"""Integridad de N01 con metadatos/predicciones sintéticos, sin imágenes reales."""
import json

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import balanced_accuracy_score, f1_score
import torch

from src import new_approach_protocol as protocol
from src import train_new_approach as runner


def rows():
    return pd.DataFrame([dict(patient_id=f'p{fold}_{y}_{i}', sample_id=f's{fold}_{y}_{i}_{z}',
                              pCR=y, fold=fold, split='train', slice_index=z)
                         for fold in range(5) for y in (0, 1) for i in range(5) for z in range(i % 3+1)])


def test_fixed_splits_are_grouped_complete_reproducible_and_disjoint():
    frame = rows()
    outer_seen = []
    for fold in range(5):
        parts = protocol.split_rows(frame, fold)
        repeated = protocol.split_rows(frame.sample(frac=1, random_state=31), fold)
        ids = [set(p.patient_id) for p in parts.values()]
        assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])
        assert set.union(*ids) == set(frame.patient_id)
        assert parts['outer'].fold.eq(fold).all()
        assert all(set(parts[n].patient_id) == set(repeated[n].patient_id) for n in parts)
        assert sum(len(p) for p in parts.values()) == len(frame)
        outer_seen.extend(ids[2])
    assert len(outer_seen) == len(set(outer_seen)) == frame.patient_id.nunique()


@pytest.mark.parametrize('problem', ['test', 'label', 'fold', 'duplicate', 'missing'])
def test_rejects_metadata_leakage_and_inconsistency(problem):
    frame = rows()
    if problem == 'test': frame.loc[0, 'split'] = 'test'
    if problem == 'label': frame.loc[1, 'pCR'] = 1-frame.loc[1, 'pCR']
    if problem == 'fold': frame.loc[1, 'fold'] = 3
    if problem == 'duplicate': frame.loc[1, 'sample_id'] = frame.loc[0, 'sample_id']
    if problem == 'missing': frame.loc[0, 'patient_id'] = None
    with pytest.raises(ValueError):
        protocol.split_rows(frame, 0)


def test_threshold_matches_independent_macro_f1_and_tie_rules():
    p = np.array([.1, .1, .22, .3, .31, .33, .5, .6])
    y = np.array([0, 0, 1, 0, 1, 1, 0, 1])
    frame = pd.DataFrame(dict(patient_id=list('abcdefgh'), target=y, probability=p))
    t, m = protocol.select_threshold(frame)
    u = np.unique(p)
    candidates = np.unique(np.r_[0., .5, 1., (u[1:]+u[:-1])/2])
    best = max(candidates, key=lambda x: (f1_score(y, p>=x, average='macro'),
                                         balanced_accuracy_score(y, p>=x), -abs(x-.5), -x))
    assert t == best
    assert m['f1_macro'] == pytest.approx(f1_score(y, p>=t, average='macro'))
    frame.probability = .2
    frame.target = [0, 1]*4
    assert protocol.select_threshold(frame)[0] == .5


@pytest.mark.parametrize('problem', ['duplicate', 'nonbinary', 'nan', 'range'])
def test_threshold_rejects_bad_internal_predictions(problem):
    frame = pd.DataFrame(dict(patient_id=['a', 'b'], target=[0, 1], probability=[.2, .3]))
    if problem == 'duplicate': frame.loc[1, 'patient_id'] = 'a'
    if problem == 'nonbinary':
        frame['target'] = frame.target.astype(float)
        frame.loc[1, 'target'] = 1.2
    if problem == 'nan': frame.loc[1, 'probability'] = np.nan
    if problem == 'range': frame.loc[1, 'probability'] = 1.01
    with pytest.raises(ValueError): protocol.select_threshold(frame)


def test_weight_counts_patients_not_cuts_and_undefined_precision():
    from src.patient_training import patient_level_pos_weight
    frame = pd.DataFrame(dict(patient_id=['a', 'b']+['c']*10, pCR=[0, 0]+[1]*10))
    assert patient_level_pos_weight(frame) == 2.
    pred = pd.DataFrame(dict(target=[0, 1], probability=[.1, .2]))
    m = protocol.metrics(pred, .5)
    assert not m['precision_defined'] and m['precision'] == 0.
    assert m['balanced_accuracy'] == .5


def test_freeze_is_idempotent_and_rejects_changes(tmp_path, monkeypatch):
    current = {'version': 1}
    monkeypatch.setattr(protocol, 'current_manifest', lambda root: (dict(current), rows()))
    protocol.prepare(tmp_path)
    before = (tmp_path/protocol.TECH/'protocol/manifest.json').read_bytes()
    protocol.prepare(tmp_path)
    protocol.verify(tmp_path)
    assert (tmp_path/protocol.TECH/'protocol/manifest.json').read_bytes() == before
    current['version'] = 2
    with pytest.raises(ValueError): protocol.prepare(tmp_path)
    with pytest.raises(ValueError): protocol.verify(tmp_path)


def test_lock_never_overwrites_and_releases_on_error(tmp_path):
    with pytest.raises(RuntimeError):
        with runner.exclusive_lock(tmp_path):
            with pytest.raises(FileExistsError):
                with runner.exclusive_lock(tmp_path): pass
            raise RuntimeError('Interrupción simulada')
    assert not (tmp_path/protocol.TECH/'execution.lock').exists()


def test_eval_loader_does_not_consume_training_random_stream(tmp_path):
    loader = runner.make_loader(rows(), tmp_path, 0, 1)
    before = torch.get_rng_state().clone()
    # Crear el iterador basta para comprobar el consumo de la semilla base del loader.
    iter(loader)
    assert torch.equal(before, torch.get_rng_state())
    with pytest.raises(ValueError): runner.make_loader(rows().assign(split='test'), tmp_path, 0, 1)


def test_existing_run_and_missing_smoke_block_before_training(tmp_path, monkeypatch):
    manifest = {'config': {'reference': 'unused'}}
    monkeypatch.setattr(runner, 'verify', lambda root: (manifest, rows()))
    out, _ = runner.run_dirs(tmp_path, 'normal', 0)
    out.mkdir(parents=True)
    with pytest.raises(FileExistsError): runner.run('normal', 0, 'cpu', root=tmp_path)
    with pytest.raises(RuntimeError, match='Falta smoke'):
        runner.run('weighted', 0, 'cpu', root=tmp_path)


def test_run_selects_inner_epoch_and_freezes_before_any_outer_loader(tmp_path, monkeypatch):
    """Orquestación simulada: no imágenes ni pasos de entrenamiento reales."""
    from src.train_patient_features import load_spec
    spec, typed = load_spec(protocol.ROOT/'configs/patient_features/E57_patient_feature_mean_mac.json')
    manifest = {'config': dict(reference='unused', epochs=3, patience=10)}
    protocol_dir = tmp_path/protocol.TECH/'protocol'
    protocol_dir.mkdir(parents=True)
    protocol.write_json(protocol_dir/'manifest.json', manifest)
    monkeypatch.setattr(runner, 'verify', lambda root: (manifest, rows()))
    monkeypatch.setattr(runner, 'load_spec', lambda path: (spec, typed))
    smoke, _ = runner.run_dirs(tmp_path, 'normal', 0, True)
    smoke.mkdir(parents=True)
    protocol.write_json(smoke/'summary.json', dict(status='complete', scientific_result=False,
                                                  protocol_sha256=protocol.sha(protocol_dir/'manifest.json')))
    out, weights = runner.run_dirs(tmp_path, 'normal', 0)
    step, outer_calls, loads = [0], [], []

    def loader(frame, root, fold, role, training=False):
        if role == 3:
            decision = json.loads((out/'decision_before_outer.json').read_text())
            assert decision['best_epoch'] == 1
            assert decision['checkpoint_sha256'] == protocol.sha(weights/'best.pt')
            loads.append('outer-created-after-decision')
        return role, frame

    def train(model, loader, optimizer, criterion, device):
        step[0] += 1
        # Cambios sintéticos en un parámetro permiten comprobar la restauración.
        with torch.no_grad(): next(model.parameters()).fill_(step[0])
        return .7

    def evaluate(model, loader, device):
        role, frame = loader
        p = frame[['patient_id', 'pCR']].drop_duplicates().rename(columns={'pCR': 'target'})
        good = role != 3 and (role == 1 or float(next(model.parameters()).detach().flatten()[0]) == 1.)
        p['probability'] = (.1+.8*p.target) if good else (.9-.8*p.target)
        if role == 3: outer_calls.append(True)
        return dict(bce=.6, cut_bce=.61, metrics=protocol.metrics(p, .5)), p, p

    monkeypatch.setattr(runner, 'make_loader', loader)
    monkeypatch.setattr(runner, 'train_epoch', train)
    monkeypatch.setattr(runner, 'evaluate', evaluate)
    monkeypatch.setattr(runner, 'refresh', lambda root: None)
    result = runner.run('normal', 0, 'cpu', root=tmp_path)
    assert result['best_epoch'] == 1 and result['epochs_completed'] == 3
    assert result['inner_at_best']['metrics']['roc_auc'] == 1.
    assert result['outer_at_05']['metrics']['roc_auc'] == 0.
    assert result['status'] == 'complete' and not result['test_evaluated']
    assert len(outer_calls) == 1 and len(loads) == 1
    assert not (tmp_path/protocol.TECH/'execution.lock').exists()
    checkpoint = torch.load(weights/'best.pt', weights_only=False)
    assert checkpoint['config']['experiment']['experiment_id'] == 'N01_normal_fold_0_seed_42'


@pytest.mark.parametrize('positive_weight', [1., 2.5])
def test_train_objective_uses_patient_labels_and_full_positive_weight(positive_weight):
    from src.patient_feature_model import group_features

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(1, 1, bias=False)
            with torch.no_grad(): self.linear.weight.fill_(.4)
        def forward_patients(self, x, y, ids):
            f, labels, names = group_features(x, y, ids)
            return self.linear(f).squeeze(-1), labels, names

    model = Tiny()
    x, y, ids = torch.tensor([[1.], [3.], [2.]]), torch.tensor([0., 0., 1.]), ['a', 'a', 'b']
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(positive_weight))
    expected = criterion(torch.tensor([.8, .8]), torch.tensor([0., 1.])).item()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.)
    actual = runner.train_epoch(model, [dict(image=x, target=y, patient_id=ids)], optimizer, criterion, torch.device('cpu'))
    assert actual == pytest.approx(expected)
    assert torch.isfinite(model.linear.weight.grad).all()
