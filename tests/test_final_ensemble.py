"""Modelo de entrega: reglas, congelación y guardas del test. Nunca toca el test real."""
import hashlib
import json
import shutil
from pathlib import Path

import pytest
import torch

from src.final_ensemble import ROOT, RULES, EnsembleModel, choose, evaluate_test, freeze, load_ensemble
from src.patient_feature_model import PatientFeatureNet
from src.train_patient_features import load_spec

E57_CONFIG = ROOT/'configs/patient_features/E57_patient_feature_mean_mac.json'


def summary(diff_ci=(-.02, .03), diff=.005, auc=.6, folds=5):
    s = {l: dict(n_folds=folds, n_pacientes=1000, auc_conjunta=auc - (0 if l == 'normal' else diff),
                 auc_conjunta_ic95=[auc-.04, auc+.04], auc_media_folds=auc) for l in ('normal', 'weighted')}
    s['pareada'] = dict(n=1000, diferencia_auc_normal_menos_ponderada=diff, ic95=list(diff_ci))
    return s


def test_rules_default_to_normal_when_difference_unclear():
    assert choose(summary())[0] == 'normal'


def test_rules_pick_weighted_only_when_clearly_better():
    assert choose(summary(diff_ci=(-.06, -.01), diff=-.03))[0] == 'weighted'


def test_rules_keep_e57_below_minimum_auc():
    loss, why = choose(summary(auc=.54))
    assert loss is None and 'E57' in why


def test_rules_refuse_incomplete_folds():
    with pytest.raises(ValueError):
        choose(summary(folds=4))


def save_fake_checkpoint(path, seed):
    spec, cfg = load_spec(E57_CONFIG)
    torch.manual_seed(seed)
    model = PatientFeatureNet(cfg.model)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(model_class='PatientFeatureNet_v1', config=spec, model_state_dict=model.state_dict()), path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_ensemble_is_mean_of_member_probabilities(tmp_path):
    members = []
    for k in range(3):
        p = tmp_path/f'm{k}.pt'
        members.append(dict(fold=k, checkpoint=p.name, sha256=save_fake_checkpoint(p, k)))
    ens = load_ensemble(dict(members=members), root=tmp_path)
    x = torch.rand(2, 3, 256, 256)
    manual = torch.stack([torch.sigmoid(m(x)) for m in ens.members]).mean(0)
    assert torch.allclose(ens.predict_proba(x), manual)
    assert not ens.training


def test_ensemble_refuses_modified_weights(tmp_path):
    p = tmp_path/'m.pt'
    save_fake_checkpoint(p, 0)
    with pytest.raises(ValueError):
        load_ensemble(dict(members=[dict(fold=0, checkpoint='m.pt', sha256='0'*64)]), root=tmp_path)


def fake_project(tmp_path):
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from test_cv_report import fake_runs
    cohorts = fake_runs(tmp_path)
    (tmp_path/RULES).parent.mkdir(parents=True)
    shutil.copy(ROOT/RULES, tmp_path/RULES)
    runs = tmp_path/'reports/new_approach/P02_aumento_geometrico/runs_cuda'
    for run in runs.iterdir():
        ckpt = tmp_path/'checkpoints/new_approach/P02_aumento_geometrico/runs_cuda'/run.name/'best.pt'
        ckpt.parent.mkdir(parents=True)
        ckpt.write_bytes(run.name.encode())
        d = json.loads((run/'decision_before_outer.json').read_text())
        d['checkpoint_sha256'] = hashlib.sha256(ckpt.read_bytes()).hexdigest()
        (run/'decision_before_outer.json').write_text(json.dumps(d))
    return cohorts


def test_freeze_writes_decision_once(tmp_path, monkeypatch):
    cohorts = fake_project(tmp_path)
    import src.final_ensemble as fe
    from src.cv_report import build
    monkeypatch.setattr(fe, 'build_cv', lambda v, m, root: build(v, m, root=root, cohorts=cohorts))
    d = freeze('p02', 'runs_cuda', root=tmp_path)
    assert d['entregable'].startswith('P02_') and len(d['members']) == 5
    assert d['threshold'] == pytest.approx(.3) and d['test_evaluated'] is False
    with pytest.raises(FileExistsError):
        freeze('p02', 'runs_cuda', root=tmp_path)


def test_freeze_refuses_modified_rules(tmp_path):
    fake_project(tmp_path)
    (tmp_path/RULES).write_text('reglas cambiadas')
    with pytest.raises(ValueError):
        freeze('p02', 'runs_cuda', root=tmp_path)


def test_test_runs_at_most_once_and_never_for_e57(tmp_path):
    out = tmp_path/'reports/new_approach/ENTREGA'
    out.mkdir(parents=True)
    from src.final_ensemble import RULES_SHA
    (out/'decision.json').write_text(json.dumps(dict(rules_sha256=RULES_SHA, entregable='E57_original')))
    with pytest.raises(ValueError):
        evaluate_test("cpu", root=tmp_path)
    (out/'decision.json').write_text(json.dumps(dict(rules_sha256=RULES_SHA, entregable='P02_normal_promedio_5_folds',
                                                     threshold=.3, members=[])))
    (out/'test').mkdir()
    with pytest.raises(FileExistsError):
        evaluate_test("cpu", root=tmp_path)
