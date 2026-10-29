"""P01: sampler y subconjunto de overfit con datos sintéticos; sin imágenes ni entrenamiento."""
from pathlib import Path

import pandas as pd
import pytest
import torch

from src.training.train_p01 import PatientCutSampler, overfit_subset


def rows(n_patients=25, cuts=10):
    data = [dict(patient_id=f'P{p:03d}', pCR=p % 3 == 0, split='train')
            for p in range(n_patients) for _ in range(cuts)]
    frame = pd.DataFrame(data)
    frame['pCR'] = frame.pCR.astype(int)
    return frame


def test_sampler_batches_have_distinct_patients_and_two_cuts():
    frame = rows()
    sampler = PatientCutSampler(frame, 10, 2, 5, torch.Generator().manual_seed(42))
    batches = list(sampler)
    assert len(batches) == len(sampler) == 5 * 3
    for batch in batches:
        patients = frame.patient_id.iloc[batch]
        assert patients.value_counts().eq(2).all()
        assert len(set(batch)) == len(batch)
    # Cada recorrido contiene a todas las pacientes exactamente una vez.
    per_pass = [sum(batches[i:i + 3], []) for i in range(0, 15, 3)]
    for indices in per_pass:
        assert frame.patient_id.iloc[indices].value_counts().eq(2).all()
        assert frame.patient_id.iloc[indices].nunique() == 25


def test_sampler_images_per_epoch_match_e57():
    frame = rows()
    images = sum(len(b) for b in PatientCutSampler(frame, 10, 2, 5, torch.Generator().manual_seed(0)))
    assert images == len(frame)


def test_sampler_handles_short_patients_and_is_reproducible():
    frame = pd.concat([rows(4, 10), rows(1, 1).assign(patient_id='SHORT')], ignore_index=True)
    a = list(PatientCutSampler(frame, 10, 2, 2, torch.Generator().manual_seed(7)))
    b = list(PatientCutSampler(frame, 10, 2, 2, torch.Generator().manual_seed(7)))
    assert a == b
    assert all((frame.patient_id.iloc[x] == 'SHORT').sum() == 1 for x in a)


def test_sampler_rejects_invalid_sizes():
    with pytest.raises(ValueError):
        PatientCutSampler(rows(), 0, 2, 5, torch.Generator())


def test_overfit_subset_is_balanced_and_deterministic():
    frame = rows(90)
    a, b = overfit_subset(frame, 20), overfit_subset(frame, 20)
    assert a.equals(b)
    counts = a.drop_duplicates('patient_id').pCR.value_counts()
    assert counts[0] == counts[1] == 20


def test_p02_config_only_adds_augmentation_to_p01():
    from src.training.train_p01 import load_config
    p01, p02 = load_config(variant='p01'), load_config(variant='p02')
    assert 'augmentation' not in p01
    diff = {k for k in set(p01) | set(p02) if p01.get(k) != p02.get(k)}
    # 'overfit' es solo el diagnóstico de memorización de P01; no afecta al entrenamiento.
    assert diff == {'schema', 'description', 'augmentation', 'overfit'}


def test_sync_augmentation_applies_same_geometry_to_all_phases():
    from src.training.train_p01 import SyncFlipAffine
    torch.manual_seed(0)
    base = torch.rand(256, 256)
    image = torch.stack([base, base * 0.5, base * 3.0]).float()
    aug = SyncFlipAffine(dict(horizontal_flip_probability=1., vertical_flip_probability=1.,
                              max_rotation_degrees=10., max_translation_fraction=0.03))
    out = aug(image)
    assert out.shape == (3, 256, 256) and torch.isfinite(out).all()
    # Fases proporcionales siguen proporcionales píxel a píxel (también en el borde
    # rellenado con ceros) solo si las tres reciben la misma geometría.
    assert not torch.allclose(out[0], image[0])
    assert torch.allclose(out[1], out[0] * 0.5, atol=1e-5)
    assert torch.allclose(out[2], out[0] * 3.0, atol=1e-4)


def test_flip_only_is_exact_and_eval_has_no_transform():
    from src.training.train_p01 import SyncFlipAffine, train_transform, load_config
    image = torch.arange(3 * 256 * 256, dtype=torch.float32).reshape(3, 256, 256)
    aug = SyncFlipAffine(dict(horizontal_flip_probability=1., vertical_flip_probability=0.,
                              max_rotation_degrees=0., max_translation_fraction=0.))
    assert torch.equal(aug(image), torch.flip(image, dims=(-1,)))
    assert train_transform(load_config(variant='p01')) is None


def test_gpu_outputs_are_separate_from_mac(tmp_path):
    from src.training.train_p01 import run_dirs
    mac = run_dirs(tmp_path, 'normal', 1, False, 'p02', 'mps')
    gpu = run_dirs(tmp_path, 'normal', 1, False, 'p02', 'cuda')
    smoke = run_dirs(tmp_path, 'weighted', 0, True, 'p02', 'cuda')
    assert mac[0].parent.name == 'runs' and gpu[0].parent.name == 'runs_cuda'
    assert gpu[1].parent.name == 'runs_cuda' and smoke[0].parent.name == 'smoke_cuda'
    assert 'P02_aumento_geometrico' in str(gpu[0]) and mac[0] != gpu[0]


def test_night_check_accepts_identical_protocol(monkeypatch, tmp_path):
    import src.campaigns.run_p02_noche as noche
    from src.training.new_approach_protocol import ROOT
    monkeypatch.setattr(noche.torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(noche.torch.cuda, 'get_device_name', lambda i: 'GPU simulada')
    noche.check(Path(ROOT), tmp_path/'log.txt')
    assert 'idéntico' in (tmp_path/'log.txt').read_text()


def test_p05_config_only_changes_learning_rate_from_p02():
    from src.training.train_p01 import load_config
    p02, p05 = load_config(variant='p02'), load_config(variant='p05')
    diff = {k for k in set(p02) | set(p05) if p02.get(k) != p05.get(k)}
    # outer_predictions_per_epoch solo guarda predicciones para el análisis secundario.
    assert diff == {'schema', 'description', 'learning_rate', 'outer_predictions_per_epoch'}
    assert p05['learning_rate'] == p02['learning_rate'] / 10


def test_optimizer_reads_config_and_keeps_p01_p02_at_0001():
    import inspect
    from src.training.train_p01 import load_config, run
    assert "lr=float(cfg['learning_rate'])" in inspect.getsource(run)
    for variant in ('p01', 'p02'):
        cfg = load_config(variant=variant)
        assert cfg['learning_rate'] == 0.001 and cfg['weight_decay'] == 0.001
        assert not cfg.get('outer_predictions_per_epoch', False)


def test_seed_default_keeps_old_paths_and_new_seeds_are_separate(tmp_path):
    from src.training.train_p01 import run_dirs
    old = run_dirs(tmp_path, 'normal', 2, False, 'p02', 'cuda')
    assert old == run_dirs(tmp_path, 'normal', 2, False, 'p02', 'cuda', 42)
    assert old[0].name == 'normal_fold_2_seed_42'
    new = run_dirs(tmp_path, 'normal', 2, False, 'p02', 'cuda', 17)
    assert new[0].name == 'normal_fold_2_seed_17' and new[0].parent == old[0].parent
    assert run_dirs(tmp_path, 'normal', 0, True, 'p02', 'cuda', 123)[0].name == 'normal_fold_0_seed_123'
    with pytest.raises(ValueError):
        run_dirs(tmp_path, 'normal', 0, False, 'p02', 'cuda', 7)


def test_seed_changes_initial_weights_but_not_partitions():
    import torch
    from src.training.train_p01 import new_model, state_sha
    from src.training.new_approach_protocol import ROOT
    a = state_sha(new_model(ROOT, torch.device('cpu'), 42)[2])
    b = state_sha(new_model(ROOT, torch.device('cpu'), 17)[2])
    assert a != b and a == state_sha(new_model(ROOT, torch.device('cpu'))[2])
