"""Pruebas de protocolo, arquitecturas y cola E32–E41, sin ciencia nueva."""
from dataclasses import replace
import json
from pathlib import Path
import shutil

import pytest
import torch
from torch import nn

from src.models.architectures import BreastPCRNet, trainable_parameter_count
from src.training.experiment_config import load_experiment_config
from src.campaigns.run_mac_batch import SPECS, EXPERIMENTS, MANIFEST, ROOT, validate_plan, changed_settings


@pytest.mark.parametrize("name,parent,changes,_", SPECS)
def test_parent_changes_and_smoke_budget(name, parent, changes, _):
    old = load_experiment_config(f"configs/experiments/{parent}.json")
    new = load_experiment_config(f"configs/experiments/{name}.json")
    smoke = load_experiment_config(f"configs/experiments/SMOKE_{name}.json")
    assert changed_settings(old, new) == changes
    assert new.training.epochs == 50 and new.training.early_stopping_patience == 10
    assert new.training.seed == 42 and new.data.validation_fold == 0
    assert new.data.train_patient_fraction == 1
    assert new.data.train_patients_per_class is None and new.data.validation_patients_per_class is None
    assert smoke.model == new.model
    assert smoke.data == replace(new.data, train_patients_per_class=4, validation_patients_per_class=2)
    assert smoke.training == replace(new.training, epochs=2, early_stopping_patience=3)
    assert smoke.checkpoint_dir != new.checkpoint_dir and smoke.output_dir != new.output_dir


@pytest.mark.parametrize("name", EXPERIMENTS)
def test_actual_layers_dimensions_and_trace_parameter_sum(name):
    from src.reporting.document_architectures import trace
    cfg = load_experiment_config(f"configs/experiments/{name}.json")
    model = BreastPCRNet(cfg.model).eval()
    steps = trace(model)
    assert sum(step['parameters'] for step in steps) == trainable_parameter_count(model)
    convs = [layer for block in model.blocks for layer in block.modules() if isinstance(layer, nn.Conv2d)]
    assert len(convs) == len(cfg.model.channels) * cfg.model.convolutions_per_block
    assert all(layer.kernel_size == (cfg.model.kernel_size,) * 2 for layer in convs)
    assert all(layer.padding == (cfg.model.kernel_size // 2,) * 2 for layer in convs)
    with torch.no_grad():
        output = model(torch.zeros(2, 3, 256, 256))
    assert output.shape == (2,) and torch.isfinite(output).all()
    assert all(step['shape'][-1] > 0 for step in steps)


def sandbox_plan(tmp_path):
    directory = tmp_path / 'configs/experiments'
    directory.mkdir(parents=True)
    for name in set(EXPERIMENTS) | {s[1] for s in SPECS}:
        shutil.copyfile(ROOT / f'configs/experiments/{name}.json', directory / f'{name}.json')
    shutil.copyfile(ROOT / MANIFEST, tmp_path / MANIFEST)
    return directory


def test_preview_never_launches_or_writes(tmp_path, monkeypatch, capsys):
    import src.campaigns.run_mac_batch as queue
    sandbox_plan(tmp_path)
    monkeypatch.setattr(queue, 'ROOT', tmp_path)
    monkeypatch.setattr(queue, 'run_child', lambda *args: pytest.fail('Vista previa entrenó'))
    assert queue.main(['--only', 'E32', 'E37']) == 0
    assert 'VISTA PREVIA' in capsys.readouterr().out
    assert not (tmp_path / 'reports').exists()
    assert not (tmp_path / 'checkpoints').exists()


@pytest.mark.parametrize('parent', [False, True])
def test_modified_configs_rejected_before_execution(tmp_path, parent):
    directory = sandbox_plan(tmp_path)
    path = directory / f'{SPECS[0][1 if parent else 0]}.json'
    config = json.loads(path.read_text())
    config['training']['epochs'] = 51
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match='modificada'):
        validate_plan(tmp_path)


def test_partial_run_and_orphan_log_are_preserved(tmp_path, monkeypatch):
    import src.campaigns.run_mac_batch as queue
    sandbox_plan(tmp_path)
    monkeypatch.setattr(queue, 'ROOT', tmp_path)
    cfg = load_experiment_config(tmp_path / f'configs/experiments/{EXPERIMENTS[0]}.json')
    checkpoint = tmp_path / cfg.checkpoint_dir
    checkpoint.mkdir(parents=True)
    with pytest.raises(FileExistsError, match='parcial'):
        queue.main(['--only', 'E32'])
    assert checkpoint.is_dir()
    checkpoint.rmdir()
    log = tmp_path / 'reports/mac_batch_E32_E41' / f'{EXPERIMENTS[0]}.log'
    log.parent.mkdir(parents=True)
    log.write_text('conservar')
    with pytest.raises(FileExistsError, match='Log existente'):
        queue.main(['--only', 'E32'])
    assert log.read_text() == 'conservar'


def test_error_stops_queue_and_releases_only_own_lock(tmp_path, monkeypatch):
    import src.campaigns.run_mac_batch as queue
    sandbox_plan(tmp_path)
    monkeypatch.setattr(queue, 'ROOT', tmp_path)
    monkeypatch.setattr(queue, 'require_device', lambda _: torch.device('cpu'))
    called = []
    def fail(command, root, path):
        called.append(command)
        return 7
    monkeypatch.setattr(queue, 'run_child', fail)
    assert queue.main(['--execute', '--device', 'cpu', '--only', 'E32', 'E33']) == 7
    assert len(called) == 1
    assert not (tmp_path / 'reports/mac_batch_E32_E41/active.lock').exists()
    lock = tmp_path / 'reports/mac_batch_E32_E41/active.lock'
    lock.write_text('ajeno')
    with pytest.raises(FileExistsError):
        queue.main(['--execute', '--device', 'cpu', '--only', 'E32'])
    assert lock.read_text() == 'ajeno'


def test_completed_runs_skipped_and_explicit_device_checked(tmp_path, monkeypatch):
    import src.campaigns.run_mac_batch as queue
    sandbox_plan(tmp_path)
    monkeypatch.setattr(queue, 'ROOT', tmp_path)
    monkeypatch.setattr(queue, 'execution_state', lambda *args: 'complete')
    monkeypatch.setattr(queue, 'run_child', lambda *args: pytest.fail('Reentrenó un completo'))
    monkeypatch.setattr(queue, 'require_device', lambda _: torch.device('cpu'))
    assert queue.main(['--execute', '--device', 'cpu', '--only', 'E32']) == 0
    def unavailable(_):
        raise RuntimeError('MPS no disponible')
    monkeypatch.setattr(queue, 'require_device', unavailable)
    with pytest.raises(RuntimeError, match='MPS'):
        queue.main(['--execute', '--only', 'E33'])
    assert not (tmp_path / 'reports/mac_batch_E32_E41/active.lock').exists()


def test_duplicate_requests_rejected():
    import src.campaigns.run_mac_batch as queue
    with pytest.raises(SystemExit):
        queue.main(['--only', 'E32', 'E32'])


def test_kernel5_documentation_reports_real_padding(tmp_path):
    from src.reporting.document_architectures import trace, write_sheet
    cfg = load_experiment_config('configs/experiments/E34_kernel5_compact_mac.json')
    model = BreastPCRNet(cfg.model)
    write_sheet(tmp_path, 'E34', model, 'Prueba', [], trace(model))
    content = (tmp_path / 'README.md').read_text()
    assert 'kernel 5×5, padding 2, stride 1' in content
    assert 'tienen stride 1 y padding 1' not in content


@pytest.mark.parametrize('name', ['E37_rotation5_mac', 'E39_dropout_affine_mac'])
def test_geometry_is_shared_and_within_prefixed_bounds(name):
    from src.pipeline.augmentation import SharedRandomAffine
    cfg = load_experiment_config(f'configs/experiments/{name}.json')
    transform = SharedRandomAffine(cfg.data.augmentation)
    plane = torch.rand(256, 256)
    image = torch.stack((plane, plane * .5, plane * .25))
    original = image.clone()
    out = transform(image)
    assert torch.allclose(out[1], out[0] * .5)
    assert torch.allclose(out[2], out[0] * .25)
    assert torch.equal(image, original)
    for _ in range(25):
        p = transform.sample_parameters()
        assert abs(p.angle_degrees) <= 5
        assert abs(p.translation_x_pixels) <= cfg.data.augmentation.max_translation_fraction * 256
        assert abs(p.translation_y_pixels) <= cfg.data.augmentation.max_translation_fraction * 256


def test_dropout_maps_train_only_and_same_parameter_count():
    cfg = load_experiment_config('configs/experiments/E38_spatial_dropout010_mac.json')
    model = BreastPCRNet(cfg.model)
    layers = [layer for layer in model.modules() if isinstance(layer, nn.Dropout2d)]
    assert len(layers) == 4 and all(layer.p == .1 for layer in layers)
    assert trainable_parameter_count(model) == 294129
    x = torch.ones(8, 128, 2, 2)
    layer = layers[-1]
    layer.train()
    torch.manual_seed(42)
    y = layer(x)
    assert (y == 0).any() and (y != 0).any()
    assert torch.equal(y[:, :, 0, 0], y[:, :, 1, 1])
    layer.eval()
    assert torch.equal(layer(x), x)


def test_child_interruption_terminates_process_and_preserves_log(tmp_path, monkeypatch):
    import src.campaigns.run_mac_batch as queue
    actions = []
    class InterruptedChild:
        @property
        def stdout(self):
            yield 'antes de interrumpir\n'
            raise KeyboardInterrupt
        def terminate(self):
            actions.append('terminate')
        def wait(self, timeout=None):
            actions.append(('wait', timeout))
            return 0
    monkeypatch.setattr(queue.subprocess, 'Popen', lambda *a, **k: InterruptedChild())
    log = tmp_path / 'output.log'
    with pytest.raises(KeyboardInterrupt):
        queue.run_child(['dummy'], tmp_path, log)
    assert actions == ['terminate', ('wait', 10)]
    assert log.read_text() == 'antes de interrumpir\n'
    with pytest.raises(FileExistsError):
        queue.run_child(['dummy'], tmp_path, log)


def test_mps_availability_is_verified_not_just_device_label(monkeypatch):
    import src.campaigns.run_mac_batch as queue
    monkeypatch.setattr(torch.backends.mps, 'is_available', lambda: False)
    with pytest.raises(RuntimeError, match='MPS no disponible'):
        queue.require_device('mps')
    assert queue.require_device('cpu') == torch.device('cpu')
