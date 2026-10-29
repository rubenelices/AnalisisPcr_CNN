"""E42: ablación basal, signos/gradientes y contrato común de inferencia."""
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch

from src.models.architectures import BreastPCRNet, BaseCNNConfig, trainable_parameter_count
from src.reporting.document_architectures import trace
from src.training.experiment_config import load_experiment_config, experiment_config_from_dict
from src.web.inference import DCEInferenceEngine
from src.pipeline.phase_representation import PhaseRepresentation

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'configs/experiments/E42_temporal_differences_mac.json'
SMOKE = ROOT / 'configs/experiments/SMOKE_E42_temporal_differences_mac.json'


def test_only_representation_changes_against_e19():
    ref = load_experiment_config(ROOT / 'configs/experiments/E19_weight_decay_001_mac.json')
    cand = load_experiment_config(CONFIG)
    assert cand.data == ref.data and cand.training == ref.training
    assert cand.model == replace(ref.model, input_representation='temporal_differences')
    assert cand.model.spatial_dropout == 0 and cand.training.loss == 'normal'
    assert not cand.data.augmentation.enabled and not cand.training.mixup.enabled
    assert cand.output_dir != ref.output_dir and cand.checkpoint_dir != ref.checkpoint_dir
    assert experiment_config_from_dict(cand.to_dict()) == cand


def test_smoke_preserves_model_and_main_settings_but_is_isolated():
    cand, smoke = load_experiment_config(CONFIG), load_experiment_config(SMOKE)
    assert smoke.model == cand.model
    assert smoke.data == replace(cand.data, train_patients_per_class=4, validation_patients_per_class=2)
    assert smoke.training == replace(cand.training, epochs=2, early_stopping_patience=3)
    assert smoke.output_dir != cand.output_dir and smoke.checkpoint_dir != cand.checkpoint_dir


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_signed_two_channel_output_preserves_dtype_device_and_input(dtype):
    image = torch.tensor([.8, .9, .2], dtype=dtype).view(1,3,1,1)
    original = image.clone()
    transform = PhaseRepresentation('temporal_differences')
    before = torch.random.get_rng_state()
    result = transform(image)
    assert result.shape == (1,2,1,1)
    assert torch.allclose(result.flatten(), torch.tensor([.1, -.7], dtype=dtype))
    assert result.dtype == dtype and result.device == image.device
    assert torch.equal(image, original) and torch.equal(before, torch.random.get_rng_state())
    assert transform.output_channels == 2 and transform.state_dict() == {}
    assert list(transform.parameters()) == []


def test_no_clipping_absolute_value_or_independent_normalization():
    image = torch.tensor([1., 0., 1.]).view(1,3,1,1)
    assert torch.equal(PhaseRepresentation('temporal_differences')(image).flatten(),
                       torch.tensor([-1., 1.]))


def test_gradient_reaches_pre_early_and_late():
    image = torch.rand(2,3,4,4,requires_grad=True)
    weights = torch.tensor([2.,4.]).view(1,2,1,1)
    (PhaseRepresentation('temporal_differences')(image)*weights).sum().backward()
    assert torch.equal(image.grad, torch.tensor([-2.,-2.,4.]).view(1,3,1,1).expand_as(image))


def test_common_spatial_offset_is_lost_not_recoverable_baseline():
    image = torch.rand(2,3,5,5,dtype=torch.float64)*.5
    offset = torch.rand(2,1,5,5,dtype=torch.float64)*.2
    phase = PhaseRepresentation('temporal_differences')
    assert not torch.equal(image, image+offset)
    assert torch.allclose(phase(image), phase(image+offset), atol=1e-15, rtol=0)


@pytest.mark.parametrize('image', [torch.zeros(3,4,4), torch.zeros(1,2,4,4),
                                   torch.zeros(1,3,4,4,dtype=torch.uint8)])
def test_reject_non_triplet_or_non_float_external_input(image):
    with pytest.raises(ValueError):
        PhaseRepresentation('temporal_differences')(image)


def test_model_receives_only_two_maps_once_and_all_weights_get_gradient():
    config = BaseCNNConfig(channels=(4,8), input_representation='temporal_differences')
    model = BreastPCRNet(config)
    image = torch.rand(2,3,32,32,requires_grad=True)
    received = []
    hook = model.blocks[0].features[0].register_forward_pre_hook(
        lambda layer, inputs: received.append(inputs[0].detach().clone()))
    output = model(image)
    hook.remove()
    assert output.shape == (2,) and len(received) == 1
    assert received[0].shape == (2,2,32,32)
    assert torch.equal(received[0], PhaseRepresentation('temporal_differences')(image))
    torch.nn.functional.binary_cross_entropy_with_logits(output, torch.tensor([0.,1.])).backward()
    assert torch.isfinite(image.grad).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_real_shapes_parameter_count_and_receptive_field():
    model = BreastPCRNet(load_experiment_config(CONFIG).model)
    first = model.blocks[0].features[0]
    assert first.in_channels == 2 and first.out_channels == 16
    assert first.kernel_size == (3,3) and first.padding == (1,1) and first.stride == (1,1)
    assert first.weight.numel() == 288 and trainable_parameter_count(model) == 293985
    steps = trace(model)
    assert steps[0]['shape'] == (3,256,256) and steps[1]['shape'] == (2,256,256)
    assert steps[1]['label'] == 'EARLY−PRE\nLATE−EARLY' and steps[1]['parameters'] == 0
    assert sum(s['parameters'] for s in steps) == 293985
    assert len([s for s in steps if s['kind']=='conv']) == 8
    assert len([s for s in steps if s['kind']=='pool']) == 4
    assert [s for s in steps if s['kind']=='conv'][-1]['rf'] == 106
    shapes = model.trace_shapes(1)
    assert shapes[1].shape == (1,2,256,256)
    assert [s.shape for s in shapes[2:6]] == [(1,16,128,128),(1,32,64,64),
                                            (1,64,32,32),(1,128,16,16)]


@pytest.mark.parametrize('mode', ['raw', 'pre_differences'])
def test_old_modes_keep_three_channels_and_reload_identically(mode):
    torch.manual_seed(42)
    config = BaseCNNConfig(input_representation=mode)
    model = BreastPCRNet(config).eval()
    restored = BreastPCRNet(BaseCNNConfig(**config.to_dict())).eval()
    restored.load_state_dict(model.state_dict(), strict=True)
    assert model.input_transform.output_channels == 3
    assert model.blocks[0].features[0].in_channels == 3
    assert trainable_parameter_count(model) == 294129
    image = torch.rand(1,3,32,32)
    with torch.no_grad():
        assert torch.equal(model(image),restored(image))


def test_new_checkpoint_inference_matches_paths_bytes_and_five_repeated_uploads(tmp_path):
    config = load_experiment_config(CONFIG)
    model = BreastPCRNet(config.model).eval()
    path = tmp_path/'synthetic.pt'
    torch.save({'model_state_dict':model.state_dict(),'config':config.to_dict(),
                'experiment_id':config.experiment_id,'epoch':1},path)
    engine = DCEInferenceEngine.from_checkpoint(path,device_name='cpu')
    assert engine.model.input_transform.output_channels == 2
    sources = {}
    for phase,value in zip(['PRE','EARLY','LATE'],[50,150,90]):
        target = tmp_path/(phase+'.png')
        Image.fromarray(np.full((256,256),value,dtype=np.uint8)).save(target)
        sources[phase] = target
    from src.pipeline.data import load_dce_triplet
    image = load_dce_triplet(sources)
    with torch.inference_mode():
        expected = torch.sigmoid(model(image.unsqueeze(0))).item()
    assert engine.predict_sources(sources).probability_pcr == pytest.approx(expected)
    byte_sources = {phase:path.read_bytes() for phase,path in sources.items()}
    repeated = [engine.predict_sources(byte_sources).probability_pcr for _ in range(5)]
    assert repeated == pytest.approx([expected]*5) and not engine.model.training
    with pytest.raises(ValueError,match='Forma invalida'):
        engine.predict_tensor(image[:2])


def test_registered_e42_is_complete_from_scientific_not_smoke_metrics():
    rows=json.loads((ROOT/'experiments/notes/resultados_registrados.json').read_text())['runs']
    row=next(r for r in rows if r['experiment_id']=='E42_temporal_differences_mac')
    assert row['status']=='complete' and row['source']=='local_artifacts'
    assert row['roc_auc']==pytest.approx(0.6071572580645161)
    assert row['best_epoch']==13 and row['epochs_completed']==23
    assert row['validation_patients']==219 and row['test_evaluated'] is False
    assert row['validation_fold']==0 and row['seed']==42
