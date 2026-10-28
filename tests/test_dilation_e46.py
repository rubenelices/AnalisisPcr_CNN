"""Modelo real E46: muestreo, formas, RF, gradientes e inferencia compartida."""
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from torch import nn

from src.architectures import BaseCNNConfig, BreastPCRNet, trainable_parameter_count
from src.document_architectures import trace
from src.experiment_config import experiment_config_from_dict, load_experiment_config
from src.inference import DCEInferenceEngine

ROOT = Path(__file__).resolve().parents[1]

def config(name):
    return load_experiment_config(ROOT/'configs/experiments'/f'{name}.json')


def test_only_dilation_changes_against_e19_and_smokes_are_isolated():
    old,new = config('E19_weight_decay_001_mac'),config('E46_dilation2_mac')
    assert new.model == replace(old.model,dilation=2)
    assert new.data == old.data and new.training == old.training
    assert not new.training.sam.enabled
    assert new == experiment_config_from_dict(new.to_dict())
    for name in ('SMOKE_E46_dilation2_cpu','SMOKE_E46_dilation2_mps'):
        small = config(name)
        assert small.model == new.model
        assert small.training == replace(new.training,epochs=2,early_stopping_patience=3)
        assert small.data == replace(new.data,train_patients_per_class=4,validation_patients_per_class=2)
        assert small.output_dir != new.output_dir and small.checkpoint_dir != new.checkpoint_dir


@pytest.mark.parametrize('value',[0,3,True,1.0,None,'2'])
def test_invalid_dilation_rejected(value):
    with pytest.raises(ValueError,match='dilation'):
        BaseCNNConfig(dilation=value)


def test_same_initial_weights_keys_rng_and_count_without_loading_reference():
    torch.manual_seed(42)
    old = BreastPCRNet(config('E19_weight_decay_001_mac').model)
    rng = torch.get_rng_state()
    torch.manual_seed(42)
    new = BreastPCRNet(config('E46_dilation2_mac').model)
    assert torch.equal(rng,torch.get_rng_state())
    assert trainable_parameter_count(old) == trainable_parameter_count(new) == 294129
    assert old.state_dict().keys() == new.state_dict().keys()
    assert all(torch.equal(value,new.state_dict()[name]) for name,value in old.state_dict().items())
    layers = [layer for layer in new.modules() if isinstance(layer,nn.Conv2d)]
    assert len(layers) == 8
    assert all(layer.kernel_size==(3,3) and layer.dilation==(2,2) and layer.padding==(2,2) and layer.stride==(1,1) for layer in layers)
    assert sum(isinstance(layer,nn.MaxPool2d) for layer in new.modules()) == 4


def test_real_trace_shapes_rf_parameter_sum_and_gradients():
    model = BreastPCRNet(config('E46_dilation2_mac').model)
    steps = trace(model)
    pools = [step for step in steps if step['kind']=='pool']
    assert [step['shape'] for step in pools] == [(16,128,128),(32,64,64),(64,32,32),(128,16,16)]
    convs = [step for step in steps if step['kind']=='conv']
    assert convs[-1]['rf']==196 and convs[-1]['jump']==16
    assert sum(step['parameters'] for step in steps)==294129
    assert all('dilation=2' in step['detail'] for step in convs)
    assert [step for step in steps if step['kind']=='head'][0]['shape']==(128,1,1)
    model.train()
    x = torch.rand(2,3,256,256,requires_grad=True)
    logits = model(x)
    assert logits.shape == (2,)
    nn.BCEWithLogitsLoss()(logits,torch.tensor([0.,1.])).backward()
    assert torch.isfinite(x.grad).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert next(step['rf'] for step in reversed(trace(BreastPCRNet(config('E19_weight_decay_001_mac').model))) if step['kind'] in {'conv','pool'}) == 106


@pytest.mark.parametrize('new',[False,True])
def test_old_and_dilated_checkpoint_shared_inference_five_triplets(tmp_path,new):
    cfg = config('E46_dilation2_mac' if new else 'E19_weight_decay_001_mac')
    model = BreastPCRNet(cfg.model).eval()
    raw = cfg.to_dict()
    if not new:
        raw['model'].pop('dilation')
        assert experiment_config_from_dict(raw).model.dilation==1
    path = tmp_path/'own_synthetic.pt'
    torch.save({'config':raw,'model_state_dict':model.state_dict(),'experiment_id':cfg.experiment_id,'epoch':0},path)
    engine = DCEInferenceEngine.from_checkpoint(path,device_name='cpu')
    for _ in range(5):
        image = torch.rand(3,256,256)
        with torch.inference_mode():
            expected = torch.sigmoid(model(image.unsqueeze(0))).item()
        assert engine.predict_tensor(image).probability_pcr==expected
