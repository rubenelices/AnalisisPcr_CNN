"""Activación: único cambio, inicialización compartida y ficha real."""
from dataclasses import replace
from pathlib import Path
import pytest
import torch
from torch import nn
from src.architectures import BaseCNNConfig,BreastPCRNet,trainable_parameter_count
from src.experiment_config import experiment_config_from_dict,load_experiment_config
from src.document_architectures import trace
from src.inference import DCEInferenceEngine
ROOT=Path(__file__).resolve().parents[1]
def config(name):
    return load_experiment_config(ROOT/'configs/experiments'/f'{name}.json')
def test_sole_activation_change_and_smokes():
    old,new=config('E19_weight_decay_001_mac'),config('E52_silu_mac')
    assert new.data==old.data and new.training==old.training
    assert new.model==replace(old.model,activation='silu')
    assert new==experiment_config_from_dict(new.to_dict())
    for name in ('SMOKE_E52_silu_cpu','SMOKE_E52_silu_mps'):
        tiny=config(name)
        assert tiny.model==new.model
        assert tiny.training==replace(new.training,epochs=2,early_stopping_patience=3)
        assert tiny.data==replace(new.data,train_patients_per_class=4,validation_patients_per_class=2)
        assert tiny.output_dir!=new.output_dir and tiny.checkpoint_dir!=new.checkpoint_dir
@pytest.mark.parametrize('value',['gelu','',None,True,1])
def test_activation_guard(value):
    with pytest.raises(ValueError,match='activation'):
        BaseCNNConfig(activation=value)
def test_identical_initial_parameters_rng_and_state_keys():
    old,new=config('E19_weight_decay_001_mac'),config('E52_silu_mac')
    torch.manual_seed(42);a=BreastPCRNet(old.model);rng=torch.get_rng_state()
    torch.manual_seed(42);b=BreastPCRNet(new.model)
    assert torch.equal(rng,torch.get_rng_state())
    assert a.state_dict().keys()==b.state_dict().keys()
    assert all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items())
    assert sum(isinstance(x,nn.SiLU) for x in b.modules())==8
    assert sum(isinstance(x,nn.ReLU) for x in b.modules())==0
    assert trainable_parameter_count(a)==trainable_parameter_count(b)==294129
def test_trace_silu_shapes_rf_counts_and_default_relu():
    model=BreastPCRNet(config('E52_silu_mac').model)
    steps=trace(model)
    assert sum(s['parameters'] for s in steps)==294129
    assert sum(s['kind']=='activation' and s['label']=='SiLU' for s in steps)==8
    assert all('SiLU' in s['label'] for s in steps if s['kind']=='conv')
    assert next(s['rf'] for s in reversed(steps) if s['kind']=='conv')==106
    assert next(s['shape'] for s in reversed(steps) if s['kind']=='conv')==(128,16,16)
    old=trace(BreastPCRNet(config('E19_weight_decay_001_mac').model))
    assert sum(s['kind']=='activation' and s['label']=='ReLU' for s in old)==8
def test_finite_gradients_and_legacy_parser():
    cfg=config('E52_silu_mac');model=BreastPCRNet(cfg.model)
    y=model(torch.rand(2,3,32,32));y.square().sum().backward()
    assert y.shape==(2,)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    raw=config('E19_weight_decay_001_mac').to_dict();raw['model'].pop('activation')
    assert experiment_config_from_dict(raw).model.activation=='relu'
def test_checkpoint_inference_uses_silu(tmp_path):
    cfg=config('E52_silu_mac');model=BreastPCRNet(cfg.model).eval()
    path=tmp_path/'synthetic_silu.pt'
    torch.save({'config':cfg.to_dict(),'model_state_dict':model.state_dict(),
        'experiment_id':cfg.experiment_id,'epoch':1},path)
    engine=DCEInferenceEngine.from_checkpoint(path,device_name='cpu')
    assert sum(isinstance(x,nn.SiLU) for x in engine.model.modules())==8
    for _ in range(5):
        x=torch.rand(3,256,256)
        with torch.inference_mode():
            expected=torch.sigmoid(model(x.unsqueeze(0))).item()
        assert engine.predict_tensor(x).probability_pcr==expected
