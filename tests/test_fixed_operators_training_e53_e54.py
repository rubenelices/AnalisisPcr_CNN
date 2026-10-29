"""E53/E54: cambios aislados, default exacto e inferencia compartida."""
from dataclasses import replace
from pathlib import Path
import pytest
import torch
from torch import nn
from src.models.architectures import BaseCNNConfig,BreastPCRNet,trainable_parameter_count
from src.training.experiment_config import experiment_config_from_dict,load_experiment_config
from src.reporting.document_architectures import trace
from src.pipeline.triplet_standardization import TripletStandardization
from src.models.power_mean_pool import FixedPowerMeanPool
from src.web.inference import DCEInferenceEngine
ROOT=Path(__file__).resolve().parents[1]
def config(name):
    return load_experiment_config(ROOT/'configs/experiments'/f'{name}.json')
@pytest.mark.parametrize('name,changes',[('E53_shared_standardization_mac',{'input_normalization':'shared_mean_std'}),
    ('E54_fixed_gem_mac',{'global_pooling':'power_mean'})])
def test_sole_operator_change_and_smokes(name,changes):
    old,new=config('E19_weight_decay_001_mac'),config(name)
    assert new.data==old.data and new.training==old.training
    assert new.model==replace(old.model,**changes)
    assert new==experiment_config_from_dict(new.to_dict())
    assert trainable_parameter_count(BreastPCRNet(new.model))==294129
    for device in ('cpu','mps'):
        tiny=config(f'SMOKE_{name[:-4]}_{device}')
        assert tiny.model==new.model
        assert tiny.training==replace(new.training,epochs=2,early_stopping_patience=3)
        assert tiny.data==replace(new.data,train_patients_per_class=4,validation_patients_per_class=2)
        assert tiny.output_dir!=new.output_dir and tiny.checkpoint_dir!=new.checkpoint_dir
@pytest.mark.parametrize('name',['E53_shared_standardization_mac','E54_fixed_gem_mac'])
def test_initial_parameters_keys_and_rng_identical(name):
    old,new=config('E19_weight_decay_001_mac'),config(name)
    torch.manual_seed(42);a=BreastPCRNet(old.model);rng=torch.get_rng_state()
    torch.manual_seed(42);b=BreastPCRNet(new.model)
    assert torch.equal(rng,torch.get_rng_state()) and a.state_dict().keys()==b.state_dict().keys()
    assert all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items())
def test_shared_standardization_precedes_phase_differences_and_trace_global_dependency():
    cfg=config('E53_shared_standardization_mac');model=BreastPCRNet(cfg.model).eval()
    x=torch.rand(2,3,32,32);captured=[]
    handle=model.input_transform.register_forward_pre_hook(lambda m,args:captured.append(args[0].clone()))
    with torch.inference_mode():model(x)
    handle.remove()
    expected=TripletStandardization('shared_mean_std')(x)
    torch.testing.assert_close(captured[0],expected)
    steps=trace(model)
    assert steps[1]['global_input_statistics'] and steps[1]['parameters']==0
    assert steps[2]['label'].startswith('PRE')
    assert next(s['rf'] for s in reversed(steps) if s['kind']=='conv')==106
    assert sum(s['parameters'] for s in steps)==294129
    assert model.trace_shapes(1)[1].operation=='normalización conjunta por tripleta'
def test_fixed_gem_head_actual_not_trainable_and_does_not_add_channels():
    model=BreastPCRNet(config('E54_fixed_gem_mac').model)
    assert isinstance(model.global_pool,FixedPowerMeanPool)
    assert model.global_pool.power==3 and not list(model.global_pool.parameters())
    steps=trace(model);step=next(s for s in steps if s['label'].startswith('GeM'))
    assert step['shape']==(128,1,1) and step['parameters']==0
    assert sum(s['parameters'] for s in steps)==294129
    assert model.trace_shapes(1)[-3].operation=='power mean pooling p3 fixed'
@pytest.mark.parametrize('mode',[None,True,'per_phase','imagenet',''])
def test_invalid_normalization(mode):
    with pytest.raises(ValueError,match='input_normalization'):
        BaseCNNConfig(input_normalization=mode)
def test_gem_rejects_negative_activation_and_grid():
    with pytest.raises(ValueError,match='ReLU'):
        BaseCNNConfig(global_pooling='power_mean',activation='silu')
    with pytest.raises(ValueError,match='grid'):
        BaseCNNConfig(global_pooling='power_mean',global_pool_grid_size=2)
def test_legacy_parser_defaults_no_preprocess_or_activation_change():
    raw=config('E19_weight_decay_001_mac').to_dict()
    raw['model'].pop('input_normalization');raw['model'].pop('activation')
    cfg=experiment_config_from_dict(raw)
    assert cfg.model.input_normalization=='none' and cfg.model.activation=='relu'
@pytest.mark.parametrize('name',['E53_shared_standardization_mac','E54_fixed_gem_mac'])
def test_shared_checkpoint_inference_uses_fixed_operator(tmp_path,name):
    cfg=config(name);model=BreastPCRNet(cfg.model).eval()
    path=tmp_path/f'{name}.pt'
    torch.save({'config':cfg.to_dict(),'model_state_dict':model.state_dict(),
        'experiment_id':cfg.experiment_id,'epoch':1},path)
    engine=DCEInferenceEngine.from_checkpoint(path,device_name='cpu')
    for _ in range(5):
        x=torch.rand(3,256,256)
        with torch.inference_mode():expected=torch.sigmoid(model(x.unsqueeze(0))).item()
        assert engine.predict_tensor(x).probability_pcr==expected
