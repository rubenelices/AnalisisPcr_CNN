"""Contrato E49, compatibilidad por defecto y composición sincronizada."""
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from src.models.architectures import BreastPCRNet, trainable_parameter_count
from src.pipeline.augmentation import AugmentationConfig, IntensityAugmentationConfig, build_train_transform
from src.training.experiment_config import DataExperimentConfig, experiment_config_from_dict, load_experiment_config
from src.pipeline.horizontal_flip import HorizontalFlipConfig

ROOT=Path(__file__).resolve().parents[1]

def config(name):
    return load_experiment_config(ROOT/'configs/experiments'/f'{name}.json')

def test_protocol_sole_change_and_technical_scopes():
    old,new=config('E19_weight_decay_001_mac'),config('E49_horizontal_flip_mac')
    assert new.model==old.model and new.training==old.training
    assert new.data==replace(old.data,horizontal_flip=HorizontalFlipConfig(True,.5))
    assert new==experiment_config_from_dict(new.to_dict())
    assert trainable_parameter_count(BreastPCRNet(new.model))==294129
    for name in ('SMOKE_E49_horizontal_flip_cpu','SMOKE_E49_horizontal_flip_mps'):
        tiny=config(name)
        assert tiny.data==replace(new.data,train_patients_per_class=4,validation_patients_per_class=2)
        assert tiny.training==replace(new.training,epochs=2,early_stopping_patience=3)
        assert tiny.model==new.model
        assert tiny.output_dir!=new.output_dir and tiny.checkpoint_dir!=new.checkpoint_dir

@pytest.mark.parametrize('horizontal',[None,HorizontalFlipConfig(),HorizontalFlipConfig(True,0)])
def test_default_pipeline_preserves_rng_and_exact_old_affine(horizontal):
    x=torch.rand(3,256,256)
    cfg=AugmentationConfig(True)
    old=build_train_transform(cfg)
    new=build_train_transform(cfg,horizontal_config=horizontal)
    torch.manual_seed(17); a=old(x); state=torch.get_rng_state()
    torch.manual_seed(17); b=new(x)
    assert torch.equal(a,b) and torch.equal(state,torch.get_rng_state())
    assert build_train_transform(AugmentationConfig(),horizontal_config=horizontal) is None

def test_flip_first_then_shared_gain():
    x=torch.rand(3,256,256); original=x.clone()
    transform=build_train_transform(AugmentationConfig(),IntensityAugmentationConfig(True),HorizontalFlipConfig(True,1))
    torch.manual_seed(42); expected=torch.flip(x,[-1])* (1+(2*torch.rand(()).item()-1)*.1)
    state=torch.get_rng_state()
    torch.manual_seed(42); actual=transform(x)
    assert torch.equal(actual,expected) and torch.equal(state,torch.get_rng_state())
    assert torch.equal(x,original)

def test_old_saved_configuration_and_initial_weights_unchanged():
    raw=config('E19_weight_decay_001_mac').to_dict();raw['data'].pop('horizontal_flip')
    old=experiment_config_from_dict(raw)
    assert old.data.horizontal_flip==HorizontalFlipConfig()
    torch.manual_seed(42);a=BreastPCRNet(old.model); state=torch.get_rng_state()
    torch.manual_seed(42);b=BreastPCRNet(config('E49_horizontal_flip_mac').model)
    assert torch.equal(state,torch.get_rng_state())
    assert all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items())

def test_untyped_horizontal_configuration_rejected():
    with pytest.raises(ValueError,match='horizontal_flip'):
        DataExperimentConfig(horizontal_flip={})
