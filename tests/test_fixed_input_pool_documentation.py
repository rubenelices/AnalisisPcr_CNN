"""Fichas de operadores independientes; sin modelos científicos nuevos."""
from torch import nn
from src.document_architectures import trace
from src.phase_representation import PhaseRepresentation
from src.triplet_standardization import TripletStandardization
from src.power_mean_pool import FixedPowerMeanPool


def synthetic(normalization='none',pool=None):
    return nn.Sequential(TripletStandardization(normalization),PhaseRepresentation('pre_differences'),
        nn.Conv2d(3,2,3,padding=1,bias=False),nn.BatchNorm2d(2),nn.ReLU(),
        pool if pool is not None else nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.Linear(2,1))


def test_standardization_documented_before_differences_without_fake_parameters():
    model=synthetic('shared_mean_std');steps=trace(model)
    assert steps[1]['global_input_statistics'] is True
    assert steps[1]['shape']==(3,256,256) and steps[1]['parameters']==0
    assert steps[2]['label'].startswith('PRE')
    assert sum(s['parameters'] for s in steps)==61
    assert next(s['rf'] for s in reversed(steps) if s['kind']=='conv')==3
    assert model.training


def test_default_standardization_identity_not_shown_and_fixed_gem_head_actual():
    steps=trace(synthetic(pool=FixedPowerMeanPool()))
    assert not any(s.get('global_input_statistics') for s in steps)
    gem=next(s for s in steps if s['label'].startswith('GeM'))
    assert gem['shape']==(2,1,1) and gem['parameters']==0
    assert 'NO aprendible' in gem['detail']
    assert sum(s['parameters'] for s in steps)==61
