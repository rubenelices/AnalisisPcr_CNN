"""Documentación SiLU en una red sintética, sin integración científica."""
import torch
from torch import nn
from src.document_architectures import trace,visible_diagram_steps


def test_synthetic_silu_trace_has_actual_activation_without_changing_mode():
    model=nn.Sequential(nn.Conv2d(3,2,3,padding=1,bias=False),nn.BatchNorm2d(2),
        nn.SiLU(),nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.Linear(2,1))
    steps=trace(model)
    assert model.training
    assert sum(s['parameters'] for s in steps)==61
    assert [s['label'] for s in steps if s['kind']=='activation']==['SiLU']
    assert all('SiLU' in s['label'] for s in steps if s['kind']=='conv')
    assert next(s['rf'] for s in reversed(steps) if s['kind']=='conv')==3


def test_parallel_group_label_uses_actual_activation():
    steps=[{'label':'SiLU','kind':'activation'},
        {'kind':'parallel_join','label':'join','shape':(16,256,256),
         'parameters':0,'branch_parameters':432,'rf':5,'jump':1}]
    visible=visible_diagram_steps(steps)
    assert len(visible)==1 and 'BN + SiLU' in visible[0]['label']
    assert visible[0]['parameters']==432
