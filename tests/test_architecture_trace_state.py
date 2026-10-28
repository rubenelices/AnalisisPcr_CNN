"""Dibujar/trazar no debe cambiar modo, buffers BN ni dejar hooks."""
from pathlib import Path
import pytest
import torch
from src.architectures import BreastPCRNet
from src.document_architectures import trace
from src.experiment_config import load_experiment_config

ROOT=Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('experiment',['E19_weight_decay_001_mac','E50_parallel_context_mac'])
def test_trace_restores_mixed_training_modes_buffers_and_hooks(experiment):
    model=BreastPCRNet(load_experiment_config(ROOT/'configs/experiments'/f'{experiment}.json').model)
    model.train();model.blocks[1].eval()
    modes={m:m.training for m in model.modules()}
    state={k:v.clone() for k,v in model.state_dict().items()}
    hooks={m:(len(m._forward_hooks),len(m._forward_pre_hooks)) for m in model.modules()}
    steps=trace(model)
    assert steps
    assert all(m.training==old for m,old in modes.items())
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in state.items())
    assert all((len(m._forward_hooks),len(m._forward_pre_hooks))==old for m,old in hooks.items())


def test_trace_failure_also_cleans_hooks_and_restores_modes(monkeypatch):
    model=BreastPCRNet();model.train();model.blocks[0].eval()
    modes={m:m.training for m in model.modules()}
    def failed(*args,**kwargs):
        raise RuntimeError('synthetic forward failure')
    monkeypatch.setattr(model,'forward',failed)
    with pytest.raises(RuntimeError,match='synthetic forward failure'):
        trace(model)
    assert all(m.training==old for m,old in modes.items())
    assert all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules())


def test_shape_trace_preserves_modes_bn_rng_and_supports_float64():
    model=BreastPCRNet().double();model.train();model.blocks[1].eval()
    modes={m:m.training for m in model.modules()}
    state={k:v.clone() for k,v in model.state_dict().items()}
    rng=torch.get_rng_state().clone()
    steps=model.trace_shapes(1)
    assert steps[-1].shape==(1,)
    assert all(m.training==old for m,old in modes.items())
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in state.items())
    assert torch.equal(rng,torch.get_rng_state())
