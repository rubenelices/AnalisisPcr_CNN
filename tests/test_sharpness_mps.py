"""Comprobación sintética específica de MPS; lanzar fuera del sandbox y sin ciencia activa.

Un skip por falta de MPS no demuestra compatibilidad. Nunca carga PNG ni pesos.
"""
from copy import deepcopy

import pytest
import torch
from torch import nn

from src.sharpness_step import sam_step


@pytest.fixture
def mps_device():
    if not torch.backends.mps.is_available():
        pytest.skip("MPS no disponible aquí; verificar explícitamente en el Mac fuera del sandbox")
    cpu_rng, mps_rng = torch.get_rng_state(), torch.mps.get_rng_state()
    try:
        torch.manual_seed(42)
        torch.mps.manual_seed(42)
        yield torch.device('mps')
    finally:
        torch.set_rng_state(cpu_rng)
        torch.mps.set_rng_state(mps_rng)


def test_mps_zero_radius_matches_adamw_and_preserves_dropout_rng(mps_device):
    reference = nn.Sequential(nn.Linear(3,4),nn.BatchNorm1d(4),nn.Dropout(0.3),nn.Linear(4,1)).to(mps_device)
    candidate = deepcopy(reference)
    x,y = torch.randn(8,3,device=mps_device), torch.randn(8,1,device=mps_device)
    old_opt = torch.optim.AdamW(reference.parameters(),lr=0.001,weight_decay=0.001)
    new_opt = torch.optim.AdamW(candidate.parameters(),lr=0.001,weight_decay=0.001)
    cpu_rng,mps_rng = torch.get_rng_state(),torch.mps.get_rng_state()
    loss = (reference(x)-y).square().mean()
    loss.backward()
    old_opt.step()
    after_cpu,after_mps = torch.get_rng_state(),torch.mps.get_rng_state()
    torch.set_rng_state(cpu_rng)
    torch.mps.set_rng_state(mps_rng)
    masks = []
    handle = candidate[2].register_forward_hook(lambda m,i,o: masks.append(o.detach().eq(0).cpu()))
    result = sam_step(candidate,new_opt,lambda: (candidate(x)-y).square().mean(),rho=0)
    handle.remove()
    assert torch.equal(masks[0],masks[1])
    assert torch.equal(after_cpu,torch.get_rng_state()) and torch.equal(after_mps,torch.mps.get_rng_state())
    assert torch.allclose(result,loss.detach(),atol=1e-7,rtol=1e-6)
    for name,value in reference.state_dict().items():
        assert torch.allclose(value,candidate.state_dict()[name],atol=1e-7,rtol=1e-6), name
    assert candidate[1].num_batches_tracked.item() == 1


def test_real_reference_cnn_nonzero_sam_updates_finite_on_mps(mps_device):
    from src.architectures import BreastPCRNet
    from src.experiment_config import load_experiment_config
    model = BreastPCRNet(load_experiment_config('configs/experiments/E19_weight_decay_001_mac.json').model).to(mps_device).train()
    optimizer = torch.optim.AdamW(model.parameters(),lr=0.001,weight_decay=0.001)
    images = torch.rand(4,3,32,32,device=mps_device)
    targets = torch.tensor([0.,1.,0.,1.],device=mps_device)
    before = model.classifier.weight.detach().clone()
    loss = sam_step(model,optimizer,lambda: nn.functional.binary_cross_entropy_with_logits(model(images),targets))
    assert torch.isfinite(loss).item()
    assert not torch.equal(before,model.classifier.weight)
    assert all(p.grad is not None and torch.isfinite(p.grad).all().item() for p in model.parameters())
    assert all(m.num_batches_tracked.item() == 1 for m in model.modules() if isinstance(m,nn.BatchNorm2d))


def test_mps_second_pass_failure_restores_all_state(mps_device):
    model = nn.Sequential(nn.Linear(2,3),nn.BatchNorm1d(3),nn.Dropout(0.3),nn.Linear(3,1)).to(mps_device)
    optimizer = torch.optim.AdamW(model.parameters(),lr=0.001)
    before = deepcopy(model.state_dict())
    x = torch.ones(4,2,device=mps_device)
    cpu_rng,mps_rng = torch.get_rng_state(),torch.mps.get_rng_state()
    calls = []
    def closure():
        calls.append(1)
        loss = model(x).square().mean()
        if len(calls)==2:
            raise RuntimeError('fallo sintético')
        return loss
    with pytest.raises(RuntimeError,match='sintético'):
        sam_step(model,optimizer,closure)
    for name,value in before.items():
        assert torch.equal(value,model.state_dict()[name])
    assert torch.equal(cpu_rng,torch.get_rng_state()) and torch.equal(mps_rng,torch.mps.get_rng_state())
    assert not optimizer.state and all(p.grad is None for p in model.parameters())
