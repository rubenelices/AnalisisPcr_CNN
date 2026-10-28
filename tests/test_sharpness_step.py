"""Controles matemáticos y de estado; no son resultados de pCR."""
from copy import deepcopy

import pytest
import torch
from torch import nn

from src.sharpness_step import SAMConfig, sam_step


def test_quadratic_perturbs_then_updates_original_weight_once():
    model = nn.Linear(1,1,bias=False).double()
    with torch.no_grad():
        model.weight.fill_(2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    seen = []
    def closure():
        seen.append(model.weight.item())
        return model.weight.square().sum()/2
    loss = sam_step(model, optimizer, closure, rho=0.05)
    assert seen == pytest.approx([2,2.05])
    assert model.weight.item() == pytest.approx(1.795)
    assert loss.item() == 2 and not loss.requires_grad


def test_rho_zero_matches_adamw_dropout_and_bn_one_pass():
    torch.manual_seed(17)
    reference = nn.Sequential(nn.Linear(3,4),nn.BatchNorm1d(4),nn.Dropout(0.3),nn.Linear(4,1))
    candidate = deepcopy(reference)
    x, y = torch.randn(8,3), torch.randn(8,1)
    old_opt = torch.optim.AdamW(reference.parameters(),lr=0.001,weight_decay=0.001)
    new_opt = torch.optim.AdamW(candidate.parameters(),lr=0.001,weight_decay=0.001)
    rng = torch.get_rng_state()
    loss = (reference(x)-y).square().mean()
    loss.backward()
    old_opt.step()
    after = torch.get_rng_state()
    torch.set_rng_state(rng)
    masks = []
    handle = candidate[2].register_forward_hook(lambda m,i,o: masks.append(o.detach().clone()))
    result = sam_step(candidate,new_opt,lambda: (candidate(x)-y).square().mean(),rho=0)
    handle.remove()
    assert torch.equal(masks[0],masks[1])
    assert torch.equal(after,torch.get_rng_state())
    assert torch.equal(result,loss.detach())
    for name, value in reference.state_dict().items():
        assert torch.equal(value,candidate.state_dict()[name]), name
    assert candidate[1].num_batches_tracked.item() == 1
    for old, new in zip(old_opt.state.values(),new_opt.state.values()):
        for name in old:
            assert torch.equal(old[name],new[name])


def test_second_pass_failure_restores_weights_buffers_rng_and_no_step():
    model = nn.Sequential(nn.Linear(2,2),nn.BatchNorm1d(2),nn.Dropout(0.5),nn.Linear(2,1))
    optimizer = torch.optim.AdamW(model.parameters())
    weights = deepcopy(model.state_dict())
    x = torch.ones(4,2)
    rng = torch.get_rng_state()
    calls = []
    def closure():
        calls.append(1)
        output = model(x).square().mean()
        if len(calls) == 2:
            raise RuntimeError("fallo deliberado")
        return output
    with pytest.raises(RuntimeError,match="deliberado"):
        sam_step(model,optimizer,closure)
    for name,value in weights.items():
        assert torch.equal(value,model.state_dict()[name])
    assert torch.equal(rng,torch.get_rng_state())
    assert not optimizer.state and all(p.grad is None for p in model.parameters())


def test_zero_gradient_remains_finite_and_stationary():
    model = nn.Linear(1,1,bias=False)
    with torch.no_grad():
        model.weight.zero_()
    optimizer = torch.optim.SGD(model.parameters(),lr=0.1)
    sam_step(model,optimizer,lambda: model.weight.square().sum())
    assert model.weight.item() == 0


@pytest.mark.parametrize("rho",[-0.1,float("nan"),float("inf"),True,"0.05"])
def test_invalid_radius_rejected_before_changes(rho):
    model = nn.Linear(1,1)
    optimizer = torch.optim.SGD(model.parameters(),lr=0.1)
    with pytest.raises(ValueError,match="rho"):
        sam_step(model,optimizer,lambda: model.weight.sum(),rho=rho)


def test_nonfinite_loss_rolls_back():
    model = nn.Linear(1,1)
    optimizer = torch.optim.SGD(model.parameters(),lr=0.1)
    with pytest.raises(ValueError,match="no finita"):
        sam_step(model,optimizer,lambda: model.weight.sum()*float("nan"))
    assert all(p.grad is None for p in model.parameters())


def test_global_norm_is_shared_across_parameters():
    model = nn.Linear(1,1)
    with torch.no_grad():
        model.weight.fill_(3)
        model.bias.fill_(4)
    optimizer = torch.optim.SGD(model.parameters(),lr=0.1)
    seen = []
    def closure():
        seen.append((model.weight.item(),model.bias.item()))
        return (model.weight.square().sum()+model.bias.square().sum())/2
    sam_step(model,optimizer,closure,rho=0.1)
    assert seen[1] == pytest.approx((3.06,4.08))
    assert model.weight.item() == pytest.approx(2.694)
    assert model.bias.item() == pytest.approx(3.592)


def test_nonzero_sam_keeps_first_pass_bn_buffers_only():
    model = nn.Sequential(nn.Linear(2,3),nn.BatchNorm1d(3),nn.Linear(3,1))
    optimizer = torch.optim.SGD(model.parameters(),lr=0.1)
    first_buffers = []
    def closure():
        loss = model(torch.tensor([[1.,2.],[3.,4.],[5.,7.],[9.,10.]])).square().mean()
        first_buffers.append({name:b.clone() for name,b in model.named_buffers()})
        return loss
    sam_step(model,optimizer,closure)
    assert first_buffers[1]['1.num_batches_tracked'].item() == 2
    for name,b in model.named_buffers():
        assert torch.equal(b,first_buffers[0][name])


def test_nonfinite_gradient_rolls_back_even_when_loss_finite():
    model = nn.Linear(1,1,bias=False)
    with torch.no_grad():
        model.weight.zero_()
    optimizer = torch.optim.SGD(model.parameters(),lr=0.1)
    with pytest.raises(ValueError,match="Gradientes"):
        sam_step(model,optimizer,lambda: model.weight.sqrt().sum())
    assert model.weight.item() == 0 and model.weight.grad is None


def test_half_precision_and_missing_optimizer_parameters_rejected():
    model = nn.Linear(1,1).half()
    optimizer = torch.optim.SGD(model.parameters(),lr=0.1)
    with pytest.raises(ValueError,match="float32"):
        sam_step(model,optimizer,lambda: model.weight.sum())
    model = nn.Linear(1,1)
    optimizer = torch.optim.SGD([model.weight],lr=0.1)
    with pytest.raises(ValueError,match="exactamente"):
        sam_step(model,optimizer,lambda: model.weight.sum())


def test_typed_configuration_default_is_disabled_without_rng_consumption():
    rng = torch.get_rng_state()
    assert SAMConfig() == SAMConfig(enabled=False,rho=0.05)
    assert torch.equal(rng,torch.get_rng_state())


def test_autocast_rejected_before_closure_or_update():
    model = nn.Linear(1,1)
    optimizer = torch.optim.SGD(model.parameters(),lr=0.1)
    calls = []
    with torch.autocast('cpu',dtype=torch.bfloat16):
        with pytest.raises(ValueError,match="autocast"):
            sam_step(model,optimizer,lambda: calls.append(1))
    assert not calls and not optimizer.state


@pytest.mark.parametrize("settings", [{"enabled":1},{"enabled":"false"},
    {"rho":0},{"rho":-0.1},{"rho":2},{"rho":True},{"rho":float("nan")},{"rho":float("inf")}])
def test_typed_configuration_rejects_invalid_values(settings):
    with pytest.raises(ValueError,match="sam"):
        SAMConfig(**settings)


def test_real_reference_cnn_updates_all_parameters_with_single_adamw_step():
    from src.architectures import BreastPCRNet
    from src.experiment_config import load_experiment_config
    class CountAdamW(torch.optim.AdamW):
        calls = 0
        def step(self,closure=None):
            self.calls += 1
            return super().step(closure)
    torch.manual_seed(42)
    model = BreastPCRNet(load_experiment_config('configs/experiments/E19_weight_decay_001_mac.json').model).train()
    optimizer = CountAdamW(model.parameters(),lr=0.001,weight_decay=0.001)
    images,targets = torch.rand(4,3,32,32),torch.tensor([0.,1.,0.,1.])
    before = {name:p.clone() for name,p in model.named_parameters()}
    loss = sam_step(model,optimizer,lambda: nn.functional.binary_cross_entropy_with_logits(model(images),targets))
    assert torch.isfinite(loss) and optimizer.calls == 1
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert all(not torch.equal(before[name],p) for name,p in model.named_parameters())
    assert all(m.num_batches_tracked.item() == 1 for m in model.modules() if isinstance(m,nn.BatchNorm2d))
