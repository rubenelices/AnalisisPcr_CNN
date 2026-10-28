"""Ramas propias: formas, pesos, gradientes y aritmética espacial; no calidad."""
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from src.parallel_context import ParallelContextConv


@pytest.mark.parametrize('cin,cout',[(3,16),(16,32),(32,64),(64,128)])
def test_same_parameters_and_spatial_shape_as_dense_conv(cin,cout):
    branch=ParallelContextConv(cin,cout)
    assert sum(p.numel() for p in branch.parameters())==9*cin*cout
    assert branch(torch.randn(2,cin,16,16)).shape==(2,cout,16,16)
    assert branch.local.dilation==(1,1) and branch.local.padding==(1,1)
    assert branch.context.dilation==(2,2) and branch.context.padding==(2,2)
    assert not any(isinstance(m,(nn.ReLU,nn.BatchNorm2d)) for m in branch.modules())
    assert branch.local.groups==branch.context.groups==1


def test_equivalence_to_constrained_sparse_five_kernel_and_gradients():
    model=ParallelContextConv(3,8,bias=True).double()
    x=torch.randn(2,3,9,9,dtype=torch.float64,requires_grad=True)
    kernel=torch.zeros(8,3,5,5,dtype=torch.float64)
    kernel[:4,:,1:4,1:4]=model.local.weight
    kernel[4:,:,::2,::2]=model.context.weight
    bias=torch.cat((model.local.bias,model.context.bias))
    actual=model(x);expected=F.conv2d(x,kernel,bias,padding=2)
    assert torch.allclose(actual,expected,atol=1e-12,rtol=1e-12)
    a=torch.autograd.grad(actual.square().sum(),x,retain_graph=True)[0]
    b=torch.autograd.grad(expected.square().sum(),x,retain_graph=True)[0]
    assert torch.allclose(a,b,atol=1e-11,rtol=1e-11)
    actual.square().sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0 for p in model.parameters())


def test_gradcheck_input_and_no_mutation():
    model=ParallelContextConv(2,4).double()
    x=torch.randn(1,2,5,5,dtype=torch.float64,requires_grad=True)
    before=x.detach().clone()
    assert torch.autograd.gradcheck(model,(x,),fast_mode=True)
    assert torch.equal(x.detach(),before)


def test_branch_order_and_common_input_hook():
    model=ParallelContextConv(3,16)
    seen=[]
    handles=[m.register_forward_pre_hook(lambda layer,inputs:seen.append(inputs[0])) for m in (model.local,model.context)]
    x=torch.rand(1,3,12,12);result=model(x)
    for h in handles:h.remove()
    assert seen[0] is x and seen[1] is x
    assert torch.equal(result[:,:8],model.local(x))
    assert torch.equal(result[:,8:],model.context(x))


@pytest.mark.parametrize('cin,cout,bias',[(0,16,False),(3,15,False),(True,16,False),(3,True,False),(3,16,1)])
def test_invalid_configuration_rejected(cin,cout,bias):
    with pytest.raises(ValueError):ParallelContextConv(cin,cout,bias)

def test_initialization_uses_total_output_channels(monkeypatch):
    import math
    model=ParallelContextConv(3,16,bias=True)
    calls=[]
    original=nn.init.normal_
    def capture(tensor,mean=0.,std=1.,**kwargs):
        calls.append((tensor,mean,std))
        return original(tensor,mean,std,**kwargs)
    monkeypatch.setattr(nn.init,'normal_',capture)
    model.reset_branch_parameters()
    assert len(calls)==2
    assert calls[0][0] is model.local.weight and calls[1][0] is model.context.weight
    assert all(mean==0 and std==math.sqrt(2/(16*9)) for _,mean,std in calls)
    assert torch.count_nonzero(model.local.bias)==torch.count_nonzero(model.context.bias)==0

def test_initialization_reproducible_and_finite():
    model=ParallelContextConv(32,64)
    torch.manual_seed(17);model.reset_branch_parameters()
    a={name:p.clone() for name,p in model.named_parameters()};state=torch.get_rng_state()
    torch.manual_seed(17);model.reset_branch_parameters()
    assert torch.equal(state,torch.get_rng_state())
    assert all(torch.equal(a[name],p) and torch.isfinite(p).all() for name,p in model.named_parameters())
