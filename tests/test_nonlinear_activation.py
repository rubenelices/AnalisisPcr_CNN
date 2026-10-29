"""Activaciones: fórmula, derivada/default/RNG, no modelo científico."""
import pytest
import torch
from torch import nn
from src.models.nonlinear_activation import make_activation


def test_default_relu_identical_no_rng_or_parameters():
    state=torch.get_rng_state();activation=make_activation()
    x=torch.tensor([-2.,-.2,0.,1.,3.]);before=x.clone()
    assert isinstance(activation,nn.ReLU) and not activation.inplace
    assert torch.equal(activation(x),nn.ReLU()(x)) and torch.equal(x,before)
    assert torch.equal(state,torch.get_rng_state())
    assert sum(p.numel() for p in activation.parameters())==0

def test_silu_formula_derivative_and_gradcheck():
    activation=make_activation('silu')
    x=torch.linspace(-6,6,25,dtype=torch.float64,requires_grad=True);before=x.detach().clone()
    output=activation(x);sigmoid=torch.sigmoid(x)
    assert torch.allclose(output,x*sigmoid,atol=1e-12,rtol=1e-12)
    output.sum().backward()
    assert torch.allclose(x.grad,sigmoid*(1+x*(1-sigmoid)),atol=1e-12,rtol=1e-12)
    assert torch.autograd.gradcheck(activation,(x,),fast_mode=True)
    assert torch.equal(x.detach(),before) and not activation.inplace
    assert sum(p.numel() for p in activation.parameters())==0

@pytest.mark.parametrize('value',[None,True,1,[],{},'swish',''])
def test_invalid_name(value):
    with pytest.raises(ValueError,match='activation'):make_activation(value)
