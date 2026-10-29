"""GeM fijo: álgebra, extremos, gradiente, dtype y estabilidad; no científico."""
import pytest
import torch
from src.models.power_mean_pool import FixedPowerMeanPool


def test_formula_shapes_no_parameters_mutation_or_rng():
    x=torch.rand(2,7,8,8,dtype=torch.float64)+.1;before=x.clone();state=torch.get_rng_state()
    model=FixedPowerMeanPool();result=model(x)
    assert result.shape==(2,7,1,1) and result.dtype==x.dtype
    assert torch.allclose(result,x.pow(3).mean((-2,-1),keepdim=True).pow(1/3),atol=1e-12,rtol=1e-12)
    assert torch.equal(x,before) and torch.equal(state,torch.get_rng_state()) and not model.state_dict()
    assert sum(p.numel() for p in model.parameters())==0

def test_one_exact_average_and_three_between_average_and_maximum():
    x=torch.rand(2,4,9,9)
    assert torch.equal(FixedPowerMeanPool(1)(x),x.mean((-2,-1),keepdim=True))
    result=FixedPowerMeanPool()(x)
    assert (result>=x.mean((-2,-1),keepdim=True)-1e-6).all()
    assert (result<=x.amax((-2,-1),keepdim=True)+1e-6).all()

def test_gradcheck_and_finite_gradient():
    x=(torch.rand(1,2,5,5,dtype=torch.float64)+.1).requires_grad_();model=FixedPowerMeanPool()
    assert torch.autograd.gradcheck(model,(x,),fast_mode=True)
    model(x).sum().backward();assert torch.isfinite(x.grad).all() and (x.grad>0).all()

def test_constant_zero_and_extreme_values_finite():
    model=FixedPowerMeanPool()
    for value in (0.,2.,1e20):
        x=torch.full((2,3,4,4),value,requires_grad=True)
        y=model(x);assert torch.isfinite(y).all()
        expected=max(value,model.epsilon)
        assert torch.allclose(y,torch.full_like(y,expected),rtol=1e-6,atol=1e-12)
        y.sum().backward();assert torch.isfinite(x.grad).all()

@pytest.mark.parametrize('dtype',[torch.float16,torch.bfloat16,torch.float32,torch.float64])
def test_dtype_preserved(dtype):
    x=torch.rand(2,3,5,5).to(dtype)
    result=FixedPowerMeanPool()(x)
    assert result.dtype==dtype and torch.isfinite(result).all()

@pytest.mark.parametrize('power',[True,0,9,float('nan'),float('inf'),'3'])
def test_invalid_power(power):
    with pytest.raises(ValueError):FixedPowerMeanPool(power)

@pytest.mark.parametrize('image',[torch.rand(3,5,5),torch.empty(1,3,0,5),torch.ones(1,3,5,5,dtype=torch.int64),torch.full((1,3,5,5),float('nan')),torch.full((1,3,5,5),-.1)])
def test_invalid_input(image):
    with pytest.raises(ValueError):FixedPowerMeanPool()(image)
