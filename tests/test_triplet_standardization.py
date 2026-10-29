"""Álgebra y gradientes de normalización conjunta; no rendimiento ni dataset."""
import pytest
import torch
from src.pipeline.triplet_standardization import TripletStandardization


def test_default_identity_no_rng_or_parameters():
    state=torch.get_rng_state();model=TripletStandardization();x=torch.rand(2,3,12,12)
    current=torch.get_rng_state()
    assert model(x) is x and torch.equal(current,torch.get_rng_state())
    assert len(model.state_dict())==0 and sum(p.numel() for p in model.parameters())==0

def test_each_triplet_one_shared_mean_std_and_batch_independence():
    model=TripletStandardization('shared_mean_std');x=torch.rand(2,3,13,13,dtype=torch.float64)
    original=x.clone();actual=model(x)
    variance,mean=torch.var_mean(x,dim=(1,2,3),correction=0,keepdim=True)
    assert torch.allclose(actual,(x-mean)/variance.sqrt(),atol=1e-12,rtol=1e-12)
    assert torch.allclose(actual.mean((1,2,3)),torch.zeros(2,dtype=x.dtype),atol=1e-12)
    assert torch.allclose(actual.var((1,2,3),correction=0),torch.ones(2,dtype=x.dtype),atol=1e-12)
    assert torch.allclose(actual,torch.cat([model(x[i:i+1]) for i in range(2)]),atol=1e-12)
    assert torch.equal(x,original) and actual.shape==x.shape and actual.dtype==x.dtype
    # Restas firmadas solo se escalan; una fase NO se centra por separado.
    scale=variance.sqrt()[:,0]
    assert torch.allclose(actual[:,1]-actual[:,0],(x[:,1]-x[:,0])/scale,atol=1e-12)
    assert torch.allclose(actual[:,2]-actual[:,1],(x[:,2]-x[:,1])/scale,atol=1e-12)

def test_positive_affine_invariance_above_floor_and_finite_gradients():
    model=TripletStandardization('shared_mean_std');x=torch.rand(2,3,5,5,dtype=torch.float64,requires_grad=True)
    assert torch.allclose(model(x),model(x*1.2+.3),atol=1e-12,rtol=1e-12)
    assert torch.autograd.gradcheck(model,(x,),fast_mode=True)
    model(x).square().sum().backward();assert torch.isfinite(x.grad).all()

@pytest.mark.parametrize('value',[0.,.8])
def test_constant_triplet_output_and_gradient_finite(value):
    x=torch.full((2,3,5,5),value,requires_grad=True);model=TripletStandardization('shared_mean_std')
    y=model(x);assert torch.isfinite(y).all() and torch.count_nonzero(y)==0
    y.sum().backward();assert torch.isfinite(x.grad).all()

@pytest.mark.parametrize('mode',[None,True,{},[],1,'imagenet','per_phase'])
def test_invalid_mode(mode):
    with pytest.raises(ValueError):TripletStandardization(mode)

@pytest.mark.parametrize('epsilon',[True,0,-1,float('nan'),float('inf')])
def test_invalid_epsilon(epsilon):
    with pytest.raises(ValueError):TripletStandardization(epsilon=epsilon)

@pytest.mark.parametrize('image',[torch.rand(3,5,5),torch.rand(1,2,5,5),torch.ones(1,3,5,5,dtype=torch.int64),torch.full((1,3,5,5),float('nan'))])
def test_invalid_active_input(image):
    with pytest.raises(ValueError):TripletStandardization('shared_mean_std')(image)


@pytest.mark.parametrize('shape',[(0,3,5,5),(1,3,0,5),(1,3,5,0)])
def test_active_rejects_empty_triplets_but_default_keeps_identity(shape):
    image=torch.empty(shape)
    assert TripletStandardization()(image) is image
    with pytest.raises(ValueError,match='no vacío'):
        TripletStandardization('shared_mean_std')(image)
