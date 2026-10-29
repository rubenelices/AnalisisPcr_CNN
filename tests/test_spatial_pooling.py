"""Matemática de las dos reducciones antes de conectar una variante científica."""
import pytest
import torch
from torch import nn

from src.models.spatial_pooling import make_downsampling_pool


@pytest.mark.parametrize("mode", ["max", "average"])
@pytest.mark.parametrize("shape", [(2,3,256,256), (1,16,8,8), (3,32,9,7)])
def test_reduction_keeps_channels_halves_space_no_weights_or_rng(mode, shape):
    before = torch.random.get_rng_state()
    pool = make_downsampling_pool(mode)
    assert torch.equal(before, torch.random.get_rng_state())
    assert pool.state_dict() == {} and list(pool.parameters()) == []
    x = torch.arange(torch.tensor(shape).prod(),dtype=torch.float64).reshape(shape)
    original = x.clone()
    y = pool(x)
    assert y.shape == (shape[0],shape[1],shape[2]//2,shape[3]//2)
    assert y.dtype == x.dtype and y.device == x.device
    assert torch.equal(x,original)
    assert pool.kernel_size==2 and pool.stride==2 and pool.padding==0


def test_average_equation_four_values_and_finite_signed_gradient():
    x = torch.tensor([[[[-4.,-2.],[2.,8.]]]],dtype=torch.float64,requires_grad=True)
    y = make_downsampling_pool("average")(x)
    assert y.item()==1.
    y.sum().backward()
    assert torch.equal(x.grad,torch.full_like(x,.25))
    assert torch.autograd.gradcheck(make_downsampling_pool("average"),(x.detach().requires_grad_(),))


def test_max_equation_and_localised_gradient():
    x = torch.tensor([[[[-4.,-2.],[2.,8.]]]],requires_grad=True)
    y = make_downsampling_pool("max")(x)
    assert y.item()==8.
    y.sum().backward()
    assert torch.equal(x.grad,torch.tensor([[[[0.,0.],[0.,1.]]]]))


def test_peak_and_distributed_signal_can_be_lost_differently():
    peak = torch.tensor([[[[4.,0.],[0.,0.]]]])
    distributed = torch.ones_like(peak)
    average,max_pool=make_downsampling_pool("average"),make_downsampling_pool("max")
    assert torch.equal(average(peak),average(distributed))
    assert not torch.equal(max_pool(peak),max_pool(distributed))
    # Esta diferencia matemática no implica superioridad clínica ni de AUC.


def test_default_matches_original_maxpool_exactly():
    x=torch.randn(2,8,32,32)
    assert torch.equal(make_downsampling_pool()(x),nn.MaxPool2d(2,2)(x))


@pytest.mark.parametrize("mode", ["", "mean", "median", "stride_conv", None, True])
def test_unknown_pooling_rejected(mode):
    with pytest.raises(ValueError,match="max o average"):
        make_downsampling_pool(mode)
