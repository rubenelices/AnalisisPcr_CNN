"""Controles del operador aislado; no entrenamientos científicos ni AUC."""
import pytest
import torch
from torch import nn
from src.factorized_conv import FactorizedSpatialConv


@pytest.mark.parametrize('channels',[(3,16),(16,16),(16,32),(64,128)])
@pytest.mark.parametrize('bias',[False,True])
def test_shapes_weight_count_and_gradient(channels,bias):
    inputs,outputs = channels
    module = FactorizedSpatialConv(inputs,outputs,bias=bias)
    assert sum(p.numel() for p in module.parameters()) == inputs*9+inputs*outputs+(outputs if bias else 0)
    assert module.depthwise.groups == inputs and module.depthwise.bias is None
    x = torch.rand(2,inputs,11,13,requires_grad=True)
    output = module(x)
    assert output.shape == (2,outputs,11,13)
    output.square().mean().backward()
    assert torch.isfinite(x.grad).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in module.parameters())
    assert not any(isinstance(m,(nn.BatchNorm2d,nn.ReLU)) for m in module.modules())


def test_equivalent_dense_kernel_and_gradients_are_exact_factorization():
    module = FactorizedSpatialConv(2,3,bias=True).double()
    x = torch.randn(1,2,7,7,dtype=torch.float64,requires_grad=True)
    weight = module.pointwise.weight[:,:,0,0,None,None] * module.depthwise.weight[:,0][None,:,:,:]
    dense = nn.functional.conv2d(x,weight,module.pointwise.bias,padding=1)
    actual = module(x)
    assert torch.allclose(actual,dense,atol=1e-12,rtol=1e-12)
    parameters = (x,*module.parameters())
    first = torch.autograd.grad(actual.square().sum(),parameters,retain_graph=True)
    second = torch.autograd.grad(dense.square().sum(),parameters)
    assert all(torch.allclose(a,b,atol=1e-11,rtol=1e-11) for a,b in zip(first,second))


def test_depthwise_never_mixes_channels_before_pointwise():
    module = FactorizedSpatialConv(2,2)
    x = torch.zeros(1,2,7,7)
    x[:,0,3,3] = 1
    spatial = module.depthwise(x)
    assert spatial[:,1].count_nonzero().item() == 0
    with torch.no_grad():
        module.pointwise.weight.fill_(1)
    output = module.pointwise(spatial)
    assert torch.equal(output[:,0],output[:,1])


@pytest.mark.parametrize('settings',[{'in_channels':0},{'out_channels':True},
    {'kernel_size':2},{'kernel_size':3.0},{'dilation':0},{'bias':1}])
def test_invalid_settings_rejected(settings):
    values = {'in_channels':3,'out_channels':16}
    values.update(settings)
    with pytest.raises(ValueError):
        FactorizedSpatialConv(**values)


def test_dilation_retains_shapes_and_parameters_for_operator_only():
    torch.manual_seed(42)
    normal = FactorizedSpatialConv(3,16)
    torch.manual_seed(42)
    dilated = FactorizedSpatialConv(3,16,dilation=2)
    x = torch.rand(2,3,16,16)
    assert normal(x).shape == dilated(x).shape
    for name,value in normal.state_dict().items():
        assert torch.equal(value,dilated.state_dict()[name])


def test_component_count_for_proposed_seven_factorizations_after_dense_stem():
    pairs = [(3,16),(16,16),(16,32),(32,32),(32,64),(64,64),(64,128),(128,128)]
    convolutions = [nn.Conv2d(3,16,3,padding=1,bias=False)]
    convolutions += [FactorizedSpatialConv(a,b) for a,b in pairs[1:]]
    norm = [nn.BatchNorm2d(b) for a,b in pairs]
    head = nn.Linear(128,1)
    count = lambda items: sum(p.numel() for module in items for p in module.parameters())
    assert count(convolutions) == 36112 and count(norm) == 960 and count([head]) == 129
    assert count(convolutions+norm+[head]) == 37201
    # Solo componentes/cálculo, no una CNN integrada ni su inferencia.


def test_prescribed_depthwise_initialization_scales_by_per_channel_fan_in():
    # Cálculo aislado previo a integración: aún no verifica reset de BreastPCRNet.
    module = FactorizedSpatialConv(128,128)
    torch.manual_seed(17)
    linear_fan_in = torch.empty_like(module.depthwise.weight)
    nn.init.kaiming_normal_(linear_fan_in,mode='fan_in',nonlinearity='linear')
    torch.manual_seed(17)
    relu_fan_out = torch.empty_like(module.depthwise.weight)
    nn.init.kaiming_normal_(relu_fan_out,mode='fan_out',nonlinearity='relu')
    # sqrt((1/9)/(2/(128*9))) = 8, mismas normales muestreadas.
    assert torch.allclose(linear_fan_in,8*relu_fan_out)
    assert abs(linear_fan_in.var().item()-1/9) < 0.02
