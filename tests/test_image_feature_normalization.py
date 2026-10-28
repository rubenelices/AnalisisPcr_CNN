import json
import math
from pathlib import Path
import pytest
import torch
from torch import nn

from src.image_feature_normalization import FixedImageFeatureNorm
from src.multimodal_model import MultimodalPCRNet,load_multimodal_checkpoint
from src.train_multimodal import load_spec
from src.clinical_features import ClinicalPreprocessor
from test_multimodal_clinical import patients


@pytest.mark.parametrize('device',['cpu','mps'])
@pytest.mark.parametrize('magnitude',[0.,1e-30,1e-7,1.,1e30])
def test_stable_norm_extremes_gradients_and_batch_independence(device,magnitude):
    if device=='mps' and not torch.backends.mps.is_available():pytest.skip('MPS no disponible')
    x=torch.ones(2,128,device=device)*magnitude;x.requires_grad_()
    layer=FixedImageFeatureNorm().to(device);before=x.detach().clone()
    y=layer(x)
    assert torch.equal(x.detach(),before) and torch.isfinite(y).all()
    assert not list(layer.parameters()) and not layer.state_dict()
    if magnitude>=1e-7:assert torch.allclose(y,torch.ones_like(y),atol=3e-6)
    else:assert torch.allclose(y,x/1e-6*math.sqrt(128),atol=1e-20)
    assert torch.equal(y[:1],layer(x[:1]))
    y.sum().backward();assert torch.isfinite(x.grad).all()


def test_formula_gain_invariance_no_batch_stats():
    torch.manual_seed(42);x=torch.randn(3,128)
    layer=FixedImageFeatureNorm()
    assert torch.allclose(layer(x),math.sqrt(128)*nn.functional.normalize(x,dim=1,eps=1e-6),atol=2e-6)
    assert torch.allclose(layer(x),layer(7*x),atol=2e-6)
    assert torch.allclose(layer(x).norm(dim=1),torch.full((3,),math.sqrt(128)),atol=2e-6)
    assert torch.equal(layer(x)[[2,0,1]],layer(x[[2,0,1]]))
    assert torch.equal(layer.train()(x),layer.eval()(x))


@pytest.mark.parametrize('x',[torch.ones(128),torch.ones(1,127),torch.ones(0,128),
    torch.ones(1,128,dtype=torch.float64),torch.full((1,128),float('nan'))])
def test_invalid(x):
    with pytest.raises(ValueError):FixedImageFeatureNorm()(x)


def test_default_identity_exact_initialization_rng_and_gradient():
    _,cfg=load_spec('configs/multimodal/E58_clinical_fusion_mac.json')
    torch.manual_seed(42);a=MultimodalPCRNet(cfg.model)
    rng=torch.random.get_rng_state().clone()
    torch.manual_seed(42);b=MultimodalPCRNet(cfg.model,feature_normalization='none')
    assert torch.equal(rng,torch.random.get_rng_state())
    assert isinstance(a.feature_normalizer,nn.Identity)
    assert a.state_dict().keys()==b.state_dict().keys()
    assert all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items())
    x=torch.randn(2,3,64,64);c=torch.randn(2,8)
    a.eval();b.eval();y=a(x,c);z=b(x,c)
    assert torch.equal(y,z);y.sum().backward();z.sum().backward()
    assert all(torch.equal(p.grad,q.grad) for p,q in zip(a.parameters(),b.parameters()))


@pytest.mark.parametrize('device',['cpu','mps'])
def test_own_model_update_restoration_and_zero_extra_parameters(tmp_path,device):
    if device=='mps' and not torch.backends.mps.is_available():pytest.skip('MPS no disponible')
    torch.set_num_threads(2);torch.manual_seed(42)
    spec,cfg=load_spec('experimentos_datos/D19_norma_features/config.json')
    model=MultimodalPCRNet(cfg.model,feature_normalization=spec['feature_normalization']).to(device)
    assert sum(p.numel() for p in model.parameters())==294289
    x=torch.rand(2,3,256,256,device=device);c=torch.randn(2,8,device=device)
    before={n:p.detach().clone() for n,p in model.named_parameters()}
    opt=torch.optim.AdamW(model.parameters(),lr=.001)
    loss=nn.functional.binary_cross_entropy_with_logits(model(x,c),torch.tensor([0.,1.],device=device))
    loss.backward();assert torch.isfinite(loss)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    opt.step();assert any(not torch.equal(before[n],p) for n,p in model.named_parameters())
    model.eval();expected=model(x,c).detach()
    path=tmp_path/'own.pt';torch.save(dict(model_class='MultimodalPCRNet_normalized_v1',config=spec,
        clinical_preprocessor=ClinicalPreprocessor.fit(patients()).to_dict(),model_state_dict=model.state_dict()),path)
    restored,_=load_multimodal_checkpoint(path,device)
    assert torch.equal(expected,restored(x,c))
    ck=torch.load(path,weights_only=False);ck['model_class']='MultimodalPCRNet_v1';torch.save(ck,path)
    with pytest.raises(ValueError,match='Normalización'):load_multimodal_checkpoint(path,device)


@pytest.mark.parametrize('value',[True,1,None,'batchnorm','imagenet'])
def test_invalid_spec(tmp_path,value):
    spec=json.loads(Path('configs/multimodal/E58_clinical_fusion_mac.json').read_text())
    spec['feature_normalization']=value
    p=tmp_path/'bad.json';p.write_text(json.dumps(spec))
    with pytest.raises(ValueError):load_spec(p)
