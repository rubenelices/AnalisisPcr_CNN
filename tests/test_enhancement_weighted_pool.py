import json
from pathlib import Path
import pytest
import torch
from src.models.enhancement_weighted_pool import EnhancementWeightedPool
from src.models.multimodal_model import MultimodalPCRNet, load_multimodal_checkpoint
from src.training.train_multimodal import load_spec
from src.pipeline.clinical_features import ClinicalPreprocessor


def test_pool_has_zero_parameters_and_uniform_fallback():
    pool=EnhancementWeightedPool()
    features=torch.arange(8,dtype=torch.float32).reshape(1,2,2,2)
    phases=torch.zeros(1,3,4,4)
    assert not list(pool.parameters()) and not pool.state_dict()
    assert torch.allclose(pool(features,phases),features.mean((-2,-1),keepdim=True))
    phases[:,1]=.5
    assert torch.allclose(pool(features,phases),features.mean((-2,-1),keepdim=True))


def test_pool_prefers_realce_region_and_gain_invariance():
    f=torch.tensor([[[[1.,2.],[3.,4.]]]],requires_grad=True)
    x=torch.zeros(1,3,4,4); x[:,1,:2,:2]=.5
    y=EnhancementWeightedPool()(f,x)
    assert torch.allclose(y,torch.tensor([[[[1.]]]]))
    assert torch.allclose(y,EnhancementWeightedPool()(f,x*.5))
    y.sum().backward()
    assert torch.equal(f.grad,torch.tensor([[[[1.,0.],[0.,0.]]]]))


@pytest.mark.parametrize('bad',[-.1,1.1,float('nan'),float('inf')])
def test_bad_raw_phase_rejected(bad):
    x=torch.zeros(1,3,4,4);x[0,0,0,0]=bad
    with pytest.raises(ValueError):EnhancementWeightedPool()(torch.ones(1,2,2,2),x)


def test_initial_weights_rng_and_count_match_e58():
    _,cfg=load_spec('configs/multimodal/E58_clinical_fusion_mac.json')
    torch.manual_seed(42);old=MultimodalPCRNet(cfg.model);rng=torch.get_rng_state()
    torch.manual_seed(42);new=MultimodalPCRNet(cfg.model,image_pooling='enhancement_weighted')
    assert torch.equal(rng,torch.get_rng_state())
    assert sum(p.numel() for p in new.parameters())==294289
    assert old.state_dict().keys()==new.state_dict().keys()
    assert all(torch.equal(v,new.state_dict()[k]) for k,v in old.state_dict().items())


def test_model_backward_and_loader_roundtrip(tmp_path):
    spec,cfg=load_spec('experiments/D14_pooling_realce/config.json')
    model=MultimodalPCRNet(cfg.model,image_pooling=spec['image_pooling'])
    x=torch.rand(2,3,256,256);clinical=torch.zeros(2,8)
    out=model(x,clinical);assert out.shape==(2,) and torch.isfinite(out).all()
    out.sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    prep=ClinicalPreprocessor((50.,0.,0.,0.),50.,10.,878)
    path=tmp_path/'model.pt'
    torch.save(dict(model_class='MultimodalPCRNet_enhancement_v1',config=spec,
                    clinical_preprocessor=prep.to_dict(),model_state_dict=model.state_dict()),path)
    restored,_=load_multimodal_checkpoint(path)
    model.eval()
    with torch.no_grad(): assert torch.equal(model(x,clinical),restored(x,clinical))


def test_gpu_pool_agrees_with_cpu():
    if not torch.backends.mps.is_available():pytest.skip('MPS unavailable')
    f=torch.rand(2,4,16,16);x=torch.rand(2,3,256,256)
    cpu=EnhancementWeightedPool()(f,x)
    gpu=EnhancementWeightedPool()(f.to('mps'),x.to('mps')).cpu()
    assert torch.allclose(cpu,gpu,atol=1e-6,rtol=1e-5)


def test_legacy_checkpoints_keep_average(tmp_path):
    spec,cfg=load_spec('configs/multimodal/E58_clinical_fusion_mac.json')
    model=MultimodalPCRNet(cfg.model)
    prep=ClinicalPreprocessor((50.,0.,0.,0.),50.,10.,878)
    path=tmp_path/'old.pt'
    torch.save(dict(model_class='MultimodalPCRNet_v1',config=spec,
                    clinical_preprocessor=prep.to_dict(),model_state_dict=model.state_dict()),path)
    restored,_=load_multimodal_checkpoint(path)
    assert restored.image_pooling=='average'


def test_negative_realce_fallback_and_late_not_used_for_weights():
    f=torch.arange(8,dtype=torch.float32).reshape(1,2,2,2)
    x=torch.ones(1,3,4,4);x[:,1]=0
    pool=EnhancementWeightedPool()
    assert torch.allclose(pool(f,x),f.mean((-2,-1),keepdim=True))
    x[:,0]=0;x[:,1,:2,:2]=.5
    a=pool(f,x);x[:,2]=0
    assert torch.equal(a,pool(f,x))


def test_checkpoint_version_requires_pool_mode(tmp_path):
    spec,cfg=load_spec('experiments/D14_pooling_realce/config.json')
    model=MultimodalPCRNet(cfg.model,image_pooling='enhancement_weighted')
    prep=ClinicalPreprocessor((50.,0.,0.,0.),50.,10.,878)
    spec.pop('image_pooling')
    path=tmp_path/'wrong.pt'
    torch.save(dict(model_class='MultimodalPCRNet_enhancement_v1',config=spec,
                    clinical_preprocessor=prep.to_dict(),model_state_dict=model.state_dict()),path)
    with pytest.raises(ValueError):load_multimodal_checkpoint(path)
