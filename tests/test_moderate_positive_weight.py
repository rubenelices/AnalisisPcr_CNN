from dataclasses import replace
import json
import math
from pathlib import Path
import pandas as pd
import pytest
import torch
from src.train_multimodal import load_spec,train_epoch,evaluate
from src.moderate_positive_weight import build_multimodal_criterion,validate_positive_weight_exponent
from src.train import build_criterion


def config(loss='weighted'):
    cfg=load_spec('configs/multimodal/E58_clinical_fusion_mac.json')[1]
    return replace(cfg,training=replace(cfg.training,loss=loss))


@pytest.mark.parametrize('loss',['normal','weighted'])
def test_default_exactly_preserves_old_loss_gradient_and_rng(loss):
    rows=pd.DataFrame({'pCR':[0,0,0,1]});cfg=config(loss)
    torch.manual_seed(42);rng=torch.get_rng_state().clone()
    old,w=build_criterion(cfg,rows,torch.device('cpu'))
    new,v=build_multimodal_criterion(cfg,rows,torch.device('cpu'))
    assert w==v and torch.equal(torch.get_rng_state(),rng)
    x=torch.tensor([-.5,0.,.5,1.],requires_grad=True);y=torch.tensor([0.,0.,0.,1.])
    a=old(x,y);a.backward();g=x.grad.clone();x.grad.zero_();b=new(x,y);b.backward()
    assert torch.equal(a,b) and torch.equal(g,x.grad)


def test_half_exponent_uses_only_train_counts_and_correct_gradient():
    criterion,w=build_multimodal_criterion(config(),pd.DataFrame({'pCR':[0,0,0,0,1]}),torch.device('cpu'),.5)
    assert w==2 and criterion.pos_weight.item()==2
    x=torch.zeros(5,requires_grad=True);y=torch.tensor([0.,0.,0.,0.,1.])
    loss=criterion(x,y);loss.backward()
    assert loss.item()==pytest.approx(6/5*math.log(2))
    assert x.grad.tolist()==pytest.approx([.1,.1,.1,.1,-.2])


@pytest.mark.parametrize('value',[0,-1,1.1,True,False,'0.5',None,float('nan'),float('inf')])
def test_invalid_exponent_rejected(value):
    with pytest.raises(ValueError):validate_positive_weight_exponent(value,'weighted')


def test_normal_rejects_moderate_exponent_and_spec_validates(tmp_path):
    with pytest.raises(ValueError):build_multimodal_criterion(config('normal'),pd.DataFrame({'pCR':[0,1]}),torch.device('cpu'),.5)
    s=json.loads(Path('configs/multimodal/E58_clinical_fusion_mac.json').read_text());s['positive_weight_exponent']=.5
    path=tmp_path/'config.json';path.write_text(json.dumps(s))
    with pytest.raises(ValueError):load_spec(path)
    s['experiment']['training']['loss']='weighted';path.write_text(json.dumps(s))
    spec,cfg=load_spec(path);assert spec['positive_weight_exponent']==.5 and cfg.training.loss=='weighted'


def test_validation_criterion_fixed_scores_unchanged():
    from tests.test_multimodal_weighted_loss import Tiny,batch
    model=Tiny();x=batch();rows=pd.DataFrame({'pCR':[0,0,0,0,1]})
    a,_=build_multimodal_criterion(config(),rows,torch.device('cpu'),.5)
    b,_=build_multimodal_criterion(config(),rows,torch.device('cpu'),1)
    first=evaluate(model,[x],torch.device('cpu'),.5,a)
    second=evaluate(model,[x],torch.device('cpu'),.5,b)
    assert first[0]!=second[0] and first[1]==second[1] and first[2].equals(second[2])


@pytest.mark.skipif(not torch.backends.mps.is_available(),reason='MPS unavailable')
def test_mps_noninteger_weight_matches_cpu_gradient():
    rows=pd.DataFrame({'pCR':[0,0,0,1]});cpu,w=build_multimodal_criterion(config(),rows,torch.device('cpu'),.5)
    mps,v=build_multimodal_criterion(config(),rows,torch.device('mps'),.5);assert w==v
    x=torch.tensor([-.5,0.,.5,1.],requires_grad=True);gx=x.detach().to('mps').requires_grad_()
    y=torch.tensor([0.,0.,0.,1.]);a=cpu(x,y);b=mps(gx,y.to('mps'));a.backward();b.backward()
    assert torch.allclose(a,b.cpu(),atol=1e-6,rtol=1e-5)
    assert torch.allclose(x.grad,gx.grad.cpu(),atol=1e-6,rtol=1e-5)


@pytest.mark.parametrize('device_name',['cpu','mps'])
def test_actual_cnn_update_with_nonunit_moderate_weight(device_name):
    if device_name=='mps' and not torch.backends.mps.is_available():pytest.skip('MPS unavailable')
    from src.multimodal_model import MultimodalPCRNet
    torch.manual_seed(42);cfg=config();device=torch.device(device_name)
    model=MultimodalPCRNet(cfg.model).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001)
    x=torch.rand(4,3,256,256).to(device);y=torch.tensor([0.,0.,0.,1.],device=device)
    criterion,w=build_multimodal_criterion(cfg,pd.DataFrame({'pCR':[0,0,0,1]}),device,.5)
    assert w==pytest.approx(math.sqrt(3)) and criterion.pos_weight.item()!=1
    before=model.image_encoder.blocks[0].features[0].weight.detach().clone()
    loss=criterion(model(x,torch.zeros(4,8,device=device)),y);loss.backward()
    assert torch.isfinite(loss) and all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    optimizer.step()
    assert not torch.equal(before,model.image_encoder.blocks[0].features[0].weight)
