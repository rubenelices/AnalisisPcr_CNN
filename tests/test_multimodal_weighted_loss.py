"""Controles para pérdida ponderada; referencia normal compatible."""
from dataclasses import replace
import copy
import pandas as pd
import pytest
import torch
from torch import nn

from src.training.train import build_criterion
from src.training.train_multimodal import load_spec, train_epoch, evaluate


def config():
    return load_spec('configs/multimodal/E58_clinical_fusion_mac.json')[1]


def test_weight_depends_only_on_supplied_train_cuts():
    cfg=config(); cfg=replace(cfg,training=replace(cfg.training,loss='weighted'))
    train=pd.DataFrame({'pCR':[0,0,0,1]})
    criterion,weight=build_criterion(cfg,train,torch.device('cpu'))
    assert weight==3 and criterion.pos_weight.item()==3
    logits=torch.zeros(4,requires_grad=True); labels=torch.tensor([0.,0.,0.,1.])
    loss=criterion(logits,labels)
    assert loss.item()==pytest.approx(1.5*torch.log(torch.tensor(2.)).item())
    loss.backward()
    assert logits.grad.tolist()==pytest.approx([.125,.125,.125,-.375])


def test_normal_criterion_does_not_consume_rng_or_use_class_weight():
    torch.manual_seed(42);before=torch.get_rng_state().clone()
    criterion,weight=build_criterion(config(),pd.DataFrame({'pCR':[0,0,0,1]}),torch.device('cpu'))
    assert weight is None and criterion.pos_weight is None
    assert torch.equal(before,torch.get_rng_state())


class Tiny(nn.Module):
    def __init__(self):
        super().__init__(); self.fc=nn.Linear(2,1)
    def forward(self,image,clinical):
        return self.fc(image+clinical).squeeze(1)


def batch():
    return dict(image=torch.tensor([[1.,0.],[0.,1.],[1.,1.],[0.,0.]]),
                clinical=torch.zeros(4,2),target=torch.tensor([0.,0.,0.,1.]),
                sample_id=['a','b','c','d'],patient_id=['a','b','c','d'],
                slice_index=torch.tensor([0,0,0,0]),split=['validation']*4,
                fold=torch.zeros(4,dtype=torch.long))


def test_normal_epoch_preserves_original_updates():
    torch.manual_seed(42); old=Tiny(); new=copy.deepcopy(old)
    a=torch.optim.AdamW(old.parameters(),lr=.001); b=torch.optim.AdamW(new.parameters(),lr=.001)
    x=batch(); old.train(); a.zero_grad(set_to_none=True)
    loss=nn.functional.binary_cross_entropy_with_logits(old(x['image'],x['clinical']),x['target'])
    loss.backward(); a.step()
    observed=train_epoch(new,[x],b,torch.device('cpu'),nn.BCEWithLogitsLoss())
    assert observed==loss.item()
    for p,q in zip(old.parameters(),new.parameters()):assert torch.equal(p,q)


def test_evaluation_uses_fixed_train_weight_without_changing_scores():
    torch.manual_seed(42); model=Tiny(); x=batch()
    unweighted=evaluate(model,[x],torch.device('cpu'),.5,nn.BCEWithLogitsLoss())
    weighted=evaluate(model,[x],torch.device('cpu'),.5,
                      nn.BCEWithLogitsLoss(pos_weight=torch.tensor(3.)))
    assert weighted[0]>unweighted[0]
    assert weighted[1]==unweighted[1]
    assert weighted[2].equals(unweighted[2])


def test_weighted_spec_allowed_without_other_changes(tmp_path):
    import json
    from pathlib import Path
    s=json.loads(Path('configs/multimodal/E58_clinical_fusion_mac.json').read_text())
    s['experiment']['training']['loss']='weighted'
    path=tmp_path/'weighted.json'; path.write_text(json.dumps(s))
    spec,cfg=load_spec(path)
    assert cfg.training.loss=='weighted'
    assert cfg.training.objective=='cut' and not cfg.training.mixed_precision


@pytest.mark.skipif(not torch.backends.mps.is_available(),reason='MPS no disponible')
def test_nonunit_positive_weight_mps_loss_and_gradient():
    device=torch.device('mps')
    cpu=torch.tensor([-.4,0.,.2,1.],requires_grad=True)
    mps=cpu.detach().to(device).requires_grad_()
    y=torch.tensor([0.,0.,1.,1.])
    expected=nn.BCEWithLogitsLoss(pos_weight=torch.tensor(3.))(cpu,y)
    actual=nn.BCEWithLogitsLoss(pos_weight=torch.tensor(3.,device=device))(mps,y.to(device))
    expected.backward();actual.backward()
    assert actual.item()==pytest.approx(expected.item(),abs=1e-6)
    assert torch.allclose(mps.grad.cpu(),cpu.grad,atol=1e-6)
