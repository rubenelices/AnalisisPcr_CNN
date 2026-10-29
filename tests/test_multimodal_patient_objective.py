"""Control emparejado: mismos lotes completos, distinto nivel de BCE."""
import json
from pathlib import Path
from dataclasses import replace

import pytest
import torch
import pandas as pd


def test_configs_only_objective_differs():
    from src.training.train_multimodal import load_spec
    _,control=load_spec('experiments/D21_lotes_paciente_control/config.json')
    _,patient=load_spec('experiments/D22_perdida_paciente/config.json')
    assert control.data==patient.data and control.model==patient.model
    assert replace(patient.training,objective='cut')==control.training
    assert control.data.train_patients_per_batch==2


def test_complete_loader_coverage_and_validation_order():
    from src.training.multimodal_patient_objective import build_complete_patient_loaders
    rows=pd.DataFrame({'patient_id':['a','a','a','b','c','c'],'pCR':[0,0,0,1,0,0]})
    samples=[{'image':torch.tensor(float(i)), 'patient_id':row.patient_id,
              'target':torch.tensor(float(row.pCR))} for i,row in rows.iterrows()]
    train,val=build_complete_patient_loaders(samples,samples,rows,rows,patients_per_batch=2,
        seed=42,num_workers=0,pin_memory=False)
    batches=list(train);seen=[int(x) for batch in batches for x in batch['image']]
    assert sorted(seen)==list(range(6)) and len(batches)==2
    for batch in batches:
        for pid in set(batch['patient_id']):
            assert batch['patient_id'].count(pid)==int(rows.patient_id.eq(pid).sum())
    first=[b['patient_id'] for b in val];second=[b['patient_id'] for b in val]
    assert first==second==[['a','a','a','b'],['c','c']]


@pytest.mark.parametrize('device',['cpu','mps'])
def test_actual_cnn_train_patient_objective(device):
    from src.training.train_multimodal import load_spec,train_epoch
    from src.models.multimodal_model import MultimodalPCRNet
    if device=='mps' and not torch.backends.mps.is_available():pytest.skip('MPS no disponible')
    torch.set_num_threads(2);torch.manual_seed(42)
    _,cfg=load_spec('experiments/D22_perdida_paciente/config.json')
    model=MultimodalPCRNet(cfg.model).to(device)
    assert sum(p.numel() for p in model.parameters())==294289
    clinical=torch.randn(2,8).repeat_interleave(2,0)
    batch={'image':torch.rand(4,3,256,256),'clinical':clinical,
           'target':torch.tensor([0.,0.,1.,1.]),'patient_id':['a','a','b','b']}
    before={n:p.detach().clone() for n,p in model.named_parameters()}
    loss=train_epoch(model,[batch],torch.optim.AdamW(model.parameters(),lr=.001),
                     torch.device(device),torch.nn.BCEWithLogitsLoss(),objective='patient_mean_probability')
    assert torch.isfinite(torch.tensor(loss))
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert any(not torch.equal(before[n],p) for n,p in model.named_parameters())


@pytest.mark.parametrize('schema,objective,batch',[
    ('multimodal_e19_v1','patient_mean_probability',2),
    ('multimodal_patient_e58_v1','patient_mean_probability',None),
    ('multimodal_patient_e58_v1','cut',3)])
def test_bad_protocol_rejected(tmp_path,schema,objective,batch):
    from src.training.train_multimodal import load_spec
    spec=json.loads(Path('experiments/D22_perdida_paciente/config.json').read_text())
    spec['schema']=schema;spec['experiment']['training']['objective']=objective
    spec['experiment']['data']['train_patients_per_batch']=batch
    p=tmp_path/'bad.json';p.write_text(json.dumps(spec))
    with pytest.raises(ValueError):load_spec(p)


def test_evaluate_weights_patients_not_cuts():
    from src.training.train_multimodal import evaluate
    class SyntheticModel(torch.nn.Module):
        def forward(self,image,clinical):return image[:,0]
    z=torch.tensor([-4.,4.,1.]);y=torch.tensor([0.,0.,1.])
    batch={'image':z[:,None],'clinical':torch.zeros(3,8),'target':y,
        'patient_id':['a','a','b'],'sample_id':['a0','a1','b0'],
        'slice_index':torch.tensor([0,1,0]),'split':['train']*3,'fold':torch.tensor([0]*3)}
    loss,metrics,frame=evaluate(SyntheticModel(),[batch],torch.device('cpu'),.5,
        torch.nn.BCEWithLogitsLoss(),objective='patient_mean_probability')
    expected=(torch.nn.functional.binary_cross_entropy_with_logits(torch.tensor(0.),torch.tensor(0.))
              +torch.nn.functional.binary_cross_entropy_with_logits(torch.tensor(1.),torch.tensor(1.)))/2
    assert loss==pytest.approx(float(expected),abs=1e-7)
    assert metrics['patient']['n']==2 and len(frame)==3
    cut_loss,cut_metrics,_=evaluate(SyntheticModel(),[batch],torch.device('cpu'),.5)
    assert cut_loss==pytest.approx(float(torch.nn.functional.binary_cross_entropy_with_logits(z,y)))
    assert cut_metrics==metrics


@pytest.mark.parametrize('key,value',[('feature_normalization','l2_fixed'),('fusion_hidden',32),
    ('image_intensity_field',{'max_amplitude':.1}),('image_weight_decay_factor',10)])
def test_patient_protocol_rejects_stacked_changes(tmp_path,key,value):
    from src.training.train_multimodal import load_spec
    spec=json.loads(Path('experiments/D22_perdida_paciente/config.json').read_text())
    spec[key]=value;p=tmp_path/'bad.json';p.write_text(json.dumps(spec))
    with pytest.raises(ValueError):load_spec(p)
