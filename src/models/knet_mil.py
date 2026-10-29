"""Bolsas por paciente y pérdida MIL; no carga datos ni entrena al importar."""
import hashlib
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset
from sklearn.metrics import precision_recall_curve,auc
from src.models.knet_model import KNet,KNetConfig
from src.pipeline.clinical_features import ClinicalPreprocessor
from src.evaluation.metrics import binary_metrics
from src.campaigns.campaign_20261015_fast import enrich


class PatientBags(Dataset):
    """Una paciente por elemento; tres cortes sin reemplazo en train, todos en eval."""
    def __init__(self,rows,cuts,k=None,seed=42):
        self.rows=rows.reset_index(drop=True).copy();self.cuts=cuts;self.k=k;self.seed=seed;self.epoch=0
        if len(self.rows)!=len(cuts):raise ValueError('Filas y dataset deben conservar el mismo orden')
        if k is not None and (type(k) is not int or k<1):raise ValueError('k positivo')
        if self.rows.sample_id.duplicated().any():raise ValueError('Cortes duplicados')
        if self.rows.groupby('patient_id').pCR.nunique().ne(1).any():raise ValueError('Etiquetas contradictorias por paciente')
        self.groups=[(str(pid),group.index.to_numpy()) for pid,group in self.rows.groupby('patient_id',sort=True)]
        self.exposure_count=sum(min(k,len(ix)) if k else len(ix) for _,ix in self.groups)
    def set_epoch(self,epoch):self.epoch=int(epoch)
    def __len__(self):return len(self.groups)
    def indices(self,index):
        pid,ix=self.groups[index]
        if self.k is None:return ix
        token=hashlib.sha256(f'{self.seed}:{self.epoch}:{pid}'.encode()).digest()
        rng=np.random.default_rng(int.from_bytes(token[:8],'little'))
        return np.sort(rng.choice(ix,size=min(self.k,len(ix)),replace=False))
    def __getitem__(self,index):
        pid,_=self.groups[index];items=[self.cuts[int(i)] for i in self.indices(index)]
        if any(str(x['patient_id'])!=pid for x in items):raise ValueError('Dataset y filas desalineados')
        if any(not torch.equal(x['clinical'],items[0]['clinical']) for x in items):raise ValueError('Clínica inconsistente')
        return dict(image=torch.stack([x['image'] for x in items]),clinical=items[0]['clinical'],target=items[0]['target'],patient_id=pid,sample_ids=[x['sample_id'] for x in items])


def collate_bags(items):
    lengths=[len(x['image']) for x in items]
    return dict(image=torch.cat([x['image'] for x in items]),clinical=torch.stack([x['clinical'] for x in items]),target=torch.stack([x['target'] for x in items]),patient_id=[x['patient_id'] for x in items],sample_ids=[x['sample_ids'] for x in items],lengths=lengths)


def patient_forward(model,batch,device,return_cuts=False):
    """Sin padding: cada corte válido pasa una vez por la CNN compartida.

    Train: un único forward (BN conjunta del lote). Eval: fragmentos de16 cortes
    limitan memoria sin alterar BN/dropout, ambos en modo eval.
    """
    image=batch['image'].to(device);clinical=batch['clinical'].to(device)
    if sum(batch['lengths'])!=len(image) or len(batch['lengths'])!=len(clinical):raise ValueError('Bolsa inválida')
    if model.training:
        features=model.image_features(image)
    else:
        features=torch.cat([model.image_features(x) for x in image.split(16)])
    chunks=features.split(batch['lengths'])
    pooled=torch.stack([x.mean(0) for x in chunks])
    clinical_features=model.clinical_head(clinical)
    logits=model.fusion(torch.cat((pooled,clinical_features),1)).squeeze(-1)
    if not return_cuts:return logits
    repeated=torch.repeat_interleave(clinical_features,torch.tensor(batch['lengths'],device=device),dim=0)
    cut_logits=model.fusion(torch.cat((features,repeated),1)).squeeze(-1)
    return logits,cut_logits


def train_epoch(model,loader,optimizer,device,criterion):
    model.train();total=0.;count=0
    for batch in loader:
        optimizer.zero_grad(set_to_none=True)
        logits=patient_forward(model,batch,device);target=batch['target'].to(device)
        loss=criterion(logits,target)
        if not torch.isfinite(loss):raise ValueError('Pérdida no finita')
        loss.backward();optimizer.step();total+=float(loss.detach())*len(target);count+=len(target)
    return total/count


@torch.inference_mode()
def evaluate(model,loader,weight,device):
    model.eval();zs=[];ys=[];patients=[];cut_rows=[];members=[]
    for batch in loader:
        logits,cut_logits=patient_forward(model,batch,device,True)
        if not torch.isfinite(logits).all() or not torch.isfinite(cut_logits).all():raise ValueError('Logits no finitos')
        zs.append(logits.cpu());ys.append(batch['target']);patients.extend(batch['patient_id'])
        offset=0
        for pid,ids,target,length in zip(batch['patient_id'],batch['sample_ids'],batch['target'],batch['lengths']):
            probs=cut_logits[offset:offset+length].sigmoid().cpu().tolist();offset+=length
            for sid,prob in zip(ids,probs):
                cut_rows.append(dict(sample_id=sid,patient_id=pid,target=float(target),probability=prob))
                members.append(dict(patient_id=pid,sample_id=sid,feature_weight=1/length))
    z=torch.cat(zs);y=torch.cat(ys)
    primary=pd.DataFrame(dict(patient_id=patients,target=y.numpy(),probability=z.sigmoid().numpy()))
    if primary.patient_id.duplicated().any():raise ValueError('Evaluación requiere una bolsa por paciente')
    m=enrich(binary_metrics(primary.target,primary.probability,.5));cm=m['confusion_matrix'];m['precision_defined']=bool(cm['tp']+cm['fp'])
    precision,recall,_=precision_recall_curve(y.numpy(),primary.probability)
    result=dict(patient_bce=float(nn.functional.binary_cross_entropy_with_logits(z,y)),objective_patient_bce=float(nn.functional.binary_cross_entropy_with_logits(z,y,pos_weight=torch.tensor(weight))),patient_metrics=m,patient_pr_auc_trapezoid=float(auc(recall,precision)),aggregation='mean_features_then_patient_logit',evaluation_cuts='all')
    return result,pd.DataFrame(cut_rows),primary,pd.DataFrame(members)


def load_mil_checkpoint(path,device='cpu'):
    """Devuelve encoder/cabezas. Inferir con patient_forward, NO promediar sigmoid."""
    ck=torch.load(path,map_location=device,weights_only=False)
    if ck.get('model_class')!='KNet_PatientMIL_v1':raise ValueError('Checkpoint MIL requerido')
    model=KNet(KNetConfig(**ck['config']['model'])).to(device);model.load_state_dict(ck['model_state_dict'],strict=True)
    prep=ClinicalPreprocessor.from_dict(ck['clinical_preprocessor'])
    if prep.active_columns!=('age','menopause'):raise ValueError('Solo edad/menopausia')
    return model.eval(),prep
