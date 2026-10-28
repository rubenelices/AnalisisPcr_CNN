"""Repetición17 y promedio50:50, sin escoger semilla con exteriores.

Perfil, número de épocas y secuenciaLR se heredan de la selección interna42.
La repetición usa la última época prefijada, no un nuevo máximo interno.
"""
from datetime import datetime,timezone,timedelta
from pathlib import Path
import hashlib
import json
import time

import numpy as np
import pandas as pd
import torch
from torch import nn

from src.architectures import BaseCNNConfig
from src.patient_feature_model import PatientFeatureNet
from src.campaign_20261015_models import CompactPatientFeatureNet
from src.campaign_20261015_fast import ROOT,DEADLINE,loader,train_epoch,evaluate,select_threshold,enrich,write_json
from src.campaign_20261015_checkpoint import load_nested_checkpoint
from src.data import load_samples
from src.train import set_reproducibility
from src.metrics import binary_metrics


class EqualCNNEnsemble(nn.Module):
    """Dos CNNpropias; logit del promedio de probabilidades por corte."""
    def __init__(self,first,second):
        super().__init__();self.first=first;self.second=second

    def forward(self,x):
        logits=torch.stack((self.first(x),self.second(x)))
        log_positive=torch.logsumexp(nn.functional.logsigmoid(logits),dim=0)
        log_negative=torch.logsumexp(nn.functional.logsigmoid(-logits),dim=0)
        return log_positive-log_negative


def combine(first,second):
    first=first.sort_values('patient_id').reset_index(drop=True)
    second=second.sort_values('patient_id').reset_index(drop=True)
    if first.patient_id.duplicated().any() or second.patient_id.duplicated().any():raise ValueError('Paciente duplicada')
    if not first[['patient_id','target','cuts']].equals(second[['patient_id','target','cuts']]):
        raise ValueError('Pacientes, etiquetas o cortes distintos entre semillas')
    result=first.copy();result.probability=(first.probability+second.probability)/2
    return result


def load_ensemble(path,device='cpu'):
    artifact=json.loads(Path(path).read_text())
    if artifact.get('schema')!='equal_cnn_20261015' or artifact.get('weights')!=[.5,.5]:raise ValueError('Ensemble inválido')
    models=[]
    for branch in artifact['branches']:
        ck=Path(branch['checkpoint'])
        if hashlib.sha256(ck.read_bytes()).hexdigest()!=branch['sha256']:raise ValueError('Pesos cambiados')
        models.append(load_nested_checkpoint(ck,device)[0])
    if len(models)!=2:raise ValueError('Se requieren dos ramas')
    return EqualCNNEnsemble(*models).eval()


def prepare():
    plans=[]
    for fold in range(5):
        folder=ROOT/f'nested_cv_fold{fold}/resultados';summary=json.loads((folder/'summary.json').read_text())
        history=pd.read_csv(folder/'history.csv').iloc[:summary['best_epoch']]
        config=json.loads((folder/'config_resolved.json').read_text())
        # Presupuesto basado en tiempos observados del MISMO perfil, sin consultar su AUCexterior.
        estimate=float(history.elapsed_seconds.quantile(.9))*1.15*len(history)+30
        plans.append(dict(fold=fold,folder=folder,summary=summary,history=history,config=config,estimate=estimate))
    return sorted(plans,key=lambda plan:plan['estimate'])


def run_fold(plan):
    fold=plan['fold'];base=plan['summary'];history42=plan['history'];config42=plan['config']
    name=f'seed17_fixed_fold{fold}';out=ROOT/'ensemble_fixed'/f'fold{fold}'/'resultados'
    weights=Path('checkpoints/campana_20261015')/name
    if out.exists() or weights.exists():raise FileExistsError('No sobrescribir repetición')
    original=Path(base['inference_checkpoint'])
    if hashlib.sha256(original.read_bytes()).hexdigest()!=base['checkpoint_sha256']:raise ValueError('Fuente cambiada')
    raw=torch.load(original,map_location='cpu',weights_only=False)
    model_config=BaseCNNConfig(**raw['config']['experiment']['model'])
    set_reproducibility(17)
    model=(CompactPatientFeatureNet(model_config) if config42['compact'] else PatientFeatureNet(model_config)).to('mps')
    spec=dict(config42,experiment_id=name,phase='replication_fixed',seed=17,max_epochs=base['best_epoch'],
        epoch_selection='Fixed epoch inherited from inner42; no reselect with seed17',
        output_dir=str(out),checkpoint_dir=str(weights))
    out.mkdir(parents=True);weights.mkdir(parents=True);write_json(out/'config_resolved.json',spec)
    manifest=json.loads((plan['folder']/'split_manifest.json').read_text());rows=load_samples();rows=rows[rows.split=='train']
    frames={name:rows[rows.patient_id.isin(manifest[name])].copy() for name in ('fit','inner','outer')}
    for part,frame in frames.items():
        if set(frame.patient_id)!=set(manifest[part]):raise ValueError('Particiones cambiadas')
    if any(set(manifest[a])&set(manifest[b]) for a,b in [('fit','inner'),('fit','outer'),('inner','outer')]):
        raise ValueError('Solape de pacientes')
    batches=loader(frames['fit'],training=True,seed=17,augment=bool(config42['augmentation']),train_batch_size=32)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001);history=[]
    provenance={name:hashlib.sha256(Path('src',name).read_bytes()).hexdigest() for name in
        ('campaign_20261015_ensemble.py','campaign_20261015_fast.py','campaign_20261015_cache.py','campaign_20261015_models.py','architectures.py','patient_feature_model.py')}
    for row42 in history42.itertuples():
        started=time.monotonic()
        for group in optimizer.param_groups:group['lr']=float(row42.learning_rate)
        loss=train_epoch(model,batches,optimizer,torch.device('mps'))
        internal,_=evaluate(model,frames['inner'],torch.device('mps'))
        history.append(dict(epoch=int(row42.epoch),train_loss=loss,validation_loss=internal['patient_probability_bce'],
            patient_roc_auc=internal['metrics']['roc_auc'],patient_pr_auc=internal['metrics']['pr_auc'],
            learning_rate=float(row42.learning_rate),elapsed_seconds=time.monotonic()-started,
            used_for_epoch_selection=False))
        pd.DataFrame(history).to_csv(out/'history.csv',index=False)
        checkpoint=dict(model_class=raw['model_class'],epoch=int(row42.epoch),model_state_dict=model.state_dict(),
            config={'experiment':{'model':model_config.__dict__}},campaign_config=spec,source_sha256=provenance)
        torch.save(checkpoint,weights/'last.pt')
        print('SEED17',fold,'epoch',row42.epoch,'/',base['best_epoch'],'inner',internal['metrics']['roc_auc'],flush=True)
    records={}
    for part in ('fit','inner','outer'):
        _,predictions=evaluate(model,frames[part],torch.device('mps'));records[part]=predictions
        predictions.to_csv(out/f'predictions_seed17_{part}.csv',index=False)
    combined={part:combine(pd.read_csv(plan['folder']/f'predictions_{part}.csv'),pred) for part,pred in records.items()}
    report={}
    for regime,predictions in [('seed17_fixed',records),('ensemble_equal',combined)]:
        threshold=select_threshold(predictions['inner'])
        report[regime]=dict(threshold=threshold,metrics={part:enrich(binary_metrics(frame.target,frame.probability,threshold)) for part,frame in predictions.items()})
    for part,pred in combined.items():pred.to_csv(out/f'predictions_ensemble_{part}.csv',index=False)
    artifact=dict(schema='equal_cnn_20261015',weights=[.5,.5],public_test_evaluated=False,
        branches=[dict(checkpoint=str(original),sha256=base['checkpoint_sha256']),
            dict(checkpoint=str(weights/'last.pt'),sha256=hashlib.sha256((weights/'last.pt').read_bytes()).hexdigest())],
        patient_aggregation='Mean of cut probabilities; equal mean over seeds commutes with patient mean',
        threshold=report['ensemble_equal']['threshold'],seeds=[42,17],no_best_seed_selection=True)
    write_json(out/'ensemble.json',artifact)
    result=dict(fold=fold,profile=base['selected_profile'],epochs=int(base['best_epoch']),results=report,
        epoch_and_lr_selection_scope='inner42 only; no seed17 checkpoint selection',source_sha256=provenance)
    write_json(out/'summary.json',result)
    print('ENSEMBLE_FOLD',fold,'seed17',report['seed17_fixed']['metrics']['outer']['roc_auc'],
        'equal',report['ensemble_equal']['metrics']['outer']['roc_auc'],flush=True)
    del model;torch.mps.empty_cache();return result


def main():
    if not (ROOT/'nested_complete.json').exists():raise ValueError('Esperar cinco folds')
    if (ROOT/'nested_process.json').exists() or (ROOT/'followup_process.json').exists():raise ValueError('MPS ocupado')
    plans=prepare();now=datetime.now(timezone.utc)
    estimate=sum(p['estimate'] for p in plans)
    if (now+timedelta(seconds=estimate)>DEADLINE-timedelta(minutes=10)
            or now+timedelta(seconds=estimate-plans[-1]['estimate'])>DEADLINE-timedelta(minutes=32)):
        print('ENSEMBLE_SKIP no cabe comparación completa con margen de cierre',flush=True);return
    torch.set_num_threads(4);results=[]
    for plan in plans:
        if datetime.now(timezone.utc)>=DEADLINE-timedelta(minutes=30) or (ROOT/'STOP').exists():break
        results.append(run_fold(plan))
    aggregate=[]
    if len(results)==5:
        for regime in ('seed17','ensemble'):
            heldout=pd.concat([pd.read_csv(ROOT/f'ensemble_fixed/fold{r["fold"]}/resultados/predictions_{regime}_outer.csv').assign(outer_fold=r['fold']) for r in results])
            if len(heldout)!=1097 or heldout.patient_id.duplicated().any():raise ValueError('Cobertura exterior incorrecta')
            heldout.to_csv(ROOT/f'ensemble_fixed/heldout_{regime}.csv',index=False)
    for regime in ('seed17_fixed','ensemble_equal'):
        row=dict(regime=regime,complete_fivefold=len(results)==5,folds=[r['fold'] for r in results])
        row['fit_auc_mean']=float(np.mean([r['results'][regime]['metrics']['fit']['roc_auc'] for r in results]))
        row['inner_auc_mean']=float(np.mean([r['results'][regime]['metrics']['inner']['roc_auc'] for r in results]))
        for metric in ('roc_auc','pr_auc','brier_score','f1_macro','balanced_accuracy','precision','sensitivity','specificity'):
            scores=[r['results'][regime]['metrics']['outer'][metric] for r in results]
            row[metric+'_mean']=float(np.mean(scores));row[metric+'_sd']=float(np.std(scores,ddof=1)) if len(scores)>1 else None
        row['confusion_sum']={key:sum(r['results'][regime]['metrics']['outer']['confusion_matrix'][key] for r in results) for key in ('tn','fp','fn','tp')}
        aggregate.append(row)
    write_json(ROOT/'ensemble_fixed/aggregate.json',aggregate);print('ENSEMBLE_AGGREGATE',aggregate,flush=True)


if __name__=='__main__':main()
