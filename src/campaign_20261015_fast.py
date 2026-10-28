"""Control de CNN E57 entrenada por corte; evaluación anidada por paciente."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import platform
import resource

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from src.architectures import BaseCNNConfig, trainable_parameter_count
from src.data import load_samples
from src.campaign_20261015_cache import CachedDataset as BreastDCEDataset
from src.augmentation import AugmentationConfig,build_train_transform
from src.horizontal_flip import HorizontalFlipConfig
from src.campaign_20261015_models import CompactPatientFeatureNet
from dataclasses import replace
from src.patient_training import PatientBatchSampler
from src.patient_feature_model import PatientFeatureNet
from src.train import set_reproducibility
from src.train_patient_features import evaluate_features
from src.metrics import binary_metrics, aggregate_by_patient

ROOT=Path('experimentos_equilibrados/campana_20261015')
DEADLINE=datetime.fromisoformat('2026-10-15T10:52:26+00:00')
MODEL=BaseCNNConfig(pool_position='between_convolutions',input_representation='pre_differences')


class DeadlineReached(RuntimeError):pass


def check_deadline():
    if (ROOT/'STOP').exists():raise DeadlineReached('Parada solicitada')
    if datetime.now(timezone.utc)>=DEADLINE:raise DeadlineReached('Fin de la ventana humana')


def write_json(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False));tmp.replace(path)


def nested_split(rows,fold,seed=42):
    if not rows.split.eq('train').all():raise ValueError('Solo train público')
    if rows.groupby('patient_id').pCR.nunique().max()!=1 or rows.groupby('patient_id').fold.nunique().max()!=1:
        raise ValueError('Paciente con etiqueta/fold incoherente')
    outer=rows.loc[rows.fold==fold].copy();remaining=rows.loc[rows.fold!=fold].copy()
    patients=remaining[['patient_id','pCR']].drop_duplicates('patient_id')
    fitids,innerids=train_test_split(patients.patient_id.to_numpy(),test_size=.2,
        stratify=patients.pCR.to_numpy(),random_state=seed)
    fit=remaining.loc[remaining.patient_id.isin(fitids)].copy()
    inner=remaining.loc[remaining.patient_id.isin(innerids)].copy()
    frames=[x.reset_index(drop=True) for x in (fit,inner,outer)]
    sets=[set(x.patient_id) for x in frames]
    if any(sets[i]&sets[j] for i,j in ((0,1),(0,2),(1,2))):raise ValueError('Solape de pacientes')
    if any(x.pCR.nunique()!=2 for x in frames):raise ValueError('Se requieren ambas clases')
    return frames


def select_threshold(frame):
    if frame.patient_id.duplicated().any():raise ValueError('Paciente repetida')
    p=frame.probability.to_numpy(float);y=frame.target.to_numpy(int)
    if set(y)!={0,1} or not np.isfinite(p).all() or np.any((p<0)|(p>1)):
        raise ValueError('Predicciones internas inválidas')
    u=np.unique(p);candidates=np.unique(np.r_[0.,.5,1.,(u[:-1]+u[1:])/2])
    best=None
    for t in candidates:
        pred=p>=t;tp=int(((y==1)&pred).sum());tn=int(((y==0)&~pred).sum())
        fp=int(((y==0)&pred).sum());fn=int(((y==1)&~pred).sum())
        f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.
        f0=2*tn/(2*tn+fp+fn) if 2*tn+fp+fn else 0.
        balanced=(tp/(tp+fn)+tn/(tn+fp))/2
        key=((f0+f1)/2,balanced,-abs(float(t)-.5))
        if best is None or key>best[0]:best=(key,float(t))
    return best[1]


def enrich(metrics):
    c=metrics['confusion_matrix'];tn,fp,fn,tp=[c[k] for k in ('tn','fp','fn','tp')]
    f0=2*tn/(2*tn+fp+fn) if 2*tn+fp+fn else 0.
    return dict(metrics,f1_negative=f0,f1_macro=(f0+metrics['f1'])/2,
        baseline_macro=(tn+fp)/(2*(tn+fp)+fn+tp),baseline_balanced_accuracy=.5)


def loader(rows,batch_size=32,training=False,patients_per_batch=2,seed=42,augment=False,train_batch_size=16):
    transform=build_train_transform(AugmentationConfig(True,5.,.02),horizontal_config=HorizontalFlipConfig(True,.5)) if training and augment else None
    ds=BreastDCEDataset(rows,'breastdcedl',transform=transform)
    if training:
        return DataLoader(ds,batch_size=train_batch_size,shuffle=True,num_workers=0,generator=torch.Generator().manual_seed(seed))
    return DataLoader(ds,batch_size=batch_size,shuffle=False,num_workers=0)


def subset(rows,n=2):
    patients=rows[['patient_id','pCR']].drop_duplicates('patient_id')
    ids=patients.groupby('pCR',group_keys=False).head(n).patient_id
    return rows.loc[rows.patient_id.isin(ids)].reset_index(drop=True)


def train_epoch(model,batches,optimizer,device):
    model.train();loss_sum=0.;count=0;criterion=nn.BCEWithLogitsLoss()
    for batch in batches:
        check_deadline();optimizer.zero_grad(set_to_none=True)
        z=model(batch['image'].to(device));y=batch['target'].to(device)
        loss=criterion(z,y);loss.backward();optimizer.step()
        loss_sum+=float(loss.detach().cpu())*len(y);count+=len(y)
    return loss_sum/count


@torch.no_grad()
def evaluate(model,rows,device):
    check_deadline()
    loss,bagloss,metrics,bagmetrics,cuts,bags=evaluate_features(model,loader(rows),device,.5)
    patients=aggregate_by_patient(cuts.patient_id,cuts.target,cuts.probability)
    return dict(patient_probability_bce=loss,bag_bce=bagloss,metrics=metrics['patient'],bag_secondary=bagmetrics),patients


def run(config_path):
    spec=json.loads(Path(config_path).read_text())
    if spec['schema']!='nested_e57_20261015':raise ValueError('Esquema de campaña inválido')
    if spec['patients_per_batch'] not in (2,4,8):raise ValueError('Lote de pacientes inválido')
    out=Path(spec['output_dir']);weights=Path(spec['checkpoint_dir'])
    if (not out.is_relative_to(ROOT) or not weights.is_relative_to('checkpoints/campana_20261015')
            or '..' in out.parts or '..' in weights.parts or out.is_absolute() or weights.is_absolute()):
        raise ValueError('Salidas de campaña inválidas')
    if out.exists() or weights.exists():raise FileExistsError('Sin sobrescribir/resume')
    check_deadline();device=torch.device(spec.get('device','mps'))
    if device.type=='mps' and not torch.backends.mps.is_available():raise RuntimeError('Sin MPS, no fallback')
    if device.type not in ('cpu','mps'):raise ValueError('Solo dispositivos locales CPU/MPS')
    torch.set_num_threads(4);set_reproducibility(spec.get('seed',42))
    rows=load_samples('breastdcedl');rows=rows.loc[rows.split=='train'].copy()
    fit,inner,outer=nested_split(rows,spec['outer_fold'])
    if spec.get('debug'):fit,inner,outer=[subset(x) for x in (fit,inner,outer)]
    out.mkdir(parents=True);weights.mkdir(parents=True)
    write_json(out/'config_resolved.json',spec)
    write_json(out/'split_manifest.json',dict(fit=fit.patient_id.unique().tolist(),
        inner=inner.patient_id.unique().tolist(),outer=outer.patient_id.unique().tolist(),
        test_evaluated=False))
    sources=['campaign_20261015_fast.py','campaign_20261015_cache.py','campaign_20261015_models.py','augmentation.py','horizontal_flip.py','patient_feature_model.py','architectures.py','data.py','patient_training.py','train_patient_features.py']
    provenance={x:hashlib.sha256(Path(__file__).with_name(x).read_bytes()).hexdigest() for x in sources}
    model_config=replace(MODEL,channels=(8,16,32,64)) if spec.get('compact',False) else MODEL
    model=(CompactPatientFeatureNet(model_config) if spec.get('compact',False) else PatientFeatureNet(model_config)).to(device)
    expected_count=75961 if spec.get('compact',False) else 298161
    assert trainable_parameter_count(model)==expected_count
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=spec.get('weight_decay',.001))
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='max',factor=.5,patience=3)
    batches=loader(fit,training=True,patients_per_batch=spec['patients_per_batch'],seed=spec.get('seed',42),augment=bool(spec.get('augmentation')),train_batch_size=spec.get('training_batch_size',16))
    history=[];best=-float('inf');best_monitor=-float('inf');bestepoch=0;stale=0;start=time.monotonic()
    maxepochs=spec.get('max_epochs',24);warmup=min(3,maxepochs)
    print(spec['experiment_id'], 'fit/inner/outer',*[x.patient_id.nunique() for x in (fit,inner,outer)],
        'training','cut batch',spec.get('training_batch_size',16),'device',device,flush=True)
    try:
        for epoch in range(1,maxepochs+1):
            before=time.monotonic();lr=optimizer.param_groups[0]['lr']
            trainloss=train_epoch(model,batches,optimizer,device)
            evaluation,_=evaluate(model,inner,device)
            row=dict(epoch=epoch,train_loss=trainloss,validation_loss=evaluation['patient_probability_bce'],learning_rate=lr,
                patient_roc_auc=evaluation['metrics']['roc_auc'],patient_pr_auc=evaluation['metrics']['pr_auc'])
            history.append(row)
            monitor=float(np.mean([x['patient_roc_auc'] for x in history[-3:]]))
            row.update(monitor_inner_auc_ma3=monitor,elapsed_seconds=time.monotonic()-before)
            pd.DataFrame(history).to_csv(out/'history.csv',index=False)
            ck=dict(model_class='CompactPatientFeatureNet_v1' if spec.get('compact',False) else 'PatientFeatureNet_v1',epoch=epoch,model_state_dict=model.state_dict(),
                config=dict(experiment=dict(experiment_id=spec['experiment_id'],model=model_config.__dict__)),campaign_config=spec,source_sha256=provenance)
            torch.save(ck,weights/'last.pt')
            if epoch>=warmup:
                scheduler.step(monitor)
                if row['patient_roc_auc']>best:
                    best=row['patient_roc_auc'];bestepoch=epoch;torch.save(ck,weights/'best.pt')
                if monitor>best_monitor:best_monitor=monitor;stale=0
                else:stale+=1
            print(f'epoch={epoch:02d} loss={trainloss:.4f} innerloss={evaluation["patient_probability_bce"]:.4f} '
                f'innerAUC={row["patient_roc_auc"]:.4f} MA3={monitor:.4f} AP={row["patient_pr_auc"]:.4f} '
                f'lr={lr:g} sec={row["elapsed_seconds"]:.1f}',flush=True)
            if stale>=spec.get('patience',6):break
        ck=torch.load(weights/'best.pt',map_location=device,weights_only=False)
        model.load_state_dict(ck['model_state_dict'])
        result=dict(status='complete',experiment_id=spec['experiment_id'],phase=spec['phase'],
            scientific_result=not spec.get('debug',False),debug_patient_subset=bool(spec.get('debug',False)),
            test_evaluated=False,outer_fold=spec['outer_fold'],patients_per_batch=None,training_unit='cut',
            best_epoch=bestepoch,epochs_completed=len(history),best_inner_smoothed_auc=best_monitor,best_inner_roc_auc=best,
            seed=spec.get('seed',42),training_batch_size=spec.get('training_batch_size',16),parameter_count=expected_count,source_sha256=provenance,elapsed_seconds=time.monotonic()-start,
            environment=dict(torch=torch.__version__,python=platform.python_version(),platform=platform.platform(),device=str(device),
                host_maxrss_raw=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                host_maxrss_units='bytes' if platform.system()=='Darwin' else 'KiB'),
            patients={k:x.patient_id.nunique() for k,x in [('fit',fit),('inner',inner),('outer',outer)]})
        train_eval,trainpred=evaluate(model,fit,device);inner_eval,innerpred=evaluate(model,inner,device)
        threshold=select_threshold(innerpred)
        result.update(train_eval=train_eval,inner_eval=inner_eval,selected_threshold=threshold,
            threshold_rule='F1macro inner only; balanced accuracy then closeness0.5 ties')
        trainpred.to_csv(out/'predictions_fit.csv',index=False);innerpred.to_csv(out/'predictions_inner.csv',index=False)
        result['inner_decision_metrics']=enrich(binary_metrics(innerpred.target,innerpred.probability,threshold))
        if spec['phase']!='screen':
            outer_eval,outerpred=evaluate(model,outer,device)
            result.update(outer_eval=outer_eval,outer_decision_metrics=enrich(binary_metrics(outerpred.target,outerpred.probability,threshold)))
            outerpred.to_csv(out/'predictions_outer.csv',index=False)
        write_json(out/'summary.json',result)
        print('COMPLETE',spec['experiment_id'],'best',bestepoch,'inner',inner_eval['metrics']['roc_auc'],
            'outer',result.get('outer_eval',{}).get('metrics',{}).get('roc_auc'),'threshold',threshold,flush=True)
        return result
    except (DeadlineReached,KeyboardInterrupt) as error:
        write_json(out/'partial.json',dict(status='partial',reason=str(error),epochs=len(history),test_evaluated=False))
        raise
    finally:
        if device.type=='mps':torch.mps.empty_cache()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',required=True)
    run(parser.parse_args().config)
