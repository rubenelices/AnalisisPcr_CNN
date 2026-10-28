"""C17: receta publicada de la CNN97.809, preparada para ejecución del estudiante.

Implementación propia con componentes existentes. Solo train público, fold0,
BCE normal/ponderada por corte; selección en el mismo fold de validación.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import resource
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import auc, precision_recall_curve

from src.campaign_20261015_cache import CachedDataset
from src.campaign_20261015_fast import write_json, enrich
from src.data import BreastDCEDataset, load_samples
from src.metrics import binary_metrics, aggregate_by_patient
from src.simple_cnn_comparison import make_model, model_config

ROOT=Path('experimentos_equilibrados/C17_receta_publicada')
WEIGHTS=Path('checkpoints/experimentos_equilibrados/C17_receta_publicada')
COMMIT='572fa3fdd95a0853d4871424a1f5233ff437c80c'
SCHEMA='published_recipe_97809_v1'


def recipe(debug=False):
    return dict(schema=SCHEMA,experiment_id='C17_receta_publicada',fold=0,seed=42,
                batch_size=16,num_workers=0,max_epochs=1 if debug else 40,patience=8,
                learning_rate=.001,weight_decay=.0001,optimizer='Adam',scheduler=None,
                dropout=.3,horizontal_flip_probability=.5,vertical_flip_probability=.5,
                losses=['normal','ponderada'],weighted_rule='N0/N1 training cuts',
                aggregation='mean_probabilities',threshold=.5,debug=debug,use_exact_cache=True,
                input_representation='raw',source_repository='Antonioaguilera2005/caso_analisis_imagenes_cancer',
                source_commit=COMMIT,model=model_config('raw').to_dict())


def load_recipe(path):
    spec=json.loads(Path(path).read_text())
    expected=json.loads(json.dumps(recipe(spec.get('debug',False))))
    if type(spec.get('debug')) is not bool or spec!=expected:
        raise ValueError('La configuración no coincide con esta receta prefijada; crear otra identidad para cambiarla.')
    return spec


def synchronized_flips(image):
    """Dos sorteos CPU, mismos volteos en las tres fases; sin interpolar."""
    if torch.rand(())<.5:
        image=torch.flip(image,dims=(-1,))
    if torch.rand(())<.5:
        image=torch.flip(image,dims=(-2,))
    return image


def set_seed(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True,warn_only=True)


def split_rows(rows,fold=0,debug=False):
    # No se usa test, ni clínica ni otra partición interna en esta reproducción.
    rows=rows.loc[rows.split.eq('train')].copy()
    if rows.groupby('patient_id').pCR.nunique().max()!=1 or rows.groupby('patient_id').fold.nunique().max()!=1:
        raise ValueError('Paciente con etiqueta/fold incoherente')
    train=rows.loc[rows.fold.ne(fold)].reset_index(drop=True)
    validation=rows.loc[rows.fold.eq(fold)].reset_index(drop=True)
    if debug:
        def limit(frame,negative):
            labels=frame.groupby('patient_id',sort=True).pCR.first()
            ids=list(labels[labels==0].index[:negative])+list(labels[labels==1].index[:2])
            return frame.loc[frame.patient_id.isin(ids)].reset_index(drop=True)
        train,validation=limit(train,3),limit(validation,2)
    if set(train.patient_id)&set(validation.patient_id):
        raise ValueError('Solape de pacientes')
    if any(f.pCR.nunique()!=2 for f in (train,validation)):
        raise ValueError('Se necesitan ambas clases')
    return train,validation


def positive_weight(rows,loss):
    if not rows.split.eq('train').all():raise ValueError('Peso solo a partir de train público')
    if loss=='normal':return 1.
    if loss!='ponderada':raise ValueError('Pérdida inválida')
    labels=rows.pCR.to_numpy()
    n0,n1=int((labels==0).sum()),int((labels==1).sum())
    if not n0 or not n1:raise ValueError('Ambas clases necesarias en train')
    return n0/n1


def loader(rows,spec,training=False):
    cls=CachedDataset if spec['use_exact_cache'] else BreastDCEDataset
    ds=cls(rows,'breastdcedl',transform=synchronized_flips if training else None)
    # También los loaders de evaluación tienen generador propio como el publicado:
    # iterarlos no cambia el RNG global que usa dropout en la época siguiente.
    return DataLoader(ds,batch_size=spec['batch_size'],shuffle=training,
                      num_workers=0,generator=torch.Generator().manual_seed(spec['seed']))


def hashes():
    names=['train_c17_published_recipe.py','simple_cnn_comparison.py','architectures.py',
           'phase_representation.py','triplet_standardization.py','data.py',
           'campaign_20261015_cache.py','metrics.py']
    return {n:hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest() for n in names}


def synchronize(device):
    if device.type=='mps':torch.mps.synchronize()


def stop_requested():
    if (ROOT/'STOP').exists():raise InterruptedError('Parada solicitada mediante archivo STOP')


@torch.inference_mode()
def evaluate(model,batches,rows,device,weight):
    model.eval();logits=[];ids=[]
    for batch in batches:
        stop_requested()
        logits.append(model(batch['image'].to(device)).cpu())
        ids.extend(batch['sample_id'])
    if ids!=rows.sample_id.tolist():raise ValueError('Inferencia desordenada/incompleta')
    z=torch.cat(logits);y=torch.tensor(rows.pCR.to_numpy(),dtype=torch.float32)
    if not torch.isfinite(z).all():raise ValueError('Logits no finitos')
    # Como el código publicado: sigmoid y BCE de evaluación sobre logits CPU.
    probabilities=torch.sigmoid(z).numpy()
    cuts=pd.DataFrame(dict(sample_id=rows.sample_id,patient_id=rows.patient_id,target=y.numpy(),probability=probabilities))
    patients=aggregate_by_patient(cuts.patient_id,cuts.target,cuts.probability)
    metric=enrich(binary_metrics(patients.target,patients.probability,.5))
    metric['precision_defined']=bool(metric['confusion_matrix']['tp']+metric['confusion_matrix']['fp'])
    precision,recall,_=precision_recall_curve(patients.target,patients.probability)
    p=np.clip(patients.probability.to_numpy(float),1e-7,1-1e-7);target=patients.target.to_numpy(float)
    result=dict(cut_bce=float(nn.functional.binary_cross_entropy_with_logits(z,y)),
                objective_cut_bce=float(nn.functional.binary_cross_entropy_with_logits(z,y,pos_weight=torch.tensor(weight))),
                patient_bce=float(np.mean(-(target*np.log(p)+(1-target)*np.log1p(-p)))),
                patient_metrics=metric,patient_pr_auc_trapezoid=float(auc(recall,precision)))
    return result,cuts,patients


def save_checkpoint(path,value):
    tmp=path.with_suffix('.pt.tmp');torch.save(value,tmp);tmp.replace(path)


def load_checkpoint(path,device='cpu'):
    ck=torch.load(path,map_location=device,weights_only=False)
    if ck.get('model_class')!='C17_BreastPCRNet_97809_v1' or ck.get('recipe',{}).get('schema')!=SCHEMA:
        raise ValueError('Checkpoint ajeno a C17')
    if ck['recipe']!=json.loads(json.dumps(recipe(ck['recipe'].get('debug',False)))):
        raise ValueError('Receta del checkpoint distinta de C17')
    model=make_model('raw').to(device);model.load_state_dict(ck['model_state_dict'],strict=True)
    if not all(torch.isfinite(x).all() for x in model.state_dict().values()):raise ValueError('Pesos no finitos')
    return model.eval(),ck


def run_one(spec,loss,device,live=False):
    name=f'{loss}/fold_0_seed_42'
    out=(ROOT/'smoke_resultados'/str(device) if spec['debug'] else ROOT/'resultados')/name
    weights=(WEIGHTS/'smoke'/str(device) if spec['debug'] else WEIGHTS/'cientifico')/name
    resolved=dict(recipe=spec,loss=loss,device=str(device),output_dir=str(out),checkpoint_dir=str(weights))
    if (out/'summary.json').exists():
        old=json.loads((out/'summary.json').read_text())
        if (json.loads((out/'config_resolved.json').read_text())!=resolved or
                old['source_sha256']!=hashes() or old['status']!='complete'):
            raise ValueError('Ejecución existente distinta; conservarla y usar otra identidad')
        saved=Path(old['selection']['checkpoint'])
        if not saved.is_file() or hashlib.sha256(saved.read_bytes()).hexdigest()!=old['selection']['checkpoint_sha256']:
            raise ValueError('Checkpoint completo ausente o alterado; no reutilizar')
        print(f'{loss}: ya completo, se conserva sin volver a entrenar.',flush=True)
        return old
    if out.exists() or weights.exists():raise FileExistsError(f'Parcial existente: {out}; sin sobrescribir ni reanudar automáticamente')
    set_seed(spec['seed']);torch.set_num_threads(4)
    train,validation=split_rows(load_samples(),spec['fold'],spec['debug'])
    weight=positive_weight(train,loss)
    model=make_model('raw').to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=spec['learning_rate'],weight_decay=spec['weight_decay'])
    train_batches=loader(train,spec,True);validation_batches=loader(validation,spec);train_eval_batches=loader(train,spec)
    criterion=nn.BCEWithLogitsLoss(pos_weight=torch.tensor(weight,device=device))
    out.mkdir(parents=True);weights.mkdir(parents=True)
    write_json(out/'config_resolved.json',resolved)
    write_json(out/'split_manifest.json',dict(train=train.patient_id.unique().tolist(),validation=validation.patient_id.unique().tolist(),test_evaluated=False))
    train[['sample_id','patient_id','pCR','fold']].to_csv(out/'particion_train.csv',index=False)
    validation[['sample_id','patient_id','pCR','fold']].to_csv(out/'particion_val.csv',index=False)
    provenance=hashes();environment=dict(python=platform.python_version(),torch=str(torch.__version__),numpy=np.__version__,pandas=pd.__version__,
        platform=platform.platform(),device=str(device),source_sha256=provenance,
        samples_sha256=hashlib.sha256(Path('breastdcedl/metadata/samples.csv').read_bytes()).hexdigest(),
        cache_note='Caché uint8 exacta, sin estadísticas aprendidas; solo train público',
        runtime_adaptations='MPS local, cuatro hilos CPU, datasets propios con IDs; misma receta publicada, sin garantía de igualdad CUDA/MPS')
    write_json(out/'environment.json',environment)
    def log(message):
        print(message,flush=True)
        with (out/'training.log').open('a') as file:file.write(message+'\n')
    log(f'{loss.upper()} | CNN97809 desde cero | MPS/CPU local={device} | train={train.patient_id.nunique()} val={validation.patient_id.nunique()} | pos_weight={weight:.6f}')
    log(f'Adam lr0.001 fijo WD0.0001 batch16 | flips H/V0.5 | max{spec["max_epochs"]}/paciencia8 | selección por AUCval desde época1')
    if spec['debug']:log('SMOKE TÉCNICO: 1época,5train/4val; no es resultado científico.')
    history=[];best=-float('inf');best_epoch=0;stale=0;start=time.perf_counter();display_handle=None
    try:
        for epoch in range(1,spec['max_epochs']+1):
            stop_requested();synchronize(device);before=time.perf_counter();model.train();total=0.;count=0
            for batch in train_batches:
                stop_requested();optimizer.zero_grad(set_to_none=True)
                y=batch['target'].to(device);value=criterion(model(batch['image'].to(device)),y)
                if not torch.isfinite(value):raise ValueError('Pérdida no finita')
                value.backward();optimizer.step();total+=float(value.detach().cpu())*len(y);count+=len(y)
            train_eval,train_cuts,train_patients=evaluate(model,train_eval_batches,train,device,weight)
            val_eval,val_cuts,val_patients=evaluate(model,validation_batches,validation,device,weight)
            synchronize(device)
            score=val_eval['patient_metrics']['roc_auc']
            row=dict(epoch=epoch,train_loss=total/count,train_eval_loss=train_eval['cut_bce'],validation_loss=val_eval['cut_bce'],
                train_objective_eval_loss=train_eval['objective_cut_bce'],validation_objective_loss=val_eval['objective_cut_bce'],
                train_patient_bce=train_eval['patient_bce'],validation_patient_bce=val_eval['patient_bce'],
                train_patient_auc=train_eval['patient_metrics']['roc_auc'],patient_roc_auc=score,
                patient_average_precision=val_eval['patient_metrics']['pr_auc'],
                patient_pr_auc_trapezoid=val_eval['patient_pr_auc_trapezoid'],learning_rate=optimizer.param_groups[0]['lr'],
                elapsed_seconds=time.perf_counter()-before)
            history.append(row);row['mean_epoch_auc']=float(np.mean([x['patient_roc_auc'] for x in history]))
            pd.DataFrame(history).to_csv(out/'history.csv',index=False)
            checkpoint=weights/f'epoch_{epoch:03d}.pt'
            save_checkpoint(checkpoint,dict(model_class='C17_BreastPCRNet_97809_v1',recipe=spec,loss=loss,epoch=epoch,
                model_state_dict=model.state_dict(),optimizer_state_dict=optimizer.state_dict(),source_sha256=provenance,
                train_eval=train_eval,validation_eval=val_eval))
            if score>best:
                best,best_epoch,stale=score,epoch,0
                for partition,cuts,patients in [('train',train_cuts,train_patients),('validation',val_cuts,val_patients)]:
                    cuts.to_csv(out/f'predictions_{partition}_cuts_best.csv',index=False)
                    patients.to_csv(out/f'predictions_{partition}_patients_best.csv',index=False)
                selected=dict(epoch=epoch,checkpoint=str(checkpoint),checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                              train_eval=train_eval,validation_eval=val_eval,selection='Máxima AUCval real por paciente; primer máximo en empate')
                write_json(out/'selection.json',selected)
            else:stale+=1
            write_json(out/'progress.json',dict(status='running',epoch=epoch,best_epoch=best_epoch,stale=stale,mean_epoch_auc=row['mean_epoch_auc']))
            log(f'epoch={epoch:02d}/{spec["max_epochs"]} BCEeval train={row["train_eval_loss"]:.4f} val={row["validation_loss"]:.4f} '
                f'AUCpaciente train={row["train_patient_auc"]:.4f} val={score:.4f} media_épocas={row["mean_epoch_auc"]:.4f} '
                f'AP={row["patient_average_precision"]:.4f} best={best:.4f}@{best_epoch} sinmejora={stale}/8 sec={row["elapsed_seconds"]:.1f}')
            from src.c17_results import build,figure
            if spec['debug']:figure(out,selected)
            build()
            if live:
                from IPython import get_ipython
                if get_ipython() is not None:
                    from IPython.display import Image,display
                    picture=Image(filename=str(out/'curvas_y_diagnostico.png'))
                    if display_handle is None:display_handle=display(picture,display_id=True)
                    else:display_handle.update(picture)
            if stale>=spec['patience']:
                log('Parada: 8épocas sin superar la máxima AUCval; no se prolonga el entrenamiento.');break
        summary=dict(status='complete',experiment_id=f'C17_{loss}',scientific_result=not spec['debug'],
                     debug_patient_subset=spec['debug'],test_evaluated=False,parameter_count=97809,fold=spec['fold'],
                     seed=spec['seed'],loss=loss,positive_weight=weight,training_negative_cuts=int(train.pCR.eq(0).sum()),
                     training_positive_cuts=int(train.pCR.eq(1).sum()),best_epoch=best_epoch,epochs_completed=len(history),
                     mean_epoch_auc=float(np.mean([x['patient_roc_auc'] for x in history])),
                     source_sha256=provenance,selection=selected,elapsed_seconds=time.perf_counter()-start,
                     memory=dict(host_maxrss_raw=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                                 units='bytes' if platform.system()=='Darwin' else 'KiB',
                                 mps_allocated_at_end=torch.mps.current_allocated_memory() if device.type=='mps' else None,
                                 mps_driver_at_end=torch.mps.driver_allocated_memory() if device.type=='mps' else None,
                                 note='RSS es pico del proceso acumulado; valores MPS al cierre, no pico GPU'),
                     note='Fold de desarrollo usado para seleccionar época; no exterior independiente ni media5folds. Media épocas es descriptiva, no criterio de selección.')
        write_json(out/'summary.json',summary);write_json(out/'progress.json',dict(status='complete',epoch=len(history)))
        log(f'COMPLETO {loss}: bestAUC={best:.4f}, media{len(history)}épocas={summary["mean_epoch_auc"]:.4f}')
        from src.c17_results import build
        build();return summary
    except BaseException as error:
        write_json(out/'partial.json',dict(status='partial',error=str(error),epochs_completed=len(history)))
        raise
    finally:
        del model,optimizer
        if device.type=='mps':torch.mps.empty_cache()


def run_config(path,device_name='mps',live=False):
    spec=load_recipe(path)
    if device_name not in ('cpu','mps'):raise ValueError('Solo CPU/MPS locales')
    device=torch.device(device_name)
    if device.type=='mps' and not torch.backends.mps.is_available():raise RuntimeError('MPS no disponible; sin fallback automático')
    ROOT.mkdir(parents=True,exist_ok=True)
    lock=ROOT/'active_process.json';fd=os.open(lock,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    with os.fdopen(fd,'w') as file:json.dump(dict(pid=os.getpid(),started_utc=datetime.now(timezone.utc).isoformat()),file)
    try:
        if not spec['debug']:
            # Preflight obligatorio al ejecutar el estudiante: reutiliza una prueba
            # técnica idéntica ya completa, o la ejecuta antes de los científicos.
            technical=json.loads(json.dumps(recipe(True)))
            for loss in technical['losses']:run_one(technical,loss,device,live)
        return [run_one(spec,loss,device,live) for loss in spec['losses']]
    finally:lock.unlink()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    parser.add_argument('--device',choices=['cpu','mps'],default='mps')
    args=parser.parse_args();run_config(args.config,args.device)
