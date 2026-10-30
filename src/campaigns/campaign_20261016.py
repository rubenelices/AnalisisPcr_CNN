"""Campaña local de tres horas: reutiliza datos, evaluación y figuras de C17."""
from src import source_path
from datetime import datetime, timezone
from dataclasses import replace
from pathlib import Path
import argparse
import copy
import hashlib
import json
import os
import platform
import resource
import time

import nbformat
import numpy as np
import pandas as pd
import torch
from torch import nn

from src.training import train_c17_published_recipe as base
from src.models.architectures import BreastPCRNet
from src.campaigns.campaign_20261015_fast import write_json
from src.campaigns.campaign_deadline import WallClockDeadline
from src.reporting.c17_results import figure, png_cell

ROOT=Path('experiments/campana_20261016')
WEIGHTS=Path('checkpoints/experimentos_equilibrados/campana_20261016')
DEADLINE=datetime.fromisoformat('2026-10-16T12:41:45+00:00')
TAG='campaign_20261016'


def remaining():
    return (DEADLINE-datetime.now(timezone.utc)).total_seconds()


def check():
    if remaining()<=0 or (ROOT/'STOP').exists():raise InterruptedError('Fin de ventana o parada humana')


def spec(name,fold=0,loss='normal',compact=False,representation='raw',debug=False,**changes):
    cfg=replace(base.model_config(representation),channels=(8,16,32,64) if compact else (16,32,64,128))
    return dict(schema='local_campaign_20261016_v1',experiment_id=name,fold=fold,seed=42,
        loss=loss,model=json.loads(json.dumps(cfg.to_dict())),batch_size=16,num_workers=0,
        learning_rate=.0003,weight_decay=.0001,max_epochs=1 if debug else 32,patience=8,
        optimizer='Adam',scheduler=None,aggregation='mean_probabilities',threshold=.5,
        selection='maximum_validation_patient_auc',debug=debug,use_exact_cache=True,
        augmentation='synchronized_horizontal_vertical_flips_0.5',**changes)


def make_model(s):
    from src.models.architectures import BaseCNNConfig
    cfg=dict(s['model']);cfg['channels']=tuple(cfg['channels'])
    return BreastPCRNet(BaseCNNConfig(**cfg))


def source_hashes():
    names=set(base.hashes())|{'campaign_20261016.py','campaign_deadline.py','c17_results.py','generalization_panels.py'}
    return {n:hashlib.sha256(source_path(n).read_bytes()).hexdigest() for n in sorted(names)}


class PatientCutSampler(torch.utils.data.Sampler):
    """Un corte por paciente y época; RNG propio, sin pacientes de validación."""
    def __init__(self,rows,seed):
        if not rows.split.eq('train').all():raise ValueError('Solo train público')
        if not rows.index.equals(pd.RangeIndex(len(rows))):raise ValueError('Índice debe ser consecutivo')
        self.groups=[torch.tensor(g.index.to_numpy()) for _,g in rows.groupby('patient_id',sort=True)]
        self.generator=torch.Generator().manual_seed(seed)
    def __len__(self):return len(self.groups)
    def __iter__(self):
        selected=[int(g[torch.randint(len(g),(1,),generator=self.generator)]) for g in self.groups]
        order=torch.randperm(len(selected),generator=self.generator).tolist()
        return iter([selected[i] for i in order])


def horizontal_only(image):
    """Retira solo V; conserva dos sorteos CPU para el control de C17."""
    if torch.rand(())<.5:image=torch.flip(image,(-1,))
    torch.rand(())
    return image


def c23_folders(completed_only=False):
    """Una identidad por fold; el reinicio completo sustituye solo al parcial en CV."""
    result=[]
    for fold in range(5):
        original=ROOT/'resultados'/f'C23_un_corte_paciente_normal_fold{fold}'
        restart=ROOT/'resultados'/f'C23_un_corte_paciente_reinicio_fold{fold}'
        folder=original if (original/'summary.json').exists() else restart if (restart/'summary.json').exists() else original
        if not completed_only and not (original/'summary.json').exists() and (restart/'history.csv').exists():folder=restart
        if (folder/'history.csv').exists() and (not completed_only or (folder/'summary.json').exists()):result.append(folder)
    return result


def report():
    from src.evaluation.generalization_panels import diagnostics,fold_overview
    rows=[];details=[]
    for folder in sorted((ROOT/'resultados').glob('*')):
        if not (folder/'history.csv').exists():continue
        h=pd.read_csv(folder/'history.csv');s=json.loads((folder/'config_resolved.json').read_text())
        selected=json.loads((folder/'selection.json').read_text())
        m=selected['validation_eval']['patient_metrics'];tr=selected['train_eval']['patient_metrics']
        state='Completo' if (folder/'summary.json').exists() else 'Parcial/interrumpido' if (folder/'partial.json').exists() else 'En curso'
        rows.append(dict(Experimento=s['experiment_id'],Fold=s['fold'],Estado=state,Épocas=len(h),
            MejorÉpoca=selected['epoch'],AUCval=m['roc_auc'],AUCtrain=tr['roc_auc'],
            Brecha=tr['roc_auc']-m['roc_auc'],MediaÉpocas=float(h.patient_roc_auc.mean()),
            AP=m['pr_auc'],Brier=m['brier_score'],F1macro=m['f1_macro'],BA=m['balanced_accuracy'],
            Precision=m['precision'] if m['precision_defined'] else None,Sensibilidad=m['sensitivity'],
            **{k.upper():v for k,v in m['confusion_matrix'].items()}))
        figure(folder,selected,title=f'{s["experiment_id"]} · fold{s["fold"]} · {len(h)} épocas · mejor {selected["epoch"]}')
        rel=folder.relative_to(ROOT).as_posix()
        text=f'## {s["experiment_id"]} · fold{s["fold"]} · {state}\n\n'
        text+='AUCval = antiguo patient_roc_auc. Train se evalúa sin aumentos/dropout. '
        text+='La validación selecciona época; no es evaluación independiente. Matriz a0,5, sin ajustar umbral.\n\n'
        text+=f'[Config]({rel}/config_resolved.json) · [Historial]({rel}/history.csv) · [Selección]({rel}/selection.json)\n\n'
        text+=h.round(4).to_markdown(index=False)
        cell=png_cell(folder/'curvas_y_diagnostico.png')
        diagnostic=diagnostics(folder,selected,f'{s["experiment_id"]} · fold{s["fold"]}')
        text+='\n\n**Diagnóstico de sobreajuste:** priorizar pérdidas/AUC por paciente en eval. La pérdida online mezcla actualizaciones de pesos, aumentos y dropout; se muestra aparte, no como comparación equivalente. Barras de la misma época; test reservado.\n'
        details.extend([nbformat.v4.new_markdown_cell(text),cell,png_cell(diagnostic)])
    table=pd.DataFrame(rows);table.to_csv(ROOT/'resumen.csv',index=False)
    intro='# Experimentos locales · 16 octubre\n\n'
    intro+='Solo train público; test cerrado. '
    intro+='CNN desde cero, sin clínica ni pesos externos. Búsqueda exploratoria: '
    intro+='no declarar ganador por un pico ni presentar selección en validación como test.\n\n'
    intro+=table.round(4).to_markdown(index=False) if len(table) else 'Primer experimento en preparación.'
    cv=[]
    completed=c23_folders(completed_only=True)
    for folder in completed:
        summary=json.loads((folder/'summary.json').read_text())
        c=json.loads((folder/'config_resolved.json').read_text())
        cv.append(dict(fold=c['fold'],**summary['selection']['validation_eval']['patient_metrics']))
    intro+=f'\n\n**C23 · muestreo de un corte por paciente: {len(cv)}/5 folds completos.** '
    intro+='Cada época procesa una vez cada paciente; eval usa todos sus cortes. No comparar mediaépocas con la unidad de épocaC17. '
    if len(cv)==5:
        if {r['fold'] for r in cv}!={0,1,2,3,4}:raise ValueError('Folds repetidos o incompletos')
        frames=[pd.read_csv(folder/'predictions_validation_patients_best.csv') for folder in completed]
        heldout=pd.concat(frames,ignore_index=True)
        if heldout.patient_id.duplicated().any() or len(heldout)!=1097:raise ValueError('CV no cubre1097pacientes disjuntas')
        heldout.to_csv(ROOT/'C23_validation_patients.csv',index=False)
        stats=dict(folds=5,patients=1097,selection_independent=False,
            roc_auc_mean=float(np.mean([r['roc_auc'] for r in cv])),roc_auc_sd=float(np.std([r['roc_auc'] for r in cv],ddof=1)),
            average_precision_mean=float(np.mean([r['pr_auc'] for r in cv])),
            balanced_accuracy_mean=float(np.mean([r['balanced_accuracy'] for r in cv])),f1_macro_mean=float(np.mean([r['f1_macro'] for r in cv])),
            confusion_matrix_sum={k:sum(r['confusion_matrix'][k] for r in cv) for k in ('tn','fp','fn','tp')},
            note='Media de cincofolds de desarrollo: cada validación seleccionó época; no exterior independiente. Trains solapados.')
        write_json(ROOT/'C23_cv_summary.json',stats)
        intro+=f'ROC-AUC medio de validación={stats["roc_auc_mean"]:.4f}, SD={stats["roc_auc_sd"]:.4f}. '
    intro+='**La validación de cada fold selecciona checkpoint; no es evaluación exterior independiente.** No adoptar un ganador por esta búsqueda exploratoria.'
    intro+='\n\n[Plan y decisiones](README.md). Uso educativo, sin validez clínica.'
    overview=fold_overview(c23_folders(),ROOT/'C23_sobreajuste_folds.png')
    nb=nbformat.v4.new_notebook(cells=[nbformat.v4.new_markdown_cell(intro)]+([png_cell(overview)] if overview else [])+details)
    nb.metadata.kernelspec=dict(name='python3',display_name='Python 3',language='python')
    p=ROOT/'RESULTADOS.ipynb';tmp=p.with_suffix('.tmp.ipynb');nbformat.write(nb,tmp);tmp.replace(p)
    main=ROOT.parent/'RESULTADOS.ipynb';old=nbformat.read(main,as_version=4)
    keep=[c for c in old.cells if c.metadata.get('campaign_overview')!=TAG and c.metadata.get('embedded_campaign')!=TAG]
    new=copy.deepcopy(nb.cells)
    import re
    for i,c in enumerate(new):
        c.metadata['campaign_overview' if i==0 else 'embedded_campaign']=TAG
        if c.cell_type=='markdown':
            c.source=re.sub(r'\]\((?![A-Za-z]+:|/|#)([^)]+)\)',lambda m:'](campana_20261016/'+m.group(1)+')',c.source)
    old.cells=[new[0]]+keep+new[1:]
    tmp=main.with_name('RESULTADOS.camp16.tmp.ipynb');nbformat.write(old,tmp);tmp.replace(main)


def run(s,device_name='mps'):
    check()
    if not s['debug'] and remaining()<900:raise InterruptedError('Últimos15min reservados al cierre')
    if device_name not in ('mps','cpu'):raise ValueError('Solo local')
    device=torch.device(device_name)
    if device.type=='mps' and not torch.backends.mps.is_available():raise RuntimeError('MPS no disponible')
    name=s['experiment_id']+f'_fold{s["fold"]}'
    out=ROOT/('smoke' if s['debug'] else 'resultados')/name
    weights=WEIGHTS/('smoke' if s['debug'] else 'cientifico')/name
    if out.exists() or weights.exists():raise FileExistsError('Identidad existente: no sobrescribir')
    base.set_seed(s['seed']);torch.set_num_threads(4)
    train,val=base.split_rows(base.load_samples(),s['fold'],s['debug'])
    weight_rows=train.drop_duplicates('patient_id') if s.get('weighted_rule')=='training_patient_counts' else train
    weight=base.positive_weight(weight_rows,s['loss']);model=make_model(s).to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=s['learning_rate'],weight_decay=s['weight_decay'])
    criterion=nn.BCEWithLogitsLoss(pos_weight=torch.tensor(weight,device=device))
    tr=base.loader(train,s,True);te=base.loader(train,s);va=base.loader(val,s)
    if s.get('one_cut_per_patient',False) or s['augmentation']=='horizontal_only':
        transform=horizontal_only if s['augmentation']=='horizontal_only' else base.synchronized_flips
        sampler=PatientCutSampler(train,s['seed']) if s.get('one_cut_per_patient',False) else None
        tr=torch.utils.data.DataLoader(base.CachedDataset(train,transform=transform),
            sampler=sampler,shuffle=sampler is None,batch_size=s['batch_size'],num_workers=0,
            generator=torch.Generator().manual_seed(s['seed']))
    out.mkdir(parents=True);weights.mkdir(parents=True)
    provenance=source_hashes()
    for file,sha in provenance.items():
        snapshot=ROOT/'source_snapshots'/sha;snapshot.mkdir(parents=True,exist_ok=True)
        (snapshot/(file+'.txt')).write_bytes(source_path(file).read_bytes())
    write_json(out/'config_resolved.json',s)
    write_json(out/'environment.json',dict(device=str(device),python=platform.python_version(),torch=str(torch.__version__),
        source_sha256=provenance,parameters=sum(p.numel() for p in model.parameters()),
        started_utc=datetime.now(timezone.utc).isoformat(),test_evaluated=False))
    write_json(out/'deadline.json',dict(deadline_utc=DEADLINE.isoformat()))
    write_json(out/'split_manifest.json',dict(train=train.patient_id.unique().tolist(),validation=val.patient_id.unique().tolist(),test_evaluated=False))
    history=[];best=-float('inf');stale=0;selected=None;start=time.perf_counter();bce_best=float('inf');mps_observed=0
    # Tanto evaluación reutilizada como entrenamiento consultan el mismo plazo.
    previous_check=base.stop_requested;base.stop_requested=check
    try:
        for epoch in range(1,s['max_epochs']+1):
            check();before=time.perf_counter();model.train();total=0.;n=0
            for batch in tr:
                check();optimizer.zero_grad(set_to_none=True)
                y=batch['target'].to(device);loss=criterion(model(batch['image'].to(device)),y)
                if not torch.isfinite(loss):raise ValueError('Pérdida no finita')
                loss.backward();optimizer.step();total+=float(loss.detach().cpu())*len(y);n+=len(y)
                if device.type=='mps':mps_observed=max(mps_observed,torch.mps.current_allocated_memory())
            train_eval,tc,tp=base.evaluate(model,te,train,device,weight)
            val_eval,vc,vp=base.evaluate(model,va,val,device,weight)
            score=val_eval['patient_metrics']['roc_auc']
            row=dict(epoch=epoch,train_loss=total/n,train_eval_loss=train_eval['cut_bce'],validation_loss=val_eval['cut_bce'],
                train_objective_eval_loss=train_eval['objective_cut_bce'],validation_objective_loss=val_eval['objective_cut_bce'],
                train_patient_bce=train_eval['patient_bce'],validation_patient_bce=val_eval['patient_bce'],
                train_patient_auc=train_eval['patient_metrics']['roc_auc'],patient_roc_auc=score,
                patient_average_precision=val_eval['patient_metrics']['pr_auc'],learning_rate=optimizer.param_groups[0]['lr'],
                training_cuts=n,elapsed_seconds=time.perf_counter()-before)
            history.append(row);row['mean_epoch_auc']=float(np.mean([r['patient_roc_auc'] for r in history]))
            pd.DataFrame(history).to_csv(out/'history.csv',index=False)
            ck=dict(model_class='BreastPCRNet_campaign_20261016_v1',config=s,epoch=epoch,
                model_state_dict=model.state_dict(),optimizer_state_dict=optimizer.state_dict(),
                train_eval=train_eval,validation_eval=val_eval,source_sha256=provenance)
            base.save_checkpoint(weights/f'epoch_{epoch:03d}.pt',ck)
            if val_eval['patient_bce']<bce_best:
                bce_best=val_eval['patient_bce'];write_json(out/'minimum_bce.json',dict(epoch=epoch,train_eval=train_eval,validation_eval=val_eval))
            if score>best:
                best=score;stale=0
                selected=dict(epoch=epoch,checkpoint=str(weights/f'epoch_{epoch:03d}.pt'),train_eval=train_eval,validation_eval=val_eval)
                write_json(out/'selection.json',selected)
                for part,cuts,patients in [('train',tc,tp),('validation',vc,vp)]:
                    cuts.to_csv(out/f'predictions_{part}_cuts_best.csv',index=False)
                    patients.to_csv(out/f'predictions_{part}_patients_best.csv',index=False)
            else:stale+=1
            line=f'{name} epoch={epoch:02d} BCE train/val={train_eval["cut_bce"]:.4f}/{val_eval["cut_bce"]:.4f} AUC train/val={row["train_patient_auc"]:.4f}/{score:.4f} best={best:.4f}@{selected["epoch"]} stale={stale}/{s["patience"]} sec={row["elapsed_seconds"]:.1f}'
            print(line,flush=True)
            with (out/'training.log').open('a') as f:f.write(line+'\n')
            if not s['debug']:report()
            if stale>=s['patience']:break
        if source_hashes()!=provenance:raise RuntimeError('Fuentes cambiadas durante fit')
        write_json(out/'summary.json',dict(status='complete',scientific_result=not s['debug'],test_evaluated=False,
            epochs_completed=len(history),selection=selected,positive_weight=weight,elapsed_seconds=time.perf_counter()-start,
            memory=dict(host_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss if platform.system()=='Darwin' else resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                mps_max_observed_after_steps_bytes=mps_observed,
                note='RSS pico del proceso; MPS máximo muestreado tras pasos, no pico de temporales GPU'),
            source_sha256=provenance,note='Validación de desarrollo selecciona checkpoint, no test independiente'))
        if not s['debug']:report()
    except BaseException as e:
        write_json(out/'partial.json',dict(status='partial',error=str(e),epochs_completed=len(history)))
        raise
    finally:
        base.stop_requested=previous_check
        del model,optimizer
        if device.type=='mps':torch.mps.empty_cache()


def main():
    global DEADLINE
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--device',default='mps',choices=['mps','cpu']);p.add_argument('--deadline');args=p.parse_args()
    if args.deadline:
        DEADLINE=datetime.fromisoformat(args.deadline)
        if DEADLINE.tzinfo is None:raise ValueError('Plazo requiere zona horaria')
    ROOT.mkdir(exist_ok=True);lock=ROOT/'active_process.json'
    fd=os.open(lock,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    with os.fdopen(fd,'w') as f:json.dump(dict(pid=os.getpid(),started=datetime.now(timezone.utc).isoformat()),f)
    guard=WallClockDeadline(remaining,lambda:os._exit(124));guard.start()
    try:run(json.loads(Path(args.config).read_text()),args.device)
    finally:guard.cancel();lock.unlink(missing_ok=True)


if __name__=='__main__':main()
