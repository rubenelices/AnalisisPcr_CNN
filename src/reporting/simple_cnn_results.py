"""Tablas, diagnósticos y notebook de C16; solo lee resultados terminados."""
from datetime import datetime, timezone
import base64
import copy
import hashlib
import json
from pathlib import Path
import re

import nbformat
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score

from src.training.simple_cnn_comparison import ROOT, REPRESENTATIONS, source_hashes
from src.evaluation.metrics import binary_metrics
from src.campaigns.campaign_20261015_fast import enrich, write_json

TAG='comparison_C16_97809'
REFERENCE=Path('local/campana_20261015')
LABELS={'raw':'CNN 97.809 · fases originales',
        'pre_differences':'CNN 97.809 · diferencias',
        'reference16':'E57 · referencia16'}


def image_cell(paths):
    source="from pathlib import Path\nfrom IPython.display import Image, display\n_project = next(p for p in (Path.cwd(), *Path.cwd().parents) if (p/'src').is_dir() and (p/'app.py').is_file())\n"
    outputs=[]
    for path in paths:
        path=Path(path)
        source+=f'display(Image(filename=str(_project / {path.as_posix()!r})))\n'
        outputs.append(nbformat.v4.new_output('display_data',data={
            'image/png':base64.b64encode(path.read_bytes()).decode('ascii'),
            'text/plain':f'Figura guardada: {path.name}'}))
    return nbformat.v4.new_code_cell(source,outputs=outputs,
        metadata=dict(jupyter=dict(source_hidden=True),collapsed=True))


def finalize_notebook():
    """Ejecuta solo lectura/visualización de PNG: no entrena ni infiere."""
    from nbclient import NotebookClient
    path=ROOT/'RESULTADOS.ipynb'
    nb=nbformat.read(path,as_version=4)
    NotebookClient(nb,timeout=60,kernel_name='python3',resources={'metadata':{'path':str(Path.cwd())}}).execute()
    if any(o.output_type=='error' for c in nb.cells if c.cell_type=='code' for o in c.outputs):
        raise ValueError('Notebook con errores')
    nbformat.write(nb,path)
    embed(nb,nb.cells[0])
    return sum(c.cell_type=='code' for c in nb.cells)


def records():
    found=[]
    for path in sorted(ROOT.glob('*/resultados/summary.json')):
        s=json.loads(path.read_text())
        if s.get('scientific_result') is True and s.get('status')=='complete':
            found.append((path.parent,s))
    return found


def metric_row(profile, fold, s, reference=False):
    m=s['outer_eval']['metrics'];d=s['outer_decision_metrics']
    fit=s['train_eval' if reference else 'fit_eval']['metrics']
    return dict(Perfil=profile,Fold=fold,AUC=m['roc_auc'],AP=m['pr_auc'],
                Brier=m['brier_score'],F1macro=d['f1_macro'],BalancedAccuracy=d['balanced_accuracy'],
                Precision=d['precision'],Sensibilidad=d['sensitivity'],Especificidad=d['specificity'],
                AUCfit=fit['roc_auc'],AUCinterna=s['inner_eval']['metrics']['roc_auc'],
                BCEfit=s['train_eval' if reference else 'fit_eval']['patient_probability_bce'],
                BCEinterna=s['inner_eval']['patient_probability_bce'],
                BCEexterior=s['outer_eval']['patient_probability_bce'],
                MejorEpoca=s['best_epoch'],Epocas=s['epochs_completed'],Umbral=s['selected_threshold'],
                **d['confusion_matrix'])


def load_table():
    rows=[]
    for _,s in records():
        rows.append(metric_row(s['representation'],s['outer_fold'],s))
    for fold in range(5):
        s=json.loads((REFERENCE/f'reference16_fold{fold}/resultados/summary.json').read_text())
        rows.append(metric_row('reference16',fold,s,True))
    return pd.DataFrame(rows)


def class_table(m):
    c=m['confusion_matrix'];tn,fp,fn,tp=[c[k] for k in ('tn','fp','fn','tp')]
    div=lambda a,b:float(a/b) if b else None
    return pd.DataFrame([
        dict(Clase='no pCR',Precision=div(tn,tn+fn),Recall=div(tn,tn+fp),
             F1=div(2*tn,2*tn+fp+fn),Pacientes=tn+fp),
        dict(Clase='pCR',Precision=div(tp,tp+fp),Recall=div(tp,tp+fn),
             F1=div(2*tp,2*tp+fp+fn),Pacientes=tp+fn)])


def matrix(ax, m, title):
    c=m['confusion_matrix'];a=np.array([[c['tn'],c['fp']],[c['fn'],c['tp']]])
    ax.imshow(a,cmap='Blues',vmin=0)
    for (i,j),v in np.ndenumerate(a):
        ax.text(j,i,str(v),ha='center',va='center',fontsize=13,
                color='white' if v>a.max()/2 else '#243746')
    ax.set(xticks=[0,1],yticks=[0,1],xticklabels=['no pCR','pCR'],yticklabels=['no pCR','pCR'],
           xlabel='Predicción',ylabel='Real',title=title)


def equivalent_losses(predictions):
    """Mismo checkpoint/eval, misma unidad y agregación en las tres particiones."""
    fit_prevalence=float(predictions['fit'].target.mean())
    rows=[]
    for name,frame in predictions.items():
        y=frame.target.to_numpy(float)
        p=np.clip(frame.probability.to_numpy(float),1e-7,1-1e-7)
        model_bce=float(np.mean(-(y*np.log(p)+(1-y)*np.log1p(-p))))
        baseline_bce=float(np.mean(-(y*np.log(fit_prevalence)+(1-y)*np.log1p(-fit_prevalence))))
        rows.append(dict(Particion=name,BCEpacienteEval=model_bce,
                         BCEbaselineConstanteFit=baseline_bce,
                         AUC=roc_auc_score(y,p)))
    return pd.DataFrame(rows)


def run_figure(folder,s):
    import matplotlib.pyplot as plt
    from sklearn.calibration import calibration_curve
    history=pd.read_csv(folder/'history.csv')
    pred={k:pd.read_csv(folder/f'predictions_{k}.csv') for k in ('fit','inner','outer')}
    fig,axes=plt.subplots(2,3,figsize=(17,10),constrained_layout=True)
    ax=axes[0,0]
    ax.plot(history.epoch,history.train_loss,label='BCE corte train online')
    ax.plot(history.epoch,history.validation_loss,label='BCE paciente interna eval')
    ax.axvline(s['best_epoch'],ls=':',color='#666');ax.set(title='Pérdidas: unidades distintas',xlabel='Época',ylabel='BCE');ax.legend(fontsize=8)
    ax=axes[0,1]
    ax.plot(history.epoch,history.patient_roc_auc,label='AUC interna real')
    ax.plot(history.epoch,history.monitor_inner_auc_ma3,label='MA3 interna')
    ax.plot(history.epoch,history.patient_pr_auc,label='AP interna')
    ax.axvline(s['best_epoch'],ls=':',color='#666');ax.set(title='Selección solo interna',xlabel='Época',ylim=(0,1));ax.legend(fontsize=8)
    ax=axes[0,2]
    for k,label in [('fit','Fit'),('inner','Interna'),('outer','Exterior')]:
        p=pred[k];fpr,tpr,_=roc_curve(p.target,p.probability)
        ax.plot(fpr,tpr,label=f'{label}: {roc_auc_score(p.target,p.probability):.3f}')
    ax.plot([0,1],[0,1],'--',color='#999');ax.set(title='Mismo checkpoint · eval · paciente',xlabel='FPR',ylabel='TPR');ax.legend(fontsize=8)
    ax=axes[1,0];p=pred['outer']
    precision,recall,_=precision_recall_curve(p.target,p.probability)
    ax.plot(recall,precision,label=f'AP {s["outer_eval"]["metrics"]["pr_auc"]:.3f}')
    ax.axhline(p.target.mean(),ls='--',color='#999',label=f'Prevalencia {p.target.mean():.3f}')
    ax.set(title='Precisión–recall exterior',xlabel='Sensibilidad',ylabel='Precisión',ylim=(0,1));ax.legend(fontsize=8)
    matrix(axes[1,1],s['outer_eval']['metrics'],'Exterior · umbral fijo 0,5')
    matrix(axes[1,2],s['outer_decision_metrics'],f'Exterior · umbral interno {s["selected_threshold"]:.3f}')
    fig.suptitle(f'{LABELS[s["representation"]]} · fold{s["outer_fold"]} · best{s["best_epoch"]}/{s["epochs_completed"]}',fontsize=16)
    fig.savefig(folder/'diagnostico.png',dpi=130);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(16,4),constrained_layout=True)
    for y,label in [(0,'no pCR'),(1,'pCR')]:
        axes[0].hist(p.loc[p.target==y,'probability'],bins=np.linspace(0,1,21),alpha=.6,density=True,label=label)
    axes[0].axvline(.5,color='#777',ls='--');axes[0].axvline(s['selected_threshold'],color='#333',ls=':')
    axes[0].set(title='Distribución exterior por clase',xlabel='Media de probabilidades',ylabel='Densidad');axes[0].legend()
    fraction,mean=calibration_curve(p.target,p.probability,n_bins=5,strategy='quantile')
    axes[1].plot(mean,fraction,'o-');axes[1].plot([0,1],[0,1],'--',color='#999')
    axes[1].set(title=f'Calibración exterior · Brier{s["outer_eval"]["metrics"]["brier_score"]:.3f}',xlabel='Probabilidad media',ylabel='Fracción pCR',xlim=(0,1),ylim=(0,1))
    losses=equivalent_losses(pred)
    losses.to_csv(folder/'loss_comparison_eval.csv',index=False)
    xx=np.arange(3)
    axes[2].bar(xx-.18,losses.BCEpacienteEval,width=.36,label='Modelo')
    axes[2].bar(xx+.18,losses.BCEbaselineConstanteFit,width=.36,label='Constante: prevalencia fit',alpha=.6)
    axes[2].set(xticks=xx,xticklabels=['Fit','Interna','Exterior'],ylabel='BCE por paciente',
                title='Mismo checkpoint · eval · media probs')
    axes[2].legend(fontsize=8)
    fig.savefig(folder/'probabilidades_calibracion.png',dpi=130);plt.close(fig)


def summary(table):
    rows=[]
    for profile in ('reference16',*REPRESENTATIONS):
        g=table.loc[table.Perfil==profile]
        if len(g)!=5:
            continue
        rows.append(dict(Perfil=profile,AUC=g.AUC.mean(),SD=g.AUC.std(ddof=1),
                         AP=g.AP.mean(),Brier=g.Brier.mean(),F1macro=g.F1macro.mean(),
                         BalancedAccuracy=g.BalancedAccuracy.mean(),Precision=g.Precision.mean(),
                         Sensibilidad=g.Sensibilidad.mean(),Especificidad=g.Especificidad.mean(),
                         AUCfit=g.AUCfit.mean(),BCEfit=g.BCEfit.mean(),BCEinterna=g.BCEinterna.mean(),
                         BCEexterior=g.BCEexterior.mean(),BrechaAUC=(g.AUCfit-g.AUC).mean(),
                         **{k:int(g[k].sum()) for k in ('tn','fp','fn','tp')}))
    return pd.DataFrame(rows)


def verify_patients(profile):
    pieces=[]
    for fold in range(5):
        folder=(REFERENCE/f'reference16_fold{fold}' if profile=='reference16' else ROOT/f'{profile}_fold{fold}')/'resultados'
        pred=pd.read_csv(folder/'predictions_outer.csv');pred['outer_fold']=fold
        manifest=json.loads((folder/'split_manifest.json').read_text())
        if set(pred.patient_id)!=set(manifest['outer']) or len(pred)!=len(manifest['outer']):
            raise ValueError('Exterior no coincide con manifest')
        if not np.isfinite(pred.probability).all() or not pred.probability.between(0,1).all():
            raise ValueError('Probabilidades inválidas')
        pieces.append(pred)
    pred=pd.concat(pieces,ignore_index=True)
    if len(pred)!=1097 or pred.patient_id.nunique()!=1097:
        raise ValueError('Exteriores duplicados/incompletos')
    return pred


def paired_comparisons():
    # Intervalos condicionales a predicciones ya fijas; no incertidumbre del fit.
    frames={p:verify_patients(p) for p in ('reference16',*REPRESENTATIONS)}
    for profile,frame in frames.items():
        frame.to_csv(ROOT/f'heldout_{profile}.csv',index=False)
    rows=[]
    for a,b in [('raw','reference16'),('pre_differences','reference16'),('raw','pre_differences')]:
        joined=frames[a].merge(frames[b],on=['patient_id','target','outer_fold'],suffixes=('_a','_b'),validate='one_to_one')
        if len(joined)!=1097:
            raise ValueError('Emparejamiento incompleto')
        rng=np.random.default_rng(20261015);values=[]
        groups=[g.reset_index(drop=True) for _,g in joined.groupby('outer_fold')]
        deltas=[roc_auc_score(g.target,g.probability_a)-roc_auc_score(g.target,g.probability_b) for g in groups]
        for _ in range(1000):
            diffs=[]
            for g in groups:
                indices=np.concatenate([rng.choice(np.flatnonzero(g.target.to_numpy()==y),size=int((g.target==y).sum()),replace=True) for y in (0,1)])
                sample=g.iloc[indices]
                diffs.append(roc_auc_score(sample.target,sample.probability_a)-roc_auc_score(sample.target,sample.probability_b))
            values.append(np.mean(diffs))
        low,high=np.quantile(values,[.025,.975])
        rows.append(dict(A=a,B=b,delta_auc_media=float(np.mean(deltas)),ci_conditional_low=float(low),
                         ci_conditional_high=float(high),fold_deltas=deltas,
                         note='Bootstrap paciente estratificado dentro fold; predicciones fijas, no variabilidad de entrenar ni test externo.'))
    write_json(ROOT/'paired_comparison.json',rows)
    return rows


def cohort_metrics():
    metadata=pd.read_csv('breastdcedl/metadata/patients.csv',usecols=['pid','dataset','split'])
    metadata=metadata.loc[metadata.split.eq('train'),['pid','dataset']].rename(columns={'pid':'patient_id'})
    rows=[]
    for profile in ('reference16',*REPRESENTATIONS):
        pred=verify_patients(profile).merge(metadata,on='patient_id',validate='many_to_one')
        if len(pred)!=1097:
            raise ValueError('Cohortes incompletas')
        for (fold,cohort),g in pred.groupby(['outer_fold','dataset']):
            rows.append(dict(Perfil=profile,Fold=int(fold),Cohorte=cohort,N=len(g),
                             Positivas=int(g.target.sum()),
                             AUC=roc_auc_score(g.target,g.probability) if g.target.nunique()==2 else None))
    result=pd.DataFrame(rows)
    result.to_csv(ROOT/'cohort_metrics.csv',index=False)
    return result


def embed(nb, overview):
    main=Path('experiments/RESULTADOS.ipynb')
    base=nbformat.read(main,as_version=4)
    old=[c for c in base.cells if c.metadata.get('embedded_campaign')!=TAG and c.metadata.get('campaign_overview')!=TAG]
    overview=copy.deepcopy(overview);overview.metadata['campaign_overview']=TAG
    overview.source=re.sub(r'\]\((?![A-Za-z]+:|/|#)([^)]+)\)',r'](C16_cnn_97809/\1)',overview.source)
    details=[]
    for original in nb.cells[1:]:
        cell=copy.deepcopy(original);cell.metadata['embedded_campaign']=TAG
        if cell.cell_type=='markdown':
            cell.source=re.sub(r'\]\((?![A-Za-z]+:|/|#)([^)]+)\)',r'](C16_cnn_97809/\1)',cell.source)
        details.append(cell)
    base.cells=[overview]+old+details
    temp=main.with_name('RESULTADOS.C16.tmp.ipynb')
    nbformat.write(base,temp);temp.replace(main)


def build():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ROOT.mkdir(exist_ok=True)
    runs=records();table=load_table();totals=summary(table)
    table.to_csv(ROOT/'fold_metrics.csv',index=False)
    totals.to_csv(ROOT/'comparison.csv',index=False)
    done=len(runs)==10
    status='COMPLETO · diez entrenamientos científicos' if done else f'EN CURSO · {len(runs)}/10 científicos completos'
    compact_table=totals[['Perfil','AUC','SD','AP','F1macro','BalancedAccuracy','Precision','Sensibilidad']]
    loss_table=totals[['Perfil','AUCfit','AUC','BrechaAUC','BCEfit','BCEinterna','BCEexterior']]
    interpretation=(ROOT/'LECTURA_BREVE.md').read_text() if (ROOT/'LECTURA_BREVE.md').exists() else ''
    overview_text=f'''# C16 · CNN de 97.809 parámetros · {status}

{interpretation}

Cuatro bloques Conv3×3/BN/ReLU/MaxPool16→32→64→128, GAP y salida lineal.
Fases originales (`raw`) y control con diferencias (`pre_differences`), ambos
prefijados. E57 referencia16 reutilizada, sin reentrenar. Selección/umbral
solo internos; mismos IDs fit/inner/outer. Sin clínica. Test cerrado; todo local.

{compact_table.round(4).to_markdown(index=False)}

### Generalización del checkpoint elegido

{loss_table.round(4).to_markdown(index=False)}

Pérdidas BCE por paciente, mismo checkpoint y modo eval; medias por fold.
BrechaAUC = AUCfit − AUCexterior. La selección usa solo interna. Menor brecha
sin aprendizaje por encima del baseline tampoco cumple el objetivo. No afirmar
que la red generaliza por tener menos parámetros ni por una curva más próxima.

Tabla: medias por fold; matrices sumadas sobre pacientes exteriores disjuntas.
Precisión/recall/F1 usan umbral interno, no0,5; se muestran ambas matrices abajo.
No son folds independientes (train solapado), ni test externo nuevo: datos usados
en desarrollo previo. raw frente E57 cambia arquitectura y representación;
pre_differences frente E57 compara paquete arquitectónico con entrada idéntica.

[Protocolo](PLAN.md) · [Informe](README.md) · [Notebook propio](RESULTADOS.ipynb).
Uso educativo y de investigación, sin validez clínica.
'''
    cells=[nbformat.v4.new_markdown_cell(overview_text)]
    cells.append(nbformat.v4.new_markdown_cell('## Resultados de todos los folds\n\n'+table.round(4).to_markdown(index=False)))
    if done:
        pairs=paired_comparisons()
        cohorts=cohort_metrics()
        note='\n'.join(f'- {x["A"]} − {x["B"]}: ΔAUC media{x["delta_auc_media"]:+.4f}; IC condicional95% [{x["ci_conditional_low"]:+.4f},{x["ci_conditional_high"]:+.4f}]; deltas por fold: '+', '.join(f'{v:+.4f}' for v in x['fold_deltas']) for x in pairs)
        cells.append(nbformat.v4.new_markdown_cell('## Comparación emparejada\n\n'+note+'\n\nBootstrap estratificado por paciente dentro de fold,1000 réplicas/seed20261015, predicciones fijas. No incorpora variabilidad del entrenamiento ni selección histórica; IC descriptivo condicionado. No ROC pooled.'))
        cells.append(nbformat.v4.new_markdown_cell('## Diagnóstico por cohorte y fold\n\n'+cohorts.round(4).to_markdown(index=False)+'\n\nCohorte solo para análisis descriptivo; nunca como entrada. No concluir diferencias firmes en subgrupos pequeños ni usar estos resultados para seleccionar modelos.'))
        fig,axes=plt.subplots(2,4,figsize=(15,8),constrained_layout=True)
        for i,profile in enumerate(('reference16',*REPRESENTATIONS)):
            fixed={k:0 for k in ('tn','fp','fn','tp')};inner={k:0 for k in fixed}
            for fold in range(5):
                folder=(REFERENCE/f'reference16_fold{fold}' if profile=='reference16' else ROOT/f'{profile}_fold{fold}')/'resultados'
                s=json.loads((folder/'summary.json').read_text())
                for k in fixed:
                    fixed[k]+=s['outer_eval']['metrics']['confusion_matrix'][k]
                    inner[k]+=s['outer_decision_metrics']['confusion_matrix'][k]
            matrix(axes[0,i],dict(confusion_matrix=fixed),LABELS[profile]+'\nUmbral0,5')
            matrix(axes[1,i],dict(confusion_matrix=inner),LABELS[profile]+'\nUmbral interno por fold')
        baseline_pred=verify_patients('reference16')
        baseline=dict(confusion_matrix=dict(tn=int((baseline_pred.target==0).sum()),fp=0,fn=int(baseline_pred.target.sum()),tp=0))
        matrix(axes[0,3],baseline,'Baseline: siempre no pCR')
        axes[1,3].axis('off');axes[1,3].text(0,.9,'1.097 pacientes exteriores por perfil\nSin ajustar umbral en exterior.\n\nCada columna usa exactamente\nlas mismas pacientes.\n\nPrecisión sin predicciones positivas:\nno definida (histórico registra0).',va='top',fontsize=11)
        fig.savefig(ROOT/'matrices_comparadas.png',dpi=150);plt.close(fig)
        cells.append(nbformat.v4.new_markdown_cell('## Matrices de confusión\n\n[PNG de las matrices](matrices_comparadas.png)'))
        cells.append(image_cell([ROOT/'matrices_comparadas.png']))
    for folder,s in runs:
        run_figure(folder,s)
        relative=folder.relative_to(ROOT).as_posix()
        text=f'## {s["experiment_id"]} · {s["epochs_completed"]} épocas · best{s["best_epoch"]}\n\n'
        text+=f'ROCfit{s["fit_eval"]["metrics"]["roc_auc"]:.4f} / interna{s["inner_eval"]["metrics"]["roc_auc"]:.4f} / exterior{s["outer_eval"]["metrics"]["roc_auc"]:.4f}. Mismo checkpoint, eval, sin aumentos.\n\n'
        text+=f'[Configuración]({relative}/config_resolved.json) · [Métricas]({relative}/summary.json) · [Todas las épocas]({relative}/history.csv) · [Predicciones exteriores]({relative}/predictions_outer.csv).\n\n'
        text+=f'[Seis paneles]({relative}/diagnostico.png) · [Distribución, calibración y pérdidas equivalentes]({relative}/probabilidades_calibracion.png).\n\n'
        text+='### Ambas clases · umbral0,5\n\n'+class_table(s['outer_eval']['metrics']).round(4).to_markdown(index=False)
        text+='\n\n### Ambas clases · umbral interno\n\n'+class_table(s['outer_decision_metrics']).round(4).to_markdown(index=False)
        text+='\n\n### Todas las épocas\n\n'+pd.read_csv(folder/'history.csv').round(4).to_markdown(index=False)
        losses=equivalent_losses({k:pd.read_csv(folder/f'predictions_{k}.csv') for k in ('fit','inner','outer')})
        text+='\n\n### Pérdidas equivalentes del checkpoint elegido\n\n'+losses.round(4).to_markdown(index=False)
        text+='\n\nMisma pérdida por paciente, modoeval y media de probabilidades en las tres particiones. Baseline constante estimada solo en fit. Son mediciones del checkpoint elegido: no reconstruyen curvas por época; no se conservaron todos los checkpoints intermedios. Curvas cercanas sin superar baseline pueden indicar aprendizaje insuficiente. El objetivo es mejora sostenida interna y generalización, no igualdad visual. No usar exterior para cambiar checkpoint ni umbral.'
        text+='\n\nLa pérdida train es por corte y online; la interna por paciente y en eval. La comparación de sobreajuste usa AUCfit/interna del checkpoint elegido, no su resta de pérdidas. Precisión NaN sin positivas = no definida. AP es average precision, no integral trapezoidal.'
        cells.append(nbformat.v4.new_markdown_cell(text))
        cells.append(image_cell([folder/'diagnostico.png',folder/'probabilidades_calibracion.png']))
    # Outputs PNG ya incluidos; finalización ejecuta solo estos lectores de figuras.
    nb=nbformat.v4.new_notebook(cells=cells)
    nb.metadata.kernelspec=dict(display_name='Python3',language='python',name='python3')
    nbformat.write(nb,ROOT/'RESULTADOS.ipynb')
    (ROOT/'README.md').write_text(overview_text+'\n\n'+('\n\n'.join(c.source for c in cells[2:5]) if done else '')+
                                '\n\nResultados individuales bajo <perfil>_foldN/resultados; pesos en checkpoints/C16_cnn_97809. Smoke excluido de tablas científicas.\n')
    embed(nb,cells[0])
    write_json(ROOT/'report_state.json',dict(status='complete' if done else 'running',scientific_runs=len(runs),
        generated_utc=datetime.now(timezone.utc).isoformat(),source_hashes_current=source_hashes()))
    print(status,flush=True)


if __name__=='__main__':
    build()
