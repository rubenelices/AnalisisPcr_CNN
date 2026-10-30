"""Vista C17: pendiente/en curso/completo, sin ejecutar entrenamiento al abrir."""
import base64
import copy
import json
from pathlib import Path
import re

import nbformat
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve,precision_recall_curve
from src.reporting.simple_cnn_results import matrix,class_table

ROOT=Path('experiments/C17_receta_publicada')
TAG='recipe_C17_97809'


def figure(folder,selection,title=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    h=pd.read_csv(folder/'history.csv')
    train=pd.read_csv(folder/'predictions_train_patients_best.csv')
    val=pd.read_csv(folder/'predictions_validation_patients_best.csv')
    fig,axes=plt.subplots(2,3,figsize=(17,10),constrained_layout=True)
    axes[0,0].plot(h.epoch,h.train_eval_loss,label='Train eval sin aumentos')
    axes[0,0].plot(h.epoch,h.validation_loss,label='Validación eval')
    axes[0,0].set(title='BCE normal por corte · mismo modo/unidad',xlabel='Época',ylabel='BCE')
    axes[0,1].plot(h.epoch,h.train_patient_auc,label='Train por paciente')
    axes[0,1].plot(h.epoch,h.patient_roc_auc,label='Validación por paciente')
    axes[0,1].plot(h.epoch,h.mean_epoch_auc,ls=':',label='Media descriptiva épocas val')
    axes[0,1].set(title='ROC-AUC por paciente · sin aumentos',xlabel='Época',ylim=(0,1))
    for ax in axes[0,:2]:
        ax.axvline(selection['epoch'],ls='--',color='#555',label='Checkpoint elegido');ax.legend(fontsize=8);ax.grid(alpha=.2)
    for name,p in [('Train',train),('Validación',val)]:
        fpr,tpr,_=roc_curve(p.target,p.probability)
        axes[0,2].plot(fpr,tpr,label=name)
    axes[0,2].plot([0,1],[0,1],'--',color='#999');axes[0,2].set(title='ROC · checkpoint elegido',xlabel='FPR',ylabel='TPR');axes[0,2].legend()
    precision,recall,_=precision_recall_curve(val.target,val.probability)
    axes[1,0].plot(recall,precision,label='Validación')
    axes[1,0].axhline(val.target.mean(),ls='--',color='#999',label='Prevalencia')
    axes[1,0].set(title='Precisión–recall · checkpoint elegido',xlabel='Sensibilidad',ylabel='Precisión',ylim=(0,1));axes[1,0].legend()
    matrix(axes[1,1],selection['validation_eval']['patient_metrics'],'Validación · umbral fijo 0,5')
    for y,label in [(0,'no pCR'),(1,'pCR')]:
        axes[1,2].hist(val.loc[val.target==y,'probability'],bins=np.linspace(0,1,21),density=True,alpha=.6,label=label)
    axes[1,2].axvline(.5,ls='--',color='#555');axes[1,2].set(title='Probabilidades por clase',xlabel='Media por paciente',ylabel='Densidad');axes[1,2].legend()
    fig.suptitle(title or f'C17 · {folder.parent.name} · fold0 · {len(h)}épocas · best{selection["epoch"]}',fontsize=16)
    for ext in ('png',):
        path=folder/f'curvas_y_diagnostico.{ext}';temp=folder/f'curvas_y_diagnostico.tmp.{ext}'
        fig.savefig(temp,dpi=130);temp.replace(path)
    plt.close(fig)


def embed(nb):
    main=Path('experiments/RESULTADOS.ipynb')
    if not main.exists():return
    base=nbformat.read(main,as_version=4)
    keep=[c for c in base.cells if c.metadata.get('campaign_overview')!=TAG and c.metadata.get('embedded_campaign')!=TAG]
    new=[]
    prefix=ROOT.relative_to(main.parent).as_posix()+'/'
    for i,original in enumerate(nb.cells):
        c=copy.deepcopy(original)
        c.metadata['campaign_overview' if i==0 else 'embedded_campaign']=TAG
        if c.cell_type=='markdown':
            c.source=re.sub(r'\]\((?![A-Za-z]+:|/|#)([^)]+)\)',lambda m:']('+prefix+m.group(1)+')',c.source)
        new.append(c)
    base.cells=[new[0]]+keep+new[1:]
    temp=main.with_name('RESULTADOS.C17.tmp.ipynb');nbformat.write(base,temp);temp.replace(main)


def png_cell(path):
    # Outputs precomputados, sin fingir ejecución. La celda solo vuelve a mostrar PNG.
    source="from pathlib import Path\nfrom IPython.display import Image, display\n_project = next(p for p in (Path.cwd(), *Path.cwd().parents) if (p/'src').is_dir() and (p/'app.py').is_file())\n"
    source+=f'display(Image(filename=str(_project / {path.as_posix()!r})))'
    return nbformat.v4.new_code_cell(source,metadata=dict(jupyter=dict(source_hidden=True)),outputs=[
        nbformat.v4.new_output('display_data',data={'image/png':base64.b64encode(path.read_bytes()).decode(),
                                                'text/plain':path.name})])


def build():
    ROOT.mkdir(parents=True,exist_ok=True)
    rows=[];details=[]
    for loss in ('normal','ponderada'):
        folder=ROOT/'resultados'/loss/'fold_0_seed_42'
        h=pd.read_csv(folder/'history.csv') if (folder/'history.csv').exists() else None
        selected=json.loads((folder/'selection.json').read_text()) if (folder/'selection.json').exists() else None
        done=json.loads((folder/'summary.json').read_text()) if (folder/'summary.json').exists() else None
        state='Completo' if done else 'En curso/parcial' if h is not None else 'Pendiente · se ejecuta a mano'
        row=dict(Pérdida=loss,Estado=state,Épocas=0 if h is None else len(h))
        if selected is not None:
            m=selected['validation_eval']['patient_metrics'];tr=selected['train_eval']['patient_metrics']
            row.update(AUCvalMediaÉpocas=float(h.patient_roc_auc.mean()),AUCvalMáxima=m['roc_auc'],
                AUCFitElegido=tr['roc_auc'],ÉpocaElegida=selected['epoch'],AP=m['pr_auc'],
                Precisión=m['precision'] if m['precision_defined'] else None,Sensibilidad=m['sensitivity'],
                F1macro=m['f1_macro'],BalancedAccuracy=m['balanced_accuracy'],Brier=m['brier_score'])
        rows.append(row)
        text=f'## BCE {loss} · {state}\n\n'
        if selected is None:
            details.append(nbformat.v4.new_markdown_cell(text+'Sin resultados científicos todavía.'))
            continue
        figure(folder,selected)
        from src.evaluation.generalization_panels import diagnostics
        diagnostic=diagnostics(folder,selected,f'C17 · BCE {loss} · fold0')
        rel=folder.relative_to(ROOT).as_posix()
        text+=f'[{len(h)} épocas completas]({rel}/history.csv) · [Checkpoint elegido]({rel}/selection.json) · [Configuración]({rel}/config_resolved.json).\n\n'
        text+='Pérdida train-eval y val-eval: BCE **normal por corte**, ambas sin aumentos ni dropout. Objetivo ponderado y pérdida online se conservan aparte en el historial. ROC y matriz son por paciente, media de probabilidades, umbral fijo0,5.\n\n'
        text+=class_table(selected['validation_eval']['patient_metrics']).round(4).to_markdown(index=False)
        metric=selected['validation_eval']['patient_metrics'];counts=metric['confusion_matrix']
        n0=counts['tn']+counts['fp'];n1=counts['fn']+counts['tp']
        text+=f'\n\nBaseline siempre no pCR: accuracy={n0/(n0+n1):.4f}, F1macro={n0/(2*n0+n1):.4f}, balancedaccuracy=0,5, sensibilidad=0; precisión pCR no definida. AUC de puntuación constante=0,5.'
        text+='\n\nPrecisión vacía/NaN: no definida al no predecir positivos.\n\n### Todas las épocas\n\n'+h.round(4).to_markdown(index=False)
        text+='\n\nPanel adicional: BCE/AUC por paciente en eval para vigilar sobreajuste; pérdida online y BCE por corte se muestran aparte. Test reservado.\n'
        details.extend([nbformat.v4.new_markdown_cell(text),png_cell(folder/'curvas_y_diagnostico.png'),png_cell(diagnostic)])
    table=pd.DataFrame(rows)
    table.to_csv(ROOT/'resumen.csv',index=False)
    intro='''# C17 · Receta publicada · fold 0

CNN propia97.809 parámetros, fases originales, flips H/V0,5, Adam LR0,001 fijo,
WD0,0001, batch16, seed42, máximo40épocas y paciencia8 por AUCval real.
Dos pérdidas separadas, normal y ponderada N0/N1 de cortes train. Sin clínica.
878pacientes train y219validación; esta validación elige checkpoint y NO es
evaluación exterior independiente. Test cerrado. No es una media de cincofolds.

**Media entre épocas y mejor época se muestran separadas**: las épocas comparten
pacientes y no son ensayos independientes. La media acumulada no elige pesos.
Cambian varias decisiones frente C16: paquete de reproducción, no ablación.
Esta es la receta por defecto del commit revisado; no verifica el resultado
ni los argumentos reales usados por el compañero. Runtime adaptado al Mac local.

[Instrucciones](README.md) · [Lanzar y ver gráficas](EJECUTAR.ipynb) · [Configuración](config.json).
Abrir este notebook solo lee resultados; no inicia entrenamientos.

'''+table.round(4).to_markdown(index=False)+'\n\nUso educativo y de investigación, sin validez clínica.'
    nb=nbformat.v4.new_notebook(cells=[nbformat.v4.new_markdown_cell(intro)]+details)
    nb.metadata.kernelspec=dict(display_name='Python3',language='python',name='python3')
    temp=ROOT/'RESULTADOS.tmp.ipynb';nbformat.write(nb,temp);temp.replace(ROOT/'RESULTADOS.ipynb')
    embed(nb)
    return ROOT/'RESULTADOS.ipynb'


if __name__=='__main__':build()
