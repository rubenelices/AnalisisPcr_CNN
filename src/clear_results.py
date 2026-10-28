"""Vista esencial de resultados: épocas y seis gráficas, sin entrenar ni mover fuentes."""
import argparse
import base64
import json
import re
from io import BytesIO
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
SEARCH_ROOTS=('reports/experiments','reports/multimodal','reports/patient_features',
              'reports/confirmations','reports/learning_curves/runs','experimentos_datos','experimentos_equilibrados')


def completed_runs(root=ROOT):
    runs=[]
    for relative in SEARCH_ROOTS:
        for path in sorted((root/relative).rglob('summary.json')):
            if path.is_relative_to(root/'experimentos_equilibrados/campana_20261015'):continue
            if path.is_relative_to(root/'experimentos_equilibrados/C16_cnn_97809'):continue
            if path.is_relative_to(root/'experimentos_equilibrados/C17_receta_publicada'):continue
            if path.is_relative_to(root/'experimentos_equilibrados/campana_20261016'):continue
            if relative in {'experimentos_datos','experimentos_equilibrados'} and 'resultados' not in path.parts: continue
            s=json.loads(path.read_text())
            if (s.get('status')!='complete' or s.get('scientific_result') is not True
                    or s.get('test_evaluated') is not False or s.get('debug_patient_subset') is not False): continue
            # D00 es una copia de presentación de E58, no otra ejecución.
            if 'D00_E58_referencia' in path.parts: continue
            if not (path.parent/'history.csv').exists(): continue
            if not re.fullmatch(r'[A-Za-z0-9_-]+',s['experiment_id']): raise ValueError('Identidad inválida')
            runs.append((path.parent,s))
    return sorted(runs,key=lambda item:(0 if item[0].is_relative_to(root/'experimentos_datos') or
                    item[0].is_relative_to(root/'reports/multimodal') else 1,item[1]['experiment_id'],str(item[0])))


def show_run(folder,summary,as_outputs=False):
    """Mostrar todas las épocas sin IDs y curvas originales del BEST."""
    import matplotlib.pyplot as plt
    from IPython.display import display, Markdown
    from sklearn.metrics import roc_curve,precision_recall_curve
    from sklearn.calibration import calibration_curve
    folder=Path(folder); h=pd.read_csv(folder/'history.csv').rename(columns={'validation_loss':'val_loss','elapsed_seconds':'seconds'})
    m=summary['validation_metrics']['patient'];outputs=[]
    best=summary['best_epoch']
    title=(f"**{summary['experiment_id']}** · mejor época {best}/{len(h)} · "
        f"{m['n']} pacientes · ROC-AUC {m['roc_auc']:.4f} · AP {m['pr_auc']:.4f} · "
        f"precisión {m['precision']:.1%} · sensibilidad {m['sensitivity']:.1%} · F1 {m['f1']:.3f}")
    if as_outputs: outputs.append(nbformat.v4.new_output('display_data',data={'text/markdown':title,'text/plain':title}))
    else: display(Markdown(title))
    keys=['epoch','train_loss','val_loss','patient_roc_auc','patient_pr_auc','learning_rate','seconds']
    columns=[k for k in keys if k in h]
    table=h[columns].round(5).rename(columns={'epoch':'Época','train_loss':'Loss train',
        'val_loss':'Loss validación','patient_roc_auc':'ROC-AUC paciente','patient_pr_auc':'AP paciente',
        'learning_rate':'LR','seconds':'Segundos'})
    if as_outputs: outputs.append(nbformat.v4.new_output('display_data',data={
        'text/html':table.to_html(index=False),'text/plain':table.to_string(index=False)}))
    else:
        with pd.option_context('display.max_rows',None,'display.max_columns',None): display(table)
    fig,axes=plt.subplots(2,3,figsize=(15,8)); ax=axes.ravel()
    for col,label in [('train_loss','Train'),('val_loss','Validación')]: ax[0].plot(h.epoch,h[col],label=label)
    ax[0].set(title='Pérdida por época',xlabel='Época');ax[0].legend()
    for col,label in [('patient_roc_auc','ROC-AUC'),('patient_pr_auc','AP')]:
        if col in h: ax[1].plot(h.epoch,h[col],label=label)
    ax[1].axvline(best,ls='--',color='gray',label='Best');ax[1].legend();ax[1].set(title='AUC / AP por época',xlabel='Época')
    c=m['confusion_matrix']; matrix=np.array([[c['tn'],c['fp']],[c['fn'],c['tp']]])
    ax[4].imshow(matrix,cmap='Blues')
    for i in range(2):
        for j in range(2):ax[4].text(j,i,str(matrix[i,j]),ha='center',va='center',fontsize=18)
    ax[4].set(title='Matriz · umbral del run '+str(m['threshold']),xlabel='Predicho',ylabel='Real',xticks=[0,1],yticks=[0,1])
    csv=folder/'predictions_patient.csv'
    if csv.exists():
        p=pd.read_csv(csv)
        if p.patient_id.duplicated().any() or len(p)!=m['n']: raise ValueError('Pacientes inválidas')
        from src.metrics import binary_metrics
        recalculated=binary_metrics(p.target,p.probability,m['threshold'])
        for key in ('roc_auc','pr_auc','brier_score'):
            if not np.isclose(recalculated[key],m[key],atol=1e-9,rtol=0): raise ValueError('Métricas no coinciden')
        y,prob=p.target.to_numpy(),p.probability.to_numpy()
        fpr,tpr,_=roc_curve(y,prob);precision,recall,_=precision_recall_curve(y,prob)
        ax[2].plot(fpr,tpr);ax[2].plot([0,1],[0,1],'--',color='gray')
        ax[2].set(title=f"ROC del best · {m['roc_auc']:.4f}",xlabel='1 − especificidad',ylabel='Sensibilidad')
        ax[3].plot(recall,precision);ax[3].axhline(y.mean(),ls='--',color='gray')
        ax[3].set(title=f"Precision–recall · AP {m['pr_auc']:.4f}",xlabel='Sensibilidad',ylabel='Precisión')
        observed,predicted=calibration_curve(y,prob,n_bins=8,strategy='quantile')
        ax[5].plot(predicted,observed,'o-');ax[5].plot([0,1],[0,1],'--',color='gray')
        ax[5].set(title=f"Calibración · Brier {m['brier_score']:.4f}",xlabel='Probabilidad',ylabel='Fracción pCR')
    else:
        for i in (2,3,5):ax[i].text(.5,.5,'Predicciones originales no disponibles\nNo reconstruir de logs',ha='center');ax[i].axis('off')
    fig.suptitle(summary['experiment_id']+' · desarrollo por paciente · curvas del best',fontsize=13)
    fig.tight_layout()
    if as_outputs:
        buffer=BytesIO();fig.savefig(buffer,format='png',dpi=100,bbox_inches='tight',facecolor='white')
        outputs.append(nbformat.v4.new_output('display_data',data={'image/png':base64.b64encode(buffer.getvalue()).decode(),
                         'text/plain':'Pérdida, AUC/AP, ROC, PR, matriz y calibración del mejor checkpoint.'}))
    else: plt.show()
    plt.close(fig)
    return outputs if as_outputs else None


def build(destination=None):
    target=Path(destination) if destination else ROOT/'notebooks/07_RESULTADOS_CLAROS.ipynb'
    runs=completed_runs()
    nb=nbformat.v4.new_notebook()
    nb.metadata.kernelspec={'display_name':'Python 3','language':'python','name':'python3'}
    nb.cells=[nbformat.v4.new_markdown_cell('''# Resultados del proyecto · vista sencilla

Un apartado por ejecución: **todas las épocas y seis gráficas importantes**.
Primero clínica + imágenes y controles; después historial solo imagen.
No necesitas abrir CSV/JSON: son respaldos técnicos que se conservan intactos.
Curvas y matriz del mejor checkpoint, no de la última época; tablas completas.
Fold/semilla se mantienen separados. No comparar folds como si fueran las mismas pacientes.
Las curvas LC usan fracciones de pacientes; el tamaño train se indica en la tabla.
Test cerrado. Umbral fijo del run, no ajustado aquí. Uso educativo, sin validez clínica.
'''),nbformat.v4.new_code_cell('''from pathlib import Path
import sys, json
root=Path.cwd()
if not (root/'src').is_dir(): root=root.parent
if not (root/'src').is_dir(): raise RuntimeError('Ejecutar desde proyecto o notebooks')
if str(root) not in sys.path: sys.path.insert(0,str(root))
from src.clear_results import completed_runs, show_run
import pandas as pd
from IPython.display import display
runs=completed_runs(root)
rows=[]
for folder,s in runs:
    m=s['validation_metrics']['patient']
    config=json.loads((folder/'config_resolved.json').read_text()) if (folder/'config_resolved.json').exists() else {}
    config=config.get('experiment',config)
    rows.append({'Experimento':s['experiment_id'],'Fold':config.get('data',{}).get('validation_fold'),
                 'Semilla':config.get('training',{}).get('seed'),'Best':s['best_epoch'],
                 'Épocas':s['epochs_completed'],'Pacientes train':s['patients']['train'],
                 'ROC-AUC':m['roc_auc'],'AP':m['pr_auc'],
                 'Precisión':m['precision'],'Sensibilidad':m['sensitivity'],'F1':m['f1']})
with pd.option_context('display.max_rows',None): display(pd.DataFrame(rows).round(4))
''')]
    reference=('E58_clinical_fusion_mac','D07_BCE_ponderada','D11_fusion_no_lineal')
    focus=(*reference,'D13_occlusion32','D14_pooling_realce','D12_wd_imagen001','D15_BCE_moderada',
           'D16_afin_suave','D17_ganancia_compartida','D18_espejo_horizontal','D19_norma_features','D20_campo_intensidad',
           'D21_lotes_paciente_control','D22_perdida_paciente','D23_cnn_compacta')
    current={s['experiment_id']:s for _,s in runs if s['experiment_id'] in focus}
    if all(name in current for name in reference):
        quick=['\n## Lectura rápida · comparación de desarrollo fold0 / seed42',
               '\nE58 es la referencia; D07 cambia la pérdida, D11 la cabeza, '
               'D13 la oclusión en train, D14 el pooling, D12 la regularización de imagen, D15 la ponderación moderada '
               'D16/D17/D18 aumentos independientes, D19 norma de features y D20 campo de intensidad. '
               'D21/D22 son control/objetivo por paciente con los mismos lotes completos; '
               'frente E58 también cambia la agrupación, no solo la pérdida. '
               'D23 reduce canales y mantiene la resolución. '
               'No se elige un ganador solo por una columna. Todos al umbral0,5.',
               '\n| Modelo | ROC-AUC | AP | Brier ↓ | Sensibilidad | Precisión | F1 | Falsos positivos |',
               '|---|---:|---:|---:|---:|---:|---:|---:|']
        for name in focus:
            if name not in current: continue
            m=current[name]['validation_metrics']['patient']
            quick.append(f"| {name} | {m['roc_auc']:.4f} | {m['pr_auc']:.4f} | {m['brier_score']:.4f} | "
                         f"{m['sensitivity']:.1%} | {m['precision']:.1%} | {m['f1']:.4f} | {m['confusion_matrix']['fp']} |")
        quick.append('\nD07 detecta más positivas pero genera más falsas alarmas; '
                     'Las variantes nuevas deben confirmar cualquier mejora fuera del fold0. '
                     'Cambiar el umbral no cambia ROC-AUC/AP. '
                     'Comparación exploratoria: confirmar fuera del fold de selección. '
                     'La tabla completa de abajo enlaza a todas las épocas y gráficas.')
        nb.cells[0].source+='\n'.join(quick)+'\n'
    for folder,s in runs:
        relative=folder.relative_to(ROOT).as_posix()
        nb.cells.append(nbformat.v4.new_markdown_cell(f"<a id='run_{s['experiment_id']}'></a>\n\n## {s['experiment_id']} · {folder.name}\n\n[Respaldo técnico](../{relative}/) · sin mover ni borrar originales."))
        nb.cells.append(nbformat.v4.new_code_cell(f"folder=root/{relative!r}\nsummary=json.loads((folder/'summary.json').read_text())\nshow_run(folder,summary)"))
    target.parent.mkdir(parents=True,exist_ok=True)
    # Actualizar catálogo sin borrar resultados ya mostrados en el notebook.
    if target.exists():
        old=nbformat.read(target,as_version=4)
        outputs={c.source:(c.get('outputs',[]),c.get('execution_count')) for c in old.cells if c.cell_type=='code'}
        for cell in nb.cells:
            if cell.cell_type=='code' and cell.source in outputs:
                cell.outputs,cell.execution_count=outputs[cell.source]
    # Cada entrenamiento nuevo queda visible SIN ejecutar el notebook a mano.
    # Render solo de celdas nuevas; fuentes/predicciones originales son inmutables.
    for cell in nb.cells:
        if cell.cell_type=='code':
            cell.metadata['jupyter']={'source_hidden':True}
            cell.metadata['tags']=['hide-input']
    for index,(folder,s) in enumerate(runs):
        cell=nb.cells[3+index*2]
        if not cell.outputs:
            cell.outputs=show_run(folder,s,as_outputs=True)
            cell.execution_count=None
            cell.metadata['rendered_from_original_artifacts']=True
    overview=[]
    for folder,s in runs:
        config=json.loads((folder/'config_resolved.json').read_text())
        config=config.get('experiment',config);m=s['validation_metrics']['patient']
        overview.append({'Experimento':s['experiment_id'],'Fold':config['data']['validation_fold'],
            'Semilla':config['training']['seed'],'Best':s['best_epoch'],'Épocas':s['epochs_completed'],
            'Pacientes train':s['patients']['train'],'ROC-AUC':m['roc_auc'],'AP':m['pr_auc'],
            'Precisión':m['precision'],'Sensibilidad':m['sensitivity'],'F1':m['f1']})
    overview=pd.DataFrame(overview).round(4)
    linked=overview.copy()
    if not linked.empty: linked['Experimento']=linked['Experimento'].map(lambda n:f"<a href='#run_{n}'>{n}</a>")
    nb.cells[1].outputs=[nbformat.v4.new_output('display_data',data={
        'text/html':linked.to_html(index=False,escape=False),'text/plain':overview.to_string(index=False)})]
    nb.cells[1].metadata['rendered_from_original_artifacts']=True
    nbformat.write(nb,target)
    print(f'Vista sencilla: {len(runs)} ejecuciones, {target}')
    return target


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output');args=p.parse_args();build(args.output)
