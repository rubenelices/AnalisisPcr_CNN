"""Diagnóstico descriptivo de decisión: nunca selecciona ni adopta un umbral."""
from pathlib import Path
from io import BytesIO
import base64
import nbformat
import numpy as np
import pandas as pd
from src.metrics import binary_metrics, validate_binary_inputs

THRESHOLDS=(.25,.30,.325,.35,.40,.425,.45,.50)


def validate_patients(frame):
    if not {'patient_id','target','probability'}.issubset(frame):
        raise ValueError('Faltan predicciones por paciente')
    if frame.patient_id.isna().any() or frame.patient_id.duplicated().any():
        raise ValueError('Una fila por paciente, sin identidades vacías')
    y,p=validate_binary_inputs(frame.target,frame.probability)
    if len(np.unique(y))!=2:raise ValueError('Se necesitan ambas clases')
    return y,p


def threshold_table(frame,thresholds=THRESHOLDS):
    y,p=validate_patients(frame);rows=[]
    for threshold in thresholds:
        m=binary_metrics(y,p,threshold);c=m['confusion_matrix'];npositive=c['tp']+c['fp']
        fnegative=2*c['tn']/(2*c['tn']+c['fp']+c['fn']) if 2*c['tn']+c['fp']+c['fn'] else 0.
        rows.append({'Umbral':threshold,'Positivas predichas':npositive,
            'Precisión pCR':f"{m['precision']:.3f}" if npositive else 'No definida',
            'Sensibilidad':m['sensitivity'],'Especificidad':m['specificity'],
            'F1 macro':(fnegative+m['f1'])/2,'Balanced accuracy':m['balanced_accuracy'],
            'TP':c['tp'],'FN':c['fn'],'FP':c['fp'],'TN':c['tn']})
    return pd.DataFrame(rows).round(3)


def decision_outputs(folder):
    import matplotlib.pyplot as plt
    p=pd.read_csv(Path(folder)/'predictions_patient.csv');y,prob=validate_patients(p)
    npositive=int((prob>=.5).sum());prevalence=float(y.mean())
    mean0=float(prob[y==0].mean());mean1=float(prob[y==1].mean())
    message=(f'**Diagnóstico de la matriz y precisión · sin umbral adoptado.** '
        f'A 0,5 hay {npositive} predicciones positivas de {len(p)} pacientes. '
        f'Probabilidad máxima: {prob.max():.3f}. Media no-pCR: {mean0:.3f}; media pCR: {mean1:.3f}. '
        f'Proporción pCR: {prevalence:.1%}, referencia de precisión sin discriminación. '
        'Cuando no se predicen positivas, la precisión de pCR no está definida; '
        'el resumen histórico la representa como cero por convención. '
        'La siguiente tabla explora umbrales fijos sobre validación ya utilizada para seleccionar '
        'el checkpoint. No es rendimiento independiente ni elige un umbral final; '
        'cambiar umbral no mejora AUC ni AP. Bajar umbral intercambia FN por FP.')
    table=threshold_table(p)
    outputs=[nbformat.v4.new_output('display_data',data={'text/markdown':message,'text/plain':message}),
        nbformat.v4.new_output('display_data',data={'text/html':table.to_html(index=False),
        'text/plain':table.to_string(index=False)})]
    fig,ax=plt.subplots(figsize=(10,4))
    bins=np.linspace(0,1,41)
    for value,label,color in [(0,'no-pCR','#346789'),(1,'pCR','#b8553c')]:
        values=prob[y==value]
        ax.hist(values,bins=bins,weights=np.ones(len(values))/len(values),
            alpha=.45,color=color,label=f'{label} · n={len(values)}')
    ax.axvline(.5,color='#303030',ls='--',label='Umbral histórico 0,5')
    ax.set(xlabel='Probabilidad predicha por paciente',ylabel='Fracción dentro de cada clase',
        title='Distribución de probabilidades: separación entre clases y umbral',xlim=(0,1))
    ax.legend();fig.tight_layout();buffer=BytesIO()
    fig.savefig(buffer,format='png',dpi=110,bbox_inches='tight',facecolor='white');plt.close(fig)
    outputs.append(nbformat.v4.new_output('display_data',data={'image/png':base64.b64encode(buffer.getvalue()).decode(),
        'text/plain':'Distribución de probabilidades de ambas clases y umbral fijo0,5.'}))
    return outputs


def display_decision(folder):
    from IPython.display import display
    for output in decision_outputs(folder):display(output['data'],raw=True)
