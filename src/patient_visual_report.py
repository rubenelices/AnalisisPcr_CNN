"""Figuras separadas a partir del best ya evaluado, sin entrenamiento ni test.

Explorar umbrales en desarrollo NO valida su rendimiento fuera de ese fold.
No modifica probabilidades, pesos, predicciones originales ni umbral del modelo.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, precision_recall_curve
from sklearn.calibration import calibration_curve
from src.metrics import binary_metrics


def validate_patients(frame):
    required={'patient_id','split','fold','target','probability'}
    if not required.issubset(frame):
        raise ValueError('Faltan columnas de predicciones por paciente')
    if frame.empty or frame.patient_id.isna().any() or frame.patient_id.duplicated().any():
        raise ValueError('Se exige una fila por paciente')
    if set(frame.split)!={'validation'} or frame.fold.nunique()!=1 or not frame.fold.isin(range(5)).all():
        raise ValueError('Solo fold interno de validación; test cerrado')
    if set(frame.target)!={0,1} or not np.isfinite(frame.probability).all() or not frame.probability.between(0,1).all():
        raise ValueError('Etiquetas o probabilidades inválidas')


def threshold_table(frame):
    validate_patients(frame)
    rows=[]
    for threshold in np.round(np.arange(.05,1.,.05),2):
        m=binary_metrics(frame.target,frame.probability,float(threshold))
        rows.append({k:v for k,v in m.items() if k!='confusion_matrix'}|m['confusion_matrix'])
    return pd.DataFrame(rows)


def save_patient_visual_report(source, destination=None):
    source=Path(source)
    destination=Path(destination) if destination else source
    for folder in (source,destination):
        resolved=folder.resolve()
        root=Path.cwd().resolve()
        relative=resolved.relative_to(root)
        if not (relative.is_relative_to('reports') or
                (relative.is_relative_to('experimentos_datos') and 'resultados' in relative.parts)):
            raise ValueError('Informes solo en reports o experimentos_datos/.../resultados')
    original=source/'predictions_patient.csv'
    frame=pd.read_csv(original)
    summary=json.loads((source/'summary.json').read_text())
    if summary.get('test_evaluated') is not False:
        raise ValueError('El resumen no acredita test cerrado')
    if not {'split','fold'}.issubset(frame):
        # E56/E57 exportaban sin contexto del fold. Recuperarlo SOLO contra el
        # índice público y configuración completos; no deducirlo por nombre.
        spec=json.loads((source/'config_resolved.json').read_text())
        if (spec.get('schema')!='patient_features_v1' or
                summary.get('model_class')!='PatientFeatureNet_v1' or
                summary.get('scientific_result') is not True or
                summary.get('debug_patient_subset') is not False):
            raise ValueError('CSV antiguo sin contexto de validación verificable')
        data=spec['experiment']['data']; fold=data['validation_fold']
        index=pd.read_csv(Path(data['root'])/'metadata/samples.csv',
                          usecols=['patient_id','split','fold','pCR'])
        index=index[(index.split=='train') & (index.fold==fold)].copy()
        if (index.groupby('patient_id').pCR.nunique()!=1).any():
            raise ValueError('Objetivo inconsistente en metadatos')
        index=index.drop_duplicates('patient_id').set_index('patient_id')
        if set(frame.patient_id)!=set(index.index):
            raise ValueError('CSV antiguo no contiene exactamente validación completa')
        if not np.array_equal(frame.target.to_numpy(),frame.patient_id.map(index.pCR).to_numpy()):
            raise ValueError('Etiquetas antiguas no coinciden con índice de validación')
        frame=frame.assign(split='validation',fold=fold)
    validate_patients(frame)
    history=pd.read_csv(source/'history.csv')
    epoch=summary['best_epoch']
    if epoch not in history.epoch.tolist():
        raise ValueError('Best no está en historial')
    metrics=binary_metrics(frame.target,frame.probability,.5)
    stored=summary['validation_metrics']['patient']
    if metrics['n']!=stored['n'] or metrics['confusion_matrix']!=stored['confusion_matrix']:
        raise ValueError('Recuento/matriz del best no coinciden')
    for key in ('roc_auc','pr_auc','brier_score','sensitivity','specificity','precision','f1'):
        if not np.isclose(metrics[key],stored[key],rtol=0,atol=1e-9):
            raise ValueError('Predicciones no coinciden con métricas del best')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    destination.mkdir(parents=True,exist_ok=True)
    def save(name,plot):
        fig,ax=plt.subplots(figsize=(6.5,5.2))
        plot(ax)
        ax.grid(alpha=.15)
        fig.suptitle(f'{summary["experiment_id"]} · validación por paciente · best {epoch}',fontsize=10)
        fig.text(.5,.01,'Desarrollo exploratorio · no test · uso educativo, sin validez clínica',ha='center',fontsize=8)
        fig.tight_layout(rect=(0,.04,1,.96))
        fig.savefig(destination/name,dpi=160,bbox_inches='tight',facecolor='white'); plt.close(fig)
    y,p=frame.target.to_numpy(),frame.probability.to_numpy()
    fpr,tpr,_=roc_curve(y,p); precision,recall,_=precision_recall_curve(y,p)
    def roc(ax):
        ax.plot(fpr,tpr,color='#126e82',label=f'AUC = {metrics["roc_auc"]:.4f}')
        ax.plot([0,1],[0,1],'--',color='gray'); ax.legend()
        ax.set(xlabel='1 − especificidad',ylabel='Sensibilidad',title='Curva ROC',xlim=(0,1),ylim=(0,1))
    def pr(ax):
        ax.plot(recall,precision,color='#a44a3f',label=f'AP = {metrics["pr_auc"]:.4f}')
        ax.axhline(y.mean(),ls='--',color='gray',label=f'Prevalencia = {y.mean():.3f}'); ax.legend()
        ax.set(xlabel='Recall / sensibilidad',ylabel='Precision (valor predictivo positivo)',title='Precision–recall',xlim=(0,1),ylim=(0,1.03))
    def confusion(ax):
        c=metrics['confusion_matrix']; matrix=np.array([[c['tn'],c['fp']],[c['fn'],c['tp']]])
        ax.imshow(matrix,cmap='Blues')
        for i in range(2):
            for j in range(2): ax.text(j,i,str(matrix[i,j]),ha='center',va='center',fontsize=20)
        ax.set_xticks([0,1],labels=['Predice no pCR','Predice pCR']); ax.set_yticks([0,1],labels=['Real no pCR','Real pCR'])
        ax.set(title='Matriz de confusión · umbral fijo 0,5',xlabel='Predicción',ylabel='Realidad')
        ax.grid(False)
    def calibration(ax):
        observed,predicted=calibration_curve(y,p,n_bins=10,strategy='quantile')
        ax.plot(predicted,observed,'o-',label=f'Brier = {metrics["brier_score"]:.4f}')
        ax.plot([0,1],[0,1],'--',color='gray'); ax.legend()
        ax.set(xlabel='Probabilidad media predicha',ylabel='Fracción pCR observada',title='Calibración',xlim=(0,1),ylim=(0,1))
    def evolution(ax,key,title):
        ax.plot(history.epoch,history[key],marker='.',color='#126e82')
        ax.axvline(epoch,ls='--',color='#a44a3f',label='Checkpoint conservado'); ax.legend()
        ax.set(xlabel='Época',ylabel=title,title=f'{title} a lo largo del entrenamiento')
    for name,plot in [('roc_paciente.png',roc),('precision_recall_paciente.png',pr),
                      ('matriz_confusion_umbral_050.png',confusion),('calibracion_paciente.png',calibration),
                      ('auc_por_epoca.png',lambda ax:evolution(ax,'patient_roc_auc','ROC-AUC por paciente')),
                      ('pr_auc_por_epoca.png',lambda ax:evolution(ax,'patient_pr_auc','AP por paciente'))]:
        save(name,plot)
    threshold_table(frame).to_csv(destination/'umbrales_exploratorios.csv',index=False)
    output=dict(experiment_id=summary['experiment_id'],best_epoch=epoch,patients=len(frame),
        threshold=.5,metrics=metrics,test_evaluated=False,threshold_selected=False,
        source_directory=str(source),source_prediction_sha256=hashlib.sha256(original.read_bytes()).hexdigest(),
        source_summary_sha256=hashlib.sha256((source/'summary.json').read_bytes()).hexdigest(),
        scientific_result=summary.get('scientific_result',True),
        warning='Umbrales solo exploratorios del fold usado para seleccionar best; no rendimiento independiente.')
    (destination/'metricas_visuales.json').write_text(json.dumps(output,indent=2,ensure_ascii=False))
    rows='\n'.join(f'| {key} | {metrics[key]:.4f} |' for key in
        ['roc_auc','pr_auc','sensitivity','specificity','precision','f1','balanced_accuracy','brier_score'])
    (destination/'LEEME_RESULTADOS.md').write_text(f'''# {summary['experiment_id']} · best época {epoch}

Métricas del mejor checkpoint, no de la última época. {len(frame)} pacientes.
Umbral de referencia fijo 0,5; no se ha elegido ni aplicado otro umbral.

| Métrica | Valor |
|---|---:|
{rows}

![ROC](roc_paciente.png)
![Precision recall](precision_recall_paciente.png)
![Matriz](matriz_confusion_umbral_050.png)
![Calibración](calibracion_paciente.png)
![AUC por época](auc_por_epoca.png)
![AP por época](pr_auc_por_epoca.png)

`umbrales_exploratorios.csv` muestra compromisos entre sensibilidad y especificidad.
No convierte un umbral elegido aquí en una evaluación independiente. Ajustar umbral
no cambia ROC-AUC ni AP. Validar/calibrar con predicciones fuera de fold y luego
congelar protocolo antes de test. No modificar el checkpoint para mejorar una cifra.

Origen: `{source}`. Predicciones originales intactas. Uso educativo, no clínico.
''',encoding='utf-8')
    return output


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',required=True)
    parser.add_argument('--destination')
    args=parser.parse_args()
    result=save_patient_visual_report(args.source,args.destination)
    print(json.dumps(result,indent=2,ensure_ascii=False))
