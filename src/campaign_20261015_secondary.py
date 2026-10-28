"""Análisis secundarios prefijados: parámetros ajustados solo en selección interna.

No entrena CNN, no selecciona hiperparámetros y no carga imágenes de test.
"""
import hashlib
import json

import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.linear_model import LogisticRegression

from src.campaign_20261015 import ROOT, select_threshold, enrich, write_json
from src.clinical_features import load_patient_clinical, ClinicalPreprocessor
from src.metrics import binary_metrics

FEATURES=('cnn_patient_logit','age_standardized','menopause','age_missing','menopause_missing')


def probability_logit(values):
    p=np.asarray(values,dtype=np.float64)
    if not np.isfinite(p).all() or np.any((p<0)|(p>1)):
        raise ValueError('Probabilidades inválidas')
    p=np.clip(p,1e-6,1-1e-6)
    return np.log(p)-np.log1p(-p)


class FrozenPatientFusion(nn.Module):
    """Media de probabilidades CNN ya calculada + cuatro campos clínicos.

    Capa lineal real5→1:6 parámetros. No sustituye el extractor CNN propio.
    """
    def __init__(self,weight,bias):
        super().__init__()
        self.linear=nn.Linear(5,1,dtype=torch.float64)
        with torch.no_grad():
            self.linear.weight.copy_(torch.as_tensor(weight,dtype=torch.float64).reshape(1,5))
            self.linear.bias.copy_(torch.as_tensor(bias,dtype=torch.float64).reshape(1))

    def forward(self,patient_probability,clinical_four):
        p=patient_probability.to(torch.float64).clamp(1e-6,1-1e-6)
        x=torch.cat((torch.logit(p).reshape(-1,1),clinical_four.to(torch.float64)),dim=1)
        return self.linear(x).squeeze(1)


def validate_parts(parts,manifest):
    sets=[]
    for name in ('fit','inner','outer'):
        frame=parts[name]
        if frame.patient_id.duplicated().any() or set(frame.patient_id)!=set(manifest[name]):
            raise ValueError('Partición de pacientes incoherente')
        if set(frame.target)!={0,1}:
            raise ValueError('Se requieren ambas clases')
        probability_logit(frame.probability)
        sets.append(set(frame.patient_id))
    if any(sets[a]&sets[b] for a,b in ((0,1),(0,2),(1,2))):
        raise ValueError('Solape de pacientes')


def fit_internal(x_inner,y_inner,c):
    # No se recibe ni se consulta el exterior para ajustar coeficientes.
    return LogisticRegression(C=c,solver='lbfgs',max_iter=2000,random_state=42).fit(x_inner,y_inner)


def document():
    from pathlib import Path
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    module=FrozenPatientFusion(np.zeros(5),[0.]).eval()
    with torch.no_grad():assert module(torch.full((2,),.3),torch.zeros(2,4)).shape==(2,)
    count=sum(p.numel() for p in module.parameters());assert count==6
    folder=Path('experimentos/C15_fusion_tardia');folder.mkdir(exist_ok=True)
    fig,ax=plt.subplots(figsize=(14,5));ax.set(xlim=(0,14),ylim=(0,5));ax.axis('off')
    ax.text(.2,4.6,'Fusión tardía fija · CNN congelada + seis parámetros',fontsize=20,color='#213344')
    nodes=[(.3,2.4,2.5,1.,'CNN propia por corte\n298.161 o75.961 params'),(3.3,2.4,2.5,1.,'Sigmoide → media\npor paciente → logit'),
        (3.3,.6,2.5,1.,'Edad/menopausia\n+ dos ausencias'),(6.7,1.5,2.1,1.,'Concatenar [B,5]\nsin parámetros'),
        (9.4,1.5,1.8,1.,'Linear5→1\n6 parámetros'),(12,1.5,1.5,1.,'Sigmoide\n[B]')]
    for x,y,w,h,label in nodes:
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.05',facecolor='#e1eee9',edgecolor='#568c7b'))
        ax.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=10)
    for a,b in (((2.85,2.9),(3.2,2.9)),((5.85,2.9),(6.6,2.1)),((5.85,1.1),(6.6,1.9)),((8.85,2),(9.3,2)),((11.25,2),(11.9,2))):
        ax.annotate('',xy=b,xytext=a,arrowprops=dict(arrowstyle='->',color='#213344'))
    ax.text(.3,.13,'Clínica: estadísticas solo fit · coeficientes y umbral solo interna · C=0,1 fijo · exterior solo evaluación',fontsize=11)
    for ext in ('png','svg'):fig.savefig(folder/f'arquitectura.{ext}',dpi=150,bbox_inches='tight',facecolor='white')
    plt.close(fig)
    (folder/'README.md').write_text('''# C15 · Fusión tardía regularizada

![Composición](arquitectura.png)

CNN propia desde cero, congelada después de seleccionar el checkpoint interno.
No se reentrena ni se importan arquitecturas o pesos preentrenados. Cada corte
produce una probabilidad; su media por paciente se recorta a[1e-6,1−1e-6] y pasa
a logit. Se concatena con edad estandarizada, menopausia y dos indicadores de
ausencia. HR/HER2 nunca se leen. Estadísticas e imputación solo pacientes fit;
regresión logísticaC0,1 y umbralF1macro solo pacientes internos. Estos mismos
internos también seleccionaron el checkpoint: métricas internas optimistas.

| Operación | Entrada | Salida | Parámetros nuevos | Geometría |
|---|---|---|---:|---|
| CNN propia congelada | `[cortes,3,256,256]` | `[cortes]` logits | 0 | Ver ficha CNN |
| Sigmoide y media por paciente | `[cortes]` | `[B,1]` | 0 | Sin kernel/stride/padding |
| Clip y logit | `[B,1]` | `[B,1]` | 0 | Sin geometría espacial |
| Clínica transformada | edad/menopausia | `[B,4]` | 0 | Estadísticas solo fit |
| Concatenación | `[B,1]`+`[B,4]` | `[B,5]` | 0 | Sin geometría espacial |
| Linear y sigmoide | `[B,5]` | `[B]` | 6 | Sin kernel/stride/padding |

Formas y seis parámetros comprobados contra FrozenPatientFusion real; prueba
de reconstrucción reproduce scikit-learn a1e-12. Total composición:298.167 con
CNNancha o75.967 con compacta. Convoluciones/RF/resolución se heredan íntegros
de [E57](../E57_patient_feature_mean_mac/README.md) o
[compacta](../C15_E57_compacta_cortes/README.md); esta capa no altera los mapas.

Motivo: comprobar si dos variables basales aportan información complementaria
sin aumentar capacidad de la CNN. No arregla su sobreajuste ni demuestra que
las imágenes aporten valor por sí solas. Se evalúa como análisis secundario
prefijado en los cinco exteriores, separado del régimen solo imágenes.
Disponibilidad de clínica en demo no confirmada. Coeficientes, preprocesador y
checkpoint fuente se guardan por fold para reconstrucción. Sin uso de test.

La calibración de un logit es un análisis distintoC1, dos parámetros; al ser
monótona creciente conserva ROC/AP y decisiones al transformar el umbral.
Su finalidad es medir Brier/calibración, no mejorar discriminación.

Regenerar: `python -m src.document_architectures --only C15_fusion_tardia`.
Uso educativo y de investigación, sin validez clínica.
''')
    return count


def evaluate_saved_fold(folder,output):
    summary=json.loads((folder/'summary.json').read_text())
    manifest=json.loads((folder/'split_manifest.json').read_text())
    if summary.get('phase')!='cv' or summary.get('test_evaluated') is not False:
        raise ValueError('Solo confirmación exterior local sin test')
    parts={name:pd.read_csv(folder/f'predictions_{name}.csv',dtype={'patient_id':str}) for name in ('fit','inner','outer')}
    validate_parts(parts,manifest)
    clinical={name:load_patient_clinical('breastdcedl',frame.patient_id.tolist(),columns=('age','menopause')) for name,frame in parts.items()}
    prep=ClinicalPreprocessor.fit(clinical['fit'],active_columns=('age','menopause'))
    xlogit={name:probability_logit(frame.probability).reshape(-1,1) for name,frame in parts.items()}
    xfusion={name:np.column_stack((xlogit[name],prep.transform(clinical[name]).numpy()[:,[0,3,4,7]])) for name in parts}
    reports=[]
    for regime,features,c in (('calibrated_images',xlogit,1.),('images_age_menopause',xfusion,.1)):
        model=fit_internal(features['inner'],parts['inner'].target,c)
        predictions={name:frame.assign(probability=model.predict_proba(features[name])[:,1]) for name,frame in parts.items()}
        if regime=='calibrated_images':
            if model.coef_[0,0]<=0:raise ValueError('Calibración no monótona creciente; revisar sin evaluar exterior')
            threshold=float(model.predict_proba(probability_logit([summary['selected_threshold']]).reshape(1,1))[0,1])
            for name in parts:
                assert np.array_equal(parts[name].probability>=summary['selected_threshold'],predictions[name].probability>=threshold)
        else:threshold=select_threshold(predictions['inner'])
        metrics={name:enrich(binary_metrics(frame.target,frame.probability,threshold)) for name,frame in predictions.items()}
        out=output/regime;out.mkdir(parents=True)
        for name,frame in predictions.items():frame.to_csv(out/f'predictions_{name}.csv',index=False)
        artifact=dict(schema='fixed_secondary_20261015',regime=regime,C=c,
            coefficients=model.coef_.tolist(),intercept=model.intercept_.tolist(),
            features=['cnn_patient_logit'] if regime=='calibrated_images' else list(FEATURES),
            clinical_preprocessor=prep.to_dict() if regime!='calibrated_images' else None,
            source_checkpoint=summary['inference_checkpoint'],source_checkpoint_sha256=summary['checkpoint_sha256'],
            source_summary_sha256=hashlib.sha256((folder/'summary.json').read_bytes()).hexdigest(),
            coefficient_fit_scope='inner_only',clinical_statistics_scope='fit_only',threshold_fit_scope='inner_only',
            threshold=threshold,cnn_frozen=True,test_evaluated=False,
            inner_reused_for_checkpoint_selection=True)
        if regime=='images_age_menopause':
            module=FrozenPatientFusion(model.coef_,model.intercept_).eval()
            with torch.no_grad():
                restored=torch.sigmoid(module(torch.from_numpy(parts['outer'].probability.to_numpy()),
                    torch.from_numpy(features['outer'][:,1:]))).numpy()
            np.testing.assert_allclose(restored,predictions['outer'].probability,atol=1e-12,rtol=1e-12)
            assert sum(p.numel() for p in module.parameters())==6
            torch.save(dict(state_dict=module.state_dict(),artifact=artifact),out/'linear_fusion.pt')
        write_json(out/'artifact.json',artifact)
        report=dict(regime=regime,fold=summary['outer_fold'],selected_profile=summary['selected_profile'],
            metrics=metrics,threshold=threshold,cnn_fit_auc=summary['train_eval']['metrics']['roc_auc'],
            cnn_outer_auc=summary['outer_eval']['metrics']['roc_auc'])
        write_json(out/'summary.json',report);reports.append(report)
    return reports


def main():
    out=ROOT/'secondary_fixed'
    if out.exists():raise FileExistsError('No sobrescribir análisis')
    folders=[ROOT/f'nested_cv_fold{fold}/resultados' for fold in range(5)]
    if not all((folder/'summary.json').exists() for folder in folders):
        raise ValueError('Esperar cinco folds completos; no adaptar análisis a resultados parciales')
    reports=[]
    for fold,folder in enumerate(folders):reports.extend(evaluate_saved_fold(folder,out/f'fold{fold}'))
    aggregate=[]
    for regime in ('calibrated_images','images_age_menopause'):
        group=[r for r in reports if r['regime']==regime]
        row=dict(regime=regime,folds=5)
        for metric in ('roc_auc','pr_auc','brier_score','f1_macro','balanced_accuracy','precision','sensitivity','specificity'):
            values=[r['metrics']['outer'][metric] for r in group]
            row[metric+'_mean']=float(np.mean(values));row[metric+'_sd']=float(np.std(values,ddof=1))
        row['confusion_sum']={key:sum(r['metrics']['outer']['confusion_matrix'][key] for r in group) for key in ('tn','fp','fn','tp')}
        aggregate.append(row)
    write_json(out/'aggregate.json',dict(protocol='fixed before first outer',results=aggregate,
        independent_trials=False,public_test_evaluated=False))
    pd.DataFrame(aggregate).to_csv(out/'aggregate.csv',index=False)
    print(json.dumps(aggregate,indent=2))


if __name__=='__main__':main()
