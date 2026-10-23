"""Exporta las cifras de la sección Resultados de la web a frontend/public/resultados/.

Lee salidas ya guardadas (test de E57, informe de 5 folds de P02, historiales de E57);
no entrena ni evalúa nada. Así la web no lleva números escritos a mano:

    python -m src.web_resultados          # y después: npm --prefix frontend run build

Si todavía falta una pérdida de la validación cruzada, su apartado sale como pendiente.
"""
import json
import shutil
from datetime import date
from pathlib import Path

import pandas as pd

from src import cv_report

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT/'frontend/public/resultados'
CV = ROOT/'reports/new_approach/P02_aumento_geometrico/cv_runs_cuda'


def _test_e57():
    t = json.loads((ROOT/'reports/final_e57_test_v1/test_evaluation.json').read_text())
    m, ic = t['patient_metrics_primary'], t['bootstrap']['intervals']['roc_auc']
    c = m['confusion_matrix']
    return dict(auc=m['roc_auc'], ic95=[ic['lower'], ic['upper']], ap=m['pr_auc'],
                sensibilidad=m['sensitivity'], especificidad=m['specificity'], precision=m['precision'],
                balanced_accuracy=m['balanced_accuracy'], matriz=c, pacientes=c['tn'] + c['fp'] + c['fn'] + c['tp'],
                pcr=c['fn'] + c['tp'], umbral=t['threshold'])


def _picos_e57():
    picos = [pd.read_csv(ROOT/'reports/patient_features/E57_patient_feature_mean_mac/fold_0_seed_42/history.csv').patient_roc_auc.max()]
    for k in range(1, 5):
        picos.append(pd.read_csv(ROOT/f'experimentos_equilibrados/E57_estabilidad/resultados/fold_{k}_seed_42/history.csv').patient_roc_auc.max())
    return dict(fold0=float(picos[0]), media=float(sum(picos) / len(picos)))


def _cv():
    path = CV/'resumen.json'
    if not path.is_file():
        return {}
    s = json.loads(path.read_text())
    salida = {}
    for loss in cv_report.LOSSES:
        x = s.get(loss)
        if not x or x['n_folds'] != 5:
            continue
        mi = x['matrices']['umbral interno (por fold)']
        salida[loss] = dict(
            auc=x['auc_conjunta'], ic95=x['auc_conjunta_ic95'], media_folds=x['auc_media_folds'],
            auc_corte=x['auc_corte_conjunta'], ap=x['ap_conjunta'], prevalencia=x['prevalencia'],
            pacientes=x['n_pacientes'], interna_media=x['auc_interna_media'], train_media=x['auc_fit_media'],
            folds=[dict(fold=f['fold'], epoca=f['epoca_elegida'], exterior=f['auc_exterior'], interna=f['auc_interna'])
                   for f in x['folds']],
            cohortes={c: dict(n=v['n'], pcr=v['pcr'], auc=v['auc']) for c, v in x['cohortes'].items()},
            matriz=dict(tn=mi['tn'], fp=mi['fp'], fn=mi['fn'], tp=mi['tp'], sensibilidad=mi['sensitivity'],
                        especificidad=mi['specificity'], balanced_accuracy=mi['balanced_accuracy']))
    if 'pareada' in s and len(salida) == 2:
        salida['pareada'] = s['pareada']
    return salida


def build():
    DEST.mkdir(parents=True, exist_ok=True)
    datos = dict(actualizado=date.today().isoformat(), test_e57=_test_e57(), picos_e57=_picos_e57(), cv_p02=_cv(),
                 referencia=dict(auc=cv_report.AUC_REFERENCIA, bce=cv_report.bce_referencia(), bce_constante=cv_report.bce_constante()))
    (DEST/'resumen.json').write_text(json.dumps(datos, indent=2, ensure_ascii=False))
    # Figuras propias (sin imágenes de pacientes), generadas por cv_report / cuaderno_resultados.
    for origen, nombre in ((CV/'curvas_normal.png', 'curvas_normal.png'), (CV/'curvas_weighted.png', 'curvas_ponderada.png'),
                           (CV/'por_fold/normal_fold_0_roc_interna_exterior.png', 'roc_fold0.png'),
                           (CV/'matrices.png', 'matrices.png'), (CV/'roc_pr.png', 'roc_pr.png'),
                           (CV/'cohortes_folds.png', 'cohortes_folds.png'), (CV/'calibracion.png', 'calibracion.png')):
        if origen.is_file():
            shutil.copy(origen, DEST/nombre)
    return DEST/'resumen.json'


if __name__ == '__main__':
    print(build())
