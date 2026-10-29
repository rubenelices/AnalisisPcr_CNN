"""Datos de las gráficas de la web: frontend/public/results/graficas.json.

La web dibuja sus propias gráficas (SVG) con estos datos, en lugar de mostrar PNG de matplotlib.
Solo lee salidas ya guardadas: historiales, predicciones de validación y la única evaluación de
test de E57. No entrena, no evalúa y no vuelve a consultar test.

    python -m src.web.web_graficas          # y después: npm --prefix frontend run build
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

from src.evaluation import cv_report

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT/'frontend/public/results/graficas.json'
E57 = ROOT/'reports/patient_features/E57_patient_feature_mean_mac/fold_0_seed_42'
E57_FOLDS = [E57] + [ROOT/f'experiments/E57_patient_feature_mean_mac/stability/resultados/fold_{k}_seed_42' for k in range(1, 5)]
TEST = ROOT/'reports/final_e57_test_v1'
P02 = ROOT/'reports/new_approach/P02_aumento_geometrico/runs_cuda'
OVERFIT = ROOT/'reports/new_approach/P01_lotes_pacientes/overfit'
E57_CV_ANIDADA = 0.527152   # validación anidada de 5 folds de E57 (ver tabla_e57 en cuaderno_resultados)


def _r(v, d=4):
    return None if v is None or (isinstance(v, float) and not np.isfinite(v)) else round(float(v), d)


def _serie(df, columnas):
    return [{c: (int(f[c]) if c == 'epoch' else _r(f[c])) for c in columnas} for _, f in df.iterrows()]


def reducir(x, y, n=120):
    """Muestrea una curva a ~n puntos conservando los extremos (para que el JSON pese poco)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) <= n:
        idx = np.arange(len(x))
    else:
        idx = np.unique(np.r_[np.linspace(0, len(x) - 1, n).round().astype(int), len(x) - 1])
    return [[_r(x[i]), _r(y[i])] for i in idx]


def curvas(y, p):
    """ROC, PR, AUC y AP de unas predicciones por paciente."""
    fpr, tpr, _ = roc_curve(y, p)
    prec, rec, _ = precision_recall_curve(y, p)
    orden = np.argsort(rec)
    return dict(n=int(len(y)), pcr=int(np.sum(y)), auc=_r(roc_auc_score(y, p)), ap=_r(average_precision_score(y, p)),
                roc=reducir(fpr, tpr), pr=reducir(rec[orden], prec[orden]))


def calibracion(y, p, grupos=8):
    q = pd.qcut(pd.Series(p), grupos, duplicates='drop')
    g = pd.DataFrame({'p': p, 'y': y}).groupby(q, observed=True).agg(p=('p', 'mean'), y=('y', 'mean'), n=('y', 'size'))
    return [[_r(a), _r(b), int(c)] for a, b, c in g.itertuples(index=False)]


def histograma(y, p, umbral, cajas=20):
    bordes = np.linspace(0, max(0.6, float(np.max(p)) + 0.02), cajas + 1)
    return dict(bordes=[_r(b, 3) for b in bordes],
                sin_pcr=np.histogram(p[y == 0], bordes)[0].tolist(),
                pcr=np.histogram(p[y == 1], bordes)[0].tolist(), umbral=umbral)


def umbrales(y, p, hasta=0.7, paso=0.005):
    """Matriz de confusión por paciente en cada umbral (p ≥ umbral es pCR, como en la web): [umbral, VP, FP, VN, FN].

    Es para el deslizador del laboratorio: enseña el compromiso entre sensibilidad y especificidad. El umbral
    entregado (0,325) se fijó con la validación; mover el deslizador sobre el test no elige nada."""
    y, p = np.asarray(y).astype(bool), np.asarray(p)
    filas = []
    for u in np.round(np.arange(0, hasta + paso / 2, paso), 3):
        si = p >= u
        filas.append([float(u), int((si & y).sum()), int((si & ~y).sum()), int((~si & ~y).sum()), int((~si & y).sum())])
    return filas


def referencia(prevalencia):
    fpr, tpr, rec, prec, ap = cv_report.referencia(prevalencia=prevalencia)
    return dict(auc=cv_report.AUC_REFERENCIA, ap=_r(ap), roc=reducir(fpr, tpr, 60), pr=reducir(rec, prec, 60))


def _e57():
    h = pd.read_csv(E57/'history.csv')
    s = json.loads((E57/'summary.json').read_text())
    val = pd.read_csv(E57/'predictions_patient.csv')
    test = pd.read_csv(TEST/'predictions_patient.csv')
    t = json.loads((TEST/'test_evaluation.json').read_text())
    umbral = float(t['threshold'])
    yt, pt = test.target.to_numpy(), test.probability.to_numpy()
    return dict(
        epoca_elegida=int(s['best_epoch']), epocas=int(s['epochs_completed']), umbral=umbral,
        historia=_serie(h, ['epoch', 'train_loss', 'validation_loss', 'patient_roc_auc', 'patient_pr_auc']),
        perdidas_folds=[[_r(v) for v in pd.read_csv(f/'history.csv').train_loss] for f in E57_FOLDS],
        picos_folds=[_r(pd.read_csv(f/'history.csv').patient_roc_auc.max()) for f in E57_FOLDS],
        validacion=curvas(val.target.to_numpy(), val.probability.to_numpy()),
        test=curvas(yt, pt) | dict(calibracion=calibracion(yt, pt), histograma=histograma(yt, pt, umbral),
                                   corte=dict(auc=_r(t['cut_metrics_secondary']['roc_auc']), n=t['cut_metrics_secondary']['n']),
                                   brier=_r(t['patient_metrics_primary']['brier_score']),
                                   ece=_r(t['calibration']['expected_calibration_error'])),
        umbrales=dict(validacion=umbrales(val.target.to_numpy(), val.probability.to_numpy()), test=umbrales(yt, pt)),
        cv_anidada=E57_CV_ANIDADA,
        referencia_validacion=referencia(float(val.target.mean())),
        referencia_test=referencia(float(yt.mean())),
    )


def _p02():
    salida = {}
    for loss in ('normal', 'weighted'):
        folds = []
        for k in range(5):
            d = P02/f'{loss}_fold_{k}_seed_42'
            if not (d/'summary.json').is_file():
                break
            s = json.loads((d/'summary.json').read_text())
            exterior = pd.read_csv(d/'predictions_outer_patient.csv')
            folds.append(dict(fold=k, epoca=int(s['best_epoch']),
                              exterior=_r(roc_auc_score(exterior.target, exterior.probability)),
                              historia=_serie(pd.read_csv(d/'history.csv'), ['epoch', 'fit_bce', 'inner_bce', 'fit_auc', 'inner_auc'])))
        if folds:
            salida[loss] = folds
    return salida


def _memorizacion():
    return {receta: _serie(pd.read_csv(OVERFIT/receta/'history.csv'), ['epoch', 'train_online_objective', 'eval_auc'])
            for receta in ('e57', 'p01') if (OVERFIT/receta/'history.csv').is_file()}


def build():
    datos = dict(bce_constante=_r(cv_report.bce_constante()), bce_referencia=_r(cv_report.bce_referencia()),
                 auc_referencia=cv_report.AUC_REFERENCIA, e57=_e57(), p02=_p02(), memorizacion=_memorizacion())
    DEST.parent.mkdir(parents=True, exist_ok=True)
    DEST.write_text(json.dumps(datos, ensure_ascii=False, separators=(',', ':')))
    return DEST


if __name__ == '__main__':
    print(build())
