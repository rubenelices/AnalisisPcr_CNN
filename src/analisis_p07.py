"""P07 · análisis con los criterios escritos antes de entrenar (nuevo_enfoque/P07_SEMILLAS.md).

Ensemble de P02 normal con semillas 42, 17 y 123 frente a la semilla 42 sola y frente a la
línea base de cohorte, en las 1.097 pacientes de los 5 folds. Solo train; nunca toca test.

    python -m src.analisis_p07          # escribe reports/new_approach/P07_semillas/
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.analisis_p05 import linea_base_cohorte, pacientes
from src.analisis_paso0 import COHORTES, bootstrap_pareado, metadatos
from src.new_approach_protocol import select_threshold

ROOT = Path(__file__).resolve().parents[1]
P02 = ROOT/'reports/new_approach/P02_aumento_geometrico'
# Desviación anotada en P07_SEMILLAS.md: la 42 se entrenó en la GPU y la 17 y la 123 en el Mac.
CARPETAS = {42: P02/'runs_cuda', 17: P02/'runs', 123: P02/'runs'}
OUT = ROOT/'reports/new_approach/P07_semillas'
SEMILLAS = (42, 17, 123)
CRITERIO_DOC_SHA = 'd6012995eb763b94fdc4a60ccc29c7ac6db9b6fc657195c81df37ce7fda17fb9'
AUC_MINIMA = 0.55


def leer(carpetas, fold, seed, nombre):
    """`carpetas`: una carpeta para todas las semillas o un dict semilla → carpeta."""
    carpeta = carpetas[seed] if isinstance(carpetas, dict) else carpetas
    ruta = Path(carpeta)/f'normal_fold_{fold}_seed_{seed}'
    if json.loads((ruta/'summary.json').read_text()).get('status') != 'complete':
        raise RuntimeError(f'Fold {fold} semilla {seed} incompleto: no se decide con folds incompletos')
    return pd.read_csv(ruta/nombre)


def promedio(carpeta, fold, nombre, clave='patient_id'):
    """Media de probabilidades de las 3 semillas (por paciente o por corte)."""
    partes = [leer(carpeta, fold, s, nombre).set_index(clave)[['probability', 'target']] for s in SEMILLAS]
    if any(set(p.index) != set(partes[0].index) for p in partes):
        raise RuntimeError(f'Las semillas del fold {fold} no tienen las mismas filas')
    media = pd.concat([p.probability for p in partes], axis=1).mean(axis=1)
    return pd.DataFrame({'probability': media, 'target': partes[0].target}).reset_index()


def analizar(carpeta, pac):
    df = pac.copy()
    df['cohorte_base'] = linea_base_cohorte(df)
    umbrales, cortes = [], []
    for k in range(5):
        interna = promedio(carpeta, k, 'predictions_inner_patient.csv')
        umbrales.append(float(select_threshold(interna)[0]))  # solo validación interna
        cortes.append(promedio(carpeta, k, 'predictions_outer_cut.csv', 'sample_id').assign(fold=k))
    exteriores = {}
    for s in SEMILLAS:
        e = pd.concat([leer(carpeta, k, s, 'predictions_outer_patient.csv') for k in range(5)])
        exteriores[f'semilla_{s}'] = e.set_index('patient_id').probability
    for nombre, serie in exteriores.items():
        df[nombre] = df.patient_id.map(serie)
    df['ensemble'] = df[list(exteriores)].mean(axis=1)
    if df[['ensemble', *exteriores]].isna().any().any():
        raise RuntimeError('Faltan pacientes en las predicciones exteriores')
    df['umbral'] = df.fold.map(dict(enumerate(umbrales)))
    df['prediccion'] = (df.ensemble >= df.umbral).astype(int)

    auc = lambda col, d=df: round(float(roc_auc_score(d.target, d[col])), 4)
    dif = lambda a, b: {'diferencia': round(auc(a) - auc(b), 4), 'ic95': bootstrap_pareado(df, a, b)}
    individuales = {n: auc(n) for n in exteriores}
    c = pd.concat(cortes)
    tn, fp, fn, tp = [int(((df.target == t) & (df.prediccion == p)).sum()) for t, p in ((0, 0), (0, 1), (1, 0), (1, 1))]
    salida = {
        'criterio_doc_sha256_antes_de_entrenar': CRITERIO_DOC_SHA,
        'pacientes': len(df),
        'auc_conjunta': {'ensemble': auc('ensemble'), **individuales, 'cohorte_base': auc('cohorte_base')},
        'media_semillas_individuales': round(float(np.mean(list(individuales.values()))), 4),
        'ensemble_menos_semilla_42': dif('ensemble', 'semilla_42'),
        'ensemble_menos_cohorte': dif('ensemble', 'cohorte_base'),
        'auc_por_corte_ensemble': round(float(roc_auc_score(c.target, c.probability)), 4),
        'dentro_de_cohorte': {COHORTES.get(k, k): {m: auc(m, g) for m in ('ensemble', 'semilla_42')}
                              for k, g in df.groupby('cohorte')},
        'umbrales_internos_ensemble': [round(u, 5) for u in umbrales],
        'umbral_entrega_mediana': round(float(np.median(umbrales)), 5),
        'matriz_conjunta_umbral_interno': {'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp,
                                           'sensibilidad': round(tp/(tp+fn), 3), 'especificidad': round(tn/(tn+fp), 3)},
    }
    salida['criterios'] = {
        'principal_ensemble_mejora_semilla_42': bool(salida['ensemble_menos_semilla_42']['ic95'][0] > 0),
        'auc_conjunta_minima_055': bool(salida['auc_conjunta']['ensemble'] >= AUC_MINIMA),
        'supera_cohorte': bool(salida['ensemble_menos_cohorte']['ic95'][0] > 0),
    }
    salida['candidato_a_entrega'] = all(salida['criterios'].values())
    return salida, df


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    salida, df = analizar(CARPETAS, pacientes(metadatos()))
    salida['carpetas'] = {s: str(c.relative_to(ROOT)) for s, c in CARPETAS.items()}
    df.to_csv(OUT/'predicciones_exteriores.csv', index=False)
    (OUT/'resumen.json').write_text(json.dumps(salida, indent=2, ensure_ascii=False))
    print(json.dumps(salida, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
