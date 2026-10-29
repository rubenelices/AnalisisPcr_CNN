"""P04 · paso 0b: ¿lleva el realce relativo señal sobre pCR más allá de la cohorte?
(ver research/P04_PASO0B.md, escrito antes de calcular).

Variables de realce por paciente (relativas, PE y SER, frente a absolutas, restas), regresión
logística con los 5 folds del profesor y comparaciones pareadas con la línea base de cohorte.
Solo train; no entrena la CNN ni toca test. Las variables no entran en ningún modelo de entrega.

    python -m src.analysis.analisis_paso0b          # escribe reports/new_approach/P04_paso0b/
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.analysis.analisis_paso0 import COHORTES, bootstrap_pareado, leer, metadatos

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'reports/new_approach/P04_paso0b'
TEJIDO = 0.05          # PRE mínimo para considerar un píxel tejido (valores / 255)
PE_UMBRAL = 0.70       # realce porcentual del volumen tumoral funcional de I-SPY
SER_LAVADO = 1.1
ABS_UMBRAL = 0.10
RELATIVAS = ['frac_pe70', 'pe_top5', 'frac_lavado']
ABSOLUTAS = ['frac_abs10', 'abs_top5', 'frac_abs_lavado']
CRITERIO_DOC_SHA = '3325f5d2120b9a0d6334a531f994c1e50fc61490f3e113cfc81e834ce6731cf7'


def variables_corte(pre, early, late):
    """Seis variables de un corte: tres relativas (PE, SER) y sus equivalentes con restas."""
    tejido = pre >= TEJIDO
    if tejido.sum() < 50:
        return {k: np.nan for k in RELATIVAS + ABSOLUTAS}
    p, e, l = pre[tejido], early[tejido], late[tejido]
    pe = (e - p) / p
    d = e - p
    caliente = pe >= PE_UMBRAL
    denom = l - p
    valido = caliente & (denom > 0.01)
    ser = np.where(valido, d / np.where(valido, denom, 1), 0)
    caliente_abs = d >= ABS_UMBRAL
    return {
        'frac_pe70': float(caliente.mean()),
        'pe_top5': float(pe[pe >= np.percentile(pe, 95)].mean()),
        'frac_lavado': float((ser[valido] > SER_LAVADO).sum() / max(caliente.sum(), 1)),
        'frac_abs10': float(caliente_abs.mean()),
        'abs_top5': float(d[d >= np.percentile(d, 95)].mean()),
        'frac_abs_lavado': float(((l - e)[caliente_abs] < 0).sum() / max(caliente_abs.sum(), 1)),
    }


def variables_paciente(m):
    filas = [{'patient_id': s.patient_id, **variables_corte(leer(s.path_pre), leer(s.path_early), leer(s.path_late))}
             for s in m.itertuples(index=False)]
    v = pd.DataFrame(filas).groupby('patient_id').mean()
    pac = m.groupby('patient_id').agg(fold=('fold', 'first'), cohorte=('dataset', 'first'), target=('pCR', 'first'))
    return pac.join(v).reset_index()


def predicciones_exteriores(df, columnas, con_cohorte=True):
    """Regresión logística entrenada en 4 folds y evaluada en el quinto, para los 5."""
    X = df[columnas].copy() if columnas else pd.DataFrame(index=df.index)
    if con_cohorte:
        X = X.join(pd.get_dummies(df.cohorte, prefix='coh', dtype=float))
    X = X.fillna(X.median())
    salida = pd.Series(np.nan, index=df.index)
    for k in range(5):
        tr, te = df.fold != k, df.fold == k
        modelo = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
        modelo.fit(X[tr], df.target[tr])
        salida[te] = modelo.predict_proba(X[te])[:, 1]
    return salida


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = variables_paciente(metadatos())
    df['p_cohorte'] = predicciones_exteriores(df, [])
    df['p_cohorte_relativas'] = predicciones_exteriores(df, RELATIVAS)
    df['p_cohorte_absolutas'] = predicciones_exteriores(df, ABSOLUTAS)
    df['p_relativas'] = predicciones_exteriores(df, RELATIVAS, con_cohorte=False)
    df['p_absolutas'] = predicciones_exteriores(df, ABSOLUTAS, con_cohorte=False)
    df.to_csv(OUT/'variables_y_predicciones.csv', index=False)

    auc = lambda col, d=df: round(float(roc_auc_score(d.target, d[col])), 4)
    modelos = ['p_cohorte', 'p_cohorte_relativas', 'p_cohorte_absolutas', 'p_relativas', 'p_absolutas']
    dif = lambda a, b: {'diferencia': round(auc(a) - auc(b), 4), 'ic95': bootstrap_pareado(df, a, b)}
    salida = {
        'criterio_doc_sha256_antes_de_calcular': CRITERIO_DOC_SHA,
        'pacientes': len(df),
        'auc_conjunta': {m: auc(m) for m in modelos},
        'dentro_de_cohorte': {COHORTES[c]: {m: round(float(roc_auc_score(g.target, g[m])), 4) for m in ('p_relativas', 'p_absolutas')}
                              for c, g in df.groupby('cohorte')},
        'cohorte_relativas_menos_cohorte': dif('p_cohorte_relativas', 'p_cohorte'),
        'cohorte_relativas_menos_cohorte_absolutas': dif('p_cohorte_relativas', 'p_cohorte_absolutas'),
        'cohorte_absolutas_menos_cohorte': dif('p_cohorte_absolutas', 'p_cohorte'),
        'medias_por_cohorte': df.groupby('cohorte')[RELATIVAS + ABSOLUTAS].mean().round(4).to_dict(orient='index'),
    }
    salida['preparar_P04'] = bool(salida['cohorte_relativas_menos_cohorte']['ic95'][0] > 0)
    (OUT/'resumen.json').write_text(json.dumps(salida, indent=2, ensure_ascii=False))
    print(json.dumps(salida, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
