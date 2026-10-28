"""P06 · paso 0c: ¿hay señal sobre pCR en la región que más realza? (ver nuevo_enfoque/P06_PASO0C.md,
escrito antes de calcular). Mide el realce solo en una ventana candidata a tumor, sin máscara, y
compara cohorte + región frente a solo cohorte. Solo train; no entrena la CNN ni toca test.

    python -m src.analisis_paso0c          # escribe reports/new_approach/P06_paso0c/
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter
from sklearn.metrics import roc_auc_score

from src.analisis_paso0 import COHORTES, bootstrap_pareado, leer, metadatos
from src.analisis_paso0b import predicciones_exteriores

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'reports/new_approach/P06_paso0c'
TEJIDO, PE_UMBRAL, SER_LAVADO = 0.05, 0.70, 1.1
SUAVIZADO, MARGEN, VENTANA, ENTORNO = 15, 16, 48, 96
BASE = ['pe_medio', 'pe_p90', 'frac_pe70', 'frac_lavado', 'heterogeneidad', 'tamano']
CRITERIO_DOC_SHA = 'f0667e73d98a9467d9026e5fb794201b75ef823f57b69fa2725db4dcd09468c0'


def ventana(centro, lado, tam=256):
    a = int(np.clip(centro - lado // 2, 0, tam - lado))
    return slice(a, a + lado)


def region_candidata(pre, early, late):
    """Centro de la región que más realza (PE suavizado, fuera del margen) y sus variables."""
    tejido = pre >= TEJIDO
    pe = np.where(tejido, (early - pre) / np.maximum(pre, TEJIDO), 0.0)
    suave = uniform_filter(pe, SUAVIZADO)
    suave[:MARGEN], suave[-MARGEN:], suave[:, :MARGEN], suave[:, -MARGEN:] = -np.inf, -np.inf, -np.inf, -np.inf
    y, x = np.unravel_index(np.argmax(suave), suave.shape)
    sy, sx = ventana(y, VENTANA), ventana(x, VENTANA)
    t = tejido[sy, sx]
    if t.sum() < 20:
        return (int(y), int(x)), {k: np.nan for k in BASE}
    p, e, l = pre[sy, sx][t], early[sy, sx][t], late[sy, sx][t]
    pe_v = (e - p) / p
    caliente = pe_v >= PE_UMBRAL
    denom = l - p
    valido = caliente & (denom > 0.01)
    ser = np.where(valido, (e - p) / np.where(valido, denom, 1), 0)
    ey, ex = ventana(y, ENTORNO), ventana(x, ENTORNO)
    variables = {
        'pe_medio': float(pe_v.mean()), 'pe_p90': float(np.percentile(pe_v, 90)), 'frac_pe70': float(caliente.mean()),
        'frac_lavado': float((ser[valido] > SER_LAVADO).sum() / max(caliente.sum(), 1)),
        'heterogeneidad': float((e - p).std()), 'tamano': float(((pe >= PE_UMBRAL) & tejido)[ey, ex].sum()),
    }
    return (int(y), int(x)), variables


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    m = metadatos()
    filas = []
    for s in m.itertuples(index=False):
        (y, x), v = region_candidata(leer(s.path_pre), leer(s.path_early), leer(s.path_late))
        filas.append({'patient_id': s.patient_id, 'y': y, 'x': x, **v})
    cortes = pd.DataFrame(filas)
    cortes.to_csv(OUT/'region_por_corte.csv', index=False)
    agg = cortes.groupby('patient_id')[BASE].agg(['mean', 'max'])
    agg.columns = [f'{a}_{b}' for a, b in agg.columns]
    pac = m.groupby('patient_id').agg(fold=('fold', 'first'), cohorte=('dataset', 'first'), target=('pCR', 'first'))
    df = pac.join(agg).reset_index()
    cols = list(agg.columns)
    df['p_cohorte'] = predicciones_exteriores(df, [])
    df['p_cohorte_region'] = predicciones_exteriores(df, cols)
    df['p_region'] = predicciones_exteriores(df, cols, con_cohorte=False)
    df.to_csv(OUT/'variables_y_predicciones.csv', index=False)

    auc = lambda col, d=df: round(float(roc_auc_score(d.target, d[col])), 4)
    centrado = ((cortes.y - 127.5).abs() < 64) & ((cortes.x - 127.5).abs() < 64)
    cortes['cohorte'] = cortes.patient_id.map(pac.cohorte)
    salida = {
        'criterio_doc_sha256_antes_de_calcular': CRITERIO_DOC_SHA,
        'pacientes': len(df),
        'auc_conjunta': {k: auc(k) for k in ('p_cohorte', 'p_cohorte_region', 'p_region')},
        'cohorte_region_menos_cohorte': {'diferencia': round(auc('p_cohorte_region') - auc('p_cohorte'), 4),
                                         'ic95': bootstrap_pareado(df, 'p_cohorte_region', 'p_cohorte')},
        'region_dentro_de_cohorte': {COHORTES[c]: round(float(roc_auc_score(g.target, g.p_region)), 4) for c, g in df.groupby('cohorte')},
        'fraccion_regiones_en_centro_128': {COHORTES[c]: round(float(centrado[cortes.cohorte == c].mean()), 3) for c in COHORTES},
    }
    salida['preparar_P06'] = bool(salida['cohorte_region_menos_cohorte']['ic95'][0] > 0)
    (OUT/'resumen.json').write_text(json.dumps(salida, indent=2, ensure_ascii=False))
    print(json.dumps(salida, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
