"""P03 · paso 0: análisis previo al recorte central (ver research/P03_PASO0.md).

Tres medidas, todas sin test y sin entrenar:

A. Dónde está el realce (EARLY − PRE positivo) en los cortes de train, por cohorte.
B. Mapas de oclusión de E57 en 60 cortes de su validación (fold 0): cuánto efecto cae en
   la periferia y en el borde mama–aire frente a lo que les correspondería por área.
C. Línea base de cohorte frente a P02 normal en la validación cruzada exterior.

    python -m src.analysis.analisis_paso0          # escribe reports/new_approach/P03_paso0/
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
DATOS = ROOT/'breastdcedl'
OUT = ROOT/'reports/new_approach/P03_paso0'
P02 = ROOT/'reports/new_approach/P02_aumento_geometrico/runs_cuda'
E57_CKPT = ROOT/'checkpoints/E57_patient_feature_mean_mac/fold_0_seed_42/best.pt'
E57_DECISION = ROOT/'reports/final_e57_test_v1/protocolo_previo.json'
TAMANOS = (128, 160, 192)
LADO = 256
SEMILLA = 42
COHORTES = {'spy1': 'I-SPY1', 'spy2': 'I-SPY2', 'duke': 'Duke'}
CRITERIO = dict(realce_minimo=0.60, centro_max_fraccion=1 / 8, razon_periferia=1.2, razon_borde=1.5)


def metadatos():
    muestras = pd.read_csv(DATOS/'metadata/samples.csv')
    pacientes = pd.read_csv(DATOS/'metadata/patients.csv', usecols=['pid', 'dataset'])
    m = muestras[muestras.split == 'train'].merge(pacientes, left_on='patient_id', right_on='pid')
    return m.drop(columns='pid')


def leer(ruta):
    return np.asarray(Image.open(DATOS/ruta), dtype=np.float32) / 255.0


def cuadrado(tam, lado=LADO):
    a = (lado - tam) // 2
    return slice(a, a + tam)


def fracciones_dentro(r, tamanos=TAMANOS):
    total = r.sum()
    if total <= 0:
        return {t: np.nan for t in tamanos}
    return {t: float(r[cuadrado(t), cuadrado(t)].sum() / total) for t in tamanos}


def centro_de_masas(mapa):
    total = mapa.sum()
    ys, xs = np.indices(mapa.shape)
    return float((ys * mapa).sum() / total), float((xs * mapa).sum() / total)


# ---------------------------------------------------------------- A. realce

def analisis_realce(m):
    filas, mapas, cuenta = [], {c: np.zeros((LADO, LADO)) for c in COHORTES}, {c: 0 for c in COHORTES}
    for s in m.itertuples(index=False):
        r = np.clip(leer(s.path_early) - leer(s.path_pre), 0, None)
        mapas[s.dataset] += r
        cuenta[s.dataset] += 1
        filas.append({'sample_id': s.sample_id, 'cohorte': s.dataset, 'realce_total': float(r.sum()),
                      **{f'dentro_{t}': v for t, v in fracciones_dentro(r).items()}})
    cortes = pd.DataFrame(filas)
    resumen = {}
    for c in COHORTES:
        medio = mapas[c] / max(cuenta[c], 1)
        np.save(OUT/f'realce_medio_{c}.npy', medio.astype(np.float32))
        cy, cx = centro_de_masas(medio)
        sub = cortes[cortes.cohorte == c]
        resumen[c] = {
            'cortes': int(cuenta[c]),
            'centro_de_masas_yx': [round(cy, 1), round(cx, 1)],
            'distancia_al_centro_px': round(float(np.hypot(cy - (LADO - 1) / 2, cx - (LADO - 1) / 2)), 1),
            **{f'mediana_dentro_{t}': round(float(sub[f'dentro_{t}'].median()), 3) for t in TAMANOS},
            **{f'p25_dentro_{t}': round(float(sub[f'dentro_{t}'].quantile(0.25)), 3) for t in TAMANOS},
            **{f'mapa_medio_dentro_{t}': round(fracciones_dentro(medio)[t], 3) for t in TAMANOS},
        }
    cumple = {t: all(resumen[c][f'mediana_dentro_{t}'] >= CRITERIO['realce_minimo']
                     and resumen[c]['distancia_al_centro_px'] < t * CRITERIO['centro_max_fraccion'] for c in COHORTES)
              for t in TAMANOS}
    elegido = min([t for t, ok in cumple.items() if ok], default=None)
    cortes.to_csv(OUT/'realce_por_corte.csv', index=False)
    return {'por_cohorte': resumen, 'area_dentro': {t: round(t * t / LADO ** 2, 3) for t in TAMANOS},
            'cumple_por_tamano': cumple, 'tamano_elegido': elegido}


# ---------------------------------------------------------------- B. oclusión

def elegir_cortes_validacion(m, por_grupo=10):
    rng = np.random.default_rng(SEMILLA)
    v = m[m.fold == 0]
    elegidos = []
    for c in COHORTES:
        for clase in (0, 1):
            pacientes = sorted(v[(v.dataset == c) & (v.pCR == clase)].patient_id.unique())
            for p in rng.permutation(pacientes)[:por_grupo]:
                cortes = v[v.patient_id == p].reset_index(drop=True)
                elegidos.append(cortes.iloc[rng.integers(len(cortes))])
    return pd.DataFrame(elegidos)


def razones(deltas, pre, tam, patch=16):
    efecto = np.abs(np.asarray(deltas))
    n = efecto.shape[0]
    dentro = np.zeros_like(efecto, dtype=bool)
    a, b = (LADO - tam) // 2 // patch, ((LADO - tam) // 2 + tam) // patch
    dentro[a:b, a:b] = True
    aire = (pre < 0.05).reshape(n, patch, n, patch).mean(axis=(1, 3))
    borde = (aire >= 0.2) & (aire <= 0.8)
    total = efecto.sum()
    if total <= 0:
        return np.nan, np.nan, 0.0
    r_perif = (efecto[~dentro].sum() / total) / (~dentro).mean()
    r_borde = (efecto[borde].sum() / total) / borde.mean() if borde.any() else np.nan
    return float(r_perif), float(r_borde), float(efecto.max())


def analisis_oclusion(m, tam):
    from src.pipeline.data import load_dce_triplet
    from src.web.inference import DCEInferenceEngine
    motor = DCEInferenceEngine.from_checkpoint(E57_CKPT, decision_path=E57_DECISION)
    filas = []
    for s in elegir_cortes_validacion(m).itertuples(index=False):
        imagen = load_dce_triplet({'PRE': DATOS/s.path_pre, 'EARLY': DATOS/s.path_early, 'LATE': DATOS/s.path_late})
        mapa = motor.occlusion_tensor(imagen)
        r_perif, r_borde, maximo = razones(mapa.deltas, imagen[0].numpy(), tam)
        filas.append({'sample_id': s.sample_id, 'cohorte': s.dataset, 'pCR': int(s.pCR), 'p_base': mapa.base_probability,
                      'efecto_max': maximo, 'razon_periferia': r_perif, 'razon_borde': r_borde})
    t = pd.DataFrame(filas)
    t.to_csv(OUT/'oclusion_por_corte.csv', index=False)
    mediana = lambda s: round(float(s.median()), 3)
    return {
        'cortes': len(t), 'tamano_recorte': tam, 'dispositivo': str(motor.device),
        'mediana_razon_periferia': mediana(t.razon_periferia), 'mediana_razon_borde': mediana(t.razon_borde),
        'mediana_efecto_max': round(float(t.efecto_max.median()), 4),
        'por_cohorte': {c: {'cortes': int((t.cohorte == c).sum()), 'razon_periferia': mediana(t[t.cohorte == c].razon_periferia),
                            'razon_borde': mediana(t[t.cohorte == c].razon_borde)} for c in COHORTES},
        'cumple': bool(t.razon_periferia.median() >= CRITERIO['razon_periferia'] or t.razon_borde.median() >= CRITERIO['razon_borde']),
    }


# ---------------------------------------------------------------- C. línea base de cohorte

def bootstrap_pareado(df, a, b, n=2000, semilla=SEMILLA):
    rng = np.random.default_rng(semilla)
    grupos = [g.index.to_numpy() for _, g in df.groupby(['fold', 'cohorte', 'target'])]
    difs = []
    for _ in range(n):
        idx = np.concatenate([rng.choice(g, len(g)) for g in grupos])
        d = df.loc[idx]
        difs.append(roc_auc_score(d.target, d[a]) - roc_auc_score(d.target, d[b]))
    return [round(float(np.percentile(difs, 2.5)), 4), round(float(np.percentile(difs, 97.5)), 4)]


def analisis_cohorte(m):
    pac = m.groupby('patient_id').agg(fold=('fold', 'first'), cohorte=('dataset', 'first'), pcr=('pCR', 'first')).reset_index()
    filas = []
    for k in range(5):
        ext = pd.read_csv(P02/f'normal_fold_{k}_seed_42/predictions_outer_patient.csv')
        resto = pac[pac.fold != k]
        prev = resto.groupby('cohorte').pcr.mean()
        e = ext.merge(pac[['patient_id', 'cohorte']], on='patient_id')
        e['fold'] = k
        e['cohorte_base'] = e.cohorte.map(prev)
        filas.append(e)
    df = pd.concat(filas, ignore_index=True)
    auc_p02, auc_coh = roc_auc_score(df.target, df.probability), roc_auc_score(df.target, df.cohorte_base)
    return {'pacientes': len(df), 'auc_p02_normal': round(auc_p02, 4), 'auc_linea_base_cohorte': round(auc_coh, 4),
            'diferencia_p02_menos_cohorte': round(auc_p02 - auc_coh, 4),
            'ic95_pareado': bootstrap_pareado(df, 'probability', 'cohorte_base'),
            'auc_p02_dentro_de_cohorte': {COHORTES[c]: round(roc_auc_score(g.target, g.probability), 4) for c, g in df.groupby('cohorte')}}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEMILLA)
    m = metadatos()
    a = analisis_realce(m)
    tam = a['tamano_elegido'] or 160
    b = analisis_oclusion(m, tam)
    c = analisis_cohorte(m)
    decision = bool(a['tamano_elegido'] and b['cumple'])
    salida = {'criterios': CRITERIO, 'criterios_doc_sha256_antes_de_calcular': '7f851e4fe734cf94fc0a9b3180e075446a5211ecd19f376dacd0418a0347ce9f',
              'A_realce': a, 'B_oclusion': b, 'C_cohorte': c,
              'decision': {'proponer_P03': decision, 'tamano_recorte': a['tamano_elegido']}}
    (OUT/'resumen.json').write_text(json.dumps(salida, indent=2, ensure_ascii=False, default=str))
    print(json.dumps(salida, indent=2, ensure_ascii=False, default=str))


if __name__ == '__main__':
    main()
