"""P05 · análisis con los criterios escritos antes de entrenar (research/P05_LR_BAJO.md).

Compara las predicciones exteriores de P05 (tasa 0,0001) con P02 normal y con la línea
base de cohorte, en las 1.097 pacientes de los 5 folds. Solo train; nunca toca test.

    python -m src.analysis.analisis_p05          # GPU (runs_cuda), escribe reports/new_approach/P05_lr_bajo/analisis/
    python -m src.analysis.analisis_p05 --mac    # si P05 se entrenó en el Mac (runs)
"""
import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score

from src.analysis.analisis_paso0 import COHORTES, bootstrap_pareado, metadatos

ROOT = Path(__file__).resolve().parents[2]
P02 = ROOT/'reports/new_approach/P02_aumento_geometrico/runs_cuda'
P05 = ROOT/'reports/new_approach/P05_lr_bajo'
CRITERIO_DOC_SHA = 'b761a9a9a9a8879f5accab360dd44a148f2442c3e775629b0d64fb991bf58fdf'
AUC_MINIMA = 0.55


def pacientes(m):
    return m.groupby('patient_id').agg(fold=('fold', 'first'), cohorte=('dataset', 'first'),
                                       target=('pCR', 'first')).reset_index()


def exteriores(carpeta, nombre='predictions_outer_patient.csv'):
    """Predicciones exteriores de los 5 folds (BCE normal). Falla si falta alguno (regla 2)."""
    partes = []
    for k in range(5):
        ruta = Path(carpeta)/f'normal_fold_{k}_seed_42'
        resumen = json.loads((ruta/'summary.json').read_text())
        if resumen.get('status') != 'complete':
            raise RuntimeError(f'Fold {k} incompleto en {carpeta}: no se decide con folds incompletos')
        partes.append(pd.read_csv(ruta/nombre).assign(fold=k))
    return pd.concat(partes, ignore_index=True)


def linea_base_cohorte(pac):
    """Proporción de pCR de la cohorte en los otros 4 folds."""
    salida = pd.Series(index=pac.index, dtype=float)
    for k in range(5):
        prev = pac[pac.fold != k].groupby('cohorte').target.mean()
        dentro = pac.fold == k
        salida[dentro] = pac.cohorte[dentro].map(prev)
    return salida


def epoca_fija(carpeta, pac):
    """Secundario: E* = época de mayor AUC interna media entre los 5 folds (solo épocas
    completadas por los 5) y AUC exterior conjunta de las predicciones de esa época."""
    historias = [pd.read_csv(Path(carpeta)/f'normal_fold_{k}_seed_42/history.csv') for k in range(5)]
    ultima = min(h.epoch.max() for h in historias)
    media = pd.concat([h.set_index('epoch').inner_auc.loc[:ultima] for h in historias], axis=1).mean(axis=1)
    e = int(media.idxmax())
    pred = exteriores(carpeta, 'predictions_outer_patient_by_epoch.csv')
    pred = pred[pred.epoch == e].merge(pac[['patient_id', 'target']], on='patient_id', suffixes=('_csv', ''))
    return {'epoca': e, 'auc_interna_media': round(float(media.max()), 4), 'epocas_comunes': int(ultima),
            'auc_exterior_conjunta': round(float(roc_auc_score(pred.target, pred.probability)), 4)}


def por_fold(carpeta):
    filas = []
    for k in range(5):
        s = json.loads((Path(carpeta)/f'normal_fold_{k}_seed_42/summary.json').read_text())
        fit, interna = s['fit_at_best']['metrics']['roc_auc'], s['inner_at_best']['metrics']['roc_auc']
        filas.append({'fold': k, 'epoca': s['best_epoch'], 'epocas': s['epochs_completed'],
                      'auc_fit': round(fit, 4), 'auc_interna': round(interna, 4),
                      'brecha_fit_interna': round(fit - interna, 4),
                      'auc_exterior': round(s['outer_at_05']['metrics']['roc_auc'], 4),
                      'umbral_interno': round(s['threshold'], 5)})
    return filas


def analizar(carpeta_p05, carpeta_p02, pac):
    df = pac.copy()
    df['cohorte_base'] = linea_base_cohorte(df)
    for nombre, carpeta in (('p05', carpeta_p05), ('p02', carpeta_p02)):
        e = exteriores(carpeta)[['patient_id', 'probability']].rename(columns={'probability': nombre})
        df = df.merge(e, on='patient_id', how='left')
    if df[['p05', 'p02']].isna().any().any():
        raise RuntimeError('Faltan pacientes en las predicciones exteriores')
    auc = lambda col, d=df: round(float(roc_auc_score(d.target, d[col])), 4)
    dif = lambda a, b: {'diferencia': round(auc(a) - auc(b), 4), 'ic95': bootstrap_pareado(df, a, b)}
    cortes = {}
    for nombre, carpeta in (('p05', carpeta_p05), ('p02', carpeta_p02)):
        c = exteriores(carpeta, 'predictions_outer_cut.csv')
        cortes[nombre] = round(float(roc_auc_score(c.target, c.probability)), 4)
    salida = {
        'criterio_doc_sha256_antes_de_entrenar': CRITERIO_DOC_SHA,
        'pacientes': len(df),
        'auc_conjunta': {k: auc(k) for k in ('p05', 'p02', 'cohorte_base')},
        'auc_por_corte': cortes,
        'p05_menos_p02': dif('p05', 'p02'),
        'p05_menos_cohorte': dif('p05', 'cohorte_base'),
        'dentro_de_cohorte': {COHORTES.get(c, c): {m: round(float(roc_auc_score(g.target, g[m])), 4) for m in ('p05', 'p02')}
                              for c, g in df.groupby('cohorte')},
        'por_fold': {'p05': por_fold(carpeta_p05), 'p02': por_fold(carpeta_p02)},
    }
    try:
        salida['secundario_epoca_fija'] = epoca_fija(carpeta_p05, pac)
    except FileNotFoundError as exc:
        salida['secundario_epoca_fija'] = f'no disponible: {exc}'
    principal = salida['p05_menos_p02']['ic95'][0] > 0
    salida['criterios'] = {
        'principal_p05_mejora_p02': bool(principal),
        'auc_conjunta_minima_055': bool(salida['auc_conjunta']['p05'] >= AUC_MINIMA),
        'supera_cohorte': bool(salida['p05_menos_cohorte']['ic95'][0] > 0),
    }
    salida['candidato_a_entrega'] = all(salida['criterios'].values())
    return salida, df


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--mac', action='store_true', help='leer P05 de runs (Mac) en vez de runs_cuda')
    args = parser.parse_args()
    out = P05/'analisis'
    out.mkdir(parents=True, exist_ok=True)
    salida, df = analizar(P05/('runs' if args.mac else 'runs_cuda'), P02, pacientes(metadatos()))
    df.to_csv(out/'predicciones_exteriores.csv', index=False)
    (out/'resumen.json').write_text(json.dumps(salida, indent=2, ensure_ascii=False))
    print(json.dumps(salida, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
