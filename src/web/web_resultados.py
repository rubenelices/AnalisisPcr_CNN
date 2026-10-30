"""Exporta las cifras de la sección Resultados de la web a frontend/public/results/.

Lee salidas ya guardadas (test de E57, validación cruzada de su receta N01, informe de 5 folds de P02,
historiales de E57);
no entrena ni evalúa nada. Así la web no lleva números escritos a mano:

    python -m src.web.web_resultados          # y después: npm --prefix frontend run build

Si todavía falta una pérdida de la validación cruzada, su apartado sale como pendiente.
"""
import json
import shutil
from datetime import date
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score

from src.evaluation import cv_report

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT/'frontend/public/results'
CV = ROOT/'reports/new_approach/P02_aumento_geometrico/cv_runs_cuda'
RUNS = ROOT/'reports/new_approach/P02_aumento_geometrico/runs_cuda'
N01 = ROOT/'reports/new_approach/N01_e57_bce'
HISTORIAL = ROOT/'research/HISTORIAL_EXPERIMENTOS.md'
METADATA = ROOT/'breastdcedl/metadata'

# Diez experimentos del historial, elegidos por lo que enseñan. Las cifras salen de la tabla del historial;
# aquí solo se escribe qué cambia en cada uno. Las filas «clínica» usan edad/menopausia o HR/HER2 y no son
# comparables con las de solo imagen (el modelo entregado es solo imagen).
EXPERIMENTOS = [
    ('C17_normal', 'Referencia inicial', 'CNN propia con BCE normal, fold 0.'),
    ('C22_groupnorm_normal', 'GroupNorm', 'Sustituye BatchNorm por GroupNorm.'),
    ('C23_un_corte_paciente_normal_fold2', 'Un corte por paciente', 'Cada paciente aporta un solo corte, fold 2.'),
    ('E57_patient_feature_mean_mac', 'E57 (modelo entregado)', 'Media de rasgos por paciente, fold 0.'),
    ('N01_e57_bce_weighted_fold_0_seed_42', 'E57 con BCE ponderada', 'Misma red con pos_weight = N0/N1.'),
    ('bag_fixed_fold1', 'Época fija', 'Pérdida por paciente y 10 épocas, sin elegir el pico.'),
    ('B08_solo_edad_menopausia', 'Solo edad y menopausia', 'Sin imágenes: control con dos datos clínicos.'),
    ('D08_clinica_sola', 'Solo datos clínicos', 'Sin imágenes: control con la clínica basal.'),
    ('D17_ganancia_compartida', 'Imagen + clínica', 'La CNN recibe además datos clínicos.'),
    ('D36_normal_40', 'Imagen + clínica, 40 épocas', 'Entrenamiento largo para forzar la memorización.'),
]


def leer_historial(path=HISTORIAL):
    """Filas de la tabla del historial como diccionarios (las columnas numéricas, en float o None)."""
    filas, cab = [], None
    for linea in Path(path).read_text().splitlines():
        if not linea.startswith('|') or linea.startswith('|:'):
            continue
        celdas = [c.strip() for c in linea.strip().strip('|').split('|')]
        if cab is None:
            cab = celdas
            continue
        f = dict(zip(cab, celdas))
        for k in ('epocas', 'mejor_epoca', 'auc_val_max', 'auc_val_final', 'auc_train_final'):
            try:
                v = float(f.get(k, ''))
                f[k] = None if v != v else v
            except ValueError:
                f[k] = None
        filas.append(f)
    return filas


def _experimentos(path=HISTORIAL, seleccion=EXPERIMENTOS):
    por_id = {f['experimento']: f for f in leer_historial(path)}
    salida = []
    for clave, nombre, cambia in seleccion:
        f = por_id[clave]
        pico, final, fit = f['auc_val_max'], f['auc_val_final'], f['auc_train_final']
        salida.append(dict(id=clave, nombre=nombre, cambia=cambia, clinica='clínica' in f['entradas'], fold=f['fold'] or None,
                           epocas=f['epocas'], mejor_epoca=f['mejor_epoca'], pico=pico, final=final, fit=fit,
                           brecha=None if fit is None or final is None else fit - final,
                           caida=None if pico is None or final is None else pico - final))
    return salida


def _particiones(metadata=METADATA):
    """Reparto de pacientes y cortes en los 5 folds del profesor, más el test público y las cohortes.

    Solo cuenta (solo lee `samples.csv` y tres columnas de `patients.csv`); del test no se expone ninguna
    etiqueta, solo cuántas pacientes y cortes hay."""
    s = pd.read_csv(Path(metadata)/'samples.csv', usecols=['sample_id', 'patient_id', 'split', 'pCR', 'fold'])
    p = pd.read_csv(Path(metadata)/'patients.csv', usecols=['pid', 'dataset', 'split']).rename(columns={'split': 'split_p'})
    tr = s[s.split == 'train']
    por_paciente = tr.drop_duplicates('patient_id')[['patient_id', 'fold', 'pCR']]
    folds = []
    for k in sorted(por_paciente.fold.unique()):
        c, pa = tr[tr.fold == k], por_paciente[por_paciente.fold == k]
        pc = pa.merge(p, left_on='patient_id', right_on='pid')
        por_cohorte = {d: dict(pacientes=int(len(g)), pcr=int(g.pCR.sum())) for d, g in pc.groupby('dataset')}
        folds.append(dict(fold=int(k), pacientes=int(len(pa)), pcr=int(pa.pCR.sum()), cortes=int(len(c)), cortes_pcr=int(c.pCR.sum()),
                          cohortes=por_cohorte))
    pacientes = por_paciente.merge(p, left_on='patient_id', right_on='pid')
    te = s[s.split == 'test']
    test_pac = te.drop_duplicates('patient_id').merge(p, left_on='patient_id', right_on='pid')
    cohortes = {}
    for d in sorted(p.dataset.unique()):
        t = pacientes[pacientes.dataset == d]
        cohortes[d] = dict(train=int(len(t)), pcr_train=float(t.pCR.mean()), test=int((test_pac.dataset == d).sum()))
    return dict(folds=folds, test=dict(pacientes=int(test_pac.shape[0]), cortes=int(len(te))), cohortes=cohortes)


def metricas_macro(c):
    """Precisión, recall y F1 de la clase pCR y su media con la clase sin pCR (macro), desde la matriz de confusión."""
    tn, fp, fn, tp = c['tn'], c['fp'], c['fn'], c['tp']
    f1 = lambda p, r: 2*p*r/(p + r) if p + r else 0.0
    p1, r1 = tp/(tp + fp), tp/(tp + fn)          # clase pCR
    p0, r0 = tn/(tn + fn), tn/(tn + fp)          # clase sin pCR
    return dict(precision_macro=(p1 + p0)/2, recall_macro=(r1 + r0)/2, f1_macro=(f1(p1, r1) + f1(p0, r0))/2,
                recall_pcr=r1, f1_pcr=f1(p1, r1), precision_pcr=p1)


def _test_e57():
    t = json.loads((ROOT/'reports/final_e57_test_v1/test_evaluation.json').read_text())
    m, ic = t['patient_metrics_primary'], t['bootstrap']['intervals']['roc_auc']
    c = m['confusion_matrix']
    return dict(auc=m['roc_auc'], ic95=[ic['lower'], ic['upper']], ap=m['pr_auc'],
                sensibilidad=m['sensitivity'], especificidad=m['specificity'], precision=m['precision'],
                balanced_accuracy=m['balanced_accuracy'], matriz=c, pacientes=c['tn'] + c['fp'] + c['fn'] + c['tp'],
                pcr=c['fn'] + c['tp'], umbral=t['threshold'], **metricas_macro(c))


def _picos_e57():
    picos = [pd.read_csv(ROOT/'reports/patient_features/E57_patient_feature_mean_mac/fold_0_seed_42/history.csv').patient_roc_auc.max()]
    for k in range(1, 5):
        picos.append(pd.read_csv(ROOT/f'experiments/E57_patient_feature_mean_mac/stability/resultados/fold_{k}_seed_42/history.csv').patient_roc_auc.max())
    return dict(fold0=float(picos[0]), media=float(sum(picos) / len(picos)))


def matriz_conjunta(runs=RUNS, prefijo='normal', folds=5, semilla=42):
    """Matriz de confusión de los 5 folds juntos: cada paciente, con el modelo del fold en que fue
    exterior y el umbral interno de ese fold (summary.json). No usa el test."""
    tn = fp = fn = tp = 0
    ids = set()
    for k in range(folds):
        d = Path(runs)/f'{prefijo}_fold_{k}_seed_{semilla}'
        umbral = json.loads((d/'summary.json').read_text())['threshold']
        p = pd.read_csv(d/'predictions_outer_patient.csv')
        ids |= set(p.patient_id)
        pred, y = p.probability >= umbral, p.target == 1
        tp += int((pred & y).sum()); fp += int((pred & ~y).sum())
        fn += int((~pred & y).sum()); tn += int((~pred & ~y).sum())
    n = tn + fp + fn + tp
    assert n == len(ids), 'una paciente aparece en más de un fold exterior'
    sens, esp = tp/(tp + fn), tn/(tn + fp)
    return dict(tn=tn, fp=fp, fn=fn, tp=tp, pacientes=n, pcr=tp + fn, sensibilidad=sens, especificidad=esp,
                balanced_accuracy=(sens + esp)/2, precision=tp/(tp + fp) if tp + fp else None, folds=folds)


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
    if 'normal' in salida:
        salida['normal']['matriz_conjunta'] = matriz_conjunta()
    if 'pareada' in s and len(salida) == 2:
        salida['pareada'] = s['pareada']
    return salida


def cv_n01(base=N01):
    """Validación cruzada de la receta de E57 (N01, BCE normal) en los 5 folds: cada paciente predicha por el
    modelo del fold en que quedó fuera. Lee el resumen y las predicciones exteriores guardadas; no usa el test."""
    base = Path(base)
    if not (base/'cv/resumen.json').is_file():
        return {}
    s = json.loads((base/'cv/resumen.json').read_text())
    p = pd.read_csv(base/'cv/predicciones_exteriores.csv')
    return dict(
        auc=s['auc_conjunta']['n01'], ic95=cv_report.bootstrap_auc(p.target, p.n01), auc_corte=s['auc_por_corte']['n01'],
        pacientes=int(len(p)), cohorte_base=s['auc_conjunta']['cohorte_base'],
        menos_cohorte=s['n01_menos_cohorte'], menos_p02=s['n01_menos_p02'],
        folds=[dict(fold=f['fold'], epoca=f['epoca'], exterior=f['auc_exterior'], interna=f['auc_interna'], fit=f['auc_fit'])
               for f in s['por_fold']['n01']],
        cohortes={c: dict(n=int(len(g)), pcr=int(g.target.sum()), auc=float(roc_auc_score(g.target, g.n01)))
                  for c, g in p.groupby('cohorte')},
        matriz_conjunta=matriz_conjunta(base/'runs', folds=len(s['por_fold']['n01'])),
        perdidas_fold0=perdidas_fold0(base/'runs'))


def perdidas_fold0(runs):
    """BCE normal frente a ponderada con la receta de E57 en el fold 0, el único en que se entrenaron las dos:
    AUC exterior y sensibilidad con el umbral 0,5 y con el umbral elegido en la validación interna."""
    rutas = {loss: Path(runs)/f'{loss}_fold_0_seed_42/summary.json' for loss in cv_report.LOSSES}
    if not all(r.is_file() for r in rutas.values()):
        return {}
    salida = {}
    for loss, path in rutas.items():
        s = json.loads(path.read_text())
        a, b = s['outer_at_inner_threshold'], s['outer_at_05']['metrics']
        salida[loss] = dict(pos_weight=s['pos_weight'], epoca=s['best_epoch'], interna=s['best_inner_auc'],
                            exterior=a['roc_auc'], umbral=s['threshold'], brier=a['brier_score'],
                            sensibilidad=a['sensitivity'], especificidad=a['specificity'],
                            sensibilidad_05=b['sensitivity'], especificidad_05=b['specificity'])
    return salida


def build():
    DEST.mkdir(parents=True, exist_ok=True)
    datos = dict(actualizado=date.today().isoformat(), test_e57=_test_e57(), picos_e57=_picos_e57(), cv_n01=cv_n01(), cv_p02=_cv(),
                 experimentos=_experimentos(), particiones=_particiones(),
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
