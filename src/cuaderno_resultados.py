"""Notebook único de resultados de la etapa: nuevo_enfoque/RESULTADOS.ipynb.

Mismo formato que E57/RESULTADOS.ipynb: para cada experimento, una explicación, y para
cada ejecución, la tabla de todas las épocas, las tres gráficas de curvas y el panel de
resultados. Las celdas solo llaman a funciones de este módulo, que leen salidas ya
guardadas (summary.json, history.csv, predicciones). No entrena ni lee test.

    python -m src.cuaderno_resultados

Al llegar resultados nuevos (los 5 folds de la GPU, la decisión o el test), basta con
volver a ejecutarlo: las secciones pendientes se rellenan solas.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import nbformat
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

from src import cv_report

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT/'nuevo_enfoque/RESULTADOS.ipynb'
NA = ROOT/'reports/new_approach'
PRIOR_ENTROPY = 0.6053          # BCE del predictor constante con 29,3 % pCR
PRIOR_WEIGHTED = 0.9795         # óptimo constante de la BCE ponderada (pos_weight ≈ 2,41)
TRAIN, VALID, OUTER, GRIS = cv_report.TRAIN, cv_report.VALID, cv_report.OUTER, '#9aa0a6'
PERDIDA = {'normal': 'normal', 'weighted': 'ponderada'}

N01 = NA/'N01_e57_bce/runs'
P01 = NA/'P01_lotes_pacientes/runs'
P02 = NA/'P02_aumento_geometrico/runs'
P02_GPU = NA/'P02_aumento_geometrico/runs_cuda'
RECETAS = [  # (experimento, carpeta, plataforma, qué cambia)
    ('N01', N01, 'Mac', 'Receta E57: 2 pacientes por lote'),
    ('P01', P01, 'Mac', '10 pacientes × 2 cortes por lote'),
    ('P02', P02, 'Mac', 'P01 + aumento geométrico sincronizado'),
    ('P02', P02_GPU, 'GPU', 'P01 + aumento geométrico sincronizado'),
]


def _coma(v, d=3):
    return '—' if v is None or (isinstance(v, float) and np.isnan(v)) else f'{v:.{d}f}'.replace('.', ',')


def _tabla(df, decimales=3, resaltar=None):
    cols = [c for c in df.select_dtypes('number').columns if not c.startswith(('Época', 'Fold', 'TN', 'FP', 'FN', 'TP'))]
    st = (df.style.format({c: (lambda v, d=decimales: _coma(v, d)) for c in cols})
          .hide(axis='index').set_table_styles([{'selector': 'th', 'props': 'text-align:left'}]))
    if resaltar is not None:
        st = st.apply(lambda fila: ['background-color:#fde7f1; font-weight:600' if resaltar(fila) else '' for _ in fila], axis=1)
    return st


def _run(carpeta, loss, fold):
    return cv_report.load_runs(carpeta)[(loss, fold)]


# ---------------------------------------------------------------- resumen y referencias

def tabla_resumen():
    """Una fila por ejecución completa de la etapa, con las cifras que importan."""
    filas = []
    for nombre, carpeta, plataforma, cambio in RECETAS:
        for (loss, fold), r in sorted(cv_report.load_runs(carpeta).items()):
            s = r['summary']
            filas.append({
                'Experimento': nombre, 'Qué cambia': cambio, 'Pérdida': PERDIDA[loss], 'Fold': fold,
                'Plataforma': plataforma, 'Época elegida': f"{s['best_epoch']} de {s['epochs_completed']}",
                'AUC train (FIT)': s['fit_at_best']['metrics']['roc_auc'],
                'AUC validación interna': s['inner_at_best']['metrics']['roc_auc'],
                'AUC exterior': s['outer_at_05']['metrics']['roc_auc'],
                'AUC exterior por corte': s['outer_at_05']['cut_metrics_secondary']['roc_auc'],
            })
    return _tabla(pd.DataFrame(filas))


def tabla_e57():
    """Las cifras de E57 que se citan, separando las optimistas de las honestas."""
    picos = [pd.read_csv(ROOT/'reports/patient_features/E57_patient_feature_mean_mac/fold_0_seed_42/history.csv').patient_roc_auc.max()]
    for k in range(1, 5):
        picos.append(pd.read_csv(ROOT/f'experimentos_equilibrados/E57_estabilidad/resultados/fold_{k}_seed_42/history.csv').patient_roc_auc.max())
    test = json.loads((ROOT/'reports/final_e57_test_v1/test_evaluation.json').read_text())
    ic = test['bootstrap']['intervals']['roc_auc']
    filas = [
        {'Medida': 'Pico de validación, fold 0 (época 14 de 24)', 'AUC': picos[0], 'Tipo': 'optimista: época elegida en la misma validación'},
        {'Medida': 'Media de los picos de 5 folds', 'AUC': float(np.mean(picos)), 'Tipo': 'optimista'},
        {'Medida': 'Validación anidada de 5 folds (otra receta, 15/10)', 'AUC': 0.527152, 'Tipo': 'honesta'},
        {'Medida': f"Test público, una evaluación (IC95 {_coma(ic['lower'])}–{_coma(ic['upper'])})",
         'AUC': test['patient_metrics_primary']['roc_auc'], 'Tipo': 'honesta'},
    ]
    return _tabla(pd.DataFrame(filas))


# ---------------------------------------------------------------- diagnóstico

def fig_colapso():
    """La pérdida de entrenamiento de E57 y N01 no baja de la del predictor constante."""
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.6))
    rutas = [ROOT/'reports/patient_features/E57_patient_feature_mean_mac/fold_0_seed_42/history.csv'] + [
        ROOT/f'experimentos_equilibrados/E57_estabilidad/resultados/fold_{k}_seed_42/history.csv' for k in range(1, 5)]
    for k, ruta in enumerate(rutas):
        h = pd.read_csv(ruta)
        ax[0].plot(h.epoch, h.train_loss, color=TRAIN, alpha=.35 + .13 * k, lw=1.6, label=f'fold {k}')
    ax[0].axhline(PRIOR_ENTROPY, color=VALID, ls='--', lw=1.5, label='predictor constante (0,605)')
    ax[0].set(title='E57, 5 folds: pérdida de entrenamiento', xlabel='Época', ylabel='BCE de entrenamiento', ylim=(0.55, 0.65))
    ax[0].legend(fontsize=8, ncol=2)
    for loss, color, constante in (('normal', TRAIN, PRIOR_ENTROPY), ('weighted', '#e87ba4', PRIOR_WEIGHTED)):
        h = pd.read_csv(N01/f'{loss}_fold_0_seed_42/history.csv')
        ax[1].plot(h.epoch, h.train_online_objective, color=color, lw=2, label=f'BCE {PERDIDA[loss]}')
        ax[1].axhline(constante, color=color, ls='--', lw=1.2, label=f'constante {PERDIDA[loss]} ({_coma(constante)})')
    ax[1].set(title='N01, fold 0: la pérdida que se optimiza', xlabel='Época', ylabel='Pérdida de entrenamiento')
    for a in ax:
        a.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax[1].legend(fontsize=8)
    fig.suptitle('La red apenas bajaba de la pérdida de un predictor que siempre responde la prevalencia',
                 fontweight='bold', x=0.01, ha='left')
    fig.tight_layout()
    cv_report.comas(fig)
    return fig


def fig_memorizacion():
    base = NA/'P01_lotes_pacientes/overfit'
    etiqueta = {'e57': 'E57: 2 pacientes × 10 cortes', 'p01': 'P01: 10 pacientes × 2 cortes'}
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.6))
    for receta, color in (('e57', GRIS), ('p01', TRAIN)):
        h = pd.read_csv(base/receta/'history.csv')
        ax[0].plot(h.epoch, h.train_online_objective, color=color, lw=2, label=etiqueta[receta])
        ax[1].plot(h.epoch, h.eval_auc, color=color, lw=2, marker='o', ms=3, label=etiqueta[receta])
    ax[0].axhline(PRIOR_ENTROPY, color=VALID, ls='--', lw=1.3, label='predictor constante')
    ax[1].axhline(0.9, color=VALID, ls='--', lw=1.3, label='umbral de memorización (0,90)')
    ax[0].set(title='Pérdida de entrenamiento', xlabel='Época', ylabel='BCE')
    ax[1].set(title='AUC en modo evaluación sobre las mismas 40 pacientes', xlabel='Época', ylabel='ROC-AUC', ylim=(0.5, 1.02))
    for a in ax:
        a.legend(fontsize=9)
    fig.suptitle('Con 10 pacientes por lote la red memoriza en 3 épocas; con 2 necesita 23', fontweight='bold', x=0.01, ha='left')
    fig.tight_layout()
    cv_report.comas(fig)
    return fig


def tabla_memorizacion():
    base = NA/'P01_lotes_pacientes/overfit'
    filas = []
    for receta, nombre in (('e57', 'E57 · 2 pacientes × 10 cortes'), ('p01', 'P01 · 10 pacientes × 2 cortes')):
        h = pd.read_csv(base/receta/'history.csv')
        primera = h.loc[h.eval_auc >= 0.9, 'epoch']
        filas.append({'Receta': nombre, 'AUC en la época 4': h.loc[h.epoch == 4, 'eval_auc'].item(),
                      'Primera época con AUC ≥ 0,90': str(int(primera.min())) if len(primera) else 'nunca',
                      'AUC en la época 30': h.eval_auc.iloc[-1],
                      'Pérdida media, 5 últimas épocas': h.train_online_objective.iloc[-5:].mean()})
    return _tabla(pd.DataFrame(filas))


# ---------------------------------------------------------------- una ejecución

COLUMNAS_EPOCA = [
    ('epoch', 'Época'), ('train_online_objective', 'Pérdida train (online)'), ('fit_bce', 'BCE train (eval)'),
    ('inner_bce', 'BCE interna'), ('fit_auc', 'AUC train (eval)'), ('inner_auc', 'AUC interna'),
    ('fit_cut_auc', 'AUC train por corte'), ('inner_cut_auc', 'AUC interna por corte'),
    ('learning_rate', 'LR'), ('seconds', 'Segundos'),
]


def tabla_epocas(carpeta, loss, fold):
    """Todas las épocas; la fila resaltada es la elegida con la validación interna."""
    r = _run(carpeta, loss, fold)
    h = r['history']
    df = pd.DataFrame({nuevo: h[viejo] for viejo, nuevo in COLUMNAS_EPOCA if viejo in h.columns})
    df['Segundos'] = df['Segundos'].round(0)
    best = r['summary']['best_epoch']
    return _tabla(df, resaltar=lambda fila: fila['Época'] == best).format({'LR': lambda v: f'{v:g}'.replace('.', ','),
                                                                         'Segundos': lambda v: f'{v:.0f}'})


def fig_tres(carpeta, loss, fold, titulo):
    runs = {k: v for k, v in cv_report.load_runs(carpeta).items() if k == (loss, fold)}
    return cv_report.fig_curves(runs, loss, None, titulo)


def fig_resultados(carpeta, loss, fold, titulo):
    """Panel de seis gráficas sobre las pacientes exteriores, como el de E57."""
    r = _run(carpeta, loss, fold)
    y, p = r['patients'].target.to_numpy(), r['patients'].probability.to_numpy()
    umbral = r['decision']['threshold']
    c = cv_report.confusion(y, p, umbral)
    fig, ax = plt.subplots(2, 3, figsize=(15, 8.4))
    fpr, tpr, _ = roc_curve(y, p)
    fpr_r, tpr_r, rec_r, prec_r = referencia(prevalencia=y.mean())
    ax[0, 0].plot(fpr_r, tpr_r, ':', color=VERDE, lw=2, label=f'buen modelo teórico ({_coma(AUC_REFERENCIA, 2)})')
    ax[0, 0].plot(fpr, tpr, color=TRAIN, lw=2, label='este modelo'); ax[0, 0].plot([0, 1], [0, 1], '--', color=GRIS)
    ax[0, 0].legend(fontsize=8, loc='lower right')
    ax[0, 0].set(title=f'ROC · AUC {_coma(roc_auc_score(y, p), 4)}', xlabel='1 − especificidad', ylabel='Sensibilidad')
    pr, rc, _ = precision_recall_curve(y, p)
    ax[0, 1].plot(rec_r, prec_r, ':', color=VERDE, lw=2, label='buen modelo teórico')
    ax[0, 1].plot(rc, pr, color=TRAIN, lw=2, label='este modelo'); ax[0, 1].axhline(y.mean(), ls=':', color=GRIS, label='azar (prevalencia)')
    ax[0, 1].legend(fontsize=8, loc='upper right')
    ax[0, 1].set(title=f'Precisión–recall · AP {_coma(average_precision_score(y, p), 4)}', xlabel='Sensibilidad', ylabel='Precisión', ylim=(0, 1.02))
    m = np.array([[c['tn'], c['fp']], [c['fn'], c['tp']]])
    ax[0, 2].imshow(m / m.sum(1, keepdims=True), cmap='Blues', vmin=0, vmax=1)
    for (i, j), v in np.ndenumerate(m):
        ax[0, 2].text(j, i, v, ha='center', va='center', fontsize=20, color='white' if m[i, j] / m[i].sum() > .55 else 'black')
    ax[0, 2].set(title=f'Matriz · umbral interno {_coma(umbral)}', xticks=[0, 1], yticks=[0, 1],
                 xticklabels=['no pCR', 'pCR'], yticklabels=['no pCR', 'pCR'], xlabel='Predicción', ylabel='Real')
    ax[0, 2].grid(False)
    q = pd.qcut(pd.Series(p), 8, duplicates='drop')
    g = pd.DataFrame({'p': p, 'y': y}).groupby(q, observed=True).mean()
    ax[1, 0].plot(g.p, g.y, 'o-', color=TRAIN, lw=2); ax[1, 0].plot([0, 1], [0, 1], '--', color=GRIS)
    ax[1, 0].set(title=f'Calibración · Brier {_coma(np.mean((p - y) ** 2), 4)}', xlabel='Probabilidad media', ylabel='Proporción pCR', xlim=(0, 1), ylim=(0, 1))
    bins = np.linspace(0, max(0.6, p.max() + 0.02), 26)
    ax[1, 1].hist(p[y == 0], bins=bins, density=True, alpha=.55, color=TRAIN, label='Real no pCR')
    ax[1, 1].hist(p[y == 1], bins=bins, density=True, alpha=.55, color=VALID, label='Real pCR')
    ax[1, 1].axvline(umbral, color='#333', ls='--', label=f'Umbral {_coma(umbral)}')
    ax[1, 1].set(title='Probabilidades por clase', xlabel='Probabilidad predicha', ylabel='Densidad'); ax[1, 1].legend(fontsize=8)
    nombres = ['Precisión', 'Sensib.', 'Especif.', 'BA', 'F1 macro']
    valores = [c['precision'], c['sensitivity'], c['specificity'], c['balanced_accuracy'], c['f1_macro']]
    ax[1, 2].bar(nombres, valores, color=[TRAIN, VALID, TRAIN, VALID, TRAIN])
    for i, v in enumerate(valores):
        ax[1, 2].text(i, v + .02, _coma(v, 2), ha='center', fontsize=9)
    ax[1, 2].set(title=f'Decisión con umbral {_coma(umbral)}', ylim=(0, 1.05))
    fig.suptitle(f"{titulo} · BCE {PERDIDA[loss]} · fold {fold} · {len(y)} pacientes exteriores\n"
                 f"Época {r['summary']['best_epoch']} y umbral elegidos con la validación interna, antes de mirar estas pacientes",
                 fontsize=12)
    fig.tight_layout()
    return cv_report.comas(fig)


AUC_REFERENCIA = cv_report.AUC_REFERENCIA
VERDE = cv_report.VERDE


def referencia(auc=AUC_REFERENCIA, prevalencia=0.293):
    fpr, tpr, rec, prec, _ = cv_report.referencia(auc, prevalencia)
    return fpr, tpr, rec, prec


def fig_roc_dos(carpeta, loss, fold, titulo):
    """La ROC de la validación interna (la que eligió la época) junto a la del exterior, cada una con su etiqueta."""
    r = _run(carpeta, loss, fold)
    interna = pd.read_csv(r['folder']/'predictions_inner_patient.csv')
    fig, ax = plt.subplots(figsize=(6.6, 6.2))
    for datos, color, nombre in ((interna, VALID, 'Validación interna (eligió la época; optimista)'),
                                 (r['patients'], OUTER, 'Exterior (pacientes no vistas)')):
        y, p = datos.target.to_numpy(), datos.probability.to_numpy()
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr, tpr, color=color, lw=2.4, label=f'{nombre}: AUC {_coma(roc_auc_score(y, p))} (n={len(y)})')
    fpr_r, tpr_r, _, _ = referencia()
    ax.plot(fpr_r, tpr_r, ':', color=VERDE, lw=2.2, label=f'Referencia teórica de un buen modelo: AUC {_coma(AUC_REFERENCIA, 2)}')
    ax.plot([0, 1], [0, 1], '--', color=GRIS, lw=1, label='Azar (0,5)')
    ax.set(xlabel='1 − especificidad', ylabel='Sensibilidad', xlim=(0, 1), ylim=(0, 1.01),
           title=f"{titulo} · BCE {PERDIDA[loss]} · fold {fold} · época {r['summary']['best_epoch']}")
    ax.legend(loc='lower right', fontsize=9)
    fig.tight_layout()
    return cv_report.comas(fig)


def fig_pr_dos(carpeta, loss, fold, titulo):
    """Precisión–recall de la validación interna junto a la del exterior; el azar es la prevalencia, no 0,5."""
    r = _run(carpeta, loss, fold)
    interna = pd.read_csv(r['folder']/'predictions_inner_patient.csv')
    fig, ax = plt.subplots(figsize=(6.6, 6.2))
    for datos, color, nombre in ((interna, VALID, 'Validación interna (eligió la época; optimista)'),
                                 (r['patients'], OUTER, 'Exterior (pacientes no vistas)')):
        y, p = datos.target.to_numpy(), datos.probability.to_numpy()
        prec, rec, _ = precision_recall_curve(y, p)
        ax.plot(rec, prec, color=color, lw=2.4, label=f'{nombre}: AP {_coma(average_precision_score(y, p))}')
    prev = r['patients'].target.mean()
    _, _, rec_r, prec_r = referencia(prevalencia=prev)
    ap_r = float(np.sum(np.diff(np.r_[0, rec_r]) * prec_r))  # la sensibilidad crece a lo largo de la curva
    ax.plot(rec_r, prec_r, ':', color=VERDE, lw=2.2,
            label=f'Referencia teórica (modelo con AUC {_coma(AUC_REFERENCIA, 2)}): AP {_coma(ap_r)}')
    ax.axhline(prev, ls='--', color=GRIS, lw=1, label=f'Azar = proporción de pCR ({_coma(prev)})')
    ax.set(xlabel='Sensibilidad (recall)', ylabel='Precisión', xlim=(0, 1), ylim=(0, 1.01),
           title=f"{titulo} · BCE {PERDIDA[loss]} · fold {fold} · época {r['summary']['best_epoch']}")
    ax.legend(loc='upper right', fontsize=9)
    fig.tight_layout()
    return cv_report.comas(fig)


def tabla_umbrales(carpeta, loss, fold):
    """La misma predicción exterior con tres umbrales: el interno, el de E57 (0,325) y 0,5."""
    r = _run(carpeta, loss, fold)
    y, p = r['patients'].target.to_numpy(), r['patients'].probability.to_numpy()
    filas = []
    for nombre, u in (('interno (elegido antes)', r['decision']['threshold']), ('0,325 (el de E57)', 0.325), ('0,5', 0.5)):
        c = cv_report.confusion(y, p, u)
        filas.append({'Umbral': f'{nombre}: {_coma(u)}', 'TN': c['tn'], 'FP': c['fp'], 'FN': c['fn'], 'TP': c['tp'],
                      'Sensibilidad': c['sensitivity'], 'Especificidad': c['specificity'],
                      'Precisión pCR': c['precision'], 'Balanced accuracy': c['balanced_accuracy']})
    return _tabla(pd.DataFrame(filas))


# ---------------------------------------------------------------- validación cruzada de P02

def _cohortes():
    meta = pd.read_csv(ROOT/'breastdcedl/metadata/patients.csv', usecols=['pid', 'dataset'])
    return dict(zip(meta.pid, meta.dataset))


def _runs_completos(carpeta=P02_GPU):
    """Solo las pérdidas con sus 5 folds: una pérdida a medias no se mezcla en los resultados conjuntos."""
    runs = cv_report.load_runs(carpeta)
    llenas = {l for l in cv_report.LOSSES if sum(k[0] == l for k in runs) == 5}
    return {k: v for k, v in runs.items() if k[0] in llenas}


def cv_disponible(carpeta=P02_GPU):
    runs = cv_report.load_runs(carpeta) if Path(carpeta).is_dir() else {}
    return {l: sorted(f for (ll, f) in runs if ll == l) for l in cv_report.LOSSES}


def cv_tabla_folds(carpeta=P02_GPU):
    """Resumen por fold, como la tabla de E57, con la matriz del umbral interno."""
    filas = []
    for (loss, fold), r in sorted(_runs_completos(carpeta).items()):
        s, m = r['summary'], r['summary']['outer_at_inner_threshold']
        c = m['confusion_matrix']
        filas.append({'Pérdida': PERDIDA[loss], 'Fold': fold, 'Épocas': s['epochs_completed'], 'Época elegida': s['best_epoch'],
                      'AUC train': s['fit_at_best']['metrics']['roc_auc'], 'AUC interna': s['inner_at_best']['metrics']['roc_auc'],
                      'AUC exterior': s['outer_at_05']['metrics']['roc_auc'], 'AP exterior': s['outer_at_05']['metrics']['pr_auc'],
                      'Sensibilidad': m['sensitivity'], 'Especificidad': m['specificity'], 'BA': m['balanced_accuracy'],
                      'TN': c['tn'], 'FP': c['fp'], 'FN': c['fn'], 'TP': c['tp']})
    return _tabla(pd.DataFrame(filas))


def cv_resumen(carpeta=P02_GPU):
    """AUC conjunta con IC95 y media de folds, por pérdida, y la comparación pareada."""
    s = cv_report.summarize(_runs_completos(carpeta), _cohortes())
    filas = []
    for loss in cv_report.LOSSES:
        if loss in s:
            x = s[loss]
            filas.append({'Pérdida': PERDIDA[loss], 'Folds': str(x['n_folds']), 'Pacientes': str(x['n_pacientes']),
                          'AUC exterior conjunta': x['auc_conjunta'], 'IC95 inferior': x['auc_conjunta_ic95'][0],
                          'IC95 superior': x['auc_conjunta_ic95'][1], 'Media de folds': x['auc_media_folds'],
                          'AUC por corte': x['auc_corte_conjunta'], 'AP': x['ap_conjunta'], 'Brier': x['brier_conjunto']})
    texto = ''
    if 'pareada' in s:
        d = s['pareada']
        incluye = d['ic95'][0] <= 0 <= d['ic95'][1]
        texto = (f"**Normal − ponderada** sobre las mismas {d['n']} pacientes: {_coma(d['diferencia_auc_normal_menos_ponderada'])} "
                 f"(IC95 {_coma(d['ic95'][0])} a {_coma(d['ic95'][1])}). "
                 + ('El intervalo incluye 0: **no hay diferencia demostrada** entre las dos pérdidas.' if incluye
                    else 'El intervalo excluye 0: la diferencia es clara.'))
    return _tabla(pd.DataFrame(filas)), texto


def cv_figuras(carpeta=P02_GPU):
    runs = _runs_completos(carpeta)
    cohortes = _cohortes()
    s = cv_report.summarize(runs, cohortes)
    pats = {l: cv_report.pooled(runs, l, cohortes)[0] for l in cv_report.LOSSES if l in s}
    figs = [cv_report.fig_curves(runs, l, None, 'P02 (GPU)') for l in cv_report.LOSSES if l in s]
    return figs + [cv_report.fig_matrices(s, None), cv_report.fig_roc_pr(pats, s, None),
                   cv_report.fig_cohorts(s, None), cv_report.fig_calibration(pats, s, None)]


# ---------------------------------------------------------------- entrega

def entrega():
    d_path, t_path = NA/'ENTREGA/decision.json', NA/'ENTREGA/test/test_evaluation.json'
    if not d_path.is_file():
        return ('**Pendiente.** Se decidirá con `python -m src.final_ensemble freeze --variant p02 --gpu` en cuanto estén '
                'los 10 entrenamientos de la GPU, aplicando las reglas tal como se fijaron.')
    d = json.loads(d_path.read_text())
    texto = f"**Entregable:** {d['entregable']}.\n\n**Motivo:** {d['motivo']}."
    if 'threshold' in d:
        texto += f"\n\n**Umbral:** {_coma(d['threshold'])} (mediana de los 5 umbrales internos)."
    if t_path.is_file():
        t = json.loads(t_path.read_text())
        m, ic = t['patient_metrics_primary'], t['bootstrap']['intervals']['roc_auc']
        c = m['confusion_matrix']
        texto += (f"\n\n**Test público (una sola evaluación):** AUC {_coma(m['roc_auc'])} (IC95 {_coma(ic['lower'])}–{_coma(ic['upper'])}); "
                  f"TN {c['tn']}, FP {c['fp']}, FN {c['fn']}, TP {c['tp']}; sensibilidad {_coma(m['sensitivity'])}, "
                  f"especificidad {_coma(m['specificity'])}.")
    else:
        texto += '\n\nEl test todavía no se ha evaluado; se hará una sola vez, con el visto bueno del estudiante.'
    return texto


# ---------------------------------------------------------------- después de P02

def _leer_resumen(carpeta):
    ruta = NA/carpeta/'resumen.json'
    return json.loads(ruta.read_text()) if ruta.is_file() else None


def _ic(d):
    return f"{d['diferencia']:+.3f} [{d['ic95'][0]:+.3f}, {d['ic95'][1]:+.3f}]".replace('.', ',')


def tabla_hipotesis():
    """Una fila por hipótesis posterior a P02, con su criterio fijado antes y el resultado."""
    filas = []
    p03 = _leer_resumen('P03_paso0')
    if p03:
        c = p03['C_cohorte']
        filas.append(('Línea base de cohorte', 'Cuánto de P02 es saber de qué cohorte viene la paciente',
                      f"cohorte {_coma(c['auc_linea_base_cohorte'])}; P02 − cohorte "
                      + _ic({'diferencia': c['diferencia_p02_menos_cohorte'], 'ic95': c['ic95_pareado']}),
                      'Referencia', 'P03_PASO0.md'))
        filas.append(('P03 · recorte central', 'Realce ≥ 60 % dentro del recorte en las 3 cohortes y oclusión de E57 cargada en la periferia',
                      f"solo cumple 192 px; razón periférica {_coma(p03['B_oclusion']['mediana_razon_periferia'], 2)}",
                      'Descartado', 'P03_PASO0.md'))
    p04 = _leer_resumen('P04_paso0b')
    if p04:
        filas.append(('P04 · realce relativo (PE, SER)', 'IC95 de cohorte + relativas − cohorte > 0',
                      _ic(p04['cohorte_relativas_menos_cohorte']), 'Descartado', 'P04_PASO0B.md'))
    p06 = _leer_resumen('P06_paso0c')
    if p06:
        filas.append(('P06 · región que más realza', 'IC95 de cohorte + región − cohorte > 0',
                      _ic(p06['cohorte_region_menos_cohorte']), 'Descartado', 'P06_PASO0C.md'))
    filas.append(('P05 · tasa de aprendizaje 0,0001', 'IC95 de P05 − P02 > 0',
                  'no se ejecutó: ya probado en E12 y E55', 'Retirado', 'P05_LR_BAJO.md'))
    p07 = _leer_resumen('P07_semillas')
    if p07:
        filas.append(('P07 · ensemble de 3 semillas', 'IC95 de ensemble − semilla 42 > 0',
                      f"ensemble {_coma(p07['auc_conjunta']['ensemble'])}; " + _ic(p07['ensemble_menos_semilla_42']),
                      'Cumple' if p07['criterios']['principal_ensemble_mejora_semilla_42'] else 'Descartado',
                      'P07_SEMILLAS.md'))
    return pd.DataFrame(filas, columns=['Hipótesis', 'Criterio fijado antes', 'Resultado', 'Decisión', 'Documento'])


def tabla_p07():
    """AUC exterior conjunta de cada semilla, del ensemble y de la cohorte."""
    p07 = _leer_resumen('P07_semillas')
    if not p07:
        return None
    a = p07['auc_conjunta']
    nombres = {'ensemble': 'Ensemble (media de 3)', 'semilla_42': 'Semilla 42 (P02, GPU)', 'semilla_17': 'Semilla 17 (Mac)',
               'semilla_123': 'Semilla 123 (Mac)', 'cohorte_base': 'Línea base de cohorte'}
    t = pd.DataFrame([(nombres[k], v) for k, v in a.items()], columns=['Modelo', 'AUC exterior conjunta (1.097)'])
    return _tabla(t)


def n01_cv():
    """Validación cruzada de la receta de E57 (N01, BCE normal), si ya se analizó."""
    hechos = sorted(f for (l, f) in cv_report.load_runs(N01) if l == 'normal')
    r = _leer_resumen('N01_e57_bce/cv')
    if len(hechos) < 5:
        return (f'**Pendiente.** Folds de N01 normal completos: {len(hechos)}/5 ({", ".join(map(str, hechos))}). '
                'Al terminar: `python -m src.analisis_n01_cv` y después regenerar este notebook.')
    if not r:
        return '**Pendiente.** Están los 5 folds: falta `python -m src.analisis_n01_cv`.'
    a, f = r['auc_conjunta'], r['por_fold']['n01']
    folds = ', '.join(f"fold {x['fold']}: {_coma(x['auc_exterior'])}" for x in f)
    return (f"**AUC exterior conjunta de N01 (1.097 pacientes):** {_coma(a['n01'])}; por corte {_coma(r['auc_por_corte']['n01'])}.\n\n"
            f"**Por fold:** {folds}.\n\n"
            f"**Frente a la línea base de cohorte ({_coma(a['cohorte_base'])}):** {_ic(r['n01_menos_cohorte'])}. "
            f"**Frente a P02 normal ({_coma(a['p02'])}):** {_ic(r['n01_menos_p02'])}.\n\n"
            f"**¿Aprende señal más allá de la cohorte?** {'Sí' if r['criterios']['aprende_mas_alla_de_la_cohorte'] else 'No'} "
            "(criterio: IC95 frente a la cohorte por encima de 0).\n\n"
            f"Dentro de cada cohorte: " + ', '.join(f"{k} {_coma(v['n01'])}" for k, v in r['dentro_de_cohorte'].items()) + '.')

# ---------------------------------------------------------------- construcción del notebook

def _md(texto):
    return nbformat.v4.new_markdown_cell(texto.strip())


def _code(texto):
    return nbformat.v4.new_code_cell(texto.strip())


def _secciones_runs(carpeta, nombre_carpeta, titulo, nivel='###'):
    """Una sección por ejecución: tabla de épocas, tres gráficas, panel y umbrales."""
    celdas = []
    for (loss, fold), r in sorted(cv_report.load_runs(carpeta).items()):
        s = r['summary']
        celdas += [
            _md(f"{nivel} {titulo} · BCE {PERDIDA[loss]} · fold {fold} · {s['epochs_completed']} épocas · checkpoint de época {s['best_epoch']}\n\n"
                f"**Todas las épocas.** La fila resaltada es la que eligió la validación interna."),
            _code(f"R.tabla_epocas(R.{nombre_carpeta}, '{loss}', {fold})"),
            _md('**Curvas.** Train (azul) y validación interna (naranja) en el mismo modo de evaluación; '
                'a la derecha, el AUC de las tres particiones en la época elegida.'),
            _code(f"R.fig_tres(R.{nombre_carpeta}, '{loss}', {fold}, '{titulo}')"),
            _md('**Resultados sobre las pacientes exteriores** (las del fold, nunca vistas al entrenar ni al elegir).'),
            _code(f"R.fig_resultados(R.{nombre_carpeta}, '{loss}', {fold}, '{titulo}')"),
            _md('**Curva ROC de la validación interna frente a la del exterior.** La interna es la que eligió la época, así que es optimista; '
                'la del exterior mide lo que pasa con pacientes nuevas. La línea verde punteada es una referencia teórica de un buen modelo '
                '(AUC 0,80), para comparar a simple vista; no es ningún modelo entrenado.'),
            _code(f"R.fig_roc_dos(R.{nombre_carpeta}, '{loss}', {fold}, '{titulo}')"),
            _md('**Precisión–recall, interna frente a exterior.** Aquí el azar no es 0,5 sino la proporción de pacientes con pCR (≈ 0,29): '
                'un modelo que adivinara acertaría en esa proporción al marcar a alguien como pCR.'),
            _code(f"R.fig_pr_dos(R.{nombre_carpeta}, '{loss}', {fold}, '{titulo}')"),
            _code(f"R.tabla_umbrales(R.{nombre_carpeta}, '{loss}', {fold})"),
        ]
    return celdas


def celdas():
    disp = cv_disponible()
    llenas = [l for l, v in disp.items() if len(v) == 5]
    completa = len(llenas) == 2
    c = [
        _md("""
# Resultados de la etapa actual

Este notebook reúne todo lo probado desde el 20 de octubre: por qué el modelo de referencia (E57) no aprendía, cómo se corrigió y qué rendimiento tiene la nueva receta. Las cifras se leen de las salidas guardadas en `reports/`; no hace falta abrir ningún CSV ni JSON. Para actualizarlo con resultados nuevos: `python -m src.cuaderno_resultados`.

### Cómo se mide

Cada fold docente (unas 219 pacientes) hace de **exterior**. Con el resto se entrena: el 80 % es **FIT** (unas 702 pacientes, con las que aprende la red) y el 20 % es la **validación interna** (unas 176), que elige la época y el umbral. Después, y una sola vez, se evalúan las pacientes exteriores. La probabilidad de una paciente es la media de las de sus cortes.

**Qué AUC mirar.** La cifra principal es la **AUC exterior**: pacientes que el modelo no vio, con decisiones tomadas sin mirarlas. El pico de validación es optimista, porque se elige la mejor de muchas épocas sobre las mismas pacientes. Con unas 219 pacientes, el error típico del AUC de un fold es ≈ 0,043: diferencias menores de 0,05 entre recetas en un solo fold no se distinguen del azar.

**Cómo leer las gráficas de cada ejecución.**
- *Tres gráficas*: la pérdida y el AUC de train e interna en cada época. Si train sube y la interna se queda quieta o baja, la red está memorizando. A la derecha, el AUC en la época elegida de train, interna y exterior; la diferencia train − exterior mide cuánto se pierde con pacientes nuevas.
- *Panel de resultados*: curva ROC y precisión-recall (no dependen del umbral), la matriz de confusión con el umbral interno, la calibración (¿las probabilidades significan lo que dicen?), cómo se reparten las probabilidades de cada clase y las métricas de la decisión.

### Índice

1. [La red](#red)
2. [Resumen](#resumen)
3. [Por qué E57 no aprendía (N01)](#diagnostico)
4. [P01: lotes con más pacientes](#p01)
5. [P02: aumento geométrico sincronizado](#p02)
6. [P02 en 5 folds: BCE normal frente a ponderada](#cv)
7. [Modelo de entrega y test](#entrega)
8. [E57 como referencia](#e57)
9. [Qué más se probó después de P02](#despues)
10. [N01 en 5 folds: la receta de E57 con validación cruzada](#n01cv)
"""),
        _code("""
%matplotlib inline
import matplotlib.pyplot as plt
plt.rcParams['figure.max_open_warning'] = 0  # el notebook muestra muchas figuras a propósito
from IPython.display import Image, Markdown, display
from src import cuaderno_resultados as R
"""),
        _md("""
<a id="red"></a>
## 1. La red

Todos los experimentos de esta etapa usan **la misma CNN**, la de E57, creada desde cero: lo que cambia es cómo se entrena. Tiene **298.161 parámetros**. Recibe las tres fases de un corte y las convierte en PRE, EARLY − PRE y LATE − EARLY (restas fijas, sin parámetros), para que el realce del contraste llegue explícito. Después hay cuatro bloques de dos convoluciones 3×3 (16, 32, 64 y 128 canales), cada una con BatchNorm y ReLU, y una reducción MaxPool 2×2 en cada bloque: la imagen pasa de 256 × 256 a 16 × 16 píxeles. La media espacial global resume cada canal en un número (128 en total) y una cabeza 128 → 32 → 1, con ReLU y dropout 0,3, da el logit. El campo receptivo de cada neurona final es de 106 × 106 píxeles.

En entrenamiento se promedian las 128 características de los cortes de cada paciente antes de la cabeza; en evaluación, se predice cada corte y se promedian sus probabilidades. La ficha con todas las capas está en [experimentos/E57_patient_feature_mean_mac](../experimentos/E57_patient_feature_mean_mac/README.md).
"""),
        _code("display(Image(filename=str(R.ROOT/'E57/ARQUITECTURA.png'), width=1000))"),
        _md("""
<a id="resumen"></a>
## 2. Resumen

Una fila por ejecución completa de la etapa. *AUC train* y *AUC interna* son de la época elegida; *AUC exterior* es la cifra que cuenta. *Por corte* es el AUC cuando se puntúa cada corte por separado, que es lo que hará la web en la defensa (un corte por paciente).
"""),
        _code('R.tabla_resumen()'),
        _md("""
<a id="diagnostico"></a>
## 3. Por qué E57 no aprendía (N01)

**Qué se observó.** En los 5 folds de E57 la pérdida de entrenamiento se quedó entre 0,59 y 0,63, cuando un predictor que siempre responde la prevalencia (29,3 % de pCR) obtiene 0,605. Sus probabilidades casi no variaban entre pacientes y el AUC sobre sus propias pacientes de entrenamiento era de 0,50–0,56. N01, la comparación BCE normal frente a ponderada con la misma receta, lo confirmó: con la ponderada la pérdida terminó en 0,979, exactamente el óptimo de un predictor constante con ese peso.

**Qué significa.** Los picos de validación de E57 (0,670 en el fold 0) eran un salto aislado dentro de una curva plana, del tamaño que produce el azar al elegir la mejor de unas 20 épocas. Esto explica también por qué ~200 experimentos de arquitectura no se distinguían: comparaban redes que apenas aprendían, con un fold de 219 pacientes.

**Hipótesis.** E57 entrenaba con lotes de solo 2 pacientes. BatchNorm normaliza cada canal con la media del lote: con dos pacientes, sus características quedan forzadas a compensarse y la red solo puede aprender diferencias entre esas dos; en cerca del 59 % de los lotes ambas tienen la misma etiqueta.
"""),
        _code('R.fig_colapso()'),
        _md('Detalle de las ejecuciones de N01 (el fold 0 con las dos pérdidas; los folds 1–4 de la BCE normal se añadieron después, ver sección 10). Fíjate en la tabla de épocas: el AUC de train no se separa de 0,5 y la pérdida apenas se mueve.'),
    ]
    c += _secciones_runs(N01, 'N01', 'N01')
    c += [
        _md("""
<a id="p01"></a>
## 4. P01: lotes con más pacientes

**Qué cambia.** Solo la composición del lote: 10 pacientes × 2 cortes al azar (distintos en cada recorrido) en lugar de 2 pacientes × 10 cortes. Se mantienen las mismas 20 imágenes por paso y los mismos pasos por época, la misma red E57 (298.161 parámetros), las mismas particiones, el mismo optimizador (AdamW, LR 0,001, WD 0,001, ReduceLROnPlateau) y la misma regla de selección. Criterios y lectura completa en [P01_LOTES.md](P01_LOTES.md).

### Paso 1 · ¿Puede memorizar? (40 pacientes, 30 épocas)

Antes de entrenar con todo, se comprueba si cada receta es capaz de memorizar un grupo pequeño. Una red que no memoriza 40 pacientes no puede aprender nada útil de 702.
"""),
        _code('R.tabla_memorizacion()'),
        _code('R.fig_memorizacion()'),
        _md("""
**Lectura.** P01 memoriza en 3–4 épocas y su pérdida cae hasta 0,03–0,16. E57 también llega a 0,94, pero tarda 23 épocas y su pérdida de entrenamiento no baja de 0,54. El criterio fijado era que E57 no pasara de 0,75: lo superó, así que la hipótesis de que E57 *no podía* aprender era demasiado fuerte. Aun así, la composición del lote cambia la velocidad de aprendizaje en un orden de magnitud.

### Paso 2 · Fold 0 con las 702 pacientes
"""),
    ]
    c += _secciones_runs(P01, 'P01', 'P01')
    c += [
        _md("""
**Conclusión de P01.** La red por fin aprende: el AUC en train pasa de 0,47 a 0,94. Pero lo que aprende no se traslada: la validación interna se queda plana entre 0,50 y 0,61 y su pérdida sube desde la época 3. Es memorización. AUC exterior 0,570 (N01 normal, 0,617). La siguiente palanca, fijada de antemano, era regularizar con aumento de datos.

<a id="p02"></a>
## 5. P02: aumento geométrico sincronizado

**Qué cambia.** P01 más aumento geométrico **solo en entrenamiento**: volteo horizontal y vertical (probabilidad 0,5 cada uno), giro de hasta ±10° y traslación de hasta ±3 %. La misma transformación se aplica a PRE, EARLY y LATE, para no desalinear las fases y no destruir el realce. Evaluación y web no cambian. Es la primera vez que el aumento se prueba sobre una receta que aprende; las pruebas anteriores (E37, E49, D16, D18) eran sobre recetas que no aprendían. Criterios en [P02_AUMENTO.md](P02_AUMENTO.md).

La imagen muestra un corte real con cuatro aumentos: el mapa de realce sigue nítido, prueba de que las fases se transforman juntas.
"""),
        _code("display(Image(filename=str(R.NA/'P02_aumento_geometrico/vista_aumento.png'), width=950))"),
    ]
    c += _secciones_runs(P02, 'P02', 'P02 (Mac)')
    c += [
        _md("""
**Conclusión de P02 en el fold 0.** El aumento frena la memorización: la diferencia final entre train e interna baja de 0,37 a 0,14 y la pérdida interna se mantiene estable. Pero la interna no pasa de 0,625 y el exterior es 0,535. Entre N01, P01 y P02 la interna sube (0,572, 0,611, 0,625) y el exterior baja (0,617, 0,570, 0,535): con un fold no se pueden ordenar recetas. Por la regla fijada antes, se pasa a la validación cruzada completa con P02.

<a id="cv"></a>
## 6. P02 en 5 folds: BCE normal frente a ponderada

**Qué se mide.** Los 5 folds con BCE normal y con BCE ponderada (`pos_weight = N0/N1` de las pacientes FIT, ≈ 2,4), entrenados en la GPU de la universidad con la misma receta y particiones. Así cada una de las 1.097 pacientes de train tiene una predicción de un modelo que no la vio, y el error típico del AUC baja a ≈ 0,018. Es además la comparación de pérdidas que exige el enunciado.
"""),
    ]
    if llenas:
        if not completa:
            faltan = ', '.join(f"{PERDIDA[l]}: {len(v)}/5" for l, v in disp.items() if len(v) < 5)
            c.append(_md(f'**Resultados parciales.** Ya están los 5 folds con BCE {", ".join(PERDIDA[l] for l in llenas)}; '
                         f'faltan entrenamientos ({faltan}). Hasta tenerlos no hay comparación pareada ni se aplican las reglas de entrega.'))
        c += [
            _md('### Resumen de los 5 folds\n\nUna fila por entrenamiento. Sensibilidad, especificidad, BA y matriz usan el umbral interno de cada fold.'),
            _code('R.cv_tabla_folds()'),
            _md('### AUC conjunta y comparación pareada\n\nLa *AUC conjunta* junta las predicciones exteriores de los 5 modelos; la *media de folds* evalúa cada modelo por separado. Se reportan las dos. La comparación pareada usa exactamente las mismas pacientes con cada pérdida.'),
            _code("tabla, texto = R.cv_resumen()\ndisplay(tabla)\ndisplay(Markdown(texto))"),
            _md('### Gráficas conjuntas\n\nCurvas de los 5 folds para cada pérdida, matrices sumando los folds (umbral interno, 0,325 y 0,5), ROC y PR, AUC por cohorte y por corte, y calibración.'),
            _code("for fig in R.cv_figuras():\n    display(fig)"),
            _md('### Detalle por entrenamiento'),
        ]
        c += _secciones_runs(P02_GPU, 'P02_GPU', 'P02 (GPU)', nivel='####')
    else:
        hechos = ', '.join(f"{PERDIDA[l]}: {len(v)}/5" for l, v in disp.items())
        c.append(_md(f'**Pendiente.** Entrenamientos disponibles de la GPU ({hechos}). Esta sección se rellena sola al regenerar el notebook cuando estén los 10.'))
    c += [
        _md("""
<a id="entrega"></a>
## 7. Modelo de entrega y test

Las reglas para elegir el modelo final se fijaron el 20 de octubre, antes de ver los resultados de la GPU, y se publicaron en GitHub (commit `830ccd2`) para que la fecha sea verificable: [REGLAS_ENTREGA.md](REGLAS_ENTREGA.md). En resumen: manda la AUC exterior conjunta; se elige BCE normal salvo que la ponderada sea claramente mejor; P02 sustituye a E57 si su AUC conjunta es al menos 0,55; se entrega el promedio de los 5 modelos con la mediana de los umbrales internos; y el test se evalúa una sola vez.
"""),
        _code('display(Markdown(R.entrega()))'),
        _md("""
<a id="e57"></a>
## 8. E57 como referencia

E57 es la red con la que empezó esta etapa (descrita en la sección 1) entrenada con su receta original: lotes de 2 pacientes y sin aumento.

Sus cifras optimistas y honestas no son comparables entre sí. El detalle de sus 5 folds y del test está en [E57/RESULTADOS.ipynb](../E57/RESULTADOS.ipynb); la versión anterior de este notebook, con N01 completo, en [anteriores/RESULTADOS_N01.ipynb](anteriores/RESULTADOS_N01.ipynb).
"""),
        _code('R.tabla_e57()'),
        _md("""
<a id="despues"></a>
## 9. Qué más se probó después de P02

P02 apenas superó a la línea base de cohorte. Antes de entrenar nada más se pasaron filtros baratos sobre las imágenes de train (pasos 0, 0b y 0c), y después se probó si promediar semillas reducía el ruido. Cada hipótesis tiene su documento con el criterio escrito **antes** de calcular o entrenar; el SHA-256 de ese documento queda anotado en su `resumen.json`. Todas las comparaciones son pareadas sobre las 1.097 pacientes, con bootstrap estratificado por fold × cohorte × clase.
"""),
        _code('R.tabla_hipotesis()'),
        _md("""
### P07 · ensemble de 3 semillas

La misma receta P02 normal entrenada con tres semillas por fold (15 modelos) y promediada. La semilla sola mueve el resultado ≈ 0,03; el ensemble queda por encima de la media de sus modelos, en la dirección prevista, pero la mejora frente a la semilla 42 no es demostrable. No se elige una semilla por su AUC exterior: sería seleccionar con la exterior.
"""),
        _code('R.tabla_p07()'),
        _md("""
<a id="n01cv"></a>
## 10. N01 en 5 folds: la receta de E57 con validación cruzada

E57 no tenía una validación cruzada limpia propia: su cifra honesta es el test. N01 es su receta (2 pacientes × 10 cortes por lote, sin aumento) con el protocolo FIT / interna / exterior; solo existía el fold 0 (exterior 0,617, la más alta del proyecto, con AUC de FIT 0,505). Se completaron los folds 1–4 para medirla en las 1.097 pacientes. No es candidata a entrega. Criterios en [N01_CV.md](N01_CV.md).
"""),
        _code('display(Markdown(R.n01_cv()))'),
        _md("""
## Cómo leer estos resultados

El AUC mide si el modelo ordena bien a las pacientes y no depende del umbral; la matriz de confusión sí depende de él. La pregunta de la etapa no ha sido qué arquitectura es mejor, sino si la red aprendía: con la receta de E57 apenas lo hacía; con 10 pacientes por lote aprende; con aumento sincronizado memoriza menos. Cuánto de eso se generaliza a pacientes nuevas lo dice la validación cruzada de la sección 6 y, al final, una única evaluación en test.

Uso educativo, sin validez clínica.
"""),
    ]
    return c


def guardar_por_fold(carpeta=P02_GPU, titulo='P02 (GPU)'):
    """Imágenes sueltas por entrenamiento (curvas y panel de resultados), para verlas o usarlas en diapositivas."""
    destino = Path(str(carpeta).replace('/runs', '/cv_runs'))/'por_fold'
    rutas = []
    for (loss, fold) in sorted(cv_report.load_runs(carpeta)):
        destino.mkdir(parents=True, exist_ok=True)
        for nombre, fig in (('curvas', fig_tres(carpeta, loss, fold, titulo)),
                            ('resultados', fig_resultados(carpeta, loss, fold, titulo)),
                            ('roc_interna_exterior', fig_roc_dos(carpeta, loss, fold, titulo)),
                            ('pr_interna_exterior', fig_pr_dos(carpeta, loss, fold, titulo))):
            ruta = destino/f'{loss}_fold_{fold}_{nombre}.png'
            fig.savefig(ruta, dpi=130)
            plt.close(fig)
            rutas.append(ruta)
    return rutas


def build(ejecutar=True):
    from nbclient import NotebookClient
    anterior = ROOT/'nuevo_enfoque/anteriores/RESULTADOS_N01.ipynb'
    if NB.is_file() and not anterior.exists():
        anterior.parent.mkdir(parents=True, exist_ok=True)
        NB.replace(anterior)   # el notebook previo (N01 y apéndice E57) se conserva
    nb = nbformat.v4.new_notebook(cells=celdas())
    nb.metadata['kernelspec'] = {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'}
    if ejecutar:
        NotebookClient(nb, timeout=900, kernel_name='python3', resources={'metadata': {'path': str(ROOT)}}).execute()
    if P02_GPU.is_dir():
        guardar_por_fold()
    tmp = NB.with_suffix('.tmp.ipynb')
    nbformat.write(nb, tmp)
    tmp.replace(NB)
    return NB


if __name__ == '__main__':
    print(build())
