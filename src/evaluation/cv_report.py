"""Informe de validación cruzada de P01/P02 a partir de salidas ya guardadas.

No entrena, no carga modelos ni imágenes y nunca lee test. Lee cada
`<loss>_fold_<k>_seed_42/` completo (history.csv, summary.json,
decision_before_outer.json y predicciones exteriores) y genera:

  curvas_<loss>.png      pérdida y AUC FIT/interna por época y fold + barras en la época elegida
  matrices.png           matrices sumando los 5 folds: umbral interno, 0,325 fijo y 0,5
  roc_pr.png             ROC y precisión-recall con las predicciones exteriores de todas las pacientes
  cohortes_folds.png     AUC por cohorte, por corte y por fold
  calibracion.png        calibración y distribución de probabilidades por clase
  RESUMEN.md / resumen.json

Uso:
    python -m src.evaluation.cv_report --variant p02 --gpu     # resultados de la universidad (runs_cuda)
    python -m src.evaluation.cv_report --variant p02           # resultados del Mac (runs)
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.transforms
from matplotlib.ticker import FuncFormatter, ScalarFormatter
import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, brier_score_loss, precision_recall_curve,
                             roc_auc_score, roc_curve)

ROOT = Path(__file__).resolve().parents[2]
VARIANT_DIRS = {'p01': 'P01_lotes_pacientes', 'p02': 'P02_aumento_geometrico'}
LOSSES = ('normal', 'weighted')
LOSS_LABEL = {'normal': 'BCE normal', 'weighted': 'BCE ponderada (N0/N1)'}
COHORT_LABEL = {'duke': 'Duke', 'spy1': 'I-SPY1', 'spy2': 'I-SPY2'}
FIXED_THRESHOLDS = (0.325, 0.5)  # 0,325: umbral E57 fijado antes de estos datos; 0,5: referencia

# Colores (validados con la guía de visualización): mismos papeles que las figuras del estudiante.
TRAIN, VALID, OUTER = '#2a78d6', '#eb6834', '#3d4148'
LOSS_COLOR = {'normal': '#1baf7a', 'weighted': '#e87ba4'}
LOSS_STYLE = {'normal': '-', 'weighted': '--'}  # codificación secundaria (daltonismo)
INK, MUTED, GRID = '#1f2328', '#6b7280', '#e5e7eb'
VERDE = '#2f7d4f'

# Referencia teórica de un buen modelo para comparar a simple vista (no es ningún modelo entrenado).
AUC_REFERENCIA = 0.80
PREVALENCIA_TRAIN = 0.293


def referencia(auc=AUC_REFERENCIA, prevalencia=PREVALENCIA_TRAIN):
    """Curvas ROC y PR de un modelo «binormal» con ese AUC: las puntuaciones de cada clase son dos
    normales de igual varianza separadas d = √2·Φ⁻¹(AUC)."""
    from scipy.stats import norm
    d = np.sqrt(2) * norm.ppf(auc)
    fpr = np.linspace(1e-4, 1, 400)
    tpr = norm.cdf(d + norm.ppf(fpr))
    precision = prevalencia * tpr / (prevalencia * tpr + (1 - prevalencia) * fpr)
    ap = float(np.sum(np.diff(np.r_[0, tpr]) * precision))
    return np.r_[0, fpr], np.r_[0, tpr], tpr, precision, ap


def bce_constante(prevalencia=PREVALENCIA_TRAIN):
    """BCE del modelo que responde siempre la prevalencia: por encima de este nivel no aprende nada."""
    return float(-(prevalencia * np.log(prevalencia) + (1 - prevalencia) * np.log(1 - prevalencia)))


def bce_referencia(auc=AUC_REFERENCIA, prevalencia=PREVALENCIA_TRAIN):
    """BCE de ese mismo modelo binormal si sus probabilidades están bien calibradas."""
    from scipy.stats import norm
    d = np.sqrt(2) * norm.ppf(auc)
    s = np.linspace(-8, 8 + d, 20001)
    f0, f1 = norm.pdf(s), norm.pdf(s - d)
    p = prevalencia * f1 / (prevalencia * f1 + (1 - prevalencia) * f0)
    p = np.clip(p, 1e-12, 1 - 1e-12)
    integrando = prevalencia * f1 * -np.log(p) + (1 - prevalencia) * f0 * -np.log(1 - p)
    return float(np.sum(integrando) * (s[1] - s[0]))

plt.rcParams.update({'font.size': 10, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK,
                     'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.spines.top': False,
                     'axes.spines.right': False, 'axes.grid': True, 'grid.color': GRID,
                     'grid.linewidth': 0.8, 'legend.frameon': False})


def load_runs(run_root):
    """{(loss, fold): datos} solo para ejecuciones completas y científicas."""
    runs = {}
    for folder in sorted(Path(run_root).glob('*_fold_*_seed_42')):
        summary_path = folder/'summary.json'
        if not summary_path.is_file():
            continue
        summary = json.loads(summary_path.read_text())
        if summary.get('status') != 'complete' or not summary.get('scientific_result'):
            continue
        if summary.get('test_evaluated'):
            raise ValueError(f'{folder.name} declara test evaluado: no se usa')
        key = (summary['loss'], int(summary['fold']))
        runs[key] = dict(
            folder=folder, summary=summary,
            history=pd.read_csv(folder/'history.csv'),
            decision=json.loads((folder/'decision_before_outer.json').read_text()),
            patients=pd.read_csv(folder/'predictions_outer_patient.csv'),
            cuts=pd.read_csv(folder/'predictions_outer_cut.csv'))
    return runs


def confusion(y, p, threshold):
    y, pred = np.asarray(y).astype(int), np.asarray(p) >= threshold
    tp, fn = int((pred & (y == 1)).sum()), int((~pred & (y == 1)).sum())
    fp, tn = int((pred & (y == 0)).sum()), int((~pred & (y == 0)).sum())
    sens = tp/(tp+fn) if tp+fn else float('nan')
    spec = tn/(tn+fp) if tn+fp else float('nan')
    prec = tp/(tp+fp) if tp+fp else float('nan')
    f1p = 2*tp/(2*tp+fp+fn) if tp else 0.
    f1n = 2*tn/(2*tn+fp+fn) if tn else 0.
    return dict(tn=tn, fp=fp, fn=fn, tp=tp, sensitivity=sens, specificity=spec, precision=prec,
                balanced_accuracy=(sens+spec)/2, f1_macro=(f1p+f1n)/2)


def bootstrap_auc(y, p, n=2000, seed=42, other=None):
    """IC95 % percentil por remuestreo de pacientes. Con `other`, diferencia pareada AUC(p)−AUC(other)."""
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y), np.asarray(p)
    other = None if other is None else np.asarray(other)
    values = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        a = roc_auc_score(y[i], p[i])
        values.append(a - roc_auc_score(y[i], other[i]) if other is not None else a)
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def pooled(runs, loss, cohorts):
    """Predicciones exteriores de todas las pacientes, con el umbral interno de su fold."""
    frames, cut_frames = [], []
    for (l, fold), r in sorted(runs.items()):
        if l != loss:
            continue
        frames.append(r['patients'].assign(fold=fold, threshold=r['decision']['threshold']))
        cut_frames.append(r['cuts'].assign(fold=fold))
    if not frames:
        return None, None
    pat = pd.concat(frames, ignore_index=True)
    if pat.patient_id.duplicated().any():
        raise ValueError('Una paciente aparece en más de un fold exterior')
    pat['cohort'] = pat.patient_id.map(cohorts)
    return pat, pd.concat(cut_frames, ignore_index=True)


def summarize(runs, cohorts):
    result = {}
    for loss in LOSSES:
        pat, cuts = pooled(runs, loss, cohorts)
        if pat is None:
            continue
        folds = []
        for (l, fold), r in sorted(runs.items()):
            if l != loss:
                continue
            s = r['summary']
            folds.append(dict(
                fold=fold, epoca_elegida=s['best_epoch'], epocas=s['epochs_completed'],
                auc_fit=s['fit_at_best']['metrics']['roc_auc'],
                auc_interna=s['inner_at_best']['metrics']['roc_auc'],
                auc_exterior=s['outer_at_05']['metrics']['roc_auc'],
                auc_exterior_corte=s['outer_at_05']['cut_metrics_secondary']['roc_auc'],
                ap_exterior=s['outer_at_05']['metrics']['pr_auc'],
                brier_exterior=s['outer_at_05']['metrics']['brier_score'],
                umbral_interno=r['decision']['threshold'],
                minutos=s.get('seconds', float('nan'))/60))
        f = pd.DataFrame(folds)
        y, p = pat.target.to_numpy(), pat.probability.to_numpy()
        internal = np.asarray(p) >= pat.threshold.to_numpy()
        cm_internal = confusion(y, internal.astype(float), 0.5)
        result[loss] = dict(
            folds=folds, n_folds=len(f), n_pacientes=len(pat), n_pcr=int(y.sum()),
            auc_media_folds=float(f.auc_exterior.mean()), auc_sd_folds=float(f.auc_exterior.std(ddof=1)) if len(f) > 1 else None,
            auc_conjunta=float(roc_auc_score(y, p)), auc_conjunta_ic95=bootstrap_auc(y, p),
            auc_corte_conjunta=float(roc_auc_score(cuts.target, cuts.probability)),
            ap_conjunta=float(average_precision_score(y, p)), prevalencia=float(y.mean()),
            brier_conjunto=float(brier_score_loss(y, p)),
            brier_constante=float(brier_score_loss(y, np.full(len(y), y.mean()))),
            auc_fit_media=float(f.auc_fit.mean()), auc_interna_media=float(f.auc_interna.mean()),
            matrices={'umbral interno (por fold)': cm_internal,
                      **{f'{t:.3f}'.rstrip('0').replace('.', ','): confusion(y, p, t) for t in FIXED_THRESHOLDS}},
            cohortes={c: dict(n=int(len(g)), pcr=int(g.target.sum()),
                              auc=float(roc_auc_score(g.target, g.probability)) if g.target.nunique() == 2 else None)
                      for c, g in pat.groupby('cohort')})
    if all(l in result for l in LOSSES):
        a, b = pooled(runs, 'normal', cohorts)[0], pooled(runs, 'weighted', cohorts)[0]
        m = a.merge(b, on='patient_id', suffixes=('_n', '_w'))
        if len(m) == len(a) == len(b) and (m.target_n == m.target_w).all():
            diff = roc_auc_score(m.target_n, m.probability_n) - roc_auc_score(m.target_n, m.probability_w)
            result['pareada'] = dict(n=len(m), diferencia_auc_normal_menos_ponderada=float(diff),
                                     ic95=bootstrap_auc(m.target_n, m.probability_n, other=m.probability_w))
    return result


# ---------------------------------------------------------------- figuras
# Con `out` se guarda el PNG y se cierra la figura; sin `out` se devuelve para verla en el notebook.

def comas(fig):
    """Coma decimal en los ejes numéricos (los ejes con etiquetas de texto no se tocan)."""
    for ax in fig.axes:
        for eje in (ax.xaxis, ax.yaxis):
            if isinstance(eje.get_major_formatter(), ScalarFormatter):
                eje.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}'.replace('.', ',')))
    return fig


def _save(fig, out):
    comas(fig)
    if out is None:
        return fig
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def fig_curves(runs, loss, out, title):
    keys = sorted(k for k in runs if k[0] == loss)
    fig, ax = plt.subplots(1, 3, figsize=(17, 5.2), gridspec_kw=dict(width_ratios=[1.25, 1.25, 1]))
    for _, fold in keys:
        r = runs[(loss, fold)]
        h, best = r['history'], r['summary']['best_epoch']
        b = h.loc[h.epoch == best].iloc[0]
        for i, (fit, inner) in enumerate((('fit_bce', 'inner_bce'), ('fit_auc', 'inner_auc'))):
            ax[i].plot(h.epoch, h[fit], color=TRAIN, lw=1.6, alpha=.85)
            ax[i].plot(h.epoch, h[inner], color=VALID, lw=1.6, alpha=.85)
            ax[i].plot(best, b[inner], 'o', ms=8, color=VALID, mec='white', mew=2, zorder=5)
            if len(keys) == 1:  # un solo fold: línea vertical en la época elegida, como en la figura de referencia
                ax[i].axvline(best, color=MUTED, ls=':', lw=1.2, zorder=1)
    handles = [plt.Line2D([], [], color=TRAIN, lw=2, label='entrenamiento (FIT, modo eval)'),
               plt.Line2D([], [], color=VALID, lw=2, label='validación interna'),
               plt.Line2D([], [], color=VALID, marker='o', ls='', ms=8, mec='white', label='época elegida')]
    ax[0].set(title='Pérdida BCE por paciente', xlabel='Época', ylabel='BCE (misma unidad y modo)')
    ax[1].set(title='ROC-AUC por paciente', xlabel='Época', ylabel='ROC-AUC', ylim=(0.4, 1.0))
    ax[1].axhline(0.5, color=MUTED, lw=1, ls=':')
    constante, buena = bce_constante(), bce_referencia()
    ax[0].axhline(constante, color=MUTED, lw=1.2, ls='--')
    ax[0].axhline(buena, color=VERDE, lw=2, ls=':')
    ax[1].axhline(AUC_REFERENCIA, color=VERDE, lw=2, ls=':')
    extra0 = [plt.Line2D([], [], color=MUTED, ls='--', lw=1.2, label=f'modelo que no mira la imagen ({_fmt(constante)})'),
              plt.Line2D([], [], color=VERDE, ls=':', lw=2, label=f'referencia teórica, buen modelo AUC {_fmt(AUC_REFERENCIA, 2)} ({_fmt(buena)})')]
    extra1 = [plt.Line2D([], [], color=VERDE, ls=':', lw=2, label=f'referencia teórica, buen modelo ({_fmt(AUC_REFERENCIA, 2)})')]
    ax[0].legend(handles=handles + extra0, loc='best', fontsize=8.5)
    ax[1].legend(handles=handles + extra1, loc='best', fontsize=8.5)
    fit = [runs[k]['summary']['fit_at_best']['metrics']['roc_auc'] for k in keys]
    inner = [runs[k]['summary']['inner_at_best']['metrics']['roc_auc'] for k in keys]
    outer = [runs[k]['summary']['outer_at_05']['metrics']['roc_auc'] for k in keys]
    names = ['entrenamiento', 'validación\ninterna', 'EXTERIOR\n(no vistas)']
    for i, (vals, color) in enumerate(zip((fit, inner, outer), (TRAIN, VALID, OUTER))):
        ax[2].bar(i, np.mean(vals), width=.62, color=color, zorder=2)
        ax[2].scatter(np.full(len(vals), i) + np.linspace(-.15, .15, len(vals)), vals, s=22,
                      color='white', edgecolor=INK, lw=.8, zorder=3)
        ax[2].text(i, max(vals) + .02, _fmt(float(np.mean(vals)), 2), ha='center', fontweight='bold', color=INK)
    ax[2].axhline(0.5, color=MUTED, lw=1, ls=':')
    ax[2].axhline(AUC_REFERENCIA, color=VERDE, lw=2, ls=':')
    ax[2].text(2.42, AUC_REFERENCIA + 0.01, f'buen modelo\n(teórico, {_fmt(AUC_REFERENCIA, 2)})', ha='right', va='bottom', fontsize=8.5, color=VERDE)
    ax[2].set(xticks=range(3), xticklabels=names, ylim=(0.4, 1.05), ylabel='ROC-AUC en la época elegida',
              title=f'Media de {len(keys)} folds (puntos = cada fold)' if len(keys) > 1 else 'En la época elegida')
    gap = np.mean(fit) - np.mean(outer)
    ax[2].text(0.98, 0.97, f'entrenamiento − exterior = {_fmt(float(gap), 2)}', transform=ax[2].transAxes,
               ha='right', va='top', fontweight='bold', color=INK)
    ax[2].grid(axis='x', visible=False)
    alcance = f'curvas de los {len(keys)} folds' if len(keys) > 1 else f'fold {keys[0][1]}'
    fig.suptitle(f'{title} · {LOSS_LABEL[loss]} · {alcance}', fontweight='bold', x=0.01, ha='left')
    fig.tight_layout()
    return _save(fig, out)


def fig_matrices(summary, out):
    losses = [l for l in LOSSES if l in summary]
    cols = list(summary[losses[0]]['matrices'])
    fig, axes = plt.subplots(len(losses), len(cols), figsize=(4.6*len(cols), 4.4*len(losses)), squeeze=False)
    for i, loss in enumerate(losses):
        for j, name in enumerate(cols):
            c, a = summary[loss]['matrices'][name], axes[i][j]
            m = np.array([[c['tn'], c['fp']], [c['fn'], c['tp']]])
            rows = m / m.sum(1, keepdims=True)
            a.imshow(rows, cmap='Blues', vmin=0, vmax=1)
            for (r, k), v in np.ndenumerate(m):
                a.text(k, r, f'{v}\n{rows[r, k]:.0%}', ha='center', va='center', fontsize=12,
                       color='white' if rows[r, k] > .55 else INK, fontweight='bold')
            a.set(xticks=[0, 1], yticks=[0, 1], xticklabels=['pred. no pCR', 'pred. pCR'],
                  yticklabels=['real no pCR', 'real pCR'])
            a.grid(False)
            label = 'umbral interno de cada fold' if name.startswith('umbral') else f'umbral fijo {name}'
            a.set_title(f'{LOSS_LABEL[loss]}\n{label}', fontsize=10)
            a.set_xlabel(f"sens {c['sensitivity']:.2f} · espec {c['specificity']:.2f} · "
                         f"prec {c['precision']:.2f}\nBA {c['balanced_accuracy']:.2f} · F1 macro {c['f1_macro']:.2f}",
                         fontsize=9)
    fig.suptitle(f"Matrices de confusión sumando los folds exteriores ({summary[losses[0]]['n_pacientes']} pacientes); "
                 "% por fila", fontweight='bold', x=0.01, ha='left')
    fig.tight_layout()
    return _save(fig, out)


def fig_roc_pr(pats, summary, out):
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.4))
    for loss, pat in pats.items():
        s = summary[loss]
        fpr, tpr, _ = roc_curve(pat.target, pat.probability)
        lo, hi = s['auc_conjunta_ic95']
        ax[0].plot(fpr, tpr, LOSS_STYLE[loss], color=LOSS_COLOR[loss], lw=2,
                   label=f"{LOSS_LABEL[loss]}: AUC {_fmt(s['auc_conjunta'])} [{_fmt(lo)}–{_fmt(hi)}]")
        prec, rec, _ = precision_recall_curve(pat.target, pat.probability)
        ax[1].plot(rec, prec, LOSS_STYLE[loss], color=LOSS_COLOR[loss], lw=2,
                   label=f"{LOSS_LABEL[loss]}: AP {_fmt(s['ap_conjunta'])}")
    prev = next(iter(summary.values()))['prevalencia'] if summary else 0
    fpr_r, tpr_r, rec_r, prec_r, ap_r = referencia(prevalencia=prev)
    ax[0].plot(fpr_r, tpr_r, ':', color=VERDE, lw=2.2, label=f'referencia teórica, buen modelo: AUC {_fmt(AUC_REFERENCIA, 2)}')
    ax[1].plot(rec_r, prec_r, ':', color=VERDE, lw=2.2, label=f'referencia teórica (AUC {_fmt(AUC_REFERENCIA, 2)}): AP {_fmt(ap_r)}')
    ax[0].plot([0, 1], [0, 1], ':', color=MUTED, lw=1, label='azar (0,5)')
    ax[1].axhline(prev, ls=':', color=MUTED, lw=1, label=f'prevalencia ({_fmt(prev)})')
    ax[0].set(xlabel='1 − especificidad', ylabel='sensibilidad', title='ROC exterior (todas las pacientes)')
    ax[1].set(xlabel='recall (sensibilidad)', ylabel='precisión', title='Precisión-recall exterior', ylim=(0, 1))
    for a in ax:
        a.legend(loc='lower right' if a is ax[0] else 'upper right', fontsize=9)
    sub = ''
    if 'pareada' in summary:
        d = summary['pareada']
        sgn = lambda v: f'{v:+.3f}'.replace('.', ',')
        sub = (f" · normal − ponderada = {sgn(d['diferencia_auc_normal_menos_ponderada'])} "
               f"[IC95 {sgn(d['ic95'][0])} a {sgn(d['ic95'][1])}], mismas {d['n']} pacientes")
    fig.suptitle('Predicciones exteriores de los 5 folds juntas' + sub, fontweight='bold', x=0.01, ha='left', fontsize=11)
    fig.tight_layout()
    return _save(fig, out)


def fig_cohorts(summary, out):
    losses = [l for l in LOSSES if l in summary]
    groups = ['global\n(paciente)', 'global\n(corte)'] + [COHORT_LABEL[c] for c in ('duke', 'spy1', 'spy2')]
    fig, ax = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw=dict(width_ratios=[1.6, 1]))
    width = 0.8/len(losses)
    for i, loss in enumerate(losses):
        s = summary[loss]
        vals = [s['auc_conjunta'], s['auc_corte_conjunta']] + [s['cohortes'].get(c, {}).get('auc') for c in ('duke', 'spy1', 'spy2')]
        ns = [s['n_pacientes'], None] + [s['cohortes'].get(c, {}).get('n') for c in ('duke', 'spy1', 'spy2')]
        x = np.arange(len(groups)) + (i - (len(losses)-1)/2)*width
        ax[0].bar(x, [v or 0 for v in vals], width*0.92, color=LOSS_COLOR[loss], label=LOSS_LABEL[loss],
                  hatch='' if loss == 'normal' else '//', edgecolor='white', zorder=2)
        for xi, v, n in zip(x, vals, ns):
            if v is not None:
                ax[0].text(xi, v + .01, _fmt(v, 2) + (f'\nn={n}' if n else ''), ha='center', fontsize=8, color=INK)
        fold_aucs = [f['auc_exterior'] for f in s['folds']]
        ax[1].scatter(np.full(len(fold_aucs), i), fold_aucs, s=70, color=LOSS_COLOR[loss],
                      marker='o' if loss == 'normal' else 'D', edgecolor='white', lw=1.5, zorder=3)
        ax[1].hlines(np.mean(fold_aucs), i-.18, i+.18, color=INK, lw=2.5, zorder=4)
        ax[1].text(i + .22, np.mean(fold_aucs), f'media {_fmt(float(np.mean(fold_aucs)))}',
                   va='center', ha='left', fontsize=9, color=INK, fontweight='bold',
                   bbox=dict(facecolor='white', edgecolor='none', pad=1))
    if len(losses) == 2:  # misma fold, mismas pacientes: línea que une la pareja
        by = {l: {f['fold']: f['auc_exterior'] for f in summary[l]['folds']} for l in losses}
        for fold in sorted(set(by['normal']) & set(by['weighted'])):
            ax[1].plot([0, 1], [by['normal'][fold], by['weighted'][fold]], color=MUTED, lw=1, zorder=2)
            ax[1].text(-.06, by['normal'][fold], f'fold {fold}', ha='right', va='center', fontsize=8, color=MUTED)
    ax[0].axhline(0.5, color=MUTED, lw=1, ls=':')
    ax[0].set(xticks=range(len(groups)), xticklabels=groups, ylim=(0.3, 0.9), ylabel='ROC-AUC exterior',
              title='Por cohorte y por corte (la demo evalúa un corte)')
    ax[0].legend(loc='upper right', fontsize=9)
    ax[0].grid(axis='x', visible=False)
    ax[1].axhline(0.5, color=MUTED, lw=1, ls=':')
    ax[1].set(xticks=range(len(losses)), xticklabels=[LOSS_LABEL[l] for l in losses], xlim=(-.6, len(losses)-.1),
              ylim=(0.35, 0.85), ylabel='ROC-AUC exterior del fold', title='Cada fold: normal frente a ponderada')
    ax[1].grid(axis='x', visible=False)
    fig.suptitle('Dónde funciona y cuánto varía', fontweight='bold', x=0.01, ha='left')
    fig.tight_layout()
    return _save(fig, out)


def fig_calibration(pats, summary, out):
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    ax[0].plot([0, 1], [0, 1], ':', color=MUTED, lw=1, label='calibración perfecta')
    for loss, pat in pats.items():
        q = pd.qcut(pat.probability, 10, duplicates='drop')
        g = pat.groupby(q, observed=True).agg(p=('probability', 'mean'), y=('target', 'mean'))
        ax[0].plot(g.p, g.y, LOSS_STYLE[loss], marker='o', ms=6, color=LOSS_COLOR[loss], lw=2,
                   label=f"{LOSS_LABEL[loss]} (Brier {_fmt(summary[loss]['brier_conjunto'])})")
    ref = next(iter(summary.values()))
    ax[0].set(xlabel='probabilidad media predicha (deciles)', ylabel='proporción real de pCR',
              title=f"Calibración (Brier del predictor constante {_fmt(ref['brier_constante'])})", xlim=(0, 1), ylim=(0, 1))
    ax[0].legend(loc='upper left', fontsize=9)
    data, labels, colors, pos = [], [], [], []
    for i, (loss, pat) in enumerate(pats.items()):
        for y in (0, 1):
            data.append(pat.probability[pat.target == y].to_numpy())
            labels.append(f"{'ponderada' if loss == 'weighted' else 'normal'}\n{'pCR' if y else 'no pCR'}")
            colors.append(LOSS_COLOR[loss]); pos.append(i*2.6 + y)
    box = ax[1].boxplot(data, positions=pos, widths=.7, orientation='horizontal', patch_artist=True, showfliers=False,
                        medianprops=dict(color=INK, lw=2))
    for patch, color, y in zip(box['boxes'], colors, [0, 1]*len(pats)):
        patch.set(facecolor=color, alpha=.35 if y == 0 else .9, edgecolor=color)
    for t in FIXED_THRESHOLDS:
        ax[1].axvline(t, color=MUTED, lw=1, ls=':')
        ax[1].text(t, 0.005, f' {t:g}'.replace('.', ','), color=MUTED, fontsize=8, ha='left', va='bottom',
                   transform=matplotlib.transforms.blended_transform_factory(ax[1].transData, ax[1].transAxes))
    ax[1].set(yticks=pos, yticklabels=labels, xlim=(0, 1), xlabel='probabilidad predicha de pCR',
              title='Probabilidades por clase real (cajas: cuartiles)')
    ax[1].grid(axis='y', visible=False)
    fig.suptitle('¿Las probabilidades significan lo que dicen?', fontweight='bold', x=0.01, ha='left')
    fig.tight_layout()
    return _save(fig, out)


# ---------------------------------------------------------------- informe

def _fmt(v, d=3):
    return '—' if v is None or (isinstance(v, float) and np.isnan(v)) else f'{v:.{d}f}'.replace('.', ',')


def write_markdown(summary, out_dir, title, run_root):
    lines = [f'# {title} · validación cruzada', '',
             f'Generado con `python -m src.evaluation.cv_report` a partir de `{run_root}`. Sin entrenar ni leer test. '
             'Época y umbral de cada fold se eligieron con su validación interna antes de mirar el exterior.', '']
    for loss in LOSSES:
        if loss not in summary:
            continue
        s = summary[loss]
        lines += [f'## {LOSS_LABEL[loss]}', '',
                  f"**AUC exterior conjunta ({s['n_pacientes']} pacientes, {s['n_pcr']} pCR): "
                  f"{_fmt(s['auc_conjunta'])} [IC95 {_fmt(s['auc_conjunta_ic95'][0])}–{_fmt(s['auc_conjunta_ic95'][1])}]** · "
                  f"media de folds {_fmt(s['auc_media_folds'])} (DE {_fmt(s['auc_sd_folds'])}) · por corte {_fmt(s['auc_corte_conjunta'])} · "
                  f"AP {_fmt(s['ap_conjunta'])} (prevalencia {_fmt(s['prevalencia'])}) · "
                  f"Brier {_fmt(s['brier_conjunto'])} (constante {_fmt(s['brier_constante'])})", '',
                  f'![curvas](curvas_{loss}.png)', '',
                  '| Fold | Época | AUC FIT | AUC interna | AUC exterior | Exterior por corte | AP | Brier | Umbral interno | Min |',
                  '|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
        for f in s['folds']:
            lines.append(f"| {f['fold']} | {f['epoca_elegida']}/{f['epocas']} | {_fmt(f['auc_fit'])} | {_fmt(f['auc_interna'])} | "
                         f"**{_fmt(f['auc_exterior'])}** | {_fmt(f['auc_exterior_corte'])} | {_fmt(f['ap_exterior'])} | "
                         f"{_fmt(f['brier_exterior'])} | {_fmt(f['umbral_interno'])} | {_fmt(f['minutos'], 0)} |")
        lines += ['', '| Umbral | TN | FP | FN | TP | Sensib. | Especif. | Precisión | BA | F1 macro |',
                  '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
        for name, c in s['matrices'].items():
            lines.append(f"| {name} | {c['tn']} | {c['fp']} | {c['fn']} | {c['tp']} | {_fmt(c['sensitivity'])} | "
                         f"{_fmt(c['specificity'])} | {_fmt(c['precision'])} | {_fmt(c['balanced_accuracy'])} | {_fmt(c['f1_macro'])} |")
        lines.append('')
    if 'pareada' in summary:
        d = summary['pareada']
        lines += ['## Normal frente a ponderada', '',
                  f"Mismas {d['n']} pacientes: AUC normal − ponderada = **{d['diferencia_auc_normal_menos_ponderada']:+.3f}** "
                  f"[IC95 bootstrap pareado {d['ic95'][0]:+.3f} a {d['ic95'][1]:+.3f}]. "
                  + ('El intervalo incluye 0: no hay diferencia demostrada.' if d['ic95'][0] <= 0 <= d['ic95'][1]
                     else 'El intervalo excluye 0.'), '']
    lines += ['## Figuras', '', '![matrices](matrices.png)', '', '![ROC y PR](roc_pr.png)', '',
              '![cohortes](cohortes_folds.png)', '', '![calibración](calibracion.png)', '',
              'La AUC conjunta junta predicciones de 5 modelos distintos; la media de folds evalúa cada modelo por separado. '
              'Se reportan las dos. El umbral 0,325 se fijó con E57 antes de estos datos; 0,5 es solo referencia.', '',
              'Uso educativo, sin validez clínica.', '']
    (out_dir/'RESUMEN.md').write_text('\n'.join(lines))


def build(variant='p02', mode='runs', root=ROOT, cohorts=None, out_dir=None):
    root = Path(root)
    tech = root/'reports/new_approach'/VARIANT_DIRS[variant]
    run_root = tech/mode
    runs = load_runs(run_root)
    if not runs:
        raise FileNotFoundError(f'No hay ejecuciones completas en {run_root}')
    if cohorts is None:
        meta = pd.read_csv(root/'breastdcedl/metadata/patients.csv', usecols=['pid', 'dataset'])
        cohorts = dict(zip(meta.pid, meta.dataset))
    out_dir = Path(out_dir) if out_dir else tech/f'cv_{mode}'
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize(runs, cohorts)
    title = f"{variant.upper()} ({'GPU' if mode.endswith('cuda') else 'Mac'})"
    for loss in LOSSES:
        if loss in summary:
            fig_curves(runs, loss, out_dir/f'curvas_{loss}.png', title)
    pats = {l: pooled(runs, l, cohorts)[0] for l in LOSSES if l in summary}
    fig_matrices(summary, out_dir/'matrices.png')
    fig_roc_pr(pats, summary, out_dir/'roc_pr.png')
    fig_cohorts(summary, out_dir/'cohortes_folds.png')
    fig_calibration(pats, summary, out_dir/'calibracion.png')
    (out_dir/'resumen.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    write_markdown(summary, out_dir, title, run_root.relative_to(root) if run_root.is_relative_to(root) else run_root)
    return out_dir, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--variant', choices=sorted(VARIANT_DIRS), default='p02')
    parser.add_argument('--gpu', action='store_true', help='leer runs_cuda (universidad)')
    args = parser.parse_args()
    out, summary = build(args.variant, 'runs_cuda' if args.gpu else 'runs')
    for loss in LOSSES:
        if loss in summary:
            s = summary[loss]
            print(f"{LOSS_LABEL[loss]}: {s['n_folds']} folds, AUC exterior conjunta {s['auc_conjunta']:.3f} "
                  f"[{s['auc_conjunta_ic95'][0]:.3f}–{s['auc_conjunta_ic95'][1]:.3f}], media folds {s['auc_media_folds']:.3f}")
    if 'pareada' in summary:
        d = summary['pareada']
        print(f"normal − ponderada: {d['diferencia_auc_normal_menos_ponderada']:+.3f} [{d['ic95'][0]:+.3f}, {d['ic95'][1]:+.3f}]")
    print(f'Informe: {out}/RESUMEN.md')


if __name__ == '__main__':
    main()
