"""Dossier legible de E57 a partir de resultados existentes; no entrena ni evalúa test."""
import base64
import copy
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nbformat
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_curve, precision_recall_curve

from src.reporting.e57_stability_results import completed
from src.evaluation.metrics import binary_metrics

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / 'experiments/E57_patient_feature_mean_mac/results'
THRESHOLD = .325
BLUE, ORANGE = '#2f75b5', '#ed7134'


def load_cases():
    audit = json.loads((ROOT / 'local/campana_20261015/analysis/original_reference_eval.json').read_text())
    cases = []
    for fold, folder, summary, history in completed():
        patients = pd.read_csv(folder / 'predictions_patient.csv')
        m = binary_metrics(patients.target, patients.probability, THRESHOLD)
        if not np.isclose(m['roc_auc'], summary['validation_metrics']['patient']['roc_auc'], atol=1e-12):
            raise ValueError('AUC distinta de la ejecución original')
        train_auc = (audit['train_eval']['metrics']['roc_auc'] if fold == 0
                     else summary['train_eval_metrics_at_best']['patient']['roc_auc'])
        cases.append(dict(fold=fold, folder=folder, summary=summary, history=history,
                          predictions=patients, metrics=m, train_auc=train_auc,
                          train_bce=(audit['train_eval']['patient_probability_bce'] if fold == 0
                                     else summary['train_eval_loss_at_best'])))
    if len(cases) != 5:
        raise ValueError('El dossier requiere los cinco folds completos')
    return cases


def three_panels(case, path):
    h, s, fold = case['history'], case['summary'], case['fold']
    best = s['best_epoch']
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    a, b, c = axes
    if fold == 0:
        a.plot(h.epoch, h.train_loss, color=BLUE, ls=':', label='Train online · histórico')
        a.scatter([best], [case['train_bce']], color=BLUE, marker='D', s=45, label='Train eval · punto auditado', zorder=5)
        a.text(.02, .03, 'Train online y val eval no son equivalentes.\nNo existe serie histórica de train eval.',
               transform=a.transAxes, fontsize=8, bbox=dict(facecolor='white', alpha=.85, edgecolor='none'))
        b.scatter([best], [case['train_auc']], color=BLUE, marker='D', s=45, label='Train eval · punto auditado', zorder=5)
    else:
        a.plot(h.epoch, h.train_eval_loss, color=BLUE, label='Train eval')
        b.plot(h.epoch, h.train_patient_auc, color=BLUE, label='Train eval')
    a.plot(h.epoch, h.validation_loss, color=ORANGE, label='Validación eval')
    a.scatter([best], [s['validation_loss']], color=ORANGE, edgecolor='white', s=60, zorder=5)
    b.plot(h.epoch, h.patient_roc_auc, color=ORANGE, label='Validación eval')
    b.scatter([best], [case['metrics']['roc_auc']], color=ORANGE, edgecolor='white', s=60, zorder=5)
    a.set(title='1. Evolución de la pérdida', xlabel='Época', ylabel='BCE por paciente')
    b.set(title='2. Evolución del ROC-AUC', xlabel='Época', ylabel='ROC-AUC por paciente', ylim=(.38, 1.02))
    for ax in (a, b):
        ax.axvline(best, color='0.4', linestyle='--', linewidth=1, label=f'Época elegida: {best}')
        ax.grid(alpha=.22)
        ax.legend(fontsize=8)
    values = [case['train_auc'], case['metrics']['roc_auc']]
    c.bar([0, 1], values, color=[BLUE, ORANGE], width=.55)
    for x, value in enumerate(values):
        c.text(x, value + .025, f'{value:.3f}', ha='center', fontsize=12, fontweight='bold')
    final_path = ROOT / 'reports/final_e57_test_v1/test_evaluation.json'
    if fold == 0 and final_path.is_file():
        test = json.loads(final_path.read_text())
        value = test['patient_metrics_primary']['roc_auc']
        c.bar([2], [value], color='#424e57', width=.55)
        c.text(2, value + .025, f'{value:.3f}', ha='center', fontsize=12, fontweight='bold')
    else:
        c.text(2, .52, 'No evaluado\npara este modelo', ha='center', va='center', color='0.45', fontsize=10)
    c.set(title=f'3. AUC en la misma época elegida ({best})', ylabel='ROC-AUC por paciente',
          xticks=[0, 1, 2], xticklabels=['Train', 'Validación', 'TEST'], ylim=(0, 1.05), xlim=(-.6, 2.6))
    c.axhline(.5, color='0.5', ls=':', linewidth=1)
    c.grid(axis='y', alpha=.2)
    fig.suptitle(f'E57 · Fold {fold} · comparación con los tres paneles de referencia', fontsize=17)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def metrics_panels(case, path):
    m, p = case['metrics'], case['predictions']
    y, prob = p.target.to_numpy(), p.probability.to_numpy()
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    a = axes.ravel()
    fpr, tpr, _ = roc_curve(y, prob)
    a[0].plot(fpr, tpr, color=BLUE); a[0].plot([0, 1], [0, 1], '--', color='0.6')
    a[0].set(title=f'ROC · AUC {m["roc_auc"]:.4f}', xlabel='1 − especificidad', ylabel='Sensibilidad')
    precision, recall, _ = precision_recall_curve(y, prob)
    a[1].plot(recall, precision, color=BLUE); a[1].axhline(y.mean(), ls=':', color='0.5')
    a[1].set(title=f'Precisión–recall · AP {m["pr_auc"]:.4f}', xlabel='Sensibilidad', ylabel='Precisión')
    c = m['confusion_matrix']; cm = np.array([[c['tn'], c['fp']], [c['fn'], c['tp']]])
    a[2].imshow(cm, cmap='Blues', vmin=0)
    for i in range(2):
        for j in range(2):
            a[2].text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=20,
                       color='white' if cm[i, j] > cm.max() * .6 else 'black')
    a[2].set(title='Matriz · umbral 0,325', xlabel='Predicción', ylabel='Real',
             xticks=[0, 1], yticks=[0, 1], xticklabels=['no pCR', 'pCR'], yticklabels=['no pCR', 'pCR'])
    observed, predicted = calibration_curve(y, prob, n_bins=8, strategy='quantile')
    a[3].plot(predicted, observed, 'o-', color=BLUE); a[3].plot([0, 1], [0, 1], '--', color='0.6')
    a[3].set(title=f'Calibración · Brier {m["brier_score"]:.4f}', xlabel='Probabilidad media', ylabel='Proporción pCR')
    for label, color in ((0, BLUE), (1, ORANGE)):
        subset = prob[y == label]
        a[4].hist(subset, bins=np.linspace(0, .55, 23), alpha=.5, color=color,
                  weights=np.ones(len(subset)) / len(subset), label=f'Real {label}')
    a[4].axvline(THRESHOLD, color='0.3', ls='--', label='Umbral 0,325')
    a[4].set(title='Probabilidades por clase', xlabel='Probabilidad predicha', ylabel='Fracción dentro de clase')
    a[4].legend(fontsize=8)
    keys = ['precision', 'sensitivity', 'specificity', 'balanced_accuracy', 'f1']
    a[5].bar(range(5), [m[k] for k in keys], color=[BLUE, ORANGE, BLUE, ORANGE, BLUE])
    a[5].set(title='Decisión con umbral 0,325', ylim=(0, 1.05), xticks=range(5),
             xticklabels=['Precisión', 'Sensib.', 'Especif.', 'BA', 'F1 pCR'])
    for ax in (a[0], a[1], a[3], a[4], a[5]): ax.grid(alpha=.18)
    fig.suptitle(f'E57 · Fold {case["fold"]} · checkpoint de época {case["summary"]["best_epoch"]}\n'
                 'Umbral 0,325 elegido después de ver validación · análisis exploratorio', fontsize=15)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def overview_panel(cases, path):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for case in cases:
        h, s, f = case['history'], case['summary'], case['fold']
        if f:
            axes[0].plot(h.epoch, h.train_eval_loss, color=BLUE, alpha=.35)
            axes[1].plot(h.epoch, h.train_patient_auc, color=BLUE, alpha=.35)
        axes[0].plot(h.epoch, h.validation_loss, color=ORANGE, alpha=.5, ls=':' if f == 0 else '-')
        axes[1].plot(h.epoch, h.patient_roc_auc, color=ORANGE, alpha=.5)
        axes[0].scatter(s['best_epoch'], s['validation_loss'], color=ORANGE, edgecolor='white', zorder=4)
        axes[1].scatter(s['best_epoch'], case['metrics']['roc_auc'], color=ORANGE, edgecolor='white', zorder=4)
    for ax in axes[:2]:
        ax.plot([], [], color=BLUE, label='Train eval · folds 1–4')
        ax.plot([], [], color=ORANGE, label='Validación eval · folds 0–4')
        ax.grid(alpha=.2); ax.legend(fontsize=8); ax.set_xlabel('Época')
    axes[0].set(title='1. Pérdida en los cinco folds', ylabel='BCE por paciente')
    axes[0].text(.02, .03, 'Fold 0: no hay serie train eval registrada.', transform=axes[0].transAxes, fontsize=8)
    axes[1].set(title='2. AUC en los cinco folds', ylabel='ROC-AUC por paciente', ylim=(.38, 1.02))
    x = np.arange(5)
    axes[2].bar(x - .18, [c['train_auc'] for c in cases], width=.36, color=BLUE, label='Train eval')
    axes[2].bar(x + .18, [c['metrics']['roc_auc'] for c in cases], width=.36, color=ORANGE, label='Validación eval')
    axes[2].set(title='3. Misma época elegida en cada fold', ylabel='ROC-AUC paciente',
                xticks=x, xticklabels=[f'Fold {i}' for i in range(5)], ylim=(0, 1))
    final_exists = (ROOT / 'reports/final_e57_test_v1/test_evaluation.json').is_file()
    axes[2].text(.5, .95, 'TEST: solo modelo original de fold 0' if final_exists else 'TEST pendiente: no se ha evaluado',
                 transform=axes[2].transAxes, ha='center', color='0.4', fontsize=9)
    axes[2].legend(fontsize=8); axes[2].grid(axis='y', alpha=.2)
    fig.suptitle('E57 · vista conjunta según los tres paneles de referencia', fontsize=17)
    fig.tight_layout(); fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white'); plt.close(fig)


def picture_cell(path, label):
    relative = path.relative_to(ROOT).as_posix()
    cell = nbformat.v4.new_code_cell(
        'from pathlib import Path\nfrom IPython.display import display, Image\n'
        "root=next(p for p in (Path.cwd(),*Path.cwd().parents) if (p/'src').is_dir() and (p/'app.py').is_file())\n"
        f'display(Image(filename=str(root/{relative!r})))')
    cell.execution_count = 1
    cell.outputs = [nbformat.v4.new_output('display_data', data={
        'image/png': base64.b64encode(path.read_bytes()).decode(), 'text/plain': label})]
    cell.metadata.update(tags=['hide-input'], jupyter={'source_hidden': True})
    return cell


def epoch_table(case):
    h = case['history'].copy()
    if case['fold'] == 0:
        h['train_eval_loss'] = np.nan
        h['train_patient_auc'] = np.nan
        chosen = h.epoch.eq(case['summary']['best_epoch'])
        h.loc[chosen, 'train_eval_loss'] = case['train_bce']
        h.loc[chosen, 'train_patient_auc'] = case['train_auc']
    h['Elegida'] = np.where(h.epoch.eq(case['summary']['best_epoch']), 'Sí', '')
    columns = ['epoch', 'train_loss', 'train_eval_loss', 'validation_loss',
               'train_patient_auc', 'patient_roc_auc', 'patient_pr_auc',
               'learning_rate', 'elapsed_seconds', 'Elegida']
    names = ['Época', 'BCE train online', 'BCE train eval', 'BCE val eval',
             'AUC train eval', 'AUC val', 'AP val', 'LR', 'Segundos', 'Elegida']
    table = h[columns].round(5)
    table.columns = names
    return table.fillna('No registrado').to_markdown(index=False)


def build():
    cases = load_cases()
    if DEST.exists() and any(p.suffix.lower() in {'.json', '.csv', '.pt'} for p in DEST.rglob('*')):
        raise ValueError('La carpeta de presentación debe contener solo documentos e imágenes')
    DEST.mkdir(exist_ok=True)
    pictures = DEST / 'GRAFICAS'; pictures.mkdir(exist_ok=True)
    shutil.copyfile(ROOT / 'experiments/E57_patient_feature_mean_mac/arquitectura.png', DEST / 'ARQUITECTURA.png')
    rows = []
    for case in cases:
        m, s = case['metrics'], case['summary']
        rows.append({'Fold': case['fold'], 'Épocas': s['epochs_completed'], 'Mejor época': s['best_epoch'],
                     'AUC paciente': m['roc_auc'], 'AP paciente': m['pr_auc'],
                     'Precisión pCR': m['precision'], 'Sensibilidad': m['sensitivity'],
                     'Especificidad': m['specificity'], 'BA': m['balanced_accuracy'],
                     **{k.upper(): v for k, v in m['confusion_matrix'].items()}})
        three_panels(case, pictures / f'FOLD_{case["fold"]}_TRES_GRAFICAS.png')
        metrics_panels(case, pictures / f'FOLD_{case["fold"]}_RESULTADOS.png')
    overview_panel(cases, pictures / 'COMPARACION_CINCO_FOLDS.png')
    table = pd.DataFrame(rows).round(4).to_markdown(index=False)
    explanation = '''E57 es una CNN 2D propia entrenada desde cero, con **298.161 parámetros**. Recibe PRE, EARLY y LATE y construye PRE, EARLY−PRE y LATE−EARLY. Tiene cuatro bloques de canales 16/32/64/128, dos convoluciones por bloque, BatchNorm, ReLU y cuatro reducciones MaxPool. Después usa promedio espacial global y una cabeza 128→32→1 con ReLU y dropout 0,3.

En entrenamiento se agrupan dos pacientes por lote y se promedian las características de todos sus cortes antes de la cabeza. La pérdida es una BCE normal por paciente. En la evaluación principal se predice cada corte y se promedian sus probabilidades por paciente; esta agregación es distinta de promediar características antes de la cabeza.

AdamW: LR inicial 0,001 y weight decay 0,001; ReduceLROnPlateau según AUC paciente, factor 0,5 y paciencia 3. Semilla 42, máximo 50 épocas y parada tras 10 épocas sin mejorar el AUC de validación. No se usan aumentos, pesos preentrenados, edad ni menopausia. El escalado docente de las intensidades se conserva.

Los cinco folds agrupan cortes de la misma paciente. El fold 0 es histórico y los folds 1–4 se ejecutaron después con la misma receta. La época se eligió con el AUC del mismo fold que se reporta: la media **0,6224** de los cinco máximos es descriptiva y optimista, no una evaluación exterior independiente. Un protocolo anterior con selección interna distinta obtuvo AUC exterior media 0,5272.

Las matrices de este dossier usan **umbral 0,325**, solicitado después de observar la validación. El AUC y las probabilidades no cambian por ese umbral. Los resultados se presentan como exploratorios; bajar el corte aumenta tanto las positivas detectadas como las falsas alarmas. La evaluación de test sigue pendiente.

Las tres gráficas por fold reproducen el propósito de la referencia del profesor: pérdida por época, AUC por época y barras de AUC en la misma época elegida. Para el fold 0 falta la serie train-eval completa: se muestran sus pérdidas históricas con la advertencia correspondiente y un único punto train-eval auditado en la época 14. **No se inventan curvas ni barras de test.**
'''
    if (ROOT / 'reports/final_e57_test_v1/test_evaluation.json').is_file():
        explanation = explanation.replace('La evaluación de test sigue pendiente.',
            'Test ya evaluado para el checkpoint original de época 14; ver la sección final de test.')
        explanation = explanation.replace('**No se inventan curvas ni barras de test.**',
            '**La barra real de test solo se añade al fold 0, que es el modelo evaluado; los otros cuatro modelos no se evalúan en test.**')
    readme = '# E57 · carpeta de presentación\n\nEmpieza por [RESULTADOS.ipynb](RESULTADOS.ipynb): todo está mostrado sin ejecutar celdas.\n\n**Etapa activa:** [research](../../../research/README.md) · [Plan actual](../../../research/PLAN.md). Este dossier conserva la referencia E57.\n\n[Protocolo de cierre](CIERRE.md).\n\n'
    readme += 'La carpeta contiene solo este resumen, el notebook, la imagen de la arquitectura y las gráficas. Las fuentes originales permanecen en sus ubicaciones y en los notebooks generales.\n\n'
    readme += explanation + '\n## Resumen de los cinco folds\n\n' + table
    readme += '\n\n![Arquitectura](ARQUITECTURA.png)\n\nUso educativo, sin validez clínica.\n'
    (DEST / 'README.md').write_text(readme)
    nb = nbformat.v4.new_notebook()
    nb.metadata.kernelspec = {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}
    nb.cells = [nbformat.v4.new_markdown_cell('# E57 · arquitectura y resultados de los cinco folds\n\n' + explanation),
                picture_cell(DEST / 'ARQUITECTURA.png', 'Arquitectura real de E57'),
                nbformat.v4.new_markdown_cell('## Resumen de resultados · umbral 0,325\n\n' + table),
                picture_cell(pictures / 'COMPARACION_CINCO_FOLDS.png', 'Comparación conjunta de los cinco folds')]
    for case in cases:
        fold, s = case['fold'], case['summary']
        note = ('En esta tabla, «No registrado» significa que no existe la medida train-eval por época; '
                'el único valor mostrado en la época 14 procede de la auditoría posterior.\n\n' if fold == 0 else '')
        nb.cells.append(nbformat.v4.new_markdown_cell(
            f'## Fold {fold} · {s["epochs_completed"]} épocas · checkpoint de época {s["best_epoch"]}\n\n'
            + note + '### Todas las épocas\n\n' + epoch_table(case)))
        nb.cells.append(picture_cell(pictures / f'FOLD_{fold}_TRES_GRAFICAS.png', f'Fold {fold} · tres paneles de referencia'))
        nb.cells.append(picture_cell(pictures / f'FOLD_{fold}_RESULTADOS.png', f'Fold {fold} · resultados con umbral 0,325'))
    nb.cells.append(nbformat.v4.new_markdown_cell('## Cómo leer estos resultados\n\n'
        'AUC describe ordenación y no depende del umbral. La matriz depende del corte elegido. '
        'Los cinco folds reúnen 1.097 pacientes: con 0,325 hay TN381, FP394, FN123 y TP199. '
        'Esto corresponde a sensibilidad 61,8 %, especificidad 49,2 % y precisión pCR 33,6 %. '
        'Las matrices tienen más predicciones positivas, pero aún hay muchos errores. '
        'La arquitectura se mantiene como referencia de trabajo; su desempeño final necesita '
        'selección interna y evaluación separada.\n\nUso educativo, sin validez clínica.'))
    test_notebook = DEST / 'TEST.ipynb'
    if test_notebook.is_file():
        test_cells = nbformat.read(test_notebook, 4).cells
        nb.cells[0:0] = copy.deepcopy(test_cells)
        readme_path = DEST / 'README.md'
        readme_path.write_text(readme_path.read_text()
            + '\n\n**TEST final ya evaluado para el checkpoint original:** [TEST.md](TEST.md) · [TEST.ipynb](TEST.ipynb). '
              'Las referencias a test pendiente en la descripción histórica corresponden a la fase previa al cierre.\n')
    nbformat.write(nb, DEST / 'RESULTADOS.ipynb')
    if (ROOT / 'research/ESTADO.md').is_file():
        from src.analysis.new_approach_results import build as build_active_results
        build_active_results(ROOT)
    return DEST


if __name__ == '__main__':
    print(build())
