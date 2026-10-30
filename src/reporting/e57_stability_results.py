"""Genera una vista estática de los folds E57 ya completados; nunca entrena."""
import base64
import copy
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nbformat
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'experiments/E57_patient_feature_mean_mac/stability'
OLD = ROOT / 'reports/patient_features/E57_patient_feature_mean_mac/fold_0_seed_42'


def completed():
    folders = [(0, OLD)] + [(fold, BASE / f'resultados/fold_{fold}_seed_42')
                            for fold in range(1, 5)]
    result = []
    for fold, folder in folders:
        if not (folder / 'summary.json').is_file():
            continue
        summary = json.loads((folder / 'summary.json').read_text())
        if (summary.get('status') != 'complete' or summary.get('scientific_result') is not True
                or summary.get('test_evaluated') is not False):
            raise ValueError(f'Resultado E57 incompleto o inválido: {folder}')
        history = pd.read_csv(folder / 'history.csv')
        m = summary['validation_metrics']['patient']
        selected = history.loc[history.epoch == summary['best_epoch']]
        if len(selected) != 1 or not np.isclose(selected.iloc[0].patient_roc_auc, m['roc_auc']):
            raise ValueError(f'Checkpoint e historial no coinciden: {folder}')
        result.append((fold, folder, summary, history))
    return result


def panel(items):
    newer = [(fold, history, summary) for fold, _, summary, history in items if fold > 0]
    if not newer:
        return None
    fig, axes = plt.subplots(2, 4, figsize=(19, 7), sharex='col', sharey='row')
    for index in range(4):
        ax_loss, ax_auc = axes[:, index]
        if index >= len(newer):
            ax_loss.axis('off'); ax_auc.axis('off'); continue
        fold, h, s = newer[index]
        ax_loss.plot(h.epoch, h.train_eval_loss, label='Train eval', color='#2878b5')
        ax_loss.plot(h.epoch, h.validation_loss, label='Validación eval', color='#e46b2e')
        ax_auc.plot(h.epoch, h.train_patient_auc, label='Train eval', color='#2878b5')
        ax_auc.plot(h.epoch, h.patient_roc_auc, label='Validación eval', color='#e46b2e')
        ax_loss.set_title(f'Fold {fold} · mejor época {s["best_epoch"]}')
        for ax in (ax_loss, ax_auc):
            ax.axvline(s['best_epoch'], color='0.4', linestyle='--', linewidth=1)
            ax.grid(alpha=.2)
        ax_auc.set_xlabel('Época')
    axes[0, 0].set_ylabel('BCE paciente · misma evaluación')
    axes[1, 0].set_ylabel('ROC-AUC paciente')
    axes[1, 0].set_ylim(.38, .72)
    axes[0, 0].legend(fontsize=9)
    fig.suptitle('E57: folds 1–4 · train y validación en el mismo modo', fontsize=17)
    fig.tight_layout()
    target = BASE / 'panel_curvas_comparables.png'
    fig.savefig(target, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return target


def image_cell(path, label):
    cell = nbformat.v4.new_code_cell(f'# {label}\n# Imagen guardada; visible sin ejecutar esta celda.\n')
    cell.execution_count = 1
    cell.outputs = [nbformat.v4.new_output('display_data', data={
        'image/png': base64.b64encode(path.read_bytes()).decode(),
        'text/plain': label,
    })]
    cell.metadata.update(tags=['hide-input'], jupyter={'source_hidden': True})
    return cell


def detail_cells(fold, folder, summary):
    from src.reporting.clear_results import show_run
    from src.reporting.balanced_results import class_outputs
    from src.evaluation.decision_diagnostics import decision_outputs
    relative = folder.relative_to(ROOT).as_posix()
    heading = nbformat.v4.new_markdown_cell(
        f'## E57 · Fold {fold} · todas las épocas, gráficas y matriz\n\n'
        f'Mejor época {summary["best_epoch"]}/{summary["epochs_completed"]}. '
        + ('Referencia histórica. ' if fold == 0 else '')
        + 'La tabla siguiente contiene todas las épocas; después aparecen los seis paneles '
          'habituales y el diagnóstico de probabilidades.')
    source = f'''from pathlib import Path
import sys, json
from IPython.display import display, Image
root=next(p for p in (Path.cwd(),*Path.cwd().parents) if (p/'src').is_dir() and (p/'app.py').is_file())
if str(root) not in sys.path: sys.path.insert(0,str(root))
from src.reporting.clear_results import show_run
from src.reporting.balanced_results import class_table
from src.evaluation.decision_diagnostics import display_decision
folder=root/{relative!r}
summary=json.loads((folder/'summary.json').read_text())
display(class_table(summary['validation_metrics']['patient']))
show_run(folder,summary)
display_decision(folder)
if (folder/'curvas_comparables.png').exists():
    display(Image(filename=str(folder/'curvas_comparables.png')))
'''
    cell = nbformat.v4.new_code_cell(source)
    cell.execution_count = 1
    cell.metadata.update(tags=['hide-input'], jupyter={'source_hidden': True})
    cell.outputs = (class_outputs(summary) + show_run(folder, summary, as_outputs=True)
                    + decision_outputs(folder))
    matched = folder / 'curvas_comparables.png'
    if matched.is_file():
        cell.outputs.extend(image_cell(matched, 'Train y validación en el mismo modo de evaluación').outputs)
    for index, item in enumerate((heading, cell)):
        item.metadata['campaign_overview'] = f'E57_estabilidad_detalle_f{fold}_{index}'
        item.metadata['e57_grouped_directory'] = relative
    return [heading, cell]


def decision_audit_cells(items):
    from sklearn.metrics import confusion_matrix
    from src.evaluation.decision_diagnostics import threshold_table
    ranges, tables = [], []
    fig, axes = plt.subplots(3, len(items), figsize=(3.3 * len(items), 8.5), squeeze=False)
    for column, (fold, folder, summary, _) in enumerate(items):
        patients = pd.read_csv(folder / 'predictions_patient.csv')
        cuts = pd.read_csv(folder / 'predictions_cut.csv')
        means = cuts.groupby('patient_id').probability.mean()
        saved = patients.set_index('patient_id').probability
        if not np.allclose(means.sort_index(), saved.sort_index(), atol=1e-12, rtol=0):
            raise ValueError('La agregación por paciente no coincide con los cortes')
        if not cuts.groupby('patient_id').target.nunique().eq(1).all():
            raise ValueError('Etiquetas inconsistentes entre cortes de una paciente')
        original = summary['validation_metrics']['patient']['confusion_matrix']
        actual = confusion_matrix(patients.target, patients.probability >= .5, labels=[0, 1])
        if actual.tolist() != [[original['tn'], original['fp']], [original['fn'], original['tp']]]:
            raise ValueError('Matriz guardada incorrecta')
        ranges.append({'Fold': fold, 'pCR reales': int(patients.target.sum()),
                       'Prob. mínima': round(float(saved.min()), 4),
                       'Prob. media': round(float(saved.mean()), 4),
                       'Prob. máxima': round(float(saved.max()), 4),
                       'Predicciones positivas a 0,5': int((saved >= .5).sum())})
        table = threshold_table(patients)
        table.insert(0, 'Fold', fold)
        tables.append(table)
        for row, threshold in enumerate((.5, .325, .35)):
            cm = confusion_matrix(patients.target, patients.probability >= threshold, labels=[0, 1])
            ax = axes[row, column]
            ax.imshow(cm, cmap='Blues', vmin=0, vmax=170)
            for a in range(2):
                for b in range(2):
                    ax.text(b, a, str(cm[a, b]), ha='center', va='center',
                            color='white' if cm[a, b] > 90 else 'black', fontsize=14)
            ax.set_title(f'Fold {fold} · umbral {threshold:g}')
            ax.set(xticks=[0, 1], yticks=[0, 1], xlabel='Predicho (0 / 1)', ylabel='Real (0 / 1)')
    fig.suptitle('E57 · comparación descriptiva de umbrales fijos, sin evaluación independiente', fontsize=15)
    fig.tight_layout()
    picture = BASE / 'diagnostico_matrices_umbrales.png'
    fig.savefig(picture, dpi=140, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    all_tables = pd.concat(tables, ignore_index=True)
    all_tables.to_csv(BASE / 'diagnostico_umbrales_fijos.csv', index=False)
    text = ('## Por qué la matriz tiene ceros en la columna positiva\n\n'
            + pd.DataFrame(ranges).to_markdown(index=False)
            + '\n\nLa matriz y la media de probabilidades por paciente se han recalculado y coinciden '
              'con los ficheros originales. Todas las probabilidades están por debajo de 0,5: '
              'con ese umbral todas las pacientes se clasifican como 0. Esto no prueba por sí solo '
              'un error de software ni mala calibración; con prevalencia cercana al 29 % y señal '
              'limitada puede ocurrir incluso con probabilidades razonables.\n\n'
              'Las matrices siguientes usan 0,5 y dos umbrales fijos que ya figuraban en el '
              'diagnóstico previo. Son ilustrativas sobre validación usada para seleccionar época. '
              '**No son una mejora validada de forma independiente.** La versión 0,325 '
              'se identifica aparte como exploratoria. Bajar el umbral '
              'permite detectar positivas, pero también genera falsos positivos y no aumenta el AUC. '
              'La escala cambia entre folds, por lo que un único 0,325 no arregla todas las matrices.\n\n'
              '**Solución que debe evaluarse:** mantener E57, reservar validación interna por paciente '
              'dentro de cada train, elegir allí checkpoint y umbral con una regla prefijada '
              '(F1 macro; desempate por balanced accuracy y proximidad a 0,5), y evaluar ambos '
              'umbrales en el fold exterior sin reajustarlos. La calibración es una evaluación '
              'aparte; no es necesaria para elegir un umbral ni garantiza que aparezcan probabilidades '
              'superiores a 0,5. Si persiste poca separación, comparar BCE normal con ponderada '
              'N0/N1 calculada solo en FIT, manteniendo la CNN. No se inicia entrenamiento.\n\n'
              'La ponderación moderada E57 ya probada también daba todas negativas a 0,5. '
              'El contraste anterior con selección interna alcanzó AUC exterior media 0,5272 '
              'con otra receta; tampoco podemos prometer que un nuevo umbral resuelva su '
              'discriminación. Test sigue reservado.\n')
    (BASE / 'DIAGNOSTICO_MATRIZ.md').write_text('# E57 · decisión y matriz de confusión\n\n' + text
        + '\n\n### Todos los umbrales fijos explorados\n\n' + all_tables.to_markdown(index=False) + '\n')
    heading = nbformat.v4.new_markdown_cell(text)
    heading.metadata['campaign_overview'] = 'E57_estabilidad_diagnostico_decision'
    image = image_cell(picture, 'Matrices con umbrales fijos: comparación descriptiva')
    image.metadata['campaign_overview'] = 'E57_estabilidad_diagnostico_matrices'
    return [heading, image]


def requested_threshold_cells(items):
    """Vista explícita; conserva los runs de 0,5."""
    from src.evaluation.metrics import binary_metrics
    config_path = BASE / 'decision_0325.json'
    if not config_path.exists() or not items:
        return []
    config = json.loads(config_path.read_text())
    if config.get('threshold') != .325 or config.get('selection_status') != 'human_requested_posthoc':
        raise ValueError('Configuración de decisión inesperada')
    target = BASE / 'decision_umbral_0325'
    target.mkdir(exist_ok=True)
    rows, all_predictions = [], []
    fig, axes = plt.subplots(1, len(items), figsize=(3.3 * len(items), 3.8), squeeze=False)
    for col, (fold, folder, summary, _) in enumerate(items):
        frame = pd.read_csv(folder / 'predictions_patient.csv')
        metrics = binary_metrics(frame.target, frame.probability, .325)
        matrix = metrics['confusion_matrix']
        tn, fp, fn, tp = [matrix[k] for k in ('tn', 'fp', 'fn', 'tp')]
        f0 = 2 * tn / (2 * tn + fp + fn) if 2 * tn + fp + fn else 0.
        metrics['f1_macro'] = (f0 + metrics['f1']) / 2
        rows.append({'Fold': fold, 'AUC': metrics['roc_auc'], 'AP': metrics['pr_auc'],
                     'Precisión pCR': metrics['precision'], 'Sensibilidad': metrics['sensitivity'],
                     'Especificidad': metrics['specificity'], 'BA': metrics['balanced_accuracy'],
                     'F1 macro': metrics['f1_macro'], 'TN': tn, 'FP': fp, 'FN': fn, 'TP': tp})
        out = target / f'fold_{fold}'
        out.mkdir(exist_ok=True)
        report = dict(schema='e57_posthoc_decision_v1', threshold=.325,
                      selection_status='human_requested_posthoc',
                      independent_evaluation=False, test_evaluated=False,
                      source=str(folder.relative_to(ROOT)),
                      checkpoint_epoch=summary['best_epoch'], metrics=metrics)
        (out / 'metrics.json').write_text(json.dumps(report, indent=2, ensure_ascii=False))
        frame['prediction'] = (frame.probability >= .325).astype(int)
        frame['threshold'] = .325
        frame['fold'] = fold
        frame.to_csv(out / 'predictions_patient.csv', index=False)
        all_predictions.append(frame)
        ax = axes[0, col]
        cm = np.array([[tn, fp], [fn, tp]])
        ax.imshow(cm, cmap='Blues', vmin=0, vmax=170)
        for a in range(2):
            for b in range(2):
                ax.text(b, a, str(cm[a, b]), ha='center', va='center', fontsize=15,
                        color='white' if cm[a, b] > 90 else 'black')
        ax.set(title=f'Fold {fold} · umbral 0,325', xlabel='Predicho (0 / 1)',
               ylabel='Real (0 / 1)', xticks=[0, 1], yticks=[0, 1])
    fig.suptitle('E57 · umbral 0,325 solicitado después de ver validación · exploratorio', fontsize=14)
    fig.tight_layout()
    picture = target / 'matrices_0325.png'
    fig.savefig(picture, dpi=140, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    table = pd.DataFrame(rows)
    table.to_csv(target / 'metricas_por_fold.csv', index=False)
    combined = pd.concat(all_predictions, ignore_index=True)
    aggregate = binary_metrics(combined.target, combined.probability, .325)
    # Esta agregación se limita a decisiones; no presentamos un AUC pooled de modelos distintos.
    for key in ('roc_auc', 'pr_auc', 'brier_score'):
        aggregate.pop(key, None)
    (target / 'conteos_y_decisiones_agregadas.json').write_text(json.dumps(aggregate, indent=2))
    c = aggregate['confusion_matrix']
    message = ('## E57 · versión con umbral 0,325\n\n'
        + table.round(4).to_markdown(index=False)
        + f'\n\nConteos de los cinco folds: **TN {c["tn"]}, FP {c["fp"]}, FN {c["fn"]}, TP {c["tp"]}**. '
        + f'Sensibilidad {aggregate["sensitivity"]:.1%}; especificidad {aggregate["specificity"]:.1%}; '
        + f'precisión pCR {aggregate["precision"]:.1%}; balanced accuracy {aggregate["balanced_accuracy"]:.4f}.\n\n'
        + '**Cambiar el umbral es válido y no requiere reentrenar.** Aquí 0,325 se ha elegido después '
          'de ver estos resultados: esta tabla es exploratoria y no una estimación independiente '
          'del rendimiento futuro. La comparación conserva la matriz original a 0,5 '
          'y declara cómo se eligió la nueva. Para estimar generalización, fijar el procedimiento '
          'de selección usando solo train/validación interna y evaluar en pacientes distintas. '
          'El AUC, AP, probabilidades, pesos y curvas por época son los mismos.\n\n'
          'Clasificar más positivas también incrementa los falsos positivos. Ningún cambio '
          'de umbral puede crear separación entre clases que el modelo no ha aprendido.\n')
    (target / 'README.md').write_text(message)
    heading = nbformat.v4.new_markdown_cell(message)
    heading.metadata['campaign_overview'] = 'E57_estabilidad_umbral_0325_resumen'
    image = image_cell(picture, 'E57 · matrices con umbral 0,325; vista exploratoria')
    image.metadata['campaign_overview'] = 'E57_estabilidad_umbral_0325_matrices'
    return [heading, image]


def build():
    items = completed()
    rows = []
    for fold, folder, summary, history in items:
        m = summary['validation_metrics']['patient']; c = m['confusion_matrix']
        rows.append({'Fold': fold, 'Mejor época': summary['best_epoch'],
                     'Épocas': summary['epochs_completed'],
                     'AUC val': round(m['roc_auc'], 4), 'AP val': round(m['pr_auc'], 4),
                     'AUC final': round(float(history.patient_roc_auc.iloc[-1]), 4),
                     'TN': c['tn'], 'FP': c['fp'], 'FN': c['fn'], 'TP': c['tp']})
    table = pd.DataFrame(rows).to_markdown(index=False) if rows else 'Sin científicos completos.'
    note = ('Los máximos se eligieron usando la validación del propio fold; su media es una '
            'descripción optimista, no una estimación exterior independiente. El fold 0 es '
            'histórico y los folds 1–4 se ejecutaron después con la misma receta. '
            'El umbral se mantuvo fijo en 0,5.')
    if len(items) == 5:
        values = [s['validation_metrics']['patient']['roc_auc'] for _, _, s, _ in items]
        note += (f' Media de los cinco máximos: {np.mean(values):.4f}; '
                 f'DE entre folds: {np.std(values, ddof=1):.4f}.')
    nb = nbformat.v4.new_notebook()
    nb.metadata.kernelspec = {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}
    nb.cells.append(nbformat.v4.new_markdown_cell(
        '# E57 · resultados visibles sin ejecutar celdas\n\n'
        + table + '\n\n' + note + '\n\n'
        + '[Protocolo](README.md) · [Notebook principal](../RESULTADOS.ipynb).'))
    requested = requested_threshold_cells(items)
    nb.cells.extend(copy.deepcopy(requested))
    overview = panel(items)
    if overview:
        nb.cells.append(nbformat.v4.new_markdown_cell('## Comparación directa de los cuatro folds nuevos'))
        nb.cells.append(image_cell(overview, 'Panel de BCE y AUC comparables por fold'))
    details = []
    for fold, folder, summary, history in items:
        details.extend(detail_cells(fold, folder, summary))
    nb.cells.extend(copy.deepcopy(details))
    diagnostics = decision_audit_cells(items) if items else []
    nb.cells.extend(copy.deepcopy(diagnostics))
    target = BASE / 'RESULTADOS.ipynb'
    temporary = BASE / 'RESULTADOS.rebuild.tmp.ipynb'
    nbformat.write(nb, temporary)
    temporary.replace(target)
    # La tabla y el panel también quedan visibles al abrir el notebook principal.
    main = ROOT / 'experiments/RESULTADOS.ipynb'
    if main.is_file():
        original = nbformat.read(main, as_version=4)
        directories = [folder.relative_to(ROOT).as_posix() for _, folder, _, _ in items]
        remove = set()
        for index, c in enumerate(original.cells):
            marker = c.metadata.get('campaign_overview', '')
            if marker.startswith('E57_estabilidad_'):
                remove.add(index)
            elif c.cell_type == 'code' and any(f'display_entry({directory!r},root)' in c.source
                                               for directory in directories):
                remove.add(index)
                if index and original.cells[index - 1].cell_type == 'markdown':
                    remove.add(index - 1)
        original.cells = [c for index, c in enumerate(original.cells) if index not in remove]
        overview_cell = nbformat.v4.new_markdown_cell(
            '# E57 · estabilidad en cinco folds\n\n' + table + '\n\n' + note
            + '\n\n**Debajo están los cinco folds completos: todas las épocas, seis paneles, matrices y curvas comparables.** '
              '[Conclusiones](E57_estabilidad/CONCLUSIONES.md).')
        overview_cell.metadata['campaign_overview'] = 'E57_estabilidad_resumen'
        original.cells.insert(1, overview_cell)
        if overview:
            figure_cell = image_cell(overview, 'E57 · panel de curvas comparables, folds 1–4')
            figure_cell.metadata['campaign_overview'] = 'E57_estabilidad_panel'
            original.cells.insert(2, figure_cell)
        insertion = 3 if overview else 2
        original.cells[insertion:insertion] = copy.deepcopy(requested + details + diagnostics)
        main_temp = main.with_name('RESULTADOS.e57.tmp.ipynb')
        nbformat.write(original, main_temp)
        main_temp.replace(main)
    return target


if __name__ == '__main__':
    print(build())
