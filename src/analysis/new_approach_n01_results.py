"""Registro de N01 en el notebook activo, solo a partir de salidas guardadas."""
import base64
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nbformat
import numpy as np
import pandas as pd

from src.evaluation.evaluation import EvaluationConfig, save_evaluation_figure
from src.training.new_approach_protocol import ROOT, TECH, metrics, sha, write_json
from src.analysis.new_approach_paired import paired_intervals
from src.analysis.new_approach_results import build as build_active, write_notebook

OWNER = 'n01_generated'


def md(source):
    cell = nbformat.v4.new_markdown_cell(source)
    cell.metadata[OWNER] = True
    cell.metadata['new_approach_run'] = 'N01'
    return cell


def picture(path, root):
    rel = path.relative_to(root).as_posix()
    cell = nbformat.v4.new_code_cell(
        'from pathlib import Path\nfrom IPython.display import display, Image\n'
        'root=next(p for p in (Path.cwd(), *Path.cwd().parents) if (p/"src").is_dir() and (p/"app.py").is_file())\n'
        f'display(Image(filename=str(root / {rel!r})))')
    cell.metadata.update({OWNER: True, 'new_approach_run': 'N01',
                          'tags': ['hide-input'], 'jupyter': {'source_hidden': True}})
    cell.outputs = [nbformat.v4.new_output('display_data', data={
        'image/png': base64.b64encode(path.read_bytes()).decode(),
        'text/plain': f'N01 · {path.stem}'})]
    return cell


def curves(history, summary, target):
    h = history
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    for column, label, color in (('fit_bce', 'FIT eval', '#2f75b5'),
                                  ('inner_bce', 'Interna eval', '#ed7134')):
        axes[0].plot(h.epoch, h[column], label=label, color=color)
    axes[0].set(title='BCE normal por paciente · mismo modo', ylabel='BCE normal')
    axes[1].plot(h.epoch, h.fit_auc, label='FIT eval', color='#2f75b5')
    axes[1].plot(h.epoch, h.inner_auc, label='Interna eval', color='#ed7134')
    axes[1].set(title='AUC por paciente · solo FIT e interna', ylabel='ROC-AUC', ylim=(0, 1.02))
    for a in axes[:2]:
        a.axvline(summary['best_epoch'], ls='--', color='.45', label='Checkpoint interno')
        a.set_xlabel('Época'); a.legend(fontsize=8); a.grid(alpha=.2)
    if summary['status'] == 'complete':
        values = [summary[n]['metrics']['roc_auc'] for n in ('fit_at_best', 'inner_at_best', 'outer_at_05')]
        axes[2].bar(['FIT', 'Interna', 'Exterior'], values, color=['#2f75b5', '#ed7134', '#424e57'])
        for i, value in enumerate(values):
            axes[2].text(i, value+.02, f'{value:.3f}', ha='center')
        axes[2].set(title=f'Mismo checkpoint · época {summary["best_epoch"]}', ylabel='ROC-AUC', ylim=(0, 1.08))
        axes[2].axhline(.5, color='.5', ls=':'); axes[2].grid(axis='y', alpha=.2)
    else:
        axes[2].axis('off')
        axes[2].text(.5, .5, 'Exterior pendiente\nNo se consulta durante las épocas', ha='center', va='center')
    fig.suptitle(f'N01 · {summary["loss"]} · fold {summary["fold"]} · desarrollo, no test')
    fig.tight_layout(); fig.savefig(target, dpi=140, bbox_inches='tight'); plt.close(fig)


def metric_table(summary):
    rows = []
    for label, m in [('0,5 fijo', summary['outer_at_05']['metrics']),
                     ('Seleccionado en interna', summary['outer_at_inner_threshold'])]:
        rows.append(dict(Decisión=label, Umbral=m['threshold'], AUC=m['roc_auc'], AP=m['pr_auc'],
                         Brier=m['brier_score'], Sensibilidad=m['sensitivity'],
                         Especificidad=m['specificity'],
                         Precisión=m['precision'] if m['precision_defined'] else 'Indefinida',
                         F1_pCR=m['f1'], F1_no_pCR=m['f1_negative'],
                         F1_macro=m['f1_macro'], BA=m['balanced_accuracy'],
                         **{k.upper(): v for k, v in m['confusion_matrix'].items()}))
    return pd.DataFrame(rows).round(5).to_markdown(index=False)


def cut_curves(history, summary, target):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    axes[0].plot(history.epoch, history.fit_cut_bce, label='FIT eval', color='#2f75b5')
    axes[0].plot(history.epoch, history.inner_cut_bce, label='Interna eval', color='#ed7134')
    axes[0].set(title='BCE normal por corte · mismo modo/unidad', ylabel='BCE por corte')
    axes[0].legend()
    axes[1].plot(history.epoch, history.train_online_objective, color='#2f75b5')
    axes[1].set(title=f'Objetivo online · {summary["loss"]} · features/paciente', ylabel='Pérdida de entrenamiento')
    for a in axes:
        a.axvline(summary['best_epoch'], ls='--', color='.45')
        a.set_xlabel('Época'); a.grid(alpha=.2)
    fig.suptitle('Unidades separadas: estas dos gráficas no miden la misma pérdida')
    fig.tight_layout(); fig.savefig(target, dpi=140, bbox_inches='tight'); plt.close(fig)


def build(root=ROOT):
    root = Path(root)
    path = build_active(root)
    nb = nbformat.read(path, 4)
    nb.cells = [c for c in nb.cells if not c.metadata.get(OWNER)]
    records = []
    for p in sorted((root/TECH/'runs').glob('*/summary.json')):
        s = json.loads(p.read_text())
        if s.get('schema') != 'new_approach_n01_run_v1' or not s.get('scientific_result'):
            raise ValueError(f'Salida científica N01 inválida: {p}')
        records.append((p.parent, s))
    cells = [md('### N01 · comparación pareada de pérdidas\n\n'
                '[Protocolo completo](PROTOCOLO_N01.md) · [Ejecución manual](EJECUTAR_N01.md).\n\n'
                'Misma E57, solo imágenes. La interna elige época y umbral; '
                'el exterior se evalúa una vez al final de cada condición/fold. '
                'No se evalúa test. Los smokes técnicos no se incluyen como resultados científicos.')]
    manifest_path = root/TECH/'protocol/manifest.json'
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        split_rows = []
        for fold, parts in manifest['splits'].items():
            fit, inner, outer = [parts[n] for n in ('fit', 'inner', 'outer')]
            split_rows.append(dict(Fold=int(fold), Pacientes_FIT=fit['patients'],
                                   pCR_FIT=fit['positive'], Pacientes_interna=inner['patients'],
                                   pCR_interna=inner['positive'], Pacientes_exterior=outer['patients'],
                                   pCR_exterior=outer['positive'],
                                   Peso_positivo_FIT=fit['negative']/fit['positive']))
        cells.append(md('#### Particiones congeladas antes de entrenar\n\n'
                        + pd.DataFrame(split_rows).round(6).to_markdown(index=False)
                        + '\n\nTodas las particiones usan pacientes distintas dentro de cada fold. '
                        'Cada paciente pertenece a un único exterior entre los cinco folds.'))
    if not records:
        cells.append(md('**Preparado; todavía sin entrenamientos científicos.** '
                        'Las particiones se verifican con metadatos y los controles de software con datos sintéticos.'))
    complete = []
    for folder, s in records:
        key = f'{s["loss"]} · fold {s["fold"]}'
        status = {'complete': 'Completo', 'running': 'En ejecución',
                  'interrupted': 'Interrumpido', 'failed': 'Fallido'}.get(s['status'], s['status'])
        cells.append(md(f'### N01 · {key} · {status}\n\n'
                        f'Épocas completas: **{s["epochs_completed"]}**. Peso positivo: **{s["pos_weight"]:.6f}**. '
                        f'Pacientes FIT/interna/exterior: **{s["counts"]["fit"]["patients"]}/'
                        f'{s["counts"]["inner"]["patients"]}/{s["counts"]["outer"]["patients"]}**.\n\n'
                        + (f'Incidencia: {s["error"]}\n\n' if 'error' in s else '')
                        + 'La pérdida online es el objetivo de entrenamiento de esta condición. '
                          'Las BCE FIT/interna son normales y comparables, medidas en eval por paciente.'))
        hpath = folder/'history.csv'
        if hpath.is_file():
            h = pd.read_csv(hpath)
            readable = h.rename(columns=dict(epoch='Época', train_online_objective='Objetivo train online',
                                             fit_bce='BCE paciente FIT', inner_bce='BCE paciente interna',
                                             fit_auc='AUC paciente FIT', inner_auc='AUC paciente interna',
                                             fit_cut_bce='BCE corte FIT', inner_cut_bce='BCE corte interna',
                                             inner_ap='AP paciente interna', learning_rate='LR', seconds='Segundos'))
            cells.append(md('#### Todas las épocas\n\n' + readable.round(5).to_markdown(index=False)))
            curves(h, s, folder/'curvas_comparables.png')
            cells.append(picture(folder/'curvas_comparables.png', root))
            if {'fit_cut_bce', 'inner_cut_bce'}.issubset(h.columns):
                cut_curves(h, s, folder/'curvas_por_corte_y_online.png')
                cells.append(picture(folder/'curvas_por_corte_y_online.png', root))
        if s['status'] != 'complete':
            continue
        complete.append(s)
        decision = json.loads((folder/'decision_before_outer.json').read_text())
        if decision != s['decision'] or decision['threshold'] != s['threshold']:
            raise ValueError('Decisión final y resumen discrepantes')
        patients = pd.read_csv(folder/'predictions_outer_patient.csv')
        for label, threshold in [('05', .5), ('inner', s['threshold'])]:
            m = metrics(patients, threshold)
            expected = s['outer_at_05']['metrics'] if label == '05' else s['outer_at_inner_threshold']
            if any(not np.isclose(m[k], expected[k], atol=1e-10) for k in ('roc_auc', 'f1_macro', 'brier_score')):
                raise ValueError('Las predicciones guardadas no reproducen el resumen')
            summary = dict(patient_metrics_primary=m, calibration={'brier_score': m['brier_score']})
            dest = folder/f'exterior_{label}.png'
            save_evaluation_figure(patients, summary, EvaluationConfig(threshold=threshold), dest,
                                   title=f'N01 · {key} · exterior de desarrollo · no test')
        cells.append(md(f'#### Exterior del checkpoint de época {s["best_epoch"]}\n\n'
                        f'Umbral interno congelado: **{s["threshold"]:.6f}**.\n\n' + metric_table(s)))
        cells.extend([picture(folder/'exterior_05.png', root), picture(folder/'exterior_inner.png', root)])
        baseline = patients.copy(); baseline['probability'] = 0.
        b = metrics(baseline, .5)
        prior = patients.copy()
        prior['probability'] = s['counts']['fit']['positive']/s['counts']['fit']['patients']
        prior_brier = metrics(prior, .5)['brier_score']
        cells.append(md(f'Clasificador siempre negativo en este exterior: BA **{b["balanced_accuracy"]:.3f}**, '
                        f'F1 macro **{b["f1_macro"]:.3f}**, sensibilidad **0**; precisión pCR indefinida.'))
        cells.append(md(f'Brier del predictor constante con prevalencia de FIT: **{prior_brier:.5f}**. '
                        'Sirve como referencia de probabilidades sin información de imágenes; no usa la prevalencia exterior para predecir.'))
    if complete:
        rows = []
        for s in complete:
            m = s['outer_at_inner_threshold']
            rows.append(dict(Pérdida=s['loss'], Fold=s['fold'], Épocas=s['epochs_completed'],
                             Elegida=s['best_epoch'], Umbral=s['threshold'], AUC=m['roc_auc'],
                             AP=m['pr_auc'], Brier=m['brier_score'], F1_macro=m['f1_macro'],
                             BA=m['balanced_accuracy']))
        frame = pd.DataFrame(rows)
        cells.insert(1, md('#### Ejecuciones completas · no seleccionar por un pico\n\n'
                           + frame.round(5).to_markdown(index=False)))
        expected = {(loss, fold) for loss in ('normal', 'weighted') for fold in range(5)}
        if {(s['loss'], s['fold']) for s in complete} == expected:
            by_key = {(s['loss'], s['fold']): s for s in complete}
            for fold in range(5):
                a, b = by_key['normal', fold], by_key['weighted', fold]
                if (a['protocol_sha256'] != b['protocol_sha256']
                        or a['initial_state_sha256'] != b['initial_state_sha256']
                        or a['counts'] != b['counts']):
                    raise ValueError('No comparar condiciones con protocolo, inicialización o particiones diferentes')
            measures = ['AUC', 'AP', 'Brier', 'F1_macro', 'BA']
            stats = frame.groupby('Pérdida')[measures].agg(['mean', 'std']).round(5)
            stats.columns = [f'{metric} {"media" if stat == "mean" else "DE"}' for metric, stat in stats.columns]
            diffs = frame.set_index(['Pérdida', 'Fold']).loc['weighted', measures] - frame.set_index(['Pérdida', 'Fold']).loc['normal', measures]
            cells.append(md('### Comparación completa de los cinco folds\n\n'
                            'Media y desviación entre folds (no entre épocas):\n\n' + stats.to_markdown()
                            + '\n\nDiferencias ponderada−normal por el mismo fold. En Brier, menor es mejor:\n\n'
                            + diffs.round(5).to_markdown()
                            + '\n\nEvaluación de desarrollo con arquitectura elegida previamente; no test nuevo.'))
            pairs, hashes = [], {}
            folders = {(s['loss'], s['fold']): folder for folder, s in records}
            for fold in range(5):
                paths = [folders[loss, fold]/'predictions_outer_patient.csv' for loss in ('normal', 'weighted')]
                hashes.update({str(p.relative_to(root)): sha(p) for p in paths})
                pairs.append((*[pd.read_csv(p) for p in paths],
                              by_key['normal', fold]['threshold'], by_key['weighted', fold]['threshold']))
            analysis = root/TECH/'analysis'
            analysis.mkdir(exist_ok=True)
            interval_path = analysis/'paired_intervals.json'
            cache_key = dict(predictions_sha256=hashes,
                             thresholds=[list(pair[2:]) for pair in pairs],
                             implementation_sha256=sha(root/'src/analysis/new_approach_paired.py'))
            saved = json.loads(interval_path.read_text()) if interval_path.is_file() else None
            if saved is None or saved['inputs'] != cache_key:
                saved = dict(inputs=cache_key, result=paired_intervals(pairs))
                write_json(interval_path, saved)
            intervals = saved['result']
            table = pd.DataFrame(intervals['intervals']).T.round(5)
            table.columns = ['Diferencia ponderada−normal', 'IC95 inferior', 'IC95 superior']
            cells.append(md('#### Incertidumbre de la diferencia\n\n' + table.to_markdown()
                            + '\n\n2.000 remuestreos pareados de pacientes por clase y fold, semilla 42. '
                            'Condicional a los modelos, umbrales y prevalencias observados. '
                            'No incluye la incertidumbre de elegir arquitectura o volver a entrenar. '
                            'En Brier, una diferencia negativa favorece a la ponderada.'))
        else:
            cells.append(md('**Comparación incompleta.** No se presenta una media definitiva de cinco folds '
                            'ni se declara una condición ganadora.'))
    position = next(i+1 for i, c in enumerate(nb.cells)
                    if c.metadata.get('new_approach_generated') == 'new-runs')
    nb.cells[position:position] = cells
    nbformat.validate(nb)
    write_notebook(nb, path)
    return path
