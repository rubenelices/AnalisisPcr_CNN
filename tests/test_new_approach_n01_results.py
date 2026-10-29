"""La vista N01 muestra parciales sin inventar exterior y conserva históricos."""
import json

import nbformat
import pandas as pd

from src.analysis import new_approach_n01_results as reporting
from src.analysis.new_approach_n01_results import build
from src.training.new_approach_protocol import TECH, metrics


def test_partial_report_keeps_reference_and_notes_without_external_metrics(tmp_path):
    dest = tmp_path/'research'
    reference = tmp_path/'experiments/E57_patient_feature_mean_mac/results'
    dest.mkdir(); reference.mkdir(parents=True)
    (dest/'ESTADO.md').write_text('Preparado, sin nuevos resultados completos')
    old = nbformat.v4.new_markdown_cell('Antecedente histórico que no se puede perder')
    nbformat.write(nbformat.v4.new_notebook(cells=[old]), reference/'RESULTADOS.ipynb')
    run = tmp_path/TECH/'runs/normal_fold_0_seed_42'
    run.mkdir(parents=True)
    summary = dict(schema='new_approach_n01_run_v1', scientific_result=True, loss='normal',
                   fold=0, status='interrupted', epochs_completed=1, best_epoch=1, pos_weight=1.,
                   counts={n: {'patients': 4} for n in ('fit', 'inner', 'outer')}, error='Interrupción')
    (run/'summary.json').write_text(json.dumps(summary))
    pd.DataFrame([dict(epoch=1, train_online_objective=.7, fit_bce=.6, inner_bce=.65,
                       fit_auc=.7, inner_auc=.6, inner_ap=.4, learning_rate=.001, seconds=1)]).to_csv(run/'history.csv', index=False)
    path = build(tmp_path)
    nb = nbformat.read(path, 4)
    note = nbformat.v4.new_markdown_cell('Nota manual del estudiante')
    nb.cells.insert(3, note)
    nbformat.write(nb, path)
    build(tmp_path)
    result = nbformat.read(path, 4)
    text = '\n'.join(c.source for c in result.cells if c.cell_type == 'markdown')
    assert 'Interrumpido' in text and 'Todas las épocas' in text
    assert 'Nota manual del estudiante' in text and old.source in text
    assert 'Exterior del checkpoint' not in text
    assert sum(c.source.startswith('### N01 · normal') for c in result.cells) == 1
    assert not (run/'exterior_05.png').exists()
    assert any(c.cell_type == 'code' and c.outputs for c in result.cells)
    assert nbformat.read(reference/'RESULTADOS.ipynb', 4).cells == [old]


def test_complete_pairs_report_both_matrices_summary_and_conditional_intervals(tmp_path, monkeypatch):
    dest = tmp_path/'research'; dest.mkdir()
    reference = tmp_path/'experiments/E57_patient_feature_mean_mac/results'; reference.mkdir(parents=True)
    (dest/'ESTADO.md').write_text('Estado de prueba sintética')
    nbformat.write(nbformat.v4.new_notebook(), reference/'RESULTADOS.ipynb')
    (tmp_path/'src/analysis').mkdir(parents=True)
    (tmp_path/'src/analysis/new_approach_paired.py').write_text('Marcador de implementación sintético para SHA')
    actual_bootstrap = reporting.paired_intervals
    monkeypatch.setattr(reporting, 'paired_intervals', lambda pairs: actual_bootstrap(pairs, repetitions=5))
    # No hace falta renderizar veinte figuras para verificar ensamblaje/tablas/intervalos.
    # El test parcial anterior sí renderiza e incrusta una figura real.
    def fake_figure(patients, summary, config, dest, **kwargs): dest.write_bytes(b'fake-png-for-notebook-test')
    monkeypatch.setattr(reporting, 'save_evaluation_figure', fake_figure)
    for fold in range(5):
        for loss in ('normal', 'weighted'):
            folder = tmp_path/TECH/'runs'/f'{loss}_fold_{fold}_seed_42'; folder.mkdir(parents=True)
            p = pd.DataFrame(dict(patient_id=[f'{fold}_{i}' for i in range(4)],
                                  target=[0, 0, 1, 1], probability=[.1, .4, .3, .8]))
            p.to_csv(folder/'predictions_outer_patient.csv', index=False)
            decision = dict(threshold=.35, best_epoch=1)
            (folder/'decision_before_outer.json').write_text(json.dumps(decision))
            m = metrics(p, .5)
            summary = dict(schema='new_approach_n01_run_v1', scientific_result=True, loss=loss,
                           fold=fold, status='complete', epochs_completed=1, best_epoch=1, pos_weight=1.,
                           counts={n: {'patients': 4, 'positive': 2} for n in ('fit', 'inner', 'outer')},
                           protocol_sha256='same', initial_state_sha256=f'same-initial-fold-{fold}',
                           threshold=.35, decision=decision, outer_at_05={'metrics': m},
                           outer_at_inner_threshold=metrics(p, .35))
            (folder/'summary.json').write_text(json.dumps(summary))
    path = build(tmp_path)
    result = nbformat.read(path, 4)
    text = '\n'.join(c.source for c in result.cells if c.cell_type == 'markdown')
    assert 'Comparación completa de los cinco folds' in text
    assert 'IC95 inferior' in text and 'Condicional' in text
    assert text.count('Exterior del checkpoint') == 10
    assert (tmp_path/TECH/'analysis/paired_intervals.json').is_file()
    assert len([c for c in result.cells if c.cell_type == 'code']) == 20
    # La segunda actualización reutiliza los intervalos guardados, no recalcula bootstrap.
    monkeypatch.setattr(reporting, 'paired_intervals', lambda pairs: (_ for _ in ()).throw(AssertionError('Cache no usada')))
    build(tmp_path)
