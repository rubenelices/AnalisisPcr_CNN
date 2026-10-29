"""Cifras de la web calculadas desde predicciones guardadas (datos sintéticos, sin tocar reports/)."""
import json

import pandas as pd

from src.web.web_resultados import matriz_conjunta


def _fold(raiz, k, umbral, filas):
    d = raiz/f'normal_fold_{k}_seed_42'
    d.mkdir(parents=True)
    (d/'summary.json').write_text(json.dumps({'threshold': umbral}))
    pd.DataFrame(filas, columns=['patient_id', 'probability', 'target', 'cuts']).to_csv(d/'predictions_outer_patient.csv', index=False)


def test_matriz_conjunta_usa_el_umbral_de_cada_fold(tmp_path):
    # Con umbral 0,3 la paciente a (0,4) es positiva; con 0,5 en el otro fold, la d (0,4) no lo es.
    _fold(tmp_path, 0, 0.3, [('a', 0.4, 0, 3), ('b', 0.2, 0, 3), ('c', 0.9, 1, 3)])
    _fold(tmp_path, 1, 0.5, [('d', 0.4, 1, 3), ('e', 0.6, 1, 3), ('f', 0.1, 0, 3)])
    m = matriz_conjunta(tmp_path, folds=2)
    assert (m['tn'], m['fp'], m['fn'], m['tp']) == (2, 1, 1, 2)
    assert m['pacientes'] == 6 and m['pcr'] == 3
    assert m['sensibilidad'] == 2/3 and m['especificidad'] == 2/3


def test_matriz_conjunta_rechaza_paciente_repetida(tmp_path):
    import pytest
    _fold(tmp_path, 0, 0.5, [('a', 0.4, 0, 3)])
    _fold(tmp_path, 1, 0.5, [('a', 0.6, 1, 3)])
    with pytest.raises(AssertionError):
        matriz_conjunta(tmp_path, folds=2)


def test_experimentos_se_leen_del_historial(tmp_path):
    from src.web.web_resultados import _experimentos
    md = tmp_path/'h.md'
    md.write_text(
        '| experimento | entradas | fold | seed | epocas | mejor_epoca | auc_val_max | auc_val_final | auc_train_final | descripcion |\n'
        '|:--|:--|:--|--:|--:|--:|--:|--:|--:|:--|\n'
        '| A | imagen | 0 | 42 | 10 | 3 | 0.6000 | 0.5000 | 0.7000 | x |\n'
        '| B | imagen+clínica |  | 42 | 40 | 7 | 0.7000 | 0.6500 | nan | y |\n')
    a, b = _experimentos(md, [('A', 'Uno', 'c1'), ('B', 'Dos', 'c2')])
    assert (a['pico'], a['final'], a['fit'], a['clinica'], a['fold']) == (0.6, 0.5, 0.7, False, '0')
    assert round(a['brecha'], 4) == 0.2 and round(a['caida'], 4) == 0.1
    assert b['clinica'] and b['fit'] is None and b['brecha'] is None and b['fold'] is None


def test_particiones_cuentan_pacientes_y_cortes_sin_etiquetas_de_test(tmp_path):
    from src.web.web_resultados import _particiones
    pd.DataFrame([
        ('a_z1', 'a', 'train', 1, 0), ('a_z2', 'a', 'train', 1, 0), ('b_z1', 'b', 'train', 0, 0),
        ('c_z1', 'c', 'train', 0, 1), ('t_z1', 't', 'test', 1, -1), ('t_z2', 't', 'test', 1, -1),
    ], columns=['sample_id', 'patient_id', 'split', 'pCR', 'fold']).to_csv(tmp_path/'samples.csv', index=False)
    pd.DataFrame([('a', 1, 'spy1', 'train'), ('b', 0, 'spy1', 'train'), ('c', 0, 'duke', 'train'), ('t', 1, 'duke', 'test')],
                 columns=['pid', 'pCR', 'dataset', 'split']).to_csv(tmp_path/'patients.csv', index=False)
    r = _particiones(tmp_path)
    assert r['folds'] == [
        dict(fold=0, pacientes=2, pcr=1, cortes=3, cortes_pcr=2, cohortes={'spy1': dict(pacientes=2, pcr=1)}),
        dict(fold=1, pacientes=1, pcr=0, cortes=1, cortes_pcr=0, cohortes={'duke': dict(pacientes=1, pcr=0)})]
    assert r['test'] == dict(pacientes=1, cortes=2)
    assert r['cohortes']['spy1'] == dict(train=2, pcr_train=0.5, test=0) and r['cohortes']['duke']['test'] == 1
    assert 'pcr' not in r['test']


def test_metricas_macro_promedian_las_dos_clases():
    from src.web.web_resultados import metricas_macro
    m = metricas_macro(dict(tn=92, fp=31, fn=35, tp=18))
    assert round(m['precision_pcr'], 4) == 0.3673 and round(m['recall_pcr'], 4) == 0.3396
    assert round(m['precision_macro'], 3) == 0.546 and round(m['recall_macro'], 4) == 0.5438
    assert round(m['f1_macro'], 4) == 0.5445 and round(m['f1_pcr'], 4) == 0.3529
