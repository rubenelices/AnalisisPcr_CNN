import json

import numpy as np
import pandas as pd
import pytest

from src.analysis.analisis_p05 import analizar, linea_base_cohorte


def _pacientes(n=300, seed=0):
    rng = np.random.default_rng(seed)
    pac = pd.DataFrame({'patient_id': [f'P{i:03d}' for i in range(n)], 'fold': np.arange(n) % 5,
                        'cohorte': rng.choice(['spy1', 'spy2', 'duke'], n)})
    pac['target'] = (rng.random(n) < pac.cohorte.map({'spy1': .4, 'spy2': .3, 'duke': .2})).astype(int)
    return pac


def _carpeta(tmp, pac, señal, seed, epocas=3):
    rng = np.random.default_rng(seed)
    for k in range(5):
        d = tmp/f'normal_fold_{k}_seed_42'
        d.mkdir(parents=True)
        p = pac[pac.fold == k]
        prob = 1/(1+np.exp(-(señal*(p.target-.5) + rng.normal(0, 1, len(p)))))
        pd.DataFrame({'patient_id': p.patient_id, 'probability': prob, 'target': p.target, 'cuts': 2}) \
            .to_csv(d/'predictions_outer_patient.csv', index=False)
        pd.concat([pd.DataFrame({'patient_id': p.patient_id, 'target': p.target, 'probability': prob})] * 2) \
            .to_csv(d/'predictions_outer_cut.csv', index=False)
        pd.DataFrame({'epoch': range(1, epocas+1), 'inner_auc': [.5, .6 + .01*k, .55]}) \
            .to_csv(d/'history.csv', index=False)
        pd.concat([pd.DataFrame({'patient_id': p.patient_id, 'probability': prob, 'target': p.target, 'epoch': e})
                   for e in range(1, epocas+1)]).to_csv(d/'predictions_outer_patient_by_epoch.csv', index=False)
        m = {'metrics': {'roc_auc': .6}}
        (d/'summary.json').write_text(json.dumps({'status': 'complete', 'best_epoch': 2, 'epochs_completed': epocas,
                                                  'fit_at_best': {'metrics': {'roc_auc': .7}}, 'inner_at_best': m,
                                                  'outer_at_05': m, 'threshold': .3}))
    return tmp


def test_linea_base_usa_solo_otros_folds():
    pac = _pacientes()
    base = linea_base_cohorte(pac)
    k, c = 0, 'duke'
    esperado = pac[(pac.fold != k) & (pac.cohorte == c)].target.mean()
    assert np.allclose(base[(pac.fold == k) & (pac.cohorte == c)], esperado)


def test_analizar_detecta_mejora_y_epoca_fija(tmp_path):
    pac = _pacientes()
    fuerte = _carpeta(tmp_path/'p05', pac, señal=3, seed=1)
    debil = _carpeta(tmp_path/'p02', pac, señal=0, seed=2)
    salida, df = analizar(fuerte, debil, pac)
    assert salida['pacientes'] == len(pac) and len(df) == len(pac)
    assert salida['criterios']['principal_p05_mejora_p02']
    assert salida['secundario_epoca_fija']['epoca'] == 2
    assert salida['auc_por_corte']['p05'] == salida['auc_conjunta']['p05']


def test_analizar_rechaza_folds_incompletos(tmp_path):
    pac = _pacientes()
    a = _carpeta(tmp_path/'a', pac, 1, 1)
    b = _carpeta(tmp_path/'b', pac, 1, 2)
    s = a/'normal_fold_3_seed_42/summary.json'
    s.write_text(json.dumps(dict(json.loads(s.read_text()), status='failed')))
    with pytest.raises(RuntimeError, match='incompleto'):
        analizar(a, b, pac)
