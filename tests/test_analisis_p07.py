import json

import numpy as np
import pandas as pd
import pytest

from src.analisis_p07 import SEMILLAS, analizar


def _pacientes(n=300, seed=0):
    rng = np.random.default_rng(seed)
    pac = pd.DataFrame({'patient_id': [f'P{i:03d}' for i in range(n)], 'fold': np.arange(n) % 5,
                        'cohorte': rng.choice(['spy1', 'spy2', 'duke'], n)})
    pac['target'] = (rng.random(n) < .3).astype(int)
    return pac


def _escribir(tmp, pac, ruido=1.5):
    """Cada semilla = señal común + ruido propio: el promedio debe ordenar mejor."""
    rng = np.random.default_rng(1)
    for s in SEMILLAS:
        for k in range(5):
            d = tmp/f'normal_fold_{k}_seed_{s}'
            d.mkdir(parents=True)
            ext = pac[pac.fold == k]
            interna = pac[pac.fold == (k+1) % 5]
            for nombre, p in (('outer', ext), ('inner', interna)):
                prob = 1/(1+np.exp(-((p.target-.5) + rng.normal(0, ruido, len(p)))))
                f = pd.DataFrame({'patient_id': p.patient_id.values, 'probability': prob, 'target': p.target.values})
                f.to_csv(d/f'predictions_{nombre}_patient.csv', index=False)
                if nombre == 'outer':
                    f.assign(sample_id=f.patient_id + '_z1').to_csv(d/'predictions_outer_cut.csv', index=False)
            (d/'summary.json').write_text(json.dumps({'status': 'complete'}))
    return tmp


def test_ensemble_promedia_y_mejora_con_ruido_independiente(tmp_path):
    pac = _pacientes(1000)
    salida, df = analizar(_escribir(tmp_path, pac), pac)
    assert np.allclose(df.ensemble, df[['semilla_42', 'semilla_17', 'semilla_123']].mean(axis=1))
    assert salida['auc_conjunta']['ensemble'] > salida['media_semillas_individuales']
    m = salida['matriz_conjunta_umbral_interno']
    assert m['tn'] + m['fp'] + m['fn'] + m['tp'] == len(pac)
    assert len(salida['umbrales_internos_ensemble']) == 5


def test_rechaza_semilla_incompleta(tmp_path):
    pac = _pacientes()
    carpeta = _escribir(tmp_path, pac)
    (carpeta/'normal_fold_4_seed_123/summary.json').write_text(json.dumps({'status': 'failed'}))
    with pytest.raises(RuntimeError, match='incompleto'):
        analizar(carpeta, pac)
