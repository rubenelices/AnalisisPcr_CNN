"""Elección de los cortes de ejemplo de la web con metadatos sintéticos."""

import pandas as pd

from src.web.web_ejemplos import elegir


def _metadatos():
    filas, pacientes = [], []
    for cohorte in ('duke', 'spy1', 'spy2'):
        for k in range(6):
            pid = f'{cohorte}_{k}'
            pacientes.append(dict(pid=pid, dataset=cohorte))
            for z in range(4):
                filas.append(dict(sample_id=f'{pid}_z{z:03d}', patient_id=pid, split='train', pCR=k % 2, slice_index=z))
    filas.append(dict(sample_id='duke_test_z000', patient_id='duke_test', split='test', pCR=1, slice_index=0))
    pacientes.append(dict(pid='duke_test', dataset='duke'))
    return pd.DataFrame(filas), pd.DataFrame(pacientes)


def test_un_corte_por_cohorte_y_clase_reproducible_y_sin_test():
    samples, patients = _metadatos()
    a = elegir(samples, patients, actuales=('spy1_0_z001',))
    b = elegir(samples, patients, actuales=('spy1_0_z001',))
    assert [f.sample_id for f in a] == [f.sample_id for f in b]
    assert a[0].sample_id == 'spy1_0_z001'
    nuevos = a[1:]
    assert sorted((f.dataset, int(f.pCR)) for f in nuevos) == [(c, p) for c in ('duke', 'spy1', 'spy2') for p in (0, 1)]
    assert len({f.patient_id for f in a}) == len(a) and all(f.split == 'train' for f in a)
