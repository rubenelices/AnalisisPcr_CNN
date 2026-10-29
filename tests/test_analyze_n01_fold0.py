import pandas as pd
import pytest

from src.analysis.analyze_n01_fold0 import paired_errors


def test_paired_error_counts_match_patients_even_if_rows_are_reordered():
    a = pd.DataFrame(dict(patient_id=['a', 'b', 'c', 'd'], target=[0, 0, 1, 1],
                          correct=[True, True, False, False]))
    b = a.copy(); b.correct = [True, False, True, False]
    result = paired_errors(a, b.iloc[::-1])
    assert result[0] == dict(Grupo='Todas', Ambas_bien=1, Solo_normal_bien=1,
                            Solo_ponderada_bien=1, Ambas_mal=1)
    assert sum(result[1][k]+result[2][k] for k in ['Ambas_bien', 'Solo_normal_bien',
                                                'Solo_ponderada_bien', 'Ambas_mal']) == 4


@pytest.mark.parametrize('problem', ['patient', 'target', 'duplicate'])
def test_cannot_pair_other_patients_or_labels(problem):
    a = pd.DataFrame(dict(patient_id=['a', 'b'], target=[0, 1], correct=[True, False]))
    b = a.copy()
    if problem == 'patient': b.loc[0, 'patient_id'] = 'z'
    if problem == 'target': b.loc[0, 'target'] = 1
    if problem == 'duplicate': b.loc[0, 'patient_id'] = 'b'
    with pytest.raises(ValueError): paired_errors(a, b)
