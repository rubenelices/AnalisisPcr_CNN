import numpy as np
import pandas as pd
import pytest

from src.analysis.new_approach_paired import paired_intervals


def pairs():
    result = []
    for fold in range(5):
        a = pd.DataFrame(dict(patient_id=[f'{fold}_{i}' for i in range(4)],
                              target=[0, 0, 1, 1], probability=[.1, .1, .9, .9]))
        b = a.copy(); b.probability = 1-b.probability
        result.append((a, b, .5, .5))
    return result


def test_bootstrap_is_paired_by_patient_with_known_effect_and_seed():
    result = paired_intervals(pairs(), repetitions=20)
    assert result == paired_intervals(pairs(), repetitions=20)
    assert result['intervals']['AUC'] == dict(difference=-1., lower=-1., upper=-1.)
    assert result['intervals']['F1_macro']['difference'] == -1.
    assert result['intervals']['Brier']['difference'] == pytest.approx(.8)
    equal = [(a, a.copy(), .5, .5) for a, _, _, _ in pairs()]
    assert all(all(v == 0 for v in item.values())
               for item in paired_intervals(equal, repetitions=10)['intervals'].values())


@pytest.mark.parametrize('problem', ['patient', 'target', 'duplicate', 'overlap', 'nan'])
def test_bootstrap_rejects_unpaired_or_invalid_inputs(problem):
    data = pairs()
    a, b, _, _ = data[0]
    if problem == 'patient': b.loc[0, 'patient_id'] = 'another'
    if problem == 'target': b.loc[0, 'target'] = 1
    if problem == 'duplicate': b.loc[0, 'patient_id'] = b.loc[1, 'patient_id']
    if problem == 'overlap': data[1] = data[0]
    if problem == 'nan': b.loc[0, 'probability'] = np.nan
    with pytest.raises(ValueError): paired_intervals(data, repetitions=1)
