import pandas as pd
import pytest
from src.decision_diagnostics import threshold_table


def test_no_predicted_positives_precision_undefined_and_counts_correct():
    p=pd.DataFrame(dict(patient_id=['a','b','c','d'],target=[0,0,1,1],probability=[.2,.3,.3,.4]))
    table=threshold_table(p,[.5,.3])
    a,b=table.iloc[0],table.iloc[1]
    assert a['Precisión pCR']=='No definida' and a['TP']==0 and a['FN']==2
    assert a['FP']==0 and a['TN']==2 and a['Balanced accuracy']==.5
    assert b['TP']==2 and b['FP']==1 and b['FN']==0 and b['TN']==1
    assert b['Precisión pCR']=='0.667'
    assert p.probability.tolist()==[.2,.3,.3,.4]


def test_reject_duplicate_patient_predictions():
    p=pd.DataFrame(dict(patient_id=['a','a'],target=[0,1],probability=[.2,.4]))
    with pytest.raises(ValueError,match='Una fila'):threshold_table(p)
