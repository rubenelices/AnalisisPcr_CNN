import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.generalization_panels import diagnostics


def test_loss_modes_and_two_observed_auc_bars_are_separate(tmp_path,monkeypatch):
    history=pd.DataFrame(dict(epoch=[1,2],train_loss=[.9,.8],train_eval_loss=[.6,.5],validation_loss=[.65,.7],
        train_patient_bce=[.58,.48],validation_patient_bce=[.64,.68],train_patient_auc=[.6,.7],patient_roc_auc=[.62,.6],training_cuts=[100,100]))
    history.to_csv(tmp_path/'history.csv',index=False)
    (tmp_path/'config_resolved.json').write_text(json.dumps({'loss':'normal'}))
    original=plt.subplots;captured=[]
    def capture(*a,**kw):
        fig,axes=original(*a,**kw);captured.append(axes);return fig,axes
    monkeypatch.setattr(plt,'subplots',capture)
    s=dict(epoch=1,train_eval={'patient_metrics':{'roc_auc':.6}},validation_eval={'patient_metrics':{'roc_auc':.62}})
    path=diagnostics(tmp_path,s,'Synthetic')
    axes=captured[0]
    assert np.allclose(axes[0,0].lines[0].get_ydata(),[.58,.48])
    assert np.allclose(axes[1,0].lines[0].get_ydata(),[.9,.8])
    assert np.allclose(axes[1,1].lines[0].get_ydata(),[.6,.5])
    assert len(axes[0,2].patches)==2
    assert [p.get_height() for p in axes[0,2].patches]==[.6,.62]
    assert np.allclose(axes[1,2].lines[0].get_xdata(),[100,200])
    assert path.is_file() and path.stat().st_size>10000
