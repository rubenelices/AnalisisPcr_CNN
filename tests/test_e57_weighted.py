from dataclasses import replace
import pandas as pd
import pytest
import torch
from src.training.train_e57_weighted import load_spec, patient_criterion, evaluate_features
from src.training.train_patient_features import load_spec as original_spec, evaluate_features as original_eval
from src.models.patient_feature_model import PatientFeatureNet
from test_patient_features import batches

CONFIG='experiments/E57_ponderada_moderada/config.json'


def test_only_loss_changes_against_e57():
    s,c=load_spec(CONFIG)
    ref,r=original_spec('configs/patient_features/E57_patient_feature_mean_mac.json')
    assert c.model==r.model and c.data==r.data
    assert replace(c.training,loss='normal')==r.training
    assert s['pool_stage']==ref['pool_stage']=='features'
    _,smoke=load_spec(CONFIG.replace('config.json','smoke.json'))
    assert smoke.training.epochs==2


def test_weight_counts_patients_not_correlated_cuts():
    rows=pd.DataFrame({'patient_id':['a']*10+['b','c','d','e'], 'pCR':[0]*10+[0,0,0,1]})
    criterion,weight=patient_criterion(rows,'cpu')
    assert weight==2 and criterion.pos_weight.item()==2
    rows.loc[0,'pCR']=1
    with pytest.raises(ValueError):patient_criterion(rows,'cpu')


def test_weight_changes_loss_not_predictions_or_selection_metrics():
    torch.set_num_threads(2);torch.manual_seed(42)
    _,cfg=load_spec(CONFIG);m=PatientFeatureNet(cfg.model).eval()
    assert sum(p.numel() for p in m.parameters())==298161
    loader=batches(2)
    a=original_eval(m,loader,torch.device('cpu'),.5)
    b=evaluate_features(m,loader,torch.device('cpu'),.5,torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(2.)))
    assert a[0]!=b[0] and a[1]!=b[1]
    assert a[2]==b[2] and a[3]==b[3]
    pd.testing.assert_frame_equal(a[4],b[4]);pd.testing.assert_frame_equal(a[5],b[5])


def test_checkpoint_restore_with_weighted_config(tmp_path):
    from src.models.patient_feature_model import load_feature_checkpoint
    s,c=load_spec(CONFIG);m=PatientFeatureNet(c.model).eval()
    image=torch.rand(2,3,32,32)
    with torch.no_grad():expected=m(image)
    p=tmp_path/'synthetic.pt'
    torch.save(dict(model_class='PatientFeatureNet_v1',config=s,model_state_dict=m.state_dict()),p)
    restored=load_feature_checkpoint(p)
    with torch.no_grad():torch.testing.assert_close(expected,restored(image),rtol=0,atol=0)


def test_refresh_general_notebook_only_for_scientific_run(monkeypatch):
    from src.training.train_e57_weighted import refresh_results_notebook
    import src.reporting.clear_results
    import src.reporting.balanced_results
    monkeypatch.setattr(src.reporting.balanced_results,'build',lambda: None)
    calls=[]
    monkeypatch.setattr(src.reporting.clear_results,'build',lambda: calls.append('build') or 'notebook')
    assert refresh_results_notebook(True) is None and not calls
    assert refresh_results_notebook(False)=='notebook' and calls==['build']
