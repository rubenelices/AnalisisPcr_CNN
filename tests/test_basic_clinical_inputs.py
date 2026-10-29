"""Solo datos sintéticos; no científicos ni smokes de entrenamiento."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch
from src.pipeline.clinical_features import ClinicalPreprocessor,load_patient_clinical
from src.training.train_multimodal import load_spec
from src.models.multimodal_model import MultimodalPCRNet,load_multimodal_checkpoint

CONFIG='experiments/D36_age_menopause/run/config.json'

def rows():
    return pd.DataFrame(dict(pid=['a','b','c'],age=[30.,60.,np.nan],menopause=[0.,1.,np.nan]))

def load(tmp_path):
    (tmp_path/'metadata').mkdir()
    rows().to_csv(tmp_path/'metadata/patients.csv',index=False)
    return load_patient_clinical(tmp_path,['a','b','c'],['age','menopause'])

def test_no_hr_her2_required_for_csv_and_inference(tmp_path):
    data=load(tmp_path)
    prep=ClinicalPreprocessor.fit(data,['age','menopause'])
    assert prep.imputation_values==(45.,0.,0.,0.) and prep.train_patient_count==3
    x=prep.transform(rows())
    assert x.shape==(3,8) and torch.equal(x[:,[1,2,5,6]],torch.zeros(3,4))
    assert x[2,[4,7]].tolist()==[1.,1.]
    assert torch.equal(x,ClinicalPreprocessor.from_dict(prep.to_dict()).transform(rows()))

def test_csv_whitelist_does_not_parse_forbidden_values(tmp_path):
    (tmp_path/'metadata').mkdir()
    data=rows();data['HR']='not_a_number';data['HER2']='secret';data['pCR']='hidden'
    data.to_csv(tmp_path/'metadata/patients.csv',index=False)
    out=load_patient_clinical(tmp_path,['b','a'],['age','menopause'])
    assert out.pid.tolist()==['b','a'] and out.HR.eq(0).all() and 'pCR' not in out

def test_fit_state_does_not_change_on_new_patient(tmp_path):
    prep=ClinicalPreprocessor.fit(load(tmp_path),['age','menopause']);state=prep.to_dict()
    new=rows();new.age=90
    prep.transform(new)
    assert prep.to_dict()==state

def test_active_field_required_but_nan_allowed(tmp_path):
    prep=ClinicalPreprocessor.fit(load(tmp_path),['age','menopause'])
    with pytest.raises(ValueError,match='Faltan'):prep.transform(rows().drop(columns='menopause'))
    assert torch.isfinite(prep.transform(rows())).all()

@pytest.mark.parametrize('columns',[['pCR'],['age','age'],[],['age','HR']])
def test_input_spec_requires_same_basic_active_columns(tmp_path,columns):
    spec=json.loads(Path(CONFIG).read_text());spec['clinical_input_columns']=columns
    path=tmp_path/'config.json';path.write_text(json.dumps(spec))
    with pytest.raises(ValueError,match='básica'):load_spec(path)

def test_checkpoint_basic_inputs_round_trip_without_training(tmp_path):
    torch.set_num_threads(2)
    spec,cfg=load_spec(CONFIG);model=MultimodalPCRNet(cfg.model).eval()
    prep=ClinicalPreprocessor.fit(load(tmp_path),['age','menopause'])
    image=torch.zeros(3,3,64,64);clinical=prep.transform(rows())
    with torch.inference_mode():before=model(image,clinical)
    path=tmp_path/'synthetic.pt'
    torch.save(dict(model_class='MultimodalPCRNet_v1',config=spec,
        model_state_dict=model.state_dict(),clinical_preprocessor=prep.to_dict()),path)
    restored,stats=load_multimodal_checkpoint(path,'cpu')
    with torch.inference_mode():after=restored(image,stats.transform(rows()))
    assert torch.equal(before,after) and sum(p.numel() for p in model.parameters())==294289

def test_base_config_only_changes_active_inputs_from_e58():
    spec,cfg=load_spec(CONFIG)
    _,reference=load_spec('configs/multimodal/E58_clinical_fusion_mac.json')
    assert cfg.model==reference.model and cfg.training==reference.training and cfg.data==reference.data
    assert spec['active_clinical_columns']==spec['clinical_input_columns']==['age','menopause']
    assert cfg.model.activation=='relu' and not cfg.data.augmentation.enabled
