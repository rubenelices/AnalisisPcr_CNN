"""Control B08: solo comprobaciones sintéticas, sin entrenamiento."""
import json
from pathlib import Path
import pytest
import torch
import pandas as pd
from src.train_multimodal import load_spec
from src.clinical_only import ClinicalOnlyPCRNet, load_clinical_only_checkpoint
from src.clinical_features import ClinicalPreprocessor

CONFIG='experimentos_equilibrados/B08_solo_edad_menopausia/config.json'


def test_control_preserves_reference_protocol():
    spec,cfg=load_spec(CONFIG)
    _,ref=load_spec('experimentos_datos/D36_edad_menopausia/config.json')
    assert cfg.data==ref.data and cfg.training==ref.training
    assert spec['schema']=='clinical_only_e58_v1'
    assert spec['clinical_input_columns']==spec['active_clinical_columns']==['age','menopause']
    smoke,c=load_spec(CONFIG.replace('config.json','smoke.json'))
    assert c.training.epochs==2 and c.data.train_patients_per_class==4


def test_basic_control_requires_explicit_matching_whitelist(tmp_path):
    spec=json.loads(Path(CONFIG).read_text());spec.pop('clinical_input_columns')
    p=tmp_path/'bad.json';p.write_text(json.dumps(spec))
    with pytest.raises(ValueError):load_spec(p)
    spec['clinical_input_columns']=['age','HR'];p.write_text(json.dumps(spec))
    with pytest.raises(ValueError):load_spec(p)


def test_basic_control_checkpoint_roundtrip_and_image_independence(tmp_path):
    frame=pd.DataFrame(dict(pid=['a','b','c'],age=[30.,50.,70.],menopause=[0.,1.,float('nan')]))
    prep=ClinicalPreprocessor.fit(frame.assign(HR=0., HER2=0.),['age','menopause'])
    features=prep.transform(frame)
    assert torch.equal(features[:,[1,2,5,6]],torch.zeros(3,4))
    net=ClinicalOnlyPCRNet().eval()
    assert sum(p.numel() for p in net.parameters())==161
    with torch.no_grad():
        expected=net(None,features)
        assert torch.equal(expected,net(torch.rand(3,3,16,16),features))
    p=tmp_path/'synthetic.pt';torch.save(dict(model_class='ClinicalOnlyPCRNet_v1',
        model_state_dict=net.state_dict(),clinical_preprocessor=prep.to_dict()),p)
    restored,stats=load_clinical_only_checkpoint(p)
    with torch.no_grad():assert torch.equal(expected,restored(None,stats.transform(frame)))
