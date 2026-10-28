import json
from pathlib import Path

import pandas as pd
import pytest
import torch

from src.clinical_only import ClinicalOnlyPCRNet, ClinicalCutDataset
from src.clinical_features import ClinicalPreprocessor
from src.train_multimodal import load_spec


def test_shapes_count_and_image_independence():
    model=ClinicalOnlyPCRNet().eval()
    assert sum(p.numel() for p in model.parameters())==161
    features=torch.randn(3,8)
    assert model(None,features).shape==(3,)
    assert torch.equal(model(None,features),model(torch.randn(3,3,8,8),features))
    assert not any(isinstance(m,torch.nn.Conv2d) for m in model.modules())
    for invalid in (torch.zeros(3,7),torch.full((3,8),float('nan'))):
        with pytest.raises(ValueError): model(None,invalid)


def test_dataset_repeats_clinic_not_images_or_stats():
    clinical=pd.DataFrame(dict(pid=['a','b'],age=[30.,60.],HR=[0,1],HER2=[1,0],menopause=[0,1]))
    prep=ClinicalPreprocessor.fit(clinical)
    rows=pd.DataFrame(dict(patient_id=['a','a','b'],sample_id=['a0','a1','b0'],
                           pCR=[0,0,1],slice_index=[0,1,0],split=['validation']*3,fold=[0]*3))
    data=ClinicalCutDataset(rows,clinical,prep)
    assert len(data)==3 and prep.train_patient_count==2
    assert data[0]['image'].shape==(1,)
    assert torch.equal(data[0]['clinical'],data[1]['clinical'])
    assert data[2]['target'].item()==1
    with pytest.raises(ValueError): ClinicalCutDataset(rows,clinical.iloc[:1],prep)


def test_d08_matches_e58_protocol():
    spec,cfg=load_spec('experimentos_datos/D08_clinica_sola/config.json')
    base=load_spec('configs/multimodal/E58_clinical_fusion_mac.json')[1]
    assert spec['schema']=='clinical_only_e58_v1'
    assert cfg.data==base.data and cfg.training==base.training and cfg.model==base.model


def test_checkpoint_roundtrip(tmp_path):
    from src.clinical_only import load_clinical_only_checkpoint
    clinical=pd.DataFrame(dict(pid=['a','b'],age=[30.,60.],HR=[0,1],HER2=[1,0],menopause=[0,1]))
    prep=ClinicalPreprocessor.fit(clinical); model=ClinicalOnlyPCRNet().eval()
    path=tmp_path/'checkpoint.pt'
    torch.save(dict(model_class='ClinicalOnlyPCRNet_v1',model_state_dict=model.state_dict(),
                    clinical_preprocessor=prep.to_dict()),path)
    restored,state=load_clinical_only_checkpoint(path)
    x=prep.transform(clinical)
    assert torch.equal(model(None,x),restored(None,x)) and prep==state
