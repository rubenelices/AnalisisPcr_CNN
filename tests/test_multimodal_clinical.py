import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from src.pipeline.clinical_features import ClinicalPreprocessor, load_patient_clinical
from src.models.multimodal_model import MultimodalPCRNet, load_multimodal_checkpoint
from src.training.train_multimodal import load_spec


def patients():
    return pd.DataFrame(dict(pid=['a','b','c'],age=[30.,50.,np.nan],
                             HR=[0.,1.,1.],HER2=[1.,0.,np.nan],menopause=[0.,1.,np.nan]))


def test_preprocessing_train_only_and_missing_flags():
    train=patients()
    prep=ClinicalPreprocessor.fit(train)
    assert prep.train_patient_count==3
    assert prep.imputation_values==(40.,1.,0.,0.)
    out=prep.transform(train)
    assert out.shape==(3,8) and out.dtype==torch.float32
    assert out[2].tolist()==[0.,1.,0.,0.,1.,0.,1.,1.]
    before=prep.to_dict()
    extreme=patients(); extreme.age=100
    prep.transform(extreme)
    assert prep.to_dict()==before
    assert torch.equal(out,ClinicalPreprocessor.from_dict(before).transform(train))


def test_labels_and_ids_not_features():
    train=patients(); prep=ClinicalPreprocessor.fit(train)
    original=prep.transform(train)
    train['pCR']=[1,0,1]; train['pid']=['x','y','z']; train['cohort']='arbitrary'
    assert torch.equal(original,prep.transform(train))


@pytest.mark.parametrize('column,value',[('age',-1),('HR',2),('HER2',float('inf'))])
def test_invalid_clinical(column,value):
    frame=patients(); frame.loc[0,column]=value
    with pytest.raises(ValueError): ClinicalPreprocessor.fit(frame)


def test_duplicate_patient_fit_rejected():
    frame=patients(); frame.loc[1,'pid']='a'
    with pytest.raises(ValueError): ClinicalPreprocessor.fit(frame)


def test_whitelist_and_join(tmp_path):
    (tmp_path/'metadata').mkdir()
    frame=patients(); frame['pCR']=['secret']*3
    frame.to_csv(tmp_path/'metadata/patients.csv',index=False)
    result=load_patient_clinical(tmp_path,['b','a'])
    assert result.pid.tolist()==['b','a']
    assert 'pCR' not in result
    with pytest.raises(ValueError): load_patient_clinical(tmp_path,['missing'])


def test_model_shapes_gradients_and_checkpoint(tmp_path):
    torch.set_num_threads(2)
    spec,cfg=load_spec('configs/multimodal/E58_clinical_fusion_mac.json')
    model=MultimodalPCRNet(cfg.model)
    assert sum(p.numel() for p in model.parameters())==294289
    x=torch.randn(2,3,64,64); clinical=torch.randn(2,8)
    logits=model(x,clinical)
    assert logits.shape==(2,)
    logits.sum().backward()
    assert all(p.grad is not None for p in model.parameters())
    model.eval(); before=model(x,clinical).detach()
    prep=ClinicalPreprocessor.fit(patients())
    path=tmp_path/'own.pt'
    torch.save(dict(model_class='MultimodalPCRNet_v1',config=spec,
                    model_state_dict=model.state_dict(),clinical_preprocessor=prep.to_dict()),path)
    restored,stats=load_multimodal_checkpoint(path)
    assert torch.equal(before,restored(x,clinical))
    assert stats.to_dict()==prep.to_dict()
    assert restored(x[:1],clinical[:1]).shape==(1,)
    with pytest.raises(ValueError): restored(x,clinical[:1])
    with pytest.raises(ValueError): restored(x,torch.ones(2,7))


def test_no_test_or_augmentation_in_spec():
    _,cfg=load_spec('configs/multimodal/E58_clinical_fusion_mac.json')
    assert cfg.data.validation_fold==0 and cfg.training.seed==42
    assert cfg.data.batch_size==16 and cfg.training.early_stopping_patience==10
    assert not cfg.data.augmentation.enabled
