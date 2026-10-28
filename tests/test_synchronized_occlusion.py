import pytest
import torch
from src.synchronized_occlusion import SynchronizedOcclusion, OccludedTrainingDataset


def test_disabled_preserves_values_identity_rng():
    x=torch.ones(3,256,256); before=torch.get_rng_state()
    assert SynchronizedOcclusion(0)(x) is x
    assert torch.equal(before,torch.get_rng_state())


def test_mask_sync_area_no_input_mutation_and_differences():
    x=torch.stack([torch.full((256,256),v) for v in (.2,.5,.7)])
    original=x.clone();torch.manual_seed(42)
    y=SynchronizedOcclusion(1)(x)
    mask=y[0]==0
    assert mask.sum()==32*32
    assert torch.equal(mask,y[1]==0) and torch.equal(mask,y[2]==0)
    assert torch.equal(x,original) and torch.equal(y[:,~mask],x[:,~mask])
    assert torch.all((y[1]-y[0])[mask]==0)
    assert y.shape==x.shape and y.dtype==x.dtype


def test_seed_reproducibility():
    x=torch.ones(3,256,256);t=SynchronizedOcclusion(1)
    torch.manual_seed(7);a=t(x);torch.manual_seed(7);b=t(x)
    assert torch.equal(a,b)


@pytest.mark.parametrize('p,side',[(True,32),(-1,32),(float('nan'),32),(1,True),(1,65),(1,0)])
def test_bad_settings(p,side):
    with pytest.raises(ValueError):SynchronizedOcclusion(p,side)


def test_wrapper_preserves_clinical_target_and_original():
    sample={'image':torch.ones(3,256,256),'clinical':torch.ones(8),'target':torch.tensor(1.),'patient_id':'synthetic'}
    wrapper=OccludedTrainingDataset([sample],SynchronizedOcclusion(1))
    result=wrapper[0]
    assert result['clinical'] is sample['clinical'] and result['target'] is sample['target']
    assert result['patient_id']==sample['patient_id'] and torch.all(sample['image']==1)
