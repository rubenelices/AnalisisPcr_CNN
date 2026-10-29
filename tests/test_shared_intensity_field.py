"""D20: intensidad espacial común, sin tocar clínica ni validación."""
import json
from pathlib import Path

import pytest
import torch


@pytest.mark.parametrize('device',['cpu','mps'])
def test_field_geometry_temporal_sync_and_gradients(device):
    from src.pipeline.shared_intensity_field import SharedIntensityField
    if device=='mps' and not torch.backends.mps.is_available():
        pytest.skip('MPS no disponible')
    transform=SharedIntensityField(max_amplitude=.1)
    image=torch.zeros(3,256,256,device=device)
    image[:,40:200,30:180]=torch.tensor([.2,.4,.6],device=device)[:,None,None]
    image.requires_grad_();original=image.detach().clone()
    torch.manual_seed(42);out=transform(image)
    assert torch.equal(image.detach(),original)
    assert torch.allclose(out[1],2*out[0],atol=1e-6)
    assert torch.allclose(out[2],3*out[0],atol=1e-6)
    assert torch.equal(out[:,0,:],torch.zeros_like(out[:,0,:]))
    ratios=out[0,40:200,30:180]/image[0,40:200,30:180]
    assert ratios.min()>=.9-1e-6 and ratios.max()<=1.1+1e-6
    out.sum().backward()
    assert torch.isfinite(image.grad).all()
    torch.manual_seed(42);again=transform(image)
    assert torch.equal(out,again)


def test_no_clipping_and_zero_amplitude_no_rng():
    from src.pipeline.shared_intensity_field import SharedIntensityField
    x=torch.ones(3,256,256)
    torch.manual_seed(17);out=SharedIntensityField(.1)(x)
    assert out.max()>1 and out.min()<1
    assert torch.allclose(out.mean(),torch.tensor(1.),atol=1e-6)
    state=torch.random.get_rng_state().clone()
    assert SharedIntensityField(0)(x) is x
    assert torch.equal(state,torch.random.get_rng_state())


@pytest.mark.parametrize('amplitude',[True,-.1,.3,float('nan'),'0.1'])
def test_invalid_amplitude(amplitude):
    from src.pipeline.shared_intensity_field import SharedIntensityField
    with pytest.raises(ValueError):SharedIntensityField(amplitude)


def test_wrapper_preserves_metadata():
    from src.pipeline.shared_intensity_field import SharedIntensityField,IntensityFieldTrainingDataset
    sample={'image':torch.ones(3,256,256),'clinical':torch.arange(8.),
            'target':torch.tensor(1.),'patient_id':'train_patient'}
    wrapped=IntensityFieldTrainingDataset([sample],SharedIntensityField(.1))
    result=wrapped[0]
    assert result is not sample and result['clinical'] is sample['clinical']
    assert result['patient_id']==sample['patient_id'] and result['target'] is sample['target']
    assert torch.equal(sample['image'],torch.ones(3,256,256))


def test_spec_isolated_from_e58():
    from src.training.train_multimodal import load_spec
    spec,cfg=load_spec('experiments/D20_campo_intensidad/config.json')
    base,b=load_spec('configs/multimodal/E58_clinical_fusion_mac.json')
    assert cfg.model==b.model and cfg.training==b.training and cfg.data==b.data
    assert spec['image_intensity_field']=={'max_amplitude':.1}
    assert not any([cfg.data.augmentation.enabled,cfg.data.intensity_augmentation.enabled,
                    cfg.data.horizontal_flip.enabled])


def test_clinical_only_rejects_field(tmp_path):
    from src.training.train_multimodal import load_spec
    spec=json.loads(Path('experiments/D08_clinica_sola/config.json').read_text())
    spec['image_intensity_field']={'max_amplitude':.1}
    p=tmp_path/'invalid.json';p.write_text(json.dumps(spec))
    with pytest.raises(ValueError,match='Clínica sola'):load_spec(p)


@pytest.mark.parametrize('device',['cpu','mps'])
def test_full_cnn_step_and_checkpoint(tmp_path,device):
    from src.pipeline.shared_intensity_field import SharedIntensityField
    from src.training.train_multimodal import load_spec
    from src.models.multimodal_model import MultimodalPCRNet,load_multimodal_checkpoint
    from src.pipeline.clinical_features import ClinicalPreprocessor
    from test_multimodal_clinical import patients
    if device=='mps' and not torch.backends.mps.is_available():pytest.skip('MPS no disponible')
    torch.set_num_threads(2);torch.manual_seed(42)
    spec,cfg=load_spec('experiments/D20_campo_intensidad/config.json')
    model=MultimodalPCRNet(cfg.model).to(device)
    assert sum(p.numel() for p in model.parameters())==294289
    x=torch.stack([SharedIntensityField(.1)(torch.rand(3,256,256)) for _ in range(2)]).to(device)
    c=torch.randn(2,8,device=device);y=torch.tensor([0.,1.],device=device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001)
    before={n:p.detach().clone() for n,p in model.named_parameters()}
    loss=torch.nn.functional.binary_cross_entropy_with_logits(model(x,c),y)
    loss.backward();assert torch.isfinite(loss)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    optimizer.step();assert any(not torch.equal(before[n],p) for n,p in model.named_parameters())
    model.eval();expected=model(x,c).detach().cpu();path=tmp_path/'own.pt'
    torch.save(dict(model_class='MultimodalPCRNet_v1',config=spec,model_state_dict=model.state_dict(),
                    clinical_preprocessor=ClinicalPreprocessor.fit(patients()).to_dict()),path)
    restored,_=load_multimodal_checkpoint(path,device=device)
    assert torch.allclose(expected,restored(x,c).detach().cpu(),atol=1e-6)


@pytest.mark.parametrize('shape',[(3,128,256),(2,256,256),(1,3,256,256)])
def test_invalid_shape(shape):
    from src.pipeline.shared_intensity_field import SharedIntensityField
    with pytest.raises(ValueError):SharedIntensityField(.1)(torch.ones(shape))


def test_invalid_dtype_and_nonfinite():
    from src.pipeline.shared_intensity_field import SharedIntensityField
    t=SharedIntensityField(.1)
    with pytest.raises(ValueError):t(torch.ones(3,256,256,dtype=torch.float64))
    x=torch.ones(3,256,256);x[0,0,0]=float('nan')
    with pytest.raises(ValueError):t(x)


@pytest.mark.parametrize('key,value',[('feature_normalization','l2_fixed'),('fusion_hidden',32),
    ('image_pooling','enhancement_weighted'),('image_weight_decay_factor',10)])
def test_combination_not_allowed(tmp_path,key,value):
    from src.training.train_multimodal import load_spec
    spec=json.loads(Path('experiments/D20_campo_intensidad/config.json').read_text())
    spec[key]=value;p=tmp_path/'bad.json';p.write_text(json.dumps(spec))
    with pytest.raises(ValueError,match='Campo exige'):load_spec(p)
