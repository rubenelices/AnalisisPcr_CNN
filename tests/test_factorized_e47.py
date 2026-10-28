"""CNN E47 real: orden de unidades, formas, parámetros y compatibilidad."""
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from torch import nn

from src.architectures import BaseCNNConfig, BreastPCRNet, trainable_parameter_count
from src.document_architectures import trace, visible_diagram_steps
from src.experiment_config import experiment_config_from_dict, load_experiment_config
from src.factorized_conv import FactorizedSpatialConv
from src.inference import DCEInferenceEngine

ROOT = Path(__file__).resolve().parents[1]

def config(name):
    return load_experiment_config(ROOT/'configs/experiments'/f'{name}.json')


def test_only_convolution_policy_changes_and_smokes_are_isolated():
    old,new = config('E19_weight_decay_001_mac'),config('E47_factorized_stem_mac')
    assert new.model==replace(old.model,convolution_type='factorized_after_stem')
    assert new.data==old.data and new.training==old.training
    assert not new.training.sam.enabled and new.model.dilation==1
    assert new==experiment_config_from_dict(new.to_dict())
    for name in ('SMOKE_E47_factorized_stem_cpu','SMOKE_E47_factorized_stem_mps'):
        small=config(name)
        assert small.model==new.model
        assert small.training==replace(new.training,epochs=2,early_stopping_patience=3)
        assert small.data==replace(new.data,train_patients_per_class=4,validation_patients_per_class=2)
        assert small.output_dir!=new.output_dir and small.checkpoint_dir!=new.checkpoint_dir


@pytest.mark.parametrize('value',[None,True,'','depthwise',[],1])
def test_invalid_policy_rejected(value):
    with pytest.raises(ValueError,match='convolution_type'):
        BaseCNNConfig(convolution_type=value)


def test_actual_units_count_order_shapes_and_receptive_field():
    model=BreastPCRNet(config('E47_factorized_stem_mac').model)
    assert isinstance(model.blocks[0].features[0],nn.Conv2d)
    assert model.blocks[0].features[0].in_channels==3
    for index,block in enumerate(model.blocks):
        assert isinstance(block.features[3],FactorizedSpatialConv)
        assert index==0 or isinstance(block.features[0],FactorizedSpatialConv)
        assert isinstance(block.features[1],nn.BatchNorm2d) and isinstance(block.features[2],nn.ReLU)
        assert isinstance(block.features[4],nn.BatchNorm2d) and isinstance(block.features[5],nn.ReLU)
        assert block.first_convolution_end==3
    assert sum(isinstance(layer,nn.Conv2d) for layer in model.modules())==15
    assert sum(isinstance(layer,nn.BatchNorm2d) for layer in model.modules())==8
    assert sum(isinstance(layer,nn.ReLU) for layer in model.modules())==8
    steps=trace(model)
    assert len([step for step in steps if step['kind']=='conv'])==15
    assert [step['shape'] for step in steps if step['kind']=='pool']==[(16,128,128),(32,64,64),(64,32,32),(128,16,16)]
    assert next(step['rf'] for step in reversed(steps) if step['kind'] in {'conv','pool'})==106
    assert trainable_parameter_count(model)==sum(step['parameters'] for step in steps)==37201
    dw=[step for step in steps if step['label'].startswith('DW')]
    pw=[step for step in steps if step['label'].startswith('PW')]
    assert len(dw)==len(pw)==7
    assert all('sin BN/ReLU intermedia' in step['detail'] for step in dw)
    assert all('mezcla canales' in step['detail'] for step in pw)
    diagram=visible_diagram_steps(steps)
    units=[step for step in diagram if step['kind']=='conv']
    assert len(units)==8 and sum(step['parameters'] for step in units)==36112
    assert len([step for step in steps if step['kind']=='conv'])==15  # no mutar tabla
    assert all(step['shape']==pw[index]['shape'] for index,step in enumerate(units[1:]))
    old_steps=trace(BreastPCRNet(config('E19_weight_decay_001_mac').model))
    assert len([step for step in visible_diagram_steps(old_steps) if step['kind']=='conv'])==8


def test_prescribed_initializer_is_used_for_real_groups(monkeypatch):
    calls={}
    original=nn.init.kaiming_normal_
    def recorder(tensor,*args,**kwargs):
        calls[id(tensor)]=(kwargs['mode'],kwargs['nonlinearity'])
        return original(tensor,*args,**kwargs)
    monkeypatch.setattr(nn.init,'kaiming_normal_',recorder)
    model=BreastPCRNet(config('E47_factorized_stem_mac').model)
    dw={layer.depthwise for layer in model.modules() if isinstance(layer,FactorizedSpatialConv)}
    for layer in model.modules():
        if isinstance(layer,nn.Conv2d):
            expected=('fan_in','linear') if layer in dw else ('fan_out','relu')
            assert calls[id(layer.weight)]==expected


def test_all_actual_parameters_receive_finite_gradients_and_one_adamw_step():
    model=BreastPCRNet(config('E47_factorized_stem_mac').model)
    x=torch.rand(2,3,256,256,requires_grad=True)
    logits=model(x)
    assert logits.shape==(2,)
    nn.BCEWithLogitsLoss()(logits,torch.tensor([0.,1.])).backward()
    assert torch.isfinite(x.grad).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert all(layer.weight.grad.abs().sum()>0 for layer in model.modules() if isinstance(layer,nn.Conv2d))
    optimizer=torch.optim.AdamW(model.parameters(),lr=0.001)
    optimizer.step()
    assert all(state['step'].item()==1 for state in optimizer.state.values())


@pytest.mark.parametrize('new',[False,True])
def test_old_and_factorized_checkpoints_shared_inference(tmp_path,new):
    cfg=config('E47_factorized_stem_mac' if new else 'E19_weight_decay_001_mac')
    model=BreastPCRNet(cfg.model).eval()
    raw=cfg.to_dict()
    if not new:
        raw['model'].pop('convolution_type')
        assert experiment_config_from_dict(raw).model.convolution_type=='standard'
    path=tmp_path/'own_synthetic.pt'
    torch.save({'config':raw,'model_state_dict':model.state_dict(),'experiment_id':cfg.experiment_id,'epoch':0},path)
    engine=DCEInferenceEngine.from_checkpoint(path,device_name='cpu')
    for _ in range(5):
        image=torch.rand(3,256,256)
        with torch.inference_mode():
            expected=torch.sigmoid(model(image.unsqueeze(0))).item()
        assert engine.predict_tensor(image).probability_pcr==expected
