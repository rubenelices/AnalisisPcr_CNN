"""E30: atención espacial, comparación controlada y compatibilidad anterior."""
from dataclasses import replace
import json

import pytest
import torch
from torch import nn

from src.attention_pooling import GatedSpatialPool
from src.architectures import BreastPCRNet, trainable_parameter_count
from src.experiment_config import load_experiment_config, experiment_config_from_dict
from src.document_architectures import trace
from src.compare_experiments import _comparability


def config(name):
    return load_experiment_config(f'configs/experiments/{name}.json')


def test_e30_only_changes_spatial_pooling_and_keeps_scientific_protocol():
    base,new,smoke=map(config,['E19_weight_decay_001_mac','E30_gated_attention_mac','SMOKE_gated_attention_mac'])
    assert base.data==new.data and base.training==new.training
    assert new.model==replace(base.model,global_pooling='gated_attention')
    assert not new.training.mixup.enabled and new.training.lr_schedule.policy=='plateau'
    assert not new.data.augmentation.enabled and not new.data.intensity_augmentation.enabled
    assert new.training.epochs==50 and new.training.early_stopping_patience==10
    assert len({c.output_dir for c in [base,new,smoke]})==3
    assert len({c.checkpoint_dir for c in [base,new,smoke]})==3
    assert smoke.model==new.model and smoke.training==replace(new.training,epochs=2)
    assert smoke.data==replace(new.data,train_patients_per_class=4,validation_patients_per_class=2)
    source=lambda c:{'scientific_run_verified':True,'config':c.to_dict(),
                     'environment':{'device':'mps','torch':'same','python':'same'}}
    comparison=_comparability(source(base),source(new))
    assert comparison['controlled_single_change']
    assert set(comparison['configuration_differences'])=={'model.global_pooling'}


@pytest.mark.parametrize('shape',[(2,128,16,16),(1,3,7,5),(4,8,1,1)])
def test_initial_pool_equals_gap_and_weights_sum_one_per_image(shape):
    x=torch.randn(shape);pool=GatedSpatialPool(shape[1])
    w=pool.attention_weights(x)
    assert w.shape==(shape[0],1,*shape[2:])
    assert torch.all(w>0)
    torch.testing.assert_close(w.sum((-2,-1)),torch.ones(shape[0],1))
    torch.testing.assert_close(pool(x),x.mean((-2,-1),keepdim=True))
    assert pool(x).shape==(shape[0],shape[1],1,1)


def test_attention_axis_is_spatial_not_batch_channels_or_patients():
    pool=GatedSpatialPool(4);nn.init.normal_(pool.score.weight,std=.3)
    x=torch.randn(3,4,5,7)
    with torch.no_grad():
        together=pool(x);alone=pool(x[:1])
        torch.testing.assert_close(together[:1],alone)
        # Describir mapa como una bolsa de posiciones: permutación conjunta
        # de canales espaciales no cambia resumen, no hay dependencia posicional.
        perm=torch.randperm(35);shuffled=x.flatten(2)[:,:,perm].reshape_as(x)
        torch.testing.assert_close(pool(shuffled),together)
        assert pool.attention_weights(x).std()>0


def test_equation_matches_explicit_gate_score_softmax_weighted_sum():
    pool=GatedSpatialPool(8);nn.init.normal_(pool.score.weight,std=.1)
    x=torch.randn(2,8,3,4)
    scores=pool.score(torch.tanh(pool.value(x))*torch.sigmoid(pool.gate(x)))
    w=torch.softmax(scores.flatten(2),dim=-1).reshape(2,1,3,4)
    expected=(x*w).sum((-2,-1),keepdim=True)
    torch.testing.assert_close(pool.attention_weights(x),w)
    torch.testing.assert_close(pool(x),expected)


def test_cnn_initial_common_weights_same_by_seed_no_checkpoint_reused():
    base=config('E19_weight_decay_001_mac');new=config('E30_gated_attention_mac')
    torch.manual_seed(42);a=BreastPCRNet(base.model).eval()
    torch.manual_seed(42);b=BreastPCRNet(new.model).eval()
    assert set(b.state_dict())-set(a.state_dict())=={
        'global_pool.value.weight','global_pool.value.bias','global_pool.gate.weight',
        'global_pool.gate.bias','global_pool.score.weight'}
    assert all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items())
    x=torch.rand(2,3,256,256)
    with torch.inference_mode():torch.testing.assert_close(a(x),b(x),atol=1e-6,rtol=1e-5)


def test_gradients_learn_gate_after_uniform_initial_score_without_frozen_weights():
    torch.manual_seed(42);pool=GatedSpatialPool(8)
    optimizer=torch.optim.AdamW(pool.parameters(),lr=.001,weight_decay=.001)
    x=torch.randn(2,8,4,4,requires_grad=True)
    first_grads=[]
    for step in range(2):
        optimizer.zero_grad();loss=pool(x).square().sum();loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in pool.parameters())
        assert torch.isfinite(x.grad).all()
        if step==0:
            assert pool.score.weight.grad.abs().sum()>0
            assert pool.value.weight.grad.abs().sum()==0
            assert pool.gate.weight.grad.abs().sum()==0
        else:
            assert pool.value.weight.grad.abs().sum()>0 and pool.gate.weight.grad.abs().sum()>0
        optimizer.step()
    assert pool.attention_weights(x).std()>0


def test_full_cnn_all_parameters_get_finite_gradients_and_output_binary_bce():
    model=BreastPCRNet(config('E30_gated_attention_mac').model)
    x=torch.rand(2,3,64,64);optimizer=torch.optim.AdamW(model.parameters(),lr=.001)
    for _ in range(2):
        optimizer.zero_grad();z=model(x)
        assert z.shape==(2,)
        loss=nn.BCEWithLogitsLoss()(z,torch.tensor([0.,1.]));loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        optimizer.step()
    assert model.global_pool.score.weight.abs().sum()>0


def test_double_precision_gradcheck_and_float32_stability_large_scores():
    pool=GatedSpatialPool(2,2).double();nn.init.normal_(pool.score.weight,std=.1)
    x=torch.randn(1,2,2,2,dtype=torch.float64,requires_grad=True)
    assert torch.autograd.gradcheck(pool,(x,),eps=1e-6,atol=1e-4)
    pool=pool.float();pool.score.weight.data.fill_(10000.)
    x=x.detach().float().requires_grad_();result=pool(x)
    assert torch.isfinite(result).all()
    result.sum().backward();assert torch.isfinite(x.grad).all()


def test_amp_reduction_stable_with_cpu_bfloat16():
    pool=GatedSpatialPool(8);x=torch.randn(2,8,4,4,requires_grad=True)
    with torch.autocast('cpu',dtype=torch.bfloat16):
        result=pool(x)
        assert pool.attention_weights(x).dtype==torch.float32
    result.square().sum().backward()
    assert torch.isfinite(result).all() and torch.isfinite(x.grad).all()


def test_reset_restores_uniform_attention_and_preserves_old_shapes_and_counts():
    model=BreastPCRNet(config('E30_gated_attention_mac').model)
    nn.init.ones_(model.global_pool.score.weight);model.reset_parameters()
    assert torch.count_nonzero(model.global_pool.score.weight)==0
    assert trainable_parameter_count(model)==298273
    assert trainable_parameter_count(model.global_pool)==4144
    steps=trace(model)
    assert sum(s['parameters'] for s in steps)==298273
    backbone=[s for s in steps if s['kind']=='conv']
    assert len(backbone)==8 and backbone[-1]['shape']==(128,16,16)
    assert backbone[-1]['rf']==106
    child=[s for s in steps if s['kind']=='attention_component']
    assert [s['parameters'] for s in child if s['parameters']]==[2064,2064,16]
    assert any(s['label'].startswith('Producto') and s['shape']==(16,16,16) for s in child)
    assert any(s['label']=='Softmax' and s['shape']==(1,256) for s in child)
    assert [s for s in steps if s['kind']=='head'][0]['shape']==(128,1,1)
    assert steps[-1]['shape']==(1,)


def test_new_config_roundtrip_and_legacy_defaults_strictly_load_old_weights():
    new=config('E30_gated_attention_mac')
    assert experiment_config_from_dict(json.loads(json.dumps(new.to_dict())))==new
    old=config('E19_weight_decay_001_mac');raw=old.to_dict();raw['model'].pop('global_pooling')
    restored=experiment_config_from_dict(raw)
    assert restored==old
    a,b=BreastPCRNet(old.model).eval(),BreastPCRNet(restored.model).eval()
    b.load_state_dict(a.state_dict(),strict=True)
    x=torch.rand(1,3,32,32)
    with torch.no_grad():torch.testing.assert_close(a(x),b(x),rtol=0,atol=0)


@pytest.mark.parametrize('channels,hidden',[(0,16),(True,16),(4,0),(4,True),(4,1.5)])
def test_invalid_module_dimensions(channels,hidden):
    with pytest.raises(ValueError):GatedSpatialPool(channels,hidden)


@pytest.mark.parametrize('shape',[(2,8,4),(2,7,4,4),(2,8,0,4),(2,8,4,0)])
def test_invalid_feature_shapes(shape):
    with pytest.raises(ValueError):GatedSpatialPool(8)(torch.zeros(shape))


def test_five_synthetic_web_inputs_are_identical_to_model_and_no_random_in_eval(tmp_path):
    from src.inference import DCEInferenceEngine
    from src.data import load_dce_triplet
    from test_inference import _sources
    cfg=config('E30_gated_attention_mac');model=BreastPCRNet(cfg.model).eval()
    # No científico; demostrar compatibilidad también con atención NO uniforme.
    nn.init.normal_(model.global_pool.score.weight,std=.1)
    path=tmp_path/'synthetic.pt'
    torch.save({'config':cfg.to_dict(),'model_state_dict':model.state_dict(),
                'experiment_id':cfg.experiment_id,'epoch':0},path)
    engine=DCEInferenceEngine.from_checkpoint(path,device_name='cpu')
    sources=_sources(tmp_path)
    with torch.inference_mode():expected=model(load_dce_triplet(sources)[None]).sigmoid().item()
    assert [engine.predict_sources(sources).probability_pcr for _ in range(5)]==pytest.approx([expected]*5)
