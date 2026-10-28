"""E28: LR aplicado, extremos, compatibilidad y protocolo de una variable."""
from dataclasses import replace
import json

import pytest
import torch

from src.architectures import BreastPCRNet, trainable_parameter_count
from src.compare_experiments import _comparability
from src.experiment_config import TrainingConfig, experiment_config_from_dict, load_experiment_config
from src.learning_rate_schedule import LRScheduleConfig, LearningRateSchedule, rate_for_epoch


def config(name):
    return load_experiment_config(f"configs/experiments/{name}.json")


def optimizer():
    return torch.optim.AdamW([torch.nn.Parameter(torch.ones(1))], lr=.001, weight_decay=.001)


def test_only_schedule_policy_changes_from_e19_and_paths_unique():
    old, new = config('E19_weight_decay_001_mac'), config('E28_warmup_cosine_mac')
    assert old.model == new.model and old.data == new.data
    assert new.training == replace(old.training, lr_schedule=new.training.lr_schedule)
    assert new.training.epochs == 50 and new.training.early_stopping_patience == 10
    assert new.training.objective == 'cut' and new.data.train_patients_per_batch is None
    assert not new.data.augmentation.enabled and not new.data.intensity_augmentation.enabled
    assert new.output_dir != old.output_dir and new.checkpoint_dir != old.checkpoint_dir
    assert trainable_parameter_count(BreastPCRNet(new.model)) == 294129
    source = lambda c: {'scientific_run_verified':True, 'config':c.to_dict(),
                        'environment':{'device':'mps','torch':'same','python':'same'}}
    comparison = _comparability(source(old), source(new))
    assert comparison['controlled_single_change'] is True
    assert set(comparison['configuration_differences']) == {'training.lr_schedule'}


def test_smoke_same_policy_covers_warmup_and_first_cosine_epoch():
    new, smoke = config('E28_warmup_cosine_mac'), config('SMOKE_warmup_cosine_mac')
    assert smoke.model == new.model
    assert smoke.training == replace(new.training, epochs=3)
    assert smoke.data == replace(new.data, train_patients_per_class=4, validation_patients_per_class=2)
    assert smoke.training.epochs > smoke.training.lr_schedule.warmup_epochs


def test_full_calendar_boundaries_no_restart_and_minimum_visited_before_earliest_stop():
    c = config('E28_warmup_cosine_mac')
    values = [rate_for_epoch(c.training.learning_rate, c.training.lr_schedule, e) for e in range(1, 51)]
    assert values[0] == pytest.approx(.0001)
    assert values[1] == pytest.approx(.001)
    assert values[2] < values[1]
    assert values[9] == pytest.approx(.00001)
    assert values[9:] == pytest.approx([.00001]*41)
    assert all(a >= b for a,b in zip(values[1:],values[2:]))
    assert 2+8 < 1+c.training.early_stopping_patience


def test_rate_used_by_real_optimizer_is_calendar_and_does_not_depend_on_metric():
    cfg = config('E28_warmup_cosine_mac').training
    opt = optimizer(); controller = LearningRateSchedule(opt, cfg)
    for epoch in range(1, 51):
        used = controller.start_epoch(epoch)
        assert used == opt.param_groups[0]['lr'] == rate_for_epoch(.001, cfg.lr_schedule, epoch)
        opt.zero_grad()
        opt.param_groups[0]['params'][0].sum().backward()
        opt.step()
        controller.finish_epoch(float(epoch % 2))
        assert opt.param_groups[0]['lr'] == used
    state = controller.state_dict()
    assert state['epoch'] == 50 and state['plateau'] is None
    assert state['schedule']['policy'] == 'warmup_cosine'
    json.dumps(state)


def test_old_schedule_exactly_matches_previous_plateau_and_actual_used_rate():
    cfg = config('E19_weight_decay_001_mac').training
    opt, old_opt = optimizer(), optimizer()
    new = LearningRateSchedule(opt, cfg)
    old = torch.optim.lr_scheduler.ReduceLROnPlateau(old_opt, mode='max', factor=.5, patience=3)
    for epoch, score in enumerate([.6]+[.59]*15+[.7]+[.65]*5, start=1):
        assert new.start_epoch(epoch) == old_opt.param_groups[0]['lr']
        new.finish_epoch(score); old.step(score)
        assert opt.param_groups[0]['lr'] == old_opt.param_groups[0]['lr']
        assert new.state_dict()['plateau'] == old.state_dict()


def test_old_serialized_config_keeps_defaults_and_new_config_roundtrips():
    for name in ['E19_weight_decay_001_mac','E28_warmup_cosine_mac']:
        cfg=config(name)
        assert experiment_config_from_dict(json.loads(json.dumps(cfg.to_dict()))) == cfg
    old = config('E19_weight_decay_001_mac').to_dict()
    del old['training']['lr_schedule']
    assert experiment_config_from_dict(old).training.lr_schedule == LRScheduleConfig()


@pytest.mark.parametrize('field,value', [
    ('policy','other'),('warmup_epochs',-1),('warmup_epochs',True),('cosine_epochs',1.5),
    ('start_factor',0),('start_factor',1),('start_factor',float('nan')),
    ('start_factor',True),('min_factor',0),('min_factor',float('inf')),
    ('min_factor',1),('min_factor','0.1'),('warmup_epochs',2),('cosine_epochs',8),
])
def test_schedule_invalid_settings_rejected(field,value):
    with pytest.raises(ValueError): LRScheduleConfig(**{field:value})


@pytest.mark.parametrize('warmup,cosine', [(0,8),(1,8),(2,0)])
def test_warmup_cosine_requires_valid_phase_lengths(warmup,cosine):
    with pytest.raises(ValueError): LRScheduleConfig(policy='warmup_cosine',warmup_epochs=warmup,cosine_epochs=cosine)


@pytest.mark.parametrize('epoch', [0,-1,True,1.5])
def test_bad_epoch_rejected(epoch):
    with pytest.raises(ValueError): rate_for_epoch(.001,config('E28_warmup_cosine_mac').training.lr_schedule,epoch)


def test_nonfinite_base_rate_and_invalid_config_type_and_epoch_order():
    s=config('E28_warmup_cosine_mac').training.lr_schedule
    for rate in [0,float('nan'),float('inf'),True]:
        with pytest.raises(ValueError): rate_for_epoch(rate,s,1)
    with pytest.raises(ValueError): rate_for_epoch(.001,LRScheduleConfig(),1)
    with pytest.raises(ValueError): TrainingConfig(lr_schedule={})
    controller=LearningRateSchedule(optimizer(),config('E28_warmup_cosine_mac').training)
    with pytest.raises(ValueError): controller.start_epoch(2)
    controller.start_epoch(1)
    with pytest.raises(ValueError): controller.start_epoch(1)


def test_synthetic_checkpoint_inference_remains_same_five_times(tmp_path):
    from src.inference import DCEInferenceEngine
    from src.data import load_dce_triplet
    from test_inference import _sources
    cfg=config('E28_warmup_cosine_mac')
    model=BreastPCRNet(cfg.model).eval()
    checkpoint=tmp_path/'synthetic.pt'
    torch.save({'config':cfg.to_dict(),'model_state_dict':model.state_dict(),
                'experiment_id':cfg.experiment_id,'epoch':1},checkpoint)
    engine=DCEInferenceEngine.from_checkpoint(checkpoint,device_name='cpu')
    sources=_sources(tmp_path)
    with torch.inference_mode(): expected=model(load_dce_triplet(sources)[None]).sigmoid().item()
    assert [engine.predict_sources(sources).probability_pcr for _ in range(5)] == pytest.approx([expected]*5)
