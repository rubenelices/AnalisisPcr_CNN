import copy
import json
from pathlib import Path
import torch
import pytest
from src.training.train_multimodal import load_spec,build_optimizer
from src.models.multimodal_model import MultimodalPCRNet
from src.training.learning_rate_schedule import LearningRateSchedule


def test_original_optimizer_identical():
    spec,cfg=load_spec('configs/multimodal/E58_clinical_fusion_mac.json')
    torch.manual_seed(42);model=MultimodalPCRNet(cfg.model)
    opt=build_optimizer(model,cfg,spec)
    old=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001)
    assert opt.state_dict()==old.state_dict()


def test_two_groups_cover_all_params_and_preserve_ratio_on_plateau():
    spec,cfg=load_spec('experiments/D06_lr_por_rama/config.json')
    model=MultimodalPCRNet(cfg.model); opt=build_optimizer(model,cfg,spec)
    ids=[id(p) for g in opt.param_groups for p in g['params']]
    assert len(ids)==len(set(ids))==len(list(model.parameters()))
    assert set(ids)=={id(p) for p in model.parameters()}
    assert [g['lr'] for g in opt.param_groups]==[.0001,.001]
    sch=LearningRateSchedule(opt,cfg.training)
    for epoch in range(1,7):sch.start_epoch(epoch);sch.finish_epoch(.7)
    assert [g['lr'] for g in opt.param_groups]==[.00005,.0005]
    assert all(g['weight_decay']==.001 for g in opt.param_groups)
    state=copy.deepcopy(opt.state_dict()); new=build_optimizer(model,cfg,spec);new.load_state_dict(state)
    assert [g['lr'] for g in new.param_groups]==[.00005,.0005]


@pytest.mark.parametrize('value',[0,-1,2,True,float('inf')])
def test_invalid_ratio_rejected(tmp_path,value):
    s=json.loads(Path('configs/multimodal/E58_clinical_fusion_mac.json').read_text());s['image_lr_factor']=value
    path=tmp_path/'bad.json';path.write_text(json.dumps(s))
    with pytest.raises(ValueError):load_spec(path)
