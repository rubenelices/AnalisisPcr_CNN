import json
from pathlib import Path
import pytest
from src.training.train_multimodal import load_spec, build_optimizer
from src.models.multimodal_model import MultimodalPCRNet


def test_image_decay_only_and_complete_partition():
    spec,cfg=load_spec('experiments/D12_wd_imagen001/config.json')
    model=MultimodalPCRNet(cfg.model)
    opt=build_optimizer(model,cfg,spec)
    assert [g['weight_decay'] for g in opt.param_groups]==[.01,.001]
    assert [g['lr'] for g in opt.param_groups]==[.001,.001]
    ids=[id(p) for g in opt.param_groups for p in g['params']]
    assert len(ids)==len(set(ids))==len(list(model.parameters()))
    assert set(ids)=={id(p) for p in model.parameters()}
    assert {id(p) for p in opt.param_groups[0]['params']}=={id(p) for p in model.image_encoder.parameters()}


@pytest.mark.parametrize('value',[0,-1,True,float('inf'),float('nan'),'10'])
def test_invalid_image_decay(tmp_path,value):
    spec=json.loads(Path('configs/multimodal/E58_clinical_fusion_mac.json').read_text())
    spec['image_weight_decay_factor']=value
    path=tmp_path/'invalid.json';path.write_text(json.dumps(spec))
    with pytest.raises(ValueError):load_spec(path)


def test_clinical_only_rejects_image_decay(tmp_path):
    spec=json.loads(Path('experiments/D08_clinica_sola/config.json').read_text())
    spec['image_weight_decay_factor']=10
    path=tmp_path/'invalid.json';path.write_text(json.dumps(spec))
    with pytest.raises(ValueError):load_spec(path)
