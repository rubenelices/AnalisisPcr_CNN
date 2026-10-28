import copy
import json

import pytest
import torch

from src.multimodal_model import MultimodalPCRNet
from src.train_multimodal import load_spec


def test_original_default_preserves_rng_weights():
    from torch import nn
    from src.architectures import BreastPCRNet
    class Legacy(nn.Module):
        def __init__(self,config):
            super().__init__()
            self.image_encoder=BreastPCRNet(config)
            self.image_encoder.classifier=nn.Identity()
            self.clinical_branch=nn.Sequential(nn.Linear(8,16),nn.ReLU())
            self.fusion=nn.Linear(144,1)
            for layer in (self.clinical_branch[0],self.fusion):
                nn.init.xavier_uniform_(layer.weight);nn.init.zeros_(layer.bias)
    cfg=load_spec('configs/multimodal/E58_clinical_fusion_mac.json')[1]
    torch.manual_seed(42); a=Legacy(cfg.model); ra=torch.get_rng_state()
    torch.manual_seed(42); b=MultimodalPCRNet(cfg.model,0); rb=torch.get_rng_state()
    assert torch.equal(ra,rb)
    for key,value in a.state_dict().items(): assert torch.equal(value,b.state_dict()[key])
    assert sum(p.numel() for p in a.parameters())==294289


def test_nonlinear_shapes_count_and_backward():
    torch.set_num_threads(2)
    cfg=load_spec('configs/multimodal/E58_clinical_fusion_mac.json')[1]
    model=MultimodalPCRNet(cfg.model,32)
    assert sum(p.numel() for p in model.parameters())==298817
    z=model(torch.randn(2,3,256,256),torch.randn(2,8))
    assert z.shape==(2,) and torch.isfinite(z).all()
    torch.nn.functional.binary_cross_entropy_with_logits(z,torch.tensor([0.,1.])).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_bad_heads_rejected():
    cfg=load_spec('configs/multimodal/E58_clinical_fusion_mac.json')[1]
    for bad in (True,16,32.,-1):
        with pytest.raises(ValueError): MultimodalPCRNet(cfg.model,bad)


def test_nonlinear_checkpoint_roundtrip(tmp_path):
    from src.multimodal_model import load_multimodal_checkpoint
    from src.clinical_features import ClinicalPreprocessor
    spec,cfg=load_spec('configs/multimodal/E58_clinical_fusion_mac.json')
    spec=copy.deepcopy(spec);spec['fusion_hidden']=32
    model=MultimodalPCRNet(cfg.model,32).eval()
    prep=ClinicalPreprocessor((40.,0.,0.,0.),40.,10.,878)
    path=tmp_path/'new.pt'
    torch.save(dict(model_class='MultimodalPCRNet_nonlinear_v1',config=spec,
                    clinical_preprocessor=prep.to_dict(),model_state_dict=model.state_dict()),path)
    restored,_=load_multimodal_checkpoint(path)
    x=torch.zeros(1,3,256,256); c=torch.zeros(1,8)
    with torch.no_grad(): assert torch.equal(model(x,c),restored(x,c))
