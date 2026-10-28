"""Cambio aislado de operador; dimensiones, pesos y compatibilidad reales."""
from dataclasses import replace
from pathlib import Path
import pytest
import torch
from torch import nn
from src.architectures import BaseCNNConfig, BreastPCRNet, trainable_parameter_count
from src.document_architectures import trace
from src.experiment_config import load_experiment_config, experiment_config_from_dict

ROOT = Path(__file__).resolve().parents[1]

def test_configuration_changes_only_downsampling_operator():
    reference = load_experiment_config(ROOT / "configs/experiments/E19_weight_decay_001_mac.json")
    candidate = load_experiment_config(ROOT / "configs/experiments/E44_average_pool_mac.json")
    smoke = load_experiment_config(ROOT / "configs/experiments/SMOKE_E44_average_pool_mac.json")
    assert candidate.model == replace(reference.model,downsampling_pool="average")
    assert reference.data == candidate.data and reference.training == candidate.training
    assert candidate == experiment_config_from_dict(candidate.to_dict())
    assert smoke.model == candidate.model
    assert smoke.data == replace(candidate.data,train_patients_per_class=4,validation_patients_per_class=2)
    assert smoke.training == replace(candidate.training,epochs=2,early_stopping_patience=3)
    assert len({candidate.output_dir,smoke.output_dir,reference.output_dir}) == 3

def test_initial_weights_are_identical_without_loading_reference():
    reference = load_experiment_config(ROOT / "configs/experiments/E19_weight_decay_001_mac.json").model
    torch.manual_seed(42)
    old = BreastPCRNet(reference)
    after_old = torch.get_rng_state()
    torch.manual_seed(42)
    candidate = BreastPCRNet(replace(reference,downsampling_pool="average"))
    assert torch.equal(after_old,torch.get_rng_state())
    assert old.state_dict().keys() == candidate.state_dict().keys()
    for name,value in old.state_dict().items():
        assert torch.equal(value,candidate.state_dict()[name]), name
    assert trainable_parameter_count(candidate) == trainable_parameter_count(old) == 294129
    candidate.load_state_dict(old.state_dict(),strict=True)

def test_four_average_pools_trace_shapes_rf_and_gradients():
    model = BreastPCRNet(load_experiment_config(ROOT / "configs/experiments/E44_average_pool_mac.json").model)
    assert sum(isinstance(m,nn.AvgPool2d) for m in model.modules()) == 4
    assert not any(isinstance(m,nn.MaxPool2d) for m in model.modules())
    steps = trace(model)
    pools = [s for s in steps if s["kind"] == "pool"]
    assert [s["shape"] for s in pools] == [(16,128,128),(32,64,64),(64,32,32),(128,16,16)]
    assert all(s["label"] == "AvgPool\n2×2" and s["parameters"] == 0 for s in pools)
    assert [s for s in steps if s["kind"] == "conv"][-1]["rf"] == 106
    assert sum(s["parameters"] for s in steps) == 294129
    model.train()
    loss = nn.BCEWithLogitsLoss()(model(torch.rand(2,3,32,32)),torch.tensor([0.,1.]))
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())

def test_default_max_is_exactly_compatible_for_five_synthetic_triplets():
    config = load_experiment_config(ROOT / "configs/experiments/E19_weight_decay_001_mac.json").model
    torch.manual_seed(17)
    old = BreastPCRNet(config).eval()
    torch.manual_seed(17)
    explicit = BreastPCRNet(replace(config,downsampling_pool="max")).eval()
    with torch.inference_mode():
        x = torch.rand(5,3,32,32)
        assert torch.equal(old(x),explicit(x))

@pytest.mark.parametrize("mode", ["median","",None,True])
def test_unknown_pool_rejected(mode):
    with pytest.raises(ValueError,match="downsampling_pool"):
        BaseCNNConfig(downsampling_pool=mode)
