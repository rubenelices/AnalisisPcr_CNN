"""Pooling espacial sin arquitectura importada ni cambios del contrato externo."""
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from src.models.architectures import BaseCNNConfig, BreastPCRNet, trainable_parameter_count
from src.reporting.document_architectures import trace
from src.training.experiment_config import load_experiment_config, experiment_config_from_dict

ROOT = Path(__file__).resolve().parents[1]


def test_e43_changes_only_head_grid_and_has_isolated_smoke():
    ref = load_experiment_config(ROOT / "configs/experiments/E19_weight_decay_001_mac.json")
    candidate = load_experiment_config(ROOT / "configs/experiments/E43_spatial_grid2_mac.json")
    smoke = load_experiment_config(ROOT / "configs/experiments/SMOKE_E43_spatial_grid2_mac.json")
    assert candidate.model == replace(ref.model, global_pool_grid_size=2)
    assert candidate.data == ref.data and candidate.training == ref.training
    assert smoke.model == candidate.model
    assert smoke.data == replace(candidate.data, train_patients_per_class=4, validation_patients_per_class=2)
    assert smoke.training == replace(candidate.training, epochs=2, early_stopping_patience=3)
    assert experiment_config_from_dict(candidate.to_dict()) == candidate
    assert len({ref.output_dir, candidate.output_dir, smoke.output_dir}) == 3


@pytest.mark.parametrize("grid,count", [(1,294129), (2,294513), (4,296049)])
def test_shapes_parameters_bins_and_gradient(grid, count):
    config = load_experiment_config(ROOT / "configs/experiments/E19_weight_decay_001_mac.json").model
    model = BreastPCRNet(replace(config, global_pool_grid_size=grid))
    assert trainable_parameter_count(model) == count
    assert model.classifier.in_features == 128 * grid * grid
    feature = torch.arange(256., requires_grad=True).reshape(1,1,16,16)
    pooled = model.global_pool(feature)
    bin_size = 16 // grid
    expected = feature.reshape(1,1,grid,bin_size,grid,bin_size).mean(dim=(3,5))
    assert pooled.shape == (1,1,grid,grid)
    assert torch.equal(pooled, expected)
    x = torch.rand(2,3,32,32)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(model(x), torch.tensor([0.,1.]))
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_grid_preserves_information_gap_loses():
    gap = torch.nn.AdaptiveAvgPool2d(1)
    grid = torch.nn.AdaptiveAvgPool2d(2)
    image = torch.zeros(1,1,16,16)
    image[:,:,:8,:8] = 1
    moved = image.flip(-1)
    assert torch.equal(gap(image), gap(moved))
    assert not torch.equal(grid(image), grid(moved))
    assert torch.equal(grid(image), torch.tensor([[[[1.,0.],[0.,0.]]]]))


def test_grid_documentation_uses_real_shape_rf_and_parameters():
    model = BreastPCRNet(load_experiment_config(ROOT / "configs/experiments/E43_spatial_grid2_mac.json").model)
    steps = trace(model)
    pooled = next(s for s in steps if s["label"].startswith("AvgPool"))
    assert pooled["label"] == "AvgPool\nadaptativo 2×2"
    assert pooled["shape"] == (128,2,2) and pooled["rf"] == 218
    assert pooled["parameters"] == 0 and "bins=8×8" in pooled["detail"]
    assert sum(s["parameters"] for s in steps) == 294513
    assert model.trace_shapes(1)[-3].shape == (1,128,2,2)
    assert model.trace_shapes(1)[-2].shape == (1,512)


@pytest.mark.parametrize("size", [0,-1,3,True,2.0])
def test_invalid_grid_sizes_rejected(size):
    with pytest.raises(ValueError):
        BaseCNNConfig(global_pool_grid_size=size)


@pytest.mark.parametrize("pooling", ["average_max", "gated_attention"])
def test_no_implicit_grid_on_other_heads(pooling):
    with pytest.raises(ValueError):
        BaseCNNConfig(global_pooling=pooling, global_pool_grid_size=2)


def test_old_config_state_dict_and_eval_are_identical_with_explicit_default():
    config = load_experiment_config(ROOT / "configs/experiments/E19_weight_decay_001_mac.json").model
    torch.manual_seed(42)
    old = BreastPCRNet(config).eval()
    torch.manual_seed(42)
    explicit = BreastPCRNet(replace(config, global_pool_grid_size=1)).eval()
    assert old.state_dict().keys() == explicit.state_dict().keys()
    for name, value in old.state_dict().items():
        assert torch.equal(value, explicit.state_dict()[name])
    explicit.load_state_dict(old.state_dict(), strict=True)
    with torch.inference_mode():
        x = torch.rand(2,3,32,32)
        assert torch.equal(old(x), explicit(x))
