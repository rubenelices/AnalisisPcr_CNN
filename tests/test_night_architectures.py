"""Ablaciones propias: formas, gradientes, compatibilidad y contrato de web."""
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from torch import nn

from src.architectures import BaseCNNConfig, BreastPCRNet, GlobalAverageMaxPool, trainable_parameter_count
from src.document_architectures import trace
from src.experiment_config import load_experiment_config, experiment_config_from_dict
from src.inference import DCEInferenceEngine

ROOT = Path(__file__).resolve().parents[1]
CASES = [("E21_group_norm", "group_norm_groups", 294129),
         ("E22_average_max", "global_pooling", 294257),
         ("E23_compact", "channels", 73913)]


@pytest.mark.parametrize("name,field,count", CASES)
def test_ablations_change_one_model_variable_and_keep_budget(name, field, count):
    base = load_experiment_config(ROOT / "configs/experiments/E19_weight_decay_001_mac.json")
    config = load_experiment_config(ROOT / f"configs/experiments/{name}.json")
    assert config.data == base.data and config.training == base.training
    assert {k for k, v in config.model.to_dict().items() if v != base.model.to_dict()[k]} == {field}
    assert config.output_dir != base.output_dir and config.checkpoint_dir != base.checkpoint_dir
    smoke = load_experiment_config(ROOT / f"configs/experiments/SMOKE_{name}.json")
    assert smoke.model == config.model and smoke.training.epochs == 2
    model = BreastPCRNet(config.model)
    images = torch.rand(2, 3, 256, 256)
    loss = nn.BCEWithLogitsLoss()(model(images), torch.tensor([0., 1.]))
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert trainable_parameter_count(model) == count
    steps = trace(model)
    assert sum(s["parameters"] for s in steps) == count
    block_steps = [s["kind"] for s in steps if s["kind"] in {"conv", "normalization", "activation", "pool"}]
    assert block_steps == ["conv", "normalization", "activation", "pool", "conv", "normalization", "activation"] * 4
    assert [s for s in steps if s["kind"] == "conv"][-1]["rf"] == 106
    assert steps[-1]["shape"] == (1,)


def test_groupnorm_has_no_batch_statistics_and_is_sample_independent():
    model = BreastPCRNet(BaseCNNConfig(group_norm_groups=4, dropout=0)).eval()
    assert sum(isinstance(m, nn.GroupNorm) for m in model.modules()) == 8
    assert not any(isinstance(m, nn.BatchNorm2d) for m in model.modules())
    images = torch.rand(2, 3, 32, 32)
    with torch.inference_mode():
        together = model(images)
        alone = model(images[:1])
    torch.testing.assert_close(together[:1], alone, atol=1e-5, rtol=1e-5)


def test_average_max_is_parallel_concatenation_not_sequential_pool():
    x = torch.tensor([[[[1., 3.], [5., 7.]], [[-2., -4.], [-6., -8.]]]])
    actual = GlobalAverageMaxPool()(x)
    torch.testing.assert_close(actual.flatten(), torch.tensor([4., -5., 7., -2.]))
    assert actual.shape == (1, 4, 1, 1)


@pytest.mark.parametrize("name,field,count", CASES)
def test_new_checkpoint_restores_same_web_probability(tmp_path, name, field, count):
    config = load_experiment_config(ROOT / f"configs/experiments/{name}.json")
    assert experiment_config_from_dict(config.to_dict()) == config
    model = BreastPCRNet(config.model).eval()
    path = tmp_path / "synthetic.pt"
    torch.save({"config": config.to_dict(), "model_state_dict": model.state_dict(),
                "experiment_id": name, "epoch": 0}, path)
    engine = DCEInferenceEngine.from_checkpoint(path, device_name="cpu")
    image = torch.rand(3, 256, 256)
    with torch.inference_mode():
        expected = torch.sigmoid(model(image.unsqueeze(0))).item()
    assert engine.predict_tensor(image).probability_pcr == pytest.approx(expected, abs=1e-6)


def test_legacy_config_defaults_preserve_keys_weights_and_predictions():
    config = load_experiment_config(ROOT / "configs/experiments/E19_weight_decay_001_mac.json")
    raw = config.to_dict()
    raw["model"].pop("group_norm_groups")
    raw["model"].pop("global_pooling")
    restored = experiment_config_from_dict(raw)
    assert restored == config
    torch.manual_seed(12)
    a = BreastPCRNet(config.model).eval()
    torch.manual_seed(12)
    b = BreastPCRNet(restored.model).eval()
    b.load_state_dict(a.state_dict(), strict=True)
    assert a.state_dict().keys() == b.state_dict().keys()
    x = torch.rand(1, 3, 32, 32)
    with torch.inference_mode():
        torch.testing.assert_close(a(x), b(x), rtol=0, atol=0)


@pytest.mark.parametrize("groups", [0, -1, 3, True, 2.5])
def test_group_count_must_be_valid_divisor(groups):
    with pytest.raises(ValueError, match="group_norm_groups"):
        BaseCNNConfig(group_norm_groups=groups)


def test_ambiguous_normalization_and_unknown_pool_rejected():
    with pytest.raises(ValueError):
        BaseCNNConfig(use_batch_norm=False, group_norm_groups=4)
    with pytest.raises(ValueError):
        BaseCNNConfig(global_pooling="mystery")
