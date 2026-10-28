"""Pruebas de que evaluacion y web comparten la inferencia PRE/EARLY/LATE."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

from src.data import load_dce_triplet
from src.inference import DCEInferenceEngine, prediction_as_dict


class MeanLogit(nn.Module):
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return images.mean(dim=(1, 2, 3))


def _sources(tmp_path: Path) -> dict[str, Path]:
    result = {}
    for phase, value in zip(("PRE", "EARLY", "LATE"), (0, 127, 255)):
        path = tmp_path / f"{phase}.png"
        Image.fromarray(np.full((256, 256), value, dtype=np.uint8), mode="L").save(path)
        result[phase] = path
    return result


def test_sources_and_preprocessed_tensor_produce_same_prediction(tmp_path: Path) -> None:
    sources = _sources(tmp_path)
    model = MeanLogit()
    engine = DCEInferenceEngine(
        model,
        torch.device("cpu"),
        threshold=0.6,
        model_sha256="abc",
        model_version="test@epoch-1",
    )
    from_sources = engine.predict_sources(sources)
    from_tensor = engine.predict_tensor(load_dce_triplet(sources))
    assert from_sources.probability_pcr == pytest.approx(from_tensor.probability_pcr)
    assert from_sources.predicted_class == 1
    assert model.training is False
    payload = prediction_as_dict(from_sources)
    assert payload["model_sha256"] == "abc"
    assert "no tiene validez clinica" in payload["notice"]


def test_inference_rejects_wrong_tensor_contract() -> None:
    engine = DCEInferenceEngine(MeanLogit(), torch.device("cpu"))
    with pytest.raises(ValueError, match="Forma invalida"):
        engine.predict_tensor(torch.zeros(3, 128, 128))


def test_phase_representation_is_restored_from_checkpoint_for_web(tmp_path: Path) -> None:
    from src.architectures import BreastPCRNet
    from src.experiment_config import load_experiment_config

    root = Path(__file__).resolve().parents[1]
    config = load_experiment_config(root / "configs/experiments/E13_phase_differences.json")
    model = BreastPCRNet(config.model).eval()
    path = tmp_path / "synthetic.pt"
    torch.save({"model_state_dict": model.state_dict(), "config": config.to_dict(),
                "experiment_id": config.experiment_id, "epoch": 1}, path)
    engine = DCEInferenceEngine.from_checkpoint(path, device_name="cpu")
    assert engine.model.config.input_representation == "pre_differences"
    sources = _sources(tmp_path)
    image = load_dce_triplet(sources)
    with torch.inference_mode():
        expected = torch.sigmoid(model(image.unsqueeze(0))).item()
    from_paths = engine.predict_sources(sources)
    from_bytes = engine.predict_sources({k: v.read_bytes() for k, v in sources.items()})
    assert from_paths.probability_pcr == pytest.approx(expected)
    assert from_bytes.probability_pcr == pytest.approx(expected)
    assert engine.model.training is False
