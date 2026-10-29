"""Mapa de oclusión de la web con tensores sintéticos y pesos aleatorios."""

from __future__ import annotations

import io

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

from app import create_app
from src.web.inference import DCEInferenceEngine
from src.web.occlusion_map import occluded_batch, occlusion_map


class RegionSum(nn.Module):
    """Logit = suma de las tres fases dentro de una región fija: solo esa región importa."""

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return image[:, :, 32:48, 64:80].sum(dim=(1, 2, 3)) / 100 - 3


def _image(seed: int = 0) -> torch.Tensor:
    return torch.rand(3, 256, 256, generator=torch.Generator().manual_seed(seed))


def test_each_copy_hides_only_its_square_in_the_three_phases() -> None:
    image = _image() + 0.1  # sin ceros previos: todo cero es del parche
    batch = occluded_batch(image, 16, range(16 * 16))
    for i in (0, 17, 255):
        zeros = batch[i] == 0
        top, left = (i // 16) * 16, (i % 16) * 16
        assert zeros[:, top:top + 16, left:left + 16].all()
        assert zeros.sum() == 3 * 16 * 16
        assert torch.equal(batch[i][:, :top], image[:, :top])


def test_only_the_square_the_model_reads_changes_the_probability() -> None:
    result = occlusion_map(RegionSum(), _image(), torch.device("cpu"), patch=16, batch_size=50)
    deltas = np.array(result.deltas)
    assert result.size == 16 and deltas.shape == (16, 16)
    assert deltas[2, 4] < 0  # región 32:48 × 64:80 = fila 2, columna 4
    deltas[2, 4] = 0
    assert np.abs(deltas).max() < 1e-6


def test_base_probability_matches_the_web_prediction() -> None:
    torch.manual_seed(0)
    model = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.BatchNorm2d(4), nn.ReLU(),
                          nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(4, 1), nn.Flatten(0))
    engine = DCEInferenceEngine(model, torch.device("cpu"))
    image = _image(1)
    result = engine.occlusion_tensor(image)
    assert result.base_probability == pytest.approx(engine.predict_tensor(image).probability_pcr, abs=1e-6)


def test_patch_must_divide_the_image() -> None:
    with pytest.raises(ValueError):
        occlusion_map(RegionSum(), _image(), torch.device("cpu"), patch=15)


def _uploads() -> dict:
    def png(value: int) -> io.BytesIO:
        stream = io.BytesIO()
        Image.fromarray(np.full((256, 256), value, dtype=np.uint8), mode="L").save(stream, format="PNG")
        stream.seek(0)
        return stream
    return {phase: (png(value), f"{phase}.png") for phase, value in (("PRE", 10), ("EARLY", 120), ("LATE", 200))}


def test_endpoint_returns_the_map_and_validates_uploads() -> None:
    app = create_app()
    app.config["INFERENCE_ENGINE"] = DCEInferenceEngine(RegionSum(), torch.device("cpu"))
    client = app.test_client()
    response = client.post("/api/occlusion", data=_uploads())
    assert response.status_code == 200
    body = response.json
    assert body["size"] == 16 and body["patch"] == 16 and len(body["deltas"]) == 16
    assert body["base_probability"] == pytest.approx(
        client.post("/api/predict", data=_uploads()).json["probability_pcr"], abs=1e-6)
    uploads = _uploads()
    del uploads["LATE"]
    assert client.post("/api/occlusion", data=uploads).status_code == 400
