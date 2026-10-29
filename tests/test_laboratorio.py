"""Laboratorio de la web y oclusión por fase, con tensores sintéticos y modelos de juguete."""

from __future__ import annotations

import io

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

from app import create_app
from src.pipeline.augmentation import AffineParameters, apply_shared_affine
from src.web.inference import DCEInferenceEngine
from src.web.laboratorio import Transformacion, aplicar, leer_transformacion
from src.web.occlusion_map import occluded_batch, occlusion_map


class Realce(nn.Module):
    """Logit = realce medio (EARLY − PRE): sin contraste, la probabilidad cae a sigmoid(0)."""

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return (image[:, 1] - image[:, 0]).mean(dim=(1, 2)) * 10


def _image(seed: int = 0) -> torch.Tensor:
    return torch.rand(3, 256, 256, generator=torch.Generator().manual_seed(seed))


def test_empty_form_is_the_identity() -> None:
    t = leer_transformacion({})
    assert t.identidad and t.fuera_de_rango() == []
    image = _image()
    assert torch.equal(aplicar(image, t), image)


@pytest.mark.parametrize("campo, valor", [
    ("giro", "46"), ("giro", "abc"), ("traslacion_x", "0.2"), ("ganancia", "1.3"), ("ganancia_EARLY", "2"),
    ("espejo_horizontal", "si"), ("tapar", "RGB"), ("giro", "nan"),
])
def test_out_of_range_values_are_rejected_not_clipped(campo: str, valor: str) -> None:
    with pytest.raises(ValueError):
        leer_transformacion({campo: valor})


def test_geometry_matches_the_training_order_flip_then_affine() -> None:
    image = _image(1)
    t = leer_transformacion({"espejo_horizontal": "1", "giro": "7", "traslacion_x": "0.02", "traslacion_y": "-0.01"})
    esperado = apply_shared_affine(torch.flip(image, dims=(-1,)).contiguous(), AffineParameters(7.0, 0.02 * 256, -0.01 * 256))
    assert torch.allclose(aplicar(image, t), esperado)


def test_shared_geometry_keeps_the_phases_aligned() -> None:
    plano = torch.rand(256, 256, generator=torch.Generator().manual_seed(2))
    image = torch.stack([plano, plano * 0.5, plano * 0.25])
    salida = aplicar(image, Transformacion(espejo_vertical=True, giro=-20.0, traslacion_x=0.1))
    assert torch.allclose(salida[1], salida[0] * 0.5, atol=1e-6) and torch.allclose(salida[2], salida[0] * 0.25, atol=1e-6)


def test_phase_probes() -> None:
    image = _image(3)
    sin = aplicar(image, Transformacion(sin_contraste=True))
    assert torch.equal(sin[1], image[0]) and torch.equal(sin[2], image[0])
    tapada = aplicar(image, Transformacion(tapar="EARLY"))
    assert (tapada[1] == 0).all() and torch.equal(tapada[0], image[0]) and torch.equal(tapada[2], image[2])
    por_fase = aplicar(image, Transformacion(ganancia_fase=(0.8, 1.25, 1.0)))
    assert torch.allclose(por_fase[0], image[0] * 0.8) and torch.allclose(por_fase[1], image[1] * 1.25)


def test_input_is_never_modified() -> None:
    image = _image(4)
    copia = image.clone()
    aplicar(image, Transformacion(espejo_horizontal=True, giro=5.0, ganancia=1.1, tapar="LATE", sin_contraste=True))
    assert torch.equal(image, copia)


def test_warnings_outside_the_moderate_range() -> None:
    assert Transformacion(espejo_horizontal=True, giro=10.0, traslacion_x=0.03).fuera_de_rango() == []
    assert Transformacion(ganancia=1.1).fuera_de_rango() == []
    avisos = Transformacion(giro=20.0, traslacion_y=0.05, ganancia=1.2, tapar="PRE").fuera_de_rango()
    assert len(avisos) == 4


def test_occlusion_can_hide_a_single_phase() -> None:
    image = _image(5) + 0.1
    batch = occluded_batch(image, 16, range(4), phases=(1,))
    assert (batch[0, 1, :16, :16] == 0).all()
    assert torch.equal(batch[0, 0], image[0]) and torch.equal(batch[0, 2], image[2])
    # Tapar solo LATE no mueve un modelo que solo mira EARLY − PRE.
    result = occlusion_map(Realce(), image, torch.device("cpu"), patch=32, phases=(2,))
    assert result.phases == (2,) and np.abs(np.array(result.deltas)).max() < 1e-6
    with pytest.raises(ValueError):
        occlusion_map(Realce(), image, torch.device("cpu"), phases=(3,))


def _uploads() -> dict:
    def png(value: int) -> io.BytesIO:
        stream = io.BytesIO()
        Image.fromarray(np.full((256, 256), value, dtype=np.uint8), mode="L").save(stream, format="PNG")
        stream.seek(0)
        return stream
    return {phase: (png(value), f"{phase}.png") for phase, value in (("PRE", 10), ("EARLY", 120), ("LATE", 200))}


@pytest.fixture
def client():
    app = create_app()
    app.config["INFERENCE_ENGINE"] = DCEInferenceEngine(Realce(), torch.device("cpu"))
    return app.test_client()


def test_laboratory_endpoint(client) -> None:
    base = client.post("/api/predict", data=_uploads()).json["probability_pcr"]
    igual = client.post("/api/laboratorio", data=_uploads()).json
    assert igual["probability_pcr"] == pytest.approx(base, abs=1e-6)
    assert set(igual["fases"]) == {"PRE", "EARLY", "LATE"} and igual["fases"]["PRE"].startswith("data:image/png;base64,")
    sin = client.post("/api/laboratorio", data=dict(_uploads(), sin_contraste="1")).json
    assert sin["probability_pcr"] == pytest.approx(0.5, abs=1e-6) and sin["fuera_de_rango"]
    assert client.post("/api/laboratorio", data=dict(_uploads(), giro="90")).status_code == 400


def test_occlusion_endpoint_options(client) -> None:
    body = client.post("/api/occlusion", data=dict(_uploads(), patch="32", fase="EARLY")).json
    assert body["size"] == 8 and body["patch"] == 32 and body["fase"] == "EARLY"
    assert client.post("/api/occlusion", data=dict(_uploads(), patch="12")).status_code == 400
    assert client.post("/api/occlusion", data=dict(_uploads(), fase="ROJO")).status_code == 400
