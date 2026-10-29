"""Pipeline unico de inferencia para evaluacion, pruebas y aplicacion web."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import Tensor, nn

from src.models.architectures import BaseCNNConfig, BreastPCRNet
from src.pipeline.data import ImageSource, load_dce_triplet
from src.web.occlusion_map import OcclusionMap, occlusion_map
from src.training.train import choose_device


EDUCATIONAL_NOTICE = (
    "Uso exclusivamente educativo y de investigacion. Este resultado no tiene "
    "validez clinica y no debe usarse para diagnostico ni decisiones terapeuticas."
)


@dataclass(frozen=True)
class InferenceResult:
    probability_pcr: float
    predicted_class: int
    threshold: float
    device: str
    latency_ms: float
    model_sha256: str
    model_version: str
    notice: str = EDUCATIONAL_NOTICE


class DCEInferenceEngine:
    """Convierte PRE/EARLY/LATE con ``load_dce_triplet`` y ejecuta un logit."""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        *,
        threshold: float = 0.5,
        model_sha256: str = "unavailable",
        model_version: str = "unversioned",
    ) -> None:
        if not 0 <= threshold <= 1:
            raise ValueError("threshold debe estar en [0, 1]")
        self.model = model.to(device).eval()
        self.device = device
        self.threshold = threshold
        self.model_sha256 = model_sha256
        self.model_version = model_version

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        *,
        device_name: str = "auto",
        threshold: float | None = None,
        decision_path: str | Path | None = None,
    ) -> "DCEInferenceEngine":
        path = Path(checkpoint_path)
        device = choose_device(device_name)
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        if "model_state_dict" not in checkpoint or "config" not in checkpoint:
            raise ValueError("El checkpoint no contiene modelo y configuracion reproducible")
        raw_config = checkpoint["config"]
        experiment = raw_config.get('experiment', raw_config)
        model_config = BaseCNNConfig(**experiment["model"])
        if checkpoint.get('model_class') == 'PatientFeatureNet_v1':
            from src.models.patient_feature_model import PatientFeatureNet
            model = PatientFeatureNet(model_config)
        else:
            model = BreastPCRNet(model_config)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        checksum = hashlib.sha256(path.read_bytes()).hexdigest()
        resolved_threshold = (
            float(experiment["training"]["threshold"])
            if threshold is None
            else float(threshold)
        )
        if decision_path is not None:
            decision = json.loads(Path(decision_path).read_text())
            if decision.get('schema') != 'e57_public_test_frozen_v1' or decision.get('checkpoint_sha256') != checksum:
                raise ValueError('El protocolo de decisión no corresponde al checkpoint')
            if decision.get('aggregation') != 'mean_cut_probabilities':
                raise ValueError('El protocolo exige otra agregación')
            selected = float(decision['threshold'])
            if threshold is not None and float(threshold) != selected:
                raise ValueError('El umbral solicitado contradice el protocolo fijado')
            resolved_threshold = selected
        experiment_id = checkpoint.get("experiment_id", experiment.get('experiment_id', 'unknown'))
        epoch = checkpoint.get("epoch", "unknown")
        version = f"{experiment_id}@epoch-{epoch}"
        return cls(
            model,
            device,
            threshold=resolved_threshold,
            model_sha256=checksum,
            model_version=version,
        )

    @torch.inference_mode()
    def predict_tensor(self, image: Tensor) -> InferenceResult:
        if image.shape != (3, 256, 256):
            raise ValueError(f"Forma invalida: {tuple(image.shape)}; se esperaba (3, 256, 256)")
        if image.dtype != torch.float32:
            raise ValueError("El tensor debe ser float32")
        if not torch.isfinite(image).all():
            raise ValueError("El tensor contiene valores no finitos")
        start = time.perf_counter()
        logit = self.model(image.unsqueeze(0).to(self.device))
        probability = float(torch.sigmoid(logit).squeeze().cpu())
        latency_ms = (time.perf_counter() - start) * 1_000
        return InferenceResult(
            probability_pcr=probability,
            predicted_class=int(probability >= self.threshold),
            threshold=self.threshold,
            device=str(self.device),
            latency_ms=latency_ms,
            model_sha256=self.model_sha256,
            model_version=self.model_version,
        )

    def occlusion_tensor(self, image: Tensor, patch: int = 16, phases: tuple[int, ...] = (0, 1, 2)) -> OcclusionMap:
        """Mapa de oclusión del corte con el mismo modelo y la misma entrada que la predicción."""
        if image.shape != (3, 256, 256) or image.dtype != torch.float32 or not torch.isfinite(image).all():
            raise ValueError("Se esperaba un tensor float32 finito de forma (3, 256, 256)")
        return occlusion_map(self.model, image, self.device, patch=patch, phases=phases)

    def predict_sources(self, sources: Mapping[str, ImageSource]) -> InferenceResult:
        return self.predict_tensor(load_dce_triplet(sources))

    @torch.inference_mode()
    def predict_patient_tensors(self, images: Tensor) -> InferenceResult:
        """Misma media de probabilidades utilizada en la evaluación por paciente."""
        if images.ndim != 4 or images.shape[1:] != (3, 256, 256) or len(images) == 0:
            raise ValueError('Se esperan uno o más cortes [N,3,256,256]')
        if images.dtype != torch.float32 or not torch.isfinite(images).all():
            raise ValueError('Los cortes deben ser float32 y finitos')
        started = time.perf_counter()
        self.model.eval()
        probabilities = torch.sigmoid(self.model(images.to(self.device)))
        probability = float(probabilities.mean().cpu())
        return InferenceResult(probability_pcr=probability,
            predicted_class=int(probability >= self.threshold), threshold=self.threshold,
            device=str(self.device), latency_ms=(time.perf_counter()-started)*1000,
            model_sha256=self.model_sha256, model_version=self.model_version)


def prediction_as_dict(result: InferenceResult) -> dict[str, Any]:
    return {
        "probability_pcr": result.probability_pcr,
        "predicted_class": result.predicted_class,
        "threshold": result.threshold,
        "device": result.device,
        "latency_ms": result.latency_ms,
        "model_sha256": result.model_sha256,
        "model_version": result.model_version,
        "notice": result.notice,
    }
