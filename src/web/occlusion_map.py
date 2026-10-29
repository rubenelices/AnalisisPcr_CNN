"""Mapa de sensibilidad por oclusión de un corte para la web.

Se tapa con ceros un cuadrado de ``patch × patch`` píxeles en PRE, EARLY y LATE a la
vez (las tres fases siguen alineadas), se vuelve a predecir y se anota el cambio de
probabilidad respecto al corte completo. Los cuadrados no se solapan y cubren los
256 × 256 píxeles, porque E57 mira la imagen entera.

``phases`` permite tapar el cuadrado solo en algunas fases (el laboratorio de la web
compara qué fase mueve más la salida). Tapar una fase sola rompe la relación entre
fases que el modelo vio en entrenamiento: es una sonda, no una situación realista.

El mapa indica qué zonas mueven la salida de este modelo. No localiza el tumor, no es
una máscara ni una explicación causal de la respuesta. Tapar con cero crea una imagen
que el modelo nunca vio en entrenamiento: el cambio mezcla la información quitada con
el efecto del propio parche.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class OcclusionMap:
    base_probability: float
    patch: int
    size: int  # cuadrados por lado
    phases: tuple[int, ...]  # canales tapados (0 PRE, 1 EARLY, 2 LATE)
    deltas: list[list[float]]  # probabilidad tapando el cuadrado [fila][columna] − base
    latency_ms: float


def occluded_batch(image: Tensor, patch: int, indices: range, phases: tuple[int, ...] = (0, 1, 2)) -> Tensor:
    """Copias del corte con el cuadrado ``i`` (orden por filas) tapado en las fases ``phases``."""
    size = image.shape[-1] // patch
    batch = image.unsqueeze(0).repeat(len(indices), 1, 1, 1)
    channels = list(phases)
    for row, i in enumerate(indices):
        top, left = (i // size) * patch, (i % size) * patch
        batch[row, channels, top:top + patch, left:left + patch] = 0
    return batch


@torch.inference_mode()
def occlusion_map(
    model: nn.Module, image: Tensor, device: torch.device, *, patch: int = 16, batch_size: int = 64,
    phases: tuple[int, ...] = (0, 1, 2),
) -> OcclusionMap:
    if image.ndim != 3 or image.shape[0] != 3 or image.shape[1] != image.shape[2]:
        raise ValueError(f"Se esperaba un corte [3, H, H]; llegó {tuple(image.shape)}")
    if patch < 1 or image.shape[-1] % patch:
        raise ValueError("El lado del cuadrado debe dividir el lado de la imagen")
    if not phases or len(set(phases)) != len(phases) or not set(phases) <= {0, 1, 2}:
        raise ValueError("phases debe ser un subconjunto no vacío de (0, 1, 2)")
    start = time.perf_counter()
    model.eval()
    # Misma llamada que predict_tensor: la base coincide con la probabilidad mostrada.
    base = float(torch.sigmoid(model(image.unsqueeze(0).to(device))).squeeze().cpu())
    size = image.shape[-1] // patch
    probabilities = []
    for first in range(0, size * size, batch_size):
        indices = range(first, min(first + batch_size, size * size))
        logits = model(occluded_batch(image, patch, indices, phases).to(device))
        probabilities.append(torch.sigmoid(logits.reshape(-1)).cpu())
    deltas = (torch.cat(probabilities) - base).reshape(size, size)
    return OcclusionMap(
        base_probability=base,
        patch=patch,
        size=size,
        phases=tuple(phases),
        deltas=deltas.tolist(),
        latency_ms=(time.perf_counter() - start) * 1_000,
    )
