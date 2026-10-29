"""Laboratorio de la web: transforma un corte y vuelve a predecir.

Los aumentos geométricos y la ganancia usan las mismas funciones que el entrenamiento
(``torch.flip`` y ``src.pipeline.augmentation``), en el mismo orden que el entrenamiento
con aumentos: espejos, giro y traslación con una sola rejilla para las tres fases, y
después la ganancia compartida. E57, el modelo entregado, se entrenó sin aumentos: aquí
se comprueba si su predicción se mantiene ante transformaciones que no vio.

Las manipulaciones de fases (quitar el contraste, tapar una fase, ganancia distinta por
fase) no se usaron nunca en entrenamiento. Sirven para preguntar al modelo qué usa: por
ejemplo, si la probabilidad cambia al copiar PRE en EARLY y LATE, el modelo depende del
realce. Rompen la relación entre fases de una resonancia real.
"""

from __future__ import annotations

import base64
import io
import math
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import torch
from PIL import Image
from torch import Tensor

from src.pipeline.augmentation import AffineParameters, apply_shared_affine, apply_shared_gain

FASES = ("PRE", "EARLY", "LATE")
LADO = 256

# Rango moderado: el que usó el entrenamiento con aumentos (src/training/train_p01.py). Fuera de él, la web avisa.
RANGO = {"giro": 10.0, "traslacion": 0.03, "ganancia": 0.1}
# Límites del laboratorio: algo más amplios para ver qué pasa fuera de lo entrenado.
LIMITES = {"giro": 45.0, "traslacion": 0.15, "ganancia": (0.75, 1.25), "ganancia_fase": (0.5, 1.5)}


@dataclass(frozen=True)
class Transformacion:
    espejo_horizontal: bool = False
    espejo_vertical: bool = False
    giro: float = 0.0              # grados; positivo, antihorario
    traslacion_x: float = 0.0      # fracción del lado; positivo, a la derecha
    traslacion_y: float = 0.0      # fracción del lado; positivo, hacia abajo
    ganancia: float = 1.0          # factor común a las tres fases
    sin_contraste: bool = False    # EARLY y LATE pasan a ser copias de PRE
    tapar: str | None = None       # fase que se pone a cero
    ganancia_fase: tuple[float, float, float] = (1.0, 1.0, 1.0)  # ilustración: falsea el realce

    @property
    def identidad(self) -> bool:
        return self == Transformacion()

    def fuera_de_rango(self) -> list[str]:
        """Cambios que salen del rango moderado de aumento, para avisar en la web."""
        avisos = []
        if abs(self.giro) > RANGO["giro"]:
            avisos.append(f"giro de {self.giro:g}°, fuera de ±{RANGO['giro']:g}°")
        if max(abs(self.traslacion_x), abs(self.traslacion_y)) > RANGO["traslacion"]:
            avisos.append(f"desplazamiento mayor que ±{RANGO['traslacion'] * 100:g} %")
        if abs(self.ganancia - 1) > RANGO["ganancia"] + 1e-9:
            avisos.append(f"ganancia de × {self.ganancia:g}, fuera de ±{RANGO['ganancia'] * 100:g} %")
        if self.sin_contraste or self.tapar or self.ganancia_fase != (1.0, 1.0, 1.0):
            avisos.append("manipulación de fases: no existe en una resonancia real")
        return avisos


def _numero(formulario: Mapping[str, str], clave: str, defecto: float, minimo: float, maximo: float) -> float:
    texto = formulario.get(clave)
    if texto in (None, ""):
        return defecto
    try:
        valor = float(texto)
    except ValueError as exc:
        raise ValueError(f"{clave} debe ser un número") from exc
    if not math.isfinite(valor) or not minimo <= valor <= maximo:
        raise ValueError(f"{clave} debe estar entre {minimo:g} y {maximo:g}")
    return valor


def _booleano(formulario: Mapping[str, str], clave: str) -> bool:
    texto = formulario.get(clave, "")
    if texto not in ("", "0", "1", "true", "false"):
        raise ValueError(f"{clave} debe ser 0 o 1")
    return texto in ("1", "true")


def leer_transformacion(formulario: Mapping[str, str]) -> Transformacion:
    """Valida los campos del formulario de la web. Cualquier valor fuera de rango es un error, no se recorta."""
    giro, tras = LIMITES["giro"], LIMITES["traslacion"]
    tapar = formulario.get("tapar") or None
    if tapar is not None and tapar not in FASES:
        raise ValueError(f"tapar debe ser una de {', '.join(FASES)}")
    return Transformacion(
        espejo_horizontal=_booleano(formulario, "espejo_horizontal"),
        espejo_vertical=_booleano(formulario, "espejo_vertical"),
        giro=_numero(formulario, "giro", 0.0, -giro, giro),
        traslacion_x=_numero(formulario, "traslacion_x", 0.0, -tras, tras),
        traslacion_y=_numero(formulario, "traslacion_y", 0.0, -tras, tras),
        ganancia=_numero(formulario, "ganancia", 1.0, *LIMITES["ganancia"]),
        sin_contraste=_booleano(formulario, "sin_contraste"),
        tapar=tapar,
        ganancia_fase=tuple(_numero(formulario, f"ganancia_{f}", 1.0, *LIMITES["ganancia_fase"]) for f in FASES),
    )


def aplicar(imagen: Tensor, t: Transformacion) -> Tensor:
    """Corte [3,256,256] transformado. No modifica la entrada ni recorta valores: la red recibe el tensor tal cual."""
    if imagen.shape != (3, LADO, LADO) or imagen.dtype != torch.float32:
        raise ValueError("Se esperaba un tensor float32 [3,256,256]")
    x = imagen
    # Mismo orden que SyncFlipAffine: espejos y después una rejilla afín común a las tres fases.
    if t.espejo_horizontal:
        x = torch.flip(x, dims=(-1,))
    if t.espejo_vertical:
        x = torch.flip(x, dims=(-2,))
    x = apply_shared_affine(x.contiguous(), AffineParameters(t.giro, t.traslacion_x * LADO, t.traslacion_y * LADO))
    x = apply_shared_gain(x, t.ganancia)
    # Sondas sobre las fases: nunca usadas para entrenar.
    if t.sin_contraste:
        x = x[0].expand(3, LADO, LADO)
    if t.ganancia_fase != (1.0, 1.0, 1.0):
        x = x * x.new_tensor(t.ganancia_fase).view(3, 1, 1)
    if t.tapar:
        x = x.clone()
        x[FASES.index(t.tapar)] = 0
    return x.contiguous()


def fase_png(canal: Tensor) -> str:
    """Una fase como PNG en gris (data URL) para enseñarla. Se recorta a [0, 1] solo para dibujar."""
    pixeles = (canal.clamp(0, 1).numpy() * 255).round().astype(np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(pixeles, mode="L").save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def resumen(imagen: Tensor, t: Transformacion) -> dict[str, Any]:
    """Fases transformadas y avisos para la web (la probabilidad la añade el servidor)."""
    return {
        "fases": {f: fase_png(imagen[i]) for i, f in enumerate(FASES)},
        "medias": {f: float(imagen[i].mean()) for i, f in enumerate(FASES)},
        "fuera_de_rango": t.fuera_de_rango(),
    }
