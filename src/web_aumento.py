"""Imágenes de ejemplo del aumento de datos para la web (frontend/public/aumento/).

Aplica a un corte público de ejemplo los mismos aumentos que usó el entrenamiento (`src.augmentation`
y los volteos de `src.train_p01`), sobre las tres fases a la vez, y guarda de cada variante la fase EARLY
y el mapa de realce EARLY − PRE (la misma paleta inferno de la web). No entrena ni lee datos privados:

    python -m src.web_aumento          # y después: npm --prefix frontend run build

Las dos últimas variantes son ilustraciones de lo que NO se usa: la ganancia compartida (E25) y una
ganancia distinta por fase, que falsearía el realce.
"""
from pathlib import Path

import numpy as np
import torch
from matplotlib import cm
from PIL import Image

from src.augmentation import AffineParameters, apply_shared_affine, apply_shared_gain

ROOT = Path(__file__).resolve().parents[1]
EJEMPLO = ROOT/'frontend/public/ejemplo/ISPY1_1001_z016'
DEST = ROOT/'frontend/public/aumento'
PIXELES_3_POR_CIENTO = 256*0.03
VMAX_REALCE = 0.5   # con vmin 0 y paleta inferno reproduce REALCE.png de los ejemplos


def cargar(carpeta=EJEMPLO):
    """Tensor [3,256,256] float32 en [0,1] con PRE, EARLY y LATE, como lo lee el entrenamiento."""
    fases = [np.asarray(Image.open(Path(carpeta)/f'{n}.png').convert('L'), dtype=np.float32)/255 for n in ('PRE', 'EARLY', 'LATE')]
    return torch.from_numpy(np.stack(fases))


def variantes(imagen):
    """Nombre → tensor aumentado. Siempre la misma transformación en las tres fases, salvo `por_fase`."""
    afin = lambda a, x=0.0, y=0.0: apply_shared_affine(imagen, AffineParameters(a, x, y))
    combinado = apply_shared_affine(torch.flip(imagen, dims=(-1,)), AffineParameters(7.0, -PIXELES_3_POR_CIENTO*0.6, PIXELES_3_POR_CIENTO*0.5))
    por_fase = torch.stack([imagen[0]*0.8, imagen[1]*1.25, imagen[2]])
    return {
        'original': imagen,
        'espejo_horizontal': torch.flip(imagen, dims=(-1,)),
        'espejo_vertical': torch.flip(imagen, dims=(-2,)),
        'giro': afin(10.0),
        'traslacion': afin(0.0, PIXELES_3_POR_CIENTO, -PIXELES_3_POR_CIENTO),
        'combinado': combinado,
        'ganancia': apply_shared_gain(imagen, 1.1),
        'ganancia_por_fase': por_fase,
    }


def a_gris(canal):
    return Image.fromarray((canal.clamp(0, 1).numpy()*255).round().astype(np.uint8), mode='L')


def a_realce(imagen):
    d = (imagen[1] - imagen[0]).numpy()
    x = np.clip(d/VMAX_REALCE, 0, 1)
    return Image.fromarray((cm.inferno(x)[..., :3]*255).round().astype(np.uint8))


def build(dest=DEST, carpeta=EJEMPLO):
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    nombres = []
    for nombre, img in variantes(cargar(carpeta)).items():
        a_gris(img[1]).save(dest/f'{nombre}_early.png')
        a_realce(img).save(dest/f'{nombre}_realce.png')
        nombres.append(nombre)
    return nombres


if __name__ == '__main__':
    print(build())
