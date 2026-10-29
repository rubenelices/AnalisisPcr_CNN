"""Carpeta de ensayo para la defensa: pruebas_defensa/ (no se versiona).

Simula lo que hará el profesor: un corte de cada una de cinco pacientes distintas, con los nombres
originales del dataset, más casos con errores a propósito para ver cómo responde la web. Usa solo
pacientes de train que no son los ejemplos de la web; no toca test.

    python -m src.web.preparar_pruebas_defensa
"""
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
DATOS = ROOT/'breastdcedl'
DEST = ROOT/'pruebas_defensa'
SEMILLA = 2026
EJEMPLOS_WEB = {'ISPY1_1001', 'ISPY2-923968', 'Breast_MRI_592'}
COHORTES = ['spy2', 'spy2', 'spy1', 'duke', 'duke']   # parecido al reparto de train
FASES = ('PRE', 'EARLY', 'LATE')


def elegir_cortes():
    muestras = pd.read_csv(DATOS/'metadata/samples.csv')
    pacientes = pd.read_csv(DATOS/'metadata/patients.csv', usecols=['pid', 'dataset'])
    muestras = muestras[(muestras.split == 'train') & ~muestras.patient_id.isin(EJEMPLOS_WEB)]
    muestras = muestras.merge(pacientes, left_on='patient_id', right_on='pid')
    rng = np.random.default_rng(SEMILLA)
    elegidos, usadas = [], set()
    for cohorte in COHORTES:
        candidatas = sorted(set(muestras[muestras.dataset == cohorte].patient_id) - usadas)
        paciente = candidatas[rng.integers(len(candidatas))]
        usadas.add(paciente)
        cortes = muestras[muestras.patient_id == paciente].reset_index(drop=True)
        elegidos.append(cortes.iloc[rng.integers(len(cortes))])
    return elegidos


def copiar_corte(fila, destino, nombre=None):
    destino.mkdir(parents=True, exist_ok=True)
    for fase, col in zip(FASES, ('path_pre', 'path_early', 'path_late')):
        shutil.copy(DATOS/fila[col], destino/f'{nombre or fila.sample_id}_{fase}.png')


def errores(fila, base):
    leer = lambda col: Image.open(DATOS/fila[col])
    d = base/'1_falta_una_fase'; d.mkdir(parents=True)
    for fase, col in zip(FASES[:2], ('path_pre', 'path_early')):
        leer(col).save(d/f'PRUEBA-A_z010_{fase}.png')
    d = base/'2_tamano_512'; d.mkdir()
    for fase, col in zip(FASES, ('path_pre', 'path_early', 'path_late')):
        leer(col).resize((512, 512)).save(d/f'PRUEBA-B_z010_{fase}.png')
    d = base/'3_fase_en_color'; d.mkdir()
    for fase, col in zip(FASES, ('path_pre', 'path_early', 'path_late')):
        img = leer(col).convert('RGB')
        if fase == 'LATE':
            r, g, b = img.split()
            img = Image.merge('RGB', (r, g.point(lambda v: v * 0.6), b.point(lambda v: v * 0.6)))
        img.save(d/f'PRUEBA-C_z010_{fase}.png')
    d = base/'4_no_es_png'; d.mkdir()
    leer('path_pre').convert('RGB').save(d/'PRUEBA-D_z010_PRE.jpg', quality=95)
    for fase, col in zip(FASES[1:], ('path_early', 'path_late')):
        leer(col).save(d/f'PRUEBA-D_z010_{fase}.png')
    d = base/'5_nombre_sin_fase'; d.mkdir()
    for i, col in enumerate(('path_pre', 'path_early', 'path_late'), 1):
        leer(col).save(d/f'imagen_{i}.png')
    d = base/'6_imagen_danada'; d.mkdir()
    for fase, col in zip(FASES, ('path_pre', 'path_early', 'path_late')):
        shutil.copy(DATOS/fila[col], d/f'PRUEBA-F_z010_{fase}.png')
    (d/'PRUEBA-F_z010_EARLY.png').write_bytes(b'esto no es una imagen PNG')


LEEME = """Ensayo de la defensa
====================

1_cinco_pacientes/
  Un corte de cada una de cinco pacientes distintas (I-SPY2, I-SPY1 y Duke), con los nombres
  originales del dataset: <paciente>_<corte>_PRE.png, _EARLY.png y _LATE.png. Es lo que te dará
  el profesor. Pruébalo de tres formas:
    a) Una a una: «Subir imágenes» y elige las tres fases de un corte.
    b) Todas a la vez: arrastra la carpeta entera sobre el visor o a la zona de Pruebas.
    c) Desde Pruebas: «Elegir carpeta».
  En el encabezado, la comprobación de la entrada debe salir en verde y la señal debe recorrer la red.

2_errores/   (cada subcarpeta es un caso; arrástrala sobre el visor)
  1_falta_una_fase    solo PRE y EARLY                → rojo en «3 fases por corte»
  2_tamano_512        imágenes de 512 × 512           → rojo en «256 × 256»
  3_fase_en_color     LATE en color                   → rojo en «escala de grises»
  4_no_es_png         PRE en JPG                      → rojo en «PNG» y en «3 fases» (falta PRE en PNG)
  5_nombre_sin_fase   imagen_1.png, imagen_2.png…     → rojo en «nombre con fase»
  6_imagen_danada     EARLY no es una imagen válida   → rojo en «PNG» (dañado)

Las imágenes son de train público (BreastDCEDL, CC BY-NC 4.0). No subir esta carpeta a GitHub.
"""


def main():
    if DEST.exists():
        shutil.rmtree(DEST)
    cortes = elegir_cortes()
    for fila in cortes:
        copiar_corte(fila, DEST/'1_cinco_pacientes')
    errores(cortes[0], DEST/'2_errores')
    (DEST/'LEEME.txt').write_text(LEEME)
    for fila in cortes:
        print(f'{fila.sample_id:28s} cohorte {fila.dataset}')
    print(DEST)


if __name__ == '__main__':
    main()
