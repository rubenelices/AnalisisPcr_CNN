"""Cortes de ejemplo de la web (frontend/public/examples/ y frontend/src/ejemplos.json).

Conserva los tres ejemplos que ya usaba la web y añade uno por cada cohorte × pCR, elegido al azar con
semilla fija entre las pacientes del train público: no se escogen mirando qué predice el modelo. De cada
paciente se toma un corte al azar. Copia los PNG originales sin tocarlos, genera el mapa de realce con la
misma paleta que el resto de la web y guarda la etiqueta pCR (pública en el train) para que el laboratorio
pueda comparar. No lee el test ni la validación privada:

    python -m src.web.web_ejemplos          # y después: npm --prefix frontend run build
"""
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from src.web.web_aumento import a_realce, cargar

ROOT = Path(__file__).resolve().parents[2]
DATOS = ROOT/'breastdcedl'
DEST = ROOT/'frontend/public/examples'
INDICE = ROOT/'frontend/src/ejemplos.json'
SEMILLA = 42
ACTUALES = ('ISPY1_1001_z016', 'ISPY2-923968_z035', 'Breast_MRI_592_z034')
COHORTES = ('duke', 'spy1', 'spy2')


def elegir(samples, patients, semilla=SEMILLA, actuales=ACTUALES):
    """Los ejemplos actuales y, después, un corte al azar de una paciente al azar por cohorte × pCR."""
    tr = samples[samples.split == 'train'].merge(patients[['pid', 'dataset']], left_on='patient_id', right_on='pid')
    rng = np.random.default_rng(semilla)
    elegidos = [tr[tr.sample_id == s].iloc[0] for s in actuales]
    usadas = {f.patient_id for f in elegidos}
    for cohorte in COHORTES:
        for pcr in (0, 1):
            pacientes = sorted(set(tr[(tr.dataset == cohorte) & (tr.pCR == pcr)].patient_id) - usadas)
            paciente = pacientes[rng.integers(len(pacientes))]
            cortes = tr[tr.patient_id == paciente].sort_values('slice_index')
            elegidos.append(cortes.iloc[rng.integers(len(cortes))])
            usadas.add(paciente)
    return elegidos


def build(datos=DATOS, dest=DEST, indice=INDICE):
    samples = pd.read_csv(Path(datos)/'metadata/samples.csv')
    patients = pd.read_csv(Path(datos)/'metadata/patients.csv', usecols=['pid', 'dataset'])
    salida = []
    for fila in elegir(samples, patients):
        carpeta = Path(dest)/fila.sample_id
        carpeta.mkdir(parents=True, exist_ok=True)
        for fase, columna in (('PRE', 'path_pre'), ('EARLY', 'path_early'), ('LATE', 'path_late')):
            shutil.copyfile(Path(datos)/getattr(fila, columna), carpeta/f'{fase}.png')
        imagen = cargar(carpeta)
        if not (carpeta/'REALCE.png').is_file():  # los ya publicados se conservan (difieren solo en redondeo)
            a_realce(imagen).save(carpeta/'REALCE.png')
        salida.append(dict(id=fila.sample_id, paciente=fila.patient_id, posicion=fila.sample_id.rsplit('_', 1)[1],
                           cohorte=fila.dataset, pcr=int(fila.pCR),
                           medias=[round(float(imagen[i].mean()), 3) for i in range(3)]))
    Path(indice).write_text(json.dumps(salida, ensure_ascii=False, indent=1) + '\n')
    return salida


if __name__ == '__main__':
    for e in build():
        print(e['id'], e['cohorte'], 'pCR' if e['pcr'] else 'sin pCR')
