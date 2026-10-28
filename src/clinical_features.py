"""Cuatro variables clínicas prefijadas, estadísticas SOLO de pacientes train.

No usa pCR, cohorte, IDs como features ni datos de test. El ID solo une tablas.
Vacío es ausente, no un cero observado. Estado serializable reutilizable en inferencia.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

CLINICAL_COLUMNS = ('age', 'HR', 'HER2', 'menopause')
CLINICAL_FEATURE_NAMES = tuple(CLINICAL_COLUMNS) + tuple(c + '_missing' for c in CLINICAL_COLUMNS)


def validate_clinical_frame(frame):
    missing = set(CLINICAL_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError('Faltan columnas clínicas: ' + ','.join(sorted(missing)))
    numeric = frame.loc[:, list(CLINICAL_COLUMNS)].apply(pd.to_numeric, errors='raise')
    if np.isinf(numeric.to_numpy(dtype=float)).any():
        raise ValueError('Valores clínicos infinitos')
    ages = numeric.age.dropna()
    if not ages.between(0, 120, inclusive='neither').all():
        raise ValueError('Edad fuera de rango válido (0,120)')
    for c in CLINICAL_COLUMNS[1:]:
        if not numeric[c].dropna().isin([0, 1]).all():
            raise ValueError(c + ' debe ser 0/1 o ausente')
    return numeric


def load_patient_clinical(root, requested_ids, columns=CLINICAL_COLUMNS):
    columns=tuple(columns)
    if not columns or len(set(columns))!=len(columns) or not set(columns).issubset(CLINICAL_COLUMNS):
        raise ValueError('Columnas clínicas de entrada inválidas')
    ids = list(dict.fromkeys(requested_ids))
    if not ids or any(not isinstance(pid, str) or not pid.strip() for pid in ids):
        raise ValueError('IDs solicitados inválidos')
    # Whitelist: ni siquiera cargar objetivos ni metadatos de adquisición como features.
    frame = pd.read_csv(Path(root) / 'metadata/patients.csv', usecols=['pid', *columns])
    # Slots inactivos constantes: no leer ni ajustar estadísticas de variables excluidas.
    for column in CLINICAL_COLUMNS:
        if column not in columns:frame[column]=np.nan if column=='age' else 0.
    frame = frame.loc[frame.pid.isin(ids)].copy()
    if frame.pid.duplicated().any() or frame.pid.isna().any():
        raise ValueError('Clave clínica duplicada o ausente')
    if set(frame.pid) != set(ids):
        raise ValueError('Hay pacientes sin registro clínico; no inventar la unión')
    validate_clinical_frame(frame)
    return frame.set_index('pid').loc[ids].reset_index()


@dataclass(frozen=True)
class ClinicalPreprocessor:
    imputation_values: tuple[float, ...]
    age_mean: float
    age_std: float
    train_patient_count: int
    active_columns: tuple[str, ...] = CLINICAL_COLUMNS

    def __post_init__(self):
        if (not self.active_columns or len(set(self.active_columns)) != len(self.active_columns)
                or not set(self.active_columns).issubset(CLINICAL_COLUMNS)):
            raise ValueError('Ablación clínica inválida')

    @classmethod
    def fit(cls, train_patients, active_columns=CLINICAL_COLUMNS):
        if ('pid' not in train_patients or train_patients.empty
                or train_patients.pid.duplicated().any() or train_patients.pid.isna().any()):
            raise ValueError('Fit exige una fila por paciente train')
        numeric = validate_clinical_frame(train_patients)
        values = []
        for c in CLINICAL_COLUMNS:
            observed = numeric[c].dropna()
            # Fallback fijo solo si un campo entero falta en train; indicador conserva ausencia.
            value = (observed.median() if c == 'age' else observed.mode().iloc[0]) if len(observed) else 0.
            values.append(float(value))
        ages = numeric.age.fillna(values[0]).to_numpy(dtype=float)
        std = float(ages.std(ddof=0))
        return cls(tuple(values), float(ages.mean()), std if std > 1e-6 else 1., len(numeric),tuple(active_columns))

    def transform(self, frame):
        # Inferencia de ablaciones: solo exigir columnas realmente activas.
        absent=set(CLINICAL_COLUMNS)-set(frame.columns)
        if absent:
            frame=frame.copy()
            for column in absent-set(self.active_columns):frame[column]=0.
        numeric = validate_clinical_frame(frame)
        missing = numeric.isna().to_numpy(dtype=np.float32)
        filled = numeric.fillna(dict(zip(CLINICAL_COLUMNS, self.imputation_values))).to_numpy(dtype=float)
        filled[:, 0] = (filled[:, 0] - self.age_mean) / self.age_std
        result = np.concatenate([filled.astype(np.float32), missing], axis=1)
        # Ablación: desactivar valor E indicador; mismos 8 slots y misma red/RNG.
        for i,c in enumerate(CLINICAL_COLUMNS):
            if c not in self.active_columns:
                result[:,[i,i+4]]=0
        if not np.isfinite(result).all():
            raise ValueError('Features clínicas no finitas')
        return torch.from_numpy(result)

    def to_dict(self):
        state = dict(schema='clinical_four_v1', columns=list(CLINICAL_COLUMNS),
                    feature_names=list(CLINICAL_FEATURE_NAMES), imputation_values=list(self.imputation_values),
                    age_mean=self.age_mean, age_std=self.age_std, train_patient_count=self.train_patient_count,
                    fit_scope='internal_train_patients_only')
        if self.active_columns != CLINICAL_COLUMNS:
            state['active_columns']=list(self.active_columns)
        return state

    @classmethod
    def from_dict(cls, state):
        if (state.get('schema') != 'clinical_four_v1' or state.get('columns') != list(CLINICAL_COLUMNS)
                or state.get('feature_names') != list(CLINICAL_FEATURE_NAMES)
                or state.get('fit_scope') != 'internal_train_patients_only'):
            raise ValueError('Esquema clínico incompatible')
        values = state['imputation_values']
        numbers = [*values, state['age_mean'], state['age_std']]
        if len(values) != 4 or not np.isfinite(numbers).all() or state['age_std'] <= 0:
            raise ValueError('Estado clínico no válido')
        if type(state['train_patient_count']) is not int or state['train_patient_count'] < 1:
            raise ValueError('Recuento train clínico inválido')
        return cls(tuple(values), state['age_mean'], state['age_std'], state['train_patient_count'],
                   tuple(state.get('active_columns',CLINICAL_COLUMNS)))


class ClinicalImageDataset(Dataset):
    """Mismo cargador de imágenes, clínica repetida por corte sin reajustar estadísticas."""
    def __init__(self, image_dataset, clinical_patients, preprocessor):
        if clinical_patients.pid.duplicated().any():
            raise ValueError('No duplicar pacientes al unir clínica con cortes')
        features = preprocessor.transform(clinical_patients)
        self.clinical = dict(zip(clinical_patients.pid, features))
        self.images = image_dataset
        if not set(image_dataset.rows.patient_id).issubset(self.clinical):
            raise ValueError('Falta clínica para alguna imagen')

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        sample = self.images[index]
        sample['clinical'] = self.clinical[sample['patient_id']].clone()
        return sample
