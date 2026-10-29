"""Protocolo N01: preparación y controles sin entrenamiento ni acceso a imágenes."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src import source_path
from src.evaluation.metrics import binary_metrics

ROOT = Path(__file__).resolve().parents[2]
CONFIG = Path('configs/new_approach/N01_e57_bce.json')
TECH = Path('reports/new_approach/N01_e57_bce')
WEIGHTS = Path('checkpoints/new_approach/N01_e57_bce')
SOURCES = (
    'new_approach_protocol.py', 'train_new_approach.py', 'new_approach_n01_results.py',
    'patient_feature_model.py', 'architectures.py', 'data.py', 'phase_representation.py',
    'patient_training.py', 'train_patient_features.py', 'metrics.py',
    'learning_rate_schedule.py', 'experiment_config.py', 'train.py',
    'new_approach_paired.py',
    'new_approach_results.py', 'evaluation.py',
    'nonlinear_activation.py', 'triplet_standardization.py', 'spatial_pooling.py',
    'model.py', 'factorized_conv.py', 'parallel_context.py', 'attention_pooling.py',
    'power_mean_pool.py', 'augmentation.py', 'horizontal_flip.py', 'label_smoothing.py',
    'mixup.py', 'sharpness_step.py', 'weight_average.py',
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    tmp.replace(path)


def load_config(root=ROOT):
    path = Path(root) / CONFIG
    cfg = json.loads(path.read_text())
    required = dict(schema='new_approach_n01_v1', seed=42, inner_fraction=.2,
                    folds=[0, 1, 2, 3, 4], losses=['normal', 'weighted'],
                    epochs=50, patience=10, learning_rate=.001, weight_decay=.001,
                    patients_per_batch=2, eval_batch_size=16, num_workers=0,
                    scheduler_factor=.5, scheduler_patience=3,
                    reference='configs/patient_features/E57_patient_feature_mean_mac.json')
    if cfg != required:
        raise ValueError('N01 exige la receta fijada; un cambio requiere otro protocolo/identidad')
    return cfg


def validate_rows(rows):
    required = {'patient_id', 'sample_id', 'pCR', 'fold', 'split'}
    if rows.empty or not required.issubset(rows.columns):
        raise ValueError('Metadatos incompletos')
    if rows[list(required)].isna().any().any():
        raise ValueError('Metadatos ausentes')
    if not rows.patient_id.map(lambda x: isinstance(x, str) and bool(x.strip())).all():
        raise ValueError('Identificador de paciente inválido')
    if not rows.split.eq('train').all():
        raise ValueError('N01 solo admite train público')
    if not rows.pCR.isin([0, 1]).all() or not rows.fold.isin(range(5)).all():
        raise ValueError('Etiqueta o fold inválido')
    if rows.sample_id.duplicated().any():
        raise ValueError('Cortes duplicados')
    if rows.groupby('patient_id')[['pCR', 'fold']].nunique().ne(1).any().any():
        raise ValueError('Etiqueta/fold incoherente por paciente')


def split_rows(rows, fold):
    validate_rows(rows)
    if type(fold) is not int or fold not in range(5):
        raise ValueError('Fold inválido')
    outer = rows.loc[rows.fold.eq(fold)].copy()
    remaining = rows.loc[~rows.fold.eq(fold)].copy()
    patients = remaining[['patient_id', 'pCR']].drop_duplicates('patient_id').sort_values('patient_id')
    fit_ids, inner_ids = train_test_split(patients.patient_id.to_numpy(), test_size=.2,
                                         stratify=patients.pCR.to_numpy(), random_state=42)
    frames = dict(fit=remaining.loc[remaining.patient_id.isin(fit_ids)],
                  inner=remaining.loc[remaining.patient_id.isin(inner_ids)], outer=outer)
    sets = [set(f.patient_id) for f in frames.values()]
    if any(sets[a] & sets[b] for a, b in ((0, 1), (0, 2), (1, 2))):
        raise ValueError('Solape de pacientes')
    if set.union(*sets) != set(rows.patient_id):
        raise ValueError('Las particiones no reconstruyen desarrollo')
    if any(f.pCR.nunique() != 2 for f in frames.values()):
        raise ValueError('Se necesitan las dos clases en cada partición')
    return {name: frame.reset_index(drop=True) for name, frame in frames.items()}


def split_manifest(frames):
    result = {}
    for name, frame in frames.items():
        patients = frame[['patient_id', 'pCR']].drop_duplicates('patient_id').sort_values('patient_id')
        n0, n1 = [int(patients.pCR.eq(y).sum()) for y in (0, 1)]
        result[name] = dict(patient_ids=patients.patient_id.tolist(), patients=len(patients),
                            negative=n0, positive=n1, cuts=len(frame))
    return result


def current_manifest(root=ROOT):
    root = Path(root)
    cfg = load_config(root)
    reference_path = root / cfg['reference']
    reference = json.loads(reference_path.read_text())
    if reference.get('schema') != 'patient_features_v1' or reference.get('pool_stage') != 'features':
        raise ValueError('Referencia E57 incorrecta')
    # Validación existente de la receta original, sin instanciar datos ni pesos.
    from src.training.train_patient_features import load_spec
    _, typed = load_spec(reference_path)
    from src.models.architectures import BaseCNNConfig
    expected_model = BaseCNNConfig(pool_position='between_convolutions', input_representation='pre_differences')
    if (typed.training.epochs != 50 or typed.training.early_stopping_patience != 10
            or typed.training.learning_rate != .001 or typed.training.weight_decay != .001
            or typed.training.scheduler_factor != .5 or typed.training.scheduler_patience != 3
            or typed.training.lr_schedule.policy != 'plateau'
            or typed.model != expected_model):
        raise ValueError('La receta original ya no coincide con la fijada para N01')
    metadata = root / 'breastdcedl/metadata/samples.csv'
    samples = pd.read_csv(metadata)
    train_ids = set(samples.loc[samples.split.eq('train'), 'patient_id'])
    test_ids = set(samples.loc[samples.split.eq('test'), 'patient_id'])
    if train_ids & test_ids:
        raise ValueError('Paciente compartida entre train público y test')
    rows = samples.loc[samples.split.eq('train')].copy().reset_index(drop=True)
    validate_rows(rows)
    if rows.patient_id.nunique() != 1097:
        raise ValueError('Se esperaban las 1.097 pacientes de desarrollo')
    manifest = dict(schema='new_approach_n01_frozen_v1', config=cfg,
                    config_sha256=sha(root / CONFIG), reference_sha256=sha(reference_path),
                    protocol_document_sha256=sha(root/'research/PROTOCOLO_N01.md'),
                    metadata_sha256=sha(metadata),
                    sources_sha256={n: sha(root / 'src' / source_path(n).parent.name / n) for n in SOURCES},
                    splits={str(f): split_manifest(split_rows(rows, f)) for f in cfg['folds']},
                    test_accessed=False)
    return manifest, rows


def comparable(manifest):
    """Partes del manifiesto que deben coincidir con el congelado: configuración, documento,
    metadatos y particiones. Las huellas de las fuentes se registran, pero no se comparan: el
    29 de octubre de 2026 el código se reorganizó en subpaquetes (solo imports y rutas), lo que
    cambió sus SHA sin cambiar la lógica; el manifiesto congelado no se reescribe."""
    return {k: v for k, v in manifest.items() if k != 'sources_sha256'}


def prepare(root=ROOT):
    root = Path(root)
    manifest, _ = current_manifest(root)
    dest = root / TECH / 'protocol'
    path = dest / 'manifest.json'
    if path.exists():
        if comparable(json.loads(path.read_text())) != comparable(manifest):
            raise ValueError('Protocolo existente incompatible: no sobrescribir el protocolo congelado')
        return manifest
    dest.mkdir(parents=True, exist_ok=False)
    write_json(path, manifest)
    return manifest


def verify(root=ROOT):
    root = Path(root)
    path = root / TECH / 'protocol/manifest.json'
    if not path.is_file():
        raise FileNotFoundError('Ejecuta primero prepare; no se congela implícitamente al entrenar')
    actual, rows = current_manifest(root)
    saved = json.loads(path.read_text())
    if comparable(actual) != comparable(saved):
        raise ValueError('Configuración, documento, datos o particiones cambiaron tras congelar N01')
    return saved, rows


def metrics(frame, threshold):
    m = binary_metrics(frame.target, frame.probability, threshold)
    c = m['confusion_matrix']
    den = 2*c['tn'] + c['fp'] + c['fn']
    m['f1_negative'] = 2*c['tn']/den if den else 0.
    m['f1_macro'] = (m['f1'] + m['f1_negative'])/2
    m['precision_defined'] = c['tp'] + c['fp'] > 0
    return m


def select_threshold(frame):
    if frame.patient_id.duplicated().any():
        raise ValueError('La selección de umbral requiere una fila por paciente')
    y = frame.target.to_numpy()
    p = frame.probability.to_numpy(float)
    if set(y) != {0, 1} or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError('Etiquetas/probabilidades internas inválidas')
    u = np.unique(p)
    candidates = np.unique(np.r_[0., .5, 1., (u[:-1]+u[1:])/2])
    # Conteos en numpy: evita recalcular ROC/AP cientos de veces al elegir umbral.
    def key(t):
        pred = p >= t
        tp, tn = int(((y == 1) & pred).sum()), int(((y == 0) & ~pred).sum())
        fp, fn = int(((y == 0) & pred).sum()), int(((y == 1) & ~pred).sum())
        d1, d0 = 2*tp+fp+fn, 2*tn+fp+fn
        macro = ((2*tp/d1 if d1 else 0.) + (2*tn/d0 if d0 else 0.))/2
        ba = (tp/(tp+fn) + tn/(tn+fp))/2
        return macro, ba, -abs(float(t)-.5), -float(t)
    threshold = float(max(candidates, key=key))
    return threshold, metrics(frame, threshold)
