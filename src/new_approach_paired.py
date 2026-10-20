"""Incertidumbre condicional pareada de N01, sin reentrenamiento ni test."""
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


MEASURES = ('AUC', 'AP', 'Brier', 'F1_macro', 'BA')


def values(y, p, threshold):
    pred = p >= threshold
    tp, tn = np.sum((y == 1) & pred), np.sum((y == 0) & ~pred)
    fp, fn = np.sum((y == 0) & pred), np.sum((y == 1) & ~pred)
    f1 = 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.
    f0 = 2*tn/(2*tn+fp+fn) if 2*tn+fp+fn else 0.
    return np.array([roc_auc_score(y, p), average_precision_score(y, p),
                     np.mean((p-y)**2), (f1+f0)/2, (tp/(tp+fn)+tn/(tn+fp))/2])


def paired_intervals(pairs, repetitions=2000, seed=42):
    """pairs: cinco tuplas (normal_patient_frame, weighted_patient_frame, t0, t1).

    Reutiliza los mismos índices para ambas pérdidas y remuestrea pacientes por
    clase dentro de cada fold. Condicional a modelos/umbrales y prevalencias;
    no es un intervalo de toda la selección de arquitectura/entrenamiento.
    """
    if len(pairs) != 5 or type(repetitions) is not int or repetitions < 1:
        raise ValueError('Se necesitan cinco parejas y repeticiones positivas')
    data, seen = [], set()
    for a, b, ta, tb in pairs:
        if a.patient_id.duplicated().any() or b.patient_id.duplicated().any():
            raise ValueError('El bootstrap requiere una fila por paciente')
        a, b = a.sort_values('patient_id'), b.sort_values('patient_id')
        if list(a.patient_id) != list(b.patient_id) or not np.array_equal(a.target, b.target):
            raise ValueError('Las predicciones pareadas deben corresponder a las mismas pacientes/etiquetas')
        if seen & set(a.patient_id):
            raise ValueError('Una paciente exterior no puede pertenecer a dos folds')
        seen.update(a.patient_id)
        y = a.target.to_numpy()
        p, q = a.probability.to_numpy(float), b.probability.to_numpy(float)
        if (set(y) != {0, 1} or not np.isfinite(np.r_[p, q]).all()
                or ((np.r_[p, q] < 0) | (np.r_[p, q] > 1)).any()
                or not 0 <= ta <= 1 or not 0 <= tb <= 1):
            raise ValueError('Probabilidades, clases o umbrales inválidos')
        data.append((y, p, q, ta, tb))
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(repetitions):
        fold_diffs = []
        for y, p, q, ta, tb in data:
            # Condicionar por clase evita AUC indefinido, también en tests pequeños.
            indices = np.concatenate([rng.choice(np.flatnonzero(y == c), int((y == c).sum()), replace=True)
                                      for c in (0, 1)])
            fold_diffs.append(values(y[indices], q[indices], tb)-values(y[indices], p[indices], ta))
        draws.append(np.mean(fold_diffs, axis=0))
    point = np.mean([values(y, q, tb)-values(y, p, ta) for y, p, q, ta, tb in data], axis=0)
    limits = np.quantile(draws, [.025, .975], axis=0)
    return dict(method='paired_patient_stratified_within_fold_conditional_percentile',
                repetitions=repetitions, seed=seed, confidence_level=.95,
                effect='weighted_minus_normal_mean_of_five_folds',
                intervals={name: dict(difference=float(point[i]), lower=float(limits[0, i]), upper=float(limits[1, i]))
                           for i, name in enumerate(MEASURES)},
                limitation='Condicional a modelos, umbrales y prevalencias observados; no incluye selección ni reentrenamiento.')
