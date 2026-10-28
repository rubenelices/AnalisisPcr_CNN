"""Comparacion pareada y reproducible de dos experimentos en validacion interna."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.experiment_config import experiment_config_from_dict
from src.metrics import binary_metrics, validate_binary_inputs


COMPARISON_METRICS = (
    "roc_auc",
    "pr_auc",
    "brier_score",
    "sensitivity",
    "specificity",
    "precision",
    "f1",
    "balanced_accuracy",
)


def _load_experiment(directory: Path, require_scientific: bool) -> tuple[pd.DataFrame, dict, dict]:
    predictions_path = directory / "predictions_patient.csv"
    summary_path = directory / "evaluation_summary.json"
    predictions = pd.read_csv(predictions_path, dtype={"patient_id": str}, float_precision="round_trip")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    required = {"patient_id", "split", "fold", "target", "probability"}
    if predictions.empty or not required.issubset(predictions.columns):
        raise ValueError(f"{directory}: faltan predicciones o columnas por paciente")
    if (summary.get("scope") != "internal_validation_only"
            or summary.get("statistical_unit") != "patient"
            or summary.get("test_evaluated", False) is not False):
        raise ValueError(f"{directory}: el informe no es de validacion interna por paciente")
    if set(predictions["split"].astype(str)) != {"validation"}:
        raise ValueError(f"{directory} no contiene solo validacion interna")
    if predictions["patient_id"].duplicated().any():
        raise ValueError(f"{directory} contiene pacientes duplicadas")
    if predictions.patient_id.isna().any() or predictions.patient_id.str.strip().eq("").any():
        raise ValueError("Identificadores de paciente vacíos")
    validate_binary_inputs(predictions.target, predictions.probability)
    folds = predictions.fold.to_numpy(dtype=float)
    if (not np.isfinite(folds).all() or not np.isin(folds, range(5)).all()
            or len(np.unique(folds)) != 1):
        raise ValueError("Se exige un único fold interno válido, entre 0 y 4")
    threshold = float(summary["threshold"])
    metrics = binary_metrics(predictions.target, predictions.probability, threshold)
    if "prediction" in predictions and not np.array_equal(
        predictions.prediction, (predictions.probability >= threshold).astype(int)
    ):
        raise ValueError("Las clases guardadas no corresponden al umbral del informe")
    primary = summary.get("patient_metrics_primary", {})
    if primary.get("n") != len(predictions):
        raise ValueError("El recuento del informe no coincide con el CSV por paciente")
    for name in COMPARISON_METRICS:
        if name in primary and not np.isclose(primary[name], metrics[name], atol=1e-8,
                                               rtol=1e-6, equal_nan=True):
            raise ValueError(f"Métrica {name} incoherente con las predicciones")
    paths = [predictions_path, summary_path]
    run_path, config_path = directory / "summary.json", directory / "config_resolved.json"
    provenance = {"scientific_run_verified": False, "config": None, "environment": {}}
    if run_path.exists() != config_path.exists():
        raise ValueError("Ejecución incompleta: se necesitan summary y config_resolved juntos")
    if run_path.exists():
        run = json.loads(run_path.read_text())
        raw_config=json.loads(config_path.read_text())
        modality=None
        if 'schema' in raw_config:
            from src.train_multimodal import load_spec
            from src.clinical_features import ClinicalPreprocessor,CLINICAL_COLUMNS
            spec,config=load_spec(config_path)
            clinical_path=directory/'clinical_preprocessor.json'
            prep=ClinicalPreprocessor.from_dict(json.loads(clinical_path.read_text()))
            if (prep.train_patient_count!=run.get('patients',{}).get('train')
                    or prep.train_patient_count!=run.get('clinical_fit_patient_count')
                    or list(prep.active_columns)!=spec.get('active_clinical_columns',list(CLINICAL_COLUMNS))):
                raise ValueError('Preprocesador clínico no corresponde al train interno')
            modality={name:spec.get(name,default) for name,default in {
                'schema':'multimodal_e19_v1','clinical_columns':list(CLINICAL_COLUMNS),
                'active_clinical_columns':list(CLINICAL_COLUMNS),'fusion_hidden':0,
                'image_pooling':'average','feature_normalization':'none',
                'image_lr_factor':1.,'image_weight_decay_factor':1.,
                'positive_weight_exponent':1.,'image_occlusion':None,'image_intensity_field':None}.items()}
            paths.append(clinical_path)
            if summary.get('experiment_id') is None:
                # Runner multimodal histórico no escribía el ID en evaluation_summary.
                # Derivarlo SOLO tras validar configuración propia y estadísticas train;
                # no modificar artefactos históricos ni aceptar IDs contradictorios.
                summary=dict(summary,experiment_id=config.experiment_id)
                provenance['evaluation_identity_source']='training_summary_and_resolved_config'
        else:
            config=experiment_config_from_dict(raw_config)
        if (run.get("status") != "complete" or run.get("scientific_result") is not True
                or run.get("test_evaluated") is not False
                or run.get("debug_patient_subset", False) is not False
                or run.get("experiment_id") != config.experiment_id
                or summary.get("experiment_id") != config.experiment_id
                or config.data.train_patients_per_class is not None
                or config.data.validation_patients_per_class is not None
                or config.data.validation_fold != int(folds[0])
                or config.training.threshold != threshold
                or run.get("patients", {}).get("validation") != len(predictions)):
            raise ValueError("Se requiere una ejecución científica completa y coherente, sin test/debug")
        if (type(run.get("best_epoch")) is not int or type(run.get("epochs_completed")) is not int
                or not 1 <= run["best_epoch"] <= run["epochs_completed"] <= config.training.epochs):
            raise ValueError("Época seleccionada o presupuesto de entrenamiento inválidos")
        if summary.get("aggregation") != run["validation_metrics"].get("aggregation"):
            raise ValueError("La agregación del resumen no coincide con la evaluación")
        for name in ("roc_auc", "pr_auc"):
            if not np.isclose(run["validation_metrics"]["patient"][name], metrics[name],
                              atol=1e-8, rtol=1e-6, equal_nan=True):
                raise ValueError("El resumen de entrenamiento no corresponde a estas predicciones")
        provenance.update(scientific_run_verified=True, config=config.to_dict(),
                          environment=run.get("environment", {}), best_epoch=run["best_epoch"],
                          epochs_completed=run["epochs_completed"])
        if modality is not None:provenance['modality']=modality
        paths.extend([run_path, config_path])
    elif require_scientific:
        raise ValueError("Faltan artefactos científicos: summary.json y config_resolved.json")
    provenance["source_sha256"] = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                                   for path in paths}
    return predictions, summary, provenance


def _comparability(left: dict, right: dict) -> dict:
    warnings, differences = [], {}
    if not (left["scientific_run_verified"] and right["scientific_run_verified"]):
        warnings.append("Solo artefactos de evaluación; no se verifica entrenamiento científico completo")
    else:
        if left.get('modality')!=right.get('modality'):
            if left.get('modality') is None or right.get('modality') is None:
                warnings.append('Modalidades distintas: no atribuir delta a una sola técnica')
            else:
                for key in sorted(set(left['modality'])|set(right['modality'])):
                    a,b=left['modality'].get(key),right['modality'].get(key)
                    if a!=b:differences['modality.'+key]={'left':a,'right':b}
        for section in ("model", "training", "data"):
            for key, value in left["config"][section].items():
                other = right["config"][section].get(key)
                if value != other:
                    differences[f"{section}.{key}"] = {"left": value, "right": other}
        for key in ("device", "torch", "python"):
            first, second = left["environment"].get(key), right["environment"].get(key)
            if first is None or second is None:
                warnings.append(f"Entorno {key} no documentado en ambas ejecuciones")
            elif first != second:
                warnings.append(f"Entorno {key} distinto: {first} frente a {second}")
        fixed = {"training.epochs", "training.seed", "training.monitor", "training.mixed_precision",
                 "training.early_stopping_patience", "training.scheduler_patience",
                 "training.scheduler_factor", "data.batch_size", "data.root"}
        # El batch puede ser la variable principal de una ablación propia.
        # Sigue siendo confusor si cambia junto a arquitectura u otros ajustes.
        if set(differences) == {"data.batch_size"}:
            fixed.discard("data.batch_size")
        for field in sorted(fixed & differences.keys()):
            warnings.append(f"No se aisló el cambio principal: difiere {field}")
        if len(differences) > 1:
            warnings.append("Más de un ajuste varía; no atribuir el delta a una sola técnica")
    return {"configuration_differences": differences, "warnings": warnings,
            "controlled_single_change": len(differences) == 1 and not warnings,
            "decision": "review_only_not_automatic_selection"}


def _aligned_predictions(
    left: pd.DataFrame, right: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    left = left.sort_values("patient_id").reset_index(drop=True)
    right = right.sort_values("patient_id").reset_index(drop=True)
    if left["patient_id"].tolist() != right["patient_id"].tolist():
        raise ValueError("Los experimentos no evaluaron exactamente las mismas pacientes")
    if not np.array_equal(left["target"].to_numpy(), right["target"].to_numpy()):
        raise ValueError("Las etiquetas no coinciden entre experimentos")
    if not np.array_equal(left["fold"].to_numpy(), right["fold"].to_numpy()):
        raise ValueError("Los folds no coinciden entre experimentos")
    return left, right


def compare_experiments(
    left_dir: str | Path,
    right_dir: str | Path,
    *,
    repetitions: int = 2_000,
    confidence_level: float = 0.95,
    seed: int = 42,
    stratified: bool = False,
    require_scientific_runs: bool = True,
) -> dict[str, Any]:
    """Estima diferencias right-left remuestreando las mismas pacientes."""

    if type(repetitions) is not int or repetitions < 1:
        raise ValueError("repetitions debe ser positivo")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level debe estar en (0, 1)")
    left, left_summary, left_source = _load_experiment(Path(left_dir), require_scientific_runs)
    right, right_summary, right_source = _load_experiment(Path(right_dir), require_scientific_runs)
    left, right = _aligned_predictions(left, right)
    for field in ("aggregation", "threshold"):
        if left_summary[field] != right_summary[field]:
            raise ValueError(f"La comparacion exige el mismo {field}")

    threshold = float(left_summary["threshold"])
    targets = left["target"].to_numpy(dtype=int)
    if np.unique(targets).size != 2:
        raise ValueError("Se necesitan ambas clases para comparar ROC-AUC")
    left_probabilities = left["probability"].to_numpy(dtype=float)
    right_probabilities = right["probability"].to_numpy(dtype=float)
    left_metrics = binary_metrics(targets, left_probabilities, threshold)
    right_metrics = binary_metrics(targets, right_probabilities, threshold)
    rng = np.random.default_rng(seed)
    differences: dict[str, list[float]] = {metric: [] for metric in COMPARISON_METRICS}
    strata = [np.flatnonzero(targets == label) for label in (0, 1)]
    for _ in range(repetitions):
        indices = (np.concatenate([rng.choice(group, size=len(group), replace=True) for group in strata])
                   if stratified else rng.integers(0, len(targets), size=len(targets)))
        sampled_left = binary_metrics(targets[indices], left_probabilities[indices], threshold)
        sampled_right = binary_metrics(targets[indices], right_probabilities[indices], threshold)
        for metric in COMPARISON_METRICS:
            difference = float(sampled_right[metric]) - float(sampled_left[metric])
            if np.isfinite(difference):
                differences[metric].append(difference)

    alpha = 1 - confidence_level
    comparison: dict[str, dict[str, float | int]] = {}
    for metric, values in differences.items():
        array = np.asarray(values, dtype=float)
        lower = float(np.quantile(array, alpha / 2)) if len(array) else None
        upper = (
            float(np.quantile(array, 1 - alpha / 2)) if len(array) else None
        )
        comparison[metric] = {
            "left": float(left_metrics[metric]),
            "right": float(right_metrics[metric]),
            "difference_right_minus_left": float(right_metrics[metric] - left_metrics[metric]),
            "difference_ci_lower": lower,
            "difference_ci_upper": upper,
            "probability_difference_positive": (
                float((array > 0).mean()) if len(array) else None
            ),
            "valid_repetitions": int(len(array)),
        }
    return {
        "scope": "internal_validation_only",
        "statistical_unit": "patient",
        "comparison_design": "paired_bootstrap_by_patient",
        "left_experiment": left_summary.get("experiment_id", Path(left_dir).name),
        "right_experiment": right_summary.get("experiment_id", Path(right_dir).name),
        "patients": int(len(targets)),
        "folds": sorted(left["fold"].astype(int).unique().tolist()),
        "aggregation": left_summary["aggregation"],
        "threshold": threshold,
        "confidence_level": confidence_level,
        "repetitions": repetitions,
        "seed": seed,
        "test_evaluated": False,
        "bootstrap_stratified_by_target": stratified,
        "sources": {"left": left_source, "right": right_source},
        "comparability": _comparability(left_source, right_source),
        "selection_warning": (
            "Exploratorio: el intervalo no corrige selección de checkpoint ni múltiples ensayos. "
            "La fracción bootstrap positiva no es un p-valor ni probabilidad de superioridad. "
            "Estratificar condiciona el intervalo a la composición de clases observada."
        ),
        "notice": "Uso educativo y de investigación, sin validez clínica.",
        "metrics": comparison,
        "interpretation": (
            "Un intervalo de la diferencia que cruza 0 no aporta evidencia clara de "
            "ventaja. Para Brier, una diferencia negativa favorece al experimento derecho; "
            "para el resto, una diferencia positiva lo favorece."
        ),
    }


def save_comparison(result: dict[str, Any], output_dir: str | Path) -> None:
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"El informe ya existe; elige otra carpeta: {output}")
    output.mkdir(parents=True, exist_ok=False)
    (output / "comparison.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )
    rows = [{"metric": metric, **values} for metric, values in result["metrics"].items()]
    pd.DataFrame(rows).to_csv(output / "comparison.csv", index=False)
    save_comparison_figure(result, output / "comparison.png")


def save_comparison_figure(result: dict[str, Any], path: str | Path) -> None:
    """Regenera el gráfico desde agregados, sin bootstrap ni archivos de pacientes."""

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-caso-cancer")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frame = pd.DataFrame([{"metric": metric, **values} for metric, values in result["metrics"].items()])
    positions = np.arange(len(frame))
    values = frame["difference_right_minus_left"].to_numpy()
    figure, axis = plt.subplots(figsize=(9, 5.5))
    # Un intervalo percentil no tiene por qué contener el estimador observado.
    # Dibujar límites absolutos evita barras de error negativas o recortadas.
    lower = pd.to_numeric(frame["difference_ci_lower"], errors="coerce").to_numpy(dtype=float)
    upper = pd.to_numeric(frame["difference_ci_upper"], errors="coerce").to_numpy(dtype=float)
    defined = np.isfinite(lower) & np.isfinite(upper)
    axis.hlines(positions[defined], lower[defined], upper[defined],
                color="#126E82", linewidth=2)
    axis.scatter(values, positions, color="#126E82", zorder=3)
    axis.axvline(0, color="0.5", linestyle="--")
    labels = ["Brier (menor es mejor)" if metric == "brier_score" else metric for metric in frame.metric]
    axis.set_yticks(positions, labels=labels)
    axis.set_xlabel("Diferencia: experimento derecho - izquierdo")
    axis.set_title(f"{result['left_experiment']} → {result['right_experiment']}\n"
                   "Comparación pareada por paciente en validación interna", fontsize=11)
    figure.text(.5, .01, "IC exploratorios; no corrigen selección ni múltiples ensayos. Uso educativo, no clínico.",
                ha="center", fontsize=8)
    axis.grid(axis="x", alpha=0.2)
    figure.tight_layout(rect=(0, .03, 1, 1))
    figure.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stratified", action="store_true", help="Remuestrear pacientes dentro de cada clase")
    parser.add_argument("--allow-artifact-only", action="store_true",
                        help="Solo depuración: admitir evaluación sin ejecución científica verificable")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = compare_experiments(
        args.left, args.right, repetitions=args.repetitions, seed=args.seed,
        stratified=args.stratified, require_scientific_runs=not args.allow_artifact_only,
    )
    save_comparison(result, args.output_dir)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
