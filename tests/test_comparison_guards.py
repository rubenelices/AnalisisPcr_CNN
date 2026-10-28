"""Control de procedencia, bootstrap y gráficos con artefactos sintéticos."""
import json

import numpy as np
import pandas as pd
import pytest

from src.compare_experiments import compare_experiments, save_comparison
from src.experiment_config import load_experiment_config
from src.metrics import binary_metrics


def pair(tmp_path, scientific=True):
    paths = []
    for name, config_file, offset in [("left", "E13_50epochs.json", 0),
                                       ("right", "E14_spatial_dropout_010.json", .05)]:
        path = tmp_path / name
        path.mkdir()
        config = load_experiment_config(f"configs/experiments/{config_file}")
        targets = np.array([0, 0, 0, 0, 0, 1, 1, 1])
        scores = np.array([.1, .4, .6, .2, .55, .3, .7, .65]) + offset
        frame = pd.DataFrame({"patient_id": [f"synthetic{i}" for i in range(8)],
            "split": "validation", "fold": 0, "target": targets,
            "probability": scores, "prediction": (scores >= .5).astype(int), "cuts": 2})
        frame.to_csv(path / "predictions_patient.csv", index=False)
        metrics = binary_metrics(targets, scores)
        evaluation = {"scope": "internal_validation_only", "statistical_unit": "patient",
            "experiment_id": config.experiment_id, "aggregation": "mean", "threshold": .5,
            "patient_metrics_primary": metrics}
        (path / "evaluation_summary.json").write_text(json.dumps(evaluation))
        if scientific:
            (path / "config_resolved.json").write_text(json.dumps(config.to_dict()))
            summary = {"experiment_id": config.experiment_id, "status": "complete",
                "scientific_result": True, "test_evaluated": False,
                "best_epoch": 1, "epochs_completed": 2, "patients": {"validation": 8},
                "environment": {"device": "cuda", "torch": "synthetic", "python": "synthetic"},
                "validation_metrics": {"aggregation": "mean", "patient": metrics}}
            (path / "summary.json").write_text(json.dumps(summary))
        paths.append(path)
    return paths


def edit_json(path, change):
    value = json.loads(path.read_text())
    change(value)
    path.write_text(json.dumps(value))


def test_scientific_pair_tracks_configuration_hashes_and_controlled_change(tmp_path):
    result = compare_experiments(*pair(tmp_path), repetitions=20, stratified=True)
    assert result["comparability"]["controlled_single_change"] is True
    assert list(result["comparability"]["configuration_differences"]) == ["model.spatial_dropout"]
    assert result["test_evaluated"] is False
    assert result["bootstrap_stratified_by_target"] is True
    assert result["metrics"]["roc_auc"]["valid_repetitions"] == 20
    assert len(result["sources"]["left"]["source_sha256"]) == 4
    assert "probabilidad de superioridad" in result["selection_warning"]


def test_artifact_only_is_opt_in_and_never_called_controlled_scientific_comparison(tmp_path):
    paths = pair(tmp_path, scientific=False)
    with pytest.raises(ValueError, match="Faltan artefactos científicos"):
        compare_experiments(*paths, repetitions=2)
    result = compare_experiments(*paths, repetitions=2, require_scientific_runs=False)
    assert result["comparability"]["warnings"]
    assert result["comparability"]["controlled_single_change"] is False


def test_batch_can_be_the_single_controlled_variable(tmp_path):
    paths = pair(tmp_path)
    edit_json(paths[1] / "config_resolved.json", lambda data: data["model"].update(spatial_dropout=0))
    edit_json(paths[1] / "config_resolved.json", lambda data: data["data"].update(batch_size=128))
    result = compare_experiments(*paths, repetitions=2)
    assert result["comparability"]["controlled_single_change"] is True
    assert list(result["comparability"]["configuration_differences"]) == ["data.batch_size"]


def test_batch_plus_architecture_is_still_confounded(tmp_path):
    paths = pair(tmp_path)
    edit_json(paths[1] / "config_resolved.json", lambda data: data["data"].update(batch_size=128))
    result = compare_experiments(*paths, repetitions=2)
    assert result["comparability"]["controlled_single_change"] is False
    assert any("data.batch_size" in warning for warning in result["comparability"]["warnings"])


@pytest.mark.parametrize("field,value", [("scope", "test"), ("statistical_unit", "slice"),
                                         ("test_evaluated", True)])
def test_non_validation_or_non_patient_scope_is_rejected(tmp_path, field, value):
    paths = pair(tmp_path)
    edit_json(paths[1] / "evaluation_summary.json", lambda data: data.update({field: value}))
    with pytest.raises(ValueError, match="validacion interna"):
        compare_experiments(*paths, repetitions=2)


@pytest.mark.parametrize("field,value", [("target", .7), ("probability", np.nan),
                                         ("fold", .5), ("prediction", 5)])
def test_invalid_patient_csv_is_rejected(tmp_path, field, value):
    paths = pair(tmp_path)
    csv = paths[1] / "predictions_patient.csv"
    frame = pd.read_csv(csv)
    frame[field] = frame[field].astype(float)
    frame.loc[0, field] = value
    frame.to_csv(csv, index=False)
    with pytest.raises(ValueError):
        compare_experiments(*paths, repetitions=2)


def test_reordered_rows_align_by_patient_not_file_position(tmp_path):
    paths = pair(tmp_path)
    original = compare_experiments(*paths, repetitions=20, seed=7, stratified=True)
    csv = paths[1] / "predictions_patient.csv"
    pd.read_csv(csv).iloc[::-1].to_csv(csv, index=False)
    reordered = compare_experiments(*paths, repetitions=20, seed=7, stratified=True)
    assert original["metrics"] == reordered["metrics"]


def test_distinct_patient_set_is_rejected_even_if_metrics_and_counts_match(tmp_path):
    paths = pair(tmp_path)
    csv = paths[1] / "predictions_patient.csv"
    frame = pd.read_csv(csv)
    frame.loc[0, "patient_id"] = "different_synthetic_patient"
    frame.to_csv(csv, index=False)
    with pytest.raises(ValueError, match="mismas pacientes"):
        compare_experiments(*paths, repetitions=2)


def test_mixed_devices_or_budgets_are_marked_as_confounded(tmp_path):
    paths = pair(tmp_path)
    edit_json(paths[1] / "summary.json", lambda data: data["environment"].update(device="mps"))
    edit_json(paths[1] / "config_resolved.json", lambda data: data["training"].update(epochs=10))
    result = compare_experiments(*paths, repetitions=2)
    assert result["comparability"]["controlled_single_change"] is False
    assert any("device distinto" in warning for warning in result["comparability"]["warnings"])
    assert any("training.epochs" in warning for warning in result["comparability"]["warnings"])


def test_output_is_not_overwritten_and_interval_need_not_enclose_point(tmp_path):
    result = compare_experiments(*pair(tmp_path), repetitions=10, stratified=True)
    for values in result["metrics"].values():
        values["difference_right_minus_left"] = .2
        values["difference_ci_lower"] = -.1
        values["difference_ci_upper"] = .1
    output = tmp_path / "comparison"
    save_comparison(result, output)
    assert (output / "comparison.png").is_file()
    with pytest.raises(FileExistsError):
        save_comparison(result, output)


def test_undefined_bootstrap_interval_is_saved_as_null_and_plot_still_works(tmp_path):
    result = compare_experiments(*pair(tmp_path), repetitions=2, stratified=True)
    result["metrics"]["roc_auc"].update(difference_ci_lower=None, difference_ci_upper=None,
                                        probability_difference_positive=None, valid_repetitions=0)
    output = tmp_path / "undefined_interval"
    save_comparison(result, output)
    saved = json.loads((output / "comparison.json").read_text())
    assert saved["metrics"]["roc_auc"]["difference_ci_lower"] is None
    assert (output / "comparison.png").exists()


@pytest.mark.parametrize("confidence", [0, 1, np.nan])
def test_invalid_confidence_levels_fail_before_reading_inputs(tmp_path, confidence):
    with pytest.raises(ValueError, match="confidence_level"):
        compare_experiments(tmp_path / "missing", tmp_path / "missing", confidence_level=confidence)
