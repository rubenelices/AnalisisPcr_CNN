"""Rechazar errores de configuración antes de cargar datos o reservar GPU."""
import pytest

from src.experiment_config import DataExperimentConfig, TrainingConfig


@pytest.mark.parametrize("field,value", [
    ("root", ""), ("validation_fold", -1), ("validation_fold", 5),
    ("validation_fold", 0.5), ("batch_size", 0), ("batch_size", True),
    ("num_workers", -1), ("train_patients_per_class", 0),
    ("validation_patients_per_class", 1.5),
])
def test_invalid_data_settings_fail_early(field, value):
    with pytest.raises(ValueError, match=field):
        DataExperimentConfig(**{field: value})


@pytest.mark.parametrize("field,value", [
    ("epochs", 0), ("epochs", 2.5), ("early_stopping_patience", 0),
    ("scheduler_patience", -1), ("seed", -1),
    ("learning_rate", 0), ("learning_rate", float("nan")),
    ("learning_rate", float("inf")), ("learning_rate", "0.001"),
    ("weight_decay", -1), ("weight_decay", float("inf")),
    ("threshold", -0.1), ("threshold", 1.1), ("threshold", float("nan")),
    ("scheduler_factor", 0), ("scheduler_factor", 1),
])
def test_invalid_training_settings_fail_early(field, value):
    with pytest.raises(ValueError, match=field):
        TrainingConfig(**{field: value})


def test_valid_boundaries_and_long_run_are_allowed():
    data = DataExperimentConfig(validation_fold=4, batch_size=1, num_workers=0)
    assert data.batch_size == 1
    training = TrainingConfig(epochs=50, early_stopping_patience=51,
                              scheduler_patience=0, seed=0, weight_decay=0, threshold=1)
    assert training.early_stopping_patience > training.epochs
