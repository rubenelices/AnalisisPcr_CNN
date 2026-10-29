"""Las entradas inválidas no pueden convertirse silenciosamente en métricas."""
import numpy as np
import pytest

from src.evaluation.metrics import aggregate_by_patient, binary_metrics


@pytest.mark.parametrize("label", [0.7, 1.9, -0.1, np.nan, np.inf])
@pytest.mark.parametrize("aggregation", [False, True])
def test_fractional_and_non_finite_labels_are_rejected(label, aggregation):
    with pytest.raises(ValueError, match="solo 0 y 1"):
        if aggregation:
            aggregate_by_patient(["A", "B"], [label, 1], [0.1, 0.9])
        else:
            binary_metrics([label, 1], [0.1, 0.9])


@pytest.mark.parametrize("scores", [[0.1, np.nan], [0.1, np.inf], [-0.1, 0.9], [0.1, 1.1]])
def test_aggregation_cannot_hide_invalid_probabilities(scores):
    with pytest.raises(ValueError, match="finitos"):
        aggregate_by_patient(["A", "A"], [0, 0], scores)


@pytest.mark.parametrize("identifiers", [["A", None], ["A", "  "], ["A"]])
def test_missing_or_empty_patient_ids_are_rejected(identifiers):
    with pytest.raises(ValueError, match="identificadores"):
        aggregate_by_patient(identifiers, [0, 1], [0.1, 0.9])


def test_valid_integer_and_float_binary_labels_remain_identical():
    assert binary_metrics([0, 1], [.1, .9]) == binary_metrics([0., 1.], [.1, .9])
    frame = aggregate_by_patient(["A", "A", "B"], [0., 0., 1.], [.1, .3, .9])
    assert frame.target.tolist() == [0, 1]
    assert frame.probability.tolist() == pytest.approx([.2, .9])
