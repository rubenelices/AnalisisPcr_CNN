import pandas as pd
import pytest
import torch
from torch import nn

from src import evaluate_e57_final as final
from src.metrics import aggregate_by_patient


class Loader:
    dataset = [0, 1, 2]
    def __init__(self, split='test'):
        self.split = split
    def __iter__(self):
        yield dict(split=[self.split] * 3, fold=torch.full((3,), -1),
                   image=torch.tensor([[-2.], [0.], [2.]]),
                   target=torch.tensor([0., 0., 1.]), patient_id=['a', 'a', 'b'],
                   sample_id=['a1', 'a2', 'b1'], slice_index=torch.tensor([1, 2, 1]))


class Logits(nn.Module):
    def forward(self, image):
        return image[:, 0]


def test_collector_uses_sigmoid_then_patient_mean():
    cuts = final.collect(Logits(), Loader(), torch.device('cpu'))
    patients = aggregate_by_patient(cuts.patient_id, cuts.target, cuts.probability)
    assert len(cuts) == 3 and len(patients) == 2
    assert patients.probability.tolist() == pytest.approx([
        (torch.sigmoid(torch.tensor(-2.)).item() + .5) / 2,
        torch.sigmoid(torch.tensor(2.)).item()])
    assert (patients.probability >= final.THRESHOLD).tolist() == [False, True]


def test_collector_rejects_training_before_inference():
    class Never(nn.Module):
        def forward(self, image):
            raise AssertionError('No debe ejecutar inferencia')
    with pytest.raises(ValueError, match='Solo se admite'):
        final.collect(Never(), Loader('train'), torch.device('cpu'))


def test_modified_checkpoint_is_rejected(tmp_path, monkeypatch):
    file = tmp_path / 'modified.pt'
    file.write_bytes(b'changed')
    monkeypatch.setattr(final, 'CHECKPOINT', file)
    with pytest.raises(ValueError, match='no coincide'):
        final.verify_checkpoint()


def test_existing_evaluation_cannot_be_repeated(tmp_path, monkeypatch):
    monkeypatch.setattr(final, 'OUT', tmp_path)
    monkeypatch.setattr(final, 'verify_checkpoint', lambda: None)
    def never(*args):
        raise AssertionError('No debe leer datos de test')
    monkeypatch.setattr(final, 'load_samples', never)
    with pytest.raises(FileExistsError):
        final.evaluate('cpu')
