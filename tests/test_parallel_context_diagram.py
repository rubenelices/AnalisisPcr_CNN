"""La figura de ramas usa módulos e inputs reales, no lee dataset."""
from torch import nn
import pytest
from src.models.parallel_context import ParallelContextConv
from src.reporting.parallel_context_diagram import draw_parallel_context_branches


def test_synthetic_branches_drawing_and_original_mode_preserved(tmp_path):
    model=nn.Sequential(ParallelContextConv(3,16),nn.BatchNorm2d(16),nn.ReLU(),nn.MaxPool2d(2),
        ParallelContextConv(16,32),nn.BatchNorm2d(32),nn.ReLU()).train()
    model[1].eval()  # preservar también modos mixtos, no solo el padre
    before=model[1].num_batches_tracked.item()
    draw_parallel_context_branches(model,tmp_path)
    assert model.training and not model[1].training and model[1].num_batches_tracked.item()==before
    assert (tmp_path/'ramas.png').is_file() and (tmp_path/'ramas.svg').is_file()
    assert 'padding2' in (tmp_path/'ramas.svg').read_text()

def test_no_parallel_unit_rejected(tmp_path):
    with pytest.raises(ValueError,match='unidades paralelas'):
        draw_parallel_context_branches(nn.Identity(),tmp_path)
    assert not (tmp_path/'ramas.png').exists()
