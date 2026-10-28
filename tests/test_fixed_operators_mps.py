"""E53/E54 synthetic MPS edge cases; no data, checkpoints or science runs."""
from copy import deepcopy

import pytest
import torch

from src.architectures import BreastPCRNet
from src.experiment_config import load_experiment_config
from src.power_mean_pool import FixedPowerMeanPool
from src.triplet_standardization import TripletStandardization


@pytest.fixture
def local_mps():
    if not torch.backends.mps.is_available():
        pytest.skip('GPU Apple local no disponible en este entorno')
    cpu_state, mps_state = torch.get_rng_state(), torch.mps.get_rng_state()
    try:
        yield torch.device('mps')
    finally:
        torch.set_rng_state(cpu_state)
        torch.mps.set_rng_state(mps_state)


@pytest.mark.parametrize('experiment', ['E53_shared_standardization_mac', 'E54_fixed_gem_mac'])
@pytest.mark.parametrize('pattern', ['zero', 'constant', 'phases', 'random', 'small_signal'])
def test_full_model_eval_cpu_mps_consistency(local_mps, experiment, pattern):
    cfg = load_experiment_config(f'configs/experiments/{experiment}.json')
    torch.manual_seed(73)
    cpu = BreastPCRNet(cfg.model).eval()
    gpu = deepcopy(cpu).to(local_mps)
    x = torch.rand(1, 3, 256, 256)
    if pattern == 'zero':
        x.zero_()
    elif pattern == 'constant':
        x.fill_(0.8)
    elif pattern == 'phases':
        x[:, 0].fill_(0.1)
        x[:, 1].fill_(0.4)
        x[:, 2].fill_(0.3)
    elif pattern == 'small_signal':
        x.mul_(1e-7)
    with torch.inference_mode():
        expected = cpu(x)
        actual = gpu(x.to(local_mps)).cpu()
    assert expected.shape == actual.shape == (1,)
    assert torch.isfinite(actual).all()
    # Cross-device arithmetic is not bitwise identical; this is a technical
    # consistency bound, not accuracy or a scientific equivalence claim.
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-3)


@pytest.mark.parametrize('kind', ['standardization', 'power_mean'])
@pytest.mark.parametrize('pattern', ['zero', 'constant', 'random'])
def test_fixed_operator_mps_output_and_backward_finite(local_mps, kind, pattern):
    torch.manual_seed(73)
    x = torch.rand(2, 3, 17, 17, device=local_mps)
    if pattern == 'zero':
        x.zero_()
    elif pattern == 'constant':
        x.fill_(0.8)
    x.requires_grad_()
    operator = (TripletStandardization('shared_mean_std')
                if kind == 'standardization' else FixedPowerMeanPool())
    result = operator(x)
    assert torch.isfinite(result).all()
    result.square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
