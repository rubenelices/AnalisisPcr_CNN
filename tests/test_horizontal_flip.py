"""Operación independiente: no nuevos pacientes ni calidad de clasificación."""
import pytest
import torch

from src.horizontal_flip import HorizontalFlipConfig,SharedRandomHorizontalFlip,apply_shared_horizontal_flip
from src.phase_representation import PhaseRepresentation


def test_exact_spatial_mirror_shared_channels_and_involution_without_mutation():
    plane=torch.arange(256*256,dtype=torch.float32).reshape(256,256)
    image=torch.stack((plane,plane+100000,-plane))
    before=image.clone()
    flipped=apply_shared_horizontal_flip(image)
    assert flipped.shape==image.shape and flipped.dtype==image.dtype
    assert flipped.is_contiguous() and flipped.data_ptr()!=image.data_ptr()
    assert torch.equal(image,before)
    assert all(torch.equal(flipped[channel],image[channel][:,torch.arange(255,-1,-1)]) for channel in range(3))
    assert torch.equal(apply_shared_horizontal_flip(flipped),image)
    assert torch.equal(torch.sort(flipped.flatten()).values,torch.sort(image.flatten()).values)


def test_signed_phase_differences_commute_with_shared_mirror():
    image=torch.rand(3,256,256)
    representation=PhaseRepresentation('pre_differences')
    left=representation(apply_shared_horizontal_flip(image).unsqueeze(0)).squeeze(0)
    right=apply_shared_horizontal_flip(representation(image.unsqueeze(0)).squeeze(0))
    assert torch.equal(left,right)


@pytest.mark.parametrize('config',[HorizontalFlipConfig(),HorizontalFlipConfig(True,0)])
def test_disabled_and_zero_probability_preserve_identity_and_rng(config):
    image=torch.rand(3,256,256)
    rng=torch.get_rng_state()
    assert SharedRandomHorizontalFlip(config)(image) is image
    assert torch.equal(rng,torch.get_rng_state())


def test_probability_one_mirrors_without_random_draw():
    image=torch.rand(3,256,256)
    rng=torch.get_rng_state()
    assert torch.equal(SharedRandomHorizontalFlip(HorizontalFlipConfig(True,1))(image),image.flip(-1))
    assert torch.equal(rng,torch.get_rng_state())


def test_one_shared_coin_and_seeded_sequence():
    operation=SharedRandomHorizontalFlip(HorizontalFlipConfig(True,0.5))
    torch.manual_seed(17)
    first=[operation.sample_flip() for _ in range(20)]
    torch.manual_seed(17)
    second=[operation.sample_flip() for _ in range(20)]
    assert first==second and any(first) and not all(first)
    generator=torch.Generator().manual_seed(42)
    rng=torch.get_rng_state()
    operation.sample_flip(generator=generator)
    assert torch.equal(rng,torch.get_rng_state())


@pytest.mark.parametrize('probability',[-0.1,1.1,True,float('nan'),float('inf'),None,'0.5'])
def test_bad_probability_rejected(probability):
    with pytest.raises(ValueError):
        HorizontalFlipConfig(probability=probability)


def test_bad_enable_rejected():
    with pytest.raises(ValueError):
        HorizontalFlipConfig(enabled=1)


@pytest.mark.parametrize('image',[torch.zeros(3,128,128),torch.zeros(3,256,256,dtype=torch.float64),torch.full((3,256,256),float('nan'))])
def test_bad_tensor_rejected(image):
    with pytest.raises(ValueError):
        apply_shared_horizontal_flip(image)


def test_finite_gradient_and_source_values_unchanged():
    image=torch.rand(3,256,256,requires_grad=True)
    result=apply_shared_horizontal_flip(image)
    result.square().sum().backward()
    assert torch.equal(image.grad,2*image.detach())
