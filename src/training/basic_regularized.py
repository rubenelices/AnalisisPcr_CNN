"""Contrato explícito de B09: propuesta conjunta, no cambia defaults históricos."""
from src.models.architectures import BaseCNNConfig
from src.pipeline.augmentation import AugmentationConfig


def validate_basic_regularized(spec, cfg):
    expected = BaseCNNConfig(channels=(8,16,32,64), group_norm_groups=4,
                            pool_position='between_convolutions', input_representation='pre_differences')
    if (cfg.model != expected or spec.get('clinical_input_columns') != ['age','menopause']
            or spec.get('active_clinical_columns') != ['age','menopause']
            or spec.get('fusion_hidden',0) != 0 or spec.get('image_pooling','average') != 'average'
            or spec.get('feature_normalization','none') != 'none'
            or spec.get('image_lr_factor',1) != 1 or spec.get('image_weight_decay_factor',1) != 1
            or spec.get('image_occlusion') is not None or spec.get('image_intensity_field') is not None
            or cfg.data.augmentation != AugmentationConfig(True,5.,.02)
            or cfg.data.intensity_augmentation.enabled or cfg.data.horizontal_flip.enabled
            or cfg.training.loss != 'weighted' or spec.get('positive_weight_exponent') != .5
            or cfg.training.learning_rate != .0003 or cfg.training.weight_decay != .001):
        raise ValueError('B09 exige CNN compacta GN4, clínica básica, geometría5°/2%, BCEsqrt y LR0,0003')
