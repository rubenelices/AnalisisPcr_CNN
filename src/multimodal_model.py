"""CNN propia E19 + MLP clínica mínima, inicializadas desde cero.

Fusión aditiva al nivel de logit: no atención ni interacciones aprendidas entre
modalidades en esta primera prueba. No carga pesos E19 ni ningún modelo externo.
"""
import torch
from torch import nn

from src.architectures import BreastPCRNet, BaseCNNConfig
from src.clinical_features import ClinicalPreprocessor


class MultimodalPCRNet(nn.Module):
    def __init__(self, config: BaseCNNConfig, fusion_hidden: int = 0, image_pooling: str = 'average',
                 feature_normalization: str = 'none'):
        super().__init__()
        if (config.channels not in {(16, 32, 64, 128),(8,16,32,64)} or config.global_pooling != 'average'
                or config.global_pool_grid_size != 1):
            raise ValueError('E58 exige extractor E19 y GAP128')
        if config.channels==(8,16,32,64) and (fusion_hidden!=0 or image_pooling!='average' or feature_normalization!='none'):
            raise ValueError('Compacta exige fusión lineal original sin operaciones adicionales')
        image_width=config.channels[-1]
        self.image_encoder = BreastPCRNet(config)
        self.image_encoder.classifier = nn.Identity()
        self.clinical_branch = nn.Sequential(nn.Linear(8, 16), nn.ReLU())
        if type(fusion_hidden) is not int or fusion_hidden not in (0,32):
            raise ValueError('Fusión autorizada: lineal o hidden32')
        self.fusion_hidden = fusion_hidden
        self.fusion = (nn.Linear(image_width+16, 1) if fusion_hidden==0 else
                       nn.Sequential(nn.Linear(image_width+16,32),nn.ReLU(),nn.Linear(32,1)))
        layers=[self.clinical_branch[0], *[m for m in self.fusion.modules() if isinstance(m,nn.Linear)]]
        for layer in layers:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
        if type(image_pooling) is not str or image_pooling not in {'average','enhancement_weighted'}:
            raise ValueError('Pooling de imagen inválido')
        if image_pooling!='average' and fusion_hidden!=0:
            raise ValueError('Pooling de realce exige cabeza E58 original')
        self.image_pooling=image_pooling
        if image_pooling=='enhancement_weighted':
            from src.enhancement_weighted_pool import EnhancementWeightedPool
            self.enhancement_pool=EnhancementWeightedPool()
        if type(feature_normalization) is not str or feature_normalization not in {'none','l2_fixed'}:
            raise ValueError('Normalización de features inválida')
        if feature_normalization!='none' and (fusion_hidden!=0 or image_pooling!='average'):
            raise ValueError('Normalización fija exige E58 original sin combinar cabezas/pooling')
        self.feature_normalization=feature_normalization
        if feature_normalization=='l2_fixed':
            from src.image_feature_normalization import FixedImageFeatureNorm
            self.feature_normalizer=FixedImageFeatureNorm()
        else:self.feature_normalizer=nn.Identity()

    def image_features(self, image):
        e = self.image_encoder
        phases=image
        image = e.input_transform(e.input_normalizer(image))
        for block in e.blocks:
            image = block(image)
        pooled=(e.global_pool(image) if self.image_pooling=='average' else
                self.enhancement_pool(image,phases))
        return e.dropout(self.feature_normalizer(e.flatten(pooled)))

    def forward(self, image, clinical):
        if (clinical.ndim != 2 or clinical.shape != (image.shape[0], 8)
                or clinical.device != image.device or not clinical.is_floating_point()
                or not torch.isfinite(clinical).all()):
            raise ValueError('Se requieren ocho features clínicas válidas por corte')
        image_features = self.image_features(image)
        clinical_features = self.clinical_branch(clinical)
        return self.fusion(torch.cat([image_features, clinical_features], dim=1)).squeeze(-1)


def load_multimodal_checkpoint(path, device='cpu'):
    from src.experiment_config import experiment_config_from_dict
    ck = torch.load(path, map_location=device, weights_only=False)
    if ck.get('model_class') not in {'MultimodalPCRNet_v1','MultimodalPCRNet_nonlinear_v1','MultimodalPCRNet_enhancement_v1','MultimodalPCRNet_normalized_v1','MultimodalPCRNet_compact_v1'}:
        raise ValueError('No es un checkpoint multimodal')
    cfg = experiment_config_from_dict(ck['config']['experiment'])
    if (ck['model_class']=='MultimodalPCRNet_compact_v1') != (cfg.model.channels==(8,16,32,64)):
        raise ValueError('Anchura y versión del checkpoint no coinciden')
    hidden=ck['config'].get('fusion_hidden',0)
    pooling=ck['config'].get('image_pooling','average')
    normalization=ck['config'].get('feature_normalization','none')
    if (ck['model_class']=='MultimodalPCRNet_nonlinear_v1') != (hidden==32):
        raise ValueError('Cabeza y versión del checkpoint no coinciden')
    if (ck['model_class']=='MultimodalPCRNet_enhancement_v1') != (pooling=='enhancement_weighted'):
        raise ValueError('Pooling y versión del checkpoint no coinciden')
    if (ck['model_class']=='MultimodalPCRNet_normalized_v1') != (normalization=='l2_fixed'):
        raise ValueError('Normalización y versión del checkpoint no coinciden')
    model = MultimodalPCRNet(cfg.model,hidden,pooling,normalization).to(device)
    model.load_state_dict(ck['model_state_dict'], strict=True)
    return model.eval(), ClinicalPreprocessor.from_dict(ck['clinical_preprocessor'])
