"""Restauración de checkpoints anidados; no son checkpoints del runner histórico."""
import torch
from src.models.architectures import BaseCNNConfig
from src.models.patient_feature_model import PatientFeatureNet


def load_nested_checkpoint(path,device='cpu'):
    ck=torch.load(path,map_location=device,weights_only=False)
    if ck.get('model_class') not in {'PatientFeatureNet_v1','CompactPatientFeatureNet_v1'} or ck.get('campaign_config',{}).get('schema')!='nested_e57_20261015':
        raise ValueError('No es un checkpoint anidado de esta campaña')
    config=BaseCNNConfig(**ck['config']['experiment']['model'])
    if ck['model_class']=='CompactPatientFeatureNet_v1':
        from src.campaigns.campaign_20261015_models import CompactPatientFeatureNet
        model=CompactPatientFeatureNet(config).to(device)
    else:model=PatientFeatureNet(config).to(device)
    model.load_state_dict(ck['model_state_dict'],strict=True)
    if not all(torch.isfinite(x).all() for x in model.state_dict().values()):raise ValueError('Pesos no finitos')
    return model.eval(),ck['campaign_config'],int(ck['epoch'])
