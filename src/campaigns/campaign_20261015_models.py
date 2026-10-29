"""Control de anchura: CNN propia, ochoConv/BN, sin arquitecturas importadas."""
from torch import nn
from src.models.architectures import BreastPCRNet
from src.models.patient_feature_model import PatientFeatureNet


class CompactPatientFeatureNet(PatientFeatureNet):
    """CNN8/16/32/64, GAP64 y cabeza64→32→1;75.961 parámetros."""
    def __init__(self,config):
        nn.Module.__init__(self)
        if config.channels!=(8,16,32,64) or config.global_pooling!='average' or config.global_pool_grid_size!=1:
            raise ValueError('Compacta exige canales8/16/32/64 y GAP64')
        self.encoder=BreastPCRNet(config)
        self.encoder.classifier=nn.Identity();self.encoder.dropout=nn.Identity()
        self.head=nn.Sequential(nn.Linear(64,32),nn.ReLU(),nn.Dropout(config.dropout),nn.Linear(32,1))
        for layer in self.head:
            if isinstance(layer,nn.Linear):nn.init.xavier_uniform_(layer.weight);nn.init.zeros_(layer.bias)
