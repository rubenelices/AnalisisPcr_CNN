"""K-Net propia: cuatro reducciones, cinco convoluciones, SE y GAP+GMP."""
from dataclasses import dataclass,asdict
import torch
from torch import nn
from src.phase_representation import PhaseRepresentation
from src.clinical_features import ClinicalPreprocessor


def activation(name):
    choices={'relu':nn.ReLU,'silu':nn.SiLU,'gelu':nn.GELU}
    if name not in choices:raise ValueError('Activación: relu, silu o gelu')
    return choices[name]()


@dataclass(frozen=True)
class KNetConfig:
    conv_activation:str='silu'
    head_activation:str='gelu'
    se_reduction:int=8
    dropout:float=.3
    def __post_init__(self):
        activation(self.conv_activation);activation(self.head_activation)
        if type(self.se_reduction) is not int or self.se_reduction<1:raise ValueError('Reducción SE inválida')
        if not 0<=self.dropout<1:raise ValueError('Dropout inválido')


class SqueezeExcitation(nn.Module):
    def __init__(self,channels,reduction,nonlinearity):
        super().__init__();hidden=max(4,channels//reduction)
        self.pool=nn.AdaptiveAvgPool2d(1)
        self.gate=nn.Sequential(nn.Linear(channels,hidden),activation(nonlinearity),nn.Linear(hidden,channels),nn.Sigmoid())
    def forward(self,x):
        return x*self.gate(self.pool(x).flatten(1))[:,:,None,None]


class ConvSE(nn.Module):
    def __init__(self,in_channels,out_channels,config):
        super().__init__()
        self.conv=nn.Conv2d(in_channels,out_channels,3,padding=1,bias=False)
        self.bn=nn.BatchNorm2d(out_channels)
        self.activation=activation(config.conv_activation)
        self.se=SqueezeExcitation(out_channels,config.se_reduction,config.conv_activation)
    def forward(self,x):return self.se(self.activation(self.bn(self.conv(x))))


class DualPool(nn.Module):
    def __init__(self):
        super().__init__();self.average=nn.AdaptiveAvgPool2d(1);self.maximum=nn.AdaptiveMaxPool2d(1)
    def forward(self,x):return torch.cat((self.average(x),self.maximum(x)),dim=1).flatten(1)


class KNet(nn.Module):
    """Entrada imagen PRE/EARLY/LATE y clínica [age,menopause,age_missing,menopause_missing]."""
    def __init__(self,config=KNetConfig()):
        super().__init__();self.config=config
        self.phases=PhaseRepresentation('pre_differences')
        self.input_bn=nn.BatchNorm2d(3)
        widths=[3,16,32,64,128,128]
        self.convs=nn.ModuleList([ConvSE(a,b,config) for a,b in zip(widths,widths[1:])])
        self.pools=nn.ModuleList([nn.MaxPool2d(2) for _ in range(4)])
        self.dual_pool=DualPool()
        self.image_head=nn.Sequential(nn.Dropout(config.dropout),nn.Linear(256,64),activation(config.head_activation))
        self.clinical_head=nn.Sequential(nn.Linear(4,16),nn.LayerNorm(16),activation(config.head_activation))
        self.fusion=nn.Linear(80,1)
        for layer in self.modules():
            if isinstance(layer,nn.Conv2d):nn.init.kaiming_normal_(layer.weight,mode='fan_out',nonlinearity='relu')
            elif isinstance(layer,nn.Linear):nn.init.xavier_uniform_(layer.weight);nn.init.zeros_(layer.bias)
    def image_features(self,x):
        if x.ndim!=4 or tuple(x.shape[1:])!=(3,256,256):raise ValueError('Imagen: [B,3,256,256] PRE/EARLY/LATE')
        x=self.input_bn(self.phases(x))
        for i,conv in enumerate(self.convs):
            x=conv(x)
            if i<4:x=self.pools[i](x)
        return self.image_head(self.dual_pool(x))
    def forward(self,image,clinical):
        if clinical.shape!=(len(image),4) or not torch.isfinite(clinical).all():raise ValueError('Se requieren cuatro features clínicas básicas')
        return self.fusion(torch.cat((self.image_features(image),self.clinical_head(clinical)),dim=1)).squeeze(-1)


def load_knet_checkpoint(path,device='cpu'):
    ck=torch.load(path,map_location=device,weights_only=False)
    if ck.get('model_class')!='KNet_SE_DualPool_v1':raise ValueError('No es un checkpoint K-Net')
    model=KNet(KNetConfig(**ck['config']['model'])).to(device);model.load_state_dict(ck['model_state_dict'],strict=True)
    prep=ClinicalPreprocessor.from_dict(ck['clinical_preprocessor'])
    if prep.active_columns!=('age','menopause'):raise ValueError('K-Net solo usa edad/menopausia')
    return model.eval(),prep
