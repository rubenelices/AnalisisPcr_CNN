"""Ablación multimodal: pos_weight=(N0/N1)**exponente, solo train.

El default1 conserva exactamente la pérdida normal/ponderada anterior.
No calibración automática ni ajuste de probabilidades durante inferencia.
"""
import math
import torch
from torch import nn
from src.training.train import build_criterion


def validate_positive_weight_exponent(value,loss):
    if type(value) not in (int,float) or not math.isfinite(value) or not 0<value<=1:
        raise ValueError('positive_weight_exponent exige número finito en (0,1]')
    if value!=1 and loss!='weighted':
        raise ValueError('Exponente moderado exige loss weighted explícita')


def build_multimodal_criterion(config,train_rows,device,positive_weight_exponent=1.):
    validate_positive_weight_exponent(positive_weight_exponent,config.training.loss)
    criterion,weight=build_criterion(config,train_rows,device)
    if positive_weight_exponent==1:
        return criterion,weight
    weight=weight**positive_weight_exponent
    return nn.BCEWithLogitsLoss(pos_weight=torch.tensor(weight,dtype=torch.float32,device=device)),weight
