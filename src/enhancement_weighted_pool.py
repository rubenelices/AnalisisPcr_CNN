"""Promedio espacial propio guiado por realce EARLY-PRE, cero parámetros.

Regla fija de la imagen basal, sin etiquetas, máscaras externas ni transfer.
No es segmentación ni explicación causal. Mantener raw PRE/EARLY/LATE.
"""
import torch
from torch import nn
from torch.nn import functional as F


class EnhancementWeightedPool(nn.Module):
    def forward(self, features, phases):
        if (features.ndim!=4 or phases.ndim!=4 or phases.shape[1]!=3
                or features.shape[0]!=phases.shape[0]
                or not features.is_floating_point() or not phases.is_floating_point()
                or features.device!=phases.device or features.dtype!=phases.dtype):
            raise ValueError('Features y fases flotantes compatibles B×C×H×W/B×3×H×W')
        if not torch.isfinite(features).all() or not torch.isfinite(phases).all():
            raise ValueError('Pooling exige entradas finitas')
        if ((phases<0)|(phases>1)).any():
            raise ValueError('Fases originales deben estar en [0,1]')
        realce=(phases[:,1:2]-phases[:,0:1]).clamp_min(0)
        weights=F.adaptive_avg_pool2d(realce,features.shape[-2:])
        total=weights.sum(dim=(-2,-1),keepdim=True)
        safe_total=torch.where(total>0,total,torch.ones_like(total))
        normalized=weights/safe_total
        uniform=torch.full_like(weights,1/(features.shape[-2]*features.shape[-1]))
        normalized=torch.where(total>0,normalized,uniform)
        return (features*normalized).sum(dim=(-2,-1),keepdim=True)
