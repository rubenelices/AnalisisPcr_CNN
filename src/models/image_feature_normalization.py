"""Escala L2 fija por corte antes de fusionar, sin estadísticas ni pesos."""
import math
import torch
from torch import nn


class FixedImageFeatureNorm(nn.Module):
    """v -> sqrt128*v/max(norm(v),1e-6), forma[B,128] y cero parámetros.

    Reescala internamente por el máximo absoluto para evitar overflow al calcular
    normas de float32 grandes. Es la misma fórmula, no otro preprocesamiento.
    No estima nada con pacientes, ni cruza muestras del batch, ni toca la clínica.
    """
    feature_count=128
    epsilon=1e-6

    def forward(self,features):
        if (features.ndim!=2 or features.shape[1]!=self.feature_count
                or features.shape[0]<1 or features.dtype!=torch.float32
                or not torch.isfinite(features).all()):
            raise ValueError('Normalización exige features float32 finitas[B,128]')
        scale=features.detach().abs().amax(dim=1,keepdim=True).clamp_min(self.epsilon)
        stable=features/scale
        denominator=torch.maximum(torch.linalg.vector_norm(stable,dim=1,keepdim=True),
                                  self.epsilon/scale)
        return math.sqrt(self.feature_count)*stable/denominator
