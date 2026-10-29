"""Oclusión cuadrada pequeña de train, idéntica en PRE/EARLY/LATE.

Adaptación propia de la idea Cutout/Random Erasing: no pesos externos,
no nuevos pacientes y ninguna modificación de los PNG originales.
"""
import math
import torch
from torch.utils.data import Dataset


class SynchronizedOcclusion:
    def __init__(self, probability=.25, side=32):
        if type(probability) not in (int,float) or not math.isfinite(probability) or not 0<=probability<=1:
            raise ValueError('Probabilidad de oclusión inválida')
        if type(side) is not int or not 1<=side<=64:
            raise ValueError('Lado de oclusión exige entero entre1 y64')
        self.probability=float(probability)
        self.side=side

    def __call__(self, image):
        if image.ndim!=3 or image.shape[0]!=3 or not image.is_floating_point():
            raise ValueError('Oclusión exige tensor flotante3×H×W')
        h,w=image.shape[-2:]
        if self.side>min(h,w): raise ValueError('Oclusión mayor que imagen')
        if self.probability==0: return image
        if self.probability<1 and torch.rand(()).item()>=self.probability: return image
        top=int(torch.randint(h-self.side+1,()).item())
        left=int(torch.randint(w-self.side+1,()).item())
        result=image.clone()
        result[:,top:top+self.side,left:left+self.side]=0
        return result


class OccludedTrainingDataset(Dataset):
    """Envolver SOLO train; clínica, etiquetas e identificadores intactos."""
    def __init__(self, dataset, transform):
        self.dataset=dataset
        self.transform=transform

    def __len__(self): return len(self.dataset)

    def __getitem__(self,index):
        sample=dict(self.dataset[index])
        sample['image']=self.transform(sample['image'])
        return sample
