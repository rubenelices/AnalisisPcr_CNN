"""Lotes completos para controlar el objetivo multimodal por paciente."""
from torch.utils.data import DataLoader
import torch
from src.training.patient_training import PatientBatchSampler
from src.pipeline.data import seed_worker


class OrderedPatientBatchSampler(PatientBatchSampler):
    """Validación: grupos completos y orden fijo, sin sorteos."""
    def __iter__(self):
        for start in range(0,len(self.groups),self.patients_per_batch):
            yield [index for group in self.groups[start:start+self.patients_per_batch] for index in group]


def build_complete_patient_loaders(train_dataset,val_dataset,train_rows,val_rows,
                                  patients_per_batch=2,seed=42,num_workers=0,pin_memory=False):
    if type(patients_per_batch) is not int or patients_per_batch!=2:
        raise ValueError('Control prefijado exige dos pacientes completas por lote')
    if len(train_dataset)!=len(train_rows) or len(val_dataset)!=len(val_rows):
        raise ValueError('Filas y datasets incompatibles')
    train_generator=torch.Generator().manual_seed(seed)
    val_generator=torch.Generator().manual_seed(seed+1)
    common=dict(num_workers=num_workers,pin_memory=pin_memory,worker_init_fn=seed_worker,
                persistent_workers=num_workers>0)
    return (DataLoader(train_dataset,batch_sampler=PatientBatchSampler(train_rows,2,train_generator),
                       generator=train_generator,**common),
            DataLoader(val_dataset,batch_sampler=OrderedPatientBatchSampler(val_rows,2,val_generator),
                       generator=val_generator,**common))
