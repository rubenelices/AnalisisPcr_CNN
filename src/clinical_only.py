"""Control propio 8→16→1: clínica sola, sin cargar ni procesar PNG."""
import torch
from torch import nn
from torch.utils.data import Dataset


class ClinicalOnlyPCRNet(nn.Module):
    """La misma capacidad clínica de E58; no sustituye la CNN obligatoria."""
    def __init__(self):
        super().__init__()
        self.clinical_branch = nn.Sequential(nn.Linear(8, 16), nn.ReLU())
        self.fusion = nn.Linear(16, 1)
        for layer in (self.clinical_branch[0], self.fusion):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, image_placeholder, clinical):
        if (clinical.ndim != 2 or clinical.shape[1] != 8
                or not clinical.is_floating_point() or not torch.isfinite(clinical).all()):
            raise ValueError('Se requieren ocho features clínicas válidas')
        # El primer argumento solo mantiene la interfaz del evaluador compartido.
        return self.fusion(self.clinical_branch(clinical)).squeeze(-1)


class ClinicalCutDataset(Dataset):
    """Una fila por corte para igualar BCE, batches y pasos de E58.

    La clínica se repite, pero NO crea nuevas pacientes independientes.
    Mantener el índice de cortes empareja la ponderación implícita de E58.
    """
    def __init__(self, rows, clinical_patients, preprocessor):
        if clinical_patients.pid.duplicated().any():
            raise ValueError('Paciente clínica duplicada')
        self.rows = rows.reset_index(drop=True).copy()
        self.features = dict(zip(clinical_patients.pid, preprocessor.transform(clinical_patients)))
        if not set(self.rows.patient_id).issubset(self.features):
            raise ValueError('Falta clínica para alguna paciente')

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows.iloc[index]
        return dict(image=torch.zeros(1), clinical=self.features[row.patient_id].clone(),
                    target=torch.tensor(float(row.pCR), dtype=torch.float32),
                    patient_id=str(row.patient_id), sample_id=str(row.sample_id),
                    slice_index=int(row.slice_index), split=str(row.split), fold=int(row.fold))


def load_clinical_only_checkpoint(path, device='cpu'):
    from src.clinical_features import ClinicalPreprocessor
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint.get('model_class') != 'ClinicalOnlyPCRNet_v1':
        raise ValueError('No es un checkpoint de clínica sola')
    model = ClinicalOnlyPCRNet().to(device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    return model.eval(), ClinicalPreprocessor.from_dict(checkpoint['clinical_preprocessor'])
