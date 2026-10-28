"""Campo afín suave de intensidad, compartido por PRE/EARLY/LATE solo train."""
import math
import torch
from torch.utils.data import Dataset


class SharedIntensityField:
    def __init__(self,max_amplitude=.1):
        if (type(max_amplitude) not in (int,float) or not math.isfinite(max_amplitude)
                or not 0<=max_amplitude<=.2):
            raise ValueError('Amplitud exige número finito entre0 y0,2')
        self.max_amplitude=float(max_amplitude)

    def __call__(self,image):
        if (not isinstance(image,torch.Tensor) or image.dtype!=torch.float32
                or tuple(image.shape)!=(3,256,256) or not torch.isfinite(image).all()):
            raise ValueError('Campo exige tripleta float32 finita3×256×256')
        if self.max_amplitude==0:return image
        # Dos sorteos CPU: el cargador usa CPU; MPS recibe el mismo campo matemático.
        coefficients=(torch.rand(2)-.5)*self.max_amplitude
        y=torch.linspace(-1,1,256,device=image.device,dtype=image.dtype)[:,None]
        x=torch.linspace(-1,1,256,device=image.device,dtype=image.dtype)[None,:]
        field=1+float(coefficients[0])*x+float(coefficients[1])*y
        return image*field.unsqueeze(0)


class IntensityFieldTrainingDataset(Dataset):
    """Envolver exclusivamente train, sin modificar diccionario ni PNG originales."""
    def __init__(self,dataset,transform):self.dataset,self.transform=dataset,transform
    def __len__(self):return len(self.dataset)
    def __getitem__(self,index):
        sample=dict(self.dataset[index])
        sample['image']=self.transform(sample['image'])
        return sample
