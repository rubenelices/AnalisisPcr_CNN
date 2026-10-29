"""Caché uint8 de train público: mismos píxeles, sin estadísticas aprendidas."""
from pathlib import Path
import json
import hashlib
import numpy as np
import torch
from torch.utils.data import Dataset
from src.pipeline.data import BreastDCEDataset,load_samples

CACHE=Path('reports/cache/campana_20261015')


def build():
    torch.set_num_threads(2)
    if (CACHE/'manifest.json').exists():raise FileExistsError('Caché existente, conservar')
    CACHE.mkdir(parents=True,exist_ok=True)
    rows=load_samples();rows=rows.loc[rows.split=='train'].reset_index(drop=True)
    if rows.sample_id.duplicated().any():raise ValueError('Cortes duplicados')
    dataset=BreastDCEDataset(rows,'breastdcedl')
    shape=(len(rows),3,256,256);temporary=CACHE/'train.building'
    data=np.memmap(temporary,dtype=np.uint8,mode='w+',shape=shape)
    for i in range(len(rows)):
        x=dataset[i]['image'];encoded=(x*255).round().to(torch.uint8)
        if not torch.equal(x,encoded.to(torch.float32)/255.):raise ValueError('Caché no reproduce píxeles')
        data[i]=encoded.numpy()
        if i%2000==0:print('CACHE',i,'/',len(rows),flush=True)
    data.flush();del data;temporary.replace(CACHE/'train.uint8')
    manifest=dict(shape=list(shape),sample_ids=rows.sample_id.tolist(),split='train',test_images_loaded=False,
        metadata_sha256=hashlib.sha256(Path('breastdcedl/metadata/samples.csv').read_bytes()).hexdigest(),
        preprocessing='uint8 pixels -> float32 /255; three phases PRE/EARLY/LATE',
        cache_sha256=hashlib.sha256((CACHE/'train.uint8').read_bytes()).hexdigest())
    (CACHE/'manifest.json').write_text(json.dumps(manifest))
    print('CACHE COMPLETE',len(rows),'cortes',round((CACHE/'train.uint8').stat().st_size/1e9,2),'GB',flush=True)


class CachedDataset(Dataset):
    def __init__(self,rows,root='breastdcedl',transform=None):
        self.rows=rows.reset_index(drop=True).copy();self.transform=transform
        self.manifest=json.loads((CACHE/'manifest.json').read_text())
        if not self.rows.split.eq('train').all():raise ValueError('Caché exclusivamente train público')
        actual=hashlib.sha256((Path(root)/'metadata/samples.csv').read_bytes()).hexdigest()
        if self.manifest['metadata_sha256']!=actual:raise ValueError('Metadatos distintos de la caché')
        self.indices={sid:i for i,sid in enumerate(self.manifest['sample_ids'])}
        if not set(self.rows.sample_id).issubset(self.indices):raise ValueError('Corte no presente en caché')
        self.data=np.memmap(CACHE/'train.uint8',dtype=np.uint8,mode='r',shape=tuple(self.manifest['shape']))
    def __len__(self):return len(self.rows)
    def __getitem__(self,index):
        row=self.rows.iloc[index]
        image=torch.from_numpy(np.array(self.data[self.indices[row.sample_id]],copy=True)).float()/255.
        if self.transform is not None:image=self.transform(image)
        BreastDCEDataset._validate_transformed_tensor(image)
        return dict(image=image,target=torch.tensor(float(row.pCR),dtype=torch.float32),patient_id=str(row.patient_id),
            sample_id=str(row.sample_id),slice_index=int(row.slice_index),split=str(row.split),fold=int(row.fold))


if __name__=='__main__':build()
