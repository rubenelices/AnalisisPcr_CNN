"""Ficha D19: formas y parámetros obtenidos por hooks del modelo real."""
from pathlib import Path
import torch
from torch import nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from src.train_multimodal import load_spec
from src.multimodal_model import MultimodalPCRNet


def document():
    torch.set_num_threads(2)
    spec,cfg=load_spec('experimentos_datos/D19_norma_features/config.json')
    model=MultimodalPCRNet(cfg.model,feature_normalization=spec['feature_normalization']).eval()
    rows=[];hooks=[]
    def hook(name):
        def record(layer,inputs,out):
            geo=(f'k={layer.kernel_size}, p={layer.padding}, s={layer.stride}' if isinstance(layer,nn.Conv2d)
                 else f'k={layer.kernel_size}, s={layer.stride}' if isinstance(layer,nn.MaxPool2d) else '—')
            rows.append((name,type(layer).__name__,tuple(inputs[0].shape),tuple(out.shape),
                         sum(p.numel() for p in layer.parameters(recurse=False)),geo))
        return record
    for name,layer in model.named_modules():
        if not list(layer.children()) and not isinstance(layer,nn.Identity):hooks.append(layer.register_forward_hook(hook(name)))
    with torch.inference_mode():out=model(torch.zeros(1,3,256,256),torch.zeros(1,8))
    for h in hooks:h.remove()
    count=sum(p.numel() for p in model.parameters())
    assert count==sum(r[4] for r in rows)==294289 and out.shape==(1,)
    norm=[r for r in rows if r[0]=='feature_normalizer'][0]
    assert norm[2]==norm[3]==(1,128) and norm[4]==0
    blocks=[]
    for i in range(4):
        last=[r for r in rows if r[0].startswith(f'image_encoder.blocks.{i}.')][-1]
        blocks.append((last[3],sum(p.numel() for p in model.image_encoder.blocks[i].parameters())))
    folder=Path('experimentos/D19_norma_features');folder.mkdir(parents=True,exist_ok=True)
    fig,ax=plt.subplots(figsize=(19,8));ax.set(xlim=(0,19),ylim=(0,8));ax.axis('off')
    ax.text(.3,7.45,'D19 · El embudo de la CNN con norma fija de imagen',fontsize=24,fontfamily='serif',color='#17263d')
    ax.text(.3,6.9,'Misma CNN propia E58 y clínica · único cambio: L2 en 128 características antes de dropout',fontsize=13,color='#657080')
    def volume(x,y,w,h,color,title,bottom):
        d=.2
        for points,alpha in [([(x,y),(x+w,y),(x+w,y+h),(x,y+h)],.25),
            ([(x,y+h),(x+d,y+h+d),(x+w+d,y+h+d),(x+w,y+h)],.2),
            ([(x+w,y),(x+w+d,y+d),(x+w+d,y+h+d),(x+w,y+h)],.4)]:
            ax.add_patch(Polygon(points,facecolor=color,edgecolor=color,lw=2,alpha=alpha))
        ax.text(x+w/2,y+h+.35,title,ha='center',fontsize=10)
        ax.text(x+w/2,y-.2,bottom,ha='center',va='top',fontsize=10,color='#657080')
    volume(.5,3.3,.3,2.3,'#80a6db','Entrada\n3 fases','3 × 256²')
    ax.text(.35,2.55,'PRE / EARLY−PRE / LATE−EARLY',fontsize=9)
    for i,(shape,params) in enumerate(blocks):
        x=2+i*1.8
        volume(x,3.3,.6+i*.13,2.05-i*.3,'#68bda0',f'Bloque {i+1}\nConv–BN–ReLU\nPool\nConv–BN–ReLU',f'{shape[1]} × {shape[2]}²\n{params:,} parámetros')
        ax.annotate('',xy=(x-.1,4.15),xytext=(x-.65,4.15),arrowprops=dict(arrowstyle='->',color='#17263d'))
    volume(9.3,3.3,.45,.9,'#a481c1','GAP','128')
    volume(10.6,3.3,.55,.9,'#5391cc','Norma L2 fija\n+ dropout0,3','128 · 0 parámetros')
    ax.annotate('',xy=(10.4,3.8),xytext=(9.9,3.8),arrowprops=dict(arrowstyle='->'))
    volume(10.6,1.05,.55,.8,'#ecaf65','Clínica8→16\nReLU','16 · 144 parámetros')
    ax.text(.5,1.4,'Edad · HR · HER2 · menopausia\n4 valores + 4 indicadores de ausencia\nEstadísticas ajustadas solo en train',fontsize=12)
    ax.annotate('',xy=(10.4,1.4),xytext=(7.5,1.4),arrowprops=dict(arrowstyle='->'))
    for y in [3.8,1.5]:ax.annotate('',xy=(12.1,2.8),xytext=(11.35,y),arrowprops=dict(arrowstyle='->'))
    ax.text(13.25,2.8,'Concatenar\n128 + 16 = 144',ha='center',fontsize=11)
    volume(15,2.5,.4,.6,'#a481c1','Lineal144→1','1 · 145 parámetros')
    ax.annotate('',xy=(14.8,2.8),xytext=(14.2,2.8),arrowprops=dict(arrowstyle='->'))
    ax.annotate('',xy=(16.8,2.8),xytext=(15.7,2.8),arrowprops=dict(arrowstyle='->'))
    ax.text(17.5,3.15,'Logit → sigmoide\nProbabilidad del corte',ha='center',va='bottom',fontsize=11)
    ax.text(17.5,2.4,'↓ media por paciente',ha='center',fontsize=11)
    ax.text(.5,.15,'294.289 parámetros · Conv3×3/p1/s1 · Pool2×2/s2 · RF106 · Norma sqrt128×v/max(||v||₂,1e−6)',fontsize=11)
    for ext in ['png','svg']:fig.savefig(folder/f'arquitectura.{ext}',dpi=150,bbox_inches='tight',facecolor='white')
    plt.close(fig)
    table='\n'.join(f'| {n} | {kind} | `{a}` | `{b}` | {p} | {g} |' for n,kind,a,b,p,g in rows)
    (folder/'README.md').write_text(f'''# D19 · Norma fija antes de fusionar

![Arquitectura](arquitectura.png)

Misma CNN E58 propia desde cero:8Conv/8BN-ReLU/4MaxPool entre convoluciones,
canales16-32-64-128/resolución256→128→64→32→16/GAP128/RF local106.
Única operación nueva128→128 antes del dropout0,3:
`sqrt128 * v / max(normL2(v),1e-6)`. Cero parámetros, total real{count}.
Norma por corte, sin estadísticas de batch ni ajuste a pacientes/etiquetas.
No normaliza las fases por separado ni cambia el cargador PNG. Esta operación
vive dentro del modelo y también se usa al evaluar e inferir.

Clínica8→16/ReLU y fusión lineal144→1 intactas. La capa final puede compensar
el cambio aumentando sus pesos; la magnitud de features podría ser información
útil que perdamos. No afirmar que esto resuelva el sobreajuste.
La norma se calcula reescalando internamente por maxabs para evitar overflow;
la fórmula es equivalente. Epsilon y escala prefijados, no elegidos en validación.
El vector cero sigue cero, gradientes probadosCPU/MPS. No pesos externos/transfer.
El GAP reúne todo el mapa, no extiende RF local de las convoluciones.

Inicializaciones anteriores Kaiming/Xavier se conservan: operación sin RNG ni
pesos. El default Identity preserva exactamente estados/checkpoints antiguos.
Nueva versión MultimodalPCRNet_normalized_v1 exige coherencia de configuración.
Norma sin kernel/padding/stride espacial; convoluciones/pool en tabla real.

| Capa | Tipo | Entrada | Salida | Parámetros | Geometría |
|---|---|---|---|---:|---|
{table}

[Protocolo](../../experimentos_datos/D19_norma_features/README.md).
Generador: `python -m src.document_feature_norm` (no datos ni pesos históricos).
Uso educativo y de investigación, sin validez clínica.
''',encoding='utf-8')
    print('D19 documentado:294289 parámetros, norma128→128/0pesos comprobados')


if __name__=='__main__':document()
