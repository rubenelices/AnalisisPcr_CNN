"""Ficha D23 obtenida del modelo propio real, sin datos ni pesos históricos."""
from pathlib import Path
import torch
from torch import nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from src.train_multimodal import load_spec
from src.multimodal_model import MultimodalPCRNet


def document(output_dir=None):
    torch.set_num_threads(2)
    _,cfg=load_spec('experimentos_datos/D23_cnn_compacta/config.json')
    model=MultimodalPCRNet(cfg.model).eval();rows=[];handles=[]
    def hook(name):
        def record(layer,inputs,out):
            geometry=(f'k={layer.kernel_size}, p={layer.padding}, s={layer.stride}' if isinstance(layer,nn.Conv2d)
                else f'k={layer.kernel_size}, s={layer.stride}' if isinstance(layer,nn.MaxPool2d) else '—')
            rows.append((name,type(layer).__name__,tuple(inputs[0].shape),tuple(out.shape),
                         sum(p.numel() for p in layer.parameters(recurse=False)),geometry))
        return record
    for name,layer in model.named_modules():
        if not list(layer.children()) and not isinstance(layer,nn.Identity):
            handles.append(layer.register_forward_hook(hook(name)))
    with torch.inference_mode():out=model(torch.zeros(1,3,256,256),torch.zeros(1,8))
    for h in handles:h.remove()
    count=sum(p.numel() for p in model.parameters())
    assert count==sum(r[4] for r in rows)==74073 and out.shape==(1,)
    assert model.fusion.in_features==80
    blocks=[]
    for i in range(4):
        last=[r for r in rows if r[0].startswith(f'image_encoder.blocks.{i}.')][-1]
        blocks.append((last[3],sum(p.numel() for p in model.image_encoder.blocks[i].parameters())))
    assert [b[0] for b in blocks]==[(1,8,128,128),(1,16,64,64),(1,32,32,32),(1,64,16,16)]
    folder=Path(output_dir) if output_dir is not None else Path('experimentos/D23_cnn_compacta')
    folder.mkdir(parents=True,exist_ok=True)
    fig,ax=plt.subplots(figsize=(19,8));ax.set(xlim=(0,19),ylim=(0,8));ax.axis('off')
    ax.text(.3,7.45,'D23 · Una CNN más compacta, con la misma resolución',fontsize=24,fontfamily='serif',color='#17263d')
    ax.text(.3,6.9,'Canales 8 → 16 → 32 → 64 · clínica intacta · 74.073 parámetros propios, desde cero',fontsize=13,color='#657080')
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
        volume(x,3.3,.6+i*.13,2.05-i*.3,'#68bda0',f'Bloque {i+1}\nConv–BN–ReLU\nPool\nConv–BN–ReLU',
               f'{shape[1]} × {shape[2]}²\n{params:,} parámetros')
        ax.annotate('',xy=(x-.1,4.15),xytext=(x-.65,4.15),arrowprops=dict(arrowstyle='->',color='#17263d'))
    volume(9.3,3.3,.45,.9,'#a481c1','GAP','64')
    volume(10.6,3.3,.55,.9,'#5391cc','Dropout0,3','64 · 0 parámetros')
    ax.annotate('',xy=(10.4,3.8),xytext=(9.9,3.8),arrowprops=dict(arrowstyle='->'))
    volume(10.6,1.05,.55,.8,'#ecaf65','Clínica8→16\nReLU','16 · 144 parámetros')
    ax.text(.5,1.4,'Edad · HR · HER2 · menopausia\n4 valores + 4 indicadores de ausencia\nEstadísticas ajustadas solo en train',fontsize=12)
    ax.annotate('',xy=(10.4,1.4),xytext=(7.5,1.4),arrowprops=dict(arrowstyle='->'))
    for y in [3.8,1.5]:ax.annotate('',xy=(12.1,2.8),xytext=(11.35,y),arrowprops=dict(arrowstyle='->'))
    ax.text(13.25,2.8,'Concatenar\n64 + 16 = 80',ha='center',fontsize=11)
    volume(15,2.5,.4,.6,'#a481c1','Lineal80→1','1 · 81 parámetros')
    ax.annotate('',xy=(14.8,2.8),xytext=(14.2,2.8),arrowprops=dict(arrowstyle='->'))
    ax.annotate('',xy=(16.8,2.8),xytext=(15.7,2.8),arrowprops=dict(arrowstyle='->'))
    ax.text(17.5,3.15,'Logit → sigmoide\nProbabilidad del corte',ha='center',va='bottom',fontsize=11)
    ax.text(17.5,2.4,'↓ media por paciente',ha='center',fontsize=11)
    ax.text(.5,.15,'Conv3×3/p1/s1 · Pool2×2/s2 · RF local106 · volúmenes esquemáticos, dimensiones reales indicadas',fontsize=11)
    for ext in ['png','svg']:fig.savefig(folder/f'arquitectura.{ext}',dpi=150,bbox_inches='tight',facecolor='white')
    plt.close(fig)
    table='\n'.join(f'| {n} | {kind} | `{a}` | `{b}` | {p} | {g} |' for n,kind,a,b,p,g in rows)
    (folder/'README.md').write_text(f'''# D23 · CNN compacta multimodal

![Arquitectura](arquitectura.png)

CNN propia desde cero, no arquitectura importada:8Conv3×3/p1/s1,
8BatchNorm/ReLU y4MaxPool2×2/s2 entre cada pareja de convoluciones.
Canales8/16/32/64, mapas256→128→64→32→16, GAP64 y dropout0,3.
Clínica8→16/ReLU, concatenación64+16=80, fusión lineal80→1.
Total real{count} parámetros, comprobado por hooks y conteo del modelo.

Frente E58 se cambia la anchura de CNN y, necesariamente, la entrada de
fusión144→80. Se conserva resolución y campo receptivo local106; GAP resume
todo el mapa, no aumenta ese campo receptivo local. No norma fija ni atención.
Menos capacidad puede reducir memorización o perder señal: medirlo, no asumirlo.
Misma política Kaiming/Xavier, pero formas distintas cambian consumo aleatorio;
no garantizar pesos comunes idénticos. No transfer ni pesos externos/anteriores.

Entrada por corte3×256×256 PRE/EARLY/LATE; representación interna PRE,
EARLY−PRE,LATE−EARLY. No RGB. Misma transformación al entrenar e inferir.
La salida de entrenamiento es un logit; sigmoide solo al evaluar y media de
probabilidades por paciente. Checkpoint MultimodalPCRNet_compact_v1 exige
configuración coherente; no etiquetarlo como E58 antiguo. Clínica fit solo train.

| Capa | Tipo | Entrada | Salida | Parámetros | Geometría |
|---|---|---|---|---:|---|
{table}

[Protocolo D23](../../experimentos_datos/D23_cnn_compacta/README.md).
Generador: `python -m src.document_compact_multimodal`, sin datos/pesos históricos.
Uso educativo y de investigación, sin validez clínica.
''',encoding='utf-8')
    print('D23 documentado:74073 parámetros/GAP64/fusión80→1 comprobados')
    return count,rows


if __name__=='__main__':document()
