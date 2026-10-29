"""Ficha compacta de campaña, medidas del modelo real y sin pacientes."""
from pathlib import Path
from dataclasses import replace
import torch
from torch import nn
from src.campaigns.campaign_20261015 import MODEL
from src.campaigns.campaign_20261015_models import CompactPatientFeatureNet


def document():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    torch.set_num_threads(2)
    model=CompactPatientFeatureNet(replace(MODEL,channels=(8,16,32,64))).eval();rows=[];handles=[]
    def hook(name):
        def record(layer,inputs,out):
            geometry=(f'k={layer.kernel_size}, p={layer.padding}, s={layer.stride}' if isinstance(layer,nn.Conv2d)
                else f'k={layer.kernel_size}, p={layer.padding}, s={layer.stride}' if isinstance(layer,nn.MaxPool2d) else 'Sin geometría espacial')
            rows.append((name,type(layer).__name__,tuple(inputs[0].shape),tuple(out.shape),sum(p.numel() for p in layer.parameters(recurse=False)),geometry))
        return record
    for name,layer in model.named_modules():
        if not list(layer.children()) and not isinstance(layer,nn.Identity):handles.append(layer.register_forward_hook(hook(name)))
    with torch.no_grad():out=model(torch.zeros(2,3,256,256))
    for h in handles:h.remove()
    count=sum(p.numel() for p in model.parameters());assert count==75961==sum(r[4] for r in rows) and out.shape==(2,)
    blocks=[]
    for i,b in enumerate(model.encoder.blocks):
        last=[r for r in rows if r[0].startswith(f'encoder.blocks.{i}.')][-1]
        blocks.append((last[3],sum(p.numel() for p in b.parameters())))
    assert [x[0] for x in blocks]==[(2,8,128,128),(2,16,64,64),(2,32,32,32),(2,64,16,16)]
    folder=Path('experiments/C15_E57_compacta_cortes');folder.mkdir(exist_ok=True)
    fig,ax=plt.subplots(figsize=(17,6));ax.set(xlim=(0,17),ylim=(0,6));ax.axis('off')
    ax.text(.3,5.55,'CNN compacta por corte · 75.961 parámetros',fontsize=24,fontfamily='serif',color='#213344')
    ax.text(.3,5.05,'Misma profundidad y BatchNorm que E57 · cambia la anchura · pesos propios desde cero',fontsize=12,color='#546778')
    nodes=[(.4,.4,2.3,'3 fases','3 × 256²',0)]
    nodes += [(2+i*2,.55+i*.08,2-i*.30,f'Bloque{i+1}',f'{shape[1]} × {shape[2]}²',params) for i,(shape,params) in enumerate(blocks)]
    nodes += [(10.1,.45,.8,'GAP + Flatten','64',0),(12.0,.6,.9,'Linear + ReLU\nDropout0,3','32',2080),(14.3,.35,.55,'Linear','1 logit',33)]
    for i,(x,w,h,title,shape,params) in enumerate(nodes):
        y=2.;d=.17;color='#699cba' if i==0 else '#62a895' if i<5 else '#a58abd'
        for points,alpha in [([(x,y),(x+w,y),(x+w,y+h),(x,y+h)],.28),
            ([(x,y+h),(x+d,y+h+d),(x+w+d,y+h+d),(x+w,y+h)],.18),
            ([(x+w,y),(x+w+d,y+d),(x+w+d,y+h+d),(x+w,y+h)],.4)]:
            ax.add_patch(Polygon(points,facecolor=color,edgecolor=color,alpha=alpha,lw=2))
        ax.text(x+w/2,4.55,title,ha='center',fontsize=10)
        ax.text(x+w/2,1.7,shape,ha='center',fontsize=11)
        ax.text(x+w/2,1.3,f'{params:,} parámetros',ha='center',fontsize=9,color='#546778')
        if i<len(nodes)-1:
            end=nodes[i+1][0]-.15
            ax.annotate('',xy=(end,2.55),xytext=(x+w+.22,2.55),arrowprops=dict(arrowstyle='->',color='#213344'))
    ax.text(.3,.7,'Bloque: Conv3×3 → BN → ReLU → MaxPool2×2 → Conv3×3 → BN → ReLU',fontsize=12)
    ax.text(.3,.25,'Salida: sigmoide por corte → media de probabilidades por paciente · RF Conv/Pool nominal106',fontsize=11)
    for ext in ('png','svg'):fig.savefig(folder/f'arquitectura.{ext}',dpi=150,bbox_inches='tight',facecolor='white')
    plt.close(fig)
    table='\n'.join(f'| {n} | {kind} | `{a}` | `{b}` | {p} | {g} |' for n,kind,a,b,p,g in rows)
    (folder/'README.md').write_text(f'''# C15 · CNN compacta por corte

![Arquitectura](arquitectura.png)

Modelo propio desde cero, sin pesos previos ni arquitectura importada. Ocho
convoluciones3×3/padding1/stride1, BatchNorm/ReLU y cuatroMaxPool2×2/stride2.
Canales8/16/32/64; GAP64 → Linear64→32 → ReLU → Dropout0,3 → Linear32→1.
Total real **{count} parámetros**; extractor73.848 y cabeza2.113.

Entrada externa3×256×256 PRE/EARLY/LATE. Representación interna fija PRE,
EARLY−PRE,LATE−EARLY; mismas fases/preprocesamiento en fit y predicción.
Las formas de la tabla incluyen batch2 ficticio. No se cargan imágenes reales.
RF local Conv/Pool nominal106×106; BN durante train y GAP incluyen dependencia
global adicional. La expansión de canales se reduce frente E57 para limitar
memorización manteniendo profundidad, BN, cabeza32 y dropout0,3. Menor capacidad
también puede perder señal; el experimento debe medirlo.

Entrenamiento BCE por corte, folds por paciente. Principal al evaluar: media de
probabilidades de los cortes por paciente. No implica cortes independientes.
El método forward_patients heredado existe por compatibilidad, pero no se usa
para entrenar estos controles por corte. No clínica ni atención.

| Capa | Tipo | Entrada | Salida | Parámetros | Geometría |
|---|---|---|---|---:|---|
{table}

Experimento: comparar frente CNN ancha con el MISMO régimen aumentado por
corte y selección interna/exterior. Solo cambia anchura y entrada necesaria de
la cabeza128→64; no volver a cambiar GroupNorm/LR/loss como B09. Configs y
protocolo en ../campana_20261015/PLAN.md.
Cargador: src.campaigns.campaign_20261015_checkpoint.load_nested_checkpoint.
Regenerar: `python -m src.reporting.document_architectures --only C15_E57_compacta_cortes`.

Uso educativo y de investigación, sin validez clínica.
''')
    print('C15 compacta documentada:',count,'parámetros y formas reales verificados')
    return count,rows


if __name__=='__main__':document()
