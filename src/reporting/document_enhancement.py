"""Ficha D14: formas y parámetros verificados en el modelo real."""
from pathlib import Path
import torch
from torch import nn
from src.models.multimodal_model import MultimodalPCRNet
from src.training.train_multimodal import load_spec


def document():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    torch.set_num_threads(2)
    spec,cfg=load_spec('experiments/D14_pooling_realce/config.json')
    model=MultimodalPCRNet(cfg.model,image_pooling=spec['image_pooling']).eval()
    rows=[];handles=[]
    def hook(name):
        def record(layer,inputs,output):
            geometry=(f'k={layer.kernel_size},p={layer.padding},s={layer.stride}' if isinstance(layer,nn.Conv2d)
                      else f'k={layer.kernel_size},s={layer.stride}' if isinstance(layer,nn.MaxPool2d) else '—')
            rows.append((name,type(layer).__name__,[tuple(x.shape) for x in inputs],
                         tuple(output.shape),sum(p.numel() for p in layer.parameters(recurse=False)),geometry))
        return record
    for name,layer in model.named_modules():
        if not list(layer.children()) and not isinstance(layer,nn.Identity):handles.append(layer.register_forward_hook(hook(name)))
    phases=torch.zeros(1,3,256,256);phases[:,1,:128]=.5
    with torch.no_grad():out=model(phases,torch.zeros(1,8))
    for h in handles:h.remove()
    count=sum(p.numel() for p in model.parameters())
    assert count==sum(r[4] for r in rows)==294289 and out.shape==(1,)
    pool=next(r for r in rows if r[0]=='enhancement_pool')
    assert pool[2]==[(1,128,16,16),(1,3,256,256)] and pool[3]==(1,128,1,1)
    assert not any(r[0]=='image_encoder.global_pool' for r in rows)
    folder=Path('experiments/D14_pooling_realce');folder.mkdir(parents=True,exist_ok=True)
    fig,ax=plt.subplots(figsize=(18,7));ax.set(xlim=(0,18),ylim=(0,7));ax.axis('off')
    ax.text(.3,6.4,'D14 · CNN con pooling guiado por realce',fontsize=24,fontfamily='serif',color='#17263d')
    ax.text(.3,5.9,'Misma CNN E58 · único cambio: cómo se resumen los mapas finales · cero parámetros nuevos',fontsize=12,color='#657080')
    def volume(x,y,w,h,color,title,bottom):
        d=.2
        for vertices,alpha in [([(x,y),(x+w,y),(x+w,y+h),(x,y+h)],.25),
            ([(x,y+h),(x+d,y+h+d),(x+w+d,y+h+d),(x+w,y+h)],.2),
            ([(x+w,y),(x+w+d,y+d),(x+w+d,y+h+d),(x+w,y+h)],.4)]:
            ax.add_patch(Polygon(vertices,facecolor=color,edgecolor=color,lw=2,alpha=alpha))
        ax.text(x+w/2,y+h+.4,title,ha='center',fontsize=10)
        ax.text(x+w/2,y-.25,bottom,ha='center',fontsize=10,va='top')
    volume(.5,3,.3,1.7,'#80a6db','Entrada\n3 fases','3 × 256²')
    for i in range(4):
        volume(2+i*1.8,3,.65,1.6-i*.25,'#68bda0',f'Bloque {i+1}\nConv–BN–ReLU\nPool\nConv–BN–ReLU',f'{16*2**i} × {256//2**(i+1)}²')
    volume(9.4,3,.5,.8,'#ecaf65','Media ponderada\nDropout0,3','128 imagen')
    ax.text(.5,1.8,'EARLY − PRE → positivo → media de bloques16² → normalizar pesos',fontsize=11)
    ax.text(.5,1.25,'Mapa de pesos16 × 16; si todo es cero → pesos uniformes\nRegla fija de imagen: sin etiquetas, máscaras externas ni pesos aprendidos',fontsize=11,color='#657080')
    ax.annotate('',xy=(9.6,3),xytext=(8.5,1.9),arrowprops=dict(arrowstyle='->',color='#d5a35b'))
    volume(9.4,.5,.5,.55,'#a481c1','Clínica8→16\nReLU','16 clínica')
    for y in (3.5,.9):ax.annotate('',xy=(11.4,2.6),xytext=(10.1,y),arrowprops=dict(arrowstyle='->'))
    ax.text(11.8,2.6,'Concatenar\n144',ha='center',fontsize=11)
    volume(13.2,2.2,.5,.7,'#a481c1','Lineal144→1','1 · 145 parámetros')
    ax.annotate('',xy=(15.3,2.6),xytext=(14,2.6),arrowprops=dict(arrowstyle='->'))
    ax.text(16,2.6,'Logit → sigmoide\nProbabilidad por corte\n→ media por paciente',ha='center',fontsize=11)
    ax.text(.4,-.2,'294.289 parámetros · Conv3×3/p1/s1 · Pool2×2/s2 · RF local106 · el mapa de pesos NO explica causalidad',fontsize=11)
    for ext in ('png','svg'):fig.savefig(folder/f'arquitectura.{ext}',dpi=160,bbox_inches='tight',facecolor='white')
    plt.close(fig)
    table='\n'.join(f'| {name} | {kind} | `{inp}` | `{output}` | {params} | {geo} |' for name,kind,inp,output,params,geo in rows)
    text=f'''# D14 · Pooling guiado por realce

![Arquitectura](arquitectura.png)

Misma CNN propia desde cero E58,8convs/8BNReLU/4MaxPool intermedios,
canales16/32/64/128 y campo receptivo local106. Clínica8→16/ReLU y fusión144→1.
Totalreal **{count}**; exactamente mismos pesos iniciales y RNG que E58 al
construir, nueva operación sin parámetros/buffers. No transfer ni modelos externos.

## Único cambio: resumen espacial

Entrada adicional de pooling: fases originales3×256². `r=relu(EARLY−PRE)`;
mediaadaptativa256²→16² (kernel16/stride16 en este tamaño), dividir por suma
para obtener pesos que suman1. Sin realce positivo, pesos uniformes1/256.
Salida: suma ponderada de features128×16²→128×1². Flatten/dropout0,3 idénticos.
La diferencia negativa se ignora SOLO en pesos; CNN sigue recibiendo PRE y
diferencias FIRMADAS, incluido LATE−EARLY. No cambia normalización compartida.
Pooling no tiene kernel aprendible ni parámetros; todas fases siguen en CNN.
Mapa no es máscara tumoral, Grad-CAM ni explicación causal; puede suprimir
contexto importante o amplificar artefactos. Regla idéntica train/eval/inferencia.

| Capa | Tipo | Entradas | Salida | Parámetros | Geometría |
|---|---|---|---|---:|---|
{table}

[Protocolo y resultados](../D14_pooling_realce/README.md).
Generador: `python -m src.reporting.document_enhancement`; SVG/PNG y tabla del modelo real.
Uso educativo, sin validez clínica.
'''
    (folder/'README.md').write_text(text)
    print('D14:294289parámetros, pooling(128,16,16)+(3,256,256)→(128,1,1) comprobado')


if __name__=='__main__':document()
