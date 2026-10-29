"""Ficha K-Net: volúmenes y capas comprobados con inferencia sintética en eval."""
from pathlib import Path
import json,math
import pandas as pd,torch
from torch import nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon,FancyBboxPatch
from src.models.knet_model import KNet,KNetConfig


def document():
    root=Path('experiments/K01_knet_se_dualpool');root.mkdir(parents=True,exist_ok=True)
    model=KNet().eval();torch.set_num_threads(2);rows=[];handles=[]
    def hook(name):
        def record(layer,inputs,out):
            geometry=f'kernel={layer.kernel_size}, stride={layer.stride}, padding={layer.padding}' if isinstance(layer,nn.Conv2d) else f'kernel={layer.kernel_size}, stride={layer.stride}' if isinstance(layer,nn.MaxPool2d) else '—'
            rows.append(dict(capa=name,tipo=type(layer).__name__,entrada=str(tuple(inputs[0].shape)),salida=str(tuple(out.shape)),parametros=sum(p.numel() for p in layer.parameters(recurse=False)),geometria=geometry))
        return record
    for name,layer in model.named_modules():
        if not list(layer.children()):handles.append(layer.register_forward_hook(hook(name)))
    with torch.inference_mode():result=model(torch.zeros(2,3,256,256),torch.zeros(2,4))
    for h in handles:h.remove()
    count=sum(p.numel() for p in model.parameters());assert count==272055 and result.shape==(2,) and sum(r['parametros'] for r in rows)==count
    display_count=f'{count:,}'.replace(',','.')
    pd.DataFrame(rows).to_csv(root/'capas.csv',index=False)
    # Outputs of complete blocks/pooling obtained with hooks, independent from schematic sizes.
    observed={};handles=[]
    for name,layer in [('phases',model.phases),('input_bn',model.input_bn),*[(f'conv{i}',m) for i,m in enumerate(model.convs)],*[(f'pool{i}',m) for i,m in enumerate(model.pools)],('dual',model.dual_pool),('image',model.image_head),('clinical',model.clinical_head),('fusion',model.fusion)]:
        handles.append(layer.register_forward_hook(lambda layer,args,out,name=name:observed.__setitem__(name,list(out.shape[1:]))))
    with torch.inference_mode():model(torch.zeros(2,3,256,256),torch.zeros(2,4))
    for h in handles:h.remove()
    (root/'dimensiones_verificadas.json').write_text(json.dumps(dict(parameters=count,observed=observed),indent=2))
    stages=[('Entrada\n3 fases',(3,256,256),'input'),('PRE\nEARLY−PRE\nLATE−EARLY\n+ BN',tuple(observed['input_bn']),'input')]
    for i in range(5):
        stages.append(('Conv3×3\nBN + SiLU + SE',tuple(observed[f'conv{i}']),'conv'))
        if i<4:stages.append(('MaxPool\n2×2',tuple(observed[f'pool{i}']),'pool'))
    stages += [('GAP + GMP\nConcatenar',tuple(observed['dual']),'head'),('Dropout0,3\nLineal256→64\nGELU',tuple(observed['image']),'head'),('Concatenar\n64 + 16',(80,),'head'),('Lineal80→1\nLogit',tuple(observed['fusion']),'head')]
    n=len(stages);fig,ax=plt.subplots(figsize=(max(25,n*1.75),11));ax.set(xlim=(-1,n+2),ylim=(-5.3,5));ax.axis('off');fig.patch.set_facecolor('white')
    colors={'input':('#e5f0fb','#4782a4'),'conv':('#e0f0e9','#55977e'),'pool':('#fff0d8','#bd904e'),'head':('#f0e5f5','#79528a')}
    def volume(x,y,shape,kind,label):
        c=shape[0];side=shape[-1] if len(shape)==3 else 1;h=.6+2.2*math.log2(side+1)/math.log2(257) if len(shape)==3 else .7+min(c/256,.5);w=.25+.32*min(math.log2(c+1)/8,1);l=x-w/2;b=y-h/2;dx=.18;dy=.22;fill,edge=colors[kind]
        for face in [[(l,b),(l+w,b),(l+w,b+h),(l,b+h)],[(l,b+h),(l+dx,b+h+dy),(l+w+dx,b+h+dy),(l+w,b+h)],[(l+w,b),(l+w+dx,b+dy),(l+w+dx,b+h+dy),(l+w,b+h)]]:ax.add_patch(Polygon(face,closed=True,facecolor=fill,edgecolor=edge,lw=1.7))
        ax.text(x,3.55 if y>0 else y+1.2,label,ha='center',va='center',fontsize=10 if y>0 else 11,linespacing=1.05 if y>0 else 1.5)
        dim=f'{shape[0]} @ {shape[-1]}²' if len(shape)==3 else str(shape[0]);ax.text(x,-.55 if y>0 else y-.85,dim,ha='center',fontsize=12)
        return l,l+w+dx
    def arrow(a,b):ax.annotate('',xy=b,xytext=a,arrowprops=dict(arrowstyle='->',color='#a1a7ad',lw=1.1))
    ends=[]
    for i,(label,shape,kind) in enumerate(stages):
        ends.append(volume(i,1.5,shape,kind,label))
        if i:arrow((ends[i-1][1]+.03,1.5),(ends[i][0]-.05,1.5))
    arrow((ends[-1][1]+.03,1.5),(n-.2,1.5));ax.text(n,1.7,'Sigmoide\nProbabilidad\ndel corte',fontsize=12,va='center');ax.text(n,.1,'↓ Media por paciente\nTop5: diagnóstico\nseparado',fontsize=11)
    a,b=volume(n-6,-2.3,(4,),'input','Edad y menopausia\n+ ausencias');c,d=volume(n-4.3,-2.3,(16,),'head','Lineal4→16\nLayerNorm + GELU');arrow((b+.04,-2.3),(c-.05,-2.3));join=n-2
    ax.plot([d+.03,join-.7,join-.7],[-2.3,-2.3,1.0],color='#a1a7ad');arrow((join-.7,1.0),(ends[-2][0]-.03,1.0))
    ax.text(-.7,4.55,'K-Net · Atención de canal + doble pooling + fusión clínica',fontsize=27,fontfamily='DejaVu Serif')
    ax.text(-.7,4.0,'CNN propia desde cero · cuatro bloques con reducción y un refinamiento · SiLU en convoluciones / GELU en cabezas',fontsize=15)
    ax.add_patch(FancyBboxPatch((-.7,-5.05),n+2.25,1.1,boxstyle='round,pad=.1',facecolor='#edf4fb',edgecolor='#9bb6c8'))
    ax.text(-.4,-4.3,f'{display_count} parámetros reales · SE reducción8, sigmoid de compuerta · BCEWithLogitsLoss · salida1logit',fontsize=14)
    ax.text(-.4,-4.65,'Azul: entradas · Verde: Conv+BN+SiLU+SE · Naranja: MaxPool · Violeta: cabezas/fusión. Dimensiones verificadas en el modelo.',fontsize=12)
    ax.text(-.4,-4.95,'No usa pesos preentrenados. El dibujo de referencia contiene cinco convoluciones, no cuatro. Uso educativo, sin validez clínica.',fontsize=11)
    for ext in ['png','svg']:fig.savefig(root/f'arquitectura.{ext}',dpi=120,bbox_inches='tight',metadata={'Date':None} if ext=='svg' else None)
    plt.close(fig)
    text=f'''# K01 · K-Net propia · {display_count} parámetros

Cuatro bloques con reducción espacial y un refinamiento final: **cinco Conv3×3** como en la secuencia visible del dibujo humano. La cifra aproximada312.400 del dibujo no coincide con esta implementación; el recuento real es{display_count}. No se añaden capas ocultas para forzar esa cifra.

Entrada externa PRE/EARLY/LATE `[B,3,256,256]`; dentro del modelo PRE/EARLY−PRE/LATE−EARLY y BN3. No se normalizan PNG por separado ni se altera su ventana. BN es una capa de la red, no un aumento de intensidad.

Cada ConvSE: Conv3×3/padding1/stride1/sinbias → BatchNorm → SiLU → SE. SE usa GAP y Linear C→max(4,C/8)→C con SiLU interna/sigmoid; multiplica los canales de entrada. No es segmentación ni explicación causal. Canales16,32,64,128,128. MaxPool2×2/stride2 después de las primeras4convs. Resoluciones256→128→64→32→16, refinamiento16.

GAP+GMP produce256 características; dropout0,3 → Linear256→64 → GELU. Clínica `[age,menopause,age_missing,menopause_missing]`: Linear4→16 → LayerNorm16 → GELU. Estadísticas clínicas ajustadas solo con pacientes FIT. Concatenación80→Linear1logit; sigmoid solo en evaluación/inferencia. HR/HER2 no se leen como entradas.

RF de la ruta convolucional local78píxeles; SE usa promedios espaciales globales desde el primer bloque, por lo que la dependencia total ya es global. GAP/GMP también globales. SE, doble pooling y nueva cabeza cambian varias piezas frente D36: **no es una ablación exclusivamente de activación**. SiLU/GELU no añaden parámetros, ni garantizan mejor AUC. GAP+GMP puede amplificar ruido; medir con validación.

Inicialización: Conv Kaiming/fan_out/gananciaReLU (aproximación explícita para SiLU); Linear Xavier uniform/bias0; BN/LN gamma1 beta0. Todo desde cero. Ver parámetros, formas y geometría de cada capa en [capas.csv](capas.csv), [dimensiones comprobadas](dimensiones_verificadas.json), [PNG](arquitectura.png) y [SVG](arquitectura.svg).

Agregación principal: media de probabilidades de todos los cortes. Top5 por realce EARLY−PRE se muestra aparte; no elige checkpoint porque no mejoró en la tanda anterior. Diferencia explícita frente la flechaTop-K del dibujo. Se calculan ambas, sin buscar K ni umbral en test.

Entrenamiento manual: [Ejecutar y ver curvas](../K01_knet_se_dualpool/EJECUTAR.ipynb). BCE normal/ponderada N0/N1 de cortes FIT; AdamW LR0,001/WD0,001, Plateau AUCpaciente3/factor0,5, máximo60épocas, **paciencia15** sin mejora mayor1e-6. Sin geometría por defecto para no añadir un cambio de receta; nunca intensidad/ImageNet. Paciencia15 de earlystop distinta de paciencia3 del scheduler. Cada pérdida reinicia pesos/seed42. Mejor época por val; media de épocas descriptiva, no media5folds ni test independiente.

Estado: **preparado, no entrenado**. Solo formas/parametría con tensores sintéticos en eval. Smokes y científicos los ejecuta el estudiante. Uso educativo, sin validez clínica.
'''
    (root/'README.md').write_text(text)
    print('Documentado K01',count)
    return root


if __name__=='__main__':document()
