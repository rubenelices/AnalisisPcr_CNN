"""Ficha E58 derivada del modelo real; no carga pacientes ni checkpoints."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import torch
from torch import nn
from src.models.multimodal_model import MultimodalPCRNet
from src.training.train_multimodal import load_spec


def draw_d36_layers(model, directory):
    """Mismo lenguaje visual del catálogo, trazando el extractor real completo."""
    import math
    from matplotlib.patches import FancyBboxPatch
    from src.reporting.document_architectures import trace, visible_diagram_steps, dimensions
    steps=visible_diagram_steps(trace(model.image_encoder))
    assert sum(isinstance(layer,nn.Conv2d) for layer in model.modules())==8
    assert sum(p.numel() for p in model.parameters())==294289
    assert model.clinical_branch[0].in_features==8 and model.clinical_branch[0].out_features==16
    assert model.fusion.in_features==144 and model.fusion.out_features==1
    n=len(steps);join=n+1;head=n+2;output=n+3
    fig,ax=plt.subplots(figsize=(max(24,(n+5)*1.5),11))
    fig.patch.set_facecolor('white');ax.set(xlim=(-.7,n+5),ylim=(-5.3,5));ax.axis('off')
    navy,muted='#17283f','#61748c'
    palette={'input':('#e7efff','#83ade9'),'conv':('#e3f3ed','#75bda7'),
             'pool':('#fff0df','#e8a358'),'head':('#efe7f6','#a188c1')}
    ax.text(-.5,4.55,'D36 · Imágenes + edad y menopausia',fontsize=27,color=navy,fontfamily='DejaVu Serif')
    ax.text(-.5,4.05,'CNN propia desde cero · ocho convoluciones · fusión con datos de la misma paciente',fontsize=15,color=muted)
    def volume(x,y,shape,kind,label,dimension_y,label_y):
        channels=shape[0];side=shape[-1] if len(shape)==3 else 1
        h=(.65+2.05*math.log2(side+1)/math.log2(257) if len(shape)==3 else .55+.85*min(channels/128,1))
        w=.18+.40*min(math.log2(channels+1)/8,1);left=x-w/2;bottom=y-h/2;dx,dy=.17,.23
        fill,edge=palette[kind]
        faces=[[(left,bottom),(left+w,bottom),(left+w,bottom+h),(left,bottom+h)],
               [(left,bottom+h),(left+dx,bottom+h+dy),(left+w+dx,bottom+h+dy),(left+w,bottom+h)],
               [(left+w,bottom),(left+w+dx,bottom+dy),(left+w+dx,bottom+h+dy),(left+w,bottom+h)]]
        for face in faces:ax.add_patch(Polygon(face,closed=True,facecolor=fill,edgecolor=edge,linewidth=1.8))
        ax.text(x+.05,label_y,label,ha='center',va='center',fontsize=11,color=navy,linespacing=1.5)
        ax.text(x+.05,dimension_y,dimensions(shape),ha='center',fontsize=12,color=muted)
        return left,left+w+dx
    def arrow(a,b):ax.annotate('',xy=b,xytext=a,arrowprops=dict(arrowstyle='->',color='#9baabd',lw=1.2))
    ends=[]
    for i,step in enumerate(steps):
        ends.append(volume(i,1.5,step['shape'],step['kind'],step['label'],-.3,3.55))
        if i:arrow((ends[i-1][1]+.03,1.48),(ends[i][0]-.05,1.48))
    jleft,jright=volume(join,1.5,(144,),'head','Concatenar\n128 + 16',-.3,3.55)
    hleft,hright=volume(head,1.5,(1,),'head','Lineal\n144 → 1 logit',-.3,3.55)
    arrow((ends[-1][1]+.03,1.48),(jleft-.05,1.48));arrow((jright+.03,1.48),(hleft-.05,1.48))
    arrow((hright+.03,1.48),(output-.1,1.48))
    ax.text(output,1.85,'Sigmoide',fontsize=13,color=navy)
    ax.text(output,1.2,'Probabilidad\ndel corte',fontsize=12,color=muted,linespacing=1.5)
    ax.text(output,.15,'↓ Media de probabilidades\nde todos los cortes\n→ pCR por paciente',fontsize=11,color=muted,linespacing=1.5)
    cy=-2.5;clinical_input=n-5;clinical_head=n-2.5
    aleft,aright=volume(clinical_input,cy,(8,),'input','Entrada clínica',cy-.85,cy+1.1)
    bleft,bright=volume(clinical_head,cy,(16,),'head','Lineal 8 → 16\n+ ReLU',cy-.85,cy+1.1)
    arrow((aright+.03,cy),(bleft-.08,cy))
    elbow=join-.65
    ax.plot([bright+.03,elbow,elbow],[cy,cy,1.05],color='#9baabd',lw=1.2)
    arrow((elbow,1.05),(jleft-.05,1.05))
    ax.text(-.45,-1.65,'Edad y menopausia',fontsize=16,color=navy)
    ax.text(-.45,-2.22,'Dos valores + dos indicadores de ausencia.\nCuatro posiciones constantes a cero por compatibilidad.',fontsize=12,color=muted,linespacing=1.7)
    ax.text(-.45,-3.22,'Imputación y escala ajustadas solo con pacientes train.\nHR y HER2 no son entradas de D36.',fontsize=12,color=muted,linespacing=1.7)
    ax.add_patch(FancyBboxPatch((-.5,-5.05),n+5.05,1.1,boxstyle='round,pad=.1',facecolor='#edf3fc',edgecolor='#aac5ed',lw=1.5))
    ax.text(-.2,-4.25,'294.289 parámetros · Conv 3×3 / padding 1 / stride 1 · MaxPool 2×2 / stride 2 · Dropout 0,3',fontsize=14,color=navy)
    ax.text(-.2,-4.58,'Azul: entradas · Verde: convolución + BN + ReLU · Naranja: reducción espacial · Violeta: cabeza y fusión',fontsize=12,color=navy)
    ax.text(-.2,-4.9,'Volúmenes esquemáticos; dimensiones verificadas en el modelo real. Uso educativo, sin validez clínica.',fontsize=11,color=muted)
    for ext in ('png','svg'):
        fig.savefig(directory/f'arquitectura.{ext}',dpi=120,bbox_inches='tight',metadata={'Date':None} if ext=='svg' else None)
    plt.close(fig)


def document(config_path='configs/multimodal/E58_clinical_fusion_mac.json', directory=None, basic=False, diagram_only=False):
    torch.set_num_threads(2)
    spec,cfg=load_spec(config_path)
    identifier=cfg.experiment_id if basic else 'E58'
    activation='SiLU' if cfg.model.activation=='silu' else 'ReLU'
    simple=cfg.model.convolutions_per_block==1
    raw=cfg.model.input_representation=='raw'
    expected_count=97969 if simple else 294289
    receptive_field=46 if simple else 106
    block_label=(f'Conv–BN–{activation}\nPool' if simple else
                 f'Conv–BN–{activation}\nPool\nConv–BN–{activation}')
    if basic and spec.get('clinical_input_columns')!=['age','menopause']:
        raise ValueError('Ficha básica exige edad y menopausia')
    model=MultimodalPCRNet(cfg.model).eval()
    rows=[]; handles=[]
    def hook(name):
        def record(layer,inputs,output):
            parameters=sum(p.numel() for p in layer.parameters(recurse=False))
            geometry=(f'k={layer.kernel_size}, p={layer.padding}, s={layer.stride}'
                      if isinstance(layer,nn.Conv2d) else
                      f'k={layer.kernel_size}, s={layer.stride}' if isinstance(layer,nn.MaxPool2d) else '—')
            rows.append((name,type(layer).__name__,tuple(inputs[0].shape),tuple(output.shape),parameters,geometry))
        return record
    for name,layer in model.named_modules():
        if not list(layer.children()) and not isinstance(layer,nn.Identity):
            handles.append(layer.register_forward_hook(hook(name)))
    with torch.inference_mode():
        result=model(torch.zeros(1,3,256,256),torch.zeros(1,8))
    for h in handles: h.remove()
    count=sum(p.numel() for p in model.parameters())
    assert result.shape==(1,) and sum(row[4] for row in rows)==count==expected_count
    blocks=[]
    for i in range(4):
        block=model.image_encoder.blocks[i]
        params=sum(p.numel() for p in block.parameters())
        last=[r for r in rows if r[0].startswith(f'image_encoder.blocks.{i}.')][-1]
        blocks.append((last[3],params))
    directory=Path(directory or 'experiments/E58_clinical_fusion_mac'); directory.mkdir(parents=True,exist_ok=True)
    if identifier == 'D36_edad_menopausia':
        draw_d36_layers(model, directory)
    else:
        fig,ax=plt.subplots(figsize=(18,8)); ax.set_xlim(0,18); ax.set_ylim(0,8); ax.axis('off')
        fig.patch.set_facecolor('white')
        ax.text(.4,7.4,f'{identifier} · Imagen + edad y menopausia' if basic else 'E58 · Imagen + información clínica',fontsize=25,fontfamily='serif',color='#17263d')
        ax.text(.4,6.85,'CNN propia desde cero · solo dos variables · fusión de 144 características' if basic else 'CNN E19 propia desde cero · cuatro variables prefijadas · fusión de 144 características',fontsize=13,color='#657080')
        def volume(x,y,w,h,color,label,bottom,label_y=5.85):
            d=.22
            ax.add_patch(Polygon([(x,y),(x+w,y),(x+w,y+h),(x,y+h)],facecolor=color,alpha=.24,edgecolor=color,lw=2))
            ax.add_patch(Polygon([(x+w,y),(x+w+d,y+d),(x+w+d,y+h+d),(x+w,y+h)],facecolor=color,alpha=.4,edgecolor=color,lw=2))
            ax.add_patch(Polygon([(x,y+h),(x+d,y+h+d),(x+w+d,y+h+d),(x+w,y+h)],facecolor=color,alpha=.18,edgecolor=color,lw=2))
            ax.text(x+w/2,label_y,label,ha='center',va='bottom',fontsize=10)
            ax.text(x+w/2,y-.25,bottom,ha='center',va='top',fontsize=10,color='#657080')
        volume(.7,3.2,.3,2.3,'#80a6db','Entrada\n3 fases','3 × 256²')
        ax.text(.4,2.05,'Dentro: PRE, EARLY, LATE' if raw else 'Dentro: PRE, EARLY−PRE, LATE−EARLY',fontsize=10)
        for i,(shape,params) in enumerate(blocks):
            c,h,w=shape[1:]; x=2.2+i*2
            volume(x,3.2,.65+i*.15,2.1-i*.3,'#68bda0',f'Bloque {i+1}\n{block_label}',f'{c} × {h}²\n{params:,} parámetros')
            ax.annotate('',xy=(x,4.2),xytext=(x-.65,4.2),arrowprops=dict(arrowstyle='->',color='#17263d'))
        volume(10.3,3.4,.6,1.1,'#a481c1','GAP + dropout 0,3','128 imagen')
        volume(10.3,1.,.6,1.1,'#ecaf65','Clínica: 8 → 16\n+ ReLU','16 clínica · 144 parámetros',label_y=2.35)
        ax.text(.7,1.55,'Edad · menopausia' if basic else 'Edad · HR · HER2 · menopausia',fontsize=14)
        ax.text(.7,1,'2 valores + 2 indicadores; 4 slots constantes a cero\nImputación y escala ajustadas solo con pacientes train' if basic else '4 valores + 4 indicadores de ausencia\nImputación y escala ajustadas solo con pacientes train',fontsize=11,color='#657080')
        ax.annotate('',xy=(10.2,1.3),xytext=(7.7,1.3),arrowprops=dict(arrowstyle='->',color='#17263d'))
        for y in (4,1.4):
            props=dict(arrowstyle='->',color='#17263d')
            if basic:props['connectionstyle']=f'angle,angleA={-90 if y==4 else 90},angleB=0'
            ax.annotate('',xy=(12.3,2.9),xytext=(11.6 if basic else 11.2,y),arrowprops=props)
        ax.text(12.5,3.45 if basic else 2.9,'Concatenar\n128 + 16 = 144',ha='center',fontsize=12)
        ax.annotate('',xy=(14,2.9),xytext=(13.3,2.9),arrowprops=dict(arrowstyle='->'))
        ax.text(14.7,3.45 if basic else 2.9,'Lineal 144 → 1\n145 parámetros',ha='center',fontsize=12)
        ax.annotate('',xy=(16.2,2.9),xytext=(15.6,2.9),arrowprops=dict(arrowstyle='->'))
        ax.text(16.6,3.55 if basic else 2.9,'Logit\nSigmoide\nProbabilidad',ha='center',fontsize=11)
        ax.text(.5,.1,f'{count:,} parámetros · Conv 3×3/padding 1/stride 1 · Pool 2×2/stride 2 · RF local {receptive_field} · media de probabilidades por paciente',fontsize=11)
        fig.savefig(directory/'arquitectura.png',dpi=160,bbox_inches='tight')
        fig.savefig(directory/'arquitectura.svg',bbox_inches='tight'); plt.close(fig)
    if diagram_only:
        return rows
    table='\n'.join(f'| {n} | {kind} | `{inp}` | `{out}` | {p} | {geo} |' for n,kind,inp,out,p,geo in rows)
    text=f'''# E58 · CNN E19 + clínica

![Arquitectura](arquitectura.png)

Hipótesis: edad, HR, HER2 y menopausia aportan información complementaria a DCE-MRI.
No se garantiza mejora ni 0,7. E19 es la referencia provisional, no un ganador estable.
Toda la red se inicializa desde cero; no se cargan pesos E19 ni pesos externos.

## Ramas y fusión

Extractor E19: 8 convoluciones 3×3, 8 BN/ReLU, 4 MaxPool entre convoluciones;
16/32/64/128 canales; 256→128→64→32→16; campo receptivo local 106.
GAP produce 128 características y dropout 0,3. Clínica: 8→16/ReLU.
Concatenación 144→1: fusión **aditiva en el logit**, no atención ni interacción
explícita entre modalidades. Total comprobado: **{count} parámetros**.

Edad: mediana para ausentes y estandarización; binarias: moda para ausentes.
Cuatro indicadores preservan la ausencia. Estadísticas con una fila por paciente
del train interno únicamente, guardadas con el checkpoint para reutilizar en inferencia.
IDs solo para unir registros; pCR, cohorte, split y subtipos redundantes no son entradas.
Los indicadores de ausencia podrían reflejar cohortes, no interpretarlos causalmente.

## Protocolo y resultados

Fold 0/semilla 42; 878 pacientes train y 219 validación, sin test.
Batch 16, AdamW LR/WD 0,001, BCE por corte, máximo 50 épocas,
Plateau paciencia 3/factor 0,5 y early stopping paciencia 10.
Métrica principal: ROC-AUC **por paciente**, media de probabilidades de cortes.
No aumenta pacientes ni aplica rotaciones; cambia únicamente la modalidad clínica
respecto a la referencia, con nueva cabeza y trayectoria aleatoria propia.
E58 completado: AUC0,747076613/AP0,544753123, best2 de12.
A0,5 sensibilidad0,15625 y F10,23529: AUC alto no garantiza decisiones satisfactorias.
[Nueva etapa clínica y figuras](../README.md).
Smoke técnico no demuestra calidad; no reentrenar E58 para organizar archivos.

```sh
python -u -m src.training.train_multimodal --config configs/multimodal/E58_clinical_fusion_mac.json --device mps
```

Resultados locales: `reports/multimodal/E58_clinical_fusion_mac/fold_0_seed_42/`:
summary.json, history.csv, training_curves.png, evaluation_patient.png,
predictions_cut/patient.csv, auditoría clínica y preprocesador. Pesos en checkpoints.
No sobrescribe una ejecución existente. La web actual no acepta este checkpoint:
requiere rama clínica y mismo preprocesador. Confirmar que el profesor facilitará
estas variables en la demo y que son basales, disponibles antes del tratamiento.
Para atribuir el aporte de imagen posteriormente comparar con clínica sola.

## Capas reales (batch ficticio 1)

| Capa | Tipo | Entrada | Salida | Parámetros | Kernel/padding/stride |
|---|---|---|---|---:|---|
{table}

Diagrama y tabla regenerables con `python -m src.reporting.document_multimodal`.
Uso educativo y de investigación, sin validez clínica.
'''
    if basic:
        text=f'''# {identifier} · Imágenes, edad y menopausia

![Arquitectura](arquitectura.png)

Extractor propio, no ganador definitivo ni pesos reutilizados.
Solo edad y menopausia como variables: no lee HR/HER2 del CSV. Cuatro entradas
informativas: dos valores y dos indicadores de ausencia. Se conservan ocho slots
por compatibilidad y comparación: posiciones1/2/5/6 son ceros constantes.
Esto NO añade cuatro variables clínicas ocultas. Total real: {count} parámetros;
algunos pesos de entrada multiplican siempre cero, no aportan señal.

Cuatro bloques {'Conv3×3/BN/'+activation+'→MaxPool2×2' if simple else 'Conv3×3/BN/'+activation+'→MaxPool2×2→Conv3×3/BN/'+activation};
canales16/32/64/128, resoluciones256→128→64→32→16, campo local{receptive_field}.
Entrada interna: {'PRE, EARLY, LATE originales' if raw else 'PRE, EARLY−PRE, LATE−EARLY'}.
GAP128/dropout0,3; rama clínica8→16/ReLU (144params), fusión144→1 (145params).
Fusión aditiva en logit, no atención ni interacción explícita imagen/clínica.
Edad mediana/escala y menopausia moda ajustadas solo en pacientes train.
Ausencia no equivale a cero observado; sus indicadores permanecen activos.
Los ausentes pueden reflejar cohortes; investigar, no atribuir causalidad.

Inicialización desde cero;50épocas/paciencia10, batch16, LR/WD0,001,
BCE normal por corte. Evaluación principal media de probabilidades por paciente.
Umbral0,5 provisional para comparar; no umbral final ni optimizado en test.
Preparado sin ningún entrenamiento. Comparar controles de imágenes y clínica,
otras semillas y folds antes de atribuir mejora o adoptar modelo.
El profesorado permite estas variables; falta verificar su entrega
en la defensa y su registro basal. La web todavía no integra este checkpoint.

```sh
python -u -m src.training.train_multimodal --config {config_path} --device mps
```

## Capas reales (batch ficticio1)

| Capa | Tipo | Entrada | Salida | Parámetros | Kernel/padding/stride |
|---|---|---|---|---:|---|
{table}

ActivaciónCNN:{activation}, sin parámetros adicionales. Rama clínica:ReLU.
Regenerar: `python -m src.reporting.document_architectures --only {identifier}`.
Uso educativo y de investigación, sin validez clínica.
'''
    (directory/'README.md').write_text(text,encoding='utf-8')
    print(f'{identifier} documentado: {count} parámetros; formas comprobadas contra modelo real')


if __name__=='__main__': document()
