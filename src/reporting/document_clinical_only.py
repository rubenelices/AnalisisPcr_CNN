"""Ficha del control D08 desde capas reales, sin datos ni entrenamiento."""
from pathlib import Path

import torch
from torch import nn

from src.models.clinical_only import ClinicalOnlyPCRNet


def document(basic=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    model=ClinicalOnlyPCRNet().eval()
    rows=[]; handles=[]
    def hook(name):
        def record(layer,inputs,output):
            rows.append((name,type(layer).__name__,tuple(inputs[0].shape),tuple(output.shape),
                         sum(p.numel() for p in layer.parameters(recurse=False))))
        return record
    for name,layer in model.named_modules():
        if isinstance(layer,(nn.Linear,nn.ReLU)): handles.append(layer.register_forward_hook(hook(name)))
    with torch.no_grad(): result=model(None,torch.zeros(1,8))
    for handle in handles: handle.remove()
    count=sum(p.numel() for p in model.parameters())
    assert count==sum(row[4] for row in rows)==161 and result.shape==(1,)
    name = 'B08_solo_edad_menopausia' if basic else 'D08_clinica_sola'
    title = 'B08 · Solo edad y menopausia' if basic else 'D08 · Control con clínica sola'
    inputs = ('Edad, menopausia y sus ausencias; cuatro slots inactivos a cero' if basic
              else '4 variables + 4 indicadores de ausencia')
    description = ('edad, menopausia y sus dos indicadores de ausencia; ocho slots serializados, cuatro inactivos a cero'
                   if basic else 'edad, HR, HER2, menopausia y cuatro indicadores de ausencia')
    protocol = f'../{name}/README.md' if basic else '../D08_clinica_sola/README.md'
    folder=Path('experiments')/name;folder.mkdir(parents=True,exist_ok=True)
    fig,ax=plt.subplots(figsize=(12,5));ax.set(xlim=(0,12),ylim=(0,5));ax.axis('off')
    ax.text(.3,4.5,title,fontsize=23,fontfamily='serif',color='#17263d')
    ax.text(.3,4.05,'Sin imágenes · misma rama clínica de E58 · inicialización desde cero',fontsize=12,color='#657080')
    for x,h,color,title,bottom in [(1,1.1,'#80a6db','Entrada','8 valores'),
                                  (4,1.8,'#68bda0','Lineal 8 → 16 + ReLU','144 parámetros'),
                                  (7.5,.8,'#a481c1','Lineal 16 → 1','17 parámetros')]:
        y=1.4;w=.7;d=.22
        ax.add_patch(Polygon([(x,y),(x+w,y),(x+w,y+h),(x,y+h)],facecolor=color,alpha=.3,edgecolor=color,lw=2))
        ax.add_patch(Polygon([(x,y+h),(x+d,y+h+d),(x+w+d,y+h+d),(x+w,y+h)],facecolor=color,alpha=.2,edgecolor=color,lw=2))
        ax.add_patch(Polygon([(x+w,y),(x+w+d,y+d),(x+w+d,y+h+d),(x+w,y+h)],facecolor=color,alpha=.4,edgecolor=color,lw=2))
        ax.text(x+.35,3.55,title,ha='center',fontsize=11)
        ax.text(x+.35,1.05,bottom,ha='center',fontsize=11)
    for start,end in [(2,3.7),(5,7.1),(8.5,9.0)]:
        ax.annotate('',xy=(end,2),xytext=(start,2),arrowprops=dict(arrowstyle='->',color='#17263d'))
    ax.text(10.2,2,'Logit\nSigmoide al evaluar\nProbabilidad pCR',ha='center',fontsize=11)
    ax.text(.3,.5,f'{inputs} · 161 parámetros',fontsize=11)
    ax.text(.3,.1,'Control de diagnóstico: no sustituye la CNN del trabajo · uso educativo, sin validez clínica',fontsize=10,color='#657080')
    for ext in ('png','svg'):fig.savefig(folder/f'arquitectura.{ext}',dpi=160,bbox_inches='tight',facecolor='white')
    plt.close(fig)
    table='\n'.join(f'| {n} | {kind} | `{inp}` | `{out}` | {p} |' for n,kind,inp,out,p in rows)
    (folder/'README.md').write_text(f'''# {title}

![Arquitectura](arquitectura.png)

Control propio desde cero: {description}.
Misma rama clínica que E58; sin CNN, PNG, pesos previos ni transfer learning.
Total real comprobado: **{count} parámetros**. Capas densas: sin kernel/padding/stride
espacial, sin pooling ni campo receptivo de imagen. ReLU aporta no linealidad.

| Capa | Tipo | Entrada | Salida | Parámetros |
|---|---|---|---|---:|
{table}

La salida lineal produce un logit para BCEWithLogitsLoss; sigmoide solo al evaluar.
No hay dropout de imagen: la rama de imagen no existe. La inicialización es
Xavier con bias cero como las capas clínicas E58, pero la forma y trayectoria
RNG cambian; no se afirma igualdad de pesos iniciales con E58.

Consulta [protocolo y ejecución]({protocol}).
Regenerar con `python -m src.reporting.document_architectures --only {name}`. Uso educativo, no clínico.
''',encoding='utf-8')
    print(f'{name} documentado: {count} parámetros verificados')


if __name__=='__main__': document()
