"""Ficha D11 desde la CNN real y la cabeza 144→32→1."""
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
    spec,cfg=load_spec('experiments/D11_fusion_no_lineal/config.json')
    model=MultimodalPCRNet(cfg.model,spec['fusion_hidden']).eval()
    rows=[];handles=[]
    def hook(name):
        def record(layer,inputs,output):
            geo=(f'k={layer.kernel_size},padding={layer.padding},stride={layer.stride}' if isinstance(layer,nn.Conv2d)
                 else f'k={layer.kernel_size},stride={layer.stride}' if isinstance(layer,nn.MaxPool2d) else '—')
            rows.append((name,type(layer).__name__,tuple(inputs[0].shape),tuple(output.shape),
                         sum(p.numel() for p in layer.parameters(recurse=False)),geo))
        return record
    for name,layer in model.named_modules():
        if not list(layer.children()) and not isinstance(layer,nn.Identity):handles.append(layer.register_forward_hook(hook(name)))
    with torch.no_grad():output=model(torch.zeros(1,3,256,256),torch.zeros(1,8))
    for h in handles:h.remove()
    count=sum(p.numel() for p in model.parameters())
    assert count==sum(r[4] for r in rows)==298817 and output.shape==(1,)
    folder=Path('experiments/D11_fusion_no_lineal');folder.mkdir(parents=True,exist_ok=True)
    fig,ax=plt.subplots(figsize=(18,7));ax.set(xlim=(0,18),ylim=(0,7));ax.axis('off')
    ax.text(.3,6.4,'D11 · Fusión no lineal de imagen y clínica',fontsize=24,fontfamily='serif',color='#17263d')
    ax.text(.3,5.9,'Misma CNN propia E58 · único cambio: cabeza 144 → 32 → 1',fontsize=13,color='#657080')
    def volume(x,y,w,h,color,title,bottom):
        d=.2
        for vertices,alpha in [([(x,y),(x+w,y),(x+w,y+h),(x,y+h)],.25),
            ([(x,y+h),(x+d,y+h+d),(x+w+d,y+h+d),(x+w,y+h)],.2),
            ([(x+w,y),(x+w+d,y+d),(x+w+d,y+h+d),(x+w,y+h)],.4)]:
            ax.add_patch(Polygon(vertices,facecolor=color,edgecolor=color,lw=2,alpha=alpha))
        ax.text(x+w/2,y+h+.4,title,ha='center',fontsize=10)
        ax.text(x+w/2,y-.25,bottom,ha='center',fontsize=10,va='top')
    volume(.5,2.8,.3,1.8,'#80a6db','Entrada\n3 fases','3 × 256²')
    for i in range(4):
        size=256//2**(i+1);channels=16*2**i
        volume(2+i*1.8,2.8,.65,1.7-i*.25,'#68bda0',f'Bloque {i+1}\nConv–BN–ReLU\nPool\nConv–BN–ReLU',f'{channels} × {size}²')
    volume(9.4,2.8,.5,.8,'#a481c1','GAP\nDropout0,3','128 imagen')
    volume(9.4,.7,.5,.7,'#ecaf65','Clínica 8→16\nReLU','16 clínica')
    ax.text(.5,1.05,'Edad · HR · HER2 · menopausia\n4 valores + 4 indicadores · estadísticas SOLO train',fontsize=12)
    for y in (3.3,1.1):ax.annotate('',xy=(11.2,2.3),xytext=(10.1,y),arrowprops=dict(arrowstyle='->'))
    ax.text(11.4,2.3,'Concatenar\n144',ha='center',fontsize=11)
    volume(12.7,1.8,.55,.9,'#a481c1','Lineal144→32\nReLU','32 · 4640 parámetros')
    volume(15,1.8,.4,.5,'#a481c1','Lineal32→1','1 · 33 parámetros')
    ax.text(16.6,2,'Logit\nSigmoide\nProbabilidad',ha='center',fontsize=11)
    for start,end in [(12,12.5),(13.6,14.7),(15.6,16.1)]:
        ax.annotate('',xy=(end,2.2),xytext=(start,2.2),arrowprops=dict(arrowstyle='->'))
    ax.text(.5,.1,'298.817 parámetros · Conv3×3/p1/s1 · Pool2×2/s2 · RF106 · media de probabilidades por paciente',fontsize=11)
    for extension in ('png','svg'):fig.savefig(folder/f'arquitectura.{extension}',dpi=160,bbox_inches='tight',facecolor='white')
    plt.close(fig)
    table='\n'.join(f'| {n} | {kind} | `{inp}` | `{out}` | {p} | {geo} |' for n,kind,inp,out,p,geo in rows)
    (folder/'README.md').write_text(f'''# D11 · Fusión no lineal

![Arquitectura](arquitectura.png)

CNN propia desde cero, misma E58; clínica8→16/ReLU y ambas ramas intactas.
Solo cabeza144→1 se sustituye por144→32/ReLU→1 para permitir interacciones
no lineales entre modalidades. Parámetros reales **{count}**, aumento4528.
No garantía de mejora: puede aumentar sobreajuste. No pesos externos/transfer.
Xavier/bias0 en capas clínicas y fusión; formas cambian RNG, no inicialización
idéntica garantizada entre diseños. CNN8convs/4pools intermedios16/32/64/128,
GAP128/dropout0,3,RF local106 antes de GAP; externo3×256×256,
interno PRE/diferencias firmadas. GAP agrega las posiciones del mapa completo.
Capas densas sin kernel/padding/stride espacial; ReLU introduce no linealidad.

| Capa | Tipo | Entrada | Salida | Parámetros | Geometría |
|---|---|---|---|---:|---|
{table}

[Protocolo de prueba](../D11_fusion_no_lineal/README.md).
Regenerar con `python -m src.reporting.document_fusion`. Uso educativo, no clínico.
''',encoding='utf-8')
    print('D11 documentado: 298817 parámetros, formas reales verificadas')


if __name__=='__main__':document()
