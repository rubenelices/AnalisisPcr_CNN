"""Dos CNN propias y promedio equiponderado; formas y parámetros reales."""
from dataclasses import replace
from pathlib import Path
import torch
from torch import nn
from src.campaign_20261015 import MODEL
from src.patient_feature_model import PatientFeatureNet
from src.campaign_20261015_models import CompactPatientFeatureNet
from src.campaign_20261015_ensemble import EqualCNNEnsemble


def document():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    folder=Path('experimentos/C15_ensemble_dos_semillas');folder.mkdir(exist_ok=True)
    torch.set_num_threads(2);details=[]
    for compact in (False,True):
        config=replace(MODEL,channels=(8,16,32,64)) if compact else MODEL
        cls=CompactPatientFeatureNet if compact else PatientFeatureNet
        model=EqualCNNEnsemble(cls(config),cls(config)).eval();shapes=[];hooks=[]
        for b,branch in enumerate((model.first,model.second)):
            for i,block in enumerate(branch.encoder.blocks):
                hooks.append(block.register_forward_hook(lambda layer,args,out,b=b,i=i:shapes.append((b,i,tuple(out.shape),sum(p.numel() for p in layer.parameters())))))
        with torch.no_grad():out=model(torch.zeros(2,3,256,256))
        for hook in hooks:hook.remove()
        params=sum(p.numel() for p in model.parameters());expected=151922 if compact else 596322
        assert out.shape==(2,) and params==expected and sum(isinstance(m,nn.Conv2d) for m in model.modules())==16
        for b in range(2):
            assert [shape for branch,_,shape,_ in shapes if branch==b]==[(2,c,256//2**(i+1),256//2**(i+1)) for i,c in enumerate(config.channels)]
        label='compacta' if compact else 'ancha'
        details.append(f'## Variante {label}: {params:,} parámetros\n\n| Rama | Bloque | Salida real | Parámetros |\n|---|---|---|---:|\n'+
            '\n'.join(f'| {b+1} | {i+1} | `{shape}` | {count} |' for b,i,shape,count in shapes))
        fig,ax=plt.subplots(figsize=(16,6));ax.set(xlim=(0,16),ylim=(0,6));ax.axis('off')
        ax.text(.2,5.6,f'Dos semillas · CNN {label} · {params:,} parámetros',fontsize=22,color='#213344')
        ax.text(.2,5.12,'16 Conv propias en total · promedio fijo50:50 · ningún peso externo ni semilla ganadora',fontsize=12,color='#526575')
        for branch in range(2):
            y=3.5 if branch==0 else 1.55
            ax.text(.2,y+.9,f'Semilla {42 if branch==0 else 17}',fontsize=11)
            nodes=[(.3,3,256,'Entrada',0)]+[(2.2+i*1.9,c,256//2**(i+1),f'Bloque{i+1}',sum(p.numel() for p in getattr(model,'first' if branch==0 else 'second').encoder.blocks[i].parameters())) for i,c in enumerate(config.channels)]
            nodes += [(10.,config.channels[-1],1,'GAP',0),(11.7,32,1,'Head→1',sum(p.numel() for p in model.first.head.parameters()))]
            for j,(x,c,res,title,count) in enumerate(nodes):
                w=.7;h=.85 if res>1 else .3;d=.12;color='#659aaf' if branch==0 else '#74a88f'
                for pts,alpha in [([(x,y),(x+w,y),(x+w,y+h),(x,y+h)],.3),
                    ([(x,y+h),(x+d,y+h+d),(x+w+d,y+h+d),(x+w,y+h)],.2),
                    ([(x+w,y),(x+w+d,y+d),(x+w+d,y+h+d),(x+w,y+h)],.5)]:ax.add_patch(Polygon(pts,facecolor=color,edgecolor=color,alpha=alpha))
                ax.text(x+w/2,y-.25,f'{c}×{res}²' if res>1 else f'{c}' if title=='GAP' else '32→1',ha='center',fontsize=9)
                ax.text(x+w/2,y-.57,title,ha='center',fontsize=9)
                if j<len(nodes)-1:ax.annotate('',xy=(nodes[j+1][0]-.1,y+.4),xytext=(x+w+.17,y+.4),arrowprops=dict(arrowstyle='->',color='#526575'))
            ax.annotate('',xy=(14.,2.9),xytext=(12.8,y+.3),arrowprops=dict(arrowstyle='->',color='#526575'))
        ax.text(14.6,3.15,'Sigmoides\nMedia 50:50',ha='center',fontsize=11)
        ax.text(14.6,2.35,'Media cortes\npor paciente',ha='center',fontsize=10)
        ax.text(.2,.3,'Cada bloque: Conv3×3→BN→ReLU→Pool2→Conv3×3→BN→ReLU · RF local106 por rama',fontsize=11)
        stem='arquitectura_compacta' if compact else 'arquitectura'
        for ext in ('png','svg'):fig.savefig(folder/f'{stem}.{ext}',dpi=150,bbox_inches='tight',facecolor='white')
        plt.close(fig)
    (folder/'README.md').write_text('''# C15 · Dos semillas, promedio fijo

![CNNancha](arquitectura.png)

![CNNcompacta](arquitectura_compacta.png)

Dos copias de la MISMA CNN propia, entrenadas desde cero con semillas42 y17.
Por corte, promedio fijo de dos sigmoides; por paciente, media de sus cortes.
Ambas medias conmutan. No se elige la mejor semilla ni se ajusta el peso50:50.
EqualCNNEnsemble devuelve el logit de esa media con logsigmoid/logsumexp
estables, sin clipping ni parámetros adicionales. Forma real `[B,3,256,256]`
→dos logits`[B]`→media de probabilidades`[B]`→logit`[B]`.

Se conservan PRE/EARLY/LATE de entrada y representación PRE/diferencias de
cada rama, geometría sincronizada cuando corresponda, CNN2D propia y sin pesos
externos. Por rama8Conv3×3/stride1/padding1,8BN/ReLU,4Pool2×2/stride2,
GAP y cabeza32. RF local106, salida de bloques medida con datos sintéticos.
La nueva combinación no cambia resoluciones ni RF; usa dos predicciones.

Motivo: medir variación de inicialización y si el promedio reduce su ruido.
No elimina sobreajuste ni garantiza mejora. Perfil/época/LRs provienen SOLO de
selección interna42. La repetición17 entrena exactamente ese número de épocas;
sus métricas internas se registran pero no escogen su checkpoint, ni scheduler.
Split permanece42: semilla17 cambia inicialización, barajado y aumentos.
El umbral de cada régimen se elige de nuevo SOLO en interna. Se reportan
siempre seed17 y promedio, aunque empeoren. No son ensayos independientes.

Idea semilla/ensemble mencionada antes del primer exterior; regla concreta de
épocas/LRs definida después del procedimiento base, antes de sus resultados.
Es un análisis adicional exploratorio; no confirmación externa independiente.
Ejecutar solo si el presupuesto permite cinco folds completos y comenzar cada
entrenamiento antes de10:22UTC. El orden de folds depende de tiempos, no AUC.

Manifest por fold conserva ambos checkpoints y SHA256; restaurador
`src.campaign_20261015_ensemble.load_ensemble`. Mismo preprocesamiento dentro
de cada rama al evaluar. No nueva variable clínica ni transferencia externa.

'''+ '\n\n'.join(details)+'''

Las capas individuales se comparten con [E57](../E57_patient_feature_mean_mac/README.md)
y [compacta](../C15_E57_compacta_cortes/README.md). Cada tabla de arriba cuenta
los bloques de cada copia; el resto corresponde a GAP sin parámetros y dos
cabezas32. Totales comprobados contra el módulo real, no estimados de dibujo.
Regenerar: `python -m src.document_architectures --only C15_ensemble_dos_semillas`.
Uso educativo y de investigación, sin validez clínica.
''')


if __name__=='__main__':document()
