"""Detalle de las ramas obtenido de módulos propios e inputs sintéticos reales."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import torch

from src.parallel_context import ParallelContextConv


def draw_parallel_context_branches(model, directory: Path) -> None:
    units=[m for m in model.modules() if isinstance(m,ParallelContextConv)]
    if not units:
        raise ValueError('No hay unidades paralelas en el modelo')
    shapes={}
    def record(module,inputs):
        shapes[module]=tuple(inputs[0].shape[1:])
    handles=[m.register_forward_pre_hook(record) for m in units]
    original_modes={module:module.training for module in model.modules()}
    try:
        model.eval()
        with torch.inference_mode():model(torch.zeros(1,3,256,256))
    finally:
        for handle in handles:handle.remove()
        for module,training in original_modes.items():
            module.training=training
    fig,ax=plt.subplots(figsize=(19,3*len(units)+2))
    ax.set(xlim=(0,19),ylim=(-1,3*len(units)+1.5));ax.axis('off')
    navy,muted='#17283f','#61748c'
    def box(x,y,label,color):
        ax.add_patch(FancyBboxPatch((x-1.4,y-.47),2.8,.94,
            boxstyle='round,pad=.12',facecolor=color,edgecolor='#92baa9',linewidth=1.5))
        ax.text(x,y,label,ha='center',va='center',fontsize=11,color=navy)
    def arrow(a,b):
        ax.annotate('',xy=b,xytext=a,arrowprops=dict(arrowstyle='->',lw=1.7,color=muted))
    for index,unit in enumerate(units):
        cin,h,w=shapes[unit];cout=unit.out_channels;half=cout//2
        y=3*(len(units)-index)-1
        count=lambda m:sum(p.numel() for p in m.parameters())
        box(1.8,y,f'Bloque{index+1} · MISMA entrada\n{cin} × {h} × {w}','#edf3fc')
        for branch,by,name in ((unit.local,y+.65,'Local'),(unit.context,y-.65,'Contexto')):
            count_label=f'{count(branch):,}'.replace(',','.')
            label=(f'{name} · Conv{branch.kernel_size[0]}×{branch.kernel_size[0]} · d={branch.dilation[0]}\n'
                   f'padding{branch.padding[0]}, stride{branch.stride[0]} · {half} × {h}²\n'
                   f'{count_label} parámetros')
            box(6.4,by,label,'#e3f3ed')
            arrow((3.35,y),(4.85,by));arrow((7.95,by),(9.85,y))
        box(11.4,y,f'Concatenar canales\n{cout} × {h} × {w}\n0 parámetros','#eee7f4')
        box(16.4,y,f'BN → ReLU\n{cout} × {h} × {w}\n→ pool → segundaConv','#fff0df')
        arrow((12.95,y),(14.85,y))
    ax.text(.3,3*len(units)+.8,'Contexto local y amplio en paralelo · no son convoluciones consecutivas',fontsize=21,color=navy)
    ax.text(.3,-.5,'Formas y parámetros de módulos reales; entrada sintética, sin imágenes de pacientes. Uso educativo, no clínico.',fontsize=12,color=muted)
    directory.mkdir(parents=True,exist_ok=True)
    fig.savefig(directory/'ramas.png',dpi=120,bbox_inches='tight',facecolor='white')
    fig.savefig(directory/'ramas.svg',bbox_inches='tight',facecolor='white',metadata={'Date':None})
    plt.close(fig)
