"""Ficha de la CNN 97.809: medidas obtenidas del modelo real ficticio."""
from pathlib import Path
import torch
from torch import nn
from src.simple_cnn_comparison import make_model


def document():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    torch.set_num_threads(2)
    model = make_model('raw').eval()
    rows, handles = [], []
    def hook(name):
        def capture(layer, inputs, output):
            geometry = (f'kernel={layer.kernel_size}, padding={layer.padding}, stride={layer.stride}'
                        if isinstance(layer, (nn.Conv2d, nn.MaxPool2d)) else '—')
            rows.append((name, type(layer).__name__, tuple(inputs[0].shape), tuple(output.shape),
                         sum(p.numel() for p in layer.parameters(recurse=False)), geometry))
        return capture
    for name, layer in model.named_modules():
        if not list(layer.children()):
            handles.append(layer.register_forward_hook(hook(name)))
    with torch.no_grad():
        output = model(torch.zeros(2, 3, 256, 256))
    for handle in handles:
        handle.remove()
    assert output.shape == (2,) and sum(r[4] for r in rows) == 97809
    trace = model.trace_shapes(2)
    assert [x.shape for x in trace[1:5]] == [(2,16,128,128), (2,32,64,64), (2,64,32,32), (2,128,16,16)]
    folder = Path('experimentos/C16_cnn_97809')
    folder.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.set(xlim=(0,16), ylim=(0,6)); ax.axis('off')
    ax.text(.3,5.5,'CNN propia de cuatro bloques · 97.809 parámetros',fontsize=23,color='#22384a')
    ax.text(.3,5.0,'Una Conv3×3 → BatchNorm → ReLU → MaxPool2×2 por bloque · desde cero',fontsize=12)
    labels = [('Entrada','3 × 256 × 256',0)]
    labels += [(f'Bloque {i+1}',f'{c} × {256//2**(i+1)} × {256//2**(i+1)}',sum(p.numel() for p in b.parameters()))
               for i,(c,b) in enumerate(zip((16,32,64,128),model.blocks))]
    labels += [('GAP + Flatten','128 valores',0), ('Dropout 0,3\nLinear128→1','1 logit',129)]
    for i,(title,shape,params) in enumerate(labels):
        x=.5+i*2.2; y=2.0; w=.65+i*.055; h=2.1-i*.24; d=.15
        color='#4a7f9c' if i==0 else '#388d74' if i<5 else '#8a71a5'
        ax.add_patch(Polygon([(x,y),(x+w,y),(x+w,y+h),(x,y+h)],facecolor=color,alpha=.35))
        ax.add_patch(Polygon([(x,y+h),(x+d,y+h+d),(x+w+d,y+h+d),(x+w,y+h)],facecolor=color,alpha=.6))
        ax.add_patch(Polygon([(x+w,y),(x+w+d,y+d),(x+w+d,y+h+d),(x+w,y+h)],facecolor=color,alpha=.8))
        ax.text(x+w/2,4.5,title,ha='center',fontsize=11)
        ax.text(x+w/2,1.65,shape,ha='center',fontsize=10)
        ax.text(x+w/2,1.3,f'{params:,} parámetros',ha='center',fontsize=9)
        if i<6:
            ax.annotate('',xy=(x+2.05,2.55),xytext=(x+w+.2,2.55),arrowprops=dict(arrowstyle='->',color='#22384a'))
    ax.text(.3,.7,'Entrada externa: PRE / EARLY / LATE · salida: sigmoid por corte → media por paciente',fontsize=12)
    ax.text(.3,.25,'Variante de control: PRE / EARLY−PRE / LATE−EARLY dentro del modelo, sin parámetros adicionales',fontsize=11)
    for ext in ('png','svg'):
        fig.savefig(folder/f'arquitectura.{ext}',dpi=160,bbox_inches='tight',facecolor='white')
    plt.close(fig)
    table='\n'.join(f'| {n} | {kind} | `{a}` | `{b}` | {p} | {g} |' for n,kind,a,b,p,g in rows)
    (folder/'README.md').write_text(f'''# C16 · CNN propia de 97.809 parámetros

![Volúmenes reales](arquitectura.png)

Cuatro bloques Conv3×3 (padding1,stride1,sin bias) → BatchNorm → ReLU →
MaxPool2×2(stride2,padding0), canales16/32/64/128. GAP128 → Dropout0,3 →
Linear128→1. **97.809 parámetros**, verificados contra el modelo real: pesos de
Conv97.200, BatchNorm480 y Linear129. Sin bias en Conv porque sigue BatchNorm.
Las formas incluyen batch2 ficticio; no se cargan imágenes para esta ficha.

La entrada externa siempre es PRE/EARLY/LATE, float32 PNG/255. Perfil raw sin
restas; control pre_differences transforma a PRE,EARLY−PRE,LATE−EARLY dentro del
modelo con cero parámetros, de forma idéntica al entrenar y predecir. Ninguna
normalización RGB ni pesos preentrenados. Inicialización propia existente:
Kaiming fan_out/ReLU para Conv, BN peso1/bias0, Xavier para Linear y bias0.
No conocemos inicialización, optimizador ni toda la receta del compañero:
se reproduce la arquitectura descrita con nuestro régimen prefijado.

Campo receptivo local nominal tras bloques:4,10,22,46 píxeles; saltos2,4,8,16.
Cada bloque reduce a la mitad H/W y aumenta canales. BN en train y GAP añaden
dependencia global. Una Conv por bloque reduce profundidad y cabeza lineal
elimina MLP intermedio. Es una hipótesis de mejor generalización, no una mejora
demostrada. El paquete arquitectónico cambia también la posición del pooling
respecto a E57; no atribuir una diferencia solo al número de convoluciones.

Experimento: dos representaciones prefijadas, cinco folds por paciente, BCE
normal por corte, selección de checkpoint y umbral exclusivamente interna.
Comparación con E57 referencia16 en exactamente las mismas pacientes.
El control con diferencias aísla el paquete arquitectónico; raw frente E57
cambia también representación. No se elige un perfil según métricas exteriores.
Configuraciones y resultados: ../../experimentos_equilibrados/C16_cnn_97809/.

| Capa | Tipo | Entrada | Salida | Parámetros | Geometría |
|---|---|---|---|---:|---|
{table}

[SVG](arquitectura.svg) · Regenerar: `python -m src.document_architectures --only C16_cnn_97809`.
Cargador reproducible: `src.simple_cnn_comparison.load_checkpoint`.

Uso educativo y de investigación, sin validez clínica.
''')
    return rows


if __name__ == '__main__':
    document()
