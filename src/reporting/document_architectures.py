"""Genera fichas y diagramas desde las capas reales, sin leer el dataset.

Ejecutar: python -m src.reporting.document_architectures
Las figuras SVG y PNG son documentación propia versionable.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["svg.hashsalt"] = "breastpcr-architectures-v1"
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, FancyBboxPatch
import torch
from torch import nn

from src.models.architectures import BreastPCRNet, GlobalAverageMaxPool, trainable_parameter_count
from src.training.experiment_config import load_experiment_config
from src.models.model import MinimalCNN
from src.pipeline.phase_representation import PhaseRepresentation
from src.models.attention_pooling import GatedSpatialPool
from src.models.factorized_conv import FactorizedSpatialConv
from src.models.parallel_context import ParallelContextConv, ChannelConcatenation
from src.reporting.parallel_context_diagram import draw_parallel_context_branches
from src.pipeline.triplet_standardization import TripletStandardization
from src.models.power_mean_pool import FixedPowerMeanPool


ROOT = Path(__file__).resolve().parents[2]


def trace(model: nn.Module) -> list[dict]:
    """Captura el orden real de ejecución, también cuando el pool es intermedio."""
    steps = [{"label": "Entrada\n3 fases DCE", "shape": (3, 256, 256),
              "kind": "input", "parameters": 0, "rf": 1, "jump": 1}]
    handles = []
    original_modes = {layer: layer.training for layer in model.modules()}
    rf, jump = 1, 1
    names = {layer: name for name, layer in model.named_modules()}
    has_attention = isinstance(getattr(model, 'global_pool', None), GatedSpatialPool)
    parallel_units = [layer for layer in model.modules() if isinstance(layer, ParallelContextConv)]
    parallel_branches = {branch: unit for unit in parallel_units for branch in (unit.local, unit.context)}
    parallel_joins = {unit.concatenate: unit for unit in parallel_units}
    parallel_inputs = {}

    def remember_parallel_input(layer, inputs):
        parallel_inputs[layer] = (rf, jump)

    for unit in parallel_units:
        handles.append(unit.register_forward_pre_hook(remember_parallel_input))

    def record(layer, inputs, output):
        nonlocal rf, jump
        shape = tuple(output.shape[1:])
        parameters = sum(p.numel() for p in layer.parameters(recurse=False))
        additional = {}
        if isinstance(layer, TripletStandardization):
            if layer.mode == 'none':
                return
            label, kind = 'Normalización\nconjunta 3 fases', 'input'
            detail = ('una media/std por tripleta; todos los píxeles/fases en train/eval; '
                      f'epsilon={layer.epsilon:g}; sin fit ni pesos; dependencia global, NO RF local')
            additional['global_input_statistics'] = True
        elif isinstance(layer, PhaseRepresentation):
            if layer.mode == "raw":
                return
            label, kind = ("EARLY−PRE\nLATE−EARLY" if layer.mode == "temporal_differences"
                           else "PRE\nEARLY−PRE\nLATE−EARLY"), "input"
            detail = "restas fijas; sin clipping, pesos, kernel ni cambio espacial"
            if layer.mode == "temporal_differences":
                detail += "; 3 fases externas → 2 diferencias internas; PRE no pasa como canal directo"
        elif isinstance(layer, nn.Conv2d):
            kernel, stride, padding = layer.kernel_size[0], layer.stride[0], layer.padding[0]
            dilation = layer.dilation[0]
            if layer in parallel_branches:
                rf, jump = parallel_inputs[parallel_branches[layer]]
            rf += (kernel - 1) * dilation * jump
            jump *= stride
            norm_label = "GN" if has_gn else "BN" if has_bn else ""
            label = f"Conv {kernel}×{kernel}\n{norm_label + ' + ' if norm_label else ''}{activation_name}"
            if dilation != 1:
                label = f"Conv {kernel}×{kernel}\nd={dilation}\n{norm_label + ' + ' if norm_label else ''}{activation_name}"
            kind = "conv"
            detail = f"kernel={kernel}, padding={padding}, stride={stride}"
            if dilation != 1:
                detail += f"; dilation={dilation}; área efectiva={(kernel-1)*dilation+1}×{(kernel-1)*dilation+1}, no kernel denso mayor"
            if layer in depthwise_layers:
                label = f"DW {kernel}×{kernel}\npor canal" + (f"\nd={dilation}" if dilation != 1 else "")
                detail += f"; groups={layer.groups}; {layer.in_channels}→{layer.out_channels}; sin BN/ReLU intermedia"
            elif layer in pointwise_layers:
                label = f"PW 1×1\n{norm_label + ' + ' if norm_label else ''}{activation_name}"
                detail += f"; groups=1; {layer.in_channels}→{layer.out_channels}; mezcla canales"
            if has_attention and names[layer].startswith('global_pool.'):
                # Ramas paralelas, NO Conv-BN-ReLU secuenciales adicionales.
                kind = 'attention_component'
                branch = names[layer].rsplit('.', 1)[-1]
                label = {'value': 'Rama V · Conv1×1 antes de tanh',
                         'gate': 'Rama U · Conv1×1 antes de sigmoid',
                         'score': 'Score · Conv1×1 tras el producto'}[branch]
                detail += f"; {layer.in_channels}→{layer.out_channels}; bias={layer.bias is not None}"
                if branch == 'score':
                    steps.append(dict(label='Producto · tanh ⊙ sigmoid',
                        shape=tuple(inputs[0].shape[1:]), kind=kind, parameters=0,
                        rf=rf, jump=jump, detail='producto punto a punto de las dos ramas paralelas'))
            if layer in parallel_branches:
                unit = parallel_branches[layer]
                branch = 'local' if layer is unit.local else 'contexto'
                label = f'Rama {branch} · Conv3×3\nd={dilation}'
                detail += f'; MISMA entrada; {layer.in_channels}→{layer.out_channels}; sin BN/ReLU dentro de rama'
                additional['parallel_branch'] = True
        elif isinstance(layer, ChannelConcatenation):
            unit = parallel_joins[layer]
            entry_rf, entry_jump = parallel_inputs[unit]
            rf = max(entry_rf + 2 * branch.dilation[0] * entry_jump for branch in (unit.local, unit.context))
            jump = entry_jump
            label, kind = 'Concatenar ramas', 'parallel_join'
            detail = f'local+contexto por canales; RF máximo={rf}; BN/{activation_name} después, no entre ramas'
            additional['branch_parameters'] = sum(p.numel() for p in unit.parameters())
        elif isinstance(layer, (nn.BatchNorm2d, nn.GroupNorm)):
            label, kind = type(layer).__name__, "normalization"
            detail = (f"{layer.num_groups} grupos; estadísticas por muestra en train/eval"
                      if isinstance(layer, nn.GroupNorm) else "estadísticas por lote en train; acumuladas en eval")
        elif isinstance(layer, nn.ReLU):
            label, kind, detail = "ReLU", "activation", "max(0, x); sin pesos ni cambio de forma"
        elif isinstance(layer, nn.SiLU):
            label, kind, detail = "SiLU", "activation", "x*sigmoid(x); sin pesos ni cambio de forma; sigmoid interna NO es probabilidad pCR"
        elif isinstance(layer, GlobalAverageMaxPool):
            label, kind = "Media + máximo\nConcatenar", "head"
            detail = "dos resúmenes espaciales en paralelo; concatena canales; sin pesos"
        elif isinstance(layer, FixedPowerMeanPool):
            label, kind = f'GeM\np={layer.power:g} fijo', 'head'
            detail = (f'media generalizada espacial por mapa no negativo; p={layer.power:g} '
                      f'NO aprendible, floor={layer.epsilon:g}; sin pesos ni agregación entre pacientes')
        elif isinstance(layer, (nn.Tanh, nn.Sigmoid, nn.Softmax)):
            label, kind = type(layer).__name__, 'attention_component'
            detail = ('softmax solo sobre H×W por corte; pesos positivos suman1'
                      if isinstance(layer, nn.Softmax) else 'no linealidad, sin parámetros ni cambio de forma')
        elif isinstance(layer, GatedSpatialPool):
            label, kind = 'Atención\nmedia ponderada', 'head'
            detail = 'suma espacial de features × softmax(score); un vector por corte; parámetros en filas de ramas'
        elif isinstance(layer, (nn.MaxPool2d, nn.AvgPool2d)):
            kernel, stride = layer.kernel_size, layer.stride
            rf += (kernel - 1) * jump
            jump *= stride
            label, kind = ("AvgPool\n2×2" if isinstance(layer, nn.AvgPool2d) else "MaxPool\n2×2"), "pool"
            detail = "kernel=2, padding=0, stride=2"
        elif isinstance(layer, nn.AdaptiveAvgPool2d):
            label = "GAP\npromedio global" if shape[-1] == 1 else f"AvgPool\nadaptativo {shape[-2]}×{shape[-1]}"
            kind, detail = "head", f"salida espacial={shape[-2:]}"
            if shape[-1] > 1:
                height, width = inputs[0].shape[-2:]
                if height % shape[-2] == 0 and width % shape[-1] == 0:
                    kernel = height // shape[-2]
                    detail += f"; bins={height // shape[-2]}×{width // shape[-1]}; stride igual al bin; padding0"
                    rf += (kernel - 1) * jump
                    jump *= kernel
        elif isinstance(layer, nn.Flatten):
            # El vector se documenta en la tabla; no añade un volumen redundante.
            label, kind, detail = "Flatten", "flatten", "reorganiza; sin pesos"
        elif isinstance(layer, nn.Dropout2d):
            label, kind = f"Dropout2d\np={layer.p:g}", "regularization"
            detail = "mapas completos por muestra; solo train; sin pesos ni cambio espacial"
        elif isinstance(layer, nn.Dropout):
            label, kind, detail = f"Dropout\np={layer.p:g}", "head", "solo durante entrenamiento"
        elif isinstance(layer, nn.Linear):
            label, kind = ("Lineal\n1 logit" if layer.out_features == 1
                           else f"Lineal\n{layer.out_features} características"), "head"
            detail = f"{layer.in_features} → {layer.out_features}"
        else:
            return
        steps.append(dict(label=label, shape=shape, kind=kind, parameters=parameters,
                          rf=rf, jump=jump, detail=detail, **additional))

    has_bn = any(isinstance(layer, nn.BatchNorm2d) for layer in model.modules())
    activation_name = "SiLU" if any(isinstance(layer, nn.SiLU) for layer in model.modules()) else "ReLU"
    has_gn = any(isinstance(layer, nn.GroupNorm) for layer in model.modules())
    depthwise_layers = {layer.depthwise for layer in model.modules() if isinstance(layer, FactorizedSpatialConv)}
    pointwise_layers = {layer.pointwise for layer in model.modules() if isinstance(layer, FactorizedSpatialConv)}
    for layer in model.modules():
        if isinstance(layer, (TripletStandardization, FixedPowerMeanPool, PhaseRepresentation, nn.Conv2d, nn.MaxPool2d, nn.AvgPool2d, nn.AdaptiveAvgPool2d,
                              nn.Flatten, nn.Dropout, nn.Dropout2d, nn.Linear,
                              nn.BatchNorm2d, nn.GroupNorm, nn.ReLU, nn.SiLU, GlobalAverageMaxPool)):
            handles.append(layer.register_forward_hook(record))
        elif isinstance(layer, (GatedSpatialPool, nn.Tanh, nn.Sigmoid, nn.Softmax, ChannelConcatenation)):
            handles.append(layer.register_forward_hook(record))
    try:
        model.eval()
        with torch.inference_mode():
            model(torch.zeros(1, 3, 256, 256))
    finally:
        for handle in handles:
            handle.remove()
        # Restaurar también estados mixtos; model.train(...) uniformaría todo.
        for layer, training in original_modes.items():
            layer.training = training
    if isinstance(model, MinimalCNN):
        # La CNN mínima realiza flatten con torch.flatten, sin módulo independiente.
        linear_index = next(i for i, step in enumerate(steps) if step["label"].startswith("Lineal"))
        steps.insert(linear_index, dict(label="Flatten", shape=(512,), kind="flatten",
                                      parameters=0, rf=rf, jump=jump,
                                      detail="32 × 4 × 4 = 512 valores"))
    return steps


def dimensions(shape: tuple) -> str:
    if len(shape) == 3:
        return f"{shape[0]} @ {shape[1]}²" if shape[1] == shape[2] else f"{shape[0]} @ {shape[1]}×{shape[2]}"
    return " × ".join(map(str, shape))


def visible_diagram_steps(steps: list[dict]) -> list[dict]:
    """Agrupa visualmente DW→PW; la tabla conserva ambas capas reales.

    No muta el trace ni agrupa convoluciones densas consecutivas. El volumen
    compuesto muestra la salida de PW, no los canales intermedios de DW.
    """
    visible = [step for step in steps if step["kind"] not in {"flatten", "normalization", "activation", "attention_component"}
               and not step.get('parallel_branch', False)]
    activation_name = "SiLU" if any(s['kind']=='activation' and s['label']=='SiLU' for s in steps) else "ReLU"
    grouped = []
    index = 0
    while index < len(visible):
        first = visible[index]
        if first['kind'] == 'parallel_join':
            grouped.append({**first, 'kind':'conv', 'parameters':first['branch_parameters'],
                'label':f'Conv3×3 ∥ 3×3\nd=1 ∥ d=2\nBN + {activation_name}',
                'detail':'dos ramas paralelas; tabla mantiene parámetros separados'})
            index += 1
        elif (first['label'].startswith('DW ') and index+1 < len(visible)
                and visible[index+1]['label'].startswith('PW ')):
            second = visible[index+1]
            grouped.append({**second,
                'label': first['label'].splitlines()[0]+'\n'+second['label'].splitlines()[0]+'\n'+second['label'].splitlines()[-1],
                'parameters': first['parameters']+second['parameters'],
                'detail': 'unidad compuesta DW→PW; tabla separa ambas convoluciones'})
            index += 2
        else:
            grouped.append(first)
            index += 1
    return grouped


def draw(steps: list[dict], title: str, subtitle: str, params: int, path: Path,
         *, patient_output: bool = False) -> None:
    visible = visible_diagram_steps(steps)
    if any(step['label'].startswith('DW ') for step in steps):
        subtitle = "Cada unidad DW→PW agrupa dos convoluciones reales; la tabla detalla sus canales intermedios."
    if any(step['kind'] == 'parallel_join' for step in steps):
        subtitle = "Las primeras unidades tienen ramas local/contexto paralelas; el detalle de ramas aclara la topología."
    n = len(visible)
    fig, ax = plt.subplots(figsize=(max(22, n * 1.65), 10))
    fig.patch.set_facecolor("white")
    ax.set(xlim=(-0.7, n + 1.7), ylim=(-2.4, 5.0))
    ax.axis("off")
    navy, muted = "#17283f", "#61748c"
    palette = {"input": ("#e7efff", "#83ade9"), "conv": ("#e3f3ed", "#75bda7"),
               "pool": ("#fff0df", "#e8a358"), "head": ("#efe7f6", "#a188c1"),
               "regularization": ("#fce8ed", "#c78096")}
    ax.text(-0.5, 4.55, title, fontsize=27, color=navy, fontfamily="DejaVu Serif")
    ax.text(-0.5, 4.12, subtitle, fontsize=15, color=muted)
    # Cada ancho indica canales de forma esquemática; altura disminuye con H/W.
    for i, step in enumerate(visible):
        shape = step["shape"]
        channels = shape[0]
        side = shape[-1] if len(shape) == 3 else 1
        height = 0.65 + 2.05 * math.log2(max(side, 1) + 1) / math.log2(257)
        width = 0.18 + 0.40 * min(math.log2(channels + 1) / 8, 1)
        if len(shape) != 3:
            height = 0.55 + 0.85 * min(channels / 128, 1)
        x, y, dx, dy = i - width / 2, 1.50 - height / 2, 0.17, 0.23
        fill, edge = palette[step["kind"]]
        faces = [[(x, y), (x + width, y), (x + width, y + height), (x, y + height)],
                 [(x, y + height), (x + dx, y + height + dy),
                  (x + width + dx, y + height + dy), (x + width, y + height)],
                 [(x + width, y), (x + width + dx, y + dy),
                  (x + width + dx, y + height + dy), (x + width, y + height)]]
        for face in faces:
            ax.add_patch(Polygon(face, closed=True, facecolor=fill, edgecolor=edge, linewidth=1.8))
        ax.text(i + 0.05, 3.43, step["label"], ha="center", va="center", fontsize=11, color=navy, linespacing=1.5)
        ax.text(i + 0.05, -0.28, dimensions(shape), ha="center", fontsize=12, color=muted)
        if i < n - 1:
            ax.annotate("", xy=(i + 0.62, 1.48), xytext=(i + width / 2 + dx + 0.03, 1.48),
                        arrowprops=dict(arrowstyle="->", color="#b3beca", lw=1))
    ax.annotate("", xy=(n + 0.10, 1.48), xytext=(n - 0.55, 1.48),
                arrowprops=dict(arrowstyle="-|>", color=navy, lw=2.4))
    ax.text(n + 0.13, 1.8, "Sigmoide", fontsize=13, color=navy)
    ax.text(n + 0.13, 1.28, "Probabilidad\nde la paciente" if patient_output else "Probabilidad\ndel corte", fontsize=13, color=muted, linespacing=1.6)
    if not patient_output:
        ax.text(n + 0.13, 0.42, "↓ Media de cortes\npor paciente", fontsize=12, color=muted, linespacing=1.5)
    ax.add_patch(FancyBboxPatch((-0.5, -2.15), n + 1.85, 1.25, boxstyle="round,pad=0.12",
                               facecolor="#edf3fc", edgecolor="#aac5ed", linewidth=1.5))
    ax.text(-0.22, -1.15, f"{params:,} parámetros entrenables".replace(",", "."), fontsize=15, color=navy)
    legend = "Azul: entrada · Verde: convolución · Naranja: reducción espacial · Violeta: cabeza de salida"
    if any(step["kind"] == "regularization" for step in steps):
        legend += " · Rosa: regularización en train"
    ax.text(-0.22, -1.5, legend, fontsize=13, color=navy)
    ax.text(-0.22, -1.84, "Volúmenes esquemáticos; dimensiones exactas bajo cada capa. BCEWithLogitsLoss recibe el logit. Uso educativo, no clínico.", fontsize=12, color=muted)
    # IDs y metadatos estables: regenerar no debería ensuciar Git solo por fecha.
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight", metadata={"Date": None})
    fig.savefig(path.with_suffix(".png"), dpi=120, bbox_inches="tight")
    plt.close(fig)


def draw_attention_head(pool: GatedSpatialPool, directory: Path,
                        spatial_shape: tuple[int, int]) -> None:
    """Topología de ramas; tamaños y parámetros extraídos del módulo real."""
    c, h = pool.channels, pool.hidden_channels
    height, width = spatial_shape
    hw = f'{height} × {width}'
    formatted = lambda n: f'{n:,}'.replace(',', '.')
    count = lambda layer: sum(p.numel() for p in layer.parameters())
    fig, ax = plt.subplots(figsize=(18, 7))
    ax.set(xlim=(0,18.4), ylim=(-.8,7.1));ax.axis('off')
    navy, muted = '#17283f', '#61748c'
    def box(x,y,label,color='#e5f4ef'):
        ax.add_patch(FancyBboxPatch((x-1.12,y-.55),2.24,1.1,
            boxstyle='round,pad=.1',facecolor=color,edgecolor='#92baa9',linewidth=1.5))
        ax.text(x,y,label,ha='center',va='center',fontsize=11,color=navy)
    def arrow(start,end,label=None):
        ax.annotate('',xy=end,xytext=start,arrowprops=dict(arrowstyle='->',lw=1.6,color=muted))
        if label:ax.text((start[0]+end[0])/2,(start[1]+end[1])/2+.17,label,
                        color=muted,ha='center',fontsize=10)
    box(1.4,3.5,f'Mapa final\n{c} × {hw}','#edf3fc')
    box(4.6,5,f'Conv1×1 {c}→{h}\n+ tanh\n{h} × {hw}\n{formatted(count(pool.value))} parámetros')
    box(4.6,2,f'Conv1×1 {c}→{h}\n+ sigmoid\n{h} × {hw}\n{formatted(count(pool.gate))} parámetros')
    box(7.8,3.5,f'Producto ⊙\n{h} × {hw}\n0 parámetros')
    box(10.9,3.5,f'Conv1×1 {h}→1\n1 × {hw}\n{count(pool.score)} pesos, sin bias')
    box(14,3.5,f'Softmax espacial\n{height*width} pesos, suma 1\n0 parámetros','#eee7f4')
    box(14,1,'Suma ponderada\nde los mapas originales\n0 parámetros','#eee7f4')
    box(17,1,f'{c} valores\n→ dropout → lineal\n→ 1 logit','#eee7f4')
    arrow((2.62,3.6),(3.38,4.75));arrow((2.62,3.4),(3.38,2.25))
    arrow((5.82,4.75),(6.58,3.65));arrow((5.82,2.25),(6.58,3.35))
    arrow((9.02,3.5),(9.68,3.5));arrow((12.12,3.5),(12.78,3.5))
    arrow((14,2.85),(14,1.65),'pesos')
    ax.plot([1.4,1.4,12.7],[2.85,.25,.25],color=muted,linewidth=1.5)
    arrow((12.7,.25),(12.78,.9))
    ax.text(6.8,.4,'Mapas originales: no sumar los 16 canales de la compuerta',fontsize=10,color=muted,ha='center')
    arrow((15.22,1),(15.78,1))
    ax.text(.25,6.75,'E30 · Atención espacial con dos ramas paralelas',fontsize=22,color=navy)
    ax.text(.25,6.2,f'{formatted(count(pool))} parámetros nuevos · conv1×1/padding0/stride1 · score=0 al iniciar: pesos uniformes (GAP)',fontsize=12,color=muted)
    ax.text(.25,-.45,'No son pesos entre pacientes ni una segmentación tumoral. Compuerta sigmoid ≠ probabilidad pCR. Uso educativo, no clínico.',fontsize=11,color=muted)
    fig.savefig(directory/'atencion.png',dpi=120,bbox_inches='tight',facecolor='white')
    fig.savefig(directory/'atencion.svg',bbox_inches='tight',facecolor='white',metadata={'Date':None})
    plt.close(fig)


def write_sheet(directory: Path, title: str, model: nn.Module, explanation: str,
                configs: list[str], steps: list[dict]) -> None:
    params = trainable_parameter_count(model)
    table = ["| Operación | Salida por corte | Parámetros de la operación | Detalle |",
             "|---|---|---:|---|"]
    for step in steps:
        # El diagrama agrupa Conv/Norm/ReLU; la tabla separa su orden real.
        label = step['label'].splitlines()[0] if step['kind'] == 'conv' else step['label'].replace(chr(10), ' · ')
        table.append(f"| {label} | `{step['shape']}` | {step['parameters']} | {step.get('detail', '3 fases temporales')} |")
    local_rf = next(step["rf"] for step in reversed(steps) if step["kind"] in {"conv", "pool"})
    activation_name = "SiLU" if any(isinstance(layer, nn.SiLU) for layer in model.modules()) else "ReLU"
    bn_note = f"Las filas de normalización incluyen sus parámetros propios; en el dibujo se agrupan con Conv y {activation_name}."
    content = f"# {title}\n\n![Diagrama de la arquitectura](arquitectura.png)\n\n" + explanation
    content += f"\n\nTotal: **{params:,} parámetros entrenables**.\n\n".replace(",", ".")
    content += "Configuraciones asociadas:\n\n" + "\n".join(f"- [{name}](../../configs/experiments/{name})" for name in configs)
    if not configs:
        content += "- [CNN mínima](../../src/models/model.py), utilizada por la prueba de memorización."
    content += "\n\n## Recorrido de las capas\n\n" + "\n".join(table)
    content += f"\n\nLas formas omiten el batch `B`. {bn_note}\n"
    if any(s.get('global_input_statistics') for s in steps):
        content += ('\nLa normalización de entrada usa todos los píxeles de las tres fases de '
                    'esa tripleta incluso en eval. El RF convolucional local mostrado NO es '
                    'la dependencia total del modelo: ya hay dependencia global antes de Conv. '
                    'No normaliza fases por separado ni aprende estadísticas del dataset.\n')
    if any(isinstance(layer, FactorizedSpatialConv) for layer in model.modules()):
        content += "\nEl dibujo agrupa DW→PW en una unidad cuyo volumen es la salida de PW. La tabla mantiene ambas operaciones y sus formas intermedias. No hay BN/ReLU entre ellas: solo después de PW. Quince convoluciones reales, no ocho; cuatro pools entre las dos unidades lógicas de cada bloque.\n"
    if any(isinstance(layer, ParallelContextConv) for layer in model.modules()):
        content += "\n## Ramas local/contexto\n\n![Detalle de ramas](ramas.png)\n\n[SVG de ramas](ramas.svg). Las dos ramas reciben la misma entrada; concatenación de canales antes de una BN/ReLU. No están en serie. Doce convoluciones reales, ocho unidades lógicas, cuatro pools entre unidades. Parámetros de ramas se cuentan solo en sus filas, no otra vez en concatenación. Campo receptivo máximo combina rutas; no todas tienen el mismo contexto. Inicialización Kaiming/ReLU usa Cout total para ambas ramas, no mitad.\n"
    content += f"\nAntes del pooling global, el campo receptivo **convolucional local** es **{local_rf}×{local_rf} píxeles**. Este cálculo sigue kernels y strides; no incorpora la dependencia espacial global de las estadísticas de normalización. El pooling global combina posiciones; no equivale a localizar un tumor.\n"
    extractor_convs = [layer for block in getattr(model, 'blocks', []) for layer in block.modules()
                       if isinstance(layer, nn.Conv2d)]
    if not extractor_convs:
        extractor_convs = [layer for layer in model.modules() if isinstance(layer, nn.Conv2d)]
    conv_specs = sorted({(layer.kernel_size[0], layer.padding[0], layer.stride[0], layer.dilation[0]) for layer in extractor_convs})
    conv_note = "; ".join(f"kernel {kernel}×{kernel}, padding {padding}, stride {stride}"
                          + (f", dilation {dilation}" if dilation != 1 else "")
                          for kernel, padding, stride, dilation in conv_specs)
    pool_note = "AvgPool" if any(isinstance(layer, nn.AvgPool2d) for layer in model.modules()) else "MaxPool"
    content += f"\nConvoluciones del extractor: {conv_note}; cada {pool_note} 2×2 tiene stride 2. {activation_name} aporta no linealidad. La sigmoide de salida se aplica para evaluar, después del logit. La media de probabilidades por paciente es la agregación de estas pruebas, elegida en desarrollo; test no se utiliza para ajustarla.\n"
    if isinstance(getattr(model, 'global_pool', None), GatedSpatialPool):
        content += "\n## Cabeza de atención: ramas paralelas\n\n![Detalle de atención](atencion.png)\n\nLas ramas V y U reciben el MISMO mapa `[128,16,16]`, no una detrás de otra. Conv1×1/padding0/stride1 →16 canales por rama, tanh/sigmoid, producto y Conv1×1 →1 score por posición. Softmax sobre256 posiciones y suma ponderada devuelven128 valores. No mezcla pacientes/cortes ni usa una máscara tumoral. Los4.144 parámetros nuevos se cuentan solo en las filas de las ramas; el módulo compuesto no los duplica. Inicialización score=0 → atención uniforme (GAP); la compuerta se aprende después. Sigmoid de compuerta NO es probabilidad pCR. [SVG de la cabeza](atencion.svg).\n"
    content += f"\n## Imágenes y regeneración\n\n[SVG vectorial](arquitectura.svg) · [PNG](arquitectura.png). Las etiquetas y dimensiones se obtienen ejecutando la red con una entrada ficticia; no se leen imágenes de pacientes.\n\n```bash\npython -m src.reporting.document_architectures --only {directory.name}\n```\n\nEste comando no regenera otras fichas. Los SVG tienen identificadores estables y no incluyen fechas de generación.\n\nUso educativo y de investigación, sin validez clínica.\n"
    (directory / "README.md").write_text(content, encoding="utf-8")


def document_patient_feature_architecture(folder):
    from src.models.patient_feature_model import PatientFeatureNet, group_features
    from src.training.train_patient_features import load_spec
    spec, cfg = load_spec(ROOT / 'configs/patient_features' / (folder + '.json'))
    model = PatientFeatureNet(cfg.model).eval()
    directory = ROOT / 'experiments' / folder
    directory.mkdir(parents=True, exist_ok=True)
    title = ('E56 · Cabeza no lineal: control por corte' if spec['pool_stage'] == 'probabilities'
             else 'E57 · Características combinadas por paciente')
    steps = trace(model)
    for step in steps:
        if step.get('detail') == '128 → 32':
            step['label'] = 'Lineal + ReLU\n128 → 32'
    assert sum(s['parameters'] for s in steps) == trainable_parameter_count(model) == 298161
    draw(steps, title, 'Ruta real de un corte; cabeza compartida 128 → 32 → 1.',
         trainable_parameter_count(model), directory / 'arquitectura')
    explanation = ('CNN propia desde cero compartida entre cortes. Ocho convoluciones3×3/'
        'padding1/stride1, canales16/32/64/128, ocho BN/ReLU y cuatro MaxPool2×2 intermedios. '
        'GAP128 y cabeza Linear128→32(4128 parámetros)→ReLU→Dropout0,3→Linear32→1(33). '
        'Extractor294000, total298161, RF local106. [Protocolo y restricciones](../notes/CARACTERISTICAS_PACIENTE.md). '
        'Cambios frente E19: cabeza, pérdida/lotes por paciente y agregación; no aislarlos. '
        'Solo E57−E56 compara punto de agregación con misma cabeza y lotes. '
        'La granularidad del dropout cambia necesariamente con ese punto. '
        'Esta figura es la ruta de un corte: no representa diez cortes en serie. '
        'Pendiente científico; mantener test cerrado. Los checkpoints nuevos requieren '
        'load_feature_checkpoint; NO son intercambiables con el cargador web actual.')
    write_sheet(directory, title, model, explanation,
                ['../patient_features/' + folder + '.json'], steps)
    # Formas de la media y la cabeza comprobadas con dos vectores sintéticos.
    with torch.inference_mode():
        f = torch.zeros(2, 128)
        m, _, _ = group_features(f, torch.zeros(2), ['p', 'p'])
        hidden = model.head[1](model.head[0](m))
        output = model.classify_features(m)
    bag_steps = [dict(label='Cortes compartidos\nN × 128', shape=(f.shape[1],), kind='input', parameters=0, rf=106, jump=16),
                 dict(label='Media por paciente\n128', shape=tuple(m.shape[1:]), kind='head', parameters=0, rf=106, jump=16),
                 dict(label='Lineal + ReLU\n32', shape=tuple(hidden.shape[1:]), kind='head', parameters=4128, rf=106, jump=16),
                 dict(label='Dropout\np=0,3', shape=tuple(hidden.shape[1:]), kind='head', parameters=0, rf=106, jump=16),
                 dict(label='Lineal\n1 logit', shape=(1,), kind='head', parameters=33, rf=106, jump=16)]
    assert output.shape == (1,)
    draw(bag_steps, 'Ruta de bolsa de características · E57',
         'Mismo extractor para cada corte; media sin pesos ni orden axial; N=1 también válido.',
         trainable_parameter_count(model), directory / 'bolsa', patient_output=True)
    with (directory / 'README.md').open('a') as stream:
        stream.write('\n## Ruta alternativa de bolsa\n\n![Media de características](bolsa.png)\n\n'
            '[SVG bolsa](bolsa.svg). La bolsa se usa en train SOLO en E57; en E56 es '
            'diagnóstico secundario. La media no aprende interacciones por pares ni orden axial. '
            'No tiene kernel/padding/stride; combina N vectores128→1 vector128. N=1 '
            'es idéntico al forward por corte en eval.\n')
    print('Documentado: ' + folder)


def main(only: str | None = None) -> None:
    if only == 'K02_knet_mil':
        from src.reporting.document_knet_mil import document
        document()
        return
    if only == 'K01_knet_se_dualpool':
        from src.reporting.document_knet import document
        document()
        return
    if only in {'C19_compacta_4conv','C21_pooling_local','C22_groupnorm'}:
        from src.reporting.document_campaign_20261016 import document
        document(only)
        return
    if only == 'C16_cnn_97809':
        from src.reporting.document_simple_cnn import document
        document()
        return
    if only == 'C15_ensemble_dos_semillas':
        from src.reporting.document_campaign_ensemble import document
        document()
        return
    if only == 'C15_fusion_tardia':
        from src.campaigns.campaign_20261015_secondary import document
        document()
        return
    if only == 'C15_E57_compacta_cortes':
        from src.reporting.document_campaign_compact import document
        document()
        return
    if only == 'B09_cnn_regularizada':
        from src.reporting.document_basic_regularized import document
        document()
        return
    if only in {'B05_cnn_sencilla','B06_fases_originales','B07_sencilla_fases_originales'}:
        from src.reporting.document_multimodal import document
        document(f'experiments/{only}/config.json',
                 f'experiments/{only}',basic=True)
        return
    if only == 'B03_silu_cnn':
        from src.reporting.document_multimodal import document
        document('experiments/B03_silu_cnn/config.json',
                 'experiments/B03_silu_cnn',basic=True)
        return
    if only == 'D36_edad_menopausia':
        from src.reporting.document_multimodal import document
        document('experiments/D36_age_menopause/run/config.json',
                 'experiments/D36_age_menopause',basic=True)
        return
    if only in {'D30_groupnorm4','D32_spatial_dropout01'}:
        from experimentos.document_regularized_multimodal import document
        document(only)
        return
    if only == 'D23_cnn_compacta':
        from src.reporting.document_compact_multimodal import document
        document()
        return
    if only == 'D19_norma_features':
        from src.reporting.document_feature_norm import document
        document()
        return
    if only == 'D14_pooling_realce':
        from src.reporting.document_enhancement import document
        document()
        return
    if only == 'D11_fusion_no_lineal':
        from src.reporting.document_fusion import document
        document()
        return
    if only == 'B08_solo_edad_menopausia':
        from src.reporting.document_clinical_only import document
        document(basic=True)
        return
    if only == 'D08_clinica_sola':
        from src.reporting.document_clinical_only import document
        document()
        return
    if only == 'E58_clinical_fusion_mac':
        from src.reporting.document_multimodal import document
        document()
        return
    feature_folders = ('E56_nonlinear_cut_control_mac', 'E57_patient_feature_mean_mac')
    if only in feature_folders:
        document_patient_feature_architecture(only)
        return
    catalog = [
        ("A01_minima", "A01 · CNN mínima de depuración", MinimalCNN(), [],
         "Tres bloques Conv–ReLU–Pool, canales 8/16/32. Resume a 4×4 y conecta 512 valores a un logit. Su objetivo fue memorizar un subconjunto pequeño para comprobar el cableado; no es una estimación de generalización."),
    ]
    specifications = [
        ("E31_weight_decay_003_mac", "E31 · Misma CNN, weight decay 0,003", "E31_weight_decay_003_mac.json", ["E31_weight_decay_003_mac.json"],
         "No es una arquitectura nueva ni una continuación de pesos E19: misma CNN propia de 294.129 parámetros, inicializada desde cero. Solo training.weight_decay cambia 0,001→0,003 en AdamW; datos completos, fases/restas firmadas, capas, LR0,001, BCE normal por corte, batch16, dropout0,3, Plateau, máximo50/paciencia10 y MPSfloat32 se conservan. AdamW contrae los parámetros durante cada actualización, no añade capas ni pone filtros a mano. Hipótesis: regularización mayor podría reducir memorización; riesgo de infraajuste o efecto insuficiente. Dos semillas prefijadas42/17 se comparan con los E19 correspondientes ya completos, sin relanzarlos ni elegir solo el pico. Completas: AUC42/17=0,616633/0,598286, mejores épocas1/11 y2/12. Sin mejora observada frente a sus E19; no relanzar. [Cierre y protocolo histórico](../WEIGHT_DECAY_E31.md)."),
        ("E30_gated_attention_mac", "E30 · CNN con atención espacial aprendida", "E30_gated_attention_mac.json", ["E30_gated_attention_mac.json"],
         "Nueva cabeza propia sobre el extractor convolucional E19: sustituye únicamente global_pooling=average por gated_attention. Aprende una media espacial de los128 mapas finales con dos ramas1×1 de16 canales, tanh/sigmoid, producto y score1×1 sin bias; softmax sobre16×16 posiciones. Total298.273 parámetros,4.144 más (+1,41%). Hipótesis: dar peso diferente a zonas útiles podría evitar diluir patrones locales; NO está demostrado que GAP causara el AUC bajo. Mantiene cuatro bloques Conv–BN–ReLU → Pool → Conv–BN–ReLU, fases/restas, datos, BCE normal, AdamW LR/WD0,001, Plateau, batch16, fold0seed42, máximo50/paciencia10, MPSfloat32. Sin Mixup, aumentos extra, SAM, pesos preentrenados ni arquitectura importada. Todo se inicializa desde cero; score=0 arranca como GAP. Atención espacial dentro de un corte, no atención entre pacientes ni segmentación/explicación causal. Completo en MPS: AUC paciente0,615423, AP0,395345, Brier0,203473, mejor época6/16, 1.062,63s. Sobreajuste persistente, sin mejora observada E19; delta−0,022782, IC exploratorio95% [−0,101817;+0,049206]. No relanzar ni prolongar. [Resultado y protocolo](../ATENCION_E30.md); [diagnóstico y curva de cantidad de pacientes](../CURVA_PACIENTES.md)."),
        ("E29_mixup_mac", "E29 · Misma CNN, Mixup en train", "E29_mixup_mac.json", ["E29_mixup_mac.json"],
         "Arquitectura idéntica a E19: CNN 2D propia desde cero, 294.129 parámetros, cuatro bloques Conv–BN–ReLU → Pool → Conv–BN–ReLU, canales 16/32/64/128, diferencias firmadas, GAP y dropout0,3. No se añade ninguna capa. Solo cambia training.mixup: combina dos tripletas y sus etiquetas con un mismo coeficiente Beta(0,2;0,2) por lote, solo en train. No genera pacientes independientes ni cambia validación/web. Mantiene BCE normal por corte, AdamW LR/WD0,001, Plateau (NO calendario E28), fold0/seed42, batch16, máximo50/paciencia10, MPS float32; sin rotación ni ganancia adicional. Hipótesis de regularización aún no comprobada: las mezclas pueden no ser anatómicamente plausibles y también perjudicar el aprendizaje. Pesos inicializados desde cero; no transfer learning. Completo en MPS: AUC paciente0,616835, AP0,378680, Brier0,223213, mejor época1/11, 744,43s. No mejora observada frente a E19; delta−0,021371, IC exploratorio95% [−0,103170;+0,058894]. Sobreajuste persistente, test cerrado. No relanzar. [Resultado y protocolo](../MIXUP_E29.md)."),
        ("E28_warmup_cosine_mac", "E28 · Misma CNN, calendario LR nuevo", "E28_warmup_cosine_mac.json", ["E28_warmup_cosine_mac.json"],
         "No añade capas ni parámetros: misma CNN E19 propia, 294.129 parámetros, ocho convoluciones, cuatro pools intermedios, BatchNorm, diferencias firmadas y GAP. Solo cambia training.lr_schedule: warmup lineal por época (0,0001 en la primera, 0,001 en la segunda), descenso coseno durante ocho épocas hasta 0,00001 en la décima; después suelo fijo. Política prefijada sin reinicios ni uso de test. Se mantienen AdamW/WD0,001, BCE normal por corte, batch16, fold0/seed42, MPS float32, máximo50/paciencia10 por AUC paciente. No Mixup, SAM ni aumentos. La política completa es una variable principal: no aisla warmup y coseno por separado. Hipótesis de optimización, no mejora demostrada. Completo en MPS: mejor época3/13, ROC-AUC paciente0,603427, AP0,383880, Brier0,201072, 854,22s. No mejora observada frente a E19; delta AUC−0,034778 con IC exploratorio95% [−0,116336;+0,043967]. E19 sigue provisional. No relanzar. [Resultado y protocolo](../CALENDARIO_LR_E28.md) · [Calendario previsto](learning_rate.png)."),
        ("E21_group_norm", "E21 · Normalización por grupos", "E21_group_norm.json", ["E21_group_norm.json"],
         "Solo sustituye las ocho BatchNorm de E19 por GroupNorm con cuatro grupos. Cada grupo contiene 4/8/16/32 canales en los bloques 1/2/3/4. Media y varianza se calculan por muestra, sobre canales del grupo y posiciones espaciales; no dependen de otras imágenes del lote ni de medias acumuladas. Conserva dos parámetros afines por canal y las dimensiones. No normaliza los PNG ni las fases por separado. Hipótesis: reducir dependencia de estadísticas del lote y discrepancias train/eval. Riesgo: perder información de escala de activaciones o regularización útil de BN. El diagnóstico BN anterior NO probó que BN causara el problema. Referencia E19: mismos datos, LR 0,001, weight decay 0,001, batch 16, dropout 0,3, 50 épocas/paciencia 10, sin aumentos. Completado en MPS: mejor época 5 de 15, ROC-AUC 0,526815 y AP 0,296088. No supera E19 en este fold/semilla. [Protocolo y fuentes](../TANDA_NOCTURNA.md)."),
        ("E22_average_max", "E22 · Resumen espacial doble", "E22_average_max.json", ["E22_average_max.json"],
         "Solo cambia la cabeza de E19: del último mapa [128,16,16] obtiene en paralelo una media y un máximo por canal, concatena [256,1,1], aplana, aplica dropout 0,3 y una lineal 256→1. No aplica max después de la media ni elige el corte de mayor probabilidad; la agregación por paciente sigue siendo media. La cabeza añade 128 pesos, total 294.257. Hipótesis: complementar patrones extendidos con respuestas locales intensas. Riesgo: amplificar ruido/artefactos y sobreajuste. No se importa CBAM ni ninguna arquitectura externa. Resto idéntico a E19, máximo 50 épocas/paciencia 10. Completado en MPS: mejor época 18 de 28, ROC-AUC 0,580343 y AP 0,345463. No supera E19; sobreajuste marcado. [Protocolo y fuentes](../TANDA_NOCTURNA.md)."),
        ("E23_compact", "E23 · Capacidad reducida sobre E19", "E23_compact.json", ["E23_compact.json"],
         "Solo reduce canales de E19 de 16/32/64/128 a 8/16/32/64; mantiene ocho convoluciones, cuatro pools, BatchNorm y GAP. Cabeza 64→1; total 73.913 parámetros. Hipótesis: menor capacidad podría reducir sobreajuste y coste; riesgo de infraajuste. E10 ya probó esta anchura, pero con entrada raw, weight decay 0,0001 y diez épocas: E23 mide esa decisión con diferencias firmadas y los ajustes actuales de E19. No se presenta como una idea nunca probada ni mejora asegurada. Máximo 50 épocas, paciencia 10, batch 16 y LR 0,001. Completado en MPS: mejor época 12 de 22, ROC-AUC 0,593145 y AP 0,370721. No supera E19 en este fold/semilla. [Protocolo y fuentes](../TANDA_NOCTURNA.md)."),
        ("E18_affine15_mac", "E18 · Aumentos moderados", "E18_affine15_mac.json", ["E18_affine15_mac.json"],
         "La CNN es idéntica a E17: entrada PRE/EARLY/LATE [3,256,256], diferencias firmadas dentro del modelo, cuatro bloques Conv–BN–ReLU → Pool → Conv–BN–ReLU con canales 16/32/64/128, GAP, dropout 0,3 y un logit. Son 294.129 parámetros; campo receptivo local 106×106. Kernels 3×3, padding 1 y stride 1; pooling 2×2 y stride 2. No se añaden capas ni parámetros. Solo se activan aumentos de train: giro uniforme ±15° y desplazamiento ±5 % por eje (12,8 px). La misma rejilla bilineal transforma las tres fases antes de calcular sus diferencias, con relleno cero. No se aumentan validación ni web. Hipótesis: reducir dependencia de la posición; comprobar visualmente que no se pierde información relevante. Weight decay 0,0001. Mantiene batch 16, LR 0,001, BCE normal, MPS float32, fold 0, seed 42, máximo 50 épocas y paciencia 10 por ROC-AUC paciente. E17 ya está completo: AUC 0,589415; E18 completó 14 épocas, mejor época 4: ROC-AUC 0,579435 y AP 0,360488. No supera el AUC observado de E17. Siguiente: [confirmación E19](../E19_ANALISIS_Y_CONFIRMACION.md). Ver [diseño de comparación y comandos](../REGULARIZACION_E18_E20.md)."),
        ("E20_affine15_weight_decay_mac", "E20 · Aumentos y regularización", "E20_affine15_weight_decay_mac.json", ["E20_affine15_weight_decay_mac.json"],
         "La CNN es idéntica a E17: entrada PRE/EARLY/LATE [3,256,256], diferencias firmadas dentro del modelo, cuatro bloques Conv–BN–ReLU → Pool → Conv–BN–ReLU con canales 16/32/64/128, GAP, dropout 0,3 y un logit. Son 294.129 parámetros; campo receptivo local 106×106. Kernels 3×3, padding 1 y stride 1; pooling 2×2 y stride 2. No se añaden capas ni parámetros. Combina aumentos sincronizados ±15°/±5 % solo en train con weight decay 0,001. E20−E18 aísla añadir regularización de pesos con aumentos; E20−E19 aísla añadir aumentos con regularización mayor. Frente a E17 cambian dos factores: no atribuir el resultado a uno solo. Una única semilla no demuestra una interacción reproducible. Mantiene batch 16, LR 0,001, BCE normal, MPS float32, fold 0, seed 42, máximo 50 épocas y paciencia 10 por ROC-AUC paciente. E17 ya está completo: AUC 0,589415; E20 completó 28 épocas, mejor época 18: ROC-AUC 0,563206 y AP 0,320784. No supera el AUC observado de E17. Siguiente: [confirmación E19](../E19_ANALISIS_Y_CONFIRMACION.md). Ver [diseño de comparación y comandos](../REGULARIZACION_E18_E20.md)."),
        ("E19_weight_decay_001_mac", "E19 · Weight decay 0,001 en Mac", "E19_weight_decay_001_mac.json", ["E19_weight_decay_001_mac.json", "E24_weighted_bce_mac.json", "E25_shared_gain_mac.json", "E26_grouped_cut_mac.json", "E27_patient_mean_mac.json"],
         "Ablación de E17 Mac: solo cambia weight_decay de AdamW de 0,0001 a 0,001. AdamW aplica una contracción de pesos desacoplada del gradiente de la pérdida; no añade una capa ni modifica el logit en evaluación. Hipótesis: una regularización diez veces mayor podría reducir sobreajuste, aunque también puede limitar el aprendizaje. Misma CNN E13: diferencias firmadas PRE, EARLY−PRE y LATE−EARLY, canales 16/32/64/128, ocho convoluciones, cuatro pools intermedios, BatchNorm, GAP, dropout final 0,3 y salida de un logit. Total 294.129 parámetros y campo receptivo local 106×106. Batch 16, LR inicial 0,001, BCE normal, sin aumentos, fold 0, seed 42, MPS float32, workers 0, máximo 50 épocas y paciencia 10 de ROC-AUC por paciente. Referencia E17 completada: mejor época 6, ROC-AUC 0,589415 y AP 0,357588. E19 completó 13 épocas en MPS, mejor época 3: ROC-AUC 0,638206 y AP 0,401181. Candidato provisional, no superioridad confirmada. [Análisis y réplicas pareadas](../E19_ANALISIS_Y_CONFIRMACION.md). No relanzar el original. Ver [protocolo E19](../WEIGHT_DECAY_E19.md)."),
        ("E16_batch128_mac", "E16 · Batch 128 en Mac", "E16_batch128_mac.json", ["E16_batch128_mac.json"],
         "La arquitectura es exactamente E13: diferencias firmadas PRE, EARLY−PRE, LATE−EARLY; ocho convoluciones, cuatro pools intermedios, BatchNorm, GAP y dropout final 0,3. No se añade ninguna capa ni parámetro: 294.129 parámetros y campo receptivo local 106×106. El cambio principal frente a E17 es batch 16 → 128: entrada completa [128,3,256,256], último lote menor; son cortes, no necesariamente pacientes distintas. Los mapas de un lote completo son [128,16,128,128], [128,32,64,64], [128,64,32,32] y [128,128,16,16]. Cambian las estadísticas de BatchNorm y se realizan 69 actualizaciones por época en vez de 548, con los 8.759 cortes de train completos. Mantiene LR inicial 0,001, weight decay 0,0001, BCE normal, fold 0 y seed 42; no se escala LR automáticamente. Para Mac: MPS float32, workers 0, pin_memory false, máximo 50 épocas y paciencia 10 sin mejora del ROC-AUC paciente. Es paciencia, no un corte fijo en época 10. Completado en MPS: 15 épocas, mejor época 5, ROC-AUC 0,580948 y AP 0,393112; 977,36 s. Comparación pareada E17−E16 +0,008468, IC exploratorio [−0,083800; +0,100708], sin ventaja clara. No relanzar. Ver [protocolo batch](../BATCH_MAC.md)."),
        ("E17_batch16_mac", "E17 · Referencia batch 16 en Mac", "E17_batch16_mac.json", ["E17_batch16_mac.json"],
         "Referencia emparejada de E16: misma arquitectura E13, inicialización, fold, seed, pérdidas, LR, scheduler, máximo 50 épocas y paciencia 10. Solo cambia batch 128 → 16. Conserva 294.129 parámetros, ocho convoluciones, cuatro pools y campo receptivo local 106×106; entrada completa [16,3,256,256], 548 actualizaciones por época y último lote de siete cortes. El batch no cambia las dimensiones por corte, los filtros ni el preprocesamiento. Mantiene MPS float32, workers 0 y pin_memory false, igual que E16. E13 Mac antiguo solo tuvo diez épocas y E13 largo se ejecutó en ROCm con otra precisión y criterio de parada: ninguno sustituye a esta referencia controlada. Completado en MPS: 16 épocas, mejor época 6, ROC-AUC 0,589415 y AP 0,357588; 1.099,44 s. Persiste sobreajuste. Referencia para E18–E20; no relanzar ni sobrescribir. Ver [protocolo batch](../BATCH_MAC.md)."),
        ("E14_spatial_dropout_010", "E14 · Dropout espacial 0,1", "E14_spatial_dropout_010.json", ["E14_spatial_dropout_010.json"],
         "Variante controlada de E13_50epochs: tras la segunda Conv–BN–ReLU de cada bloque se añade Dropout2d con p=0,1. Durante entrenamiento anula aleatoriamente mapas aprendidos completos por muestra y escala los supervivientes por 1/(1−p); en evaluación no anula ninguno. No se borran fases PRE/EARLY/LATE ni se modifica la imagen. Conserva ocho convoluciones, cuatro pools intermedios, 294.129 parámetros y campo receptivo local 106×106. Hipótesis: limitar la dependencia de mapas concretos podría mejorar generalización; también podría causar infraajuste. La forma de salida del bloque no cambia. Se conserva dropout final 0,3, diferencias firmadas, batch 16, LR inicial 0,001, weight decay 0,0001, BCE normal, seed 42, fold 0, 50 épocas y paciencia 51. Resultado universitario recibido: 50 épocas, mejor época 15, ROC-AUC paciente 0,584526 y AP 0,353891, en 1.010,26 s. No supera el ROC-AUC observado de E13_50epochs (0,591280); no hay CSV originales para cuantificar incertidumbre pareada. Ver [protocolo de comparación](../notes/DROPOUT_ESPACIAL.md)."),
        ("E15_spatial_dropout_020", "E15 · Dropout espacial 0,2", "E15_spatial_dropout_020.json", ["E15_spatial_dropout_020.json"],
         "Misma arquitectura de E14 con p=0,2 en los cuatro Dropout2d. Respecto a E13_50epochs solo cambia spatial_dropout de 0 a 0,2; respecto a E14 cambia solo su intensidad. No se eliminan canales de forma permanente ni se añaden parámetros. Cada bloque conserva dimensiones, kernel, padding, stride y campo receptivo. Busca comprobar si una regularización moderada evita depender de pocos mapas; no hay mejora garantizada. Se mantienen todos los ajustes de E13_50epochs, incluyendo dropout final 0,3, 50 épocas, paciencia 51 y selección por ROC-AUC por paciente. No hay resultados científicos todavía; primero probar E14, no lanzar ambos en paralelo al entrenamiento actual. Ver [protocolo](../notes/DROPOUT_ESPACIAL.md)."),
        ("E13_phase_differences", "E13 · Realce explícito entre fases", "E13_phase_differences.json", ["E13_phase_differences.json", "E13_50epochs.json"],
         "La entrada externa sigue siendo PRE/EARLY/LATE en [0,1], con forma [3,256,256]. Dentro del modelo una operación fija conserva PRE y calcula EARLY−PRE y LATE−EARLY: tres canales, cero parámetros nuevos y sin recortar negativos. Las ocho convoluciones, cuatro MaxPool, BatchNorm, GAP y dropout son los de A04/E05; total 294.129 parámetros y campo receptivo local 106×106. Se conserva LR inicial 0,001 y todos los ajustes de E05 Mac. Es un cambio de representación invertible, no información nueva ni una CNN más grande: EARLY=PRE+(EARLY−PRE), LATE=EARLY+(LATE−EARLY). Hipótesis: presentar directamente los cambios temporales podría facilitar optimización. La primera convolución original ya podría aprender restas, por lo que no hay mejora garantizada. Las escalas/correlaciones de entrada cambian y pueden afectar la optimización y BatchNorm; esto forma parte del experimento. E13 completó diez épocas en Mac MPS: mejor época 10, ROC-AUC 0,572480 y AP 0,350860; 421,48 segundos. No supera la referencia E05 Mac (ROC-AUC 0,586895). El mejor checkpoint al final no demuestra convergencia ni que más épocas ayuden; revisar diagnóstico train/validación antes de decidir. Ver [protocolo E13](../notes/REALCE_E13.md)."),
        ("A02_dos_convoluciones", "A02 · Dos convoluciones antes del pooling", "E02_base_normal.json", ["E02_base_normal.json", "E03_base_weighted.json"],
         "Cada bloque hace Conv–BN–ReLU → Conv–BN–ReLU → MaxPool. La hipótesis es combinar características antes de reducir resolución. E02 y E03 comparten esta arquitectura y cambian únicamente la pérdida. En el primer fold y semilla E02 obtuvo ROC-AUC 0,6204 y E03 0,6029; no basta para declarar una arquitectura ganadora."),
        ("A03_una_convolucion", "A03 · Una convolución por bloque", "E04_one_conv_normal.json", ["E04_one_conv_normal.json"],
         "Cada bloque hace Conv–BN–ReLU → MaxPool. E04 elimina la segunda convolución para estudiar si reducir capacidad ayuda a generalizar. Obtuvo ROC-AUC 0,5998 en el primer fold y semilla: fue más rápido, pero no mejoró el resultado observado de E02. La diferencia requiere confirmación con más folds o semillas."),
        ("A04_pool_intermedio", "A04 · Pooling entre las convoluciones", "E05_pool_between_convs.json", ["E05_pool_between_convs.json"],
         "Cada bloque hace Conv–BN–ReLU → MaxPool → Conv–BN–ReLU. Se mueve el único pooling del bloque; no se añade otro al final. Conserva los parámetros y las dimensiones finales de A02, pero la segunda convolución trabaja a menor resolución. Hipótesis: estudiar el efecto de reducir antes de refinar características. E05 completó diez épocas: el checkpoint elegido por ROC-AUC fue el de la época 1, con ROC-AUC por paciente 0,6140 y PR-AUC 0,3910. A umbral 0,5 predijo todas las pacientes como negativas; no hay mejora demostrada de generalización. Cambiar el presupuesto de nueve a diez épocas frente a los baselines limita la comparación: confirmar candidatos con igual presupuesto."),
        ("A05_cinco_bloques", "A05 · Cinco bloques con pooling intermedio", "E06_five_blocks_normal.json", ["E06_five_blocks_normal.json"],
         "Se añade un quinto bloque de 128 canales a A04/E05: Conv–BN–ReLU → MaxPool → Conv–BN–ReLU. Son diez convoluciones y cinco MaxPool, con resolución 256 → 128 → 64 → 32 → 16 → 8. No se amplía a 256 canales para no cambiar profundidad y anchura a la vez. Hipótesis: el bloque extra combina información de una región mayor antes del promedio global. El campo receptivo local aumenta de 106×106 a 218×218; es teórico, no una medida de atención ni una explicación causal. El riesgo es aumentar sobreajuste o perder detalle espacial. Se mantienen todos los ajustes de entrenamiento de E05, incluido learning rate inicial 0,001, BCE normal, seed 42, fold 0 y diez épocas. El scheduler puede reducir el learning rate durante el entrenamiento. E06 completó diez épocas en universidad: mejor época 6, ROC-AUC 0,574647 y AP 0,367224, sin superar el resultado observado de E05; más capas no garantizan alcanzar ROC-AUC 0,7. Comparación y criterio de confirmación en [el protocolo](../notes/PROTOCOLO_COMPARACION.md)."),
        ("E07_dropout_050", "E07 · Cuatro bloques y dropout 0,5", "E07_dropout_050.json", ["E07_dropout_050.json"],
         "Variante de regularización de A04/E05: se cambia únicamente el dropout de 0,3 a 0,5 en el vector de 128 características, después del promedio global. Durante entrenamiento anula aleatoriamente componentes de ese vector; en evaluación está desactivado. No elimina convoluciones, no reduce canales y no añade parámetros. Hipótesis: dificultar la dependencia de características concretas puede reducir el sobreajuste; también podría empeorar por regularización excesiva. Se conserva weight decay 0,0001, learning rate inicial 0,001, BCE normal, seed 42, fold 0 y diez épocas. E07 completó diez épocas en universidad: mejor época 1, ROC-AUC 0,595565 y AP 0,368886; no superó el resultado observado de E05. Ver [comparación E05/E07/E08](../notes/REGULARIZACION.md)."),
        ("E08_weight_decay_001", "E08 · Cuatro bloques y weight decay 0,001", "E08_weight_decay_001.json", ["E08_weight_decay_001.json"],
         "La arquitectura es exactamente A04/E05, con dropout 0,3; el dibujo se repite aquí para que el experimento tenga su ficha completa. Solo se cambia el weight decay de AdamW de 0,0001 a 0,001. No es una capa nueva: modifica la actualización de los pesos durante entrenamiento mediante decaimiento desacoplado. Hipótesis: una regularización mayor de los pesos puede reducir el sobreajuste; su utilidad se mide en validación, no por lograr una menor loss de train. Se mantienen learning rate inicial 0,001, BCE normal, seed 42, fold 0 y diez épocas. E08 no combina este cambio con el dropout 0,5 de E07. E08 completó diez épocas en Mac MPS: mejor época 3, ROC-AUC 0,582661 y AP 0,355411. Frente a E05 Mac, el intervalo pareado global incluye cero; no hay mejora clara. Ver [comparación E05/E07/E08](../notes/REGULARIZACION.md)."),
        ("E09_shared_affine", "E09 · CNN de E05 con aumentos solo en train", "E09_shared_affine.json", ["E09_shared_affine.json"],
         "La CNN es exactamente A04/E05: ocho convoluciones, cuatro MaxPool y dropout 0,3. El único cambio es un aumento aleatorio de entrada durante entrenamiento: rotaciones entre −5° y +5° y traslaciones de hasta 3 % por eje (7,68 píxeles). Se aplica una única rejilla bilineal a PRE/EARLY/LATE, con relleno cero; no se normalizan las fases por separado. No se crean pacientes nuevas ni se guardan PNG aumentados. Validación, test e inferencia no aplican el aumento; mantienen la misma carga determinista en [0,1]. El presupuesto sigue siendo diez épocas, LR inicial 0,001, weight decay 0,0001, BCE normal, fold 0 y seed 42. Hipótesis: aprender patrones menos dependientes de la posición. Puede empeorar si pierde detalle; no se garantiza reducir el sobreajuste ni alcanzar ROC-AUC 0,7. E09 completó diez épocas en universidad: mejor época 1, ROC-AUC 0,610282 y AP 0,375773; no superó el resultado observado de E05. Ver [protocolo y lámina local](../notes/AUMENTOS_E09.md)."),
        ("A06_mitad_canales", "A06 · Misma profundidad, mitad de canales", "E10_half_width.json", ["E10_half_width.json"],
         "E10 mantiene el orden Conv–BN–ReLU → MaxPool → Conv–BN–ReLU de E05, pero reduce los canales de 16/32/64/128 a 8/16/32/64. Conserva ocho convoluciones y cuatro MaxPool, las resoluciones espaciales y el campo receptivo local de 106×106. La cabeza recibe 64 valores tras GAP y devuelve un logit. Reduce capacidad sin eliminar etapas de procesamiento: no es E04, que quitaba convoluciones. Hipótesis: menos parámetros pueden limitar la adaptación a peculiaridades de train; también podrían causar infraajuste. El diagnóstico de E05 muestra AUC train/validación 0,587/0,614 en época 1 y 0,870/0,580 en época 10. E10 conserva diez épocas, LR inicial 0,001, weight decay 0,0001, dropout 0,3, BCE normal, fold 0 y seed 42, sin aumentos. E10 completó diez épocas en Mac MPS: mejor época 6, ROC-AUC 0,590625 y AP 0,379061. E05 Mac obtuvo 0,586895: delta +0,003730 con IC pareado global que incluye cero, sin ventaja clara. Ver [protocolo y ejecución en Mac](../notes/ANCHURA_E10.md)."),
        ("E11_lr_0003", "E11 · Learning rate inicial 0,0003", "E11_lr_0003.json", ["E11_lr_0003.json"],
         "La arquitectura es A04/E05: cuatro bloques Conv–BN–ReLU → MaxPool → Conv–BN–ReLU, canales 16/32/64/128 y 294.129 parámetros. El único ajuste modificado respecto a E05 Mac es el learning rate inicial de AdamW: 0,001 → 0,0003. No es una capa ni un cambio en los kernels. Se mantienen diez épocas y el scheduler, que puede reducir la tasa según AUC de validación. Hipótesis: actualizaciones menores podrían estabilizar aprendizaje; también podrían necesitar más épocas. E11 completó diez épocas en Mac MPS: mejor época 2, ROC-AUC 0,550202 y AP 0,344140; 399,15 segundos. No superó el ROC-AUC de E05 Mac (0,586895) y mantuvo señales de sobreajuste. Ver [protocolo de learning rate](../notes/LEARNING_RATE.md)."),
        ("E12_lr_0001", "E12 · Learning rate inicial 0,0001", "E12_lr_0001.json", ["E12_lr_0001.json"],
         "La arquitectura es A04/E05, idéntica a E11. El único ajuste modificado respecto a E05 Mac es el learning rate inicial de AdamW: 0,001 → 0,0001. Se mantienen datos completos, fold 0, seed 42, batch 16, diez épocas, BCE normal, dropout 0,3, weight decay 0,0001 y scheduler. No hay aumentos ni pesos preentrenados. La comparación inicial usa igual presupuesto, pero no supone convergencia idéntica; decidir una prueba más larga por separado si aún mejora al terminar. E12 completó diez épocas en Mac MPS: mejor época 3, ROC-AUC 0,557964 y AP 0,386688; 481,87 segundos. No superó el ROC-AUC de E05 Mac (0,586895). La AP algo mayor no cambia nuestro criterio de selección; el sobreajuste persiste. Ver [protocolo de learning rate](../notes/LEARNING_RATE.md)."),
    ]
    from src.campaigns.run_mac_batch import SPECS
    titles = ("Tres bloques", "Una convolución por bloque", "Kernel 5×5 en red compacta",
              "Batch 32", "Batch 64", "Giro suave ±5°", "Dropout espacial 0,1",
              "Dropout espacial y aumento geométrico", "Learning rate 0,0003", "Pooling tras dos convoluciones")
    registry_rows = json.loads((ROOT / 'experiments/notes/resultados_registrados.json').read_text())['runs']
    for (name, parent, _, hypothesis), title in zip(SPECS, titles):
        registered = next((row for row in registry_rows if row['experiment_id'] == name), None)
        status_note = (f"Completo: ROC-AUC paciente{registered['roc_auc']:.6f}, "
                       f"AP{registered['pr_auc']:.6f}, mejor época{registered['best_epoch']}/{registered['epochs_completed']}. "
                       "Ninguna prueba de esta tanda supera el pico observado E19; no mejora estable demostrada. "
                       if registered and registered['status'] == 'complete' else
                       "Pendiente de entrenamiento científico; el smoke solo verifica funcionamiento, no calidad. ")
        if name == 'E38_spatial_dropout010_mac':
            replicas = [row for row in registry_rows if row.get('series') == 'E19_E38']
            if len(replicas) == 2 and all(row['status'] == 'complete' for row in replicas):
                status_note += ("[Confirmación completa en otra semilla/fold](../CONFIRMACION_E19_E38.md): "
                                "E38 no supera a E19 en ninguna de las tres parejas observadas; "
                                "cerrar esta hipótesis sin relanzar ni alargar. No prueba inferioridad general. ")
            else:
                status_note += "[Confirmación prefijada en otra semilla/fold](../CONFIRMACION_E19_E38.md): dos réplicas pendientes, controles E19 reutilizados sin cargar sus pesos. "
        explanation = (
            f"Referencia: **{parent}**. {hypothesis} "
            "CNN 2D propia, filtros aprendidos y pesos inicializados desde cero, sin transfer learning. "
            "Entrada PRE/EARLY/LATE [3,256,256], restas firmadas dentro del modelo; "
            "878 pacientes de train y 219 de validación, fold0/seed42, BCE normal por corte, "
            "AdamW WD0,001, Plateau, máximo50/paciencia10, MPSfloat32/workers0. "
            "No cambia validación ni preprocesamiento web. " + status_note +
            "[Protocolo y cola de la tanda](../TANDA_MAC_E32_E41.md).")
        specifications.append((name, f"{name[:3]} · {title}", f"{name}.json", [f"{name}.json"], explanation))
    specifications.append((
        "E42_temporal_differences_mac", "E42 · Solo cambios entre fases",
        "E42_temporal_differences_mac.json", ["E42_temporal_differences_mac.json"],
        "Ablación de E19: solo model.input_representation cambia a temporal_differences. "
        "La entrada externa sigue siendo PRE/EARLY/LATE [3,256,256]; dentro del modelo se "
        "calculan EARLY−PRE y LATE−EARLY [2,256,256], con signo y sin clipping ni normalización extra. "
        "PRE se usa en la primera resta, pero no llega como imagen basal separada al extractor. "
        "No es una segmentación ni elimina toda la anatomía. Se pierde información absoluta: "
        "sumar la misma imagen a las tres fases deja las diferencias intactas. "
        "Hipótesis: reducir dependencia de apariencia basal; también puede perder señal útil. "
        "Primera Conv3×3/padding1/stride1 2→16:288 pesos frente a432 de E19; total293985 "
        "frente a294129 (144 menos,0,049%). No atribuir un resultado a una gran reducción de capacidad. "
        "Ocho convoluciones, cuatro pools intermedios, canales16/32/64/128, BatchNorm/ReLU, "
        "GAP/dropout0,3 y un logit, sin dropout espacial. Campo receptivo local106. "
        "Datos completos878/219 pacientes,8759/2186 cortes; fold0/seed42,batch16, "
        "AdamWLR/WD0,001,BCE normal,Plateau,máximo50/paciencia10,MPSfloat32/workers0. "
        "Sin aumentos/Mixup/ganancia/atención ni transfer learning. Desde cero, no cargar E19. "
        "Misma política de inicialización; cambiar forma de la primera conv cambia consumo RNG, "
        "no se prometen pesos comunes iniciales idénticos. Completo: AUC0,607157258,AP0,381855898, "
        "mejor época13/23; sin mejora observada frente a E19. Smoke solo técnico. "
        "[Hipótesis, límites y comando](../SOLO_DIFERENCIAS_E42.md)."))
    specifications.append((
        "E43_spatial_grid2_mac", "E43 · Conservar cuatro regiones espaciales",
        "E43_spatial_grid2_mac.json", ["E43_spatial_grid2_mac.json"],
        "Solo model.global_pool_grid_size cambia 1→2 frente a E19. Misma CNN propia desde cero, "
        "sin pesos ni arquitectura importada: ocho convoluciones y cuatro MaxPool intermedios. "
        "Mapa final128×16×16 → AdaptiveAvgPool2d(2,2) →128×2×2 →512 valores →Dropout0,3 "
        "→Lineal512→1. Cada bin promedia8×8 posiciones, kernel8/stride8/padding0 en este mapa; "
        "RF teórico local106 antes de la cabeza y218 por bin; el logit combina cuatro regiones. "
        "294513 parámetros,384 más que E19 (0,13%); no nueva capa oculta. "
        "Hipótesis: conservar distribución espacial gruesa puede aportar señal que GAP elimina; "
        "también puede aprender posición/artefactos y aumentar sobreajuste. No segmenta ni localiza tumor. "
        "Entrada PRE/EARLY/LATE y diferencias firmadas E19 intactas, NO usa modo E42. "
        "878/219 pacientes, batch16, LR/WD0,001, BCE normal, Plateau,50 épocas/paciencia10, "
        "fold0seed42, MPSfloat32, sin aumentos. Inicialización desde cero con política habitual; "
        "cambiar tamaño del clasificador cambia consumo RNG, no pesos comunes idénticos garantizados. "
        "Completo: AUC0,575302419, mejor época3/13,906,68s; no mejora observada sobre E19. "
        "[Hipótesis y protocolo](../CAMPANA_AUTONOMA_20261006.md)."))
    specifications.append((
        "E44_average_pool_mac", "E44 · Promediar en cada reducción espacial",
        "E44_average_pool_mac.json", ["E44_average_pool_mac.json"],
        "Solo model.downsampling_pool cambia max→average frente a E19. Cuatro bloques "
        "Conv3×3–BatchNorm–ReLU → AvgPool2×2 → Conv3×3–BatchNorm–ReLU, canales16/32/64/128. "
        "Pooling kernel2,stride2,padding0; conserva canales y divide el lado entre2:256→128→64→32→16. "
        "GAP sobre128×16×16 →128valores →Dropout0,3 →Lineal128→1logit. "
        "Mismas ocho convoluciones, RF local106 y294129 parámetros: no ampliar ni reducir capacidad aprendida. "
        "Hipótesis: medias locales podrían conservar señal distribuida y reducir dominancia de picos; "
        "también pueden diluir patrones intensos útiles. No segmenta ni aumenta pacientes. "
        "Mismos pesos iniciales comunes con misma semilla, verificados sin cargar E19; cambia el operador "
        "sin pesos y sin consumir RNG. PRE/EARLY/LATE y representación firmada E19 intactas. "
        "878/219pacientes,8759/2186cortes,fold0seed42,batch16,MPSfloat32,AdamWLR/WD0,001, "
        "BCE normal,Plateau,50épocas/paciencia10,sin aumentos ni Mixup. "
        "Completo: AUC0,593649194/AP0,365635002/Brier0,217082042, mejor17/27,2001,89s. "
        "No mejora observada frenteE19. [Protocolo](../SIGUIENTE_HIPOTESIS_POOLING.md)."))
    specifications.append((
        "E45_sam_mac", "E45 · Misma CNN, entrenamiento SAM",
        "E45_sam_mac.json", ["E45_sam_mac.json"],
        "Misma arquitectura E19: PRE/EARLY/LATE externos y PRE/diferencias firmadas internos; "
        "ocho Conv3×3–BN–ReLU, cuatro MaxPool entre convs, canales16/32/64/128, "
        "GAP/dropout0,3/lineal128→1,294129parámetros y RF convolucional local106. "
        "SOLO training.sam.enabled cambia false→true, rho0,05 prefijado. "
        "Implementación propia L2 sobre AdamW: gradiente del lote, perturbación normalizada de pesos, "
        "segundo gradiente, restauración exacta y una actualización sobre pesos originales. "
        "Reutiliza máscara Dropout y conserva buffers BN solo de primera pasada, no model.eval(). "
        "Hipótesis: reducir sensibilidad de pérdida a pequeñas perturbaciones; no garantía de generalización. "
        "No arquitectura ni pesos importados y no añade pacientes. Mismos pesos iniciales por semilla "
        "sin cargar E19,878/219pacientes,8759/2186cortes,fold0seed42,batch16,LR/WD0,001, "
        "BCE normal,Plateau,50épocas/paciencia10,MPSfloat32, sin aumentos/Mixup. "
        "Mismos pasos de optimizador, dos pasadas por lote: NO presupuesto computacional igual. "
        "Guardar primera train loss, validación habitual sin SAM. Política completa, no aislamiento "
        "de componentes BN/Dropout. Completo MPS: AUC0,588306452/AP0,378226093/Brier0,204071810, "
        "mejor época6/16,1770,20s; no mejora observada frente a E19, sobreajuste persistente. "
        "[Matemática, controles y límites](../PREPARACION_SAM.md)."))
    specifications.append((
        "E46_dilation2_mac", "E46 · Más contexto con los mismos pesos",
        "E46_dilation2_mac.json", ["E46_dilation2_mac.json"],
        "SOLO model.dilation cambia1→2 frente a E19, padding ajustado1→2 en las ocho Conv3×3. "
        "Nueve pesos por pareja de canales toman posiciones de un área5×5: no Conv5×5 densa. "
        "Cuatro bloques Conv–BN–ReLU/MaxPool2×2/Conv–BN–ReLU, canales16/32/64/128; "
        "formas256→128→64→32→16, GAP128/dropout0,3/lineal128→1 intactos. "
        "294129parámetros y mismos pesos iniciales por semilla desde cero, sin cargar E19. "
        "RF convolucional local106→196 antes de GAP; no incluye dependencia global de BN. "
        "Hipótesis de contexto espacial más amplio sin añadir capacidad parametrizada; "
        "riesgo de rejilla/pérdida de detalle, no mejora garantizada ni segmentación. "
        "Datos completos878/219pacientes,8759/2186cortes,fold0seed42,batch16,AdamWLR/WD0,001, "
        "BCE normal,Plateau,50épocas/paciencia10,MPSfloat32,sin aumentos/SAM/Mixup. "
        "No acumular pooling medio ni rejillaE43; mismo criterio prefijado de confirmación. "
        "Completo MPS: AUC0,602923387/AP0,365590189/Brier0,204302443, mejor1/11,790,15s. "
        "No mejora observada frente a E19. [Hipótesis y controles](../SIGUIENTE_HIPOTESIS_DILATACION.md)."))
    specifications.append((
        "E47_factorized_stem_mac", "E47 · Filtros espaciales y mezcla de canales",
        "E47_factorized_stem_mac.json", ["E47_factorized_stem_mac.json"],
        "SOLO model.convolution_type cambia standard→factorized_after_stem frente a E19. "
        "Primera Conv3→16 densa; otras7unidades: DW3×3 por canal→PW1×1 mezcla canales→BN→ReLU. "
        "No BN/ReLU entre DW y PW;15convoluciones reales,8normalizaciones/activaciones,4MaxPool intermedios. "
        "Mismos canales16/32/64/128,formas256→128→64→32→16,RF local106 y GAP128/dropout0,3/lineal128→1. "
        "37201parámetros (36112conv+960BN+129lineal),frente294129E19;87,35%menos, no estrechar mapas. "
        "Kernel denso restringido W[o,i,u,v]=P[o,i]D[i,u,v]; menor expresividad por canal, puede infraajustar. "
        "Hipótesis de capacidad distinta; no garantiza generalización ni equivale a todas las convs densas. "
        "Inicialización prefijada DWfan_in/lineal (no activación intermedia),PW+densa fan_out/ReLU; "
        "consumo RNG cambia, no pesos comunes idénticos garantizados ni aislar reducción de parámetros. "
        "Arquitectura propia de4bloques, sin MobileNet/Xception ni pesos/importaciones de redes externas. "
        "PRE/diferencias firmadas E19,878/219pacientes,fold0seed42,batch16,AdamWLR/WD0,001,BCE normal, "
        "Plateau,50/paciencia10,MPSfloat32. Sin dilatación extra/SAM/EMA/Mixup/aumentos/poolingmedio. "
        "Completo MPS: AUC0,578024194/AP0,374101506/Brier0,284502429, mejor10/20,1365,99s; "
        "no mejora observada frente a E19. [Motivo y controles](../HIPOTESIS_CONVOLUCION_FACTORIZADA.md)."))
    specifications.append((
        "E48_ema_mac", "E48 · Misma CNN, pesos suavizados",
        "E48_ema_mac.json", ["E48_ema_mac.json"],
        "SOLO training.ema.enabled cambia false→true frente a E19; decay0,99 prefijado por paso AdamW. "
        "Copia desde la inicialización propia, sin pesos externos ni cargar E19; promedio exponencial "
        "de parámetros y buffers flotantes BN, contadores enteros copiados. Sin recalibración BN. "
        "Train loss usa pesos entrenables; validación/checkpoint/Plateau/early stopping usan copia EMA en eval. "
        "No comparar esa brecha como si fueran los mismos pesos; LR puede seguir trayectoria distinta. "
        "La sombra no aprende por gradientes ni añade parámetros de arquitectura: CNN E19,294129,8convs,RF106. "
        "model_state_dict guardado corresponde a EMA; estado entrenable/optimizador/promedio separados, no resume. "
        "878/219pacientes,8759/2186cortes,fold0seed42,batch16,LR/WD0,001,BCE normal,Plateau,50/paciencia10,MPSfloat32. "
        "Sin SAM/Mixup/aumentos/dilatación/factoración ni test. Completo: AUC0,623286290/AP0,409816629/Brier0,200916572, mejor4/14,968,80s MPS; no supera E19. "
        "[Política y limitaciones](../HIPOTESIS_PROMEDIO_PESOS.md)."))
    specifications.append((
        "E49_horizontal_flip_mac", "E49 · Espejo horizontal solo en train",
        "E49_horizontal_flip_mac.json", ["E49_horizontal_flip_mac.json"],
        "Solo data.horizontal_flip.enabled cambia false→true sobre E19, probabilidad0,5 prefijada. "
        "Una moneda por tripleta y lectura; inversión del eje W idéntica para PRE/EARLY/LATE. "
        "Sin interpolación, recorte, cambios de intensidad, fases ni etiqueta; no pacientes independientes nuevos. "
        "Validación/test/web sin espejo ni TTA. Hipótesis: reducir dependencia de orientación de presentación, no causa demostrada. "
        "Misma CNN propia294129 parámetros/8convs/RF106 y datos878/219pacientes,8759/2186cortes. "
        "Batch16, AdamW LR/WD0,001, BCE normal, Plateau,50/paciencia10,fold0seed42,MPSfloat32; "
        "sin EMA/SAM/Mixup/rotación/ganancia. Inicialización idéntica con semilla igual, trayectoria RNG posterior distinta. "
        "Completo MPS: AUC0,581149194/AP0,345662984/Brier0,219747186, mejor17/27,2157,19s; no supera E19. [Hipótesis y controles](../HIPOTESIS_ESPEJO_HORIZONTAL.md)."))
    specifications.append((
        "E50_parallel_context_mac", "E50 · Contexto local y amplio en paralelo",
        "E50_parallel_context_mac.json", ["E50_parallel_context_mac.json"],
        "SOLO model.convolution_type standard→parallel_context frente E19. "
        "Primera unidad de cada bloque: dos Conv3×3 paralelas con MISMA entrada, "
        "Cin→Cout/2 cada una; dilation/padding1/1 y2/2, stride1,groups1,sin bias. "
        "Concatenar canales antes de una BN→ReLU; no hay activación dentro de ramas. "
        "MaxPool2×2 entre unidades, segunda Conv3×3d1 normal. 12convs reales,8BN/ReLU,4pools; "
        "294129parámetros, mismos canales/dimensiones, RFconvolucionalmáximo136 vs106E19. "
        "Ramas inicializadas con Kaiming/ReLU y fan_outCoutTOTAL, no mitad; topología/RNG distintos, sin cargar E19. "
        "CNN propia desde cero, no DeepLab/ASPP/ResNet ni pesos importados. "
        "Datos878/219pacientes,8759/2186cortes,fold0seed42,batch16,AdamWLRWD0,001,BCE normal, "
        "Plateau,50/paciencia10,MPSfloat32,sin espejo/EMA/SAM/Mixup/ganancia/rotación. "
        "Completo MPS: AUC0,605040323/AP0,415432039/Brier0,204974081, mejor6/16,2957,75s; no supera E19. "
        "[Motivo y controles](../HIPOTESIS_CONTEXTO_PARALELO.md)."))
    specifications.append((
        "E51_label_smoothing_mac", "E51 · Misma CNN, objetivos suaves solo en train",
        "E51_label_smoothing_mac.json", ["E51_label_smoothing_mac.json"],
        "SOLO training.label_smoothing.enabled false→true sobre E19, epsilon0,1 prefijado. "
        "BCE de train con y_s=0,9*y+0,05: objetivos0,05/0,95 sin cambiar etiquetas originales. "
        "Validación/métricas/test/web con etiquetas binarias reales; inferencia sin suavizado ni clipping. "
        "No Mixup/pseudoetiquetas/EMA/SAM/ponderación ni nuevos aumentos. CNN E19 propia idéntica: "
        "8convs/8BN/ReLU/4MaxPool intermedios/GAP/dropout0,3,294129parámetros,RF106. "
        "878/219pacientes,8759/2186cortes,fold0seed42,batch16,AdamWLRWD0,001,Plateau,50/paciencia10,MPSfloat32. "
        "Train BCE suave y val BCE binaria no tienen el mismo objetivo: mínimo suave teórico0,198515, "
        "no cero, no resultado científico. Hipótesis regularizar confianza excesiva, no mejora garantizada. "
        "CompletoMPS: AUC0,623084677/AP0,407043879/Brier0,201894989, mejor3/13,928,20s; no supera E19. "
        "[Motivo, fórmula y controles](../HIPOTESIS_SUAVIZADO_ETIQUETAS.md)."))
    specifications.append((
        "E52_silu_mac", "E52 · Activación suave SiLU, misma estructura",
        "E52_silu_mac.json", ["E52_silu_mac.json"],
        "SOLO model.activation relu→silu sobre E19: ocho SiLU=x*sigmoid(x) tras BN. "
        "Sigmoid interna NO es probabilidad pCR; un logit final con sigmoid solo al evaluar. "
        "8convs/8BN/4MaxPool intermedios/GAP/dropout0,3,294129parámetros/RF106, mismas formas. "
        "No añade parámetros ni filtros predefinidos; kernels aprendidos desde cero. "
        "Inicializador Kaiming/ReLU anterior RETENIDO para aislar activación y conservar pesos iniciales "
        "idénticos con misma semilla; no afirmar que sea el óptimo para SiLU ni cargar E19. "
        "Datos878/219pacientes,8759/2186cortes,fold0seed42,batch16,AdamWLRWD0,001,BCE normal, "
        "Plateau,50/paciencia10,MPSfloat32; sin suavizado de objetivos/EMA/SAM/Mixup/espejo/ganancia/rotaciones. "
        "CompletoMPS: AUC0,595866935/AP0,347440306/Brier0,203721930, mejor4/14,1055,40s; no supera E19. "
        "[Fórmula y controles](../HIPOTESIS_ACTIVACION_SUAVE.md)."))
    specifications.append((
        "E53_shared_standardization_mac", "E53 · Una escala común para las tres fases",
        "E53_shared_standardization_mac.json", ["E53_shared_standardization_mac.json"],
        "SOLO model.input_normalization none→shared_mean_std sobre E19. Una media/std por tripleta "
        "sobre todos los píxeles y tres fases, antes de PRE/diferencias; varianza poblacional, floorstd1e-6. "
        "Sin fit de estadísticas del dataset, normalización por fase, clipping, RGB ni ImageNet. "
        "Mismo paso dentro de CNN para train/eval/web; no cambiar PNG/cargador. Diferencias firmadas "
        "preservan signo y escala relativa; media/escala absolutas se eliminan y el fondo deja de ser cero: "
        "podría borrar señal útil. RF convolucional106 NO dependencia total: normalización usa toda la tripleta en eval. "
        "CNN294129parámetros/8convs/8BN/ReLU/4MaxPool/GAP/dropout0,3, mismo inicializador/semilla desde cero. "
        "Datos878/219pacientes,8759/2186cortes,fold0seed42,batch16,AdamWLRWD0,001,BCE normal, "
        "Plateau,50/paciencia10,MPSfloat32; sin SiLU/GeM/EMA/SAM/Mixup/labelsmoothing/aumentos nuevos. "
        "COMPLETO en MPS: AUC0,600907258/AP0,383448492/Brier0,206491021, "
        "mejor6/16,1075,90s; no supera E19. No adoptar ni relanzar. "
        "[Hipótesis y cierre](../HIPOTESIS_NORMALIZACION_CONJUNTA.md)."))
    specifications.append((
        "E54_fixed_gem_mac", "E54 · Pooling GeM con exponente tres fijo",
        "E54_fixed_gem_mac.json", ["E54_fixed_gem_mac.json"],
        "SOLO model.global_pooling average→power_mean sobre E19. Por mapa no negativo tras ReLU, "
        "(mean(x**3))**(1/3), p3 FIJO/floor1e-6, escalado por máximo para evitar overflow. "
        "128×16×16→128×1×1 como GAP, sin parámetros nuevos/exponente aprendible, sin cambiar agregación por paciente. "
        "Favorece respuestas intensas respecto a promedio, puede amplificar ruido; no máscara/segmentación/causalidad. "
        "CNN propia294129parámetros/8convs/8BN/ReLU/4MaxPool/RF106/dropout0,3, misma inicialización con semilla. "
        "Datos878/219pacientes,8759/2186cortes,fold0seed42,batch16,AdamWLRWD0,001,BCE normal, "
        "Plateau,50/paciencia10,MPSfloat32; sin normalizaciónE53/SiLU/EMA/SAM/Mixup/labelsmoothing/aumentos nuevos. "
        "Solo fórmula matemática propia, no redes/pesos/código preentrenado del artículo de retrieval. "
        "Completo MPS: AUC0,594657258/AP0,372722937/Brier0,212596906, "
        "mejor6/16,1084,61s; no supera E19. No adoptar ni relanzar. "
        "[Fórmula y controles](../HIPOTESIS_POOLING_GENERALIZADO.md)."))
    specifications.append((
        "E55_lr0001_mac", "E55 · Actualizaciones diez veces más pequeñas",
        "E55_lr0001_mac.json", ["E55_lr0001_mac.json"],
        "SOLO training.learning_rate 0,001→0,0001 frente a E19. Misma CNN propia desde cero: "
        "294129 parámetros,8convs/8BN/ReLU/4MaxPool intermedios/RF106/GAP/dropout0,3/cabeza128→1. "
        "PRE/diferencias firmadas; sin GeM/EMA/normalización conjunta/aumentos/suavizado. "
        "Fold0seed42,878/219 pacientes,8759/2186 cortes,batch16,AdamW WD0,001,BCE normal, "
        "Plateau factor0,5/paciencia3,máximo50/paciencia10,MPSfloat32. Solo LR inicial cambia; "
        "scheduler puede cambiar trayectoria según validación. E11/E12 usaban otra base; "
        "E40 probó0,0003 sobre E19. Hipótesis actualizaciones más suaves, no elimina "
        "sobreajuste ni garantiza mejora; presupuesto idéntico puede detener un aprendizaje lento. "
        "Pendiente científico. [Protocolo](../LR_E55.md)."))
    for folder, title, config, configs, explanation in specifications:
        loaded = load_experiment_config(ROOT / "configs/experiments" / config)
        catalog.append((folder, title, BreastPCRNet(loaded.model), configs, explanation))
    if only is not None and only not in {entry[0] for entry in catalog}:
        raise ValueError(f"Arquitectura desconocida: {only}")
    for folder, title, model, configs, explanation in catalog:
        if only is not None and folder != only:
            continue
        directory = ROOT / "experiments" / folder
        directory.mkdir(parents=True, exist_ok=True)
        steps = trace(model)
        draw(steps, title, "La resolución se reduce y los canales aumentan; cada operación se muestra en su orden real.",
             trainable_parameter_count(model), directory / "arquitectura")
        write_sheet(directory, title, model, explanation, configs, steps)
        if any(isinstance(layer, ParallelContextConv) for layer in model.modules()):
            draw_parallel_context_branches(model,directory)
        if isinstance(getattr(model, 'global_pool', None), GatedSpatialPool):
            spatial_shape = next(s['shape'][-2:] for s in reversed(steps) if s['kind']=='conv')
            draw_attention_head(model.global_pool, directory, spatial_shape)
        print(f"Documentado: {folder}")
    if only is None:
        for folder in feature_folders:
            document_patient_feature_architecture(folder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", help="Regenerar solo una carpeta, por ejemplo A05_cinco_bloques")
    main(parser.parse_args().only)
