"""Fichas desde modelos reales de la campaña; sin cargar imágenes."""
from pathlib import Path
import json
import torch

CARDS={
 'C19_compacta_4conv':('C19_compacta_normal','Cuatro bloques8/16/32/64, una Conv3×3→BN→ReLU→MaxPool2×2 cada uno; GAP64→dropout0,3→Linear64→1. Reduce solo canales frenteC18. Hipótesis: menor capacidad limita memorización. Contraste C19normal frenteC18normal y ponderadas por separado.'),
 'C21_pooling_local':('C21_pooling_local_normal','Misma CNN compactaC19. Único cambio: GAP→promedio generalizado espacial GeM p3fijo sobre activaciones ReLU no negativas. Sin parámetros añadidos. Hipótesis: evitar diluir respuestas locales. Puede amplificar ruido; no mejora demostrada. ContrasteC21normal frenteC19normal. No agrega pacientes ni aprende máscaras.'),
 'C22_groupnorm':('C22_groupnorm_normal','Misma CNN97.809parámetrosC18. Único cambio: BatchNorm→GroupNorm4 en cuatro bloques. Estadísticas por muestra/grupo en train/eval, sin medias móviles. Hipótesis: reducir saltos de evaluación entre épocas. Puede perder señal de intensidad; no mejora demostrada. ContrasteC22normal frenteC18normal.')}


def document(name):
 from src.campaigns.campaign_20261016 import ROOT,make_model
 from src.reporting.document_architectures import trace,draw,write_sheet
 torch.set_num_threads(2)
 experiment,reason=CARDS[name];config_path=ROOT/(experiment+'.json')
 model=make_model(json.loads(config_path.read_text()));steps=trace(model)
 directory=Path('experiments')/name;directory.mkdir(exist_ok=True)
 params=sum(p.numel() for p in model.parameters())
 explanation=reason+' Entrada externa PRE/EARLY/LATE256×256, float32PNG/255. Adam LR0,0003/WD0,0001/flips sincronizadosH/V0,5. Resolución tras bloques128/64/32/16. Convkernel3/padding1/stride1/sinbias; poolingkernel2/stride2/padding0. Campo receptivo local nominal4/10/22/46; normalización y pooling global añaden dependencia global. Toda inicialización desde cero. Cada entrada de capa coincide con la salida de la fila anterior. Ficha compartida si solo cambia pérdida. Evaluación principalmente por paciente, media de probabilidades por corte.'
 draw(steps,name,'CNN propia · contraste controlado · sin clínica',params,directory/'arquitectura')
 write_sheet(directory,name,model,explanation,[],steps)
 path=directory/'README.md';text=path.read_text().replace('- [CNN mínima](../../src/models/model.py), utilizada por la prueba de memorización.',f'- [Configuración](../../{config_path.as_posix()})\n- [Protocolo](../../{ROOT.as_posix()}/README.md)')
 path.write_text(text)
 assert sum(s['parameters'] for s in steps)==params
 return params


if __name__=='__main__':
 for name in CARDS:print(name,document(name))
