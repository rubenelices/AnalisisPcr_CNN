"""Noche en la GPU de la universidad: P02 normal y ponderada, 5 folds cada una.

Uso (desde la raíz del repo, entorno con PyTorch ROCm/CUDA):
    python -m src.campaigns.run_p02_noche            # todo: comprobaciones, smokes, 10 fits, paquete
    python -m src.campaigns.run_p02_noche --check    # solo comprobaciones (≈1 min), para lanzar tranquilo

Antes de entrenar nada exige que el protocolo N01 regenerado aquí tenga el MISMO
SHA que en el Mac: así se demuestra que código, metadatos y particiones son
idénticos. Cada fit corre en un subproceso: si uno falla, la noche continúa con
el siguiente. Al final empaqueta resultados y best.pt en un .tar.gz para traerlo.
Nunca evalúa test.
"""
import argparse
from datetime import datetime
import subprocess
import sys
import tarfile
from pathlib import Path

import torch

from src.training.new_approach_protocol import ROOT, TECH as N01_TECH, prepare, sha

EXPECTED_MANIFEST_SHA = '827fef1dfb6539431ae038ed7f97bd9bfd1596db64447727b5cb60216662edb4'
P02 = Path('reports/new_approach/P02_aumento_geometrico')
P02_WEIGHTS = Path('checkpoints/new_approach/P02_aumento_geometrico')


def log(path, message):
    line = f'[{datetime.now():%Y-%m-%d %H:%M:%S}] {message}'
    print(line, flush=True)
    with open(path, 'a') as stream:
        stream.write(line + '\n')


def check(root, logfile):
    if not torch.cuda.is_available():
        raise SystemExit('GPU no disponible (torch.cuda.is_available() es False). No se lanza nada.')
    name = torch.cuda.get_device_name(0)
    log(logfile, f'GPU: {name} · torch {torch.__version__} · HIP {torch.version.hip} · CUDA {torch.version.cuda}')
    if not (root/'breastdcedl/metadata/samples.csv').is_file():
        raise SystemExit('Falta breastdcedl/ en este ordenador.')
    prepare(root)  # crea el manifiesto si no existe; si existe, exige que coincida
    got = sha(root/N01_TECH/'protocol/manifest.json')
    if got != EXPECTED_MANIFEST_SHA:
        raise SystemExit(f'Protocolo distinto al del Mac ({got[:12]}… ≠ {EXPECTED_MANIFEST_SHA[:12]}…): '
                         'código, metadatos o particiones no coinciden. No se lanza nada.')
    log(logfile, 'Protocolo N01 idéntico al del Mac (SHA 827fef1d…): mismas particiones, configuración y datos.')


def step(cmd, logfile):
    log(logfile, 'INICIO ' + ' '.join(cmd[2:]))
    with open(logfile, 'a') as stream:
        code = subprocess.run([sys.executable, *cmd[1:]], stdout=stream, stderr=subprocess.STDOUT, cwd=ROOT).returncode
    log(logfile, ('OK ' if code == 0 else f'FALLO (código {code}) ') + ' '.join(cmd[2:]))
    return code == 0


def package(root, logfile):
    out = root/f'resultados_p02_gpu_{datetime.now():%Y%m%d_%H%M}.tar.gz'
    with tarfile.open(out, 'w:gz') as tar:
        for mode in ('runs_cuda', 'smoke_cuda'):
            folder = root/P02/mode
            if folder.is_dir():
                tar.add(folder, arcname=str(P02/mode))
        for best in sorted((root/P02_WEIGHTS/'runs_cuda').glob('*/best.pt')):
            tar.add(best, arcname=str(best.relative_to(root)))
        tar.add(logfile, arcname=str(logfile.relative_to(root)))
    log(logfile, f'Paquete listo: {out.name} ({out.stat().st_size/1e6:.1f} MB). Cópialo al Mac.')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--check', action='store_true', help='solo comprobaciones, sin entrenar')
    args = parser.parse_args()
    root = Path(ROOT)
    (root/P02).mkdir(parents=True, exist_ok=True)
    logfile = root/P02/'noche_gpu.log'
    check(root, logfile)
    if args.check:
        log(logfile, 'Comprobaciones superadas. Lanza sin --check para entrenar.')
        return
    base = ['python', '-m', 'src.training.train_p01']
    gpu = ['--variant', 'p02', '--device', 'cuda']
    ok = {}
    for loss in ('normal', 'weighted'):
        if not step(base + ['smoke', '--loss', loss, '--fold', '0'] + gpu, logfile):
            log(logfile, f'Smoke {loss} falló: se omiten sus 5 folds.')
            continue
        for fold in range(5):
            ok[(loss, fold)] = step(base + ['run', '--loss', loss, '--fold', str(fold)] + gpu, logfile)
    log(logfile, f'Completadas {sum(ok.values())} de 10 ejecuciones.')
    package(root, logfile)


if __name__ == '__main__':
    main()
