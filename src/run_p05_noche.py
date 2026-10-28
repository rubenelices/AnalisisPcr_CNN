"""P05 en la GPU de la universidad: P02 con tasa 0,0001, BCE normal, 5 folds.

Uso (desde la raíz del repo, entorno con PyTorch ROCm/CUDA):
    python -m src.run_p05_noche --check    # solo comprobaciones (≈1 min)
    python -m src.run_p05_noche            # smoke + 5 fits + paquete

Mismas comprobaciones que la noche de P02 (protocolo N01 con el SHA del Mac).
Hipótesis y criterios, escritos antes: nuevo_enfoque/P05_LR_BAJO.md. Nunca evalúa test.
"""
import argparse
from datetime import datetime
import tarfile
from pathlib import Path

from src.new_approach_protocol import ROOT
from src.run_p02_noche import check, log, step

P05 = Path('reports/new_approach/P05_lr_bajo')
P05_WEIGHTS = Path('checkpoints/new_approach/P05_lr_bajo')


def package(root, logfile):
    out = root/f'resultados_p05_gpu_{datetime.now():%Y%m%d_%H%M}.tar.gz'
    with tarfile.open(out, 'w:gz') as tar:
        for mode in ('runs_cuda', 'smoke_cuda'):
            folder = root/P05/mode
            if folder.is_dir():
                tar.add(folder, arcname=str(P05/mode))
        for best in sorted((root/P05_WEIGHTS/'runs_cuda').glob('*/best.pt')):
            tar.add(best, arcname=str(best.relative_to(root)))
        tar.add(logfile, arcname=str(logfile.relative_to(root)))
    log(logfile, f'Paquete listo: {out.name} ({out.stat().st_size/1e6:.1f} MB). Cópialo al Mac.')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--check', action='store_true', help='solo comprobaciones, sin entrenar')
    args = parser.parse_args()
    root = Path(ROOT)
    (root/P05).mkdir(parents=True, exist_ok=True)
    logfile = root/P05/'noche_gpu.log'
    check(root, logfile)
    if args.check:
        log(logfile, 'Comprobaciones superadas. Lanza sin --check para entrenar.')
        return
    base = ['python', '-m', 'src.train_p01']
    gpu = ['--variant', 'p05', '--loss', 'normal', '--device', 'cuda']
    if not step(base + ['smoke', '--fold', '0'] + gpu, logfile):
        log(logfile, 'Smoke falló: no se lanzan los 5 folds.')
        return
    ok = [step(base + ['run', '--fold', str(fold)] + gpu, logfile) for fold in range(5)]
    log(logfile, f'Completadas {sum(ok)} de 5 ejecuciones.')
    package(root, logfile)


if __name__ == '__main__':
    main()
