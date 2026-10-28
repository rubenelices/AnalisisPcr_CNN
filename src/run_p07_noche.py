"""P07 en la GPU de la universidad: P02 normal con semillas 17 y 123, 5 folds cada una.

Uso (desde la raíz del repo, entorno con PyTorch ROCm/CUDA):
    python -m src.run_p07_noche --check    # solo comprobaciones (≈1 min)
    python -m src.run_p07_noche            # 2 smokes + 10 fits + paquete

La semilla 42 ya existe (P02). Mismas comprobaciones que la noche de P02 (protocolo
N01 con el SHA del Mac). Hipótesis y criterios: nuevo_enfoque/P07_SEMILLAS.md. Nunca evalúa test.
"""
import argparse
from datetime import datetime
import tarfile
from pathlib import Path

from src.new_approach_protocol import ROOT
from src.run_p02_noche import P02, P02_WEIGHTS, check, log, step

NUEVAS = (17, 123)


def package(root, logfile):
    out = root/f'resultados_p07_gpu_{datetime.now():%Y%m%d_%H%M}.tar.gz'
    with tarfile.open(out, 'w:gz') as tar:
        for mode in ('runs_cuda', 'smoke_cuda'):
            for seed in NUEVAS:
                for folder in sorted((root/P02/mode).glob(f'normal_fold_*_seed_{seed}')):
                    tar.add(folder, arcname=str(folder.relative_to(root)))
        for seed in NUEVAS:
            for best in sorted((root/P02_WEIGHTS/'runs_cuda').glob(f'normal_fold_*_seed_{seed}/best.pt')):
                tar.add(best, arcname=str(best.relative_to(root)))
        tar.add(logfile, arcname=str(logfile.relative_to(root)))
    log(logfile, f'Paquete listo: {out.name} ({out.stat().st_size/1e6:.1f} MB). Cópialo al Mac.')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--check', action='store_true', help='solo comprobaciones, sin entrenar')
    args = parser.parse_args()
    root = Path(ROOT)
    (root/P02).mkdir(parents=True, exist_ok=True)
    logfile = root/P02/'noche_p07_gpu.log'
    check(root, logfile)
    if args.check:
        log(logfile, 'Comprobaciones superadas. Lanza sin --check para entrenar.')
        return
    base = ['python', '-m', 'src.train_p01']
    ok = {}
    for seed in NUEVAS:
        gpu = ['--variant', 'p02', '--loss', 'normal', '--device', 'cuda', '--seed', str(seed)]
        if not step(base + ['smoke', '--fold', '0'] + gpu, logfile):
            log(logfile, f'Smoke semilla {seed} falló: se omiten sus 5 folds.')
            continue
        for fold in range(5):
            ok[(seed, fold)] = step(base + ['run', '--fold', str(fold)] + gpu, logfile)
    log(logfile, f'Completadas {sum(ok.values())} de 10 ejecuciones.')
    package(root, logfile)


if __name__ == '__main__':
    main()
