"""Monta la carpeta que se sube al Space de Hugging Face y, si se pide, la sube.

El Space (SDK Gradio, hardware ZeroGPU gratuito) no descarga el código: lleva una copia de lo
necesario para servir la web. Este script la arma a partir del repositorio:

    npm --prefix frontend ci && npm --prefix frontend run build
    python deploy/huggingface/preparar_space.py                       # solo monta local/hf_space/
    python deploy/huggingface/preparar_space.py --subir RubenElices/pcr-demo

Copia app.py, los .py de src/, model/ (protocolo congelado), web/ (demo antigua) y la interfaz
compilada, más serve.py, requirements.txt y README.md de esta carpeta. No copia pesos: el servidor
descarga best.pt del repositorio de modelo y comprueba su SHA-256. Subir necesita haber iniciado
sesión con `hf auth login` y un token con permiso de escritura.
"""

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
AQUI = Path(__file__).resolve().parent
DESTINO = ROOT/'local'/'hf_space'
FICHEROS_SPACE = ('serve.py', 'requirements.txt', 'README.md')


def montar(destino=DESTINO):
    dist = ROOT/'frontend'/'dist'
    if not (dist/'index.html').is_file():
        raise SystemExit('Falta la interfaz compilada: npm --prefix frontend run build')
    destino = Path(destino)
    if destino.exists():
        shutil.rmtree(destino)
    destino.mkdir(parents=True)
    shutil.copy2(ROOT/'app.py', destino/'app.py')
    for py in (ROOT/'src').rglob('*.py'):
        objetivo = destino/py.relative_to(ROOT)
        objetivo.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(py, objetivo)
    ignorar = shutil.ignore_patterns('.DS_Store', '__pycache__')
    for carpeta in ('model', 'web'):
        shutil.copytree(ROOT/carpeta, destino/carpeta, ignore=ignorar)
    shutil.copytree(dist, destino/'frontend'/'dist', ignore=ignorar)
    for nombre in FICHEROS_SPACE:
        shutil.copy2(AQUI/nombre, destino/nombre)
    return destino


def subir(destino, repo):
    from huggingface_hub import HfApi
    # Los ficheros compilados cambian de nombre en cada build: se borran los antiguos del Space.
    return HfApi().upload_folder(repo_id=repo, repo_type='space', folder_path=str(destino),
                                 delete_patterns=['frontend/dist/assets/*'], commit_message='Update web demo')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--destino', default=DESTINO, type=Path)
    parser.add_argument('--subir', metavar='USUARIO/SPACE', help='Space al que subir la carpeta montada')
    args = parser.parse_args()
    carpeta = montar(args.destino)
    print(f'Montado en {carpeta} ({sum(1 for p in carpeta.rglob("*") if p.is_file())} ficheros)')
    if args.subir:
        print(subir(carpeta, args.subir))
