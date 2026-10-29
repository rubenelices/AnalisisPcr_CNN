"""Arranque en Hugging Face Spaces (SDK Gradio, hardware ZeroGPU gratuito).

Sirve la API Flask y la interfaz en el puerto 7860; no usa Gradio ni la GPU. Los pesos de E57
se descargan del repositorio de modelo y se verifican con el SHA-256 de model/protocolo_e57.json.
"""
import os

import spaces  # antes que torch, como pide ZeroGPU
import torch


@spaces.GPU
def _sin_gpu() -> None:
    """ZeroGPU exige al menos una función @spaces.GPU al arrancar; la web no la llama nunca."""


# ZeroGPU confirma el arranque desde Blocks.launch de Gradio; como no lanzamos Gradio,
# hacemos ese aviso a mano. Fuera de ZeroGPU (en local) no existe y no se llama.
if hasattr(spaces, "zero") and hasattr(spaces.zero, "startup"):
    spaces.zero.startup()


os.environ.setdefault("BREAST_PCR_HF_REPO", "RubenElices/BreastDCEDL")

# En ZeroGPU, fuera de @spaces.GPU, PyTorch emula CUDA: la elección automática diría "cuda"
# y la inferencia fallaría. Un corte de E57 se predice en milisegundos en CPU.
import src.web.inference as inference  # noqa: E402

inference.choose_device = lambda _nombre="auto": torch.device("cpu")

from waitress import serve  # noqa: E402

from app import app  # noqa: E402

if __name__ == "__main__":
    serve(app, host="0.0.0.0", port=int(os.environ.get("PORT", 7860)), threads=4)
