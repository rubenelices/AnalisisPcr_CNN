# Despliegue

La web completa (interfaz React y API de inferencia en Flask) corre en un **Hugging Face Space**: [rubenelices-pcr-demo.hf.space](https://rubenelices-pcr-demo.hf.space). Los pesos de E57 se descargan al arrancar del repositorio de modelo [RubenElices/BreastDCEDL](https://huggingface.co/RubenElices/BreastDCEDL), y el servidor comprueba que su SHA-256 coincide con el de [model/protocolo_e57.json](../model/protocolo_e57.json); si no coincide, no carga el modelo.

## Cómo está montado

El Space usa el SDK Gradio con el hardware gratuito ZeroGPU, pero no usa ni Gradio ni la GPU: [huggingface/serve.py](huggingface/serve.py) arranca el mismo `app.py` de este repositorio con `waitress` en el puerto 7860. La GPU no hace falta: un corte se predice en milisegundos en CPU.

- **Sin GPU.** ZeroGPU exige declarar una función `@spaces.GPU` y confirma el arranque desde Gradio. `serve.py` declara una vacía (no se llama nunca) y hace esa confirmación a mano.
- **Inferencia en CPU forzada.** En ZeroGPU, PyTorch emula CUDA fuera de esas funciones, así que `serve.py` fuerza la CPU.
- **Mismo código y mismo preprocesado** que en local y en el entrenamiento: el Space lleva una copia de `app.py`, `src/`, `model/`, `web/` y la interfaz compilada.

## Publicar una versión nueva

```bash
npm --prefix frontend ci && npm --prefix frontend run build
python deploy/huggingface/preparar_space.py --subir RubenElices/pcr-demo
```

El script monta la carpeta del Space en `local/hf_space/` (fuera de Git) y la sube; hace falta haber iniciado sesión con `hf auth login` y un token con permiso de escritura. El Space se reconstruye solo en uno o dos minutos. Sin `--subir`, solo monta la carpeta, para revisarla o probarla.

Un Space gratuito se duerme tras un rato sin uso: la primera visita tarda en despertarlo, así que conviene abrirlo unos minutos antes de una demostración.

## Crear el Space desde cero

1. En Hugging Face, **New Space** → SDK **Gradio** → plantilla **Blank** → hardware **ZeroGPU** → público.
2. Ejecuta el script anterior con `--subir <usuario>/<space>`. El `README.md` de [huggingface/](huggingface/) lleva la cabecera que configura el Space (Python 3.12 y `serve.py` como fichero de arranque).

## Interfaz en otro dominio (opcional)

Si la interfaz se publica aparte, se compila apuntando al Space y se autoriza ese origen en el servidor:

```bash
VITE_API_URL=https://rubenelices-pcr-demo.hf.space npm --prefix frontend run build
```

En el Space, añade la variable `BREAST_PCR_ALLOWED_ORIGINS=https://<tu-dominio>` (Settings → Variables).

## Variables del servidor

| Variable | Uso |
|---|---|
| `BREAST_PCR_HF_REPO` | Repositorio de Hugging Face del que descargar `best.pt` si no está en `checkpoints/` (el Space usa `RubenElices/BreastDCEDL`) |
| `BREAST_PCR_CHECKPOINT`, `BREAST_PCR_DECISION` | Rutas a otro checkpoint y su protocolo |
| `BREAST_PCR_ALLOWED_ORIGINS` | Orígenes autorizados a llamar a `/api/` desde otro dominio |

La comprobación de la etiqueta real de un corte (`/api/etiqueta`) necesita el CSV local de BreastDCEDL; en el despliegue no está y la web simplemente no muestra el acierto o fallo en Pruebas. Los ejemplos del Laboratorio sí la muestran, porque su etiqueta pública del train viaja con ellos.
