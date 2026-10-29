# Despliegue

La web completa (interfaz React y API de inferencia) corre en un único contenedor en **Hugging Face Spaces**. El código se descarga de este repositorio de GitHub y los pesos de E57 del repositorio de modelo [RubenElices/BreastDCEDL](https://huggingface.co/RubenElices/BreastDCEDL). Al cargar, el servidor comprueba que el SHA-256 de los pesos coincide con el de [model/protocolo_e57.json](../model/protocolo_e57.json); si no coincide, no arranca el modelo.

## Hugging Face Space

1. En Hugging Face: **New Space** → SDK **Docker** → plantilla en blanco → hardware CPU básico (gratis).
2. Sube al Space los dos ficheros de [huggingface/](huggingface/): `Dockerfile` y `README.md` (este último lleva la cabecera que configura el Space).
3. El Space se construye solo (unos minutos la primera vez). La web queda en `https://<usuario>-<space>.hf.space`.

Para publicar cambios del código, haz push a GitHub y pulsa **Factory rebuild** en los ajustes del Space. Un Space gratuito se duerme tras un rato sin uso: la primera visita tarda en despertarlo, así que conviene abrirlo unos minutos antes de una demostración.

## Interfaz en otro dominio (opcional)

Si la interfaz se publica aparte (Vercel, GitHub Pages…), compílala apuntando al Space y autoriza ese origen en el servidor:

```bash
VITE_API_URL=https://<usuario>-<space>.hf.space npm --prefix frontend run build
```

En el Space, añade la variable `BREAST_PCR_ALLOWED_ORIGINS=https://<tu-dominio>` (Settings → Variables).

## Variables del servidor

| Variable | Uso |
|---|---|
| `BREAST_PCR_HF_REPO` | Repositorio de Hugging Face del que descargar `best.pt` si no está en `checkpoints/` |
| `BREAST_PCR_CHECKPOINT`, `BREAST_PCR_DECISION` | Rutas a otro checkpoint y su protocolo |
| `BREAST_PCR_ALLOWED_ORIGINS` | Orígenes autorizados a llamar a `/api/` desde otro dominio |

La comprobación de la etiqueta real de un corte (`/api/etiqueta`) necesita el CSV local de BreastDCEDL; en el despliegue no está y la web simplemente no muestra el acierto o fallo.
