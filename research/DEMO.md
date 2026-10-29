# Web de la entrega

[Inicio](README.md) · [Plan](PLAN.md).

La web tiene dos partes:

- **Servidor (`app.py`, Python/Flask):** carga el modelo una vez, valida los tres PNG con el mismo preprocesado que el entrenamiento (`src.pipeline.data.load_dce_triplet`) y predice con `src.web.inference`. Endpoints: `/api/predict` (las tres fases), `/api/occlusion` (mapa de oclusión del mismo corte, `src/web/occlusion_map.py`) y `/api/modelo` (ficha del modelo: versión, checksum, umbral, dispositivo, parámetros).
- **Interfaz (`frontend/`, React + Vite):** portada, cabecera de mando (subir fases, ejemplos, estado del modelo, último resultado) y página de predicción. Toda la inferencia ocurre en el servidor; la interfaz solo valida para avisar antes y muestra resultados.

## Arrancar en local

Desde la raíz del proyecto. Primero compilar la interfaz (solo cuando cambie):

```sh
npm --prefix frontend run build
```

Después, el servidor. `python app.py` carga por defecto el modelo de entrega (E57 y su protocolo de umbral fijado, `DEFAULT_MODEL` en `app.py`):

```sh
python app.py
```

Para probar otro checkpoint sin tocar el código: `BREAST_PCR_CHECKPOINT=<ruta> BREAST_PCR_DECISION=<protocolo> python app.py`. Si cambia el modelo entregado, se actualiza `DEFAULT_MODEL`.

Abrir http://127.0.0.1:8000. La demo HTML anterior sigue en http://127.0.0.1:8000/demo.

**Para trabajar en el diseño** con recarga automática: con `app.py` en marcha, arrancar además `node frontend/node_modules/vite/bin/vite.js frontend --port 5173` y abrir http://127.0.0.1:5173 (las llamadas a `/api` se redirigen al servidor de Python).

## Comprobado (21 de octubre)

- La web y el pipeline de evaluación dan la misma probabilidad: corte `ISPY1_1001_z016`, 0,336232 en ambos.
- Ejemplos y subidas pasan por la misma validación del servidor; los errores del servidor se muestran en el panel de errores sin perder el corte anterior.
- Tests del servidor: `python -m pytest -q tests/test_app.py` (incluye ficha del modelo, servidor sin modelo y servir la interfaz compilada).

La interfaz predice **un corte**, como en la defensa. Los resultados científicos por paciente promedian las probabilidades de sus cortes. Las imágenes de ejemplo (`frontend/public/examples/`) son de train público (BreastDCEDL, CC BY-NC 4.0) y no se versionan.

Secciones (23 de octubre): **Inicio** (el trabajo, pipeline y recorrido de 10 pasos con scroll), **Predicción** (un corte en detalle), **Modelo** (arquitectura interactiva, explicación y gráficas del entrenamiento de E57 con su explicación), **Resultados** (test, formas de medir, validación cruzada normal/ponderada, cohortes, decisión) y **Pruebas** (para la defensa: se sueltan las fases de uno o varios cortes, o una carpeta, y se agrupan por nombre `<paciente>_<corte>_PRE|EARLY|LATE.png`; tabla copiable o descargable).

Las gráficas se dibujan en la web con los datos de `python -m src.web.web_graficas` y las cifras salen de `python -m src.web.web_resultados`; después, `npm --prefix frontend run build`.

**Resultados** incluye la matriz de confusión conjunta de P02 normal (5 folds, 1.097 pacientes; `matriz_conjunta` en `src/web/web_resultados.py`, desde `predictions_outer_patient.csv` y el umbral de cada `summary.json`). **Pruebas** marca cada corte como Correcta o Incorrecta comparando la clase predicha con el pCR de `breastdcedl/metadata/samples.csv` (`/api/etiqueta/<corte>`, solo lectura; si el corte no está en el CSV se indica). Hay que reiniciar `app.py` para que exista ese endpoint.

Pendiente: despliegue público.

Uso educativo, sin validez clínica.
