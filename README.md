# Predicción de pCR con DCE-MRI y una CNN 2D desde cero

Proyecto de Aprendizaje Automático (Universidad Alfonso X el Sabio, 2026–2027). Estima la probabilidad de **respuesta patológica completa (pCR)** a la quimioterapia neoadyuvante en cáncer de mama a partir de las tres fases de una DCE-MRI previa al tratamiento (PRE, EARLY y LATE), con una CNN 2D diseñada y entrenada desde cero en PyTorch, sin transfer learning ni pesos preentrenados.

**Pesos del modelo:** [huggingface.co/RubenElices/BreastDCEDL](https://huggingface.co/RubenElices/BreastDCEDL) · **Resultados:** [nuevo_enfoque/RESULTADOS.ipynb](nuevo_enfoque/RESULTADOS.ipynb) · **Presentación:** [docs/presentacion](docs/presentacion/presentacion.pdf)

## Resultado

El modelo entregado es **E57**: 298.161 parámetros, entrada por corte `[3, 256, 256]` (fases temporales, no RGB), media de las probabilidades de los cortes por paciente y umbral 0,325 fijado antes de abrir el test.

| Evaluación | AUC por paciente |
|---|---|
| Test público (176 pacientes, evaluado una sola vez) | **0,573** [IC95 0,475–0,671] |
| Validación cruzada de la receta en 5 folds (1.097 pacientes, predicciones fuera de fold) | 0,507 |
| Solo la cohorte de origen (línea base) | 0,523 |

La señal que la red extrae de la imagen no supera de forma demostrable a saber de qué cohorte procede la paciente. Las hipótesis probadas para mejorarla (recetas de entrenamiento, aumento geométrico, recortes, realce relativo, ensembles de semillas) tienen sus criterios escritos antes de ejecutar y sus resultados, también los negativos, en [nuevo_enfoque/](nuevo_enfoque/README.md).

## Estructura

| Ruta | Contenido |
|---|---|
| `src/` | Código: datos y preprocesado, arquitecturas, entrenamiento, evaluación, análisis estadísticos, inferencia y generación de resultados |
| `tests/` | Tests de software con datos sintéticos |
| `configs/` | Configuraciones de los experimentos (E57, N01, P01, P02…) |
| `nuevo_enfoque/` | Protocolo, hipótesis y decisiones de cada experimento, y el notebook de resultados |
| `E57/` | Dossier del modelo entregado: 5 folds, test y gráficas |
| `experimentos/` | Fichas de arquitectura con diagramas de volúmenes y capas reales |
| `experimentos_equilibrados/`, `experimentos_datos/` | Repeticiones de E57 en otros folds y referencia con edad/menopausia |
| `app.py`, `frontend/` | Demo web: API Flask de inferencia e interfaz React |
| `web/` | Demo HTML anterior (servida en `/demo`) |
| `modelo/` | Protocolo congelado del modelo entregado (umbral, agregación y SHA-256 del checkpoint) |
| `deploy/` | Despliegue en Hugging Face Spaces |
| `docs/` | Presentación y guion de defensa |

## Uso

```bash
pip install -r requirements.txt
python -m pytest -q tests
```

Los datos de BreastDCEDL no se incluyen: se descargan del sitio oficial y se colocan en `breastdcedl/`. Los checkpoints, informes generados (`reports/`) y pesos quedan fuera del repositorio.

```bash
python -m src.train_new_approach run --loss normal --fold 0 --device mps   # receta de E57 con el protocolo N01
python -m src.cuaderno_resultados                                          # regenera nuevo_enfoque/RESULTADOS.ipynb
```

### Demo web en local

```bash
npm --prefix frontend ci && npm --prefix frontend run build
BREAST_PCR_HF_REPO=RubenElices/BreastDCEDL python app.py
```

Abre `http://127.0.0.1:8000`. Si el checkpoint no está en `checkpoints/`, se descarga del repositorio de Hugging Face y se verifica su SHA-256 contra [modelo/protocolo_e57.json](modelo/protocolo_e57.json). El servidor no guarda las imágenes recibidas. Despliegue: [deploy/README.md](deploy/README.md).

## Protocolo experimental

- La unidad estadística es la paciente: todas las particiones están agrupadas por paciente; los cortes de una paciente nunca se reparten entre particiones.
- Cada fold separa FIT (entrenamiento), validación interna (época y umbral) y fold exterior (evaluación honesta).
- El test público no se usa para elegir arquitectura, hiperparámetros, umbral, épocas ni agregación.
- Se comparan `BCEWithLogitsLoss` normal y ponderada (`pos_weight = N0/N1`).
- Los aumentos geométricos se aplican sincronizados a las tres fases; no hay normalización ImageNet ni aumentos de color.
- El preprocesado es idéntico en entrenamiento, evaluación y web.

## Datos y licencia

BreastDCEDL (cohortes Duke, I-SPY1 e I-SPY2), licencia [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). La web incluye unos pocos cortes de ejemplo con atribución ([frontend/public/LICENCIA_IMAGENES.md](frontend/public/LICENCIA_IMAGENES.md)).

**Uso educativo, sin validez clínica.** La probabilidad no es un diagnóstico, ni supervivencia, ni una recomendación terapéutica.
