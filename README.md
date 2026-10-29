# Predicción de pCR con DCE-MRI y una CNN 2D desde cero

Proyecto de Aprendizaje Automático (Universidad Alfonso X el Sabio, 2026–2027). Estima la probabilidad de **respuesta patológica completa (pCR)** a la quimioterapia neoadyuvante en cáncer de mama a partir de las tres fases de una DCE-MRI previa al tratamiento (PRE, EARLY y LATE), con una CNN 2D diseñada y entrenada desde cero en PyTorch, sin transfer learning ni pesos preentrenados.

**Demo web:** [rubenelices-pcr-demo.hf.space](https://rubenelices-pcr-demo.hf.space) · **Pesos del modelo:** [huggingface.co/RubenElices/BreastDCEDL](https://huggingface.co/RubenElices/BreastDCEDL) · **Resultados:** [research/RESULTADOS.ipynb](research/RESULTADOS.ipynb) · **Presentación:** [docs/presentation](docs/presentation/presentacion.pdf)

## Resultado

El modelo entregado es **E57**: 298.161 parámetros, entrada por corte `[3, 256, 256]` (fases temporales, no RGB), media de las probabilidades de los cortes por paciente y umbral 0,325 fijado antes de abrir el test.

| Evaluación | AUC por paciente |
|---|---|
| Test público (176 pacientes, evaluado una sola vez) | **0,573** [IC95 0,475–0,671] |
| Validación cruzada de la receta en 5 folds (1.097 pacientes, predicciones fuera de fold) | 0,507 |
| Solo la cohorte de origen (línea base) | 0,523 |

La señal que la red extrae de la imagen no supera de forma demostrable a saber de qué cohorte procede la paciente. Las hipótesis probadas para mejorarla (recetas de entrenamiento, aumento geométrico, recortes, realce relativo, ensembles de semillas) tienen sus criterios escritos antes de ejecutar y sus resultados, también los negativos, en [research/](research/README.md).

## Estructura

```
├── app.py                 servidor web: API Flask de inferencia + interfaz compilada
├── src/
│   ├── pipeline/          carga de datos, fases DCE, aumentos sincronizados
│   ├── models/            arquitecturas propias (E57 y variantes)
│   ├── training/          bucles de entrenamiento, configuración y protocolo N01
│   ├── evaluation/        métricas por corte y por paciente, validación cruzada, test final
│   ├── analysis/          análisis estadísticos pareados (P03–P07, N01)
│   ├── reporting/         notebook de resultados y fichas de arquitectura
│   ├── web/               inferencia y datos de la web
│   └── campaigns/         runners de las campañas de entrenamiento
├── tests/                 tests de software con datos sintéticos
├── configs/               configuraciones de los experimentos
├── experiments/           una carpeta por experimento: ficha, diagrama de volúmenes y capas reales
│   ├── E57_patient_feature_mean_mac/   modelo entregado
│   │   ├── results/       5 folds, test público y figuras
│   │   └── stability/     repeticiones en los folds 1–4
│   └── notes/             notas y diario de los experimentos anteriores
├── research/              protocolo, hipótesis con criterios previos, decisiones y notebook de resultados
├── model/                 protocolo congelado del modelo entregado (umbral, agregación, SHA-256)
├── frontend/              interfaz React
├── web/                   demo HTML anterior (servida en /demo)
├── deploy/                despliegue en Hugging Face Spaces (preparar_space.py monta y sube la web)
└── docs/presentation/     presentación
```

## Uso

```bash
pip install -r requirements.txt
python -m pytest -q tests
```

Los datos de BreastDCEDL no se incluyen: se descargan del sitio oficial y se colocan en `breastdcedl/`. Los checkpoints, informes generados (`reports/`) y pesos quedan fuera del repositorio.

```bash
python -m src.training.train_new_approach run --loss normal --fold 0 --device mps   # receta de E57 con el protocolo N01
python -m src.reporting.cuaderno_resultados                                          # regenera research/RESULTADOS.ipynb
```

### Demo web en local

```bash
npm --prefix frontend ci && npm --prefix frontend run build
BREAST_PCR_HF_REPO=RubenElices/BreastDCEDL python app.py
```

Abre `http://127.0.0.1:8000`. Si el checkpoint no está en `checkpoints/`, se descarga del repositorio de Hugging Face y se verifica su SHA-256 contra [model/protocolo_e57.json](model/protocolo_e57.json). El servidor no guarda las imágenes recibidas. La web pública está en [rubenelices-pcr-demo.hf.space](https://rubenelices-pcr-demo.hf.space); cómo se despliega: [deploy/README.md](deploy/README.md).

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
