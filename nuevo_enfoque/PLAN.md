# Nuevo enfoque · plan de trabajo

Actualizado el 20 de octubre de 2026 (tarde). [Inicio](README.md) · [Estado](ESTADO.md) · [P01](P01_LOTES.md) · [Historial](HISTORIAL_EXPERIMENTOS.md).

## Qué sabemos hoy

1. **E57 y N01 apenas aprendían.** En los 5 folds de E57 y en las dos ejecuciones N01, la pérdida de entrenamiento se queda en la del predictor constante (0,605; 0,979 en la ponderada). Las probabilidades casi no varían y el AUC en FIT-eval está en torno a 0,50–0,56. Los picos de validación (0,670 en E57 fold0) son compatibles con el ruido de 219 pacientes (error típico ≈ 0,043) más la selección de la mejor época.
2. **Por eso los ~200 experimentos no distinguían arquitecturas:** las diferencias de 0,01–0,04 en un fold están dentro del ruido. No se reabre la búsqueda de arquitecturas.
3. **La receta de lote es un factor real.** Con la misma red y los mismos pasos, 10 pacientes × 2 cortes por lote memoriza 40 pacientes en 3–4 épocas; 2 pacientes × 10 cortes necesita ~23 y su pérdida de entrenamiento no baja. Detalle y lectura honesta frente a los criterios en [P01](P01_LOTES.md).
4. **Lo que sigue abierto:** si P01 generaliza con 702 pacientes o cae en memorización (régimen D36: FIT → 1, validación hacia abajo).

Decisiones del estudiante vigentes: solo imágenes (HR/HER2 no se usan ni como entrada ni como etiqueta auxiliar); los entrenamientos los lanza el estudiante; los 8 fits pendientes de N01 no se ejecutan.

## Pasos

### 1b · P02: P01 + aumento geométrico ← ahora

Resultado de P01 fold 0: aprende pero memoriza (interna plana ≤ 0,61; exterior 0,570). Edad y menopausia descartadas: AUC 0,532 con validación cruzada en train, en parte cohorte (la menopausia falta en todo I-SPY1), y la demo dependería de datos que el profesor quizá no aporte. Siguiente y única palanca: [P02](P02_AUMENTO.md), con criterios y regla de cierre fijados.

**Qué AUC reportar:** la cifra principal es la **exterior con época y umbral elegidos en la interna** (o test). El pico de validación es optimista y la media de épocas no mide rendimiento. Expectativa realista solo con imagen: 0,57–0,62 de media en 5 folds.

### 1 · P01 fold 0, BCE normal (hecho)

```bash
python -m src.train_p01 smoke --loss normal --fold 0
```

```bash
python -m src.train_p01 run --loss normal --fold 0
```

Criterios fijados en [P01](P01_LOTES.md): aprende (FIT-eval > 0,60 en la época elegida), la interna acompaña a FIT, exterior ≥ 0,62 con brecha FIT–interna < 0,15. Referencia directa: N01 normal fold 0, exterior 0,6165.

### 2a · Si cumple: validación cruzada completa

- Folds 1–4 con BCE normal y después los 5 folds con BCE ponderada (pos_weight = N0/N1). Es el contraste obligatorio del enunciado, ahora sobre un modelo que aprende.
- Resumen con las predicciones exteriores de las 1.097 pacientes: AUC conjunto con intervalo bootstrap, AP, Brier, matriz con umbral interno, **AUC por corte** (la demo evalúa un corte por paciente) y AUC por cohorte.
- Comparar normal y ponderada por pares sobre las mismas pacientes. No declarar ganador con un fold.

### 2b · Si memoriza: una sola palanca de regularización

Si FIT-eval se va a ~1 y la interna cae, el siguiente experimento cambia **una** cosa decidida antes de ver el resultado. Candidatas ya implementadas en el código: aumento geométrico sincronizado de las tres fases (flips/giro pequeño) o parada por BCE interna en vez de AUC. No volver a tocar la arquitectura.

### 2c · Si no aprende con 702 pacientes

Volver al diagnóstico antes de entrenar nada más: comparar la pérdida en train() y en eval() sobre el mismo lote y revisar LR y cabeza.

### 3 · Modelo de entrega

Reglas fijadas antes de ver los resultados de la GPU: [REGLAS_ENTREGA.md](REGLAS_ENTREGA.md) (SHA-256 `ac853cd9c2ed…`, 2026-10-20 22:05). En resumen: manda la AUC exterior conjunta de los 5 folds; normal salvo diferencia clara con la ponderada; P02 sustituye a E57 si su AUC conjunta es ≥ 0,55; se entrega el promedio de los 5 modelos con la mediana de los umbrales internos; el test se evalúa una sola vez.

### 4 · Demo, informe y presentación

- Demo: cargar el modelo elegido, mostrar la probabilidad por corte y el umbral fijado. La URL pública queda para cuando la web sea la definitiva.
- Informe: ampliar [INFORME_BORRADOR.md](INFORME_BORRADOR.md) con el diagnóstico de no aprendizaje, P01 y la comparación normal/ponderada.
- Cinco diapositivas: aplazadas hasta que el estudiante lo pida.

## Reglas de la etapa

- Test público: no se consulta para elegir nada. La validación privada del profesorado no existe para nosotros.
- Una variable por experimento; criterios escritos **antes** de ejecutar y cualquier desviación anotada.
- Se reportan siempre las curvas completas (FIT-eval, interna, BCE, AUC por corte), no solo el máximo.

## Anexo · E57 por cohorte (desarrollo)

Las cifras siguientes son de los checkpoints seleccionados con sus propias validaciones, por lo que son descriptivas y optimistas. No son nuevas evaluaciones de test. Los subgrupos son pequeños, especialmente I-SPY1; no deben sacarse conclusiones fuertes de un único valor.

|   Fold | Cohorte   |   Pacientes |   Positivas |   AUC |    AP |    BA |
|-------:|:----------|------------:|------------:|------:|------:|------:|
|      0 | duke      |          42 |          11 | 0.833 | 0.677 | 0.812 |
|      0 | spy1      |          21 |           4 | 0.647 | 0.283 | 0.449 |
|      0 | spy2      |         156 |          49 | 0.609 | 0.4   | 0.606 |
|      1 | duke      |          42 |           9 | 0.63  | 0.43  | 0.515 |
|      1 | spy1      |          21 |           7 | 0.653 | 0.666 | 0.643 |
|      1 | spy2      |         156 |          46 | 0.621 | 0.381 | 0.599 |
|      2 | duke      |          41 |           7 | 0.689 | 0.276 | 0.662 |
|      2 | spy1      |          21 |           7 | 0.724 | 0.571 | 0.679 |
|      2 | spy2      |         158 |          45 | 0.672 | 0.431 | 0.595 |
|      3 | duke      |          42 |           9 | 0.478 | 0.259 | 0.485 |
|      3 | spy1      |          21 |           3 | 0.667 | 0.289 | 0.5   |
|      3 | spy2      |         157 |          63 | 0.509 | 0.455 | 0.521 |
|      4 | duke      |          42 |           8 | 0.599 | 0.451 | 0.496 |
|      4 | spy1      |          20 |           5 | 0.573 | 0.368 | 0.5   |
|      4 | spy2      |         157 |          49 | 0.578 | 0.403 | 0.513 |

El fold3 es débil también dentro de Duke y de I-SPY2. Eso impide atribuir sus malos resultados únicamente a la mezcla de cohortes. No demuestra una causa única; descarta una explicación demasiado simple y evita cambiar folds para esconder el problema.

Uso educativo, sin validez clínica.
