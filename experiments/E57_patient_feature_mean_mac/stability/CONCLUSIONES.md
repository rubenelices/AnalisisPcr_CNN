# E57 · estabilidad en los cinco folds · 19 octubre 2026

El estudiante completó el smoke y los cuatro científicos locales preparados; el fold 0 procede del experimento E57 anterior. La CNN y la receta normal son las mismas; en los folds 1–4 se añadió evaluación de train en el mismo modo de validación para observar las curvas. No se evaluó test.

| Fold | Mejor época / épocas | AUC paciente en mejor época | AUC al final | AUC train en mejor checkpoint | Matriz a 0,5 |
|---|---:|---:|---:|---:|---|
| 0, histórico | 14 / 24 | 0,6703 | 0,5570 | 0,5577, auditoría posterior | TN155/FP0/FN64/TP0 |
| 1 | 27 / 37 | 0,6299 | 0,5844 | 0,6291 | TN157/FP0/FN62/TP0 |
| 2 | 10 / 20 | 0,6688 | 0,5678 | 0,5189 | TN161/FP0/FN59/TP0 |
| 3 | 2 / 12 | 0,5503 | 0,4490 | 0,5404 | TN145/FP0/FN75/TP0 |
| 4 | 5 / 15 | 0,5927 | 0,5474 | 0,4889 | TN157/FP0/FN62/TP0 |

**Media de los cinco máximos seleccionados: AUC 0,6224; desviación estándar entre folds 0,0514.** AP media 0,4096. Tres folds tienen máximos ≥0,6299, pero los otros dos bajan a 0,5503 y 0,5927. La media de máximos **no es una estimación exterior independiente**: cada fold se usó para elegir la época que aquí se reporta, y el fold 0 ya había motivado elegir E57. No se debe presentar 0,6224 como objetivo confirmado ni compararlo directamente con la media exterior 0,5272 del protocolo de selección interna anterior.

La lectura de las curvas cambia según el fold. En el 1, train y validación progresan inicialmente juntos y se separan moderadamente después de la época elegida. En el 2, la validación llega a 0,6688 en la época 10, se mantiene alrededor de 0,63 durante tres épocas y después cae; el AUC de train en ese checkpoint es solo 0,5189. Es una señal llamativa pero inestable, no el patrón clásico de train memorizado. En el 3, la pérdida de validación sube mientras la de train queda casi plana y el AUC de validación acaba bajo 0,5; no hay evidencia de una recuperación tardía. En el 4, el máximo de 0,5927 dura una época y el AUC vuelve aproximadamente a 0,5.

La matriz es el límite más claro: a umbral fijo 0,5, **los cinco folds predicen todas las pacientes como no pCR**. Juntando los folds hay TN775/FP0/FN322/TP0. El AUC y el AP indican cierta capacidad de ordenación en algunos folds, pero el clasificador actual no detecta ninguna positiva con ese umbral. Cambiarlo mirando estos mismos folds produciría una estimación sesgada; necesitaría selección interna por paciente y evaluación exterior separada.

Esta prueba confirma que el pico de E57 no fue exclusivo del fold 0, pero también que la señal es muy irregular y que la matriz sigue siendo inaceptable. E57 permanece como candidata de trabajo para estudiar umbral/calibración con separación interna y el efecto de cohortes; **no es todavía un resultado final favorable ni justifica otra búsqueda de arquitectura por sí solo**. No se lanzan nuevos entrenamientos automáticamente.

[Notebook con figuras visibles](RESULTADOS.ipynb) · [Panel PNG directo](panel_curvas_comparables.png) · [Protocolo](README.md).

## Auditoría posterior de las matrices

Se verificaron las etiquetas frente a metadata, las 1.097 pacientes disjuntas, la media de probabilidades de los cortes frente al CSV por paciente (error máximo 5,55e-17) y las matrices recalculadas con scikit-learn. No se detectó error en estos pasos. Los máximos por fold son 0,3595/0,4097/0,3663/0,3433/0,3826: ninguno alcanza 0,5. Las matrices a otros umbrales fijos demuestran que bajar el corte detecta positivas a costa de falsos positivos y que un único umbral no funciona igual para todos los modelos. [Diagnóstico completo y plan de selección interna](DIAGNOSTICO_MATRIZ.md). No se han cambiado los umbrales de los resultados originales ni entrenado nuevos modelos.
