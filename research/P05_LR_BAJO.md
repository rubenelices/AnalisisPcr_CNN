# P05 · P02 con tasa de aprendizaje 10 veces menor

> **Retirado sin ejecutar (24 de octubre).** Al revisar el historial: la tasa 0,0001 ya se probó en E12 y E55 sin mejora. Se sustituye por [P07](P07_SEMILLAS.md). El documento se conserva tal como se escribió.

[Inicio](README.md) · [Estado](ESTADO.md) · [Paso 0](P03_PASO0.md) · [Reglas de entrega](REGLAS_ENTREGA.md).

**Escrito el 24 de octubre de 2026, antes de entrenar nada.** Los tres filtros baratos (P03 recorte, P04 realce relativo, P06 región que más realza) se descartaron. Queda lo que señalan las curvas de P02: el problema está en la receta de entrenamiento, no en la entrada ni en la red.

## Qué vemos en P02 normal (5 folds, GPU)

- En 3 de 5 folds la época elegida por la validación interna es la **1 o la 2** (folds 2, 3 y 4); en los otros dos, la 9 y la 19.
- El AUC FIT-eval sigue subiendo después del pico interno mientras el AUC interno cae: la red memoriza pacientes de FIT antes de aprender algo que generalice.
- Con AdamW a 0,001 y ReduceLROnPlateau (paciencia 3, factor 0,5), la tasa solo baja cuando el daño ya está hecho.

## Hipótesis

Con una tasa 10 veces menor, cada paso mueve menos los pesos: la red tarda más en memorizar y el pico de validación interna llega más tarde, más alto y más estable. Si la señal de imagen existe pero se pierde por pasos demasiado grandes, P05 debería ordenar mejor a las pacientes exteriores que P02.

## Único cambio

`learning_rate`: 0,001 → **0,0001**. Todo lo demás idéntico a P02 normal: particiones N01, lotes 10 pacientes × 2 cortes × 5 recorridos, aumento geométrico sincronizado, AdamW con `weight_decay` 0,001, ReduceLROnPlateau, 50 épocas como máximo, paciencia 10, selección de época y umbral en la validación interna, semilla 42, **solo BCE normal** (P02 ya comparó normal y ponderada; la ponderada no mejoró).

Cambio técnico necesario y no experimental: el runner tenía la tasa escrita a mano (`lr=.001`); ahora la lee del JSON. Para P01 y P02 el JSON ya dice 0,001, así que su comportamiento no cambia (lo comprueba un test).

Además, P05 guarda las predicciones exteriores de **cada época** en un fichero aparte. No intervienen en la elección de época ni de umbral (que sigue siendo solo interna y se congela en disco antes de la evaluación exterior final); sirven únicamente para el análisis secundario de época fija descrito abajo.

## Criterios (fijados antes de ejecutar)

Análisis: `python -m src.analysis.analisis_p05`. Mismo método que el paso 0: predicciones exteriores de las 1.097 pacientes (5 folds), bootstrap pareado estratificado por fold × cohorte × clase, 2.000 réplicas, semilla 42.

**Principal.** P05 mejora a P02 si el **IC95 pareado de AUC(P05) − AUC(P02 normal) queda por encima de 0**.

**Para ser candidato a entrega** (aplicando las [reglas de entrega](REGLAS_ENTREGA.md) a P05 en lugar de P02) tiene que cumplir las tres:

1. criterio principal;
2. AUC exterior conjunta ≥ 0,55 (regla 4);
3. IC95 pareado de AUC(P05) − AUC(línea base de cohorte) por encima de 0: que la mejora no sea solo saber de qué cohorte viene la paciente.

Si las cumple, se propone cambiar el entregable al promedio de los 5 modelos de P05 con la mediana de los 5 umbrales internos y **una única** evaluación en test. Nada de eso se ejecuta sin su aprobación.

**Se reportan siempre, sin decidir:** AUC por corte; AUC dentro de cada cohorte (en especial Duke, que en P02 se ordenaba al revés: 0,40); época elegida por fold; brecha AUC FIT − interna en la época elegida (comprueba el mecanismo de la hipótesis); curvas completas por fold.

**Secundario (no decide):** regla de época fija. E* = la época con mayor AUC interna media entre los 5 folds, contando solo épocas que completaron los 5. Se calcula el AUC exterior conjunta de las predicciones de la época E*. Mide cuánto pierde la selección por fold por el ruido de una validación interna de ~175 pacientes.

**Si no se cumple el principal:** P05 se reporta como resultado negativo con sus curvas. No se ajusta la tasa a otro valor mirando el resultado exterior: cualquier variante posterior lleva su propio documento escrito antes.

## Coste y ejecución

Se lanza a mano. Smoke técnico (1 época, subconjunto) y 5 folds, BCE normal. Con una tasa menor la parada temprana llega más tarde: en GPU ≈ 75–90 s por época, hasta 50 épocas por fold, entre 2 y 6 horas en total. En el Mac (MPS), el triple.

```bash
python -m src.campaigns.run_p05_noche --check
python -m src.campaigns.run_p05_noche
```

## Expectativa

P02 no superó a la línea base de cohorte (+0,009 [−0,025, +0,042]) y los análisis del paso 0 no encontraron señal de imagen más allá de la cohorte. Lo más probable es que P05 tampoco cumpla. Si cumple, sería la primera receta propia que demuestra aprender algo de la imagen; si no, refuerza la conclusión con una causa más descartada (el tamaño del paso).

Uso educativo, sin validez clínica.
