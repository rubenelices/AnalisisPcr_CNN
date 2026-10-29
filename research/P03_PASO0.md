# P03 · Paso 0: análisis previo al recorte central

[Inicio](README.md) · [Estado](ESTADO.md) · [Reglas de entrega](REGLAS_ENTREGA.md).

**Escrito el 24 de octubre de 2026, antes de calcular nada.** Decide si merece la pena entrenar P03 (P02 + recorte central fijo dentro de la red). Solo usa train: imágenes de train, predicciones de validación ya guardadas y E57 aplicado a su propia validación (fold 0). No entrena, no toca test y no cambia el entregable.

Código: `python -m src.analysis.analisis_paso0`. Salidas: `reports/new_approach/P03_paso0/`.

## A. ¿Dónde está el realce?

**Hipótesis.** El realce del contraste (lo que más relación tiene con el tumor) está centrado en el recorte, de modo que un cuadrado central conserva la mayor parte en las tres cohortes.

**Medida.** Para cada corte de train: `r = max(EARLY − PRE, 0)` con valores / 255. Para los cuadrados centrales de 128, 160 y 192 px:

- fracción del realce de cada corte que cae dentro (mediana y percentil 25 por cohorte);
- centro de masas del realce medio de cada cohorte y su distancia al centro de la imagen;
- como referencia, la fracción de área: 25 %, 39 % y 56 %.

**Criterio (a).** Un tamaño S cumple si, **en las tres cohortes**, la mediana por corte del realce dentro es ≥ 60 % y el centro de masas está a menos de S/8 del centro. Si cumplen varios, P03 usaría **el menor**. Si ninguno cumple, no hay P03.

**Limitación.** El realce incluye vasos, corazón y pared torácica, no solo el tumor. Es un indicador, no una máscara.

## B. ¿En qué se fija E57?

**Hipótesis.** Si E57 usa atajos de encuadre o cohorte, el efecto de tapar zonas de la imagen se concentra en la periferia o en el borde de la mama con el aire, más de lo que corresponde a su área.

**Medida.** Mapas de oclusión de E57 (`src/web/occlusion_map.py`, cuadrados de 16 px) en **60 cortes de su validación (fold 0)**: 10 por cohorte × clase, un corte al azar por paciente, semilla 42. En cada mapa, la masa de |efecto|:

- fracción fuera del recorte central de tamaño S (el de A, o 160 si A no deja ninguno), dividida por la fracción de área fuera → **razón periférica R** (1 = proporcional al área);
- fracción en los cuadrados de **borde mama–aire** (con entre 20 % y 80 % de píxeles de aire, PRE < 0,05) dividida por su fracción de área → **razón de borde B**.

**Criterio (b).** La oclusión carga de forma apreciable fuera del centro si la **mediana de R ≥ 1,2** o la **mediana de B ≥ 1,5**, en el conjunto de los 60 cortes.

**Limitación.** Tapar con ceros crea imágenes que el modelo nunca vio; los efectos de E57 son pequeños (≤ 1 punto de probabilidad en los ejemplos de la web). Es un indicio, no una explicación causal.

## C. ¿Supera P02 a la cohorte sola?

**Medida.** Línea base de cohorte: a cada paciente exterior se le asigna la proporción de pCR de su cohorte en el resto de folds. AUC conjunta de esa línea base y diferencia pareada P02 normal − cohorte, con bootstrap estratificado por fold × cohorte × clase (2.000 réplicas, semilla 42).

**Para qué.** No decide P03; dice cuánto de lo que ordena P02 es solo saber de qué cohorte viene la paciente. Sirve para el informe y la defensa.

## Decisión

- **Se propone P03** solo si se cumplen **(a) y (b)**. Su ficha, hipótesis y criterios de éxito se escribirán aparte antes de entrenar, con el tamaño de recorte fijado aquí.
- Si no se cumplen, **no se entrena**: se mantiene E57 y el paso 0 se reporta como análisis.

## Resultados (24 de octubre, `reports/new_approach/P03_paso0/resumen.json`)

Criterios fijados antes de calcular: SHA-256 de este documento sin esta sección `7f851e4f…` (09:15 UTC), anotado en el JSON. Tiempo total: 31 s en el Mac (MPS).

### A. Realce: solo un recorte grande lo conserva

| Cohorte | Cortes | Centro de masas a… | Mediana dentro 128 | 160 | 192 |
|---|---:|---:|---:|---:|---:|
| I-SPY1 | 1.040 | 18,3 px | 50 % | 66 % | 79 % |
| I-SPY2 | 7.835 | 9,8 px | 37 % | 51 % | 66 % |
| Duke | 2.070 | 6,3 px | 35 % | 49 % | 64 % |
| *Área del cuadrado* | | | *25 %* | *39 %* | *56 %* |

El realce está centrado, pero repartido: un recorte de 128 (el del compañero) deja fuera dos tercios del realce en I-SPY2 y Duke. **Solo 192 cumple (a)**, y 192 apenas recorta (conserva el 56 % del área).

### B. Oclusión: E57 no se fija en la periferia

54 cortes (no 60: en el fold 0 solo hay 4 pacientes I-SPY1 con pCR). Recorte evaluado: 192.

| | Razón periférica R | Razón de borde B |
|---|---:|---:|
| Todos (mediana) | **0,37** | **0,53** |
| I-SPY1 | 0,08 | 0,27 |
| I-SPY2 | 0,46 | 0,34 |
| Duke | 0,37 | 0,92 |

La periferia recibe un tercio del efecto que le tocaría por área y el borde mama–aire, la mitad. Lo visto en los ejemplos de la web (efecto en el borde) **no se confirma**: E57 se apoya sobre todo en el centro. Los efectos son muy pequeños (mediana del máximo, 0,4 puntos de probabilidad). **(b) no se cumple.**

### C. P02 apenas supera a la cohorte sola

| | AUC exterior conjunta (1.097 pacientes) |
|---|---:|
| P02 normal | 0,532 |
| Línea base de cohorte | 0,523 |
| Diferencia pareada | **+0,009 [−0,025, +0,042]** |

Dentro de cada cohorte, P02 da 0,61 (I-SPY1), 0,54 (I-SPY2) y 0,40 (Duke). Lo que P02 ordena por encima del azar se explica casi entero por saber de qué cohorte viene la paciente.

## Decisión tomada

**No se propone P03.** (a) solo se cumple con 192, que casi no recorta, y (b) no se cumple: no hay periferia que quitar, porque E57 ya mira al centro. Un recorte no ataca el problema real, que C hace explícito: la señal de imagen que aprenden nuestras recetas no va más allá de la cohorte. Se mantiene E57 y este análisis se reporta como resultado.

Uso educativo, sin validez clínica.
