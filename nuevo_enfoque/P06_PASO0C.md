# P06 · Paso 0c: ¿hay señal en la región que más realza?

[Inicio](README.md) · [Estado](ESTADO.md) · [Paso 0b](P04_PASO0B.md) · [Reglas de entrega](REGLAS_ENTREGA.md).

**Escrito el 24 de octubre de 2026, antes de calcular nada.** El paso 0b midió el realce en todo el tejido y no encontró señal más allá de la cohorte, pero dejó abierta una salvedad: la información podría estar **localizada en el tumor** y diluirse al promediar el corte entero. Este filtro mide solo en la región que más realza, que es un candidato a tumor sin máscara. Solo train; no entrena la CNN ni toca test; las variables no entran en ningún modelo de entrega.

Código: `python -m src.analisis_paso0c`. Salidas: `reports/new_approach/P06_paso0c/`.

## Medida

Por corte, en el tejido (PRE ≥ 0,05): mapa de PE = (EARLY − PRE)/PRE, suavizado con una media de 15 × 15 px. Se descarta un margen de 16 px junto al borde de la imagen. La **región candidata** es la ventana de 48 × 48 px centrada en el máximo del PE suavizado. En ella:

- PE medio y percentil 90 de PE;
- fracción con PE ≥ 0,70 y, de esa, fracción con lavado (SER > 1,1);
- heterogeneidad: desviación típica de EARLY − PRE;
- tamaño: píxeles con PE ≥ 0,70 en la ventana de 96 × 96 alrededor.

Por paciente: media y máximo de cada variable entre sus cortes. Regresión logística con los 5 folds del profesor, como en el paso 0b: (1) solo cohorte; (2) cohorte + región. Bootstrap pareado estratificado por fold × cohorte × clase (2.000 réplicas, semilla 42).

## Criterio

**P06 (recorte adaptativo a la región que más realza) se prepara si** el IC95 pareado de (2) − (1) queda **por encima de 0**. Se reportan además el AUC de la región sola dentro de cada cohorte y la fracción de cortes cuya región cae en el centro de la imagen (para saber si el candidato es tumor o pared torácica).

**Limitación.** La región que más realza puede ser corazón, vasos o pared torácica, no el tumor. Si el filtro falla, no prueba que no haya señal local; prueba que este candidato no la tiene.

## Resultados (24 de octubre, `reports/new_approach/P06_paso0c/resumen.json`)

Criterio fijado antes de calcular: SHA-256 `f0667e73…` (09:26 UTC), anotado en el JSON.

| | AUC exterior conjunta (1.097 pacientes) |
|---|---:|
| Solo cohorte | 0,523 |
| Cohorte + región candidata | 0,532 |
| Región sola | 0,489 |
| **Diferencia pareada (criterio)** | **+0,009 [−0,019, +0,038]** |

La región sola da 0,51 en I-SPY1, 0,49 en I-SPY2 y 0,48 en Duke. La región candidata solo cae en el cuadrado central de 128 en el 49 % de los cortes de I-SPY1, el 22 % de I-SPY2 y el 24 % de Duke: a menudo es pared torácica, corazón o vasos, no el tumor.

## Decisión tomada

**No se prepara P06.** El candidato a tumor sin máscara no aporta información a la cohorte. Con el alcance escrito antes: no demuestra que no haya señal local, demuestra que sin una localización fiable del tumor no se puede aprovechar.

Uso educativo, sin validez clínica.
