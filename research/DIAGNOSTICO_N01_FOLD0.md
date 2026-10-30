# Diagnóstico ligero de N01 · primera pareja

Análisis descriptivo de las 219 pacientes exteriores del fold 0, utilizando exclusivamente predicciones guardadas. No se cargan pesos ni imágenes, no se entrena y no se consulta test. Los umbrales se conservan tal como se eligieron en la interna. Este análisis no se utiliza para ajustarlos.

## Separación de las clases

| Pérdida   |   Clase |   Pacientes |   Media |   Mediana |   Mínimo |   Máximo |
|:----------|--------:|------------:|--------:|----------:|---------:|---------:|
| Normal    |       0 |         155 | 0.23925 |   0.25329 |  0.04894 |  0.29965 |
| Normal    |       1 |          64 | 0.25459 |   0.26487 |  0.1218  |  0.31271 |
| Ponderada |       0 |         155 | 0.48909 |   0.497   |  0.30874 |  0.51015 |
| Ponderada |       1 |          64 | 0.49389 |   0.50035 |  0.41817 |  0.51062 |

Las distribuciones se solapan mucho. La media positiva es algo mayor, pero la diferencia es pequeña. La ponderada desplaza las probabilidades hacia 0,5 y separa poco las clases. Mover un umbral no corrige la ordenación débil que refleja el AUC. Estas probabilidades son salidas del modelo y no deben interpretarse como riesgos clínicos calibrados.

## Sensibilidad a cambios pequeños del umbral

Se cuentan las bandas descriptivas ±0,005 y ±0,01 alrededor del umbral congelado. No son nuevos criterios de selección ni se exploran umbrales alternativos buscando el mejor resultado exterior.

| Pérdida   |   Umbral |   Dentro de ±0,005 |   Dentro de ±0,01 |   Positivas a 0,5 |   Positivas al umbral interno |
|:----------|---------:|-------------------:|------------------:|------------------:|------------------------------:|
| Normal    |  0.26395 |                 29 |                70 |                 0 |                            82 |
| Ponderada |  0.50293 |                 98 |               136 |               102 |                            75 |

En la ponderada, 98 de 219 pacientes están dentro de ±0,005 del umbral interno. Pasar de 0,5 al umbral interno ya fijado cambia 27 predicciones. Esto ayuda a explicar por qué una matriz puede variar mucho con un cambio numéricamente pequeño. No permite escoger ahora el corte que dé mejor matriz exterior.

## Probabilidades y calibración

| Pérdida   |   Probabilidad media |   Frecuencia pCR |   Brier |   Brier constante FIT |
|:----------|---------------------:|-----------------:|--------:|----------------------:|
| Normal    |              0.24373 |          0.29224 | 0.20475 |               0.20684 |
| Ponderada |              0.49049 |          0.29224 | 0.24472 |               0.20684 |

El predictor constante utiliza la prevalencia de FIT, no la del exterior. La normal mejora muy ligeramente su Brier; la ponderada es peor. La ponderación cambia el objetivo y puede alterar la calibración. No se ajusta ninguna calibración con estas 219 pacientes. Una transformación estrictamente creciente de los scores conserva su ordenación y por tanto su ROC-AUC; no resolvería por sí sola la discriminación baja.

## ¿Se equivocan con las mismas pacientes?

| Grupo   |   Ambas bien |   Solo normal bien |   Solo ponderada bien |   Ambas mal |
|:--------|-------------:|-------------------:|----------------------:|------------:|
| Todas   |           91 |                 48 |                    37 |          43 |
| pCR=0   |           79 |                 27 |                    25 |          24 |
| pCR=1   |           12 |                 21 |                    12 |          19 |

Las dos aciertan 91 pacientes y fallan 43. La normal acierta 48 que la ponderada falla; la ponderada acierta 37 que la normal falla. Hay errores diferentes, pero eso no demuestra que una combinación mejore ni justifica ajustar un ensemble con este exterior. Las diferencias dependen también de los umbrales internos distintos.

## Resultado por cohorte

| Pérdida   | Cohorte   |   Pacientes |   pCR |     AUC |      AP |      BA |   Sensibilidad |   Especificidad |   TN |   FP |   FN |   TP |
|:----------|:----------|------------:|------:|--------:|--------:|--------:|---------------:|----------------:|-----:|-----:|-----:|-----:|
| Normal    | duke      |          42 |    11 | 0.63636 | 0.41395 | 0.56598 |        0.45455 |         0.67742 |   21 |   10 |    6 |    5 |
| Normal    | spy1      |          21 |     4 | 0.57353 | 0.43468 | 0.625   |        0.25    |         1       |   17 |    0 |    3 |    1 |
| Normal    | spy2      |         156 |    49 | 0.61053 | 0.43149 | 0.59327 |        0.55102 |         0.63551 |   68 |   39 |   22 |   27 |
| Ponderada | duke      |          42 |    11 | 0.74194 | 0.47114 | 0.64663 |        0.45455 |         0.83871 |   26 |    5 |    6 |    5 |
| Ponderada | spy1      |          21 |     4 | 0.60294 | 0.43624 | 0.50735 |        0.25    |         0.76471 |   13 |    4 |    3 |    1 |
| Ponderada | spy2      |         156 |    49 | 0.49304 | 0.32063 | 0.48741 |        0.36735 |         0.60748 |   65 |   42 |   31 |   18 |

Las submuestras son pequeñas, especialmente I-SPY1 (21 pacientes). Estas cifras sirven para describir variabilidad, no para elegir cohortes favorables, eliminar pacientes o declarar causas. No se añaden edad, menopausia ni otros datos clínicos al modelo.

## Qué queda aprendido y qué no

La prioridad es conseguir una separación de clases más estable y una evaluación coherente, además de una matriz útil. Las curvas juntas no bastan: también pueden aparecer cuando el modelo discrimina poco. El fold 0 favorece descriptivamente a la normal, pero no confirma que ocurra en otros folds ni permite atribuir el resultado a una causa única.

Se conserva la arquitectura y la receta N01. No se cambian pooling, aumentos, umbral, calibración o variables a partir de este diagnóstico. Cuando vuelva a ser posible entrenar, la pareja del fold 1 permitirá comprobar si el comportamiento se repite. Si se cierra aquí, el resultado se reportará como comparación de un solo fold.

Fuentes: [revisión de la pareja](REVISION_N01_FOLD0.md), [protocolo fijado](PROTOCOLO_N01.md) y [resultados completos](RESULTADOS.ipynb). Respaldo reproducible en `reports/new_approach/N01_e57_bce/diagnostics_fold0/`; regeneración: `python -m src.analysis.analyze_n01_fold0`.

Uso educativo, sin validez clínica.
