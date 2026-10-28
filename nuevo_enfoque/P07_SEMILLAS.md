# P07 · P02 normal promediando tres semillas

[Inicio](README.md) · [Estado](ESTADO.md) · [P05 (retirado)](P05_LR_BAJO.md) · [Reglas de entrega](REGLAS_ENTREGA.md).

**Escrito el 24 de octubre de 2026, antes de entrenar nada.** Sustituye a P05: la tasa 0,0001 ya se probó dos veces en el historial (E12 y E55) sin mejora.

## Por qué

La misma receta cambia con la semilla: en el fold 0, la AUC de validación al final de E19 va de 0,534 (semilla 17) a 0,552 (semilla 123), y su máximo de 0,585 a 0,616; entre folds, la misma receta va de 0,525 a 0,548 (historial). En P02, la época elegida salta entre la 1 y la 19 según el fold. Con tan poca señal, cada modelo aprende algo del patrón real y bastante ruido propio (inicialización, orden de pacientes, aumentos). El patrón se comparte entre semillas y el ruido no, así que el promedio debería ordenar mejor. No busca señal nueva: reduce la varianza.

## Único cambio

En cada uno de los 5 folds, P02 normal se entrena con las semillas **17 y 123** además de la 42 ya existente. La predicción del ensemble es la **media de las probabilidades de los 3 modelos** por corte, y por paciente la media de sus cortes (equivale a promediar las probabilidades por paciente de cada modelo). Misma configuración (`P02_aumento_geometrico.json`, mismo SHA), mismas particiones N01, misma selección de época por validación interna en cada modelo. Solo BCE normal.

La semilla controla la inicialización, el orden de los lotes y los aumentos. Las particiones FIT / interna / exterior **no** cambian con la semilla.

Cambio técnico: el runner acepta `--seed` (por defecto 42, idéntico a lo anterior). Cada semilla tiene su smoke y sus carpetas `normal_fold_k_seed_S`; nada de lo existente se sobrescribe.

## Criterios (fijados antes de ejecutar)

Análisis: `python -m src.analisis_p07`. Predicciones exteriores de las 1.097 pacientes, bootstrap pareado estratificado por fold × cohorte × clase, 2.000 réplicas, semilla 42.

**Principal.** El ensemble mejora si el **IC95 pareado de AUC(ensemble 3 semillas) − AUC(P02 normal semilla 42) queda por encima de 0**.

**Candidato a entrega** (reglas de entrega aplicadas al ensemble): principal + AUC exterior conjunta ≥ 0,55 + IC95 pareado frente a la línea base de cohorte por encima de 0. El entregable sería el promedio de los 15 modelos; el umbral, la mediana de los 5 umbrales del ensemble, cada uno elegido por F1 macro sobre la **validación interna** de su fold con la media de las 3 semillas. Nada se cambia ni se evalúa en test sin aprobación del estudiante.

**Se reportan siempre:** AUC conjunta de cada semilla por separado (¿fue la 42 afortunada?) y su media; diferencia ensemble − media de las semillas individuales (secundario); AUC por corte; AUC dentro de cada cohorte; matriz de confusión conjunta con los umbrales internos.

**Si no se cumple:** se reporta como resultado; no se añaden semillas mirando el resultado exterior.

## Ejecución

Lo lanza el estudiante. 2 smokes + 10 entrenamientos. P02 normal tardó ≈ 100 min los 5 folds en la GPU: ≈ 3,5 h.

```bash
python -m src.run_p07_noche --check
python -m src.run_p07_noche
```

## Desviación (24 de octubre, antes de calcular el ensemble)

Las semillas 17 y 123 se entrenan en el **Mac (MPS)**, no en la GPU; la 42 sigue siendo la de la GPU (CUDA). Misma configuración (SHA `a4385f77…`) y mismo protocolo de particiones (`827fef1d…`). MPS y CUDA no dan números idénticos con la misma semilla, así que el dispositivo es una fuente más de variación entre modelos, como la propia semilla; no cambia las particiones ni la receta. El criterio principal no cambia. Al anotar esto ya se habían visto en la terminal los AUC exteriores por fold de la semilla 17 (entre 0,539 y 0,569); el ensemble y la comparación pareada aún no se habían calculado.

## Expectativa

Lo esperable con ensembles en datos ruidosos es una mejora pequeña (+0,01 a +0,03). La dirección es probable; que el IC95 quede por encima de 0 con 1.097 pacientes, no está asegurado.

Uso educativo, sin validez clínica.

## Resultados (24 de octubre, `reports/new_approach/P07_semillas/resumen.json`)

15 ejecuciones completas (semilla 42 en GPU; 17 y 123 en el Mac), misma configuración y mismas particiones. Criterio fijado antes: SHA `d6012995…`, anotado en el JSON.

| AUC exterior conjunta (1.097 pacientes) | |
|---|---:|
| **Ensemble 3 semillas** | **0,543** |
| Semilla 42 (P02) | 0,532 |
| Semilla 17 | 0,553 |
| Semilla 123 | 0,521 |
| Media de las tres por separado | 0,535 |
| Línea base de cohorte | 0,523 |

| Comparación pareada | Diferencia | IC95 |
|---|---:|---|
| **Ensemble − semilla 42 (criterio principal)** | +0,012 | [−0,013, +0,036] |
| Ensemble − cohorte | +0,021 | [−0,014, +0,054] |

AUC por corte del ensemble: 0,543. Dentro de cada cohorte: Duke 0,42, I-SPY1 0,56, I-SPY2 0,55. Matriz conjunta con los umbrales internos del ensemble: TN 607, FP 168, FN 228, TP 94 (sensibilidad 0,292, especificidad 0,783). La semilla 123 eligió la época 3 en 4 de 5 folds.

## Decisión tomada

**No se cumple ningún criterio:** el IC95 del principal incluye 0, la AUC conjunta queda por debajo de 0,55 y no supera a la cohorte de forma demostrable. El entregable sigue siendo E57.

Lo que sí se observa, sin valor de decisión: el ensemble queda por encima de la media de sus tres modelos (+0,008), en la dirección prevista, y la semilla sola ya varía 0,03 entre 0,521 y 0,553. No se elige la semilla 17 por su AUC exterior: sería seleccionar con la exterior.
