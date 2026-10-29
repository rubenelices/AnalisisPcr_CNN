# N01 · validación cruzada de la receta de E57 (folds 1–4)

[Inicio](README.md) · [Estado](ESTADO.md) · [Reglas de entrega](REGLAS_ENTREGA.md).

**Escrito el 27 de octubre de 2026, antes de entrenar los folds 1–4.** Revoca la decisión anterior del estudiante de no ejecutar los fits pendientes de N01, solo para la BCE normal.

## Para qué

N01 es la receta de E57 (2 pacientes × 10 cortes por lote, sin aumento) con el protocolo limpio: FIT / validación interna / fold exterior. Solo existe el fold 0, con AUC exterior **0,6165**, la más alta medida en el proyecto. E57, el modelo entregado, no tiene una validación cruzada limpia propia: su cifra honesta es el test (0,573). Completar los folds 1–4 da la **validación cruzada de la receta del modelo entregado** sobre las 1.097 pacientes.

## Qué se espera

En el fold 0, la AUC de FIT en la época elegida (2) fue 0,505 y la BCE se quedó en la del predictor constante (≈ 0,605): la red apenas aprendió. Un 0,6165 exterior con 0,505 en FIT es compatible con el azar de 219 pacientes (error típico ≈ 0,043). **Predicción escrita antes:** la AUC exterior conjunta quedará entre 0,50 y 0,56, sin superar a la cohorte de forma demostrable. La hipótesis del estudiante es la contraria: que el fold 0 no fue suerte.

## Qué se ejecuta

`python -m src.training.train_new_approach run --loss normal --fold k` para k = 1, 2, 3, 4, en el Mac (MPS), con el código y el smoke ya existentes. Nada cambia en la receta. ≈ 17 min por fold.

## Cómo se interpretará (fijado antes)

Análisis: `python -m src.analysis.analisis_n01_cv`. Mismo método que P05 y P07 (bootstrap pareado estratificado por fold × cohorte × clase, 2.000 réplicas, semilla 42).

- **Se reporta:** AUC exterior conjunta con su IC95, por corte, por cohorte y por fold; diferencias pareadas frente a la línea base de cohorte y frente a P02 normal.
- **"La receta de E57 aprende señal más allá de la cohorte"** solo si el IC95 pareado N01 − cohorte queda por encima de 0.
- **No es un candidato a entrega.** E57 sigue siendo el entregable pase lo que pase. Si el resultado fuera muy bueno (conjunta ≥ 0,55 e IC95 frente a la cohorte por encima de 0), cambiar el entregable al promedio de los 5 modelos de N01 necesitaría reglas nuevas escritas antes y la aprobación del estudiante, con una única evaluación en test.
- El fold 0 entra en la conjunta igual que los demás: no se descarta ni se pondera.

Uso educativo, sin validez clínica.

## Resultados (27 de octubre, `reports/new_approach/N01_e57_bce/cv/resumen.json`)

5 folds completos, BCE normal, Mac. El SHA del criterio (`ebdfb36b…`) está anotado en el JSON.

| AUC exterior conjunta (1.097 pacientes) | |
|---|---:|
| **N01 (receta de E57)** | **0,507** |
| P02 normal | 0,532 |
| Línea base de cohorte | 0,523 |

| Comparación pareada | Diferencia | IC95 |
|---|---:|---|
| **N01 − cohorte (criterio)** | −0,015 | [−0,037, +0,006] |
| N01 − P02 | −0,024 | [−0,058, +0,009] |

AUC por corte: 0,508 (P02: 0,529). Dentro de cada cohorte: Duke 0,46, I-SPY1 0,50, I-SPY2 0,51.

| Fold | Época | AUC FIT | AUC interna | AUC exterior |
|---:|---:|---:|---:|---:|
| 0 | 2 | 0,505 | 0,572 | 0,617 |
| 1 | 6 | 0,561 | 0,611 | 0,494 |
| 2 | 13 | 0,529 | 0,618 | 0,542 |
| 3 | 6 | 0,585 | 0,577 | 0,472 |
| 4 | 6 | 0,541 | 0,597 | 0,445 |

## Decisión tomada

**No se cumple el criterio:** el IC95 frente a la cohorte incluye 0 (y la diferencia es negativa). La receta de E57 no demuestra aprender señal de imagen más allá de la cohorte. Se cumple la predicción escrita antes (conjunta 0,50–0,56); la hipótesis del estudiante (el fold 0 no fue suerte) no se sostiene: los otros cuatro folds dan entre 0,445 y 0,542.

Lo que enseña, sin valor de decisión:

- **La validación interna no predice la exterior.** En los 5 folds la AUC interna en la época elegida está entre 0,57 y 0,62, y la exterior entre 0,44 y 0,62, sin relación de orden (el fold con mejor interna, el 2, no es el mejor fuera). Elegir la mejor de ~15 épocas sobre 176 pacientes infla el pico.
- **AUC FIT ≈ 0,5–0,58 en la época elegida:** la red casi no ordena ni a las pacientes con las que entrena, coherente con el diagnóstico de [P01](P01_LOTES.md) (BatchNorm con 2 pacientes por lote).
- **El test de E57 (0,573 [0,475, 0,671]) es compatible con esta CV:** el intervalo contiene 0,507. No hay contradicción, pero tampoco evidencia de que E57 supere al azar más allá de la cohorte.

El entregable sigue siendo E57; su cifra oficial sigue siendo el test, consultado una sola vez.
