# P04 · Paso 0b: ¿lleva señal el realce relativo?

[Inicio](README.md) · [Estado](ESTADO.md) · [Paso 0 de P03](P03_PASO0.md) · [Reglas de entrega](REGLAS_ENTREGA.md).

**Escrito el 24 de octubre de 2026, antes de calcular nada.** Filtro barato antes de entrenar P04 (P02 con la entrada en realce relativo). Solo usa train y las 5 particiones del profesor. No entrena la CNN ni toca test. Las variables que se calculan aquí **no entran en ningún modelo de entrega**: sirven solo para decidir si P04 merece 3 horas de GPU o de Mac.

Código: `python -m src.analisis_paso0b`. Salidas: `reports/new_approach/P04_paso0b/`.

## Hipótesis

El paso 0 mostró que P02 apenas supera a la cohorte sola (+0,009 [−0,025, +0,042]) y que Duke se ordena al revés. Si la red aprende diferencias de adquisición, una entrada **relativa**, que no depende de la ganancia del escáner, debería llevar señal sobre pCR **más allá de la cohorte**. Las medidas son las de la DCE-MRI de mama (volumen tumoral funcional de I-SPY):

- **PE** = (EARLY − PRE) / PRE, realce porcentual;
- **SER** = (EARLY − PRE) / (LATE − PRE), relación de realce: > 1,1 indica lavado (*washout*).

## Medida

Por corte, en el tejido (PRE ≥ 0,05, valores / 255):

- **Relativas:** fracción de tejido con PE ≥ 0,70; PE medio del 5 % de píxeles que más realzan; fracción de esos píxeles (PE ≥ 0,70) con SER > 1,1.
- **Absolutas** (la misma forma con restas, como la entrada actual): fracción de tejido con EARLY − PRE ≥ 0,10; media del 5 % superior de EARLY − PRE; fracción de esos píxeles con LATE − EARLY < 0.

Por paciente, la media de sus cortes. Regresión logística (variables estandarizadas dentro de cada partición de entrenamiento) con validación cruzada de los 5 folds del profesor: predicciones exteriores de las 1.097 pacientes. Modelos:

1. solo cohorte;
2. cohorte + relativas;
3. cohorte + absolutas;
4. relativas solas, y su AUC dentro de cada cohorte.

Comparaciones pareadas con bootstrap estratificado por fold × cohorte × clase (2.000 réplicas, semilla 42).

## Criterio para preparar P04

**Se prepara P04 si** el IC95 pareado de (2) − (1) queda **por encima de 0**: el realce relativo añade información sobre pCR más allá de la cohorte.

Se reporta además (2) − (3), para ver si lo relativo aporta más que lo absoluto, y el AUC de (4) en Duke. No deciden por sí solos.

**Si no se cumple,** no se entrena P04: ni con las medidas que usa la radiología hay en estos cortes señal de imagen que supere a la cohorte, y eso se reporta como resultado.

**Limitaciones.** Un corte 2D por posición, sin máscara de tumor: el tejido incluye pared torácica y vasos. Es un filtro de viabilidad, no un modelo clínico.

## Resultados (24 de octubre, `reports/new_approach/P04_paso0b/resumen.json`)

Criterio fijado antes de calcular: SHA-256 de este documento sin esta sección `3325f5d2…` (09:22 UTC), anotado en el JSON. 38 s en el Mac.

| Modelo (validación cruzada, 1.097 pacientes) | AUC exterior conjunta |
|---|---:|
| Solo cohorte | 0,523 |
| Cohorte + relativas (PE, SER) | 0,536 |
| Cohorte + absolutas (restas) | 0,541 |
| Relativas solas | 0,497 |
| Absolutas solas | 0,512 |

| Comparación pareada | Diferencia | IC95 |
|---|---:|---|
| **Cohorte + relativas − cohorte** (criterio) | +0,014 | [−0,012, +0,039] |
| Cohorte + relativas − cohorte + absolutas | −0,005 | [−0,023, +0,013] |
| Cohorte + absolutas − cohorte | +0,018 | [−0,007, +0,042] |

Dentro de cada cohorte, las relativas dan 0,58 en I-SPY1, 0,49 en I-SPY2 y 0,49 en Duke.

## Decisión tomada

**No se prepara P04.** El realce relativo no añade información demostrable a la cohorte, y tampoco hace mejor papel que las restas absolutas que ya usa la red. Con las medidas de realce que usa la radiología, calculadas sobre el corte entero, no aparece en estos datos una señal sobre pCR que supere a la cohorte.

**Alcance.** Son estadísticos de todo el tejido, sin máscara de tumor; una CNN podría encontrar patrones locales que estas variables no ven. Pero las CNN que ya entrenamos (P02, con restas, en 5 folds) tampoco superaron a la cohorte (+0,009 [−0,025, +0,042]). Dos caminos independientes llegan al mismo techo.

Uso educativo, sin validez clínica.
