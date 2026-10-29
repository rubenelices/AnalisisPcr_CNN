# Reglas para elegir el modelo de entrega

**Fijadas el 20 de octubre de 2026 por la noche, mientras corren los 10 fits de P02 en la GPU y antes de ver ninguno de sus resultados.** Su SHA-256 queda anotado en [ESTADO.md](ESTADO.md); si este fichero cambia después, se notará. Acordadas con el estudiante.

## Por qué se fijan antes

Tras ver los números es fácil elegir, sin querer, la cifra o la variante que más favorece. Escribirlo antes es lo que permite decir en la defensa que la elección no se ajustó a los resultados.

## Candidatas

- **P02 · BCE normal** y **P02 · BCE ponderada (N0/N1)**: 5 folds cada una, en GPU, con el mismo protocolo (particiones N01; época y umbral elegidos en la validación interna de cada fold).
- **E57**: el modelo de referencia actual, cuyo test ya se midió (0,5728). Sus números honestos son menos precisos: la validación anidada de 5 folds con otra receta (0,527) y un solo fold con selección interna (N01 normal, 0,6165). Su pérdida de entrenamiento se queda en la del predictor constante y sus probabilidades apenas varían.

## Reglas

1. **Número que manda:** la **AUC exterior conjunta** de los 5 folds (predicciones de las 1.097 pacientes, cada una del modelo que no la vio), con IC95 bootstrap. No el pico de validación ni la media de épocas.
2. **No se decide con folds incompletos.** Si algún fit de la GPU falla, se relanza ese fit con el mismo protocolo antes de decidir.
3. **Normal frente a ponderada:** gana la de mayor AUC exterior conjunta. **Si el IC95 pareado de la diferencia incluye 0, se elige la normal**, porque sus probabilidades están mejor calibradas y es la opción por defecto.
4. **P02 frente a E57:** el ganador de la regla 3 sustituye a E57 **salvo que su AUC exterior conjunta sea menor que 0,55**. Por debajo de ese valor ninguna receta es claramente útil y se mantiene E57, con su test ya medido.
5. **Qué se entrega:** el **promedio de probabilidades de los 5 modelos** (el `best.pt` de cada fold) de la receta elegida. Por corte, la probabilidad es la media de los 5 modelos; por paciente, la media de sus cortes, como en toda la evaluación. No se reentrena nada.
6. **Umbral:** la **mediana de los 5 umbrales internos** de la receta elegida (cada uno elegido por F1 macro en su validación interna). 0,325 y 0,5 se muestran solo como referencia.
7. **Test, una sola vez:** después de fijar modelo, agregación y umbral, se evalúa el entregable en las 176 pacientes de test **una vez**, con un script que se niega a repetir. El resultado se reporta tal cual, aunque sea peor que el de E57. No se cambia nada después de verlo.
8. **Qué se reporta:** el test del modelo elegido, la validación cruzada que lo eligió, y el test anterior de E57 (0,5728) como antecedente, sin usarlo para elegir.

Uso educativo, sin validez clínica.
