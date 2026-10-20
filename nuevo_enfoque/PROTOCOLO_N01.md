# N01 · E57 con BCE normal frente a ponderada completa

Protocolo fijado antes de ejecutar N01. Fecha: 20 de octubre de 2026.

## Pregunta y alcance

¿Cómo cambia el comportamiento de la misma E57 al usar BCE normal o `pos_weight=N0/N1` calculado únicamente con las pacientes FIT? Se comparan discriminación, calibración y clasificación, sin exigir que la ponderada mejore AUC. No se busca otra arquitectura ni se cambia el umbral después de observar la evaluación exterior.

E57 tiene 298.161 parámetros, se inicializa desde cero y utiliza solo imágenes. Se conserva la representación PRE, EARLY−PRE, LATE−EARLY; media de características en entrenamiento y media de probabilidades por corte en evaluación. Esta diferencia se conserva para aislar la pérdida, no se declara resuelta mediante este contraste. No se incorporan edad, menopausia ni otros datos clínicos.

## Particiones y selección

1. Solo se utiliza el train público de 1.097 pacientes. Los cinco folds docentes se mantienen intactos. Cada paciente aparece en un único fold exterior y conserva todos sus cortes.
2. Para cada fold se reserva su parte exterior. Del resto se separa un 20 % de pacientes para validación interna, estratificando pCR y con semilla 42. Se congela la lista de pacientes FIT/interna/exterior antes del primer entrenamiento. La misma lista se utiliza en las dos pérdidas. No se seleccionan particiones por sus resultados.
3. Solo FIT actualiza pesos y BatchNorm. La validación interna selecciona época, scheduler y umbral. La parte exterior no se evalúa durante las épocas.
4. Se conserva el primer checkpoint que alcanza el mejor AUC interno observado. Empates exactos: primera época. Máximo 50 épocas, parada tras 10 sin superar el máximo. No se amplía el presupuesto al observar resultados desfavorables.
5. Restaurado ese checkpoint, se elige un único umbral con las probabilidades internas: mayor F1 macro, después mayor balanced accuracy, después menor distancia a 0,5 y finalmente menor umbral. Se exploran 0, 0,5, 1 y los puntos medios entre probabilidades distintas; decisión `p >= umbral`. Esa regla se fija antes de las evaluaciones exteriores.
6. Se guarda la decisión (época, SHA de los pesos, umbral y métricas internas) antes de cargar imágenes exteriores. Después se evalúa el exterior una vez con ese checkpoint y se muestran dos matrices: umbral 0,5 y umbral interno congelado. No se modifica la decisión con ninguna de ellas.

Los folds exteriores **ya se han observado en la historia del proyecto**. Esta separación evita que N01 seleccione su época con su propio exterior, pero no borra el historial de selección de arquitectura. Se denomina evaluación exterior de desarrollo, nunca test independiente nuevo. Se conserva el test público ya realizado de E57 original; no se vuelve a ejecutar ni se asigna a N01.

## Receta compartida

| Elemento | Decisión |
|---|---|
| CNN, representación y cabeza | Las de E57 original, sin modificaciones |
| Lote de entrenamiento | Dos pacientes completas; cada paciente una vez por época |
| Aumentos y normalización extra | Ninguno; intensidades docentes conservadas |
| Optimizador | AdamW, LR 0,001, weight decay 0,001 |
| Scheduler | ReduceLROnPlateau, AUC paciente interno, factor 0,5, paciencia 3 |
| Semilla | 42, reiniciada en cada condición/fold |
| Presupuesto | Máximo 50 épocas; paciencia 10; máximo 10 fits científicos |
| Normal | BCEWithLogitsLoss, peso positivo 1 |
| Ponderada | BCEWithLogitsLoss, peso positivo N0/N1 de pacientes FIT, sin raíz |
| Evaluación | Modo eval, sin aumentos; mismos cortes y agregación por paciente |

Los generadores de lotes y evaluación son explícitos. La evaluación no consume el generador de dropout/entrenamiento. Ambas condiciones empiezan con los mismos pesos y orden de pacientes; pueden seguir trayectorias distintas al aplicar la pérdida y el scheduler. La reproducibilidad numérica entre dispositivos no se garantiza.

## Métricas y presentación

Se guardan todas las épocas con pérdida de entrenamiento online, BCE normal por paciente en train-eval e interna-eval, AUC train/interna y LR. No se compara como si fuera la misma magnitud la pérdida online ponderada con una BCE normal eval.

Al cerrar un fold se muestran las tres gráficas de referencia: BCE comparable por época, AUC por época y barras FIT/interna/exterior del mismo checkpoint. También ROC, precision-recall, calibración y matrices de ambas decisiones. No se dibuja una curva exterior por época ni una barra de test para N01.

Por paciente: ROC-AUC, AP (average precision), Brier, sensibilidad, especificidad, precisión, F1 de ambas clases, F1 macro, balanced accuracy y matriz. La precisión sin predicciones positivas se marca como indefinida aunque el respaldo numérico utilice cero. Se incluye el clasificador siempre negativo como referencia.

El resumen final requiere los cinco folds completos en **ambas** condiciones. Presentará media y desviación entre folds y diferencias ponderada−normal por el mismo fold. No se tratarán las épocas o cortes como observaciones independientes. Los intervalos bootstrap por paciente, si se muestran, son condicionales a los modelos entrenados y no representan la incertidumbre de todo el proceso de selección. Un aumento del número de positivos predichos no basta para declarar mejora.

## Integridad y ejecución

La preparación congela configuración, fuentes y particiones en `reports/new_approach/N01_e57_bce/protocol/`. El runner comprueba esos SHA antes de entrenar. Los respaldos técnicos permanecen fuera de esta carpeta de lectura. Los pesos nuevos van a `checkpoints/new_approach/N01_e57_bce/` y nunca sustituyen pesos antiguos.

Antes de los entrenamientos científicos, el estudiante ejecuta un smoke por condición en fold 0: una época con cuatro pacientes por clase en FIT y dos por clase en interna/exterior. Las particiones de los cinco folds se verifican durante la preparación. El smoke es una comprobación técnica, queda separado y no cuenta como resultado científico. Cada ejecución científica exige el smoke de su condición completo y compatible. No se sobrescriben ni reanudan ejecuciones parciales automáticamente.

Los entrenamientos y smokes se lanzan manualmente en CPU/MPS local. No se usa nube, CUDA remota, supervisores ni colas antiguas. Este protocolo no autoriza una campaña indefinida. Los resultados se incorporan automáticamente al [notebook activo](RESULTADOS.ipynb), conservando sus antecedentes y notas.

Uso educativo, sin validez clínica.
