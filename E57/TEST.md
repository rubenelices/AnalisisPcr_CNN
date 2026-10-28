# E57 · única evaluación final de TEST

Modelo original de época14, entrenado con878 pacientes y seleccionado con219 de validación. Umbral0,325 y media de probabilidades por paciente fijados antes de abrir test. 176pacientes/1.758cortes de test público. Sin entrenamiento ni ajustes con test.

| Métrica           |   Valor | IC95          |
|:------------------|--------:|:--------------|
| ROC-AUC           |  0.5728 | 0.4753–0.6706 |
| AP                |  0.4206 | 0.3052–0.5548 |
| Sensibilidad      |  0.3396 | 0.2157–0.4630 |
| Especificidad     |  0.748  | 0.6696–0.8261 |
| Precisión pCR     |  0.3673 | 0.2321–0.5116 |
| F1 pCR            |  0.3529 | 0.2273–0.4685 |
| Balanced accuracy |  0.5438 | 0.4690–0.6196 |
| Brier             |  0.2116 | 0.1814–0.2428 |

Matriz: **TN92, FP31, FN35, TP18**.

Los intervalos son bootstrap por paciente, con2.000 remuestreos y semilla42. La validación seleccionó época y umbral; el test mide este checkpoint fijo, no la media de máximos de cinco modelos. Cualquier cambio posterior basado en test convertiría este conjunto en datos de desarrollo.

Uso educativo, sin validez clínica.
