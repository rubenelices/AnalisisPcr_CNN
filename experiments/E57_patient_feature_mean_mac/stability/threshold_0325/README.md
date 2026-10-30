## E57 · versión con umbral 0,325

|   Fold |    AUC |     AP |   Precisión pCR |   Sensibilidad |   Especificidad |     BA |   F1 macro |   TN |   FP |   FN |   TP |
|-------:|-------:|-------:|----------------:|---------------:|----------------:|-------:|-----------:|-----:|-----:|-----:|-----:|
|      0 | 0.6703 | 0.4225 |          0.439  |         0.5625 |          0.7032 | 0.6329 |     0.6199 |  109 |   46 |   28 |   36 |
|      1 | 0.6299 | 0.4108 |          0.338  |         0.7742 |          0.4013 | 0.5877 |     0.5045 |   63 |   94 |   14 |   48 |
|      2 | 0.6688 | 0.4006 |          0.3272 |         0.8983 |          0.323  | 0.6106 |     0.4773 |   52 |  109 |    6 |   53 |
|      3 | 0.5503 | 0.4161 |          0.5714 |         0.0533 |          0.9793 | 0.5163 |     0.4454 |  142 |    3 |   71 |    4 |
|      4 | 0.5927 | 0.3977 |          0.29   |         0.9355 |          0.0955 | 0.5155 |     0.3066 |   15 |  142 |    4 |   58 |

Conteos de los cinco folds: **TN 381, FP 394, FN 123, TP 199**. Sensibilidad 61.8%; especificidad 49.2%; precisión pCR 33.6%; balanced accuracy 0.5548.

**Cambiar el umbral es válido y no requiere reentrenar.** Aquí 0,325 se ha elegido después de ver estos resultados: esta tabla es exploratoria y no una estimación independiente del rendimiento futuro. La comparación conserva la matriz original a 0,5 y declara cómo se eligió la nueva. Para estimar generalización, fijar el procedimiento de selección usando solo train/validación interna y evaluar en pacientes distintas. El AUC, AP, probabilidades, pesos y curvas por época son los mismos.

Clasificar más positivas también incrementa los falsos positivos. Ningún cambio de umbral puede crear separación entre clases que el modelo no ha aprendido.
