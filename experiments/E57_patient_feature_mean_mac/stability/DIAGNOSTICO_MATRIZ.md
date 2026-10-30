# E57 · decisión y matriz de confusión

## Por qué la matriz tiene ceros en la columna positiva

|   Fold |   pCR reales |   Prob. mínima |   Prob. media |   Prob. máxima |   Predicciones positivas a 0,5 |
|-------:|-------------:|---------------:|--------------:|---------------:|-------------------------------:|
|      0 |           64 |         0.0093 |        0.289  |         0.3595 |                              0 |
|      1 |           62 |         0.008  |        0.3348 |         0.4097 |                              0 |
|      2 |           59 |         0.0116 |        0.3213 |         0.3663 |                              0 |
|      3 |           75 |         0.0844 |        0.2792 |         0.3433 |                              0 |
|      4 |           62 |         0.2352 |        0.3477 |         0.3826 |                              0 |

La matriz y la media de probabilidades por paciente se han recalculado y coinciden con los ficheros originales. Todas las probabilidades están por debajo de 0,5: con ese umbral todas las pacientes se clasifican como 0. Esto no prueba por sí solo un error de software ni mala calibración; con prevalencia cercana al 29 % y señal limitada puede ocurrir incluso con probabilidades razonables.

Las matrices siguientes usan 0,5 y dos umbrales fijos que ya figuraban en el diagnóstico previo. Son ilustrativas sobre validación usada para seleccionar época. **No son una mejora validada de forma independiente.** La versión 0,325 se identifica aparte como exploratoria. Bajar el umbral permite detectar positivas, pero también genera falsos positivos y no aumenta el AUC. La escala cambia entre folds, por lo que un único 0,325 no arregla todas las matrices.

**Solución que debe evaluarse:** mantener E57, reservar validación interna por paciente dentro de cada train, elegir allí checkpoint y umbral con una regla prefijada (F1 macro; desempate por balanced accuracy y proximidad a 0,5), y evaluar ambos umbrales en el fold exterior sin reajustarlos. La calibración es una evaluación aparte; no es necesaria para elegir un umbral ni garantiza que aparezcan probabilidades superiores a 0,5. Si persiste poca separación, comparar BCE normal con ponderada N0/N1 calculada solo en FIT, manteniendo la CNN. No se inicia entrenamiento.

La ponderación moderada E57 ya probada también daba todas negativas a 0,5. El contraste anterior con selección interna alcanzó AUC exterior media 0,5272 con otra receta; tampoco podemos prometer que un nuevo umbral resuelva su discriminación. Test sigue reservado.


### Todos los umbrales fijos explorados

|   Fold |   Umbral |   Positivas predichas | Precisión pCR   |   Sensibilidad |   Especificidad |   F1 macro |   Balanced accuracy |   TP |   FN |   FP |   TN |
|-------:|---------:|----------------------:|:----------------|---------------:|----------------:|-----------:|--------------------:|-----:|-----:|-----:|-----:|
|      0 |    0.25  |                   180 | 0.333           |          0.938 |           0.226 |      0.426 |               0.582 |   60 |    4 |  120 |   35 |
|      0 |    0.3   |                   156 | 0.346           |          0.844 |           0.342 |      0.489 |               0.593 |   54 |   10 |  102 |   53 |
|      0 |    0.325 |                    82 | 0.439           |          0.562 |           0.703 |      0.62  |               0.633 |   36 |   28 |   46 |  109 |
|      0 |    0.35  |                     4 | 0.250           |          0.016 |           0.981 |      0.426 |               0.498 |    1 |   63 |    3 |  152 |
|      0 |    0.4   |                     0 | No definida     |          0     |           1     |      0.414 |               0.5   |    0 |   64 |    0 |  155 |
|      0 |    0.425 |                     0 | No definida     |          0     |           1     |      0.414 |               0.5   |    0 |   64 |    0 |  155 |
|      0 |    0.45  |                     0 | No definida     |          0     |           1     |      0.414 |               0.5   |    0 |   64 |    0 |  155 |
|      0 |    0.5   |                     0 | No definida     |          0     |           1     |      0.414 |               0.5   |    0 |   64 |    0 |  155 |
|      1 |    0.25  |                   200 | 0.290           |          0.935 |           0.096 |      0.307 |               0.516 |   58 |    4 |  142 |   15 |
|      1 |    0.3   |                   171 | 0.316           |          0.871 |           0.255 |      0.427 |               0.563 |   54 |    8 |  117 |   40 |
|      1 |    0.325 |                   142 | 0.338           |          0.774 |           0.401 |      0.505 |               0.588 |   48 |   14 |   94 |   63 |
|      1 |    0.35  |                   110 | 0.336           |          0.597 |           0.535 |      0.531 |               0.566 |   37 |   25 |   73 |   84 |
|      1 |    0.4   |                     3 | 0.667           |          0.032 |           0.994 |      0.449 |               0.513 |    2 |   60 |    1 |  156 |
|      1 |    0.425 |                     0 | No definida     |          0     |           1     |      0.418 |               0.5   |    0 |   62 |    0 |  157 |
|      1 |    0.45  |                     0 | No definida     |          0     |           1     |      0.418 |               0.5   |    0 |   62 |    0 |  157 |
|      1 |    0.5   |                     0 | No definida     |          0     |           1     |      0.418 |               0.5   |    0 |   62 |    0 |  157 |
|      2 |    0.25  |                   206 | 0.277           |          0.966 |           0.075 |      0.284 |               0.52  |   57 |    2 |  149 |   12 |
|      2 |    0.3   |                   188 | 0.303           |          0.966 |           0.186 |      0.386 |               0.576 |   57 |    2 |  131 |   30 |
|      2 |    0.325 |                   162 | 0.327           |          0.898 |           0.323 |      0.477 |               0.611 |   53 |    6 |  109 |   52 |
|      2 |    0.35  |                    21 | 0.333           |          0.119 |           0.913 |      0.496 |               0.516 |    7 |   52 |   14 |  147 |
|      2 |    0.4   |                     0 | No definida     |          0     |           1     |      0.423 |               0.5   |    0 |   59 |    0 |  161 |
|      2 |    0.425 |                     0 | No definida     |          0     |           1     |      0.423 |               0.5   |    0 |   59 |    0 |  161 |
|      2 |    0.45  |                     0 | No definida     |          0     |           1     |      0.423 |               0.5   |    0 |   59 |    0 |  161 |
|      2 |    0.5   |                     0 | No definida     |          0     |           1     |      0.423 |               0.5   |    0 |   59 |    0 |  161 |
|      3 |    0.25  |                   183 | 0.350           |          0.853 |           0.179 |      0.391 |               0.516 |   64 |   11 |  119 |   26 |
|      3 |    0.3   |                    79 | 0.380           |          0.4   |           0.662 |      0.53  |               0.531 |   30 |   45 |   49 |   96 |
|      3 |    0.325 |                     7 | 0.571           |          0.053 |           0.979 |      0.445 |               0.516 |    4 |   71 |    3 |  142 |
|      3 |    0.35  |                     0 | No definida     |          0     |           1     |      0.397 |               0.5   |    0 |   75 |    0 |  145 |
|      3 |    0.4   |                     0 | No definida     |          0     |           1     |      0.397 |               0.5   |    0 |   75 |    0 |  145 |
|      3 |    0.425 |                     0 | No definida     |          0     |           1     |      0.397 |               0.5   |    0 |   75 |    0 |  145 |
|      3 |    0.45  |                     0 | No definida     |          0     |           1     |      0.397 |               0.5   |    0 |   75 |    0 |  145 |
|      3 |    0.5   |                     0 | No definida     |          0     |           1     |      0.397 |               0.5   |    0 |   75 |    0 |  145 |
|      4 |    0.25  |                   217 | 0.286           |          1     |           0.013 |      0.235 |               0.506 |   62 |    0 |  155 |    2 |
|      4 |    0.3   |                   207 | 0.285           |          0.952 |           0.057 |      0.273 |               0.504 |   59 |    3 |  148 |    9 |
|      4 |    0.325 |                   200 | 0.290           |          0.935 |           0.096 |      0.307 |               0.516 |   58 |    4 |  142 |   15 |
|      4 |    0.35  |                   121 | 0.355           |          0.694 |           0.503 |      0.545 |               0.598 |   43 |   19 |   78 |   79 |
|      4 |    0.4   |                     0 | No definida     |          0     |           1     |      0.418 |               0.5   |    0 |   62 |    0 |  157 |
|      4 |    0.425 |                     0 | No definida     |          0     |           1     |      0.418 |               0.5   |    0 |   62 |    0 |  157 |
|      4 |    0.45  |                     0 | No definida     |          0     |           1     |      0.418 |               0.5   |    0 |   62 |    0 |  157 |
|      4 |    0.5   |                     0 | No definida     |          0     |           1     |      0.418 |               0.5   |    0 |   62 |    0 |  157 |
