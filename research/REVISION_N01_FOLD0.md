# N01 · revisión de la primera pareja, fold 0

Ambas ejecuciones manuales están completas: BCE normal, 12 épocas; ponderada completa, 21 épocas. Comparten protocolo, inicialización y pacientes: 702 FIT, 176 internas y 219 exteriores (64 pCR). Se verificaron el SHA de ambos checkpoints elegidos, las decisiones guardadas y la reproducción de las métricas desde las predicciones ya almacenadas. No se ejecutó nueva inferencia ni se consultó test.

## Resultado exterior de desarrollo

La época y el umbral de cada condición se eligieron con sus pacientes internas, antes de evaluar el exterior.

| Medida | BCE normal | BCE ponderada |
|---|---:|---:|
| Época elegida | 2 de 12 | 11 de 21 |
| AUC FIT-eval en esa época | 0,5052 | 0,5480 |
| AUC interna que seleccionó la época | 0,5721 | 0,6497 |
| **AUC exterior** | **0,6165** | **0,5475** |
| AP exterior | 0,4117 | 0,3244 |
| Brier exterior (menor es mejor) | 0,20475 | 0,24472 |
| Umbral elegido en interna | 0,263949 | 0,502931 |
| Sensibilidad exterior | 51,56 % | 37,50 % |
| Especificidad exterior | 68,39 % | 67,10 % |
| Precisión pCR exterior | 40,24 % | 32,00 % |
| F1 macro exterior | 0,5890 | 0,5205 |
| Balanced accuracy exterior | 0,5997 | 0,5230 |
| TN / FP / FN / TP | 106 / 49 / 31 / 33 | 104 / 51 / 40 / 24 |

La normal es mejor en todas esas métricas exteriores en este fold. La ponderada obtuvo un pico interno de 0,6497, pero ese valor no se mantuvo en el exterior. Esto ilustra por qué no debemos elegir una receta solo por su máximo de validación interna. No demuestra que la normal gane en otros folds ni una diferencia estadísticamente concluyente.

La ponderada a umbral fijo 0,5 obtiene TN86/FP69/FN31/TP33. Su umbral interno 0,502931 reduce las predicciones positivas: TN104/FP51/FN40/TP24. No se cambia ese umbral tras observar cuál de las dos matrices exteriores parece mejor.

La normal a 0,5 predice todo negativo (TN155/FP0/FN64/TP0). Con el umbral elegido en la interna, detecta 33 de las 64 positivas y genera 49 falsas alarmas. El AUC no cambia al modificar el umbral. Esta vez el corte no se ha elegido mirando la matriz exterior ni se ha reutilizado el 0,325 histórico.

## Qué dicen las curvas

No aparece el patrón clásico de memorización en el que FIT-eval se acerca a AUC1 mientras empeora la interna. Aquí FIT-eval permanece aproximadamente entre 0,47 y 0,58. Las curvas cercanas no equivalen por sí solas a un modelo bueno: también pueden acompañar una discriminación débil o inestable. Estas cifras no identifican una causa única.

En la normal, las BCE de FIT e interna bajan y después oscilan cerca de 0,60–0,62. En la ponderada, ambas BCE normales suben hacia 0,69–0,70. Esta última curva mide BCE **sin ponderar**, mientras el entrenamiento optimiza BCE ponderada. La ponderación puede desplazar las probabilidades hacia valores mayores; no debe atribuirse toda la subida a sobreajuste clásico. Su Brier exterior sí es peor: 0,24472 frente a 0,20684 del predictor constante basado en prevalencia de FIT. La normal mejora muy ligeramente ese predictor constante: 0,20475 frente a 0,20684.

## Decisión de trabajo

Conservar esta pareja como comparación preliminar y mantener E57 fija. La normal es la opción más favorable **en este fold**, pero no se sustituye automáticamente el modelo de entrega ni se le asigna el test del E57 original.

No se lanzan los ocho entrenamientos restantes en bloque ni se cambian hiperparámetros para perseguir el pico interno. Si se continúa, el siguiente paso es la pareja del fold 1 con el mismo protocolo, para comprobar si el resultado se repite. Si se decide terminar aquí, se presentará como comparación de un solo fold, sin media de cinco folds ni ganador general.

Las dos ejecuciones sumaron aproximadamente 48 minutos según los tiempos registrados, incluyendo las evaluaciones y la actualización de resultados. Sus 33 épocas, curvas por paciente/corte y matrices están en [RESULTADOS.ipynb](RESULTADOS.ipynb). La referencia E57 original y su test se conservan como antecedentes separados.

Uso educativo, sin validez clínica.
