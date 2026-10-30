# Cierre y evaluación final · 20 de octubre de 2026

**Test ejecutado una única vez el 20 de octubre.** Se usó exactamente el checkpoint original de época14, con SHA verificado, media de probabilidades y umbral0,325 fijados antes de leer test. Inferencia local CPU sobre176pacientes/1.758cortes, sin entrenamiento ni cambios en pesos. AUC0,5728 (IC95%0,4753–0,6706); AP0,4206; matrizTN92/FP31/FN35/TP18. [Resultados completos](TEST.ipynb) · [Resumen](TEST.md). Los apartados siguientes conservan la propuesta previa como registro. No se volverá a usar este test para seleccionar versiones y presentar luego esas comparaciones como independientes.

La prioridad pasa de buscar arquitecturas a terminar una evaluación reproducible de E57. No es obligatorio ejecutar otra campaña de validación para poder evaluar test. La selección usando los folds de train es desarrollo; el test público debe permanecer fuera de esas decisiones y evaluarse una vez cuando se cierre el procedimiento.

## Propuesta concreta por el tiempo disponible

Usar **la referencia original de E57 ya entrenada**, checkpoint de la época 14, con 298.161 parámetros. Este modelo aprendió con 878 pacientes y usó 219 de validación para seleccionar época; no se presentará como entrenado con las 1.097 pacientes. No se elige un modelo por su rendimiento en test. Los otros cuatro modelos se conservan para describir la estabilidad de la arquitectura.

La entrada sigue siendo PRE/EARLY/LATE, con transformación interna PRE/EARLY−PRE/LATE−EARLY. Solo imágenes, sin edad ni menopausia. Inferencia por corte, sigmoide y **media de probabilidades de todos los cortes de cada paciente**. Umbral de decisión **0,325**, elegido durante el desarrollo y fijado antes de mirar test. La elección posterior a observar validación se declara; no invalida una evaluación posterior en un test que no participó en la elección.

Checkpoint propuesto: `checkpoints/E57_patient_feature_mean_mac/fold_0_seed_42/best.pt`. SHA-256 verificado: `a1fb12dfae2c3d6b730ae0b1d916727981154116077e40aa64799415b8cfde5f`.

## Pasos que faltan

1. Cerrar el checkpoint y la regla de inferencia anteriores antes de ejecutar test. Esta es una propuesta concreta, todavía no un registro de evaluación final realizada.
2. Preparar el comando de evaluación final para el cargador de E57. El evaluador genérico actual está reservado a validación y usa otra clase de modelo; no debe reutilizarse apuntándolo directamente a test.
3. Se ejecutará una sola evaluación local sobre las 176 pacientes públicas de test. Guardar ROC-AUC, AP, sensibilidad, especificidad, precisión, F1, balanced accuracy, matriz y calibración, con intervalos por paciente. Mantener el umbral prefijado aunque el resultado sea peor de lo esperado.
4. Incorporar ese resultado al notebook principal y a esta carpeta. La comparación train/validación/test debe referirse al mismo checkpoint; 0,6224 es la media de máximos de cinco modelos y no el AUC de validación de este único checkpoint.
5. Integrar el mismo cargador, preprocesamiento, agregación y umbral en la demostración. Si la demostración recibe un único corte, identificar esa limitación: no es la media de todos los cortes de la paciente.

Entrenar una red final con las 1.097 pacientes es una alternativa, no una condición para evaluar el modelo existente. Exigiría fijar de antemano las épocas y el calendario de aprendizaje, producir nuevos pesos y evaluar esos pesos, no reutilizar como si fueran equivalentes los resultados del checkpoint actual. Por el plazo disponible, esta propuesta prioriza cerrar el modelo existente.

**Estado al preparar la propuesta:** todavía no se habían leído imágenes ni calculado predicciones de test. La ejecución posterior se registra al principio de este documento. Uso educativo, sin validez clínica.
