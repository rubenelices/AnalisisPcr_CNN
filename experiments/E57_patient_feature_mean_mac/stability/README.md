# E57 · comprobar estabilidad sin cambiar la CNN

**Estado:** smoke y folds 1–4 completados. [Resultados con figuras visibles](RESULTADOS.ipynb) · [conclusiones](CONCLUSIONES.md). Las instrucciones de ejecución siguientes quedan como registro reproducible, no como indicación de repetir ni sobrescribir estas ejecuciones.

[Auditoría de la matriz y solución que debe evaluarse](DIAGNOSTICO_MATRIZ.md): media por paciente, etiquetas y matrices verificadas; las probabilidades máximas de todos los folds son inferiores a 0,5. El notebook principal incluye matrices a umbrales fijos como diagnóstico, sin presentarlas como mejora validada de forma independiente.

**Vista solicitada con umbral 0,325:** configurada en `decision_0325.json` y guardada en [decision_umbral_0325](threshold_0325/README.md). Incluye métricas y predicciones binarias por fold, además de matrices visibles en el notebook principal. Es una decisión elegida después de observar esta validación y se etiqueta como exploratoria; no modifica las predicciones probabilísticas, checkpoints ni resultados originales a 0,5. Para regenerar solo informes: `python -m src.reporting.e57_stability_results`.

Esta prueba replica **la receta E57 normal original** en los folds 1–4, con semilla 42: CNN propia de 298.161 parámetros, PRE/EARLY−PRE/LATE−EARLY, media de características por paciente en entrenamiento, BCE normal, AdamW LR 0,001/WD 0,001, scheduler Plateau, máximo 50 épocas y paciencia 10. La evaluación principal sigue siendo la **media de probabilidades por paciente**. Solo imágenes; no se usan edad, menopausia, HR ni HER2. No se añaden aumentos ni se alteran intensidades. [Ficha de la arquitectura y diagrama](../README.md).

Hipótesis: si el 0,6703 de E57 fold 0 refleja una señal reproducible, otros folds deberían mantener discriminación apreciable sin que train llegue a AUC≈1 y validación caiga a 0,5. Los cuatro folds están fijados antes de ejecutarse; **no seleccionar después el mejor fold**. Se conserva fold 0 histórico, sin reentrenarlo. El promedio de cinco folds, si se completan, mezcla esta réplica y ese resultado histórico y debe describirse así.

El entrenamiento usa el mismo código de paso E57 original. Solo se agrega una evaluación **sin gradiente y en `model.eval()`** sobre todas las pacientes de train después de cada época; el cargador de diagnóstico tiene generador privado para no alterar el orden ni el azar del entrenamiento. Se guarda `curvas_comparables.png` con BCE y ROC-AUC paciente para train/validación calculados en el mismo modo. `training_curves.png` conserva la vista histórica train online/validación eval, que no debe confundirse con la anterior.

El checkpoint se sigue eligiendo por AUC paciente **del mismo fold que se reporta**. Por eso los máximos son optimistas y esta prueba mide estabilidad descriptiva de la receta original, no una validación exterior independiente. Ya existe un contraste E57 con selección interna distinta cuya media exterior fue 0,5272; no ocultarlo ni tratar esta réplica como sustituto de ese resultado. La matriz usa umbral fijo 0,5; no se buscará un umbral favorable en estos folds. Test permanece cerrado.

## Ejecución manual local

Desde la raíz del proyecto, ejecutar primero el smoke técnico, que no se mezcla con la tabla científica:

```sh
python -m src.train_e57_stability --config experiments/E57_patient_feature_mean_mac/stability/smoke.json --device mps
```

Después, un fold cada vez. Conviene mirar el 1 antes de continuar con el 2, pero el plan está fijado para los cuatro y no se cambiará en función de los resultados:

```sh
python -m src.train_e57_stability --config experiments/E57_patient_feature_mean_mac/stability/fold_1.json --device mps
python -m src.train_e57_stability --config experiments/E57_patient_feature_mean_mac/stability/fold_2.json --device mps
python -m src.train_e57_stability --config experiments/E57_patient_feature_mean_mac/stability/fold_3.json --device mps
python -m src.train_e57_stability --config experiments/E57_patient_feature_mean_mac/stability/fold_4.json --device mps
```

También se puede lanzar cada celda de [EJECUTAR.ipynb](EJECUTAR.ipynb). Al completarse un fold científico, el runner actualiza [RESULTADOS general](../results/RESULTADOS.ipynb) y el catálogo `notebooks/07_RESULTADOS_CLAROS.ipynb`. [RESULTADOS propio](RESULTADOS.ipynb) reúne sus tablas y las curvas comparables. Si se interrumpe una ejecución, conserva su carpeta parcial y no se reanuda ni sobrescribe automáticamente. Los smokes y entrenamientos se lanzan manualmente.
