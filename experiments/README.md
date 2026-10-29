# Catálogo de arquitecturas

Cada carpeta reúne una arquitectura distinta, su explicación y un dibujo del recorrido real de sus capas. El código ejecutable continúa en `src/` y los experimentos de entrenamiento en `configs/experiments/`.

| Arquitectura | Experimentos | Orden dentro del bloque | Parámetros |
|---|---|---|---:|
| [A01 · Mínima](A01_minimal/README.md) | Memorización de depuración | Conv → ReLU → Pool | 6.545 |
| [A02 · Dos convoluciones](A02_two_convolutions/README.md) | E02 y E03 | Conv → Conv → Pool | 294.129 |
| [A03 · Una convolución](A03_one_convolution/README.md) | E04 | Conv → Pool | 97.809 |
| [A04 · Pool intermedio](A04_intermediate_pool/README.md) | E05, E07/E08 (regularización), E09 (aumentos) | Conv → Pool → Conv | 294.129 |
| [A05 · Cinco bloques](A05_five_blocks/README.md) | E06 | Conv → Pool → Conv, quinto bloque añadido | 589.553 |
| [E14 · Dropout espacial 0,1](E14_spatial_dropout_010/README.md) | E14 | Conv → Pool → Conv → Dropout2d 0,1 | 294.129 |
| [E15 · Dropout espacial 0,2](E15_spatial_dropout_020/README.md) | E15 | Conv → Pool → Conv → Dropout2d 0,2 | 294.129 |
| [E56 · Cabeza no lineal por corte](E56_nonlinear_cut_control_mac/README.md) | E56 (control de E57) | Conv → Conv → Pool, cabeza 128→32→1 | 298.161 |
| [**E57 · Media de características por paciente**](E57_patient_feature_mean_mac/README.md) | **Modelo entregado** · [resultados y test](E57_patient_feature_mean_mac/results/README.md) · [estabilidad](E57_patient_feature_mean_mac/stability/README.md) | Conv → Conv → Pool, cabeza 128→32→1 | 298.161 |
| [E58 · CNN + clínica](E58_clinical_fusion_mac/README.md) | E58 | CNN E19 + MLP clínica | 294.289 |
| [D36 · Imágenes, edad y menopausia](D36_age_menopause/README.md) | D36 · [ejecución](D36_age_menopause/run/README.md) | CNN E19 + edad/menopausia | 294.289 |

En A02, A03, A04 y A05 cada convolución va seguida de BatchNorm y ReLU. E02 y E03 usan la misma red con pérdidas distintas, así que no duplicamos su dibujo. Las alturas y anchuras de los volúmenes son esquemáticas: las dimensiones exactas aparecen bajo cada capa.

E14/E15 mantienen BatchNorm y ReLU tras cada convolución; el Dropout2d se aplica
al final del bloque, solo durante train. [Hipótesis, controles y comandos](notes/DROPOUT_ESPACIAL.md).

Abre la vista previa del README de cada carpeta en Visual Studio Code. Cada ficha incluye `arquitectura.png` para presentaciones y `arquitectura.svg` para ampliar sin perder calidad.

Para regenerar todas las imágenes y fichas a partir de las redes:

```bash
python -m src.reporting.document_architectures
```

Para regenerar únicamente una ficha: `python -m src.reporting.document_architectures --only A05_cinco_bloques`.

[Protocolo E05/E06 y siguientes pasos hacia ROC-AUC 0,7](notes/PROTOCOLO_COMPARACION.md).

## Variantes de regularización de A04

- [E07 · Dropout 0,5](E07_dropout_050/README.md): solo cambia el dropout, con su dibujo actualizado.
- [E08 · Weight decay 0,001](E08_weight_decay_001/README.md): mantiene dropout 0,3; el cambio está en AdamW, no en las capas.
- [Comparación, comandos y conservación de resultados](notes/REGULARIZACION.md).

## Aumentos de entrenamiento

- [E09 · Giros y traslaciones compartidos](E09_shared_affine/README.md): misma CNN de E05; no modifica validación ni web.
- [Protocolo, pruebas y vista original/aumentada](notes/AUMENTOS_E09.md).

## Learning rate

- [E11 · LR inicial 0,0003](E11_lr_0003/README.md) y [E12 · LR inicial 0,0001](E12_lr_0001/README.md): misma arquitectura de E05 Mac, sin otros cambios de ajustes.
- [Hipótesis, comandos y comparación](notes/LEARNING_RATE.md).

## Registro y diagnósticos

- [Notebook 05 · Diario de experimentos y resultados](notes/05_resultados_experimentos.ipynb): tablas históricas, curvas locales, diagnóstico BN y cohortes; no entrena ni abre test.
- [Respaldo agregado de las cifras comunicadas](notes/resultados_registrados.json): identifica fuente, plataforma y ejecución; no contiene datos individuales.
- [Respaldo agregado por cohorte](notes/cohortes_registradas.json): conserva las comparaciones E08/E10 y E05/E08/E10 sin archivos de pacientes.

- [Referencia E05 en Mac y diagnóstico por cohorte](notes/COHORTES_Y_REFERENCIA_MAC.md): replica con rutas separadas y comparación de predicciones existentes sin leer imágenes.

- [A06 · Mitad de canales, misma profundidad](A06_half_channels/README.md): E10 reduce capacidad manteniendo las ocho convoluciones de E05.
- [E10 · Hipótesis, comparación y ejecución en Mac](notes/ANCHURA_E10.md).

[Train frente a validación y sensibilidad a BatchNorm](notes/DIAGNOSTICO.md): análisis
sin volver a entrenar ni tocar test. Comienza con el mejor checkpoint de E05.

Toda arquitectura nueva debe incorporarse a este catálogo con hipótesis, orden de operaciones, dimensiones, parámetros, campo receptivo y configuraciones asociadas. Los resultados por paciente se guardan en `reports/`, que permanece fuera de Git.

- [E13 · Realce explícito entre fases](E13_phase_differences/README.md): misma CNN y ajustes E05 Mac; operación fija PRE, EARLY−PRE, LATE−EARLY dentro del modelo, sin parámetros nuevos.
- [Protocolo y ejecución E13](notes/REALCE_E13.md): compatibilidad train/evaluación/web, prueba técnica y comparación prevista.
- [Diagnóstico E13](notes/DIAGNOSTICO_E13.md): diferencia train/validación y copia BatchNorm; sin entrenamiento ni test.
- [E13 · Prueba de 50 épocas](notes/E13_ENTRENAMIENTO_LARGO.md): misma arquitectura, paciencia 51 y rutas nuevas; autorizada por el estudiante, sin resultados todavía.

Uso educativo y de investigación, sin validez clínica.
