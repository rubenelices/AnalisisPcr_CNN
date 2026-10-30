# P01 · lotes con más pacientes (BatchNorm)

Preparado el 20 de octubre de 2026. Ejecución manual en local. [Inicio](README.md) · [Estado](ESTADO.md).

## Por qué

Las curvas de E57 (5 folds) y de N01 muestran que la red **no aprende**. La pérdida de entrenamiento se queda en 0,59–0,63, y 0,605 es la BCE de un predictor que siempre responde la prevalencia (29,3 % pCR). En la ponderada la pérdida es 0,979, el óptimo constante exacto. Las probabilidades por paciente casi no varían (desviación típica 0,02–0,07) y el AUC en FIT-eval está en torno a 0,50–0,56. Los picos de validación (0,670 en E57 fold0) caben en la variación aleatoria de 219 pacientes (error típico ≈ 0,043).

**Hipótesis.** Con 2 pacientes por lote, BatchNorm normaliza cada canal usando solo esas dos pacientes. La media de características de ambas queda forzada a compensarse, así que la red solo puede aprender diferencias *dentro* del lote. En el ~59 % de los lotes las dos pacientes tienen la misma etiqueta y no hay nada que discriminar. En eval() se usan estadísticas acumuladas distintas de las del entrenamiento.

## Qué cambia (y qué no)

| | E57 / N01 | P01 |
|---|---|---|
| Pacientes por lote | 2 | **10** |
| Cortes por paciente y lote | todos (~10) | **2 al azar**, distintos en cada recorrido |
| Imágenes por paso | ~20 | 20 |
| Pasos por época (fold 0) | 351 | 355 (5 recorridos de las 702 pacientes) |

**Sin cambios:** red E57 (298.161 parámetros), particiones N01 (702 FIT / 176 interna / 219 exterior), AdamW con LR 0,001 y WD 0,001, plateau, máximo 50 épocas con paciencia 10, selección por AUC interna, umbral por F1 macro interno y media de probabilidades por paciente en evaluación. El sorteo de cortes es un efecto inevitable del cambio: no se puede tener más pacientes por lote con el mismo número de imágenes sin tomar menos cortes de cada una.

Comparte la ficha de arquitectura de E57: no es una red nueva.

## Paso 1 · prueba de memorización (diagnóstico, ~10 min cada una)

40 pacientes FIT del fold 0 (20 pCR y 20 no pCR). Las dos recetas hacen 30 épocas con 20 pasos cada una y LR constante. Se mide el AUC en eval() sobre esas mismas 40 pacientes. No es un resultado de rendimiento.

```bash
python -m src.training.train_p01 overfit --recipe e57
```

```bash
python -m src.training.train_p01 overfit --recipe p01
```

```bash
python -m src.training.train_p01 overfit-report
```

La figura queda en `reports/new_approach/P01_lotes_pacientes/overfit/comparacion_overfit.png`.

**Criterios fijados antes de ejecutar:**

- **Hipótesis confirmada:** P01 llega a AUC eval ≥ 0,90 con pérdida online < 0,45, y E57 se queda en AUC eval < 0,75 con pérdida cerca de 0,60. → Pasar al paso 2.
- **Ambas memorizan:** la receta no explica el colapso con 702 pacientes; la causa es otra (señal débil frente a regularización). → No lanzar el paso 2; revisarlo antes.
- **Ninguna memoriza:** el problema está en otro sitio (LR, cabeza, entrada). → Parar y diagnosticar.

### Resultado del paso 1 (ejecutado, 20 de octubre, ~2 min cada receta)

![Comparación de memorización](../reports/new_approach/P01_lotes_pacientes/overfit/comparacion_overfit.png)

| | E57: 2 pacientes × 10 cortes | P01: 10 pacientes × 2 cortes |
|---|---|---|
| AUC eval en la época 4 | 0,695 | **1,000** |
| Primera época con AUC eval ≥ 0,90 | 23 | **3** |
| AUC eval en la época 30 | 0,943 | 1,000 |
| Pérdida online (épocas 26–30) | 0,54–0,61, cerca del constante | 0,03–0,16 |
| Pérdida online < 0,45 | nunca | desde la época 4 |
| Dispersión de probabilidades al final | 0,23 | 0,48 |

**Lectura frente a los criterios.** P01 cumple su parte con holgura. E57 **no** cumple estrictamente la suya: supera AUC 0,75 desde la época 16 y llega a 0,94. Es un caso intermedio que los criterios no previeron. E57 sí aprende, pero unas 8 veces más despacio y de forma ruidosa. Además, su pérdida en modo entrenamiento no baja aunque la AUC en eval() sí suba: la discrepancia entre las estadísticas BatchNorm del lote y las acumuladas que plantea la hipótesis.

Lo que **no** demuestra: que E57 sea incapaz de aprender, ni que P01 vaya a generalizar. P01 memoriza 40 pacientes en 4 épocas, así que con 702 el riesgo es caer en el régimen D36 (FIT → 1 y validación hacia abajo).

**Decisión:** pasar al paso 2. Con las 702 pacientes, E57 nunca superó 0,58 en FIT-eval en ~8.000 pasos, mientras que aquí la diferencia de velocidad es de un orden de magnitud. La pregunta que queda solo la responde el fold 0 completo, que vigila la brecha FIT–interna. Esta desviación del criterio literal se deja escrita aquí; no se reinterpretó en silencio.

## Paso 2 · fold 0 con BCE normal (solo si el paso 1 confirma)

```bash
python -m src.training.train_p01 smoke --loss normal --fold 0
```

```bash
python -m src.training.train_p01 run --loss normal --fold 0
```

Cada época imprime el AUC por paciente **y por corte** en FIT-eval y en la interna, además de las BCE. Al cerrar se imprime el exterior una sola vez. Para la tabla:

```bash
python -m src.training.train_p01 report
```

**Criterios fijados antes de ejecutar** (referencia: N01 normal fold 0, exterior 0,6165, con época elegida en FIT-eval 0,505):

1. **Aprende:** en la época elegida, AUC FIT-eval > 0,60 y pérdida online claramente por debajo de 0,605.
2. **Generaliza:** la AUC interna sigue a la de FIT sin hundirse. Si FIT-eval sube hacia 1 mientras la interna cae, es el régimen de memorización de D36, y la siguiente palanca es la regularización/aumento, no la arquitectura.
3. **Merece los folds 1–4:** AUC exterior ≥ 0,62 por paciente, con brecha FIT−interna < 0,15. Un solo fold tiene error típico ≈ 0,043: no se declara ganador con él.
4. **Demo:** se registra el AUC exterior **por corte**, porque la evaluación en directo usa un corte por paciente.

### Resultado del paso 2 (ejecutado, 20 de octubre, 16 épocas, 23 min)

| Época | Pérdida online | AUC FIT-eval | AUC interna | BCE interna |
|---:|---:|---:|---:|---:|
| 2 | 0,613 | 0,567 | 0,597 | 0,595 |
| **6 (elegida)** | 0,603 | 0,593 | **0,611** | 0,601 |
| 10 | 0,576 | 0,722 | 0,519 | 0,628 |
| 13 | 0,530 | 0,836 | 0,556 | 0,690 |
| 16 | 0,435 | 0,938 | 0,565 | 0,763 |

En la época elegida, el exterior (219 pacientes, 64 pCR) da AUC **0,570** por paciente y 0,569 por corte, AP 0,326 y Brier 0,206. Con el umbral interno 0,323: TN 110, FP 45, FN 44, TP 20; sensibilidad 0,31, especificidad 0,71, BA 0,511. Por cohorte: Duke 0,575, I-SPY1 0,603 e I-SPY2 0,558. La correlación de Spearman con las predicciones exteriores de N01 es de solo 0,39.

**Frente a los criterios fijados:**

1. Aprende: **sí en el entrenamiento, no en la época elegida.** La pérdida baja de 0,62 a 0,44 y FIT-eval sube de 0,47 a 0,94: el colapso de E57/N01 está resuelto. Pero en la época 6 FIT-eval es 0,593, por debajo de 0,60.
2. La interna acompaña a FIT: **no.** La interna se queda plana entre 0,50 y 0,61 y su BCE sube desde la época 3. Es el régimen de memorización (D36).
3. Exterior ≥ 0,62: **no.** 0,570, frente a 0,6165 de N01 normal. La diferencia (−0,047) es del orden de un error típico, así que no se puede decir que sea peor.

**Conclusión.** El problema cambia de naturaleza: con 2 pacientes por lote la red no aprendía; con 10 aprende, pero lo que aprende de las 702 pacientes no se traslada a pacientes nuevas. En ninguna época la interna supera ~0,61, que es lo mismo que daban los picos de E57. La señal de imagen que se generaliza con estos datos parece pequeña. No se lanzan los folds 1–4 de P01 tal cual: según el plan (2b), toca una única palanca de regularización decidida antes de ver su resultado.

Si se cumple, siguen los folds 1–4 normales y después la ponderada (N0/N1), con la misma receta. Así se cumple el contraste del enunciado sobre un modelo que aprende. Los 8 fits pendientes de N01 no se lanzan: compararían dos predictores casi constantes.

## Dónde queda cada cosa

- Código: [src/training/train_p01.py](../src/training/train_p01.py) · configuración: [configs/new_approach/P01_lotes_pacientes.json](../configs/new_approach/P01_lotes_pacientes.json) · tests: [tests/test_train_p01.py](../tests/test_train_p01.py)
- Resultados: `reports/new_approach/P01_lotes_pacientes/` · pesos: `checkpoints/new_approach/P01_lotes_pacientes/`
- No se sobrescribe ninguna ejecución existente; un lock evita dos ejecuciones simultáneas. No se lee test.

Uso educativo, sin validez clínica.
