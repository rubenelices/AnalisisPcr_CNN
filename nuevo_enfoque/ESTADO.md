# Estado del trabajo · 24 de octubre de 2026

[Inicio](README.md) · [Plan](PLAN.md) · [Resultados (notebook)](RESULTADOS.ipynb) · [Web](DEMO.md) · [Reglas de entrega](REGLAS_ENTREGA.md).

## Dónde estamos

| Área | Estado | Siguiente acción |
|---|---|---|
| Diagnóstico | E57/N01 apenas aprendían (pérdida = predictor constante 0,605). Causa: BatchNorm con 2 pacientes por lote ([P01](P01_LOTES.md)) | Cerrado |
| Recetas | P01 aprende pero memoriza; P02 (+ aumento sincronizado) memoriza menos ([P02](P02_AUMENTO.md)) | Cerrado: no abrir más recetas ni arquitecturas |
| P02 · 5 folds, BCE normal (GPU) | **Completo.** AUC exterior conjunta **0,532** [IC95 0,493–0,570], media de folds 0,536; por corte 0,529. Duke 0,40 (ordena al revés), I-SPY1 0,61, I-SPY2 0,54. En 3/5 folds la interna eligió la época 1–2 | Ya en el notebook (sección 6) |
| P02 · 5 folds, BCE ponderada (GPU) | **Completo.** AUC exterior conjunta **0,500** [IC95 0,464–0,535], media de folds 0,505: nivel de azar. Normal − ponderada = +0,032 [−0,001, +0,064], sin diferencia demostrada. El 0,5000 del fold 0 era solo la época 1; terminó en la época 10 (interna 0,617). La copia parcial anterior está en `interrumpidos_20261021/parcial_weighted_fold_0` | Cerrado |
| Modelo de entrega | **Decidido (22/10, `reports/new_approach/ENTREGA/decision.json`): se mantiene E57** (regla 3: normal por defecto; regla 4: 0,532 < 0,55). No se evalúa nada más en test | Cerrado |
| Test público | E57: AUC 0,5728 (IC95 0,475–0,671), una evaluación | Cerrado: el entregable no cambia |
| Notebook de resultados | `python -m src.cuaderno_resultados` regenera todo: diagnóstico, P01, P02, 5 folds con detalle por fold, ROC/PR interna vs exterior, referencia teórica de un buen modelo (AUC 0,80, BCE 0,477) | Regenerado el 22/10 con las 10 ejecuciones |
| Informe | "Predicción de pCR con una CNN desde cero" ([presentación](../docs/presentacion/presentacion.pdf) · [guion de defensa](../docs/guion_defensa.pdf)) | Cerrado |
| Web | React (`frontend/`) + Flask (`app.py`). Hechas: portada, cabecera de mando, Predicción conectada al modelo real (misma probabilidad que el pipeline: 0,336232), Modelo y Resultados (con la ponderada). Encabezado grande en todas las secciones, en revisión de diseño | **Siguiente:** ajustes de diseño y Cómo leerlo; después despliegue |
| Despliegue | Pendiente; el estudiante preguntará al profesor si vale Hugging Face Spaces | — |
| Presentación (5 diapositivas) | Aplazada hasta que el estudiante lo pida | — |

## Resumen

1. Experimentación cerrada: entrega E57. Lo que queda es web, informe y despliegue. Último entrenamiento con recorte central (P03): **descartado por el paso 0** del 24/10 ([P03_PASO0.md](P03_PASO0.md)): el realce solo se conserva con un recorte de 192, E57 no se apoya en la periferia (R = 0,37) y P02 apenas supera a la cohorte sola (+0,009 [−0,025, +0,042]). Realce relativo (P04): **descartado por el paso 0b** ([P04_PASO0B.md](P04_PASO0B.md)): cohorte + PE/SER − cohorte = +0,014 [−0,012, +0,039]. Región que más realza (P06): **descartada por el paso 0c** ([P06_PASO0C.md](P06_PASO0C.md)): +0,009 [−0,019, +0,038]. P05 (tasa 0,0001) **retirado sin ejecutar**: ya probado en E12 y E55. **N01 en 5 folds (validación cruzada de la receta de E57): no supera a la cohorte**: conjunta 0,507; − cohorte = −0,015 [−0,037, +0,006]; − P02 = −0,024 [−0,058, +0,009]; el 0,617 del fold 0 fue el mejor de cinco (resto 0,445–0,542) ([N01_CV.md](N01_CV.md), SHA `ebdfb36b…`). **P07 descartado** (ensemble de 3 semillas: 0,543; − semilla 42 = +0,012 [−0,013, +0,036]; − cohorte = +0,021 [−0,014, +0,054]). Fue: ensemble de P02 normal con semillas 42, 17 y 123 ([P07_SEMILLAS.md](P07_SEMILLAS.md), criterios fijados antes, SHA `d6012995…`); lo lanza el estudiante con `python -m src.run_p07_noche` y se analiza con `python -m src.analisis_p07`. El entregable sigue siendo E57.
2. Web: arrancar con la línea de [DEMO.md](DEMO.md), recompilar con `npm --prefix frontend run build` y recargar el navegador (Cmd + R).

Uso educativo, sin validez clínica.
