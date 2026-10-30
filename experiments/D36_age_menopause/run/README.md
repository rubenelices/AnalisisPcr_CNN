# D36 · Referencia con información básica

## Resultado recibido · 14 octubre

Completado en local: 13 épocas, mejor3,
ROC0,609274194/AP0,352830258/Brier0,208763025/866,648s envalidación.
Matriz0,5TN155/FP0/FN64/TP0; sensibilidad/F1pCR0. Trainloss0,612→0,355,
valloss termina0,870; sobreajuste tardío. No cumple nuevoobjetivoROC0,65 y
matrizútil. No alargar/repetir, ni atribuir un único ensayo a inutilidad de edad.
Épocas y gráficos originales conservados; test no evaluado.
Las notas de preparación inferiores son historial.

Preparado el 14 de octubre. El entrenamiento se lanza manualmente en local.

Entradas: PRE/EARLY/LATE y solamente edad/menopausia. Datos basales;
CSV leído con whitelist pid/age/menopause. No HR, HER2, raza, volumen ni pCR.
Ausentes imputados solo con train; nunca inventar tabaquismo.
CNN propia de E19/E58 desde cero, no elegida por pico E57 ni transfer learning.
Se mantienen ocho slots de compatibilidad; solo cuatro transportan información.
[Ficha y PNG/SVG con dimensiones reales](../README.md).

Fold0/seed42;878train/219validación, batch16, AdamW LR/WD0,001,
50épocas/paciencia10, Plateau3/factor0,5. BCE normal por corte y media de
probabilidades por paciente. ReLU y sin aumentos para establecer referencia.
Umbral0,5 es diagnóstico provisional, no decisión final. Test cerrado.

```sh
python -u -m src.training.train_multimodal --config configs/multimodal/SMOKE_D36_edad_menopausia_mac.json --device mps
```

Primero esa comprobación técnica de2épocas/8train/4val, lanzada;
sin interpretar sus AUC. Si finaliza sin errores, ejecutar el científico:

```sh
python -u -m src.training.train_multimodal --config experiments/D36_age_menopause/run/config.json --device mps
```

Verificación: 1204 tests software correctos/cero omitidos30,57s,
sin D36 ni smoke entrenados. Carga clínica real878/219 sin solapamiento y
sin HR/HER2 leídas, estados inactivos0;1862artefactos históricos intactos.

Al completar: todasépocas/seisgráficas en notebooks/07_RESULTADOS_CLAROS;
curvas ROC/PR/matriz/calibración también en resultados/fold_0_seed_42.
No usar --overwrite: nuevas identidades para cada experimento.

## Orden de pruebas posteriores (no ejecutadas ni configuradas todavía)

1. Réplica seed17 de la referencia, aunque el resultado42 sea desfavorable;
   no elegir semilla favorable ni confundir semillas con cinco folds.
2. Aumento geométrico moderado5°/2%, sincronizado entre las tres fases, solo train.
3. SiLU SOLO en CNN frenteReLU; mantener rama clínica/fusión y todos los ajustes.
4. Ganancia de intensidad compartida±10% entre fases, sin aumentos RGB;
   independiente de la rotación. Evitar variaciones arbitrarias que destruyan realce.
5. BCE normal/ponderada, controles soloimágenes y edad/menopausia solas;
   lista corta y confirmación en cinco folds. Escalas de pérdida distintas.

Comparar ROC/AP, Brier, train/validación en mismaépoca/modo, métricas de ambas
clases y F1macro. Elegir umbral en validación con criterio explícito; cambiarlo
no mejoraROC. Congelar todo antes de evaluar test una única vez.
No promesa de superar0,7 ni selección por pico; registrar resultados negativos.
Uso educativo y de investigación, sin validez clínica.
