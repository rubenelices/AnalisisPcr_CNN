# P02 · P01 + aumento geométrico sincronizado

Preparado el 20 de octubre de 2026. Ejecución manual por el estudiante. [Inicio](README.md) · [P01](P01_LOTES.md) · [Plan](PLAN.md).

## Por qué

P01 resolvió el colapso: con 10 pacientes por lote la red aprende (FIT-eval 0,47 → 0,94). Pero memoriza: la validación interna se queda plana entre 0,50 y 0,61, su BCE sube desde la época 3 y el exterior del fold 0 es 0,570. El plan (paso 2b) prevé **una sola** palanca de regularización. Se elige el aumento geométrico porque es la primera vez que se puede medir: E37, E49, D16 y D18 lo probaron sobre recetas que no aprendían o sobre la multimodal, así que no dicen nada de este régimen.

**Hipótesis.** Si cada época presenta los cortes volteados, girados y desplazados de otra manera, memorizar cada paciente cuesta más. La red tendría que apoyarse en el patrón de realce y la textura, que no cambian con esas transformaciones. Se espera que FIT-eval suba más despacio y que la interna lo acompañe durante más épocas.

Edad y menopausia se descartaron: AUC 0,532 con validación cruzada en train, parte de esa señal es la cohorte, y complicarían la demo ([razonamiento](PLAN.md)).

## Qué cambia (y qué no)

Único cambio frente a P01, **solo en entrenamiento** (la evaluación y la web no cambian):

| Aumento | Valor |
|---|---|
| Volteo horizontal | probabilidad 0,5 |
| Volteo vertical | probabilidad 0,5 |
| Giro | uniforme en ±10° |
| Traslación | uniforme en ±3 % (±7,7 px) |

Las tres fases reciben exactamente la misma transformación (una moneda y una rejilla por tripleta). El relleno es cero, igual en las tres; no hay cambios de intensidad. [Vista de un corte real con cuatro aumentos](../reports/new_approach/P02_aumento_geometrico/vista_aumento.png): el mapa de realce EARLY−PRE sigue nítido.

Todo lo demás es idéntico a P01: red E57 (298.161 parámetros), lotes de 10 pacientes × 2 cortes, particiones N01, AdamW con LR 0,001 y WD 0,001, plateau, máximo 50 épocas con paciencia 10, selección por AUC interna y umbral por F1 macro interno.

## Ejecutar

```bash
python -m src.train_p01 smoke --variant p02 --loss normal --fold 0
```

```bash
python -m src.train_p01 run --variant p02 --loss normal --fold 0
```

```bash
python -m src.train_p01 report --variant p02
```

## Criterios fijados antes de ejecutar

Referencias en el mismo fold 0: P01 (exterior 0,570; brecha FIT–interna en la época elegida −0,02, pero interna plana y memorización desde la época 7) y N01 normal (exterior 0,6165).

1. **Retrasa la memorización:** en la última época, FIT-eval − interna < 0,15. En P01 era 0,938 − 0,565 = 0,37.
2. **La interna mejora:** su máximo supera 0,63 (en P01, 0,611).
3. **Exterior ≥ 0,62** por paciente, registrando también el AUC por corte.

- **Si se cumplen los tres:** folds 1–4 con P02 normal y después los 5 folds ponderados.
- **Si falla alguno:** se cierra la búsqueda de recetas. La comparación obligatoria normal/ponderada en 5 folds se hace con la receta de **menor brecha FIT–interna en la última época** del fold 0 (P01 o P02). Esa regla no mira el exterior.

Con un solo fold (error típico ≈ 0,043) no se declara ganador en ningún caso.

## Resultado (ejecutado por el estudiante, 20 de octubre, 24 épocas, 37 min)

| Época | Pérdida online | AUC FIT-eval | AUC interna | BCE interna |
|---:|---:|---:|---:|---:|
| 6 | 0,604 | 0,592 | 0,624 | 0,594 |
| **14 (elegida)** | 0,582 | 0,665 | **0,625** | 0,592 |
| 19 | 0,564 | 0,682 | 0,622 | 0,597 |
| 24 | 0,537 | 0,739 | 0,596 | 0,604 |

En la época elegida, el exterior da AUC **0,535** por paciente y 0,528 por corte, con AP 0,312 y Brier 0,212. Con el umbral interno 0,308: TN 105, FP 50, FN 38, TP 26 y BA 0,542. Por cohorte: Duke 0,422 (42 pacientes), I-SPY1 0,676 (21) e I-SPY2 0,549 (156).

**Frente a los criterios:**

1. Diferencia FIT − interna en la última época: 0,739 − 0,596 = **0,143 < 0,15 → cumple** (en P01, 0,37). El aumento frena la memorización: FIT-eval sube más despacio y la BCE interna se mantiene estable en ~0,60, cuando en P01 subía a 0,76.
2. Máximo de la interna > 0,63: **0,625 → no cumple** por poco.
3. Exterior ≥ 0,62: **0,535 → no cumple**.

**Lectura.** En el fold 0, la interna y el exterior se mueven en sentidos opuestos según la receta: interna 0,572 / 0,611 / 0,625 y exterior 0,6165 / 0,570 / 0,535 para N01, P01 y P02. Con unas 200 pacientes por conjunto (error típico ≈ 0,045 cada uno), eso es lo que se espera cuando el rendimiento real de las tres es parecido, en torno a 0,57–0,60, y las diferencias de un solo fold son ruido. No se interpreta como que P02 sea peor.

**Decisión según la regla fijada:** se cierra la búsqueda de recetas. La comparación normal/ponderada en 5 folds se hace con **P02**, que tiene menor diferencia FIT–interna en la última época (0,143 frente a 0,37 de P01). Esta regla no mira el exterior.

Uso educativo, sin validez clínica.
