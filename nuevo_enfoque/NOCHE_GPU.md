# P02 en la GPU de la universidad

Diez ejecuciones de [P02](P02_AUMENTO.md): BCE normal y ponderada, folds 0–4, más sus dos smokes, en serie y sin tocar test.

## En la universidad

```bash
git pull
```

```bash
nohup python -m src.run_p02_noche > noche_gpu.out 2>&1 &
```

Antes de entrenar comprueba la GPU y que el protocolo sea idéntico al del Mac (SHA 827fef1d…); si algo no cuadra, se para sin lanzar nada. Para ver que ha arrancado bien:

```bash
tail -n 20 noche_gpu.out
```

Si un fit falla, la noche sigue con el siguiente.

## Mañana

Al terminar queda `resultados_p02_gpu_<fecha>.tar.gz` en la raíz del repo, con unos 40 MB de resultados, `best.pt` y el log. Cópialo a la raíz del proyecto en el Mac y:

```bash
tar -xzf resultados_p02_gpu_*.tar.gz
```

```bash
python -m src.train_p01 report --variant p02 --gpu
```

Los resultados de la GPU van en `runs_cuda/`, separados de la ejecución del Mac.

Informe completo de los 5 folds (curvas como en la figura de referencia, matrices a umbral interno / 0,325 / 0,5, ROC y PR, cohortes, corte, comparación pareada normal–ponderada y calibración):

```bash
python -m src.cv_report --variant p02 --gpu
```

Queda en `reports/new_approach/P02_aumento_geometrico/cv_runs_cuda/RESUMEN.md`.

## Después del informe: aplicar las reglas de entrega

Con los 10 fits completos, se aplican las [reglas fijadas antes de ver los resultados](REGLAS_ENTREGA.md). Este paso no lee test:

```bash
python -m src.final_ensemble freeze --variant p02 --gpu
```

Escribe `reports/new_approach/ENTREGA/decision.json`: qué receta se entrega, por qué, los SHA de los 5 pesos y el umbral (la mediana de los internos). Solo cuando el estudiante lo decida, la **única** evaluación en test:

```bash
python -m src.final_ensemble test --device cpu
```

Si su carpeta de salida ya existe, se niega a repetir.
