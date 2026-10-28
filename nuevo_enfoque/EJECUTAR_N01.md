# Ejecutar N01 en local

[Inicio](README.md) · [Protocolo](PROTOCOLO_N01.md) · [Resultados](RESULTADOS.ipynb).

El experimento está preparado para ejecución manual. Los comandos se lanzan en la terminal, desde la raíz del proyecto. No hay que pegar código Python en zsh ni ejecutar el notebook de resultados.

## 1. Comprobar la preparación

```sh
python -m src.train_new_approach verify
```

Esto comprueba configuración, fuentes y particiones congeladas. No entrena, no abre imágenes ni evalúa test. Si indica que algo cambió, no borres la protección para continuar: hay que revisar la causa.

## 2. Comprobaciones técnicas de las dos pérdidas

```sh
python -m src.train_new_approach smoke --loss normal --fold 0 --device mps
```

```sh
python -m src.train_new_approach smoke --loss weighted --fold 0 --device mps
```

Cada smoke ejecuta una época con pocas pacientes. No aporta una medida científica del AUC. Si alguno falla, no continuar con el entrenamiento completo. Para usar CPU explícitamente, sustituye `mps` por `cpu`; no se usa nube.

## 3. Primera pareja científica: fold 0

```sh
python -m src.train_new_approach run --loss normal --fold 0 --device mps
```

```sh
python -m src.train_new_approach run --loss weighted --fold 0 --device mps
```

Ejecuta uno después del otro. El runner conserva E57 y cambia únicamente la pérdida. En pantalla se distingue **AUC FIT-eval**, **AUC INTERNA** y, al cerrar, **AUC EXTERIOR de desarrollo**. No son resultados de test. La época se elige con la interna, no con el exterior. El peso positivo de la ponderada se calcula con las pacientes FIT de ese fold; no se introduce un 2,39 fijo.

El notebook se actualiza con cada época completa. No hace falta esperar al cierre para ver las curvas; quizá debas cerrar y volver a abrir su vista para refrescarla.

## 4. Completar los otros folds

El contraste previsto incluye los folds 0–4 de ambas condiciones. Después de cerrar la primera pareja sin incidencias técnicas, se usan los mismos dos comandos sustituyendo `--fold 0` por `--fold 1`, después 2, 3 y 4. No se descartan folds porque den un AUC bajo ni se cambian los parámetros a mitad.

No se inicia una cola automática de diez entrenamientos. Cada comando ejecuta una sola condición/fold. Si se acaba el tiempo, las ejecuciones completas se conservan y el notebook indica que la comparación está incompleta; no se presenta como media definitiva de cinco folds.

## Leer y conservar los resultados

Abre [RESULTADOS.ipynb](RESULTADOS.ipynb). Incluye todas las épocas, BCE comparable, AUC, barras FIT/interna/exterior y, al cerrar cada ejecución, ROC, precision-recall, calibración y matrices con 0,5 y con el umbral interno.

Si quieres reconstruir únicamente la vista a partir de los resultados guardados:

```sh
python -m src.train_new_approach report
```

No vuelve a ejecutar inferencia ni entrenamiento. Los respaldos están en `reports/new_approach/N01_e57_bce/`, los pesos en `checkpoints/new_approach/N01_e57_bce/`. Los smokes tienen directorios propios y no aparecen en la comparación científica.

Para interrumpir un entrenamiento, usa Ctrl+C una vez y deja que termine de guardar el estado. Una ejecución parcial no se sobrescribe ni se reanuda automáticamente. Si se interrumpe de forma forzada y queda un lock, se revisará antes de retirarlo; no hay que borrar carpetas para forzar el arranque.

No cambiar la receta, añadir variables clínicas o reevaluar el test durante N01. Una incidencia técnica se resuelve y documenta antes de seguir.
