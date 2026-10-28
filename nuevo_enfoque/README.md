# Nuevo enfoque del trabajo

Punto de entrada desde el **20 de octubre de 2026**. Esta etapa se centra en cerrar un trabajo reproducible y defendible con E57 como referencia. La prioridad es resolver la evaluación, completar la comparación de pérdidas y preparar la entrega.

| Qué necesitas | Dónde está |
|---|---|
| Ver todos los resultados sin ejecutar nada | [RESULTADOS.ipynb](RESULTADOS.ipynb) |
| Saber qué está hecho y qué falta | [ESTADO.md](ESTADO.md) |
| Entender el plan y las decisiones | [PLAN.md](PLAN.md) |
| Arrancar y comprobar la demo | [DEMO.md](DEMO.md) |
| Entender y ejecutar la comparación preparada | [Protocolo N01](PROTOCOLO_N01.md) · [Ejecución](EJECUTAR_N01.md) |
| Entender los errores de N01 sin entrenar | [Diagnóstico de la primera pareja](DIAGNOSTICO_N01_FOLD0.md) |
| Revisar el texto del informe | [Borrador técnico](INFORME_BORRADOR.md) |

El notebook incluye como **antecedentes** las 108 épocas de los cinco folds E57, la arquitectura, las gráficas de pérdida y AUC, las matrices de confusión y el test ya realizado. La sección propia de N01 ya contiene las 33 épocas de la primera pareja del fold 0, ejecutada por el estudiante: [revisión y conclusiones](REVISION_N01_FOLD0.md).

## Punto de partida

E57 usa solo imágenes; no usa edad ni menopausia. La media de los cinco máximos de validación es **0,6224**, con selección de época en esas mismas validaciones. Es una cifra descriptiva y optimista. El checkpoint original de época 14 obtuvo **0,5728 en test**; ese resultado pertenece exclusivamente a esos pesos. El test ya está utilizado y no se reutilizará para elegir variantes.

La comparación **N01 ya está preparada**: BCE normal frente a ponderada completa, con selección interna de época y umbral y evaluación exterior por paciente. Consulta su [protocolo](PROTOCOLO_N01.md) y los [comandos de ejecución manual](EJECUTAR_N01.md). El estado de las ejecuciones se registra en el notebook. No se abre otra búsqueda de arquitecturas.

## Cómo mantenemos el orden

- Esta carpeta reúne la documentación y el notebook de la etapa activa. No hay que abrir ficheros técnicos para consultar los resultados.
- Cada experimento nuevo deberá tener una hipótesis, identificador, protocolo y conclusión; se incorporarán todas sus épocas, gráficas y matrices, también si no mejora.
- El código compartido permanece en `src/`; los respaldos técnicos y pesos conservan sus rutas. No se duplican datos ni checkpoints.
- Los entrenamientos se prepararán para ejecutarlos tú en local. No hay campañas automáticas abiertas.

## Antecedentes conservados

[Dossier E57](../E57/README.md) · [Historial de los ~200 experimentos](HISTORIAL_EXPERIMENTOS.md) (los originales están en el zip de respaldo del 20 de octubre).

Los originales siguen en su sitio. El nuevo notebook es una vista de consulta, no sustituye las fuentes científicas. Se puede actualizar desde los resultados guardados con `python -m src.new_approach_results`, sin entrenar ni volver a evaluar test.

Uso educativo, sin validez clínica.
