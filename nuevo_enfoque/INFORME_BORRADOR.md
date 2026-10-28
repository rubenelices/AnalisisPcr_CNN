# Borrador técnico del informe

Documento de trabajo, no versión de entrega. Describe lo que está comprobado hasta la primera pareja N01. La comparación de cinco folds sigue incompleta y la aplicación no está desplegada. Las diapositivas quedan aplazadas por petición del estudiante.

## 1. Objetivo y datos

El objetivo es predecir respuesta patológica completa (pCR) a partir de imágenes DCE-MRI anteriores al tratamiento. No se trata de detectar un tumor ni de predecir supervivencia. Cada corte contiene las fases PRE, EARLY y LATE, almacenadas como tres PNG de 256×256 píxeles. Los canales representan fases temporales, no colores.

El train público contiene 1.097 pacientes y 10.945 cortes. De esas pacientes, 322 tienen pCR=1 y 775 pCR=0. El test público contiene otras 176 pacientes y 1.758 cortes. Duke, I-SPY1 e I-SPY2 son cohortes diferentes; esa heterogeneidad se considera en la interpretación de los resultados. Las etiquetas privadas del profesorado no se utilizan para desarrollar el modelo.

La paciente es la unidad estadística. Sus cortes están correlacionados y permanecen juntos en todas las particiones. Se conservan los folds docentes y no se mezclan cortes de una misma paciente entre entrenamiento y validación.

## 2. Preprocesamiento y arquitectura

Se conserva el escalado docente conjunto entre fases. El cargador convierte los PNG grises en un tensor float32 de forma [3,256,256] mediante división por 255. No se aplican normalización ImageNet ni cambios de brillo o contraste. La red construye la representación PRE, EARLY−PRE y LATE−EARLY, manteniendo la información temporal relativa.

La arquitectura de referencia es E57, una CNN 2D propia de 298.161 parámetros, entrenada desde cero. No utiliza pesos preentrenados. Tiene cuatro bloques de 16, 32, 64 y 128 canales. Cada bloque incluye dos convoluciones 3×3 con padding 1 y stride 1, BatchNorm y ReLU; MaxPool 2×2 se sitúa entre las convoluciones. La resolución pasa de 256×256 a 128×128, 64×64, 32×32 y 16×16.

El promedio espacial global produce 128 características. La cabeza es Linear 128→32, ReLU, dropout 0,3 y Linear 32→1. La salida es un logit; la sigmoide se aplica al obtener probabilidades. El extractor aporta 294.000 parámetros y la cabeza 4.161. El campo receptivo convolucional local final es 106×106; este cálculo no demuestra qué regiones utiliza el modelo.

E57 y N01 utilizan solo imágenes. Aunque otras líneas históricas estudiaron edad y menopausia, esas variables no están activas aquí. No se puede concluir de este contraste que los datos clínicos carezcan de valor: no es la pregunta que evalúa N01.

La ficha detallada y su imagen están en [el dossier de E57](../E57/README.md). No se ha cambiado la arquitectura para favorecer los resultados de esta comparación.

## 3. Entrenamiento y evaluación: dos procedimientos distintos

### Referencia histórica E57

Durante entrenamiento se agrupan dos pacientes por lote, con todos sus cortes. Se promedian las características de cada paciente y se aplica la cabeza y una BCE normal por paciente. En la evaluación principal se predice cada corte y se promedian sus probabilidades por paciente. Estas dos agregaciones no son equivalentes por la cabeza no lineal; la diferencia se documenta y no se atribuye a ella una causa de rendimiento no demostrada.

La receta utiliza AdamW, LR inicial 0,001, weight decay 0,001, ReduceLROnPlateau por AUC paciente y semilla 42. La paciencia de parada es 10. En los cinco folds históricos se seleccionó la época con la misma validación que se reportó. Su media de máximos es descriptiva y optimista, no una evaluación exterior independiente.

### Comparación N01

N01 mantiene E57, su representación, agregaciones y receta para comparar una sola variable: BCE normal frente a BCE ponderada completa. El peso positivo se calcula como N0/N1 de pacientes FIT, no como raíz de la razón ni contando cortes como pacientes independientes.

Para cada fold docente se reserva su exterior. Dentro del resto se separa un 20 % de pacientes para validación interna, estratificando pCR con semilla 42. En fold 0 hay 702 FIT, 176 internas y 219 exteriores. Ambas condiciones utilizan las mismas pacientes y los mismos pesos iniciales.

Solo FIT actualiza parámetros y BatchNorm. La interna controla el scheduler y selecciona el primer checkpoint con mayor AUC, con máximo 50 épocas y paciencia 10. Después, usando ese checkpoint, el umbral se elige en la interna por F1macro, con desempates prefijados. La decisión queda guardada antes de evaluar el exterior. En el exterior se muestran 0,5 y el umbral interno sin escoger a posteriori cuál parece mejor.

Los folds exteriores ya fueron observados durante la historia del proyecto. N01 mejora la separación entre selección de época y evaluación dentro de su procedimiento, pero no convierte esos datos en un nuevo test independiente. El [protocolo completo](PROTOCOLO_N01.md) especifica los controles.

## 4. Resultados confirmados

### E57 original y su único test público

El checkpoint original del fold 0 se eligió en la época 14, tras entrenar con 878 pacientes y seleccionar con 219 de validación. AUC en ese checkpoint: FIT-eval 0,5577 y validación 0,6703. La media de los cinco máximos históricos fue 0,6224 y no debe confundirse con el resultado de un único modelo.

El umbral 0,325 se escogió explorando validación y se fijó antes de abrir el test público. Ese test se evaluó una vez; no se utiliza para cambiar recetas ni se vuelve a evaluar N01 en él.

| Unidad del test ya realizado | Pacientes/cortes | ROC-AUC | AP | Brier |
|---|---:|---:|---:|---:|
| Paciente: principal | 176 pacientes | 0,5728 | 0,4206 | 0,2116 |
| Corte: secundaria | 1.758 cortes | 0,5669 | 0,3932 | 0,2121 |

En pacientes, el IC95 de ROC-AUC es 0,4753–0,6706, mediante 2.000 remuestreos por paciente y semilla 42. Con umbral 0,325, la matriz es TN 92 / FP 31 / FN 35 / TP 18, sensibilidad 33,96 %, especificidad 74,80 % y precisión 36,73 %. Son resultados modestos y con incertidumbre amplia. Los cortes no constituyen 1.758 pacientes independientes y no se utiliza bootstrap ingenuo por corte para sostener conclusiones por paciente.

Las cifras por corte anteriores estaban guardadas en el informe del test original; se leen de ese registro, no proceden de otra ejecución de test.

### Primera pareja N01, exterior del fold 0

| Métrica | BCE normal | BCE ponderada |
|---|---:|---:|
| Épocas completadas | 12 | 21 |
| Época seleccionada en interna | 2 | 11 |
| AUC interna en esa época | 0,5721 | 0,6497 |
| AUC exterior | 0,6165 | 0,5475 |
| AP exterior | 0,4117 | 0,3244 |
| Brier exterior | 0,20475 | 0,24472 |
| Umbral interno | 0,263949 | 0,502931 |
| Sensibilidad exterior | 51,56% | 37,50% |
| Especificidad exterior | 68,39% | 67,10% |
| Precisión pCR exterior | 40,24% | 32,00% |
| F1macro exterior | 0,5890 | 0,5205 |
| Balanced accuracy exterior | 0,5997 | 0,5230 |
| TN/FP/FN/TP | 106/49/31/33 | 104/51/40/24 |

En este fold, la normal es más favorable. La ponderada alcanza un máximo interno mayor, pero no lo mantiene en el exterior. No hay todavía una media N01 de cinco folds ni evidencia de que una condición gane de forma general. La comparación usa menos pacientes FIT y otra selección de época que E57 original; no se interpretan las diferencias de AUC entre procedimientos como un efecto aislado de una técnica.

## 5. Interpretación de curvas y errores

Las curvas N01 no muestran el patrón clásico de AUC de entrenamiento próximo a 1 con validación estancada. FIT-eval permanece bajo y variable. Las BCE comparables cercanas pueden acompañar una discriminación débil; no bastan para declarar que el modelo aprende bien.

La ponderada concentra sus probabilidades cerca de 0,5: 98 de 219 pacientes exteriores están dentro de ±0,005 de su umbral interno. La normal también presenta solapamiento entre clases. Por tanto, cambiar el umbral puede modificar muchas etiquetas, pero no arregla por sí solo la ordenación que mide el AUC.

El Brier de la normal mejora muy ligeramente el predictor constante basado en prevalencia FIT (0,20475 frente a0,20684); la ponderada lo empeora (0,24472). La pérdida ponderada modifica el objetivo de entrenamiento y puede alterar la calibración: la subida de BCE normal en sus curvas no se atribuye automáticamente a memorización.

El diagnóstico por cohorte y el solapamiento de errores están en [el análisis descriptivo](DIAGNOSTICO_N01_FOLD0.md). Los subgrupos pequeños no permiten conclusiones causales ni selección de cohortes favorables. No se ajustan un ensemble, calibración o nuevos umbrales con este exterior.

## 6. Aplicación y reproducibilidad

La demo local recibe tres PNG del mismo corte y utiliza el cargador compartido con entrenamiento. E57 se restaura con su cabeza real y comprueba el SHA de los pesos contra el protocolo de decisión. Se muestra explícitamente que la predicción corresponde a un corte, mientras la evaluación principal agrega cortes por paciente.

El umbral de referencia0,325 fue seleccionado para la agregación por paciente. Su aplicación a un corte en la demo es una convención del prototipo; no demuestra que sea el umbral óptimo a esa unidad. Las métricas secundarias por corte del test original se reportan separadas. No se presenta una sola tripleta como si representara todos los cortes de una paciente.

Se verificaron equivalencia de probabilidades de desarrollo entre la ruta científica y la aplicación, cinco cargas consecutivas sin reiniciar y manejo de entradas inválidas. Las fases con píxeles idénticos generan un aviso pero conservan su tensor, ya que existe un caso así en los datos de desarrollo. La web continúa siendo local: todavía no hay URL pública comprobada.

Se conservan configuraciones, semillas, fuentes, historiales y pesos; los notebooks muestran tablas y figuras de los resultados guardados. Los datos y checkpoints no se añaden a Git. La inferencia se realiza en modo eval y sin gradientes. El proyecto es educativo y no tiene validez clínica.

## 7. Límites y tareas para cerrar el informe

- La comparación N01 está completa solo para fold 0. Si no se entrenan más folds, se declarará así; no se rellenarán medias ni intervalos de una CV inexistente.
- La arquitectura fue elegida tras muchos experimentos de desarrollo. No se interpreta cada época como una repetición independiente ni se ocultan resultados desfavorables.
- El test original ya se ha usado. Su resultado pertenece a sus pesos y no valida candidatos posteriores.
- No se demuestra estabilidad entre nuevas semillas, utilidad clínica, causalidad de los mapas o superioridad general de una pérdida.
- Queda decidir el alcance final de los experimentos y cerrar las figuras seleccionadas. El borrador deberá revisarse antes de la entrega.
- Falta preparar y comprobar el despliegue y la configuración final en otro equipo. No se inventa una URL ni se publica nada en esta fase.

## Fuentes de las cifras

[Dossier E57](../E57/README.md), [test original guardado](../E57/TEST.md), [revisión N01](REVISION_N01_FOLD0.md), [diagnóstico N01](DIAGNOSTICO_N01_FOLD0.md), [protocolo N01](PROTOCOLO_N01.md), [demo](DEMO.md) y [notebook completo](RESULTADOS.ipynb). El registro técnico del test es `reports/final_e57_test_v1/test_evaluation.json`; no se ha vuelto a ejecutar para redactar este documento.
