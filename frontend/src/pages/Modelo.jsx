import { useState } from 'react'
import { CAPAS } from '../components/Arquitectura.jsx'
import { ArquitecturaInteractiva } from '../components/ArquitecturaInteractiva.jsx'
import { Barras, COLOR, Diagonal, Figura, Grafico, Horizontal, Linea, Puntos, Texto, Vertical } from '../components/Graficas.jsx'
import { Revelar } from '../components/Movimiento.jsx'
import { useCanales } from '../canales.js'
import { EJEMPLOS } from '../ejemplo.js'
import { fmt, mil, useGraficas, useResumen } from '../datos.js'

const RECETA = [
  ['Pérdida', 'BCEWithLogitsLoss normal, calculada por paciente'],
  ['Optimizador', 'AdamW, tasa de aprendizaje 0,001, weight decay 0,001'],
  ['Calendario', 'La tasa se reduce a la mitad si el AUC de validación no mejora en 3 épocas'],
  ['Lotes', '2 pacientes por lote, con todos sus cortes'],
  ['Épocas', 'Máximo 50, parada tras 10 sin mejorar: se eligió la época 14 de 24'],
  ['Datos', '878 pacientes de entrenamiento y 219 de validación (fold 0), separadas por paciente'],
  ['Aumento de datos', 'Ninguno'],
  ['Semilla', '42'],
]

const DECISIONES = [
  ['Desde cero', 'El caso prohíbe pesos preentrenados y arquitecturas importadas: todo se aprende con estas 1.097 pacientes.'],
  ['Restas fijas a la entrada', 'EARLY − PRE es el realce del contraste. Darlo explícito evita que la red tenga que aprender a restar.'],
  ['Convoluciones 3×3 y cuatro reducciones', 'Patrones locales que se combinan en escalas mayores: cada neurona final ve 106 × 106 píxeles.'],
  ['Media espacial global', 'Resume cada canal en un número sin depender de dónde esté el tumor dentro del recorte.'],
  ['Red pequeña y dropout', 'Con pocas pacientes, menos parámetros y dropout reducen la memorización.'],
]

const CANALES = [
  ['PRE', 'La anatomía antes del contraste.'],
  ['EARLY − PRE', 'Cuánto realza cada zona al llegar el contraste.'],
  ['LATE − EARLY', 'Si el contraste sigue entrando o ya se lava.'],
]

function Canales() {
  const canales = useCanales(EJEMPLOS[0])
  return (
    <ol className="canales">
      {CANALES.map(([n, t], i) => (
        <Revelar as="li" key={n} orden={i}>
          {canales ? <img src={canales[i]} alt={`Canal ${n} del corte de ejemplo`} width="256" height="256" /> : <span className="canal-hueco" />}
          <p className="canal-nombre">{n}</p>
          <p className="canal-texto">{t}</p>
        </Revelar>
      ))}
    </ol>
  )
}

const epocas = (n) => Array.from({ length: Math.ceil(n / 4) + 1 }, (_, i) => Math.max(1, i * 4)).filter((v) => v <= n)

function Histograma({ h }) {
  const b = h.bordes
  const paso = b[1] - b[0]
  const max = Math.max(...h.sin_pcr, ...h.pcr)
  const tope = Math.ceil(max / 5) * 5
  return (
    <Grafico x={[b[0], b.at(-1)]} y={[0, tope]} ticksX={b.filter((_, i) => i % 4 === 0)} ticksY={Array.from({ length: tope / 5 + 1 }, (_, i) => i * 5)}
      fmtX={(v) => fmt(v, 2)} fmtY={(v) => v} etiquetaX="Probabilidad predicha" etiquetaY="Pacientes" descripcion="Histograma de probabilidades por clase real">
      <Barras ancho={paso * 0.42} datos={[
        ...h.sin_pcr.map((v, i) => ({ x: b[i] + paso * 0.28, y: v, color: COLOR.train })),
        ...h.pcr.map((v, i) => ({ x: b[i] + paso * 0.72, y: v, color: COLOR.acento })),
      ]} />
      <Vertical x={h.umbral} color={COLOR.test} etiqueta="umbral" />
    </Grafico>
  )
}

function GraficasEntrenamiento({ g, r }) {
  const e = g.e57
  const n = e.historia.length
  const t = r.test_e57
  const serie = (col) => e.historia.map((f) => [f.epoch, f[col]])
  const prevTest = e.test.pcr / e.test.n
  return (
    <div className="figuras-rejilla">
      <Figura titulo="Pérdida por época"
        leyenda={[{ texto: 'Entrenamiento', color: COLOR.train }, { texto: 'Validación', color: COLOR.valid },
          { texto: `Predictor constante (${fmt(g.bce_constante, 3)})`, color: COLOR.azar, tipo: 'discontinua' },
          { texto: `Buen modelo teórico (${fmt(g.bce_referencia, 3)})`, color: COLOR.ref, tipo: 'puntos' }]}
        explicacion={<>
          <p>La BCE mide cuánto se equivoca la probabilidad. Un modelo que respondiera siempre la proporción de pCR (29 %) tendría {fmt(g.bce_constante, 3)}: por debajo de esa línea, el modelo aprende algo de la imagen.</p>
          <p>Las dos curvas se quedan pegadas a ella. Es el síntoma que llevó al diagnóstico: con lotes de 2 pacientes, BatchNorm impedía aprender diferencias entre pacientes (Inicio, paso 6).</p>
        </>}>
        <Grafico x={[1, n]} y={[0.44, 0.66]} ticksX={epocas(n)} ticksY={[0.45, 0.5, 0.55, 0.6, 0.65]} fmtX={(v) => v} fmtY={(v) => fmt(v, 2)}
          etiquetaX="Época" etiquetaY="BCE por paciente" descripcion="Pérdida de entrenamiento y validación por época">
          <Horizontal y={g.bce_constante} />
          <Horizontal y={g.bce_referencia} color={COLOR.ref} tipo="puntos" />
          <Vertical x={e.epoca_elegida} etiqueta={`época elegida (${e.epoca_elegida})`} />
          <Linea datos={serie('train_loss')} color={COLOR.train} />
          <Linea datos={serie('validation_loss')} color={COLOR.valid} retraso={0.2} />
        </Grafico>
      </Figura>

      <Figura titulo="AUC de validación por época"
        leyenda={[{ texto: 'AUC por paciente (validación, 219 pacientes)', color: COLOR.valid },
          { texto: 'Azar (0,5)', color: COLOR.azar, tipo: 'discontinua' }, { texto: 'Buen modelo teórico (0,80)', color: COLOR.ref, tipo: 'puntos' }]}
        explicacion={<>
          <p>El AUC es la probabilidad de que, tomadas al azar una paciente con pCR y otra sin ella, el modelo dé más probabilidad a la primera. No depende del umbral.</p>
          <p>La época 14 se eligió por ser el máximo de esta curva ({fmt(e.picos_folds[0], 3)}). Como se elige mirando estas mismas pacientes, ese pico es optimista: oscila mucho de una época a otra y parte de él es suerte.</p>
        </>}>
        <Grafico x={[1, n]} y={[0.4, 0.9]} ticksX={epocas(n)} ticksY={[0.4, 0.5, 0.6, 0.7, 0.8, 0.9]} fmtX={(v) => v}
          etiquetaX="Época" etiquetaY="AUC" descripcion="AUC de validación por época">
          <Horizontal y={0.5} etiqueta="azar" />
          <Horizontal y={g.auc_referencia} color={COLOR.ref} tipo="puntos" etiqueta="buen modelo" />
          <Vertical x={e.epoca_elegida} />
          <Linea datos={serie('patient_roc_auc')} color={COLOR.valid} puntos />
        </Grafico>
      </Figura>

      <Figura titulo="AUC medido de cuatro formas"
        leyenda={[{ texto: 'Optimista: época elegida con esas pacientes', color: COLOR.valid }, { texto: 'Honesta: pacientes no vistas', color: COLOR.test }]}
        explicacion={<>
          <p>Las dos barras naranjas usan la época que mejor fue en validación, medida en esa misma validación: por eso salen altas. Las oscuras miden con pacientes que no intervinieron en ninguna decisión.</p>
          <p>La receta de E57 no evaluaba el conjunto de entrenamiento en cada época; la comparación entre entrenamiento y validación está en Resultados, con P02.</p>
        </>}>
        <Grafico x={[0.4, 4.6]} y={[0.4, 0.9]} ticksX={[]} ticksY={[0.4, 0.5, 0.6, 0.7, 0.8, 0.9]} etiquetaX="" etiquetaY="AUC" descripcion="AUC de E57 en validación, validación anidada y test">
          <Horizontal y={0.5} etiqueta="azar" lado="izquierda" />
          <Horizontal y={g.auc_referencia} color={COLOR.ref} tipo="puntos" etiqueta="buen modelo" lado="izquierda" />
          <Barras datos={[{ x: 1, y: e.picos_folds[0], color: COLOR.valid }, { x: 2, y: r.picos_e57.media, color: '#dca06f' },
            { x: 3, y: e.cv_anidada, color: '#8793a0' }, { x: 4, y: t.auc, color: COLOR.test }]} />
          {[e.picos_folds[0], r.picos_e57.media, e.cv_anidada, t.auc].map((v, i) => <Texto key={i} x={i + 1} y={v} dy={-8}>{fmt(v, 3)}</Texto>)}
          {['Validación, fold 0', 'Media 5 folds', 'Anidada 5 folds', 'Test (una vez)'].map((s, i) => <Texto key={s} x={i + 1} y={0.4} dy={20} clase="tick">{s}</Texto>)}
        </Grafico>
      </Figura>

      <Figura titulo="Curva ROC: validación frente a test"
        leyenda={[{ texto: `Validación, fold 0 (AUC ${fmt(e.validacion.auc, 3)}, n = ${e.validacion.n})`, color: COLOR.valid },
          { texto: `Test (AUC ${fmt(e.test.auc, 3)}, n = ${e.test.n})`, color: COLOR.test },
          { texto: 'Buen modelo teórico (0,80)', color: COLOR.ref, tipo: 'puntos' }, { texto: 'Azar', color: COLOR.azar, tipo: 'discontinua' }]}
        explicacion={<>
          <p>Cada punto de la curva es un umbral: a la izquierda, umbrales altos (pocas alarmas); a la derecha, bajos. Cuanto más se separa de la diagonal, mejor ordena el modelo.</p>
          <p>La curva de test queda por debajo de la de validación: es la diferencia entre medir con las pacientes que eligieron la época y medir con pacientes nuevas.</p>
        </>}>
        <Grafico x={[0, 1]} y={[0, 1]} ticksX={[0, 0.2, 0.4, 0.6, 0.8, 1]} ticksY={[0, 0.2, 0.4, 0.6, 0.8, 1]} alto={440}
          etiquetaX="1 − especificidad" etiquetaY="Sensibilidad" descripcion="Curvas ROC de validación y test">
          <Diagonal />
          <Linea datos={e.referencia_test.roc} color={COLOR.ref} tipo="puntos" />
          <Linea datos={e.validacion.roc} color={COLOR.valid} />
          <Linea datos={e.test.roc} color={COLOR.test} retraso={0.3} />
        </Grafico>
      </Figura>

      <Figura titulo="Precisión frente a sensibilidad"
        leyenda={[{ texto: `Validación (AP ${fmt(e.validacion.ap, 3)})`, color: COLOR.valid }, { texto: `Test (AP ${fmt(e.test.ap, 3)})`, color: COLOR.test },
          { texto: `Buen modelo teórico (AP ${fmt(e.referencia_test.ap, 3)})`, color: COLOR.ref, tipo: 'puntos' },
          { texto: `Azar = proporción de pCR (${fmt(prevTest, 3)})`, color: COLOR.azar, tipo: 'discontinua' }]}
        explicacion={<>
          <p>De las pacientes a las que el modelo llama «pCR», ¿cuántas lo son? Aquí el azar no es 0,5 sino la proporción de pCR del conjunto: un modelo sin información acertaría en esa proporción.</p>
          <p>En test la precisión queda algo por encima de esa línea en casi todo el recorrido, lejos de la del buen modelo.</p>
        </>}>
        <Grafico x={[0, 1]} y={[0, 1]} ticksX={[0, 0.2, 0.4, 0.6, 0.8, 1]} ticksY={[0, 0.2, 0.4, 0.6, 0.8, 1]} alto={440}
          etiquetaX="Sensibilidad (recall)" etiquetaY="Precisión" descripcion="Curvas de precisión y sensibilidad">
          <Horizontal y={prevTest} />
          <Linea datos={e.referencia_test.pr} color={COLOR.ref} tipo="puntos" />
          <Linea datos={e.validacion.pr} color={COLOR.valid} />
          <Linea datos={e.test.pr} color={COLOR.test} retraso={0.3} />
        </Grafico>
      </Figura>

      <Figura titulo={`Matriz de confusión en test (umbral ${fmt(e.umbral, 3)})`}
        explicacion={<>
          <p>El umbral se fijó con la validación, antes de mirar el test. Con él, el modelo detecta {t.matriz.tp} de {t.pcr} pCR (sensibilidad {fmt(t.sensibilidad, 2)}) y acierta en {t.matriz.tn} de {t.matriz.tn + t.matriz.fp} pacientes sin pCR (especificidad {fmt(t.especificidad, 2)}).</p>
          <p>Por qué 0,325 y no 0,5: con un 29 % de pCR, las probabilidades del modelo son bajas casi siempre y con 0,5 no diría «pCR» a nadie.</p>
        </>}>
        <div className="matriz-web" role="img" aria-label={`${t.matriz.tn} verdaderos negativos, ${t.matriz.fp} falsos positivos, ${t.matriz.fn} falsos negativos, ${t.matriz.tp} verdaderos positivos`}>
          <span /><span className="mz-cab">Predice sin pCR</span><span className="mz-cab">Predice pCR</span>
          <span className="mz-cab fila">Sin pCR real</span>
          <Revelar className="mz-celda bien" orden={0}><b>{t.matriz.tn}</b>verdaderos negativos</Revelar>
          <Revelar className="mz-celda mal" orden={1}><b>{t.matriz.fp}</b>falsos positivos</Revelar>
          <span className="mz-cab fila">pCR real</span>
          <Revelar className="mz-celda mal" orden={2}><b>{t.matriz.fn}</b>falsos negativos</Revelar>
          <Revelar className="mz-celda bien" orden={3}><b>{t.matriz.tp}</b>verdaderos positivos</Revelar>
        </div>
      </Figura>

      <Figura titulo="Calibración en test"
        leyenda={[{ texto: 'Grupos de unas 22 pacientes (8 cuantiles)', color: COLOR.test }, { texto: 'Calibración perfecta', color: COLOR.azar, tipo: 'discontinua' }]}
        explicacion={<>
          <p>Cada punto agrupa pacientes con probabilidades parecidas y compara la probabilidad media que dio el modelo con la proporción real de pCR del grupo.</p>
          <p>Casi todas las probabilidades caen entre 0,22 y 0,35: el modelo apenas se compromete, y dentro de ese margen estrecho la proporción real de pCR casi no cambia. Brier {fmt(e.test.brier, 3)}; error de calibración esperado {fmt(e.test.ece, 3)}.</p>
        </>}>
        <Grafico x={[0, 0.7]} y={[0, 0.7]} ticksX={[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]} ticksY={[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]} alto={420}
          etiquetaX="Probabilidad media predicha" etiquetaY="Proporción real de pCR" descripcion="Calibración de las probabilidades en test">
          <Diagonal />
          <Linea datos={e.test.calibracion.map(([p, y]) => [p, y])} color={COLOR.test} grosor={1.6} />
          <Puntos datos={e.test.calibracion.map(([p, y]) => [p, y])} color={COLOR.test} />
        </Grafico>
      </Figura>

      <Figura titulo="Probabilidades por clase real (test)"
        leyenda={[{ texto: 'Pacientes sin pCR', color: COLOR.train }, { texto: 'Pacientes con pCR', color: COLOR.acento }, { texto: `Umbral ${fmt(e.umbral, 3)}`, color: COLOR.test, tipo: 'discontinua' }]}
        explicacion={<>
          <p>Si el modelo separara bien, las barras rosas estarían a la derecha del umbral y las azules a la izquierda. Las dos distribuciones se solapan casi por completo.</p>
          <p>Eso es lo que resume un AUC de {fmt(t.auc, 3)}: las pacientes con pCR tienden a recibir algo más de probabilidad, con mucha superposición.</p>
        </>}>
        <Histograma h={e.test.histograma} />
      </Figura>
    </div>
  )
}

function Ficha({ modelo }) {
  if (modelo.estado !== 'listo') return <p className="nota-pie">{modelo.estado === 'cargando' ? 'Cargando la ficha del modelo…' : 'El servidor del modelo no está disponible.'}</p>
  return (
    <dl className="ficha-modelo">
      <div><dt>Versión</dt><dd>{modelo.version}</dd></div>
      <div><dt>Parámetros</dt><dd>{mil(modelo.parameters)}</dd></div>
      <div><dt>Umbral</dt><dd>{fmt(modelo.threshold, 3)}</dd></div>
      <div><dt>Pesos (SHA-256)</dt><dd title={modelo.sha256}>{modelo.sha256.slice(0, 16)}…</dd></div>
      <div><dt>Dispositivo</dt><dd>{modelo.device.toUpperCase()}</dd></div>
    </dl>
  )
}

export function Modelo({ modelo }) {
  const [activo, setActivo] = useState('general')
  const g = useGraficas(), r = useResumen()
  const total = CAPAS.reduce((s, c) => s + c.params, 0)
  return (
    <div className="pagina-modelo">
      <section className="papel-seccion">
        <h1 className="titulo-pagina">Modelo</h1>
        <p className="entradilla">
          E57: una red convolucional 2D de {mil(total)} parámetros, entrenada desde cero. Recibe las tres fases de un corte y
          devuelve la probabilidad de respuesta patológica completa; para una paciente se promedian sus cortes.
        </p>
        <Ficha modelo={modelo} />

      </section>

      <section className="arqi-seccion" aria-labelledby="t-arquitectura">
        <div className="arqi-cabeza">
          <h2 id="t-arquitectura" className="titulo-seccion">La arquitectura, parte a parte</h2>
          <p className="parrafo">Baja para recorrer la red de la entrada a la salida, o pulsa una parte del diagrama. También funciona con las flechas del teclado.</p>
        </div>
        <ArquitecturaInteractiva onCambio={setActivo} />
      </section>

      <section className="papel-seccion">
        <div className="dos-columnas">
          <div>
            <h2 className="titulo-seccion">Qué recibe</h2>
            <p>
              Las tres fases de un mismo corte en escala de grises, divididas entre 255 exactamente igual que en el entrenamiento.
              Antes de la primera convolución se transforman, sin parámetros, en tres canales. Son fases temporales, no colores:
              no se usa normalización de ImageNet ni aumentos de color.
            </p>
            <Canales />
          </div>
          <div>
            <h2 className="titulo-seccion">Por qué así</h2>
            <dl className="decisiones">
              {DECISIONES.map(([t, d], i) => <Revelar key={t} orden={i}><dt>{t}</dt><dd>{d}</dd></Revelar>)}
            </dl>
          </div>
        </div>

        <h2 className="titulo-seccion">Capa a capa</h2>
        <div className="tabla-scroll">
          <table className="tabla">
            <thead><tr><th>Etapa</th><th>Qué hace</th><th>Salida por corte</th><th className="num">Parámetros</th></tr></thead>
            <tbody>
              {CAPAS.map((c) => (
                <tr key={c.id} className={c.id === activo ? 'activa' : undefined}>
                  <td>{c.etapa}</td><td>{c.detalle}</td><td>{c.salida}</td><td className="num">{mil(c.params)}</td>
                </tr>
              ))}
              <tr className="total"><td>Total</td><td /><td /><td className="num">{mil(total)}</td></tr>
            </tbody>
          </table>
        </div>
        <p className="nota-pie">Cada convolución usa kernel 3, padding 1 y stride 1, seguida de BatchNorm y ReLU.</p>

        <h2 className="titulo-seccion">Cómo se entrenó</h2>
        <dl className="receta">{RECETA.map(([t, d]) => <div key={t}><dt>{t}</dt><dd>{d}</dd></div>)}</dl>
      </section>

      <section className="papel-seccion graficas-seccion">
        <h2 className="titulo-pagina-2">Gráficas del entrenamiento</h2>
        <p className="entradilla">
          Las del modelo que se entrega (E57, fold 0) y su única evaluación en test. Cada gráfica lleva debajo cómo leerla.
          La línea verde es la referencia de un buen modelo teórico (AUC 0,80), para tener una escala.
        </p>
        {g.estado === 'listo' && r.estado === 'listo'
          ? <GraficasEntrenamiento g={g} r={r} />
          : <p className="nota-pie">{g.estado === 'error' ? 'Faltan los datos: genera python -m src.web.web_graficas.' : 'Cargando gráficas…'}</p>}
      </section>
    </div>
  )
}
