import { useEffect, useRef, useState } from 'react'
import { AnimatePresence, motion, useReducedMotion, useScroll, useSpring } from 'framer-motion'
import { Boton } from '../componentes/Boton.jsx'
import { IconoAvanzar } from '../componentes/Iconos.jsx'
import { COLOR, Grafico, Horizontal, Linea, Puntos, Texto, Vertical } from '../componentes/Graficas.jsx'
import { Contador, EXPO, Revelar } from '../componentes/Movimiento.jsx'
import { useCanales } from '../canales.js'
import { EJEMPLOS } from '../ejemplo.js'
import { fmt, useGraficas, useResumen } from '../datos.js'

// ---------------------------------------------------------------- visuales de cada paso

function VisDatos() {
  const e = EJEMPLOS[0]
  const barras = [
    { nombre: 'Entrenamiento', total: 1097, pcr: 322 },
    { nombre: 'Test público', total: 176, pcr: 53 },
  ]
  const cohortes = [['I-SPY2', 784], ['Duke', 209], ['I-SPY1', 104]]
  return (
    <div className="vis vis-datos">
      <div className="vis-fases">
        {e.fases.map((f) => <figure key={f.id}><img src={f.src} alt={`Fase ${f.id}`} width="180" height="180" /><figcaption>{f.id}</figcaption></figure>)}
      </div>
      <div className="vis-barras">
        {barras.map((b) => (
          <div key={b.nombre} className="vis-barra">
            <span className="vis-barra-nombre">{b.nombre} · {b.total.toLocaleString('es-ES')} pacientes</span>
            <span className="vis-barra-pista" style={{ width: `${(b.total / 1097) * 100}%` }}>
              <motion.span className="vis-barra-pcr" initial={{ width: 0 }} animate={{ width: `${(b.pcr / b.total) * 100}%` }} transition={{ duration: 0.9, ease: EXPO }} />
            </span>
            <span className="vis-barra-nota">{b.pcr} con pCR ({Math.round((b.pcr / b.total) * 100)} %)</span>
          </div>
        ))}
        <p className="vis-titulo">Cohortes de entrenamiento</p>
        <div className="vis-cohortes">
          {cohortes.map(([c, n]) => <span key={c} style={{ flexGrow: n }}><b>{c}</b>{n}</span>)}
        </div>
      </div>
    </div>
  )
}

function VisCanales() {
  const canales = useCanales(EJEMPLOS[0])
  return (
    <div className="vis vis-canales">
      {['PRE', 'EARLY − PRE', 'LATE − EARLY'].map((n, i) => (
        <figure key={n}>
          {canales ? <img src={canales[i]} alt={`Canal ${n}`} width="200" height="200" /> : <span className="canal-hueco" />}
          <figcaption>{n}</figcaption>
        </figure>
      ))}
      <p className="vis-pie">PNG de 256 × 256 en escala de grises, divididos entre 255. Las dos restas son fijas, sin parámetros.</p>
    </div>
  )
}

function VisParticiones() {
  const reducir = useReducedMotion()
  const puntos = Array.from({ length: 110 }, (_, i) => {
    const fold = Math.floor(i / 22)
    if (fold === 0) return 'exterior'
    return (i - 22) % 5 === 4 ? 'interna' : 'fit'
  })
  return (
    <div className="vis vis-particiones">
      <div className="vis-puntos">
        {puntos.map((p, i) => (
          <motion.span key={i} className={`pt pt-${p}`} initial={{ opacity: 0, scale: reducir ? 1 : 0.4 }} animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: reducir ? 0 : (i % 22) * 0.012 + Math.floor(i / 22) * 0.08, duration: 0.3 }} />
        ))}
      </div>
      <div className="vis-puntos test">{Array.from({ length: 18 }, (_, i) => <span key={i} className="pt pt-test" />)}</div>
      <ul className="vis-leyenda">
        <li><span className="pt pt-exterior" />Fold exterior: 219 pacientes no vistas</li>
        <li><span className="pt pt-fit" />FIT: 80 % del resto, entrena</li>
        <li><span className="pt pt-interna" />Validación interna: 20 %, elige época y umbral</li>
        <li><span className="pt pt-test" />Test público: aparte, una sola evaluación</li>
      </ul>
      <p className="vis-pie">Cada punto son unas 10 pacientes. Todos los cortes de una paciente van siempre al mismo grupo.</p>
    </div>
  )
}

function VisModelo() {
  return (
    <div className="vis vis-modelo">
      <div className="vis-diagrama"><img src="/modelo/arquitectura.svg" alt="Diagrama de la arquitectura E57" /></div>
      <dl className="vis-cifras">
        <div><dt>Parámetros</dt><dd>298.161</dd></div>
        <div><dt>Entrada</dt><dd>3 × 256 × 256</dd></div>
        <div><dt>Salida</dt><dd>p(pCR)</dd></div>
      </dl>
    </div>
  )
}

function VisEntrenamiento({ g }) {
  const e = g.e57
  return (
    <div className="vis vis-grafico">
      <Grafico x={[1, e.historia.length]} y={[0.4, 0.9]} ticksX={[1, 6, 12, 18, 24]} ticksY={[0.4, 0.5, 0.6, 0.7, 0.8, 0.9]} fmtX={(v) => v}
        etiquetaX="Época" etiquetaY="AUC de validación" descripcion="AUC de validación de E57 por época">
        <Horizontal y={0.5} etiqueta="azar" />
        <Horizontal y={0.8} color={COLOR.ref} tipo="puntos" etiqueta="buen modelo" />
        <Vertical x={e.epoca_elegida} etiqueta={`época ${e.epoca_elegida}`} />
        <Linea datos={e.historia.map((h) => [h.epoch, h.patient_roc_auc])} color={COLOR.valid} puntos />
      </Grafico>
      <p className="vis-pie">E57, fold 0. Se guarda la época con mejor AUC de validación y, con ella, se fija el umbral.</p>
    </div>
  )
}

function VisDiagnostico({ g }) {
  const folds = g.e57.perdidas_folds
  const n = Math.max(...folds.map((f) => f.length))
  return (
    <div className="vis vis-grafico">
      <Grafico x={[1, n]} y={[0.56, 0.66]} ticksX={[1, 5, 10, 15, 20, 25].filter((v) => v <= n)} ticksY={[0.56, 0.58, 0.6, 0.62, 0.64, 0.66]} fmtX={(v) => v} fmtY={(v) => fmt(v, 2)}
        etiquetaX="Época" etiquetaY="BCE de entrenamiento" descripcion="Pérdida de entrenamiento de E57 en 5 folds frente al predictor constante">
        <Horizontal y={g.bce_constante} etiqueta={`predictor constante ${fmt(g.bce_constante, 3)}`} color={COLOR.acento} />
        {folds.map((f, i) => <Linea key={i} datos={f.map((v, k) => [k + 1, v])} color={COLOR.train} grosor={1.6} retraso={i * 0.1} />)}
      </Grafico>
      <p className="vis-pie">Pérdida de entrenamiento de E57 en los 5 folds: no baja del nivel de un modelo que responde siempre el 29 %.</p>
    </div>
  )
}

function VisRecetas({ g }) {
  const m = g.memorizacion
  return (
    <div className="vis vis-grafico">
      <Grafico x={[1, 30]} y={[0, 0.7]} ticksX={[1, 10, 20, 30]} ticksY={[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]} fmtX={(v) => v}
        etiquetaX="Época" etiquetaY="Pérdida de entrenamiento" descripcion="Prueba de memorización con 40 pacientes: pérdida de E57 frente a P01">
        <Horizontal y={g.bce_constante} etiqueta="predictor constante" color={COLOR.acento} />
        {m.e57 && <Linea datos={m.e57.map((h) => [h.epoch, h.train_online_objective])} color={COLOR.azar} />}
        {m.p01 && <Linea datos={m.p01.map((h) => [h.epoch, h.train_online_objective])} color={COLOR.train} retraso={0.2} />}
        {m.e57 && <Texto x={16} y={m.e57.at(-1).train_online_objective} dy={22} clase="nota-grafico">E57: 2 pacientes × 10 cortes por lote</Texto>}
        {m.p01 && <Texto x={16} y={0.12} dy={-10} clase="nota-grafico azul">P01: 10 pacientes × 2 cortes por lote</Texto>}
      </Grafico>
      <p className="vis-pie">Prueba de control con 40 pacientes y 30 épocas: con la receta de E57 la pérdida no baja de {fmt(m.e57?.at(-1).train_online_objective ?? 0, 2)}; con lotes de 10 pacientes llega a {fmt(m.p01?.at(-1).train_online_objective ?? 0, 2)}.</p>
    </div>
  )
}

function VisValidacion({ r }) {
  const n = r.cv_p02.normal, w = r.cv_p02.weighted
  return (
    <div className="vis vis-grafico">
      <Grafico x={[0.4, 2.6]} y={[0.4, 0.65]} ticksX={[1, 2]} ticksY={[0.4, 0.45, 0.5, 0.55, 0.6, 0.65]} fmtX={(v) => (v === 1 ? 'BCE normal' : 'BCE ponderada')} fmtY={(v) => fmt(v, 2)}
        etiquetaX="" etiquetaY="AUC exterior" descripcion="AUC exterior de los 5 folds con BCE normal y ponderada">
        <Horizontal y={0.5} etiqueta="azar" />
        <Horizontal y={0.55} color={COLOR.acento} tipo="puntos" etiqueta="listón para sustituir a E57 (0,55)" lado="izquierda" />
        <Puntos datos={n.folds.map((f, i) => [0.82 + i * 0.09, f.exterior])} color={COLOR.train} r={6} />
        {w && <Puntos datos={w.folds.map((f, i) => [1.82 + i * 0.09, f.exterior])} color={COLOR.valid} r={6} />}
        <Texto x={1} y={0.42}>{`conjunta ${fmt(n.auc, 3)}`}</Texto>
        {w && <Texto x={2} y={0.42}>{`conjunta ${fmt(w.auc, 3)}`}</Texto>}
      </Grafico>
      <p className="vis-pie">Un punto por fold: 219 pacientes no vistas cada uno, 1.097 en total.</p>
    </div>
  )
}

function VisEntrega({ r }) {
  const t = r.test_e57
  const pos = (v) => `${((v - 0.4) / 0.4) * 100}%`
  return (
    <div className="vis vis-entrega">
      <ol className="vis-reglas">
        <li><span>AUC conjunta de P02 ≥ 0,55</span><b className="no">0,532: no</b></li>
        <li><span>Ponderada claramente mejor</span><b className="no">IC incluye 0: no</b></li>
        <li><span>Modelo entregado</span><b>E57</b></li>
      </ol>
      <div className="vis-test">
        <p className="vis-titulo">Test público · {t.pacientes} pacientes · una evaluación</p>
        <p className="vis-test-cifra"><Contador valor={t.auc} decimales={3} /> <span>AUC</span></p>
        <div className="vis-ic">
          <span className="vis-ic-banda" style={{ left: pos(t.ic95[0]), width: `calc(${pos(t.ic95[1])} - ${pos(t.ic95[0])})` }} />
          <span className="vis-ic-punto" style={{ left: pos(t.auc) }} />
          <span className="vis-ic-azar" style={{ left: pos(0.5) }} />
        </div>
        <div className="vis-ic-eje"><span>0,4</span><span>0,5</span><span>0,6</span><span>0,7</span><span>0,8</span></div>
        <p className="vis-pie">Intervalo del 95 %: {fmt(t.ic95[0], 3)} a {fmt(t.ic95[1], 3)}.</p>
      </div>
    </div>
  )
}

function VisWeb() {
  return (
    <div className="vis vis-web">
      <div className="vis-iguales">
        <div><span>Pipeline de evaluación</span><b>0,336232</b></div>
        <span className="vis-igual" aria-hidden="true">=</span>
        <div><span>Web</span><b>0,336232</b></div>
      </div>
      <p className="vis-pie">Corte ISPY1_1001_z016: la misma probabilidad en los dos caminos, porque el servidor usa el mismo preprocesado y el mismo modelo en modo evaluación.</p>
    </div>
  )
}

// ---------------------------------------------------------------- pasos

const PASOS = [
  { id: 'datos', fase: 'Datos', titulo: 'Los datos', Vis: VisDatos, texto: [
    'BreastDCEDL: resonancias de mama con contraste de 1.097 pacientes de entrenamiento y 176 de test, de tres cohortes (I-SPY1, I-SPY2 y Duke).',
    'De cada paciente hay unos 10 cortes, y de cada corte tres fases: PRE (antes del contraste), EARLY y LATE. Son tiempos de adquisición, no colores.',
    'El objetivo es binario: respuesta patológica completa (pCR) o no. Solo la alcanzó el 29 % de las pacientes.'] },
  { id: 'preprocesado', fase: 'Datos', titulo: 'Preprocesado', Vis: VisCanales, texto: [
    'Cada fase es un PNG de 256 × 256 en escala de grises que se divide entre 255. Nada más: sin normalización de ImageNet ni aumentos de color.',
    'El mismo código se usa en entrenamiento, evaluación y web. Es el error más común que se busca en la defensa.'] },
  { id: 'particiones', fase: 'Datos', titulo: 'Particiones por paciente', Vis: VisParticiones, texto: [
    'La unidad estadística es la paciente: sus cortes están correlacionados y nunca se reparten entre grupos.',
    'Se usan los 5 folds del profesor. Dentro de cada uno, el 80 % entrena (FIT) y el 20 % (validación interna, semilla 42) elige la época y el umbral.',
    'El test público no participa en ninguna decisión y se evalúa una sola vez.'] },
  { id: 'arquitectura', fase: 'Modelo', titulo: 'Una CNN desde cero', Vis: VisModelo, texto: [
    'Red convolucional 2D propia de 298.161 parámetros, sin pesos preentrenados ni arquitecturas importadas.',
    'Cuatro bloques convolucionales, una media espacial global y una cabeza lineal. Cada corte da una probabilidad; la de una paciente es la media de sus cortes.'] },
  { id: 'entrenamiento', fase: 'Modelo', titulo: 'Entrenamiento y selección', Vis: VisEntrenamiento, texto: [
    'Se compararon BCEWithLogitsLoss normal y ponderada (pos_weight = N0/N1), como pide el enunciado.',
    'La época se elige por el AUC en la validación interna, y con esa época se fija el umbral de decisión (0,325 en E57), siempre antes de mirar el test.'] },
  { id: 'diagnostico', fase: 'Diagnóstico y mejora', titulo: 'Diagnóstico: la red no aprendía', Vis: VisDiagnostico, texto: [
    'Tras más de cien experimentos, el mejor modelo (E57) apenas bajaba de la pérdida de un predictor constante.',
    'La causa estaba en la receta, no en la arquitectura: con lotes de 2 pacientes, las estadísticas de BatchNorm se calculan casi dentro de cada paciente y borran las diferencias entre ellas.'] },
  { id: 'recetas', fase: 'Diagnóstico y mejora', titulo: 'P01 y P02: una receta que aprende', Vis: VisRecetas, texto: [
    'P01 forma cada lote con 10 pacientes y 2 cortes aleatorios de cada una: la red pasa a aprender.',
    'P02 añade aumento geométrico sincronizado en las tres fases (volteos, rotación de 10°, traslación del 3 %) para memorizar menos.'] },
  { id: 'validacion', fase: 'Evaluación y entrega', titulo: 'Validación cruzada en GPU', Vis: VisValidacion, texto: [
    'P02 se entrenó en los 5 folds con cada pérdida, en la GPU de la universidad: 10 entrenamientos.',
    'Con pacientes no vistas, BCE normal da AUC 0,532 y ponderada 0,500. La red aprende, pero lo aprendido se traslada poco.'] },
  { id: 'entrega', fase: 'Evaluación y entrega', titulo: 'Reglas fijadas antes y un único test', Vis: VisEntrega, texto: [
    'Las reglas de elección se registraron antes de ver la validación cruzada. P02 no llega al listón de 0,55, así que se entrega E57.',
    'E57 tiene su única evaluación en test: AUC 0,573. Un resultado modesto, medido sin tocar el test para elegir nada.'] },
  { id: 'web', fase: 'Evaluación y entrega', titulo: 'La aplicación', Vis: VisWeb, texto: [
    'Servidor en Flask con el modelo en modo evaluación e interfaz en React. Valida cada imagen y responde con probabilidad, clase, umbral, versión, dispositivo y tiempo.',
    'La sección Pruebas acepta varios cortes a la vez para la defensa.'] },
]

const FASES = ['Datos', 'Modelo', 'Diagnóstico y mejora', 'Evaluación y entrega']

function Pipeline({ onIr }) {
  const reducir = useReducedMotion()
  return (
    <div className="pipeline">
      {FASES.map((fase, k) => (
        <Revelar key={fase} className="pipeline-fase" orden={k}>
          <p className="pipeline-fase-nombre">{fase}</p>
          <ol>
            {PASOS.map((p, i) => ({ p, i })).filter(({ p }) => p.fase === fase).map(({ p, i }) => (
              <li key={p.id}>
                <button onClick={() => onIr(i)}><span className="pipeline-num">{i + 1}</span>{p.titulo}</button>
              </li>
            ))}
          </ol>
          {k < FASES.length - 1 && (
            <motion.span className="pipeline-flecha" aria-hidden="true" initial={{ scaleX: reducir ? 1 : 0 }} whileInView={{ scaleX: 1 }}
              viewport={{ once: true }} transition={{ delay: 0.3 + k * 0.15, duration: 0.6, ease: EXPO }} />
          )}
        </Revelar>
      ))}
    </div>
  )
}

// Recorrido: los textos pasan con el scroll normal y la gráfica de la derecha queda fija y cambia
// con el paso que cruza el centro de la pantalla. Cada paso mide poco más de media pantalla.
function Recorrido({ g, r, control }) {
  const [activo, setActivo] = useState(0)
  const seccion = useRef(null)
  const pasos = useRef([])
  const reducir = useReducedMotion()
  const { scrollYProgress } = useScroll({ target: seccion, offset: ['start center', 'end center'] })
  const progreso = useSpring(scrollYProgress, { stiffness: 120, damping: 30 })
  useEffect(() => {
    const obs = new IntersectionObserver((entradas) => {
      entradas.forEach((e) => { if (e.isIntersecting) setActivo(Number(e.target.dataset.paso)) })
    }, { rootMargin: '-48% 0px -48% 0px' })
    pasos.current.forEach((el) => el && obs.observe(el))
    return () => obs.disconnect()
  }, [])
  control.current = (i) => pasos.current[i]?.scrollIntoView({ behavior: reducir ? 'auto' : 'smooth', block: 'center' })
  const listo = g.estado === 'listo' && r.estado === 'listo'
  const Vis = PASOS[activo].Vis
  return (
    <section className="recorrido" ref={seccion} aria-labelledby="t-recorrido">
      <div className="recorrido-cabeza">
        <h2 id="t-recorrido" className="titulo-seccion claro">El trabajo, paso a paso</h2>
        <p>Diez pasos, de los datos a la aplicación. Al bajar, a la derecha aparece lo que produjo cada paso.</p>
      </div>
      <div className="recorrido-rejilla">
        <div className="recorrido-pasos">
          {PASOS.map((p, i) => (
            <article key={p.id} data-paso={i} ref={(el) => { pasos.current[i] = el }} className={i === activo ? 'paso activo' : 'paso'}>
              <div className="paso-tarjeta">
                <p className="paso-fase">{i + 1} de {PASOS.length} · {p.fase}</p>
                <h3>{p.titulo}</h3>
                {p.texto.map((x) => <p key={x}>{x}</p>)}
                <div className="paso-visual-movil">{listo && <p.Vis g={g} r={r} />}</div>
              </div>
            </article>
          ))}
        </div>
        <div className="recorrido-escena">
          <div className="escena-fija">
            <div className="escena-progreso" aria-hidden="true"><motion.span style={{ scaleX: progreso }} /></div>
            <p className="escena-paso">Paso {activo + 1} de {PASOS.length} · {PASOS[activo].titulo}</p>
            <div className="escena-marco">
              <AnimatePresence mode="wait">
                <motion.div key={activo} className="escena-contenido"
                  initial={{ opacity: 0, y: reducir ? 0 : 24, filter: 'blur(6px)' }} animate={{ opacity: 1, y: 0, filter: 'blur(0px)' }}
                  exit={{ opacity: 0, y: reducir ? 0 : -16, filter: 'blur(4px)' }} transition={{ duration: 0.45, ease: EXPO }}>
                  {listo ? <Vis g={g} r={r} /> : <p className="vis-pie">Cargando…</p>}
                </motion.div>
              </AnimatePresence>
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}

export function Portada({ ir }) {
  const g = useGraficas(), r = useResumen()
  const control = useRef(null)
  const irPaso = (i) => control.current?.(i)
  const cifras = [
    { v: 1097, t: 'pacientes de entrenamiento' }, { v: 322, t: 'con pCR (29 %)' }, { v: 3, t: 'cohortes' },
    { v: 176, t: 'pacientes de test' }, { v: 298161, t: 'parámetros entrenados desde cero' },
  ]
  return (
    <div className="portada">
      <section className="papel-seccion intro">
        <h1 className="titulo-pagina">Anticipar la respuesta a la quimioterapia</h1>
        <p className="entradilla">
          Antes de operar un cáncer de mama, muchas pacientes reciben quimioterapia neoadyuvante. Si tras ella no queda tumor
          invasivo, hay respuesta patológica completa (pCR), y solo se sabe después de la cirugía. Este trabajo estudia si una
          red convolucional puede anticiparla a partir de la resonancia de antes del tratamiento.
        </p>
        <div className="intro-columnas">
          <Revelar orden={0}><h2 className="subtitulo">El problema</h2><p>Clasificar cada paciente en pCR o no pCR con solo imágenes: tres fases de resonancia con contraste por corte. Sin datos clínicos como entrada.</p></Revelar>
          <Revelar orden={1}><h2 className="subtitulo">Las reglas</h2><p>CNN 2D propia, sin transfer learning. Particiones por paciente. El test no se usa para elegir nada y se evalúa una vez.</p></Revelar>
          <Revelar orden={2}><h2 className="subtitulo">El resultado</h2><p>AUC 0,573 en test. Lo importante es cómo se ha llegado: un diagnóstico de por qué no aprendía, una receta corregida y una medida honesta.</p></Revelar>
        </div>
        <dl className="cifras-clave">
          {cifras.map((c, i) => <Revelar key={c.t} orden={i}><dd><Contador valor={c.v} /></dd><dt>{c.t}</dt></Revelar>)}
        </dl>
      </section>

      <section className="papel-seccion">
        <h2 className="titulo-seccion">Pipeline del trabajo</h2>
        <p className="parrafo">Cuatro fases y diez pasos. Pulsa cualquiera para ir a su explicación.</p>
        <Pipeline onIr={irPaso} />
      </section>

      <Recorrido g={g} r={r} control={control} />

      <section className="papel-seccion cierre">
        <h2 className="titulo-seccion">Dónde seguir</h2>
        <div className="cierre-rejilla">
          <Revelar orden={0} className="cierre-item"><h3>Modelo</h3><p>La arquitectura capa a capa y todas las gráficas del entrenamiento, con su explicación.</p><Boton variante="secundario" fondo="claro" icono={<IconoAvanzar />} onClick={() => ir('modelo')}>Ver el modelo</Boton></Revelar>
          <Revelar orden={1} className="cierre-item"><h3>Resultados</h3><p>Test, validación cruzada, normal frente a ponderada, cohortes y decisión de entrega.</p><Boton variante="secundario" fondo="claro" icono={<IconoAvanzar />} onClick={() => ir('resultados')}>Ver los resultados</Boton></Revelar>
          <Revelar orden={2} className="cierre-item destacado"><h3>Pruebas</h3><p>Carga las imágenes de uno o varios cortes y obtén la predicción al momento.</p><Boton variante="principal" icono={<IconoAvanzar />} onClick={() => ir('pruebas')}>Ir a pruebas</Boton></Revelar>
        </div>
      </section>
    </div>
  )
}
