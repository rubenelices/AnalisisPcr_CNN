import { useEffect, useRef, useState } from 'react'
import { AnimatePresence, motion, useInView, useMotionValueEvent, useReducedMotion, useScroll, useSpring, useTransform } from 'framer-motion'
import { Boton } from '../components/Boton.jsx'
import { IconoAvanzar } from '../components/Iconos.jsx'
import { COLOR, Grafico, Horizontal, Linea, Puntos, Texto, Vertical } from '../components/Graficas.jsx'
import { Contador, EXPO, Revelar } from '../components/Movimiento.jsx'
import { useCanales } from '../canales.js'
import { EJEMPLOS } from '../ejemplo.js'
import { fmt, mil, useGraficas, useResumen } from '../datos.js'

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

const NOMBRE_COHORTE = { spy1: 'I-SPY1', spy2: 'I-SPY2', duke: 'Duke' }
const ORDEN_COHORTES = ['duke', 'spy1', 'spy2']   // de arriba abajo en cada columna

// Reparto real de las pacientes en los 5 folds del profesor (src/web/web_resultados.py). Un punto es una paciente:
// azul sin pCR, rosa con pCR; cada columna se agrupa por cohorte (Duke, I-SPY1, I-SPY2, de arriba abajo).
// Al pulsar un fold se usa como validación y las tablas se recalculan con el resto.
function VisParticiones({ r }) {
  const { folds, test, cohortes } = r.particiones
  const [val, setVal] = useState(folds.length - 1)
  const resto = folds.filter((f) => f.fold !== val)
  const f = folds.find((x) => x.fold === val)
  const suma = (k) => resto.reduce((a, x) => a + x[k], 0)
  const ent = { pacientes: suma('pacientes'), cortes: suma('cortes'), pcr: suma('pcr'), cortes_pcr: suma('cortes_pcr') }
  const pc = (a, n) => `${fmt((a / n) * 100, 1)} %`
  const neg = ent.cortes - ent.cortes_pcr
  const grupo = (n, p, clave) => (
    <span key={clave} className="part-grupo">{[...Array(n - p).fill('n'), ...Array(p).fill('p')].map((t, i) => <i key={i} className={t} />)}</span>
  )
  return (
    <div className="vis vis-particiones">
      <div className="part-cabeza">
        <h4>Reparto de pacientes en cinco folds</h4>
        <p>Los cortes de una paciente nunca se reparten entre entrenamiento y validación. <b>Pulsa un fold</b> para usarlo como validación.</p>
      </div>
      <div className="part-cuerpo">
        <div>
          <div className="part-folds" role="group" aria-label="Elige el fold que actúa como validación">
            {folds.map((x) => (
              <button key={x.fold} className={x.fold === val ? 'part-fold activo' : 'part-fold'} onClick={() => setVal(x.fold)} aria-pressed={x.fold === val}>
                <span className="part-nombre">{x.fold === val ? 'Validación' : `Fold ${x.fold}`}</span>
                <span className="part-puntos">{ORDEN_COHORTES.filter((c) => x.cohortes[c]).map((c) => grupo(x.cohortes[c].pacientes, x.cohortes[c].pcr, c))}</span>
              </button>
            ))}
            <span className="part-sep" aria-hidden="true" />
            <div className="part-fold test" aria-label="Test público, aparte">
              <span className="part-nombre">Test</span>
              <span className="part-puntos"><span className="part-grupo">{Array.from({ length: test.pacientes }, (_, i) => <i key={i} className="t" />)}</span></span>
            </div>
          </div>
          <p className="part-leyenda"><i className="n" /> sin pCR <i className="p" /> pCR <i className="t" /> test, aparte <span className="part-orden">Cada columna, de arriba abajo: {ORDEN_COHORTES.map((c) => NOMBRE_COHORTE[c]).join(', ')}.</span></p>
        </div>
        <div className="part-lado">
          <table className="tabla part-tabla">
            <thead><tr><th>Fold {val} como validación</th><th className="num">Pacientes</th><th className="num">Cortes</th><th className="num">pCR</th></tr></thead>
            <tbody>
              <tr><td>Entrenamiento</td><td className="num">{mil(ent.pacientes)}</td><td className="num">{mil(ent.cortes)}</td><td className="num">{pc(ent.pcr, ent.pacientes)}</td></tr>
              <tr><td>Validación</td><td className="num">{mil(f.pacientes)}</td><td className="num">{mil(f.cortes)}</td><td className="num">{pc(f.pcr, f.pacientes)}</td></tr>
            </tbody>
          </table>
          <p className="part-peso">pos_weight = {mil(neg)} / {mil(ent.cortes_pcr)} = <b>{fmt(neg / ent.cortes_pcr, 2)}</b></p>
          <table className="tabla part-tabla">
            <thead><tr><th>Cohorte</th><th className="num">Train</th><th className="num">pCR en train</th><th className="num">Test</th></tr></thead>
            <tbody>
              {ORDEN_COHORTES.slice().reverse().map((c) => (
                <tr key={c}><td>{NOMBRE_COHORTE[c]}</td><td className="num">{cohortes[c].train}</td><td className="num">{pc(cohortes[c].pcr_train, 1)}</td><td className="num">{cohortes[c].test}</td></tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}

function VisModelo() {
  return (
    <div className="vis vis-modelo">
      <div className="vis-diagrama"><img src="/model/arquitectura.svg" alt="Diagrama de la arquitectura E57" /></div>
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

// Cada variante es el mismo corte aumentado con el código del entrenamiento (src/web/web_aumento.py): a la izquierda la fase
// EARLY y a la derecha el mapa de realce EARLY − PRE. Que los dos se muevan igual demuestra que las fases van sincronizadas.
const AUMENTOS = [
  { id: 'original', nombre: 'Corte original', detalle: 'Sin aumento. Es lo que ven la evaluación y la web.', uso: 'ref', etiqueta: 'referencia' },
  { id: 'espejo_horizontal', nombre: 'Espejo horizontal', detalle: 'Probabilidad 0,5. Probado antes en E49.', uso: 'si', etiqueta: 'usado en P02' },
  { id: 'espejo_vertical', nombre: 'Espejo vertical', detalle: 'Probabilidad 0,5.', uso: 'si', etiqueta: 'usado en P02' },
  { id: 'giro', nombre: 'Giro', detalle: 'Hasta ±10° (aquí +10°). Probados ±5° (E37) y ±15° (E18, E20).', uso: 'si', etiqueta: 'usado en P02' },
  { id: 'traslacion', nombre: 'Traslación', detalle: 'Hasta ±3 %, unos ±7,7 px, con relleno cero. Probado ±5 % en E18 y E20.', uso: 'si', etiqueta: 'usado en P02' },
  { id: 'combinado', nombre: 'Todos a la vez', detalle: 'Así llega un corte a la red en una época de P02.', uso: 'si', etiqueta: 'usado en P02' },
  { id: 'ganancia', nombre: 'Ganancia de intensidad', detalle: 'Un solo factor 0,9–1,1 para las tres fases (aquí ×1,1). Probada en E25.', uso: 'probado', etiqueta: 'probado, sin mejora' },
  { id: 'ganancia_por_fase', nombre: 'Cada fase por su cuenta', detalle: 'Ilustración: PRE ×0,8 y EARLY ×1,25 inventan un realce que no existe.', uso: 'no', etiqueta: 'no se usa' },
]

function VisAumento() {
  return (
    <div className="vis vis-aumento">
      <div className="aum-cabeza">
        <h4>Aumento de datos, sobre el mismo corte</h4>
        <p>La misma transformación en PRE, EARLY y LATE: se ve EARLY y el realce EARLY − PRE. Solo en entrenamiento; la evaluación y la web no aumentan nada.</p>
      </div>
      <ol className="aum-rejilla">
        {AUMENTOS.map((a, i) => (
          <motion.li key={a.id} className={`aum-tarjeta ${a.uso}`} initial={{ opacity: 0, y: 14 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.06 * i, duration: 0.5, ease: EXPO }}>
            <div className="aum-imagenes">
              <img src={`/augmentation/${a.id}_early.png`} alt={`${a.nombre}: fase EARLY`} width="256" height="256" loading="lazy" />
              <img src={`/augmentation/${a.id}_realce.png`} alt={`${a.nombre}: realce EARLY menos PRE`} width="256" height="256" loading="lazy" />
            </div>
            <p className="aum-nombre"><strong>{a.nombre}</strong><span className={`aum-etiqueta ${a.uso}`}>{a.etiqueta}</span></p>
            <p className="aum-detalle">{a.detalle}</p>
          </motion.li>
        ))}
      </ol>
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

// Lectura de cada experimento, con sus cifras del historial (src/web/web_resultados.py).
const LECTURAS = {
  C17_normal: (e) => `Pico en la época ${e.mejor_epoca}; el entrenamiento supera a la validación en ${fmt(e.brecha, 2)}.`,
  C22_groupnorm_normal: (e) => `Sin BatchNorm el pico baja a ${fmt(e.pico, 3)}: no mejora.`,
  C23_un_corte_paciente_normal_fold2: (e) => `Pico ${fmt(e.pico, 3)} en la época ${e.mejor_epoca}, pero termina en ${fmt(e.final, 3)}.`,
  E57_patient_feature_mean_mac: (e, t) => `El pico más alto con solo imagen, elegido mirando esas mismas pacientes: prometía más de lo que dio el test (${fmt(t, 3)}).`,
  N01_e57_bce_weighted_fold_0_seed_42: (e) => `La mayor caída de todas: de ${fmt(e.pico, 3)} a ${fmt(e.final, 3)}, por debajo del azar.`,
  bag_fixed_fold1: () => 'Al no elegir el pico, el resultado es honesto: pico y final coinciden.',
  B08_solo_edad_menopausia: (e) => `Dos datos clínicos solos dan ${fmt(e.pico, 3)}: azar.`,
  D08_clinica_sola: (e) => `Sin imágenes, la clínica llega a ${fmt(e.pico, 3)}, por encima de cualquier CNN de solo imagen.`,
  D17_ganancia_compartida: (e) => `El pico más alto del historial (${fmt(e.pico, 3)}), pero termina en ${fmt(e.final, 3)}.`,
  D36_normal_40: (e) => `${fmt(e.fit, 3)} en entrenamiento frente a ${fmt(e.final, 3)} en validación: memoriza.`,
}

function VisExperimentos({ r }) {
  const filas = r.experimentos
  const solo = filas.filter((e) => !e.clinica)
  const mejor = Math.max(...solo.map((e) => e.pico)), peor = Math.max(...filas.map((e) => e.caida ?? 0))
  const celda = (v, clase) => <span className={`exp-num ${clase ?? ''}`}>{v == null ? '—' : fmt(v, 3)}</span>
  return (
    <div className="vis vis-experimentos">
      <div className="exp-cabecera" aria-hidden="true">
        <span>Experimento</span><span>Pico</span><span>Final</span><span>Caída</span><span>FIT</span>
      </div>
      <ol className="exp-lista">
        {filas.map((e, i) => (
          <motion.li key={e.id} className="exp-fila" initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.05 * i, duration: 0.45, ease: EXPO }}>
            <div className="exp-nombre">
              <strong>{e.nombre}</strong>
              <span className={e.clinica ? 'exp-etiqueta clinica' : 'exp-etiqueta'}>{e.clinica ? 'con clínica' : 'solo imagen'}</span>
              <p>{e.cambia} {LECTURAS[e.id]?.(e, r.test_e57.auc)}</p>
            </div>
            {celda(e.pico, !e.clinica && e.pico === mejor ? 'mejor' : '')}
            {celda(e.final)}
            {celda(e.caida, e.caida === peor ? 'peor' : '')}
            {celda(e.fit, e.fit >= 0.9 ? 'memoriza' : '')}
          </motion.li>
        ))}
      </ol>
      <p className="vis-pie">Fold 0 salvo indicación. Pico: mejor época de la validación (optimista). Caída: pico menos final. FIT: AUC de entrenamiento al terminar, cuando se registró.
        <span className="exp-leyenda"><i className="mejor" /> mejor pico con solo imagen <i className="peor" /> mayor caída <i className="memoriza" /> memoriza</span></p>
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
  { id: 'particiones', fase: 'Datos', titulo: 'Particiones por paciente', Vis: VisParticiones, completa: true, texto: [
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
  { id: 'recetas', fase: 'Diagnóstico y mejora', titulo: 'P01 y P02: lotes y aumento de datos', Vis: VisAumento, completa: true, textoAncho: true, texto: [
    'P01 forma cada lote con 10 pacientes y 2 cortes de cada una: la red ya puede reducir su error de entrenamiento, algo que con la receta de E57 no lograba ni con 40 pacientes.',
    'P02 añade aumentos geométricos sincronizados para memorizar menos: la misma transformación en las tres fases, para que el realce no se falsee. Frenó la memorización, pero no mejoró el AUC con pacientes nuevas.'] },
  { id: 'validacion', fase: 'Evaluación y entrega', titulo: 'Validación cruzada en GPU', Vis: VisValidacion, texto: [
    'P02 se entrenó en los 5 folds con cada pérdida, en la GPU de la universidad: 10 entrenamientos.',
    'Con pacientes no vistas, BCE normal da AUC 0,532 y ponderada 0,500. La red aprende, pero lo aprendido se traslada poco.'] },
  { id: 'entrega', fase: 'Evaluación y entrega', titulo: 'Reglas fijadas antes y un único test', Vis: VisEntrega, texto: [
    'Las reglas de elección se registraron antes de ver la validación cruzada. P02 no llega al listón de 0,55, así que se entrega E57.',
    'E57 tiene su única evaluación en test: AUC 0,573. Un resultado modesto, medido sin tocar el test para elegir nada.'] },
  { id: 'experiments', fase: 'Evaluación y entrega', titulo: 'Diez experimentos del camino', Vis: VisExperimentos, ancha: true, texto: [
    'De los casi 200 experimentos del historial, diez que enseñan algo distinto: la referencia inicial, cambios de normalización y de muestreo, la pérdida ponderada, controles con datos clínicos y un entrenamiento que memoriza.',
    'Todos los picos están medidos sobre la propia validación, así que son optimistas. Los que usan datos clínicos no son comparables con los de solo imagen: el modelo entregado no los usa.'] },
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

// Recorrido en zigzag: cada paso es una fila con el texto a un lado y su resultado al otro, y el lado se
// invierte de un paso al siguiente. Cada mitad entra desde su lado al llegar a pantalla, y el visual solo se
// monta entonces, así sus animaciones (barras, líneas, contadores) empiezan cuando se ven.
function Visual({ Vis, g, r }) {
  const ref = useRef(null)
  const visto = useInView(ref, { once: true, margin: '0px 0px -15% 0px' })
  const listo = g.estado === 'listo' && r.estado === 'listo'
  return <div ref={ref} className="zz-visual-interior">{visto && listo ? <Vis g={g} r={r} /> : <span className="zz-hueco" />}</div>
}

// Una fila del zigzag. Con el scroll, el número grande y la tarjeta se mueven a distinta velocidad (paralaje).
function Fila({ i, p, g, r, reducir, refFila }) {
  const lado = i % 2 === 0 ? -1 : 1   // -1: el texto entra por la izquierda; 1: por la derecha
  const fila = useRef(null)
  const { scrollYProgress } = useScroll({ target: fila, offset: ['start end', 'end start'] })
  const yNum = useTransform(scrollYProgress, [0, 1], reducir ? [0, 0] : [70, -70])
  const yVis = useTransform(scrollYProgress, [0, 1], reducir ? [0, 0] : [26, -26])
  const entrada = (x, retraso = 0) => ({
    initial: { opacity: 0, x: reducir ? 0 : x }, whileInView: { opacity: 1, x: 0 }, viewport: { once: true, margin: '0px 0px -12% 0px' },
    transition: { duration: 0.9, delay: retraso, ease: EXPO },
  })
  return (
    <article ref={(el) => { fila.current = el; refFila(el) }} className={`${i % 2 ? 'zz-fila invertida' : 'zz-fila'}${p.ancha ? ' ancha' : ''}${p.completa ? ' completa' : ''}${p.textoAncho ? ' texto-ancho' : ''}`} data-paso={i}>
      <motion.div className="zz-texto" {...entrada(lado * 70)}>
        <motion.span className="zz-num" aria-hidden="true" style={{ y: yNum }}>{String(i + 1).padStart(2, '0')}</motion.span>
        <p className="zz-fase">{p.fase}</p>
        <h3>{p.titulo}</h3>
        {p.texto.map((x) => <p key={x}>{x}</p>)}
      </motion.div>
      <motion.div className="zz-visual-marco" {...entrada(-lado * 70, 0.12)}>
        <motion.div className="zz-visual" style={{ y: yVis }}>
          <Visual Vis={p.Vis} g={g} r={r} />
        </motion.div>
      </motion.div>
    </article>
  )
}

// Riel de progreso fijo a la izquierda mientras se recorre la sección: una línea que se rellena con el scroll,
// un nodo por paso (pulsable) y el paso actual encendido. Discreto: solo puntos y una cifra.
function Riel({ progreso, activo, onIr, visible, reducir }) {
  return (
    <motion.nav className="riel" aria-label="Pasos del trabajo" initial={false}
      animate={{ opacity: visible ? 1 : 0, x: visible || reducir ? 0 : -16 }} transition={{ duration: 0.4, ease: EXPO }}
      style={{ pointerEvents: visible ? 'auto' : 'none' }}>
      <span className="riel-cifra">{String(activo + 1).padStart(2, '0')}<small>/{PASOS.length}</small></span>
      <div className="riel-pista">
        <span className="riel-linea" />
        <motion.span className="riel-relleno" style={{ scaleY: progreso }} />
        {PASOS.map((p, i) => (
          <button key={p.id} className={i === activo ? 'riel-nodo activo' : i < activo ? 'riel-nodo pasado' : 'riel-nodo'}
            style={{ top: `${(i / (PASOS.length - 1)) * 100}%` }} onClick={() => onIr(i)} aria-label={`Paso ${i + 1}: ${p.titulo}`} aria-current={i === activo ? 'step' : undefined}>
            <span className="riel-etiqueta">{p.titulo}</span>
          </button>
        ))}
      </div>
    </motion.nav>
  )
}

function useAnchoGrande() {
  const consulta = '(min-width: 901px)'
  const [grande, setGrande] = useState(() => window.matchMedia(consulta).matches)
  useEffect(() => {
    const m = window.matchMedia(consulta)
    const f = () => setGrande(m.matches)
    m.addEventListener('change', f)
    return () => m.removeEventListener('change', f)
  }, [])
  return grande
}

// Encoge la escena fijada para que quepa en la altura de la ventana (un portátil con Chrome al 100 % deja unos 780 px).
// Se usa transform: scale en lugar de zoom a propósito: no cambia el ancho ni la maquetación del contenido, así que el alto
// natural que se mide (offsetHeight) no depende de la escala y no puede oscilar entre dos valores.
const ALTO_RESERVADO = 116   // barra superior flotante y respiro
const ESCALA_MINIMA = 0.6
function useAjusteAlto() {
  const caja = useRef(null)
  const [escala, setEscala] = useState(1)
  useEffect(() => {
    const el = caja.current
    if (!el) return
    let actual = 1
    const medir = () => {
      const z = Math.max(ESCALA_MINIMA, Math.min(1, (window.innerHeight - ALTO_RESERVADO) / el.offsetHeight))
      if (Math.abs(z - actual) > 0.01) { actual = z; setEscala(z) }
    }
    const ro = new ResizeObserver(medir)
    ro.observe(el)
    window.addEventListener('resize', medir)
    medir()
    return () => { ro.disconnect(); window.removeEventListener('resize', medir) }
  }, [])
  return { caja, escala }
}

const VH_PASO = 80  // alto de scroll por paso, en vh

// Recorrido fijado: la escena se queda quieta mientras se baja y, paso a paso, texto y visual cambian de lado
// (zigzag). El scroll es lo que avanza la historia; el número grande se mueve con el avance dentro de cada paso.
function Escenario({ g, r, control }) {
  const contenedor = useRef(null)
  const reducir = useReducedMotion()
  const [activo, setActivo] = useState(0)
  const [dentro, setDentro] = useState(false)
  const N = PASOS.length
  const { scrollYProgress } = useScroll({ target: contenedor, offset: ['start start', 'end end'] })
  const progreso = useSpring(scrollYProgress, { stiffness: 140, damping: 30, mass: 0.4 })
  useMotionValueEvent(scrollYProgress, 'change', (v) => {
    setActivo(Math.min(N - 1, Math.max(0, Math.floor(v * N))))
    setDentro(v > 0.001 && v < 0.999)
  })
  const local = useTransform(scrollYProgress, (v) => (v >= 1 ? 1 : (v * N) % 1))
  const yNum = useTransform(local, [0, 1], reducir ? [0, 0] : [60, -60])
  control.current = (i) => {
    const el = contenedor.current
    const total = el.offsetHeight - window.innerHeight
    window.scrollTo({ top: el.getBoundingClientRect().top + window.scrollY + ((i + 0.5) / N) * total, behavior: reducir ? 'auto' : 'smooth' })
  }
  const { caja, escala } = useAjusteAlto()
  const p = PASOS[activo]
  const lado = activo % 2 === 0 ? -1 : 1
  const variantes = (x) => ({
    entra: { opacity: 0, x: reducir ? 0 : x, filter: reducir ? 'none' : 'blur(6px)' },
    en: { opacity: 1, x: 0, filter: 'blur(0px)', transition: { duration: 0.55, ease: EXPO } },
    sale: { opacity: 0, x: reducir ? 0 : -x / 2, filter: reducir ? 'none' : 'blur(4px)', transition: { duration: 0.28 } },
  })
  return (
    <div className="pin" ref={contenedor} style={{ height: `${N * VH_PASO}vh` }}>
      <Riel progreso={progreso} activo={activo} visible={dentro} onIr={(i) => control.current(i)} reducir={reducir} />
      <div className="pin-escena">
        <div className="pin-ajuste" ref={caja} style={{ transform: escala < 1 ? `scale(${escala})` : undefined }}>
        <AnimatePresence mode="wait">
          <motion.article key={activo} className={`${activo % 2 ? 'zz-fila invertida pin-fila' : 'zz-fila pin-fila'}${p.ancha ? ' ancha' : ''}${p.completa ? ' completa' : ''}${p.textoAncho ? ' texto-ancho' : ''}`} initial="entra" animate="en" exit="sale">
            <motion.div className="zz-texto" variants={variantes(lado * 90)}>
              <motion.span className="zz-num" aria-hidden="true" style={{ y: yNum }}>{String(activo + 1).padStart(2, '0')}</motion.span>
              <p className="zz-fase">{p.fase}</p>
              <h3>{p.titulo}</h3>
              {p.texto.map((x) => <p key={x}>{x}</p>)}
            </motion.div>
            <motion.div className="zz-visual-marco" variants={variantes(-lado * 90)}>
              <div className="zz-visual"><Visual Vis={p.Vis} g={g} r={r} /></div>
            </motion.div>
          </motion.article>
        </AnimatePresence>
        </div>
      </div>
    </div>
  )
}

// En pantallas pequeñas no se fija nada: los pasos se apilan, uno tras otro, con el texto sobre el visual.
function Apilado({ g, r, control }) {
  const pasos = useRef([])
  const reducir = useReducedMotion()
  control.current = (i) => pasos.current[i]?.scrollIntoView({ behavior: reducir ? 'auto' : 'smooth', block: 'center' })
  return PASOS.map((p, i) => <Fila key={p.id} i={i} p={p} g={g} r={r} reducir={reducir} refFila={(el) => { pasos.current[i] = el }} />)
}

function Recorrido({ g, r, control }) {
  const grande = useAnchoGrande()
  return (
    <section className="zigzag" aria-labelledby="t-recorrido">
      <div className="zigzag-cabeza">
        <h2 id="t-recorrido" className="titulo-seccion claro">El trabajo, paso a paso</h2>
        <p>Diez pasos, de los datos a la aplicación. Sigue bajando: la escena se queda quieta y cambia con cada paso.</p>
      </div>
      {grande ? <Escenario g={g} r={r} control={control} /> : <Apilado g={g} r={r} control={control} />}
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
