import { useCallback, useEffect, useState } from 'react'
import { animate, motion, useReducedMotion } from 'framer-motion'
import { PREVALENCIA_TRAIN } from '../ejemplo.js'
import { archivosDeEjemplo } from '../api.js'
import { MapaOclusion } from '../componentes/MapaOclusion.jsx'
import { fmtProb, useGraficas, useResumen } from '../datos.js'

// Números con coma decimal, como en el resto del trabajo.
export const fmt = (v, d = 2) => v.toFixed(d).replace('.', ',')

function Visor({ datos, children }) {
  const reducir = useReducedMotion()
  // Las fases aparecen en el orden en que se adquieren: es la única entrada escalonada.
  const aparicion = (k) => ({
    initial: { opacity: 0, y: reducir ? 0 : 10 },
    animate: { opacity: 1, y: 0 },
    transition: { delay: reducir ? 0 : 0.08 + k * 0.14, duration: 0.45, ease: [0.2, 0.8, 0.2, 1] },
  })
  return (
    <section className="sala" aria-labelledby="titulo-corte">
      <div className="sala-cabecera">
        <div>
          <h2 id="titulo-corte">{datos.titulo}</h2>
          <p>{datos.subtitulo}</p>
        </div>
      </div>

      <ol className="secuencia" aria-label="Fases en el orden en que se adquieren">
        {datos.fases.map((f, k) => (
          <motion.li key={f.id} className="fase" {...aparicion(k)}>
            <figure>
              <img src={f.src} alt={`Fase ${f.id} del corte`} width="256" height="256" />
              <figcaption>
                <span className="fase-nombre">{f.id}</span>
                <span className="fase-cuando">{f.cuando}</span>
                <span className="fase-media">intensidad media {fmt(f.media, 3)}</span>
              </figcaption>
            </figure>
            {k < datos.fases.length - 1 && <span className="fase-paso" aria-hidden="true" />}
          </motion.li>
        ))}
      </ol>

      <motion.figure className="realce" {...aparicion(3)}>
        <img src={datos.realce} alt="Mapa de realce EARLY menos PRE en color" width="256" height="256" />
        <figcaption>
          <h2>Realce de contraste</h2>
          <p>
            Diferencia EARLY − PRE: las zonas brillantes captan más contraste en la primera fase.
            El modelo recibe esta misma resta como uno de sus tres canales.
          </p>
          <div className="leyenda" aria-hidden="true">
            <span className="leyenda-barra" />
            <span className="leyenda-extremos"><span>sin realce</span><span>realce alto</span></span>
          </div>
          <p className="realce-nota">Es una ayuda para mirar la imagen, no una explicación de la predicción.</p>
        </figcaption>
      </motion.figure>
      {children}
    </section>
  )
}

export function Escala({ p, umbral }) {
  const [x, setX] = useState(0)
  useEffect(() => {
    const id = requestAnimationFrame(() => setX(p))
    return () => cancelAnimationFrame(id)
  }, [p])
  const pct = (v) => `${v * 100}%`
  return (
    <div className="escala" role="img"
      aria-label={`Probabilidad ${fmt(p, 3)} sobre una escala de 0 a 1; umbral ${fmt(umbral, 3)}`}>
      <div className="escala-pista">
        <span className="escala-zona" style={{ left: pct(umbral) }} />
        <span className="escala-umbral" style={{ left: pct(umbral) }}>
          <span>umbral {fmt(umbral, 3)}</span>
        </span>
        <span className="escala-marca" style={{ left: pct(x) }} />
      </div>
      <div className="escala-extremos"><span>0</span><span>0,5</span><span>1</span></div>
    </div>
  )
}

const COMPROBACIONES = ['Tres fases: PRE, EARLY y LATE', 'PNG de 256 × 256 en escala de grises', 'Valores / 255, igual que en el entrenamiento']

// La probabilidad cuenta desde cero cada vez que llega un resultado nuevo.
function Cifra({ p, umbral }) {
  const reducir = useReducedMotion()
  const [v, setV] = useState(reducir ? p : 0)
  useEffect(() => {
    if (reducir) { setV(p); return }
    const control = animate(0, p, { duration: 1.1, ease: [0.16, 1, 0.3, 1], onUpdate: setV })
    return () => control.stop()
  }, [p, reducir])
  return <span className="cifra">{fmtProb(v, umbral)}</span>
}

// Dónde cae este corte entre las probabilidades del test (una sola evaluación, por paciente).
function Distribucion({ p, umbral, h }) {
  const b = h.bordes, paso = b[1] - b[0]
  const total = h.sin_pcr.reduce((s, x) => s + x, 0) + h.pcr.reduce((s, x) => s + x, 0)
  let debajo = 0
  b.slice(0, -1).forEach((ini, i) => {
    const n = h.sin_pcr[i] + h.pcr[i]
    if (p >= ini + paso) debajo += n
    else if (p > ini) debajo += n * ((p - ini) / paso)
  })
  const pct = Math.round((debajo / total) * 100)
  const hasta = Math.max(0.5, Math.ceil((Math.max(p, b.at(-1)) + 0.02) * 20) / 20)
  const W = 420, H = 110, x = (v) => (v / hasta) * W
  const max = Math.max(...h.sin_pcr.map((s, i) => s + h.pcr[i]))
  const y = (n) => (n / max) * (H - 18)
  return (
    <div className="distribucion">
      <svg viewBox={`0 0 ${W} ${H + 22}`} role="img" aria-label={`Este corte está por encima del ${pct} % de las probabilidades del test`}>
        {b.slice(0, -1).map((ini, i) => {
          const s = h.sin_pcr[i], c = h.pcr[i]
          return (
            <g key={i}>
              <rect x={x(ini) + 1} width={Math.max(1, x(paso) - 2)} y={H - y(s + c)} height={y(s)} className="dist-sin" />
              <rect x={x(ini) + 1} width={Math.max(1, x(paso) - 2)} y={H - y(c)} height={y(c)} className="dist-pcr" />
            </g>
          )
        })}
        <line x1={x(umbral)} x2={x(umbral)} y1={0} y2={H} className="dist-umbral" />
        <motion.g initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.6 }}>
          <line x1={x(p)} x2={x(p)} y1={0} y2={H} className="dist-marca" />
          <circle cx={x(p)} cy={6} r={5} className="dist-punto" />
        </motion.g>
        {[0, 0.1, 0.2, 0.3, 0.4, 0.5].filter((v) => v <= hasta).map((v) => (
          <text key={v} x={x(v)} y={H + 16} className="dist-tick" textAnchor={v === 0 ? 'start' : 'middle'}>{fmt(v, 1)}</text>
        ))}
      </svg>
      <p className="distribucion-texto">
        Más alta que el <strong>{pct} %</strong> de las probabilidades que dio el modelo a las 176 pacientes del test.
        <span className="leyenda-mini"><i className="dist-sin" /> sin pCR <i className="dist-pcr" /> pCR <i className="linea" /> umbral</span>
      </p>
    </div>
  )
}

function Informe({ resultado, modelo, subida }) {
  const r = useResumen(), g = useGraficas()
  if (resultado.estado === 'cargando') {
    return (
      <section className="informe" aria-labelledby="titulo-informe" aria-busy="true">
        <h2 id="titulo-informe">Resultado del corte</h2>
        <div className="resultado-tarjeta pendiente"><span className="cifra">…</span><p>El modelo está calculando la probabilidad.</p></div>
      </section>
    )
  }
  if (resultado.estado === 'error') {
    return (
      <section className="informe" aria-labelledby="titulo-informe">
        <h2 id="titulo-informe">Resultado del corte</h2>
        <div className="resultado-tarjeta pendiente"><span className="cifra">—</span><p>No se ha podido calcular.</p></div>
        <p className="informe-error" role="alert">{resultado.error}</p>
      </section>
    )
  }
  const p = resultado.probability_pcr, u = resultado.threshold
  const si = resultado.predicted_class === 1
  const distancia = Math.abs(p - u)
  const t = r.estado === 'listo' ? r.test_e57 : null
  const e = g.estado === 'listo' ? g.e57 : null
  return (
    <section className="informe" aria-labelledby="titulo-informe">
      <div className="informe-cabeza">
        <h2 id="titulo-informe">Resultado del corte</h2>
        <span className="informe-tiempo">{Math.max(1, Math.round(resultado.latency_ms))} ms · {resultado.device.toUpperCase()}</span>
      </div>

      <div className={si ? 'resultado-tarjeta si' : 'resultado-tarjeta'}>
        <div className="resultado-fila">
          <div>
            <Cifra p={p} umbral={u} />
            <p className="cifra-texto">probabilidad estimada de respuesta patológica completa</p>
          </div>
          <motion.span key={`${resultado.clave}-${si}`} className={si ? 'clase-etiqueta si' : 'clase-etiqueta'}
            initial={{ opacity: 0, scale: 0.9 }} animate={{ opacity: 1, scale: 1 }} transition={{ delay: 0.5, duration: 0.3 }}>
            {si ? 'pCR' : 'sin pCR'}
          </motion.span>
        </div>
        <Escala p={p} umbral={u} />
        <p className="resultado-distancia">
          {fmtProb(distancia, 0)} {si ? 'por encima' : 'por debajo'} del umbral {fmt(u, 3)}
          {distancia < 0.02 && ' · muy cerca: un cambio pequeño en la imagen podría cambiar la clase'}
        </p>
      </div>
      {resultado.warnings?.length > 0 && <ul className="avisos-entrada">{resultado.warnings.map((w) => <li key={w}>{w}</li>)}</ul>}

      {e && (
        <>
          <h3>Dónde cae este corte</h3>
          <Distribucion p={p} umbral={u} h={e.test.histograma} />
        </>
      )}

      {t && (
        <>
          <h3>Cómo de fiable es (test, umbral {fmt(u, 3)})</h3>
          <div className="fiabilidad">
            <div><b>{Math.round(t.sensibilidad * 100)} %</b><span>de las pacientes con pCR, detectadas</span></div>
            <div><b>{Math.round(t.especificidad * 100)} %</b><span>de las pacientes sin pCR, bien clasificadas</span></div>
            <div><b>{Math.round(t.precision * 100)} %</b><span>de aciertos cuando dice «pCR»</span></div>
          </div>
          <p className="nota">
            AUC {fmt(t.auc, 3)} en test: ordena a las pacientes algo mejor que el azar. Como referencia, alcanzó pCR el
            {' '}{Math.round(PREVALENCIA_TRAIN * 100)} % de las pacientes de entrenamiento.
          </p>
        </>
      )}

      <h3>Modelo y entrenamiento</h3>
      <dl className="ficha-casillas">
        <div><dt>Modelo</dt><dd title={resultado.model_version}>{resultado.model_version.split('_')[0]}</dd></div>
        <div><dt>Época</dt><dd>{resultado.model_version.split('@')[1]?.replace('epoch-', '') ?? '—'}{e ? ` de ${e.epocas}` : ''}</dd></div>
        <div><dt>Pesos (SHA-256)</dt><dd title={resultado.model_sha256}>{resultado.model_sha256.slice(0, 12)}…</dd></div>
        {modelo.estado === 'listo' && <div><dt>Parámetros</dt><dd>{modelo.parameters.toLocaleString('es-ES')}</dd></div>}
        <div><dt>Umbral</dt><dd>{fmt(u, 3)}</dd></div>
        <div><dt>Dispositivo</dt><dd>{resultado.device.toUpperCase()}</dd></div>
      </dl>
      <dl className="receta-mini">
        <div><dt>Entrenamiento</dt><dd>BCE normal · AdamW, tasa 0,001 · lotes de 2 pacientes · fold 0, semilla 42</dd></div>
        {e && t && <div><dt>Métricas internas</dt><dd>AUC validación {fmt(e.validacion.auc, 3)} (época elegida) · AUC test {fmt(t.auc, 3)} [{fmt(t.ic95[0], 3)}–{fmt(t.ic95[1], 3)}] · AUC por corte {fmt(e.test.corte.auc, 3)}</dd></div>}
        <div><dt>Umbral y agregación</dt><dd>fijados con la validación antes de mirar el test; una paciente = media de sus cortes</dd></div>
      </dl>

      <h3>Comprobaciones de la entrada</h3>
      <ul className="comprobaciones-chips">
        {COMPROBACIONES.map((c) => <li key={c}>{c}</li>)}
        {subida && <li>Las tres fases son del mismo corte</li>}
      </ul>

      <p className="aviso">
        Herramienta educativa, sin validez clínica. Una probabilidad alta no garantiza pCR y una baja no la descarta; no
        debe usarse para decidir un tratamiento ni para evitar una cirugía.{subida && ' Las fases subidas no se guardan.'}
      </p>
    </section>
  )
}

function Errores({ errores }) {
  return (
    <motion.div className="errores" role="alert"
      initial={{ opacity: 0, y: -6 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.25 }}>
      <strong>No se han podido usar esos ficheros.</strong>
      <ul>{errores.map((e) => <li key={e}>{e}</li>)}</ul>
      <p>Sube de nuevo las tres fases de un mismo corte, por ejemplo ISPY1_1001_z016_PRE.png, _EARLY.png y _LATE.png. Se sigue mostrando el corte anterior.</p>
    </motion.div>
  )
}

export function Prediccion({ entrada, ejemplo, errores, resultado, modelo }) {
  const datos = ejemplo
    ? { clave: ejemplo.id, titulo: `Corte ${ejemplo.paciente}, posición ${ejemplo.posicion}`,
        subtitulo: `Ejemplo del conjunto de entrenamiento público, cohorte ${ejemplo.cohorte}`,
        fases: ejemplo.fases, realce: ejemplo.realce }
    : { clave: entrada.nombre, titulo: `Corte ${entrada.nombre}`,
        subtitulo: 'Fases subidas desde tu ordenador; no se guardan',
        fases: entrada.fases, realce: entrada.realce }
  const archivosSubidos = ejemplo ? null : entrada.archivos
  // Los ejemplos se piden de nuevo como ficheros; las subidas reutilizan los que validó Pruebas.
  const obtenerArchivos = useCallback(
    () => (ejemplo ? archivosDeEjemplo(ejemplo) : Promise.resolve(archivosSubidos)),
    [ejemplo, archivosSubidos],
  )
  const conMapa = Boolean(ejemplo || archivosSubidos)
  return (
    <div className="pagina-prediccion" data-campo="inicio">
      <div className="sala-columna">
        {errores.length > 0 && <Errores errores={errores} />}
        <Visor key={datos.clave} datos={datos}>
          {conMapa && <MapaOclusion clave={datos.clave} obtenerArchivos={obtenerArchivos} fondo={datos.fases[1].src} />}
        </Visor>
      </div>
      <Informe key={`i-${datos.clave}-${resultado.estado}`} resultado={resultado} modelo={modelo} subida={!ejemplo} />
    </div>
  )
}
