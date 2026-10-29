import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { motion, useReducedMotion } from 'framer-motion'
import { EJEMPLOS } from '../ejemplo.js'
import { archivosDeEjemplo, laboratorio } from '../api.js'
import { realceDeUrls, validarFases } from '../subida.js'
import { MapaOclusion } from '../components/MapaOclusion.jsx'
import { fmt, fmtProb, useGraficas } from '../datos.js'

// Ajustes que se envían a /api/laboratorio (src/web/laboratorio.py). Todo a cero es el corte original.
const NEUTRO = {
  espejo_horizontal: false, espejo_vertical: false, giro: 0, traslacion_x: 0, traslacion_y: 0, ganancia: 1,
  sin_contraste: false, tapar: '', ganancia_PRE: 1, ganancia_EARLY: 1, ganancia_LATE: 1,
}
const esNeutro = (a) => Object.keys(NEUTRO).every((k) => a[k] === NEUTRO[k])
// Rango moderado de aumento geométrico (el mismo que el servidor usa para avisar). E57 se entrenó sin aumentos.
const RANGO = { giro: 10, traslacion: 0.03 }
const aleatorio = () => ({
  ...NEUTRO,
  espejo_horizontal: Math.random() < 0.5, espejo_vertical: Math.random() < 0.5,
  giro: Math.round((Math.random() * 2 - 1) * RANGO.giro * 10) / 10,
  traslacion_x: Math.round((Math.random() * 2 - 1) * RANGO.traslacion * 1000) / 1000,
  traslacion_y: Math.round((Math.random() * 2 - 1) * RANGO.traslacion * 1000) / 1000,
})
const signo = (v, d = 3) => `${v > 0 ? '+' : v < 0 ? '−' : '±'}${fmt(Math.abs(v), d)}`

// ------------------------------------------------------------------ elegir corte

function Galeria({ clave, onEjemplo, onSubida }) {
  const input = useRef(null)
  const [errores, setErrores] = useState([])
  const subir = async (lista) => {
    const v = await validarFases(lista)
    setErrores(v.errores)
    if (!v.errores.length) onSubida(v)
  }
  return (
    <>
      <ol className="lab-galeria">
        {EJEMPLOS.map((e) => (
          <li key={e.id}>
            <button type="button" className={clave === e.id ? 'lab-ejemplo activo' : 'lab-ejemplo'} onClick={() => onEjemplo(e.id)}
              aria-pressed={clave === e.id}>
              <img src={e.fases[1].src} alt="" width="256" height="256" loading="lazy" />
              <span className="lab-ejemplo-texto">
                <strong>{e.cohorte}</strong>
                <span>{e.paciente.replace(/^Breast_MRI_/, 'Duke ')} · {e.posicion}</span>
              </span>
              <span className={e.pcr ? 'lab-etiqueta pcr' : 'lab-etiqueta'}>{e.pcr ? 'pCR' : 'sin pCR'}</span>
            </button>
          </li>
        ))}
        <li>
          <button type="button" className={EJEMPLOS.some((e) => e.id === clave) ? 'lab-ejemplo subir' : 'lab-ejemplo subir activo'}
            onClick={() => input.current?.click()}>
            <span className="lab-subir-icono" aria-hidden="true">+</span>
            <span className="lab-ejemplo-texto">
              <strong>Tus fases</strong>
              <span>tres PNG: _PRE, _EARLY y _LATE</span>
            </span>
          </button>
          <input ref={input} type="file" accept=".png,image/png" multiple hidden
            onChange={(e) => { subir([...e.target.files]); e.target.value = '' }} />
        </li>
      </ol>
      {errores.length > 0 && <ul className="lab-errores" role="alert">{errores.map((e) => <li key={e}>{e}</li>)}</ul>}
    </>
  )
}

// ------------------------------------------------------------------ controles

function Deslizador({ nombre, valor, min, max, paso, onCambio, formato, banda, detalle }) {
  const pct = (v) => ((v - min) / (max - min)) * 100
  const fondo = banda
    ? `linear-gradient(90deg, var(--lab-pista) ${pct(banda[0])}%, var(--lab-banda) ${pct(banda[0])}%, var(--lab-banda) ${pct(banda[1])}%, var(--lab-pista) ${pct(banda[1])}%)`
    : undefined
  return (
    <label className="lab-deslizador">
      <span className="lab-control-nombre">{nombre}<output>{formato(valor)}</output></span>
      <input type="range" min={min} max={max} step={paso} value={valor} style={{ background: fondo }}
        onChange={(e) => onCambio(Number(e.target.value))} />
      {detalle && <span className="lab-control-detalle">{detalle}</span>}
    </label>
  )
}

function Interruptor({ nombre, activo, onCambio }) {
  return (
    <button type="button" className={activo ? 'lab-interruptor activo' : 'lab-interruptor'} aria-pressed={activo} onClick={() => onCambio(!activo)}>
      <span className="lab-interruptor-punto" aria-hidden="true" />{nombre}
    </button>
  )
}

function Controles({ ajustes, cambiar, onAleatorio, onRestablecer }) {
  const pct = (v) => `${signo(v * 100, 1)} %`
  return (
    <div className="lab-controles">
      <div className="lab-grupo">
        <div className="lab-grupo-cabeza">
          <h3>Aumentos geométricos</h3>
        </div>
        <p className="lab-nota">La misma transformación en las tres fases, para que el realce no se falsee. La franja clara marca un rango moderado: hasta ±10° de giro y ±3 % de desplazamiento.</p>
        <div className="lab-fila-interruptores">
          <Interruptor nombre="Espejo horizontal" activo={ajustes.espejo_horizontal} onCambio={(v) => cambiar({ espejo_horizontal: v })} />
          <Interruptor nombre="Espejo vertical" activo={ajustes.espejo_vertical} onCambio={(v) => cambiar({ espejo_vertical: v })} />
        </div>
        <Deslizador nombre="Giro" valor={ajustes.giro} min={-45} max={45} paso={0.5} banda={[-RANGO.giro, RANGO.giro]}
          formato={(v) => `${signo(v, 1)}°`} onCambio={(v) => cambiar({ giro: v })} />
        <Deslizador nombre="Desplazamiento horizontal" valor={ajustes.traslacion_x} min={-0.15} max={0.15} paso={0.005}
          banda={[-RANGO.traslacion, RANGO.traslacion]} formato={pct} onCambio={(v) => cambiar({ traslacion_x: v })} />
        <Deslizador nombre="Desplazamiento vertical" valor={ajustes.traslacion_y} min={-0.15} max={0.15} paso={0.005}
          banda={[-RANGO.traslacion, RANGO.traslacion]} formato={pct} onCambio={(v) => cambiar({ traslacion_y: v })} />
      </div>

      <div className="lab-grupo">
        <div className="lab-grupo-cabeza">
          <h3>Intensidad</h3>
        </div>
        <Deslizador nombre="Ganancia común a las tres fases" valor={ajustes.ganancia} min={0.75} max={1.25} paso={0.01}
          banda={[0.9, 1.1]} formato={(v) => `× ${fmt(v, 2)}`} onCambio={(v) => cambiar({ ganancia: v })} />
      </div>

      <div className="lab-grupo sonda">
        <div className="lab-grupo-cabeza">
          <h3>Sondas sobre las fases</h3>
          <span className="lab-chip naranja">nunca en entrenamiento</span>
        </div>
        <p className="lab-nota">Imágenes imposibles en una resonancia real, para preguntar al modelo qué usa.</p>
        <div className="lab-fila-interruptores">
          <Interruptor nombre="Quitar el contraste (EARLY y LATE = PRE)" activo={ajustes.sin_contraste} onCambio={(v) => cambiar({ sin_contraste: v })} />
        </div>
        <div className="lab-control">
          <span className="lab-control-nombre">Apagar una fase</span>
          <div className="lab-segmentos" role="radiogroup" aria-label="Apagar una fase">
            {[['', 'Ninguna'], ['PRE', 'PRE'], ['EARLY', 'EARLY'], ['LATE', 'LATE']].map(([v, t]) => (
              <button key={v} type="button" role="radio" aria-checked={ajustes.tapar === v} className={ajustes.tapar === v ? 'activo' : ''}
                onClick={() => cambiar({ tapar: v })}>{t}</button>
            ))}
          </div>
        </div>
        {['PRE', 'EARLY', 'LATE'].map((f) => (
          <Deslizador key={f} nombre={`Brillo solo de ${f}`} valor={ajustes[`ganancia_${f}`]} min={0.5} max={1.5} paso={0.01}
            formato={(v) => `× ${fmt(v, 2)}`} onCambio={(v) => cambiar({ [`ganancia_${f}`]: v })} />
        ))}
      </div>

      <div className="lab-acciones">
        <button type="button" className="boton boton-principal sobre-claro" onClick={onAleatorio}>Transformación al azar</button>
        <button type="button" className="boton boton-secundario sobre-claro" onClick={onRestablecer} disabled={esNeutro(ajustes)}>Volver al original</button>
      </div>
    </div>
  )
}

// ------------------------------------------------------------------ imagen y resultado

function Fases({ fases, realce, cambiado }) {
  return (
    <div className="lab-fases">
      {fases.map((f) => (
        <figure key={f.id} className={`lab-fase fase-${f.id.toLowerCase()}`}>
          <img src={f.src} alt={`Fase ${f.id}${cambiado ? ' modificada' : ''}`} width="256" height="256" />
          <figcaption>{f.id}</figcaption>
        </figure>
      ))}
      <figure className="lab-fase lab-realce">
        {realce ? <img src={realce} alt="Realce EARLY − PRE" width="256" height="256" /> : <span className="lab-fase-hueco" />}
        <figcaption>EARLY − PRE</figcaption>
      </figure>
    </div>
  )
}

function Escala({ p, original, umbral }) {
  const pos = (v) => `${Math.min(100, (v / 0.7) * 100)}%`
  return (
    <div className="lab-escala" role="img" aria-label={`Probabilidad ${fmt(p, 3)}; umbral ${fmt(umbral, 3)}`}>
      <div className="lab-escala-pista">
        <span className="lab-escala-zona" style={{ left: pos(umbral) }} />
        <span className="lab-escala-umbral" style={{ left: pos(umbral) }}><span>umbral {fmt(umbral, 3)}</span></span>
        {original != null && <span className="lab-escala-original" style={{ left: pos(original) }} title="original" />}
        <span className="lab-escala-marca" style={{ left: pos(p) }} />
      </div>
      <div className="lab-escala-eje"><span>0</span><span>0,35</span><span>0,7</span></div>
    </div>
  )
}

function Resultado({ ejemplo, original, actual, umbral, cambiado, avisos, error }) {
  const reducir = useReducedMotion()
  if (error) {
    return (
      <div className="lab-resultado pendiente" role="alert">
        <span className="cifra">—</span>
        <p>No se ha podido calcular el corte modificado.</p>
        <p className="lab-errores">{error}</p>
      </div>
    )
  }
  if (!actual) return <div className="lab-resultado pendiente"><span className="cifra">…</span><p>Calculando…</p></div>
  const p = actual.probability_pcr
  const si = p >= umbral
  const real = ejemplo?.pcr
  const delta = cambiado && original ? p - original.probability_pcr : null
  const salta = delta != null && (original.probability_pcr >= umbral) !== si
  return (
    <div className={si ? 'lab-resultado si' : 'lab-resultado'}>
      <div className="lab-resultado-fila">
        <div>
          <motion.span key={p} className="cifra" initial={{ opacity: reducir ? 1 : 0.4 }} animate={{ opacity: 1 }}>{fmtProb(p, umbral)}</motion.span>
          <p className="cifra-texto">probabilidad de pCR {cambiado ? 'del corte modificado' : 'del corte'}</p>
        </div>
        <span className={si ? 'clase-etiqueta si' : 'clase-etiqueta'}>{si ? 'pCR' : 'sin pCR'}</span>
      </div>
      <Escala p={p} original={cambiado ? original?.probability_pcr : null} umbral={umbral} />
      <dl className="lab-resultado-datos">
        {delta != null && (
          <div>
            <dt>Frente al original</dt>
            <dd className={salta ? 'salta' : ''}>{fmtProb(original.probability_pcr, umbral)} → {fmtProb(p, umbral)} ({signo(delta)}){salta && ' · cambia de clase'}</dd>
          </div>
        )}
        {real != null && (
          <div>
            <dt>Etiqueta real</dt>
            <dd>{real ? 'pCR' : 'sin pCR'} · <span className={(real === 1) === si ? 'acierto' : 'fallo'}>{(real === 1) === si ? 'acierta' : 'falla'}</span> con umbral {fmt(umbral, 3)}</dd>
          </div>
        )}
      </dl>
      {avisos?.length > 0 && <ul className="lab-avisos">{avisos.map((a) => <li key={a}>{a}</li>)}</ul>}
    </div>
  )
}

// ------------------------------------------------------------------ estabilidad

function Estabilidad({ archivos, clave, original, umbral }) {
  const [puntos, setPuntos] = useState([])
  const [corriendo, setCorriendo] = useState(false)
  const turno = useRef(0)
  useEffect(() => { turno.current += 1; setPuntos([]); setCorriendo(false) }, [clave])
  const lanzar = async () => {
    const mio = ++turno.current
    setPuntos([]); setCorriendo(true)
    for (let k = 0; k < 24 && turno.current === mio; k++) {
      const ajustes = aleatorio()
      try {
        const r = await laboratorio(await archivos(), ajustes)
        if (turno.current === mio) setPuntos((prev) => [...prev, { p: r.probability_pcr, ajustes, y: (k * 0.618) % 1 }])
      } catch { break }
    }
    if (turno.current === mio) setCorriendo(false)
  }
  const ps = puntos.map((x) => x.p)
  const base = original?.probability_pcr
  const cambian = base == null ? 0 : ps.filter((p) => (p >= umbral) !== (base >= umbral)).length
  const W = 640, H = 120, M = 16
  const lo = Math.min(umbral, base ?? umbral, ...ps) - 0.03, hi = Math.max(umbral, base ?? umbral, ...ps) + 0.03
  const x = (v) => M + ((v - lo) / (hi - lo)) * (W - 2 * M)
  const ticks = []
  for (let t = Math.ceil(lo * 20) / 20; t <= hi; t += 0.05) ticks.push(Math.round(t * 100) / 100)
  return (
    <div className="lab-estabilidad">
      <div className="lab-estabilidad-cabeza">
        <p className="parrafo">
          Se aplican 24 transformaciones al azar dentro del rango moderado (espejos, giro de hasta ±10° y desplazamiento de hasta ±3 %)
          y se predice cada uno. Un modelo robusto daría casi la misma probabilidad: es el mismo corte visto de otra forma.
        </p>
        <button type="button" className="boton boton-principal sobre-claro" onClick={lanzar} disabled={corriendo || !base}>
          {corriendo ? `Probando… ${puntos.length}/24` : puntos.length ? 'Repetir' : 'Lanzar 24 transformaciones'}
        </button>
      </div>
      {puntos.length > 0 && (
        <>
          <svg className="lab-tira" viewBox={`0 0 ${W} ${H + 26}`} role="img"
            aria-label={`${puntos.length} probabilidades entre ${fmt(Math.min(...ps), 3)} y ${fmt(Math.max(...ps), 3)}`}>
            <rect x={x(umbral)} y={0} width={W - M - x(umbral)} height={H} className="lab-tira-zona" />
            <line x1={x(umbral)} x2={x(umbral)} y1={0} y2={H} className="lab-tira-umbral" />
            {puntos.map((pt, i) => (
              <motion.circle key={i} cx={x(pt.p)} cy={14 + pt.y * (H - 28)} r={6} className={pt.p >= umbral ? 'lab-tira-punto si' : 'lab-tira-punto'}
                initial={{ opacity: 0, scale: 0.4 }} animate={{ opacity: 1, scale: 1 }}>
                <title>{`${fmt(pt.p, 3)} · giro ${fmt(pt.ajustes.giro, 1)}°${pt.ajustes.espejo_horizontal ? ' · espejo H' : ''}${pt.ajustes.espejo_vertical ? ' · espejo V' : ''}`}</title>
              </motion.circle>
            ))}
            <line x1={x(base)} x2={x(base)} y1={0} y2={H} className="lab-tira-original" />
            <text x={x(base)} y={H + 22} textAnchor="middle" className="lab-tira-texto fuerte">original</text>
            {ticks.map((t) => <text key={t} x={x(t)} y={H + 22} textAnchor="middle" className="lab-tira-texto">{fmt(t, 2)}</text>)}
          </svg>
          <dl className="lab-cifras">
            <div><dt>Rango</dt><dd>{fmt(Math.min(...ps), 3)} – {fmt(Math.max(...ps), 3)}</dd></div>
            <div><dt>Mayor cambio</dt><dd>{fmt(Math.max(...ps.map((p) => Math.abs(p - base))) * 100, 1)} puntos</dd></div>
            <div><dt>Cambian de clase</dt><dd>{cambian} de {puntos.length}</dd></div>
          </dl>
          <p className="lab-nota">
            E57 se entrenó sin aumentos de datos: nunca vio estos giros ni espejos.
          </p>
        </>
      )}
    </div>
  )
}

// ------------------------------------------------------------------ umbral

const metricas = ([, vp, fp, vn, fn]) => ({
  vp, fp, vn, fn,
  sens: vp + fn ? vp / (vp + fn) : 0, esp: vn + fp ? vn / (vn + fp) : 0, prec: vp + fp ? vp / (vp + fp) : null,
})

// Por encima de ~0,36 el modelo no llama «pCR» a nadie: la gráfica y el deslizador se quedan donde algo cambia.
const UMBRAL_MIN = 0.05, UMBRAL_MAX = 0.4

function Curvas({ filas: todas, umbral, entregado }) {
  const W = 520, H = 200, M = { i: 36, d: 12, a: 12, b: 28 }
  const filas = todas.filter((r) => r[0] >= UMBRAL_MIN && r[0] <= UMBRAL_MAX)
  const x = (u) => M.i + ((u - UMBRAL_MIN) / (UMBRAL_MAX - UMBRAL_MIN)) * (W - M.i - M.d)
  const y = (v) => M.a + (1 - v) * (H - M.a - M.b)
  const linea = (f) => filas.map((r, i) => `${i ? 'L' : 'M'}${x(r[0]).toFixed(1)},${y(f(metricas(r))).toFixed(1)}`).join('')
  return (
    <svg className="lab-curvas" viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Sensibilidad y especificidad según el umbral">
      {[0, 0.5, 1].map((v) => <g key={v}><line x1={M.i} x2={W - M.d} y1={y(v)} y2={y(v)} className="lab-rejilla" /><text x={M.i - 6} y={y(v) + 4} textAnchor="end" className="lab-tira-texto">{fmt(v, 1)}</text></g>)}
      {[0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4].map((u) => <text key={u} x={x(u)} y={H - 8} textAnchor="middle" className="lab-tira-texto">{fmt(u, 2)}</text>)}
      <line x1={x(entregado)} x2={x(entregado)} y1={M.a} y2={H - M.b} className="lab-curvas-entregado" />
      <path d={linea((m) => m.sens)} className="lab-curva sens" />
      <path d={linea((m) => m.esp)} className="lab-curva esp" />
      <line x1={x(umbral)} x2={x(umbral)} y1={M.a} y2={H - M.b} className="lab-curvas-umbral" />
    </svg>
  )
}

function Matriz({ titulo, m }) {
  return (
    <div className="lab-matriz">
      <p className="lab-matriz-titulo">{titulo}</p>
      <div className="lab-matriz-celdas">
        <span /><span className="cab">dice sin pCR</span><span className="cab">dice pCR</span>
        <span className="cab fila">sin pCR</span><b className="bien">{m.vn}</b><b className="mal">{m.fp}</b>
        <span className="cab fila">pCR</span><b className="mal">{m.fn}</b><b className="bien">{m.vp}</b>
      </div>
      <dl className="lab-matriz-metricas">
        <div><dt>Sensibilidad</dt><dd>{Math.round(m.sens * 100)} %</dd></div>
        <div><dt>Especificidad</dt><dd>{Math.round(m.esp * 100)} %</dd></div>
        <div><dt>Precisión</dt><dd>{m.prec == null ? '—' : `${Math.round(m.prec * 100)} %`}</dd></div>
      </dl>
    </div>
  )
}

function Umbral({ umbral, setUmbral, entregado, actual }) {
  const g = useGraficas()
  if (g.estado !== 'listo' || !g.e57.umbrales) return null
  const { validacion, test } = g.e57.umbrales
  const fila = (tabla) => metricas(tabla.reduce((a, r) => (Math.abs(r[0] - umbral) < Math.abs(a[0] - umbral) ? r : a)))
  return (
    <div className="lab-umbral">
      <div className="lab-umbral-mando">
        <Deslizador nombre="Umbral de decisión" valor={umbral} min={UMBRAL_MIN} max={UMBRAL_MAX} paso={0.005} formato={(v) => fmt(v, 3)} onCambio={setUmbral}
          detalle={actual ? `Este corte (${fmtProb(actual.probability_pcr, umbral)}) se clasifica como ${actual.probability_pcr >= umbral ? 'pCR' : 'sin pCR'}.` : null} />
        <button type="button" className="boton boton-secundario sobre-claro" onClick={() => setUmbral(entregado)} disabled={Math.abs(umbral - entregado) < 1e-9}>
          Volver a {fmt(entregado, 3)}
        </button>
        <Curvas filas={validacion} umbral={umbral} entregado={entregado} />
        <p className="lab-leyenda"><i className="sens" /> sensibilidad <i className="esp" /> especificidad (validación interna) <i className="entregado" /> umbral entregado</p>
      </div>
      <div className="lab-umbral-matrices">
        <Matriz titulo={`Validación interna · ${validacion[0][1] + validacion[0][2]} pacientes`} m={fila(validacion)} />
        <Matriz titulo={`Test público · ${test[0][1] + test[0][2]} pacientes`} m={fila(test)} />
      </div>
    </div>
  )
}

// ------------------------------------------------------------------ página

export function Laboratorio({ entrada, ejemplo, resultado, modelo, onEjemplo, onSubida, ir }) {
  const clave = ejemplo ? ejemplo.id : entrada.nombre
  const fasesOriginales = ejemplo ? ejemplo.fases : entrada.fases
  const realceOriginal = ejemplo ? ejemplo.realce : entrada.realce
  const entregado = modelo.estado === 'listo' ? modelo.threshold : 0.325
  const [umbral, setUmbral] = useState(entregado)
  useEffect(() => { setUmbral(entregado) }, [entregado])

  // Los ejemplos se piden como ficheros una vez por corte; las subidas ya los traen.
  const cache = useRef({})
  const archivos = useCallback(() => {
    if (!ejemplo) return Promise.resolve(entrada.archivos)
    cache.current[ejemplo.id] ??= archivosDeEjemplo(ejemplo)
    return cache.current[ejemplo.id]
  }, [ejemplo, entrada.archivos])

  const [ajustes, setAjustes] = useState(NEUTRO)
  const [modificado, setModificado] = useState(null)
  const [realceMod, setRealceMod] = useState(null)
  useEffect(() => { setAjustes(NEUTRO); setModificado(null) }, [clave])
  const cambiar = (parcial) => setAjustes((a) => ({ ...a, ...parcial }))

  const turno = useRef(0)
  useEffect(() => {
    const mio = ++turno.current
    if (esNeutro(ajustes)) { setModificado(null); return undefined }
    setModificado((m) => (m ? { ...m, cargando: true } : { cargando: true }))
    const espera = setTimeout(async () => {
      try {
        const r = await laboratorio(await archivos(), ajustes)
        if (turno.current === mio) setModificado(r)
      } catch (e) {
        // Un servidor arrancado antes de añadir el laboratorio no conoce la ruta (404/405): se dice cómo arreglarlo.
        const viejo = /\((404|405)\)/.test(e.message)
        const error = viejo ? 'El servidor en marcha es una versión anterior sin laboratorio: reinícialo (python app.py).' : e.message
        if (turno.current === mio) setModificado({ error })
      }
    }, 160)
    return () => clearTimeout(espera)
  }, [ajustes, archivos])

  useEffect(() => {
    if (!modificado?.fases) { setRealceMod(null); return undefined }
    let vivo = true
    realceDeUrls(modificado.fases.PRE, modificado.fases.EARLY).then((u) => vivo && setRealceMod(u)).catch(() => {})
    return () => { vivo = false }
  }, [modificado?.fases])

  const original = resultado.estado === 'listo' && resultado.clave === clave ? resultado : null
  const cambiado = !esNeutro(ajustes)
  const actual = cambiado ? (modificado?.probability_pcr != null ? modificado : null) : original
  const fases = cambiado && modificado?.fases
    ? fasesOriginales.map((f) => ({ ...f, src: modificado.fases[f.id] }))
    : fasesOriginales
  const fondos = useMemo(() => Object.fromEntries(fasesOriginales.map((f) => [f.id, f.src])), [fasesOriginales])
  const titulo = ejemplo
    ? `${ejemplo.paciente}, corte ${ejemplo.posicion} · ${ejemplo.cohorte}`
    : `${entrada.nombre} · tus fases (no se guardan)`

  return (
    <div className="pagina-lab">
      <section className="papel-seccion lab-seccion">
        <p className="lab-paso">1 · Elige un corte</p>
        <h2 className="titulo-seccion">Nueve cortes del train público, o los tuyos</h2>
        <p className="parrafo">
          Tres de cada cohorte, con y sin pCR. Los seis nuevos se eligieron al azar con semilla fija, sin mirar qué predice el
          modelo; la etiqueta es la pública del train.
        </p>
        <Galeria clave={clave} onEjemplo={onEjemplo} onSubida={onSubida} />
      </section>

      <section className="papel-seccion lab-seccion">
        <p className="lab-paso">2 · Transforma el corte y compara</p>
        <h2 className="titulo-seccion">{titulo}</h2>
        <div className="lab-mesa">
          <div className={modificado?.cargando ? 'lab-visor recalculando' : 'lab-visor'}>
            <Fases fases={fases} realce={cambiado ? realceMod : realceOriginal} cambiado={cambiado} />
            <p className="lab-nota">
              {cambiado ? 'Así recibe la red el corte modificado (las imágenes se recortan a [0, 1] solo para dibujarlas).' : 'El corte tal como lo recibe la red: cada fase dividida entre 255.'}
              {' '}El realce EARLY − PRE es una ayuda para mirar, no una explicación.
            </p>
          </div>
          <div className="lab-lateral">
            <Resultado ejemplo={ejemplo} original={original} actual={actual} umbral={umbral} cambiado={cambiado} avisos={cambiado ? modificado?.fuera_de_rango : null} error={cambiado ? modificado?.error : null} />
            <Controles ajustes={ajustes} cambiar={cambiar} onAleatorio={() => setAjustes(aleatorio())} onRestablecer={() => setAjustes(NEUTRO)} />
          </div>
        </div>
      </section>

      <section className="papel-seccion lab-seccion">
        <p className="lab-paso">3 · ¿Es estable?</p>
        <h2 className="titulo-seccion">El mismo corte, 24 veces</h2>
        <Estabilidad archivos={archivos} clave={clave} original={original} umbral={umbral} />
      </section>

      <section className="papel-seccion lab-seccion">
        <p className="lab-paso">4 · De probabilidad a decisión</p>
        <h2 className="titulo-seccion">Mueve el umbral</h2>
        <p className="parrafo">
          El modelo da una probabilidad; el umbral la convierte en «pCR» o «sin pCR». Bajarlo detecta más pacientes con pCR a
          costa de más falsas alarmas. El umbral entregado, {fmt(entregado, 3)}, se fijó mirando la validación interna antes de
          abrir el test; mover el deslizador no cambia el modelo ni sus resultados.
        </p>
        <Umbral umbral={umbral} setUmbral={setUmbral} entregado={entregado} actual={actual} />
      </section>

      <section className="papel-seccion lab-seccion">
        <p className="lab-paso">5 · Qué zonas mueven la salida</p>
        <h2 className="titulo-seccion">Mapa de oclusión</h2>
        <MapaOclusion clave={clave} obtenerArchivos={archivos} fondos={fondos} />
      </section>

      <section className="papel-seccion lab-seccion lab-pie">
        <p className="lab-nota">
          La arquitectura y el entrenamiento están en <button type="button" className="boton-texto" onClick={() => ir('modelo')}>Modelo</button>;
          las métricas, en <button type="button" className="boton-texto" onClick={() => ir('resultados')}>Resultados</button>.
          Herramienta educativa, sin validez clínica: una probabilidad no es un diagnóstico ni una recomendación de tratamiento.
        </p>
      </section>
    </div>
  )
}
