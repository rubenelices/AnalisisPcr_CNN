import { useEffect, useLayoutEffect, useRef, useState } from 'react'
import { motion, useReducedMotion } from 'framer-motion'
import { mil } from '../datos.js'

// El diagrama de volúmenes de E57 (python -m src.document_architectures --only E57…) con una «cámara»
// que se acerca a cada parte de la red al bajar o al pulsarla. Las zonas se calculan con la misma
// rejilla con la que matplotlib dibujó el SVG: la columna i está en x = i en unidades de datos.
const ANCHO = 1671.66, ALTO = 568.8
const sx = (d) => 23.45 + (d + 0.5) * 81.24      // datos → SVG (comprobado con el título y «Sigmoide»)
const sy = (d) => 40.91 + (4.55 - d) * 74.92
const columnas = (c0, c1, y0 = -0.62, y1 = 3.98) => ({ x: sx(c0 - 0.6), y: sy(y1), w: sx(c1 + 0.66) - sx(c0 - 0.6), h: sy(y0) - sy(y1) })

export const SECCIONES = [
  { id: 'general', nombre: 'Vista general', zona: null, salida: '3 × 256 × 256 → 1', params: 298161,
    texto: ['Un corte entra por la izquierda con sus tres fases y sale por la derecha convertido en una probabilidad. Por el camino, cuatro bloques convolucionales reducen la imagen de 256 × 256 a 16 × 16 mientras aumentan los canales de 3 a 128.',
      'Colores del diagrama: azul, entrada; verde, convolución; naranja, reducción espacial; violeta, cabeza de salida. Debajo de cada volumen está su tamaño exacto.'] },
  { id: 'entrada', nombre: 'Entrada', zona: columnas(0, 1), salida: '3 × 256 × 256', params: 0,
    texto: ['Las tres fases del mismo corte (PRE, EARLY y LATE), en escala de grises y divididas entre 255, igual que en el entrenamiento.',
      'Antes de la primera convolución se convierten sin parámetros en PRE, EARLY − PRE (cuánto realza cada zona al llegar el contraste) y LATE − EARLY (si sigue entrando o se lava). Darle la resta hecha evita que la red tenga que aprenderla.'] },
  { id: 'b1', nombre: 'Bloque 1', zona: columnas(2, 4), salida: '16 × 128 × 128', params: 2800,
    texto: ['Convolución 3 × 3 con 16 filtros, BatchNorm y ReLU; MaxPool 2 × 2, que reduce la imagen a la mitad; y otra convolución 3 × 3.',
      'Los filtros de esta escala responden a bordes, texturas y diferencias locales de realce.'] },
  { id: 'b2', nombre: 'Bloque 2', zona: columnas(5, 7), salida: '32 × 64 × 64', params: 13952,
    texto: ['La misma estructura con 32 canales. Cada reducción duplica el campo de visión de las neuronas siguientes.',
      'Los patrones locales del bloque anterior se combinan en formas algo mayores.'] },
  { id: 'b3', nombre: 'Bloque 3', zona: columnas(8, 10), salida: '64 × 32 × 32', params: 55552,
    texto: ['64 canales sobre una rejilla de 32 × 32. Aquí está casi una quinta parte de los parámetros de la red.',
      'Cada posición resume ya una región amplia del corte original.'] },
  { id: 'b4', nombre: 'Bloque 4', zona: columnas(11, 13), salida: '128 × 16 × 16', params: 221696,
    texto: ['El último bloque, con 128 canales, concentra tres cuartas partes de los parámetros: 221.696 de 298.161.',
      'Al final, cada neurona ve unos 106 × 106 píxeles del corte: lo bastante para abarcar la lesión y su entorno.'] },
  { id: 'gap', nombre: 'Resumen (GAP)', zona: columnas(14, 14), salida: '128', params: 0,
    texto: ['La media espacial global resume cada uno de los 128 mapas en un único número, sin parámetros.',
      'Así el resultado no depende de en qué parte del recorte esté el tumor, y la red no puede memorizar posiciones.'] },
  { id: 'cabeza', nombre: 'Cabeza', zona: columnas(15, 17), salida: '1 logit', params: 4161,
    texto: ['Una capa lineal de 128 a 32 con ReLU, dropout del 30 % durante el entrenamiento y una lineal final de 32 a 1.',
      'La salida es un logit: la BCEWithLogitsLoss lo recibe directamente durante el entrenamiento.'] },
  { id: 'salida', nombre: 'Salida', zona: { x: sx(17.45), y: sy(2.35), w: ANCHO - sx(17.45), h: sy(0.05) - sy(2.35) }, salida: 'probabilidad', params: 0,
    texto: ['La sigmoide convierte el logit en la probabilidad de pCR del corte. Para una paciente se promedian las probabilidades de todos sus cortes.',
      'La clase sale de comparar con el umbral 0,325, fijado con la validación antes de mirar el test. En la web se predice un corte suelto, como en la defensa.'] },
]

function useTamano(ref) {
  const [t, setT] = useState({ w: 0, h: 0 })
  useLayoutEffect(() => {
    const ro = new ResizeObserver(([e]) => setT({ w: e.contentRect.width, h: e.contentRect.height }))
    ro.observe(ref.current)
    return () => ro.disconnect()
  }, [ref])
  return t
}

function camara(zona, w, h) {
  if (!w) return { x: 0, y: 0, scale: 1 }
  const k = w / ANCHO, alto = ALTO * k, dy = (h - alto) / 2
  const general = !zona
  zona ??= { x: 0, y: 0, w: ANCHO, h: ALTO }
  const s = Math.min((w * (general ? 0.98 : 0.92)) / (zona.w * k), (h * (general ? 0.98 : 0.86)) / (zona.h * k), general ? 1 : 3.4)
  const cx = (zona.x + zona.w / 2) * k, cy = dy + (zona.y + zona.h / 2) * k
  return { x: w / 2 - s * cx, y: h / 2 - s * cy, scale: s }
}

// El diagrama queda fijo arriba y las explicaciones pasan por debajo con el scroll normal.
// Cada explicación mide lo mismo que el hueco libre bajo el diagrama, así nunca queda vacío;
// la cámara se mueve cuando una explicación cruza la franja central de ese hueco.
export function ArquitecturaInteractiva({ onCambio }) {
  const [activo, setActivo] = useState(0)
  const escenario = useRef(null)
  const fija = useRef(null)
  const pasos = useRef([])
  const { w, h } = useTamano(escenario)
  const reducir = useReducedMotion()
  const n = SECCIONES.length

  useEffect(() => {
    const obs = new IntersectionObserver((entradas) => {
      entradas.forEach((e) => { if (e.isIntersecting) setActivo(Number(e.target.dataset.paso)) })
    }, { rootMargin: '-79% 0px -20% 0px' })
    pasos.current.forEach((el) => el && obs.observe(el))
    return () => obs.disconnect()
  }, [])
  useEffect(() => { onCambio?.(SECCIONES[activo].id) }, [activo, onCambio])

  const ir = (i) => {
    const el = pasos.current[Math.max(0, Math.min(n - 1, i))]
    if (!el) return
    const r = el.getBoundingClientRect()
    // Donde quedará el borde inferior del diagrama cuando esté fijo (top: 64 px), aunque ahora aún no lo esté.
    const borde = parseFloat(getComputedStyle(fija.current).top) + fija.current.offsetHeight
    window.scrollTo({ top: window.scrollY + r.top - borde, behavior: reducir ? 'auto' : 'smooth' })
  }
  const teclas = (e) => {
    if (e.key === 'ArrowRight' || e.key === 'ArrowDown') { e.preventDefault(); ir(activo + 1) }
    if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') { e.preventDefault(); ir(activo - 1) }
  }

  const sec = SECCIONES[activo]
  const cam = camara(sec.zona, w, h)
  const k = w / ANCHO, dy = (h - ALTO * k) / 2
  const z = sec.zona
  return (
    <div className="arqi">
      <div className="arqi-fija" ref={fija}>
        <div className="arqi-escenario" ref={escenario} tabIndex={0} onKeyDown={teclas}
          aria-label="Diagrama de la arquitectura. Usa las flechas para recorrer sus partes.">
          <motion.div className="arqi-lienzo" style={{ width: w, height: h, originX: 0, originY: 0 }}
            animate={cam} transition={reducir ? { duration: 0 } : { duration: 0.9, ease: [0.65, 0, 0.35, 1] }}>
            <img src="/modelo/arquitectura.svg" alt="Diagrama de volúmenes de la arquitectura E57, capa a capa"
              style={{ top: dy, width: w }} draggable="false" />
            <svg className="arqi-capa" viewBox={`0 0 ${ANCHO} ${ALTO}`} style={{ top: dy, width: w, height: ALTO * k }}>
              <motion.path fillRule="evenodd" className="arqi-velo"
                d={z ? `M0 0H${ANCHO}V${ALTO}H0Z M${z.x} ${z.y}h${z.w}v${z.h}h${-z.w}Z` : `M0 0H${ANCHO}V${ALTO}H0Z`}
                animate={{ opacity: z ? 1 : 0 }} transition={{ duration: 0.5 }} />
              {SECCIONES.map((s, i) => s.zona && (
                <rect key={s.id} x={s.zona.x} y={s.zona.y} width={s.zona.w} height={s.zona.h} rx="10"
                  className={i === activo ? 'arqi-zona activa' : 'arqi-zona'} onClick={() => ir(i)}>
                  <title>{s.nombre}</title>
                </rect>
              ))}
            </svg>
          </motion.div>
        </div>
        <nav className="arqi-chips" aria-label="Partes de la red">
          <button className="arqi-flecha" onClick={() => ir(activo - 1)} disabled={activo === 0} aria-label="Parte anterior">‹</button>
          {SECCIONES.map((s, i) => (
            <button key={s.id} className={i === activo ? 'arqi-chip activo' : 'arqi-chip'} onClick={() => ir(i)}
              aria-current={i === activo ? 'step' : undefined}>{s.nombre}</button>
          ))}
          <button className="arqi-flecha" onClick={() => ir(activo + 1)} disabled={activo === n - 1} aria-label="Parte siguiente">›</button>
        </nav>
      </div>

      <div className="arqi-pasos">
        {SECCIONES.map((s, i) => (
          <article key={s.id} data-paso={i} ref={(el) => { pasos.current[i] = el }} className={i === activo ? 'arqi-paso activo' : 'arqi-paso'}>
            <div className="arqi-paso-tarjeta">
              <div className="arqi-paso-cabeza">
                <p className="arqi-paso-num">{i === 0 ? 'Recorrido' : `${i} de ${n - 1}`}</p>
                <h3>{s.nombre}</h3>
                <dl>
                  <div><dt>Salida</dt><dd>{s.salida}</dd></div>
                  <div><dt>Parámetros</dt><dd>{mil(s.params)}</dd></div>
                </dl>
              </div>
              <div className="arqi-paso-cuerpo">{s.texto.map((x) => <p key={x}>{x}</p>)}</div>
            </div>
          </article>
        ))}
      </div>
    </div>
  )
}
