import { useEffect, useRef } from 'react'
import { useReducedMotion } from 'framer-motion'
import { CAPAS } from './Arquitectura.jsx'

// Fondo de la portada: la red E57 dibujada en perspectiva con sus dimensiones reales (CAPAS).
// Cada bloque convolucional es una pila de láminas: el lado sigue la resolución espacial (256 → 16)
// y el número de láminas crece con los canales (3 → 128). Después vienen el GAP (128 valores),
// la capa de 32 y la salida, un único nodo. Un pulso de luz recorre la red de entrada a salida:
// lento en reposo, rápido mientras se calcula una predicción y al llegar una nueva.
// Es una ilustración, no una medida: las proporciones están comprimidas para que quepa en pantalla.

const FASES = ['#6fb1ff', '#ffc65c', '#ff7aa8'] // PRE, EARLY, LATE, como las etiquetas de la zona de carga
const BLOQUES = CAPAS.filter((c) => /^b\d$/.test(c.id))
const lado = (s) => 4 * Math.pow(s, 0.45)                 // 256 → ~49, 16 → ~14
const laminas = (c) => Math.round(2 + 1.1 * Math.log2(c)) // 16 → 6, 128 → 10
const ESPACIO = 1.7
const GAP_ENTRE_BLOQUES = 9

// Geometría fija en coordenadas del mundo: X a lo largo de la red, Y arriba, Z hacia el espectador.
function construir() {
  const hojas = []   // { x, h, bloque, color? }
  const bloques = []
  let x = 0
  FASES.forEach((color, i) => hojas.push({ x: i * 2.4, h: lado(256) / 2, bloque: -1, color }))
  x = 2 * 2.4 + 12
  BLOQUES.forEach((b, k) => {
    const n = laminas(b.c), h = lado(b.s) / 2, x0 = x
    for (let i = 0; i < n; i++) hojas.push({ x: x0 + i * ESPACIO, h, bloque: k })
    bloques.push({ x0, x1: x0 + (n - 1) * ESPACIO, h })
    x = x0 + (n - 1) * ESPACIO + GAP_ENTRE_BLOQUES
  })
  const xGap = x + 2, xCapa = xGap + 11, xSalida = xCapa + 11
  const columna = (n, alto) => Array.from({ length: n }, (_, i) => ({ y: (i - (n - 1) / 2) * (alto / (n - 1)) }))
  return { hojas, bloques, xGap, xCapa, xSalida, gap: columna(16, 16), capa: columna(8, 12), fin: x + 60 }
}

export function RedFondo({ resultado }) {
  const reducir = useReducedMotion()
  const caja = useRef(null)
  const lienzo = useRef(null)
  const estado = useRef({ resultado, pulsos: [], brillo: 0, vistoClave: null, raton: { x: 0, y: 0 }, suave: { x: 0, y: 0 } })

  useEffect(() => {
    const e = estado.current
    e.resultado = resultado
    if (resultado.estado === 'listo') {
      const clave = `${resultado.clave}|${resultado.probability_pcr}`
      if (e.vistoClave !== clave) {
        if (e.vistoClave !== null) { e.pulsos.push({ x: -6, v: 75 }); e.brillo = 1 }
        e.vistoClave = clave
      }
    }
  }, [resultado])

  useEffect(() => {
    const c = lienzo.current, el = caja.current
    const ctx = c.getContext('2d')
    const red = construir()
    const e = estado.current
    const polvo = Array.from({ length: 70 }, (_, i) => ({ x: (i * 0.618) % 1, y: (i * 0.377 + 0.1) % 1, r: 0.6 + ((i * 7) % 5) * 0.28, v: 0.004 + ((i * 3) % 7) * 0.0014, t: i % 3 }))
    let W = 0, H = 0, dpr = 1, escala = 1, cx = 0, cy = 0, bucle = 0, visible = true, ultimo = 0, inactivo = 0

    // Proyección: giro alrededor de Y (la red se aleja hacia la derecha), inclinación y perspectiva.
    const yaw0 = -0.42, tilt0 = 0.14, F = 520
    const proyectar = (X, Y, Z, yaw, tilt, k = escala, ox = cx, oy = cy) => {
      const xr = X * Math.cos(yaw) + Z * Math.sin(yaw), zr = -X * Math.sin(yaw) + Z * Math.cos(yaw)
      const yr = Y * Math.cos(tilt) - zr * Math.sin(tilt), zz = Y * Math.sin(tilt) + zr * Math.cos(tilt)
      const p = F / (F + zz)
      return [ox + xr * p * k, oy - yr * p * k, p]
    }
    const ajustar = () => {
      const r = el.getBoundingClientRect()
      dpr = Math.min(window.devicePixelRatio || 1, 2)
      W = r.width; H = r.height
      c.width = Math.round(W * dpr); c.height = Math.round(H * dpr)
      // Encaja el contorno de toda la red en una caja de la pantalla.
      let x0 = 1e9, x1 = -1e9, y0 = 1e9, y1 = -1e9
      for (const hoja of red.hojas) for (const [Y, Z] of [[-1, -1], [1, 1], [-1, 1], [1, -1]]) {
        const [px, py] = proyectar(hoja.x, Y * hoja.h, Z * hoja.h, yaw0, tilt0, 1, 0, 0)
        x0 = Math.min(x0, px); x1 = Math.max(x1, px); y0 = Math.min(y0, py); y1 = Math.max(y1, py)
      }
      const [sx] = proyectar(red.xSalida, 0, 0, yaw0, tilt0, 1, 0, 0)
      x1 = Math.max(x1, sx)
      escala = Math.min((W * 0.86) / (x1 - x0), (H * 0.62) / (y1 - y0))
      cx = W * 0.5 - ((x0 + x1) / 2) * escala
      cy = H * 0.43 - ((y0 + y1) / 2) * escala
    }

    const pulso = (x, pulsos) => {
      let w = 0
      for (const p of pulsos) w = Math.max(w, Math.exp(-(((x - p.x) / 7) ** 2)))
      return w
    }

    const dibujar = (t, dt) => {
      const { pulsos } = e
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
      ctx.clearRect(0, 0, W, H)
      e.suave.x += (e.raton.x - e.suave.x) * 0.05; e.suave.y += (e.raton.y - e.suave.y) * 0.05
      const yaw = yaw0 + e.suave.x * 0.07 + (reducir ? 0 : Math.sin(t / 5200) * 0.025)
      const tilt = tilt0 + e.suave.y * 0.05
      const P = (X, Y, Z) => proyectar(X, Y, Z, yaw, tilt)

      // Polvo de luz, en pantalla.
      if (!reducir) {
        for (const d of polvo) {
          d.y -= d.v * dt / 16; if (d.y < -0.02) d.y = 1.02
          ctx.fillStyle = ['rgba(255,160,200,0.35)', 'rgba(150,180,255,0.3)', 'rgba(255,210,150,0.25)'][d.t]
          ctx.beginPath(); ctx.arc(d.x * W, d.y * H, d.r, 0, 6.283); ctx.fill()
        }
      }

      // Enlaces entre bloques: cuatro líneas de la última lámina de uno a la primera del siguiente.
      const bl = red.bloques
      const enlaces = [[red.hojas[2].x, red.hojas[2].h, bl[0].x0, bl[0].h], ...bl.slice(0, -1).map((b, i) => [b.x1, b.h, bl[i + 1].x0, bl[i + 1].h])]
      for (const [xa, ha, xb, hb] of enlaces) {
        const w = pulso((xa + xb) / 2, pulsos)
        ctx.strokeStyle = `rgba(${120 + 120 * w}, ${150 - 30 * w}, ${210 - 40 * w}, ${0.1 + 0.4 * w})`
        ctx.lineWidth = 1
        for (const [Y, Z] of [[-1, -1], [1, 1], [-1, 1], [1, -1]]) {
          const a = P(xa, Y * ha, Z * ha), b = P(xb, Y * hb, Z * hb)
          ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke()
        }
      }

      // Láminas, de la más lejana a la más cercana.
      for (const hoja of [...red.hojas].sort((a, b) => b.x - a.x)) {
        const w = pulso(hoja.x, pulsos)
        const q = [[-1, -1], [1, -1], [1, 1], [-1, 1]].map(([Y, Z]) => P(hoja.x, Y * hoja.h, Z * hoja.h))
        ctx.beginPath(); q.forEach(([px, py], i) => (i ? ctx.lineTo(px, py) : ctx.moveTo(px, py))); ctx.closePath()
        if (hoja.color) {
          ctx.fillStyle = hoja.color + '22'; ctx.strokeStyle = hoja.color + (w > 0.2 ? 'cc' : '88')
        } else {
          ctx.fillStyle = `rgba(${90 + 150 * w}, ${130 - 10 * w}, ${200 - 40 * w}, ${0.04 + 0.2 * w})`
          ctx.strokeStyle = `rgba(${105 + 135 * w}, ${150 - 30 * w}, ${215 - 45 * w}, ${0.32 + 0.6 * w})`
        }
        ctx.lineWidth = 1
        if (w > 0.25) { ctx.shadowColor = hoja.color ?? 'rgba(235,110,170,0.9)'; ctx.shadowBlur = 14 * w } else ctx.shadowBlur = 0
        ctx.fill(); ctx.stroke()
        ctx.shadowBlur = 0
      }

      // Cabeza: GAP (128 → 16 puntos), capa de 32 (8 puntos) y salida (1 nodo).
      const nodos = (columnaX, lista, radio) => lista.map((n) => ({ ...n, p: P(columnaX, n.y, 0), r: radio }))
      const g = nodos(red.xGap, red.gap, 2.2), k = nodos(red.xCapa, red.capa, 2.8)
      const salida = P(red.xSalida, 0, 0)
      ctx.lineWidth = 0.7
      const wk = pulso(red.xCapa, pulsos)
      ctx.strokeStyle = `rgba(200,140,200,${0.1 + 0.3 * wk})`
      for (const a of g) for (const b of k) { if (Math.abs(a.y * 0.75 - b.y) > 4.2) continue; ctx.beginPath(); ctx.moveTo(a.p[0], a.p[1]); ctx.lineTo(b.p[0], b.p[1]); ctx.stroke() }
      for (const b of k) { ctx.beginPath(); ctx.moveTo(b.p[0], b.p[1]); ctx.lineTo(salida[0], salida[1]); ctx.stroke() }
      const punto = (p, r, w, rosa) => {
        ctx.fillStyle = rosa ? `rgba(240,120,175,${0.55 + 0.45 * w})` : `rgba(${150 + 90 * w},${170 - 40 * w},${225 - 50 * w},${0.5 + 0.5 * w})`
        if (w > 0.2) { ctx.shadowColor = 'rgba(235,110,170,0.95)'; ctx.shadowBlur = 12 * w }
        ctx.beginPath(); ctx.arc(p[0], p[1], r * p[2] * (escala / 3.2), 0, 6.283); ctx.fill(); ctx.shadowBlur = 0
      }
      const wg = pulso(red.xGap, pulsos)
      g.forEach((n) => punto(n.p, n.r, wg, false)); k.forEach((n) => punto(n.p, n.r, wk, false))

      // Nodo de salida: halo grande que se enciende con cada predicción.
      const ws = Math.max(pulso(red.xSalida, pulsos), e.brillo)
      const rs = 70 * salida[2] * (escala / 3.2)
      const halo = ctx.createRadialGradient(salida[0], salida[1], 0, salida[0], salida[1], rs)
      halo.addColorStop(0, `rgba(255,150,200,${0.18 + 0.6 * ws})`); halo.addColorStop(0.35, `rgba(220,90,150,${0.08 + 0.25 * ws})`); halo.addColorStop(1, 'rgba(220,90,150,0)')
      ctx.fillStyle = halo; ctx.beginPath(); ctx.arc(salida[0], salida[1], rs, 0, 6.283); ctx.fill()
      punto(salida, 6.5, ws, true)

      // Avance de los pulsos.
      for (const p of pulsos) p.x += (p.v * dt) / 1000
      e.pulsos = pulsos.filter((p) => p.x < red.fin * 0.8 + 20)
      e.brillo = Math.max(0, e.brillo - dt / 2600)
    }

    const cuadro = (t) => {
      bucle = 0
      if (!visible || document.hidden) return
      const dt = Math.min(60, ultimo ? t - ultimo : 16); ultimo = t
      const { resultado: r, pulsos } = e
      if (r.estado === 'cargando') { inactivo = 0; if (!pulsos.some((p) => p.x < 14)) pulsos.push({ x: -6, v: 62 }) }
      else { inactivo += dt; if (inactivo > 6500 && !pulsos.length) { pulsos.push({ x: -6, v: 15 }); inactivo = 0 } }
      dibujar(t, dt)
      bucle = requestAnimationFrame(cuadro)
    }
    const arrancar = () => { if (!reducir && !bucle) { ultimo = 0; bucle = requestAnimationFrame(cuadro) } }

    ajustar()
    dibujar(performance.now(), 16)
    arrancar()
    const ro = new ResizeObserver(() => { ajustar(); dibujar(performance.now(), 16) })
    ro.observe(el)
    const io = new IntersectionObserver(([x]) => { visible = x.isIntersecting; if (visible) arrancar() })
    io.observe(el)
    const mover = (ev) => {
      const r = el.getBoundingClientRect()
      e.raton.x = ((ev.clientX - r.left) / r.width - 0.5) * 2; e.raton.y = ((ev.clientY - r.top) / r.height - 0.5) * 2
    }
    if (!reducir) window.addEventListener('pointermove', mover, { passive: true })
    const visibilidad = () => { if (!document.hidden) arrancar() }
    document.addEventListener('visibilitychange', visibilidad)
    return () => {
      cancelAnimationFrame(bucle); ro.disconnect(); io.disconnect()
      window.removeEventListener('pointermove', mover); document.removeEventListener('visibilitychange', visibilidad)
    }
  }, [reducir])

  return (
    <div className="red-fondo" ref={caja} aria-hidden="true">
      <span className="red-fondo-brillo" />
      <canvas ref={lienzo} />
    </div>
  )
}
