import { createContext, useContext, useId } from 'react'
import { motion, useReducedMotion } from 'framer-motion'
import { fmt } from '../datos.js'

// Gráficas en SVG propias de la web: mismos colores y tipografía que el resto, y las líneas
// se dibujan al entrar en pantalla. Cada gráfica declara sus dominios; los hijos usan la escala.
export const COLOR = {
  train: '#6aa6e0',      // entrenamiento / FIT
  valid: '#e8934f',      // validación interna
  test: '#e8ebed',       // test o pacientes no vistas
  ref: '#5fc08a',        // buen modelo teórico
  azar: '#6b7680',
  acento: '#d8689f',
}

const Escala = createContext(null)
const M = { arriba: 16, derecha: 18, abajo: 46, izquierda: 54 }

export function Grafico({ x, y, ticksX, ticksY, etiquetaX, etiquetaY, fmtX = (v) => fmt(v, 1), fmtY = (v) => fmt(v, 1),
  ancho = 640, alto = 380, descripcion, children }) {
  const w = ancho - M.izquierda - M.derecha, h = alto - M.arriba - M.abajo
  const sx = (v) => M.izquierda + ((v - x[0]) / (x[1] - x[0])) * w
  const sy = (v) => M.arriba + h - ((v - y[0]) / (y[1] - y[0])) * h
  const recorte = useId().replace(/:/g, '')
  return (
    <Escala.Provider value={{ sx, sy, x, y, w, h, recorte }}>
      <svg className="grafico" viewBox={`0 0 ${ancho} ${alto}`} role="img" aria-label={descripcion}>
        <defs><clipPath id={recorte}><rect x={M.izquierda} y={M.arriba - 4} width={w} height={h + 8} /></clipPath></defs>
        {ticksY.map((t) => (
          <g key={`y${t}`}>
            <line x1={M.izquierda} x2={M.izquierda + w} y1={sy(t)} y2={sy(t)} className="rejilla" />
            <text x={M.izquierda - 8} y={sy(t)} className="tick" textAnchor="end" dominantBaseline="middle">{fmtY(t)}</text>
          </g>
        ))}
        {ticksX.map((t) => (
          <text key={`x${t}`} x={sx(t)} y={M.arriba + h + 18} className="tick" textAnchor="middle">{fmtX(t)}</text>
        ))}
        <line x1={M.izquierda} x2={M.izquierda + w} y1={M.arriba + h} y2={M.arriba + h} className="eje" />
        <text x={M.izquierda + w / 2} y={alto - 6} className="etiqueta-eje" textAnchor="middle">{etiquetaX}</text>
        <text transform={`translate(14 ${M.arriba + h / 2}) rotate(-90)`} className="etiqueta-eje" textAnchor="middle">{etiquetaY}</text>
        <g clipPath={`url(#${recorte})`}>{children}</g>
      </svg>
    </Escala.Provider>
  )
}

const camino = (datos, sx, sy) => datos.filter(([, b]) => b !== null)
  .map(([a, b], i) => `${i ? 'L' : 'M'}${sx(a)},${sy(b)}`).join('')

// Línea que se dibuja al entrar en pantalla. tipo: 'solida' | 'discontinua' | 'puntos'
export function Linea({ datos, color, tipo = 'solida', grosor = 2.4, retraso = 0, puntos = false }) {
  const { sx, sy } = useContext(Escala)
  const reducir = useReducedMotion()
  const d = camino(datos, sx, sy)
  const raya = tipo === 'discontinua' ? '7 6' : tipo === 'puntos' ? '2 5' : undefined
  const anim = reducir || raya
    ? { initial: { opacity: 0 }, whileInView: { opacity: 1 }, transition: { duration: 0.6, delay: retraso } }
    : { initial: { pathLength: 0 }, whileInView: { pathLength: 1 }, transition: { duration: 1.3, delay: retraso, ease: [0.65, 0, 0.35, 1] } }
  return (
    <g>
      <motion.path d={d} fill="none" stroke={color} strokeWidth={grosor} strokeDasharray={raya}
        strokeLinecap="round" strokeLinejoin="round" viewport={{ once: true, margin: '0px 0px -15% 0px' }} {...anim} />
      {puntos && datos.filter(([, b]) => b !== null).map(([a, b]) => (
        <motion.circle key={a} cx={sx(a)} cy={sy(b)} r={3.2} fill={color}
          initial={{ opacity: 0 }} whileInView={{ opacity: 1 }} viewport={{ once: true }} transition={{ delay: retraso + 0.9 }} />
      ))}
    </g>
  )
}

export function Horizontal({ y: v, color = COLOR.azar, tipo = 'discontinua', etiqueta, lado = 'derecha' }) {
  const { sx, sy, x } = useContext(Escala)
  const xs = lado === 'derecha' ? sx(x[1]) - 4 : sx(x[0]) + 4
  return (
    <g>
      <line x1={sx(x[0])} x2={sx(x[1])} y1={sy(v)} y2={sy(v)} stroke={color} strokeWidth={1.6} strokeDasharray={tipo === 'puntos' ? '2 5' : '7 6'} />
      {etiqueta && <text x={xs} y={sy(v) - 6} className="nota-grafico" fill={color} textAnchor={lado === 'derecha' ? 'end' : 'start'}>{etiqueta}</text>}
    </g>
  )
}

export function Vertical({ x: v, etiqueta, color = COLOR.acento }) {
  const { sx, sy, y } = useContext(Escala)
  return (
    <g>
      <line x1={sx(v)} x2={sx(v)} y1={sy(y[0])} y2={sy(y[1])} stroke={color} strokeWidth={1.6} strokeDasharray="3 4" />
      {etiqueta && <text x={sx(v) + 6} y={sy(y[1]) + 12} className="nota-grafico" fill={color}>{etiqueta}</text>}
    </g>
  )
}

export function Diagonal() {
  const { sx, sy, x, y } = useContext(Escala)
  return <line x1={sx(x[0])} y1={sy(y[0])} x2={sx(x[1])} y2={sy(y[1])} stroke={COLOR.azar} strokeWidth={1.4} strokeDasharray="7 6" />
}

export function Puntos({ datos, color, r = 5 }) {
  const { sx, sy } = useContext(Escala)
  return datos.map(([a, b], i) => (
    <motion.circle key={i} cx={sx(a)} cy={sy(b)} r={r} fill={color} stroke="#151a21" strokeWidth={1.5}
      initial={{ opacity: 0, scale: 0.6 }} whileInView={{ opacity: 1, scale: 1 }} viewport={{ once: true }}
      transition={{ delay: 0.2 + i * 0.05, duration: 0.35 }} />
  ))
}

// Barras verticales: datos = [{x, y, color}] con x numérico (posición) y ancho en unidades de x.
export function Barras({ datos, ancho = 0.6 }) {
  const { sx, sy, y } = useContext(Escala)
  const reducir = useReducedMotion()
  return datos.map((b, i) => {
    const x0 = sx(b.x - ancho / 2), x1 = sx(b.x + ancho / 2)
    return (
      <motion.rect key={i} x={x0} width={x1 - x0} y={sy(b.y)} height={sy(y[0]) - sy(b.y)} fill={b.color}
        style={{ transformOrigin: `0px ${sy(y[0])}px`, transformBox: 'view-box' }}
        initial={{ scaleY: reducir ? 1 : 0 }} whileInView={{ scaleY: 1 }} viewport={{ once: true }}
        transition={{ duration: 0.8, delay: Math.min(i, 12) * 0.05, ease: [0.16, 1, 0.3, 1] }} />
    )
  })
}

export function Texto({ x: a, y: b, children, ancla = 'middle', clase = 'valor-grafico', dy = 0 }) {
  const { sx, sy } = useContext(Escala)
  return <text x={sx(a)} y={sy(b) + dy} textAnchor={ancla} className={clase}>{children}</text>
}

export function Leyenda({ elementos }) {
  return (
    <ul className="leyenda-grafico">
      {elementos.map((e) => (
        <li key={e.texto}><span className={`muestra-linea ${e.tipo ?? 'solida'}`} style={{ '--c': e.color }} />{e.texto}</li>
      ))}
    </ul>
  )
}

// Tarjeta de figura: título, gráfica y explicación debajo, como en el notebook.
export function Figura({ titulo, leyenda, children, explicacion, className = '' }) {
  return (
    <figure className={`figura-web ${className}`}>
      <figcaption className="figura-titulo">{titulo}</figcaption>
      {leyenda && <Leyenda elementos={leyenda} />}
      <div className="figura-cuerpo">{children}</div>
      {explicacion && <div className="figura-explicacion">{explicacion}</div>}
    </figure>
  )
}
