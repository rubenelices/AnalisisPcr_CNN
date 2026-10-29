import { useEffect, useRef, useState } from 'react'
import { useReducedMotion } from 'framer-motion'
import { fmt, fmtProb } from '../datos.js'

// El recorrido real de un corte por la red, en una línea. Con cada predicción nueva, una señal
// recorre las etapas y enciende al final la probabilidad; mientras se calcula, circula en bucle.
const ETAPAS = [
  { t: 'Corte', d: 'PRE · EARLY · LATE' },
  { t: '3 canales', d: 'restas fijas' },
  { t: '4 bloques conv', d: '16 → 128 canales' },
  { t: 'GAP', d: '128 valores' },
  { t: 'Cabeza', d: '128 → 32 → 1' },
]
const PASO_MS = 170

export function Flujo({ resultado, onAbrir }) {
  const reducir = useReducedMotion()
  const n = ETAPAS.length + 1
  const [paso, setPaso] = useState(-1)
  const ultima = useRef(null)
  useEffect(() => {
    if (resultado.estado === 'cargando') {
      if (reducir) return setPaso(-1)
      let i = 0
      const id = setInterval(() => { setPaso(i % n); i += 1 }, PASO_MS)
      return () => clearInterval(id)
    }
    if (resultado.estado !== 'listo') return setPaso(-1)
    if (ultima.current === resultado.clave + resultado.probability_pcr) return
    ultima.current = resultado.clave + resultado.probability_pcr
    if (reducir) return setPaso(n - 1)
    let i = 0
    setPaso(0)
    const id = setInterval(() => { i += 1; setPaso(Math.min(i, n - 1)); if (i >= n - 1) clearInterval(id) }, PASO_MS)
    return () => clearInterval(id)
  }, [resultado, reducir, n])

  const listo = resultado.estado === 'listo'
  const estado = (i) => (i === paso ? 'encendida' : paso >= 0 && i < paso && resultado.estado === 'listo' ? 'pasada' : '')
  return (
    <ol className="flujo" aria-label="Recorrido de un corte por la red">
      {ETAPAS.map((e, i) => (
        <li key={e.t} className={`flujo-etapa ${estado(i)}`}>
          <strong>{e.t}</strong><span>{e.d}</span>
        </li>
      ))}
      <li className={`flujo-etapa salida ${estado(n - 1)} ${listo && paso === n - 1 ? 'final' : ''}`}>
        <button onClick={onAbrir} disabled={!listo} aria-label="Ver la última predicción en detalle">
          <strong>p(pCR) {listo ? fmtProb(resultado.probability_pcr, resultado.threshold) : '…'}</strong>
          <span>{listo ? `${resultado.predicted_class === 1 ? 'pCR · supera' : 'sin pCR · bajo'} el umbral ${fmt(resultado.threshold, 3)}` : 'calculando'}</span>
        </button>
      </li>
    </ol>
  )
}
