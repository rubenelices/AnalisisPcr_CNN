import { useEffect, useRef, useState } from 'react'
import { motion, useReducedMotion } from 'framer-motion'
import { mapaOclusion } from '../api.js'

const fmt = (v, d = 1) => v.toFixed(d).replace('.', ',')
// Tapar y que baje la probabilidad: la zona empujaba hacia pCR (acento). Que suba: hacia sin pCR (azul).
const HACIA_PCR = [216, 104, 159]
const HACIA_SIN = [86, 152, 214]

// Dibuja la fase EARLY y encima un cuadro por zona tapada, con opacidad relativa al mayor cambio del corte.
function pintar(lienzo, fondo, mapa, verCuadros) {
  const ctx = lienzo.getContext('2d')
  const lado = lienzo.width, k = lado / (mapa.size * mapa.patch)
  ctx.clearRect(0, 0, lado, lado)
  ctx.drawImage(fondo, 0, 0, lado, lado)
  if (!verCuadros) return
  const maximo = Math.max(...mapa.deltas.flat().map(Math.abs))
  if (maximo === 0) return
  const c = mapa.patch * k
  mapa.deltas.forEach((fila, i) => fila.forEach((d, j) => {
    const [r, g, b] = d < 0 ? HACIA_PCR : HACIA_SIN
    ctx.fillStyle = `rgba(${r}, ${g}, ${b}, ${(0.78 * Math.abs(d)) / maximo})`
    ctx.fillRect(j * c, i * c, c, c)
  }))
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.07)'
  ctx.lineWidth = 1
  for (let n = 1; n < mapa.size; n++) {
    ctx.beginPath(); ctx.moveTo(n * c + 0.5, 0); ctx.lineTo(n * c + 0.5, lado); ctx.stroke()
    ctx.beginPath(); ctx.moveTo(0, n * c + 0.5); ctx.lineTo(lado, n * c + 0.5); ctx.stroke()
  }
}

// clave identifica el corte; obtenerArchivos devuelve { PRE, EARLY, LATE } como File; fondo es la imagen EARLY.
export function MapaOclusion({ clave, obtenerArchivos, fondo }) {
  const [mapa, setMapa] = useState({ estado: 'cargando' })
  const [verCuadros, setVerCuadros] = useState(true)
  const lienzo = useRef(null)
  const reducir = useReducedMotion()

  useEffect(() => {
    let vivo = true
    setMapa({ estado: 'cargando' })
    obtenerArchivos()
      .then((archivos) => mapaOclusion(archivos))
      .then((m) => vivo && setMapa({ estado: 'listo', ...m }))
      .catch((e) => vivo && setMapa({ estado: 'error', error: e.message }))
    return () => { vivo = false }
  }, [clave, obtenerArchivos])

  useEffect(() => {
    if (mapa.estado !== 'listo' || !lienzo.current) return
    const imagen = new Image()
    imagen.onload = () => lienzo.current && pintar(lienzo.current, imagen, mapa, verCuadros)
    imagen.src = fondo
  }, [mapa, fondo, verCuadros])

  const cambios = mapa.estado === 'listo' ? mapa.deltas.flat() : []
  const bajada = Math.min(0, ...cambios), subida = Math.max(0, ...cambios)

  return (
    <motion.figure className="realce oclusion" initial={{ opacity: 0, y: reducir ? 0 : 10 }} animate={{ opacity: 1, y: 0 }}
      transition={{ delay: reducir ? 0 : 0.5, duration: 0.45, ease: [0.2, 0.8, 0.2, 1] }}>
      <div className="oclusion-lienzo" aria-busy={mapa.estado === 'cargando'}>
        {mapa.estado === 'listo'
          ? <canvas ref={lienzo} width="512" height="512" role="img"
              aria-label={`Mapa de oclusión sobre la fase EARLY: al tapar, la probabilidad baja hasta ${fmt(-bajada * 100)} puntos y sube hasta ${fmt(subida * 100)} puntos`} />
          : <div className="oclusion-hueco">{mapa.estado === 'error' ? 'No se ha podido calcular el mapa.' : 'Tapando cada cuadro…'}</div>}
      </div>
      <figcaption>
        <h2>Mapa de oclusión</h2>
        <p>
          Se tapa el corte por cuadros de 16 × 16 píxeles, en las tres fases a la vez y uno cada vez, y se vuelve a
          calcular la probabilidad. El color indica cuánto cambia respecto al corte completo.
        </p>
        <div className="leyenda" aria-hidden="true">
          <span className="leyenda-barra oclusion-barra" />
          <span className="leyenda-extremos"><span>tapar baja P(pCR)</span><span>tapar sube P(pCR)</span></span>
        </div>
        {mapa.estado === 'listo' && (
          <dl className="oclusion-cifras">
            <div><dt>Mayor bajada</dt><dd>{fmt(-bajada * 100)} puntos</dd></div>
            <div><dt>Mayor subida</dt><dd>{fmt(subida * 100)} puntos</dd></div>
          </dl>
        )}
        {mapa.estado === 'error' && <p className="realce-nota">{mapa.error}</p>}
        <label className="oclusion-interruptor">
          <input type="checkbox" checked={verCuadros} onChange={(e) => setVerCuadros(e.target.checked)} />
          Mostrar los cuadros
        </label>
        <p className="realce-nota">
          La intensidad es relativa al mayor cambio de este corte. Indica qué zonas mueven la salida de este modelo; no
          localiza el tumor ni explica por qué una paciente responde.
        </p>
      </figcaption>
    </motion.figure>
  )
}
