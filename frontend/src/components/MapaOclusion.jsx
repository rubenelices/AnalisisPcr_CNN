import { useEffect, useRef, useState } from 'react'
import { mapaOclusion } from '../api.js'

const fmt = (v, d = 1) => v.toFixed(d).replace('.', ',')
// Tapar y que baje la probabilidad: la zona empujaba hacia pCR (acento). Que suba: hacia sin pCR (azul).
const HACIA_PCR = [216, 104, 159]
const HACIA_SIN = [86, 152, 214]
const TAMANOS = [32, 16, 8]
const FASES = [['', 'Las tres'], ['PRE', 'PRE'], ['EARLY', 'EARLY'], ['LATE', 'LATE']]

// Dibuja la fase de fondo y encima un cuadro por zona tapada, con opacidad relativa al mayor cambio del corte.
function pintar(lienzo, fondo, mapa, verCuadros, marcado) {
  const ctx = lienzo.getContext('2d')
  const lado = lienzo.width, c = lado / mapa.size
  ctx.clearRect(0, 0, lado, lado)
  ctx.drawImage(fondo, 0, 0, lado, lado)
  if (verCuadros) {
    const maximo = Math.max(...mapa.deltas.flat().map(Math.abs))
    if (maximo > 0) {
      mapa.deltas.forEach((fila, i) => fila.forEach((d, j) => {
        const [r, g, b] = d < 0 ? HACIA_PCR : HACIA_SIN
        ctx.fillStyle = `rgba(${r}, ${g}, ${b}, ${(0.78 * Math.abs(d)) / maximo})`
        ctx.fillRect(j * c, i * c, c, c)
      }))
    }
  }
  if (marcado) {
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.9)'
    ctx.lineWidth = 2
    ctx.strokeRect(marcado.j * c + 1, marcado.i * c + 1, c - 2, c - 2)
  }
}

function Selector({ nombre, opciones, valor, onCambio, desactivado }) {
  return (
    <div className="lab-segmentos" role="radiogroup" aria-label={nombre}>
      {opciones.map(([v, texto]) => (
        <button key={v} type="button" role="radio" aria-checked={valor === v} className={valor === v ? 'activo' : ''}
          disabled={desactivado} onClick={() => onCambio(v)}>{texto}</button>
      ))}
    </div>
  )
}

// clave identifica el corte; obtenerArchivos devuelve { PRE, EARLY, LATE } como File; fondos, la URL de cada fase.
export function MapaOclusion({ clave, obtenerArchivos, fondos }) {
  const [patch, setPatch] = useState(16)
  const [fase, setFase] = useState('')
  const [mapa, setMapa] = useState({ estado: 'cargando' })
  const [verCuadros, setVerCuadros] = useState(true)
  const [marcado, setMarcado] = useState(null)
  const lienzo = useRef(null)
  const imagen = useRef(null)

  useEffect(() => {
    let vivo = true
    setMapa((m) => ({ ...m, estado: 'cargando' }))
    setMarcado(null)
    obtenerArchivos()
      .then((archivos) => mapaOclusion(archivos, { patch, fase }))
      .then((m) => vivo && setMapa({ estado: 'listo', ...m }))
      .catch((e) => vivo && setMapa({ estado: 'error', error: e.message }))
    return () => { vivo = false }
  }, [clave, obtenerArchivos, patch, fase])

  const fondo = fondos[fase || 'EARLY']
  useEffect(() => {
    if (!mapa.deltas || !lienzo.current) return
    const dibujar = () => lienzo.current && pintar(lienzo.current, imagen.current, mapa, verCuadros, marcado)
    if (imagen.current?.dataset.src === fondo && imagen.current.complete) return dibujar()
    const img = new Image()
    img.dataset.src = fondo
    img.onload = () => { imagen.current = img; dibujar() }
    img.src = fondo
  }, [mapa, fondo, verCuadros, marcado])

  const mover = (e) => {
    if (!mapa.deltas) return
    const r = e.currentTarget.getBoundingClientRect()
    const j = Math.floor(((e.clientX - r.left) / r.width) * mapa.size)
    const i = Math.floor(((e.clientY - r.top) / r.height) * mapa.size)
    if (i >= 0 && j >= 0 && i < mapa.size && j < mapa.size && (marcado?.i !== i || marcado?.j !== j)) {
      setMarcado({ i, j, d: mapa.deltas[i][j] })
    }
  }

  const cambios = mapa.deltas ? mapa.deltas.flat() : []
  const bajada = Math.min(0, ...cambios), subida = Math.max(0, ...cambios)
  const cargando = mapa.estado === 'cargando'
  const lado = mapa.patch ?? patch

  return (
    <div className="lab-oclusion">
      <div className={cargando ? 'oclusion-lienzo recalculando' : 'oclusion-lienzo'} aria-busy={cargando}>
        {mapa.deltas
          ? <canvas ref={lienzo} width="512" height="512" onMouseMove={mover} onMouseLeave={() => setMarcado(null)} role="img"
              aria-label={`Mapa de oclusión: al tapar, la probabilidad baja hasta ${fmt(-bajada * 100)} puntos y sube hasta ${fmt(subida * 100)} puntos`} />
          : <div className="oclusion-hueco">{mapa.estado === 'error' ? 'No se ha podido calcular el mapa.' : 'Tapando cada cuadro…'}</div>}
        {cargando && mapa.deltas && <span className="oclusion-calculando">Recalculando…</span>}
      </div>
      <div className="lab-oclusion-panel">
        <div className="lab-control">
          <span className="lab-control-nombre">Tamaño del cuadro</span>
          <Selector nombre="Tamaño del cuadro" opciones={TAMANOS.map((t) => [t, `${t} px`])} valor={patch} onCambio={setPatch} />
        </div>
        <div className="lab-control">
          <span className="lab-control-nombre">Fase que se tapa</span>
          <Selector nombre="Fase que se tapa" opciones={FASES} valor={fase} onCambio={setFase} />
        </div>
        <div className="leyenda" aria-hidden="true">
          <span className="leyenda-barra oclusion-barra" />
          <span className="leyenda-extremos"><span>tapar baja P(pCR)</span><span>tapar sube P(pCR)</span></span>
        </div>
        {mapa.deltas && (
          <dl className="oclusion-cifras">
            <div><dt>Mayor bajada</dt><dd>{fmt(-bajada * 100)} puntos</dd></div>
            <div><dt>Mayor subida</dt><dd>{fmt(subida * 100)} puntos</dd></div>
            <div><dt>Cuadro señalado</dt><dd>{marcado ? `${marcado.d < 0 ? '−' : '+'}${fmt(Math.abs(marcado.d) * 100, 2)} puntos` : '—'}</dd></div>
          </dl>
        )}
        {mapa.estado === 'error' && <p className="lab-nota">{mapa.error}</p>}
        <label className="oclusion-interruptor">
          <input type="checkbox" checked={verCuadros} onChange={(e) => setVerCuadros(e.target.checked)} />
          Mostrar los cuadros
        </label>
        <p className="lab-nota">
          Se tapa con ceros un cuadro de {lado} × {lado} píxeles{fase ? ` solo en ${fase}` : ' en las tres fases'}, uno cada vez
          ({(256 / lado) ** 2} predicciones), y se mide cuánto cambia la probabilidad. Pasa el ratón por encima para ver cada cuadro.
          {fase && ' Tapar una sola fase crea una imagen que no existe en una resonancia real: sirve para comparar fases, no como situación realista.'}
        </p>
        <p className="lab-nota">
          La intensidad es relativa al mayor cambio de este corte. Indica qué zonas mueven la salida de este modelo; no
          localiza el tumor ni explica por qué una paciente responde.
        </p>
      </div>
    </div>
  )
}
