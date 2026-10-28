import { useEffect, useMemo } from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import { fmtProb } from '../datos.js'
import { faltan, faseDe } from '../lote.js'

// Zona de carga de la portada: las tres fases del último corte seleccionado, en tarjetas grandes con
// su probabilidad debajo. Sin ninguno todavía, tres tarjetas vacías con un resplandor suave indican qué
// hay que subir; la franja de debajo se puede pulsar para elegir imágenes (o se sueltan en la pantalla).
const FASES = ['PRE', 'EARLY', 'LATE']

export function Historial({ historial, onAbrir, onSubir, espera }) {
  const h = historial[0]
  // Con un corte a medias, las tarjetas muestran las fases que ya han llegado y dejan vacías las que faltan.
  const archivos = espera && Array.isArray(espera.archivos) ? espera.archivos : null
  const vistas = useMemo(() => (archivos ? Object.fromEntries(archivos.map((a) => [faseDe(a.name), URL.createObjectURL(a)])) : {}), [archivos])
  useEffect(() => () => Object.values(vistas).forEach((u) => URL.revokeObjectURL(u)), [vistas])
  const fuente = archivos ? (f) => vistas[f] : (f, i) => h?.fases?.[i]
  const clave = archivos ? espera.clave : h?.clave
  return (
    <div className="historial" aria-live="polite">
      <button className="historial-fases" onClick={() => (archivos ? onSubir() : h && onAbrir(h))} disabled={!h && !archivos}
        aria-label={archivos ? `Faltan fases de ${clave}: pulsa para añadirlas` : h ? `Ver en detalle ${h.clave}` : 'Todavía no se ha seleccionado ningún corte'}>
        {FASES.map((f, i) => (
          <figure key={f} className={`fase-tarjeta fase-${f.toLowerCase()}`}>
            <AnimatePresence mode="popLayout" initial={false}>
              {fuente(f, i)
                ? <motion.img key={`${clave}-${f}`} src={fuente(f, i)} alt={`${f} de ${clave}`}
                    initial={{ opacity: 0, scale: 0.96 }} animate={{ opacity: 1, scale: 1 }} exit={{ opacity: 0 }}
                    transition={{ duration: 0.3, delay: archivos ? 0 : i * 0.08 }} />
                : <span className="fase-brillo" aria-hidden="true" />}
            </AnimatePresence>
            <figcaption>{f}</figcaption>
          </figure>
        ))}
      </button>
      <button className="historial-franja" onClick={onSubir}>
        {archivos ? <>{clave} · falta {faltan(archivos).join(' y ')}: añádela para calcular</>
          : h ? <>{h.clave} · <b>{fmtProb(h.p, h.umbral)}</b> · <span className={h.clase === 1 ? 'si' : ''}>{h.clase === 1 ? 'pCR' : 'sin pCR'}</span></>
          : 'Arrastra aquí un corte: aparecerán sus tres fases'}
      </button>
    </div>
  )
}
