import { AnimatePresence, motion } from 'framer-motion'
import { fmtProb } from '../datos.js'

// Entre los dos botones del visor: las tres fases del último corte seleccionado, en miniatura, con su
// probabilidad. Sin ninguno todavía, tres casillas vacías indican qué hay que subir.
const FASES = ['PRE', 'EARLY', 'LATE']

export function Historial({ historial, onAbrir }) {
  const h = historial[0]
  return (
    <div className="historial" aria-live="polite">
      <p className="historial-titulo">{h ? 'Último corte' : 'Sin imágenes todavía'}</p>
      <button className="historial-fases" onClick={() => h && onAbrir(h)} disabled={!h}
        aria-label={h ? `Ver en detalle ${h.clave}` : 'Todavía no se ha seleccionado ningún corte'}>
        {FASES.map((f, i) => (
          <figure key={f}>
            <AnimatePresence mode="popLayout" initial={false}>
              {h?.fases?.[i]
                ? <motion.img key={`${h.clave}-${f}`} src={h.fases[i]} alt={`${f} de ${h.clave}`} width="72" height="72"
                    initial={{ opacity: 0, scale: 0.9 }} animate={{ opacity: 1, scale: 1 }} exit={{ opacity: 0 }}
                    transition={{ duration: 0.3, delay: i * 0.08 }} />
                : <span className="historial-hueco" />}
            </AnimatePresence>
            <figcaption>{f}</figcaption>
          </figure>
        ))}
      </button>
      <p className="historial-pie">
        {h ? <>{h.clave} · <b>{fmtProb(h.p, h.umbral)}</b> · <span className={h.clase === 1 ? 'si' : ''}>{h.clase === 1 ? 'pCR' : 'sin pCR'}</span></>
          : 'Sube o elige un corte: aquí aparecerán sus tres fases'}
      </p>
    </div>
  )
}
