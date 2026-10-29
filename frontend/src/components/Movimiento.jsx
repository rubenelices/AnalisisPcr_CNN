import { Fragment, useEffect, useRef, useState } from 'react'
import { animate, motion, useInView, useReducedMotion, useSpring } from 'framer-motion'

// Piezas de movimiento de toda la web, inspiradas en React Bits (SplitText, CountUp, Magnet)
// pero escritas con framer-motion. Solo mueven transform y opacity, y con
// prefers-reduced-motion dejan el estado final sin desplazamiento.

export const SALIDA = [0.22, 1, 0.36, 1]   // entradas y respuestas
export const EXPO = [0.16, 1, 0.3, 1]      // revelados largos y titulares

// Aparece una vez al entrar en pantalla, subiendo 24 px. `orden` escalona hermanos (75 ms).
export function Revelar({ as = 'div', orden = 0, children, className, desde = 24, ...props }) {
  const ref = useRef(null)
  const visto = useInView(ref, { once: true, margin: '0px 0px -10% 0px' })
  const reducir = useReducedMotion()
  const M = motion[as]
  return (
    <M ref={ref} className={className}
      initial={{ opacity: 0, y: reducir ? 0 : desde }}
      animate={visto ? { opacity: 1, y: 0 } : undefined}
      transition={{ duration: 0.8, delay: orden * 0.075, ease: EXPO }}
      {...props}>
      {children}
    </M>
  )
}

// Titular que sube palabra a palabra desde una máscara. Se repite al cambiar `clave`.
export function Palabras({ texto, clave, retraso = 0 }) {
  const reducir = useReducedMotion()
  const palabras = texto.split(' ')
  return (
    <span className="palabras" aria-label={texto}>
      {palabras.map((p, i) => (
        <Fragment key={`${clave}-${i}`}>
          <span className="palabra" aria-hidden="true">
            <motion.span
              initial={{ y: reducir ? 0 : '105%', opacity: reducir ? 0 : 1 }}
              animate={{ y: 0, opacity: 1 }}
              transition={{ duration: reducir ? 0.3 : 0.9, delay: retraso + i * 0.07, ease: EXPO }}>
              {p}
            </motion.span>
          </span>
          {i < palabras.length - 1 && ' '}
        </Fragment>
      ))}
    </span>
  )
}

// Cifra que cuenta desde cero la primera vez que se ve, con formato español.
export function Contador({ valor, decimales = 0 }) {
  const ref = useRef(null)
  const visto = useInView(ref, { once: true, margin: '0px 0px -15% 0px' })
  const reducir = useReducedMotion()
  const formato = (v) => v.toLocaleString('es-ES', { minimumFractionDigits: decimales, maximumFractionDigits: decimales, useGrouping: 'always' })
  const [texto, setTexto] = useState(formato(valor))
  useEffect(() => {
    if (!visto || reducir) return
    const control = animate(0, valor, { duration: 1.6, ease: EXPO, onUpdate: (v) => setTexto(formato(v)) })
    return () => control.stop()
  }, [visto, reducir, valor]) // eslint-disable-line react-hooks/exhaustive-deps
  return <span ref={ref} className="contador">{texto}</span>
}

// El contenido se deja atraer un poco por el puntero (solo con ratón).
export function Iman({ children, fuerza = 0.25 }) {
  const ref = useRef(null)
  const x = useSpring(0, { stiffness: 220, damping: 18, mass: 0.6 })
  const y = useSpring(0, { stiffness: 220, damping: 18, mass: 0.6 })
  const fino = typeof window !== 'undefined' && window.matchMedia?.('(hover: hover) and (pointer: fine)').matches
  const reducir = useReducedMotion()
  if (!fino || reducir) return <span className="iman">{children}</span>
  const mover = (e) => {
    const r = ref.current.getBoundingClientRect()
    x.set((e.clientX - r.left - r.width / 2) * fuerza)
    y.set((e.clientY - r.top - r.height / 2) * fuerza)
  }
  const soltar = () => { x.set(0); y.set(0) }
  return (
    <motion.span ref={ref} className="iman" style={{ x, y }} onPointerMove={mover} onPointerLeave={soltar}>
      {children}
    </motion.span>
  )
}
