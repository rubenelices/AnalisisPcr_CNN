import { useEffect, useRef, useState } from 'react'
import { AnimatePresence, motion, useReducedMotion, useScroll, useTransform } from 'framer-motion'
import { Boton } from './Boton.jsx'
import { Logo } from './Logo.jsx'
import { Flujo } from './Flujo.jsx'
import { Historial } from './Historial.jsx'
import { IconoEjemplo, IconoSubir } from './Iconos.jsx'
import { Iman, Palabras, SALIDA } from './Movimiento.jsx'
import { EJEMPLOS } from '../ejemplo.js'
import { fmt, fmtProb } from '../datos.js'
import { archivosDeArrastre } from '../lote.js'

export const RUTAS = [
  { id: '', nombre: 'Inicio' },
  { id: 'prediccion', nombre: 'Predicción' },
  { id: 'modelo', nombre: 'Modelo' },
  { id: 'resultados', nombre: 'Resultados' },
  { id: 'pruebas', nombre: 'Pruebas' },
]

function MenuEjemplos({ actual, onElegir, grande }) {
  const [abierto, setAbierto] = useState(false)
  const caja = useRef(null)
  useEffect(() => {
    if (!abierto) return
    const fuera = (e) => { if (!caja.current?.contains(e.target)) setAbierto(false) }
    const escape = (e) => { if (e.key === 'Escape') setAbierto(false) }
    document.addEventListener('pointerdown', fuera)
    document.addEventListener('keydown', escape)
    return () => { document.removeEventListener('pointerdown', fuera); document.removeEventListener('keydown', escape) }
  }, [abierto])
  return (
    <div className="menu" ref={caja}>
      <Boton variante="secundario" className={grande ? 'boton-grande' : ''} icono={<IconoEjemplo />}
        aria-haspopup="menu" aria-expanded={abierto} onClick={() => setAbierto((v) => !v)}>
        <span className="texto-largo">Elegir ejemplo</span><span className="texto-corto">Ejemplo</span>
        <span className={abierto ? 'menu-flecha abierta' : 'menu-flecha'} aria-hidden="true" />
      </Boton>
      <AnimatePresence>
        {abierto && (
          <motion.ul role="menu" className="menu-lista"
            initial={{ opacity: 0, y: -6, scale: 0.98 }} animate={{ opacity: 1, y: 0, scale: 1 }} exit={{ opacity: 0, y: -4 }}
            transition={{ duration: 0.16, ease: SALIDA }}>
            {EJEMPLOS.map((e) => (
              <li key={e.id} role="none">
                <button role="menuitem" className={e.id === actual ? 'menu-item actual' : 'menu-item'}
                  onClick={() => { setAbierto(false); onElegir(e.id) }}>
                  <img src={e.fases[1].src} alt="" width="40" height="40" />
                  <span>
                    <span className="menu-item-titulo">{e.paciente}, {e.posicion}</span>
                    <span className="menu-item-texto">Cohorte {e.cohorte}</span>
                  </span>
                </button>
              </li>
            ))}
          </motion.ul>
        )}
      </AnimatePresence>
    </div>
  )
}

function UltimoResultado({ resultado, ir }) {
  return (
    <AnimatePresence mode="popLayout">
      {resultado.estado !== 'error' && (
        <motion.button key={resultado.clave} className="mando-resultado" onClick={() => ir('prediccion')}
          initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }} transition={{ duration: 0.25 }}>
          <span className="mando-resultado-etiqueta">Último resultado</span>
          <span className="mando-resultado-valor">
            {resultado.estado === 'cargando'
              ? 'calculando…'
              : <>{fmtProb(resultado.probability_pcr, resultado.threshold)} <span>{resultado.predicted_class === 1 ? 'pCR' : 'sin pCR'}</span></>}
          </span>
        </motion.button>
      )}
    </AnimatePresence>
  )
}

function Secciones({ ruta, ir, compacta }) {
  return (
    <nav className={compacta ? 'secciones compacta' : 'secciones'} aria-label="Secciones">
      {RUTAS.map((r) => (
        <a key={r.nombre} href={`#/${r.id}`} className={r.id === ruta ? 'seccion activa' : 'seccion'}
          aria-current={r.id === ruta ? 'page' : undefined}
          onClick={(e) => { e.preventDefault(); ir(r.id) }}>
          {r.id === ruta && (
            <motion.span layoutId={compacta ? 'seccion-compacta' : 'seccion-indicador'} className="seccion-indicador"
              transition={{ type: 'spring', duration: 0.45, bounce: 0.12 }} />
          )}
          <span className="seccion-texto">{r.nombre}</span>
        </a>
      ))}
    </nav>
  )
}

// Estela rosa que sigue al puntero y se desvanece en unos 200 ms. Se dibuja en un canvas con
// requestAnimationFrame solo mientras quedan puntos vivos; con el ratón quieto no consume nada.
function useEstela(caja, activa) {
  const lienzo = useRef(null)
  const puntos = useRef([])
  const bucle = useRef(0)
  const VIDA = 200
  const dibujar = () => {
    const c = lienzo.current
    if (!c) return
    const ctx = c.getContext('2d')
    const ahora = performance.now()
    puntos.current = puntos.current.filter((p) => ahora - p.t < VIDA)
    ctx.clearRect(0, 0, c.width, c.height)
    const pts = puntos.current
    const dpr = window.devicePixelRatio || 1
    ctx.lineCap = 'round'
    ctx.shadowColor = 'rgba(216, 104, 159, 0.9)'
    ctx.shadowBlur = 4 * dpr
    for (let i = 1; i < pts.length; i++) {
      const vida = 1 - (ahora - pts[i].t) / VIDA
      ctx.strokeStyle = `rgba(216, 104, 159, ${0.85 * vida})`
      ctx.lineWidth = (0.5 + 1.1 * vida) * dpr
      ctx.beginPath()
      ctx.moveTo(pts[i - 1].x * dpr, pts[i - 1].y * dpr)
      ctx.lineTo(pts[i].x * dpr, pts[i].y * dpr)
      ctx.stroke()
    }
    bucle.current = pts.length ? requestAnimationFrame(dibujar) : 0
  }
  useEffect(() => () => cancelAnimationFrame(bucle.current), [])
  const mover = (e) => {
    if (!activa || e.pointerType === 'touch' || !lienzo.current) return
    const r = caja.current.getBoundingClientRect()
    const c = lienzo.current, dpr = window.devicePixelRatio || 1
    if (c.width !== Math.round(r.width * dpr) || c.height !== Math.round(r.height * dpr)) {
      c.width = Math.round(r.width * dpr); c.height = Math.round(r.height * dpr)
    }
    puntos.current.push({ x: e.clientX - r.left, y: e.clientY - r.top, t: performance.now() })
    if (!bucle.current) bucle.current = requestAnimationFrame(dibujar)
  }
  return { lienzo, mover }
}

// Cifras del conjunto de datos tal como las da el enunciado (GUIA.md y metadata/statistics.json).
const DATOS_ENUNCIADO = [
  ['1.273', 'pacientes'], ['12.703', 'cortes'], ['3', 'cohortes'], ['3', 'fases DCE'], ['256 × 256', 'px'], ['70,6 %', 'sin pCR'],
]
const TEXTO_DESLIZA = { '': 'el trabajo', prediccion: 'la predicción', modelo: 'el modelo', resultados: 'los resultados', pruebas: 'las pruebas' }

// La pantalla de una estación de radiología, sin imagen: anotaciones en las esquinas (cada una lleva a
// su sección), reglas en los bordes, la escala de color del realce y una estela rosa que sigue al
// puntero. Es funcional: se pueden soltar aquí imágenes o carpetas y se procesan en Pruebas.
function Pantalla({ modelo, resultado, ocupado, ir, onArchivos, nav, ruta, children }) {
  const reducir = useReducedMotion()
  const caja = useRef(null)
  const [arrastrando, setArrastrando] = useState(false)
  const capas = useRef(0)
  const estela = useEstela(caja, !reducir)
  const conFicheros = (e) => [...(e.dataTransfer?.types ?? [])].includes('Files')
  const entrar = (e) => { if (!conFicheros(e)) return; e.preventDefault(); capas.current += 1; setArrastrando(true) }
  const dejar = () => { capas.current = Math.max(0, capas.current - 1); if (!capas.current) setArrastrando(false) }
  const soltar = async (e) => {
    if (!conFicheros(e)) return
    e.preventDefault(); capas.current = 0; setArrastrando(false)
    onArchivos(await archivosDeArrastre(e.dataTransfer))
  }
  const listo = modelo.estado === 'listo'
  const r = resultado.estado === 'listo' ? resultado : null
  const aparece = (i) => ({ initial: { opacity: 0 }, animate: { opacity: 1 }, transition: { delay: 0.5 + i * 0.12, duration: 0.6 } })
  return (
    <div className={arrastrando ? 'pantalla arrastrando' : 'pantalla'} ref={caja}
      onPointerMove={estela.mover}
      onDragEnter={entrar} onDragOver={(e) => conFicheros(e) && e.preventDefault()} onDragLeave={dejar} onDrop={soltar}>
      <span className="pantalla-regla arriba" aria-hidden="true" />
      <span className="pantalla-regla izquierda" aria-hidden="true" />
      <div className="pantalla-escala" aria-hidden="true">
        <span>alto</span><span className="pantalla-escala-barra" /><span>bajo</span>
      </div>

      <div className="pantalla-fila superior">
        <div className="pantalla-marca">
          <a className="marca grande" href="#/" onClick={(e) => { e.preventDefault(); ir('') }} aria-label="Inicio">
            <Logo tamano={44} />
          </a>
        <motion.button className="esquina" {...aparece(0)} onClick={() => ir('')}>
          <b>Estudio</b> BreastDCEDL · 1.097 + 176 pacientes<br />CNN 2D desde cero · 298.161 parámetros
        </motion.button>
        </div>
        <motion.button className="esquina derecha" {...aparece(1)} onClick={() => ir('prediccion')}>
          <b>Serie</b> DCE-MRI · PRE / EARLY / LATE<br />256 × 256 · valores / 255
        </motion.button>
      </div>

      <div className="pantalla-centro">
        {children}
        <div className="pantalla-nav">
          {nav}
          <p className="pantalla-aviso">Uso educativo, sin validez clínica</p>
        </div>
      </div>

      <div className="pantalla-fila inferior">
        <motion.button className="esquina" {...aparece(2)} onClick={() => ir('modelo')} aria-live="polite">
          <span className={!listo ? 'luz luz-error' : ocupado ? 'luz luz-ocupada' : 'luz'} aria-hidden="true" />
          <b>Modelo</b> {listo ? `E57 · ${modelo.device.toUpperCase()} · umbral ${fmt(modelo.threshold, 3)}` : modelo.estado === 'cargando' ? 'cargando…' : 'no disponible'}
          <br />{listo ? `pesos ${modelo.sha256.slice(0, 12)} · arrastra imágenes a esta pantalla` : 'arranca app.py con el modelo'}
        </motion.button>
        <button className="desliza" onClick={() => {
          const hoja = document.querySelector('.hoja')
          if (hoja) window.scrollTo({ top: hoja.getBoundingClientRect().top + window.scrollY - 64, behavior: reducir ? 'auto' : 'smooth' })
        }}>
          Desliza para ver {TEXTO_DESLIZA[ruta] ?? 'la sección'}
          <span className="desliza-flecha" aria-hidden="true" />
        </button>
        <motion.button className="esquina derecha" {...aparece(3)} onClick={() => ir('prediccion')}>
          <b>Último resultado</b>{' '}
          {r ? <>p(pCR) {fmtProb(r.probability_pcr, r.threshold)} · <em>{r.predicted_class === 1 ? 'pCR' : 'sin pCR'}</em></> : 'calculando…'}
          <br />{r ? `${r.clave} · ${Math.max(1, Math.round(r.latency_ms))} ms` : '—'}
        </motion.button>
      </div>

      <canvas className="estela" ref={estela.lienzo} aria-hidden="true" />

      <AnimatePresence>
        {arrastrando && (
          <motion.div className="pantalla-soltar" initial={{ opacity: 0, scale: 0.98 }} animate={{ opacity: 1, scale: 1 }} exit={{ opacity: 0 }}
            transition={{ duration: 0.18 }}>
            <IconoSubir />
            <p>Suelta las imágenes para predecir</p>
            <span>Tres PNG por corte (_PRE, _EARLY, _LATE) o una carpeta. Los resultados aparecen en Pruebas.</span>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}

// Encabezado grande e igual en todas las secciones; al bajar, una barra fija compacta con lo mismo.
export function Cabecera({ ruta, ir, entrada, onSubir, onEjemplo, resultado, modelo, ocupado, historial, onAbrirHistorial }) {
  const input = useRef(null)
  const cabecera = useRef(null)
  const [fuera, setFuera] = useState(false)
  const reducir = useReducedMotion()
  // Efecto telón: el encabezado queda fijo detrás y la página sube por encima. Se ancla con
  // top = alto de ventana − alto del encabezado (o 0), para que nunca quede una parte sin ver.
  useEffect(() => {
    const el = cabecera.current
    const ajustar = () => {
      el.style.top = `${Math.min(0, window.innerHeight - el.offsetHeight)}px`
      setFuera(window.scrollY > el.offsetHeight - 120)
    }
    const ro = new ResizeObserver(ajustar)
    ro.observe(el)
    window.addEventListener('scroll', ajustar, { passive: true })
    window.addEventListener('resize', ajustar)
    ajustar()
    return () => { ro.disconnect(); window.removeEventListener('scroll', ajustar); window.removeEventListener('resize', ajustar) }
  }, [])
  const { scrollYProgress } = useScroll({ target: cabecera, offset: ['start start', 'end start'] })
  const atenuar = useTransform(scrollYProgress, [0, 0.9], [1, reducir ? 1 : 0.25])
  const alejar = useTransform(scrollYProgress, [0, 0.9], [1, reducir ? 1 : 0.94])
  const subir = () => input.current?.click()
  const actual = entrada.tipo === 'ejemplo' ? entrada.id : null

  return (
    <>
      <input ref={input} type="file" accept=".png,image/png" multiple hidden
        onChange={(e) => { onSubir(e.target.files); e.target.value = '' }} />

      <header className="cabecera" ref={cabecera}>
        <motion.div className="cabecera-escena" style={{ opacity: atenuar, scale: alejar }}>
        <Pantalla modelo={modelo} resultado={resultado} ocupado={ocupado} ir={ir} onArchivos={onSubir} ruta={ruta}
          nav={<Secciones ruta={ruta} ir={ir} />}>
          <div className="ancla-titulo">
          <div className="titulo-bloque">
            <p className="cabecera-antetitulo">Aprendizaje automático · CNN entrenada desde cero</p>
            <p className="cabecera-nombre"><Palabras texto="Respuesta al tratamiento neoadyuvante" clave="titulo" /></p>
            <motion.p className="cabecera-sub" initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8, delay: 0.4, ease: SALIDA }}>
              Probabilidad de respuesta patológica completa a partir de tres fases de un corte de resonancia de mama.
            </motion.p>
            <motion.ul className="cifras-enunciado" aria-label="El conjunto de datos, según el enunciado"
              initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 0.8, delay: 0.6 }}>
              {DATOS_ENUNCIADO.map(([v, t]) => <li key={t}><b>{v}</b> {t}</li>)}
            </motion.ul>
          </div>
          <div className="acciones-bloque">
          <div className="mando" role="toolbar" aria-label="Centro de mando">
            <Iman>
              <Boton className="boton-grande" icono={<IconoSubir />} onClick={subir}>
                <span className="texto-largo">Subir imágenes</span><span className="texto-corto">Subir</span>
              </Boton>
            </Iman>
            <Historial historial={historial} onAbrir={onAbrirHistorial} />
            <Iman><MenuEjemplos actual={actual} onElegir={onEjemplo} grande /></Iman>
          </div>
          </div>
          </div>
          <Flujo resultado={resultado} onAbrir={() => ir('prediccion')} />
        </Pantalla>
        </motion.div>
      </header>

      <AnimatePresence>
        {fuera && (
          <motion.div className="barra-fija" role="toolbar" aria-label="Mando compacto"
            initial={{ y: '-100%' }} animate={{ y: 0 }} exit={{ y: '-100%' }}
            transition={{ duration: 0.3, ease: SALIDA }}>
            <a className="marca" href="#/" onClick={(e) => { e.preventDefault(); ir('') }} aria-label="Inicio">
              <Logo tamano={30} />
            </a>
            <Secciones ruta={ruta} ir={ir} compacta />
            <div className="barra-acciones">
              <Boton icono={<IconoSubir />} onClick={subir}>
                <span className="texto-largo">Subir imágenes</span><span className="texto-corto">Subir</span>
              </Boton>
              <MenuEjemplos actual={actual} onElegir={onEjemplo} />
              <UltimoResultado resultado={resultado} ir={ir} />
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  )
}
