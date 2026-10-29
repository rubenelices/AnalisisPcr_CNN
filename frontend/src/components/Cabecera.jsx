import { useEffect, useRef, useState } from 'react'
import { AnimatePresence, motion, useReducedMotion, useScroll, useTransform } from 'framer-motion'
import { Boton } from './Boton.jsx'
import { Logo } from './Logo.jsx'
import { Historial } from './Historial.jsx'
import { RedFondo } from './RedFondo.jsx'
import { IconoEjemplo, IconoSubir } from './Iconos.jsx'
import { Iman, Palabras, SALIDA } from './Movimiento.jsx'
import { EJEMPLOS } from '../ejemplo.js'
import { fmt, fmtProb } from '../datos.js'
import { archivosDeArrastre } from '../lote.js'

export const RUTAS = [
  { id: '', nombre: 'Inicio' },
  { id: 'laboratorio', nombre: 'Laboratorio' },
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
        <motion.button key={resultado.clave} className="mando-resultado" onClick={() => ir('laboratorio')}
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

function Secciones({ ruta, ir, flotante }) {
  return (
    <nav className={flotante ? 'secciones flotante' : 'secciones'} aria-label="Secciones">
      {RUTAS.map((r) => (
        <a key={r.nombre} href={`#/${r.id}`} className={r.id === ruta ? 'seccion activa' : 'seccion'}
          aria-current={r.id === ruta ? 'page' : undefined}
          onClick={(e) => { e.preventDefault(); ir(r.id) }}>
          {r.id === ruta && (
            <motion.span layoutId="seccion-indicador" className="seccion-indicador"
              transition={{ type: 'spring', duration: 0.45, bounce: 0.12 }} />
          )}
          <span className="seccion-texto">{r.nombre}</span>
        </a>
      ))}
    </nav>
  )
}

// Cada sección conserva el mismo encabezado pero cambia de titular, para que se note que se ha cambiado de sitio.
const TITULARES = {
  '': ['Aprendizaje automático · CNN entrenada desde cero', 'Respuesta al tratamiento neoadyuvante',
    'Probabilidad de respuesta patológica completa a partir de tres fases de un corte de resonancia de mama.'],
  laboratorio: ['Laboratorio · análisis de un corte', 'Cómo responde el modelo',
    'Elige un corte, aplícale las transformaciones del entrenamiento, ajusta el umbral y comprueba qué zonas modifican la probabilidad que da E57.'],
  modelo: ['Modelo · E57', 'Una CNN propia, capa a capa',
    '298.161 parámetros entrenados desde cero: la arquitectura, el entrenamiento y cómo se eligió.'],
  resultados: ['Resultados · evaluación honesta', 'Lo que dicen las cifras',
    'Test público, validación cruzada de 5 folds, normal frente a ponderada y la decisión de entrega.'],
  pruebas: ['Pruebas · para la defensa', 'Carga tus propios cortes',
    'Suelta las tres fases de uno o varios cortes y compara la predicción con la etiqueta real.'],
}
const TEXTO_DESLIZA = { '': 'el trabajo', laboratorio: 'el laboratorio', modelo: 'el modelo', resultados: 'los resultados', pruebas: 'las pruebas' }

// La pantalla de una estación de radiología, sin imagen: anotaciones en las esquinas (cada una lleva a
// su sección), reglas en los bordes y la escala de color del realce. Es funcional: se pueden soltar aquí imágenes o carpetas y se procesan en Pruebas.
function Pantalla({ modelo, resultado, ocupado, ir, onArchivos, ruta, children }) {
  const reducir = useReducedMotion()
  const [arrastrando, setArrastrando] = useState(false)
  const capas = useRef(0)
  const conFicheros = (e) => [...(e.dataTransfer?.types ?? [])].includes('Files')
  const entrar = (e) => { if (!conFicheros(e)) return; e.preventDefault(); capas.current += 1; setArrastrando(true) }
  const dejar = () => { capas.current = Math.max(0, capas.current - 1); if (!capas.current) setArrastrando(false) }
  const soltar = async (e) => {
    if (!conFicheros(e)) return
    e.preventDefault(); capas.current = 0; setArrastrando(false)
    onArchivos(await archivosDeArrastre(e.dataTransfer))
  }
  const listo = modelo.estado === 'listo'
  const aparece = (i) => ({ initial: { opacity: 0 }, animate: { opacity: 1 }, transition: { delay: 0.5 + i * 0.12, duration: 0.6 } })
  return (
    <div className={arrastrando ? 'pantalla arrastrando' : 'pantalla'}
      onDragEnter={entrar} onDragOver={(e) => conFicheros(e) && e.preventDefault()} onDragLeave={dejar} onDrop={soltar}>
      <RedFondo resultado={resultado} />
      <div className="pantalla-fila superior">
        <a className="marca grande" href="#/" onClick={(e) => { e.preventDefault(); ir('') }} aria-label="Inicio">
          <Logo tamano={44} />
        </a>
        <span />
        <motion.button className="estado-modelo" {...aparece(0)} onClick={() => ir('modelo')} aria-live="polite">
          <span className={!listo ? 'luz luz-error' : ocupado ? 'luz luz-ocupada' : 'luz'} aria-hidden="true" />
          {listo ? `E57 · umbral ${fmt(modelo.threshold, 3)}` : modelo.estado === 'cargando' ? 'cargando…' : 'sin modelo'}
        </motion.button>
      </div>

      <div className="pantalla-centro">{children}</div>

      <div className="pantalla-fila inferior">
        <p className="pantalla-aviso">Uso educativo, sin validez clínica</p>
        <button className="desliza" onClick={() => {
          const hoja = document.querySelector('.hoja')
          if (hoja) window.scrollTo({ top: hoja.getBoundingClientRect().top + window.scrollY - 64, behavior: reducir ? 'auto' : 'smooth' })
        }}>
          Desliza para ver {TEXTO_DESLIZA[ruta] ?? 'la sección'}
          <span className="desliza-flecha" aria-hidden="true" />
        </button>
        <span />
      </div>

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
export function Cabecera({ ruta, ir, entrada, onSubir, onEjemplo, resultado, modelo, ocupado, historial, onAbrirHistorial, espera }) {
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
  // Mientras la hoja lo tapa, el encabezado se oscurece y se aleja un poco.
  const atenuar = useTransform(scrollYProgress, [0, 0.9], [1, reducir ? 1 : 0.22])
  const alejar = useTransform(scrollYProgress, [0, 0.9], [1, reducir ? 1 : 0.94])
  const subir = () => input.current?.click()
  const actual = entrada.tipo === 'ejemplo' ? entrada.id : null

  return (
    <>
      <input ref={input} type="file" accept=".png,image/png" multiple hidden
        onChange={(e) => { onSubir(e.target.files); e.target.value = '' }} />

      <Secciones ruta={ruta} ir={ir} flotante />

      <header className="cabecera" ref={cabecera}>
        <motion.div className="cabecera-escena" style={{ opacity: atenuar, scale: alejar }}>
        <Pantalla modelo={modelo} resultado={resultado} ocupado={ocupado} ir={ir} onArchivos={onSubir} ruta={ruta}>
          <div className="ancla-titulo">
          <div className="titulo-bloque">
            <p className="cabecera-antetitulo">{(TITULARES[ruta] ?? TITULARES[''])[0]}</p>
            <p className="cabecera-nombre"><Palabras texto={(TITULARES[ruta] ?? TITULARES[''])[1]} clave={`titulo-${ruta}`} /></p>
            <motion.p key={ruta} className="cabecera-sub" initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8, delay: 0.4, ease: SALIDA }}>
              {(TITULARES[ruta] ?? TITULARES[''])[2]}
            </motion.p>
          </div>
          <div className="acciones-bloque">
          <div className="mando" role="toolbar" aria-label="Centro de mando">
            <Iman>
              <Boton className="boton-grande" icono={<IconoSubir />} onClick={subir}>
                <span className="texto-largo">Subir imágenes</span><span className="texto-corto">Subir</span>
              </Boton>
            </Iman>
            <Historial historial={historial} onAbrir={onAbrirHistorial} onSubir={subir} espera={espera} />
            <Iman><MenuEjemplos actual={actual} onElegir={onEjemplo} grande /></Iman>
          </div>
          </div>
          </div>
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
