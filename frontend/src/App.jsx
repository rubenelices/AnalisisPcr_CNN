import { useCallback, useEffect, useRef, useState } from 'react'
import { AnimatePresence, motion, useReducedMotion, useScroll } from 'framer-motion'
import { Cabecera, RUTAS } from './componentes/Cabecera.jsx'
import { Pie } from './componentes/Pie.jsx'
import { Portada } from './paginas/Portada.jsx'
import { Prediccion } from './paginas/Prediccion.jsx'
import { Modelo } from './paginas/Modelo.jsx'
import { Resultados } from './paginas/Resultados.jsx'
import { Pruebas } from './paginas/Pruebas.jsx'
import { EJEMPLOS } from './ejemplo.js'
import { validarFases } from './subida.js'
import { agrupar, unirFases } from './lote.js'
import { archivosDeEjemplo, etiquetaReal, obtenerModelo, predecir } from './api.js'

// Rutas con almohadilla (#/prediccion): funcionan en cualquier alojamiento estático
// y el botón atrás del navegador vuelve a la sección anterior.
const leerRuta = () => {
  const r = window.location.hash.replace(/^#\/?/, '')
  return RUTAS.some((x) => x.id === r) ? r : ''
}

// Animaciones de scroll: cada bloque de una sección (título, texto, figura…) entra una vez, subiendo y
// apareciendo, con un pequeño escalonado entre hermanos. El CSS solo lo oculta si JS ha marcado el bloque.
function useRevelarAlHacerScroll(clave, reducir) {
  useEffect(() => {
    const raiz = document.querySelector('.hoja')
    if (!raiz || reducir || !('IntersectionObserver' in window)) return
    const io = new IntersectionObserver((entradas) => entradas.forEach((e) => {
      if (e.isIntersecting) { e.target.classList.add('visto'); io.unobserve(e.target) }
    }), { rootMargin: '0px 0px -8% 0px', threshold: 0.06 })
    let cuadro = 0
    const marcar = () => {
      cuadro = 0
      raiz.querySelectorAll('.papel-seccion > *').forEach((el) => {
        if (el.dataset.rv) return
        el.dataset.rv = '1'
        const hermanos = [...el.parentElement.children]
        el.style.setProperty('--rv', `${Math.min(hermanos.indexOf(el), 4) * 70}ms`)
        io.observe(el)
      })
    }
    const mo = new MutationObserver(() => { if (!cuadro) cuadro = requestAnimationFrame(marcar) })
    mo.observe(raiz, { childList: true, subtree: true })
    marcar()
    return () => { mo.disconnect(); io.disconnect(); cancelAnimationFrame(cuadro) }
  }, [clave, reducir])
}

export default function App() {
  const [ruta, setRuta] = useState(leerRuta)
  // Lo que hay en el visor de Predicción: un ejemplo o tres fases subidas.
  const [entrada, setEntrada] = useState({ tipo: 'ejemplo', id: EJEMPLOS[0].id })
  // Ficha del modelo cargado en el servidor y resultado de la última predicción.
  const [modelo, setModelo] = useState({ estado: 'cargando' })
  const [resultado, setResultado] = useState({ estado: 'cargando', clave: EJEMPLOS[0].id })
  // Cortes de la sección Pruebas (el más reciente arriba) y ficheros que no se han podido agrupar.
  const [lote, setLote] = useState([])
  const [sueltos, setSueltos] = useState([])
  const [procesando, setProcesando] = useState(false)
  // Últimas predicciones de la sesión, para el encabezado (la más reciente primero).
  const [historial, setHistorial] = useState([])
  // Mientras un corte espera alguna fase, el encabezado muestra las que ya han llegado.
  const [verEspera, setVerEspera] = useState(false)
  const registrar = useCallback((clave, r, origen, fases) => {
    setVerEspera(false)
    setHistorial((prev) => [
      { clave, p: r.probability_pcr, clase: r.predicted_class, umbral: r.threshold, origen, fases },
      ...prev.filter((h) => h.clave !== clave),
    ])
  }, [])
  const ultima = useRef(null)  // evita que una respuesta antigua pise a una más reciente
  const reducir = useReducedMotion()
  useRevelarAlHacerScroll(ruta, reducir)
  const { scrollYProgress } = useScroll()

  useEffect(() => {
    const alCambiar = () => { setRuta(leerRuta()); window.scrollTo({ top: 0 }) }
    window.addEventListener('hashchange', alCambiar)
    return () => window.removeEventListener('hashchange', alCambiar)
  }, [])

  const ir = useCallback((id) => { window.location.hash = id ? `/${id}` : '/' }, [])

  // anotar = false para la carga inicial: el historial solo recoge lo que pide quien usa la web.
  const cargarEjemplo = useCallback(async (id, anotar = true) => {
    const ejemplo = EJEMPLOS.find((e) => e.id === id)
    ultima.current = id
    setResultado({ estado: 'cargando', clave: id })
    try {
      const r = await predecir(await archivosDeEjemplo(ejemplo))
      if (ultima.current === id) setResultado({ estado: 'listo', clave: id, ...r })
      if (anotar) registrar(id, r, { tipo: 'ejemplo' }, ejemplo.fases.map((f) => f.src))
    } catch (e) {
      if (ultima.current === id) setResultado({ estado: 'error', clave: id, error: e.message })
    }
  }, [registrar])

  useEffect(() => {
    obtenerModelo().then((m) => setModelo({ estado: 'listo', ...m }))
      .catch((e) => setModelo({ estado: 'error', error: e.message }))
    cargarEjemplo(EJEMPLOS[0].id, false)
  }, [cargarEjemplo])

  const elegirEjemplo = (id) => { setEntrada({ tipo: 'ejemplo', id }); ir('prediccion'); cargarEjemplo(id) }

  // Cada corte del lote se valida y se predice en orden, uno detrás de otro. Las fases pueden llegar sueltas y en
  // varios envíos: un corte incompleto queda «esperando» y, cuando llega la fase que falta, se calcula.
  const loteRef = useRef([])
  loteRef.current = lote
  const procesar = async (ficheros) => {
    if (!ficheros?.length) return
    ir('pruebas')
    const { grupos, sueltos: fuera } = agrupar(ficheros)
    setSueltos(fuera)
    if (!grupos.length) return
    const sello = Date.now()
    const previas = new Map(loteRef.current.filter((f) => f.estado === 'esperando').map((f) => [f.clave, f]))
    const filas = grupos.map((g, i) => {
      const previa = previas.get(g.clave)
      const archivos = unirFases(previa?.archivos ?? [], g.archivos)
      const completo = archivos.length === 3
      return { ...(previa ?? {}), id: previa?.id ?? `${sello}-${i}`, clave: g.clave, archivos, estado: completo ? 'comprobando' : 'esperando', errores: undefined }
    })
    const ids = new Set(filas.map((f) => f.id))
    setLote((prev) => [...filas.filter((f) => !prev.some((p) => p.id === f.id)), ...prev.map((p) => filas.find((f) => f.id === p.id) ?? p)])
    setVerEspera(filas.some((f) => f.estado === 'esperando'))
    const cambiar = (id, datos) => setLote((prev) => prev.map((f) => (f.id === id ? { ...f, ...datos } : f)))
    setProcesando(true)
    try {
    for (const fila of filas) {
      if (fila.estado === 'esperando') continue
      const v = await validarFases(fila.archivos)
      if (v.errores.length) { cambiar(fila.id, { estado: 'error', errores: v.errores }); continue }
      cambiar(fila.id, { estado: 'calculando', fases: v.fases, realce: v.realce, archivos: v.archivos })
      try {
        const r = await predecir(v.archivos)
        const real = await etiquetaReal(fila.clave)
        cambiar(fila.id, { estado: 'listo', resultado: r, real })
        ultima.current = fila.clave
        setResultado({ estado: 'listo', clave: fila.clave, ...r })
        registrar(fila.clave, r, { tipo: 'lote', id: fila.id }, v.fases.map((f) => f.src))
      } catch (e) {
        cambiar(fila.id, { estado: 'error', errores: [e.message] })
      }
    }
    } finally { setProcesando(false) }
    return ids
  }

  const probarEjemplos = async () => {
    const ficheros = []
    for (const e of EJEMPLOS) ficheros.push(...Object.values(await archivosDeEjemplo(e)))
    procesar(ficheros)
  }

  const verDetalle = (fila) => {
    setEntrada({ tipo: 'subida', nombre: fila.clave, fases: fila.fases, realce: fila.realce, archivos: fila.archivos })
    ultima.current = fila.clave
    setResultado({ estado: 'listo', clave: fila.clave, ...fila.resultado })
    ir('prediccion')
  }

  // Abrir una predicción del historial: los ejemplos se recargan; las subidas se toman de Pruebas.
  const abrirHistorial = (h) => {
    if (h.origen.tipo === 'ejemplo') return elegirEjemplo(h.clave)
    const fila = lote.find((f) => f.id === h.origen.id && f.estado === 'listo')
    if (fila) verDetalle(fila)
    else ir('pruebas')
  }

  const ejemplo = entrada.tipo === 'ejemplo' ? EJEMPLOS.find((e) => e.id === entrada.id) : null

  let pagina
  if (ruta === '') pagina = <Portada ir={ir} />
  else if (ruta === 'prediccion') pagina = <Prediccion entrada={entrada} ejemplo={ejemplo} errores={[]} resultado={resultado} modelo={modelo} />
  else if (ruta === 'modelo') pagina = <Modelo modelo={modelo} />
  else if (ruta === 'resultados') pagina = <Resultados />
  else pagina = (
    <Pruebas lote={lote} sueltos={sueltos} modelo={modelo} ocupado={procesando}
      onArchivos={procesar} onEjemplos={probarEjemplos} onVaciar={() => { setLote([]); setSueltos([]) }} onDetalle={verDetalle} />
  )

  return (
    <>
      <Cabecera ruta={ruta} ir={ir} entrada={entrada} onSubir={(f) => procesar([...f])} onEjemplo={elegirEjemplo}
        resultado={resultado} modelo={modelo} ocupado={procesando || resultado.estado === 'cargando'} historial={historial} onAbrirHistorial={abrirHistorial}
        espera={verEspera ? lote.find((f) => f.estado === 'esperando') : null} />
      <motion.div className="progreso-pagina" style={{ scaleX: scrollYProgress }} aria-hidden="true" />
      <div className="hoja">
      <AnimatePresence mode="wait" initial={false}>
        <motion.main key={ruta || 'portada'}
          initial={{ opacity: 0, y: reducir ? 0 : 8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}
          transition={{ duration: 0.25, ease: 'easeOut' }}>
          {pagina}
        </motion.main>
      </AnimatePresence>
      <Pie ir={ir} modelo={modelo} />
      </div>
    </>
  )
}
