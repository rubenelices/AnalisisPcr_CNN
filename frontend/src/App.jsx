import { useCallback, useEffect, useRef, useState } from 'react'
import { AnimatePresence, motion, useReducedMotion } from 'framer-motion'
import { Cabecera, RUTAS } from './componentes/Cabecera.jsx'
import { Pie } from './componentes/Pie.jsx'
import { Portada } from './paginas/Portada.jsx'
import { Prediccion } from './paginas/Prediccion.jsx'
import { Modelo } from './paginas/Modelo.jsx'
import { Resultados } from './paginas/Resultados.jsx'
import { Pruebas } from './paginas/Pruebas.jsx'
import { EJEMPLOS } from './ejemplo.js'
import { validarFases } from './subida.js'
import { agrupar } from './lote.js'
import { archivosDeEjemplo, obtenerModelo, predecir } from './api.js'

// Rutas con almohadilla (#/prediccion): funcionan en cualquier alojamiento estático
// y el botón atrás del navegador vuelve a la sección anterior.
const leerRuta = () => {
  const r = window.location.hash.replace(/^#\/?/, '')
  return RUTAS.some((x) => x.id === r) ? r : ''
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
  const registrar = useCallback((clave, r, origen, fases) => setHistorial((prev) => [
    { clave, p: r.probability_pcr, clase: r.predicted_class, umbral: r.threshold, origen, fases },
    ...prev.filter((h) => h.clave !== clave),
  ]), [])
  const ultima = useRef(null)  // evita que una respuesta antigua pise a una más reciente
  const reducir = useReducedMotion()

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

  // Cada corte del lote se valida y se predice en orden, uno detrás de otro.
  const procesar = async (ficheros) => {
    if (!ficheros?.length) return
    ir('pruebas')
    const { grupos, sueltos: fuera } = agrupar(ficheros)
    setSueltos(fuera)
    if (!grupos.length) return
    const sello = Date.now()
    const filas = grupos.map((g, i) => ({ id: `${sello}-${i}`, clave: g.clave, archivos: g.archivos, estado: 'comprobando' }))
    const cambiar = (id, datos) => setLote((prev) => prev.map((f) => (f.id === id ? { ...f, ...datos } : f)))
    setLote((prev) => [...filas, ...prev])
    setProcesando(true)
    for (const fila of filas) {
      if (fila.archivos.length !== 3) {
        const fases = fila.archivos.map((f) => f.name.match(/_(PRE|EARLY|LATE)\.png$/i)[1].toUpperCase())
        const faltan = ['PRE', 'EARLY', 'LATE'].filter((f) => !fases.includes(f))
        cambiar(fila.id, { estado: 'error', errores: [faltan.length
          ? `Falta ${faltan.join(' y ')}: este corte solo tiene ${fases.join(', ')}.`
          : `Hay fases repetidas (${fases.join(', ')}): debe haber exactamente una de cada.`] })
        continue
      }
      const v = await validarFases(fila.archivos)
      if (v.errores.length) { cambiar(fila.id, { estado: 'error', errores: v.errores }); continue }
      cambiar(fila.id, { estado: 'calculando', fases: v.fases, realce: v.realce, archivos: v.archivos })
      try {
        const r = await predecir(v.archivos)
        cambiar(fila.id, { estado: 'listo', resultado: r })
        ultima.current = fila.clave
        setResultado({ estado: 'listo', clave: fila.clave, ...r })
        registrar(fila.clave, r, { tipo: 'lote', id: fila.id }, v.fases.map((f) => f.src))
      } catch (e) {
        cambiar(fila.id, { estado: 'error', errores: [e.message] })
      }
    }
    setProcesando(false)
  }

  const probarEjemplos = async () => {
    const ficheros = []
    for (const e of EJEMPLOS) ficheros.push(...Object.values(await archivosDeEjemplo(e)))
    procesar(ficheros)
  }

  const verDetalle = (fila) => {
    setEntrada({ tipo: 'subida', nombre: fila.clave, fases: fila.fases, realce: fila.realce })
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
        resultado={resultado} modelo={modelo} ocupado={procesando || resultado.estado === 'cargando'} historial={historial} onAbrirHistorial={abrirHistorial} />
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
