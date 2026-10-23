import { useRef, useState } from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import { Boton } from '../componentes/Boton.jsx'
import { IconoAvanzar, IconoEjemplo, IconoSubir } from '../componentes/Iconos.jsx'
import { SALIDA } from '../componentes/Movimiento.jsx'
import { archivosDeArrastre, tablaTexto } from '../lote.js'
import { fmt, fmtProb } from '../datos.js'

function Zona({ onArchivos, ocupado }) {
  const input = useRef(null)
  const carpeta = useRef(null)
  const [encima, setEncima] = useState(false)
  const soltar = async (e) => {
    e.preventDefault(); setEncima(false)
    onArchivos(await archivosDeArrastre(e.dataTransfer))
  }
  return (
    <div className={encima ? 'zona encima' : 'zona'}
      onDragOver={(e) => { e.preventDefault(); setEncima(true) }} onDragLeave={() => setEncima(false)} onDrop={soltar}>
      <input ref={input} type="file" accept=".png,image/png" multiple hidden onChange={(e) => { onArchivos([...e.target.files]); e.target.value = '' }} />
      <input ref={carpeta} type="file" webkitdirectory="" multiple hidden onChange={(e) => { onArchivos([...e.target.files]); e.target.value = '' }} />
      <span className="zona-icono" aria-hidden="true"><IconoSubir /></span>
      <p className="zona-titulo">Arrastra aquí las imágenes o una carpeta</p>
      <p className="zona-texto">
        Tres PNG por corte (_PRE, _EARLY y _LATE). Se pueden soltar varios cortes a la vez: se agrupan solos por el nombre.
      </p>
      <div className="zona-acciones">
        <Boton variante="principal" fondo="claro" icono={<IconoSubir />} onClick={() => input.current?.click()} disabled={ocupado}>Elegir imágenes</Boton>
        <Boton variante="secundario" fondo="claro" onClick={() => carpeta.current?.click()} disabled={ocupado}>Elegir carpeta</Boton>
      </div>
    </div>
  )
}

function Fila({ fila, onDetalle }) {
  const r = fila.resultado
  return (
    <motion.li layout className={`prueba prueba-${fila.estado}`}
      initial={{ opacity: 0, y: 14 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }} transition={{ duration: 0.35, ease: SALIDA }}>
      <div className="prueba-nombre">
        <strong>{fila.clave}</strong>
        <span>{{ comprobando: 'Comprobando las fases…', calculando: 'Calculando…', listo: `${Math.max(1, Math.round(r?.latency_ms ?? 0))} ms en ${r?.device?.toUpperCase()}`, error: 'No se ha podido usar' }[fila.estado]}</span>
      </div>
      <div className="prueba-imagenes">
        {fila.fases ? fila.fases.map((f) => <figure key={f.id}><img src={f.src} alt={`${f.id} de ${fila.clave}`} width="96" height="96" /><figcaption>{f.id}</figcaption></figure>)
          : [0, 1, 2].map((k) => <span key={k} className="prueba-hueco" />)}
        {fila.realce ? <figure><img src={fila.realce} alt={`Realce de ${fila.clave}`} width="96" height="96" /><figcaption>Realce</figcaption></figure> : <span className="prueba-hueco" />}
      </div>
      <div className="prueba-resultado">
        {fila.estado === 'listo' && (
          <>
            <span className="prueba-cifra">{fmtProb(r.probability_pcr, r.threshold)}</span>
            <span className="prueba-escala" aria-hidden="true">
              <span className="prueba-umbral" style={{ left: `${r.threshold * 100}%` }} />
              <motion.span className="prueba-marca" initial={{ left: '0%' }} animate={{ left: `${r.probability_pcr * 100}%` }} transition={{ duration: 0.9, ease: SALIDA }} />
            </span>
            <span className={r.predicted_class === 1 ? 'clase-chip si' : 'clase-chip'}>{r.predicted_class === 1 ? 'pCR' : 'sin pCR'}</span>
          </>
        )}
        {(fila.estado === 'comprobando' || fila.estado === 'calculando') && <span className="cargando" aria-label="Calculando" />}
        {fila.estado === 'error' && <ul className="prueba-errores" role="alert">{fila.errores.map((e) => <li key={e}>{e}</li>)}</ul>}
      </div>
      <div className="prueba-accion">
        {fila.estado === 'listo' && <Boton variante="texto" fondo="claro" icono={<IconoAvanzar />} onClick={() => onDetalle(fila)}>Detalle</Boton>}
      </div>
    </motion.li>
  )
}

export function Pruebas({ lote, sueltos, modelo, ocupado, onArchivos, onEjemplos, onVaciar, onDetalle }) {
  const [copiado, setCopiado] = useState(false)
  const listos = lote.filter((f) => f.estado === 'listo')
  const copiar = async () => {
    try { await navigator.clipboard.writeText(tablaTexto(lote)); setCopiado(true); setTimeout(() => setCopiado(false), 1800) } catch { /* sin portapapeles */ }
  }
  const descargar = () => {
    const url = URL.createObjectURL(new Blob([tablaTexto(lote)], { type: 'text/tab-separated-values' }))
    const a = Object.assign(document.createElement('a'), { href: url, download: 'predicciones_pcr.tsv' })
    a.click(); URL.revokeObjectURL(url)
  }
  return (
    <div className="pagina-pruebas">
      <section className="pruebas-cabeza">
        <div>
          <h1 className="titulo-pagina">Pruebas</h1>
          <p className="entradilla">
            La sección para la defensa: se cargan las imágenes de cualquier corte y el modelo de entrega responde al
            momento, con el mismo preprocesado que en el entrenamiento. Nada se guarda en el servidor.
          </p>
          <ol className="pruebas-pasos">
            <li><strong>Suelta las fases</strong> de uno o varios cortes, o una carpeta entera.</li>
            <li><strong>Cada corte se comprueba</strong>: tres fases, 256 × 256 píxeles, escala de grises.</li>
            <li><strong>El servidor predice</strong> y la fila muestra la probabilidad frente al umbral.</li>
          </ol>
        </div>
        <dl className="pruebas-modelo">
          <div><dt>Modelo</dt><dd>{modelo.estado === 'listo' ? modelo.version : '—'}</dd></div>
          <div><dt>Umbral</dt><dd>{modelo.estado === 'listo' ? fmt(modelo.threshold, 3) : '—'}</dd></div>
          <div><dt>Pesos (SHA-256)</dt><dd>{modelo.estado === 'listo' ? `${modelo.sha256.slice(0, 16)}…` : '—'}</dd></div>
          <div><dt>Dispositivo</dt><dd>{modelo.estado === 'listo' ? modelo.device.toUpperCase() : 'sin conexión'}</dd></div>
        </dl>
      </section>

      <section className="pruebas-cuerpo" aria-label="Cargar imágenes y resultados">
        <Zona onArchivos={onArchivos} ocupado={ocupado} />

        {sueltos.length > 0 && (
          <div className="sueltos" role="alert">
            <strong>{sueltos.length === 1 ? 'Un fichero no se ha usado' : `${sueltos.length} ficheros no se han usado`}</strong>
            <ul>{sueltos.map((s) => <li key={s.nombre}>{s.nombre}: {s.motivo}</li>)}</ul>
          </div>
        )}

        <div className="pruebas-barra">
          <h2>Resultados {lote.length > 0 && <span>{listos.length} de {lote.length} cortes · {listos.filter((f) => f.resultado.predicted_class === 1).length} clasificados como pCR</span>}</h2>
          <div className="pruebas-botones">
            <Boton variante="secundario" fondo="claro" icono={<IconoEjemplo />} onClick={onEjemplos} disabled={ocupado}>Probar con los 3 ejemplos</Boton>
            {listos.length > 0 && <Boton variante="secundario" fondo="claro" onClick={copiar}>{copiado ? 'Copiado' : 'Copiar tabla'}</Boton>}
            {listos.length > 0 && <Boton variante="secundario" fondo="claro" onClick={descargar}>Descargar</Boton>}
            {lote.length > 0 && <Boton variante="texto" fondo="claro" onClick={onVaciar} disabled={ocupado}>Vaciar</Boton>}
          </div>
        </div>

        {lote.length === 0
          ? <p className="pruebas-vacio">Todavía no hay cortes. Suelta imágenes arriba o prueba con los ejemplos.</p>
          : <ol className="pruebas-lista"><AnimatePresence initial={false}>{lote.map((f) => <Fila key={f.id} fila={f} onDetalle={onDetalle} />)}</AnimatePresence></ol>}

        <p className="nota-pie">
          Uso educativo, sin validez clínica. Las imágenes se validan aquí para avisar antes, pero la comprobación que cuenta
          la repite el servidor, que es quien aplica el preprocesado del entrenamiento (valores / 255).
        </p>
      </section>
    </div>
  )
}
