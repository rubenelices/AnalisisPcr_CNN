// Pruebas por lotes: el día de la defensa se pueden soltar de golpe las fases de varios cortes
// (o carpetas enteras). Los ficheros se agrupan por corte según su nombre, con el formato del
// dataset: <paciente>_<corte>_PRE.png, _EARLY.png y _LATE.png. Cada grupo pasa después por la
// misma validación y la misma petición al servidor que una subida suelta.

const FASE = /^(.*)_(PRE|EARLY|LATE)\.png$/i

export function agrupar(lista) {
  const grupos = new Map()
  const sueltos = []
  for (const a of lista) {
    if (a.name.startsWith('.')) continue  // ficheros ocultos del sistema (.DS_Store…)
    if (!/\.png$/i.test(a.name)) { sueltos.push({ nombre: a.name, motivo: 'no es un PNG' }); continue }
    const m = a.name.match(FASE)
    if (!m) { sueltos.push({ nombre: a.name, motivo: 'el nombre no termina en _PRE, _EARLY o _LATE' }); continue }
    if (!grupos.has(m[1])) grupos.set(m[1], [])
    grupos.get(m[1]).push(a)
  }
  const ordenados = [...grupos].sort(([a], [b]) => a.localeCompare(b, 'es', { numeric: true }))
  return { grupos: ordenados.map(([clave, archivos]) => ({ clave, archivos })), sueltos }
}

// Ficheros de un arrastre, entrando en las carpetas que se suelten.
export async function archivosDeArrastre(dt) {
  const entradas = [...(dt.items ?? [])].map((i) => i.webkitGetAsEntry?.()).filter(Boolean)
  if (!entradas.length) return [...(dt.files ?? [])]
  const salida = []
  const leer = async (e) => {
    if (e.isFile) {
      salida.push(await new Promise((ok, mal) => e.file(ok, mal)))
    } else if (e.isDirectory) {
      const lector = e.createReader()
      let tanda
      do {
        tanda = await new Promise((ok, mal) => lector.readEntries(ok, mal))
        for (const hijo of tanda) await leer(hijo)
      } while (tanda.length)
    }
  }
  for (const e of entradas) await leer(e)
  return salida
}

// Tabla de resultados en texto separado por tabuladores (se pega tal cual en una hoja de cálculo).
export function tablaTexto(filas) {
  const cabecera = ['corte', 'probabilidad_pcr', 'clase', 'umbral', 'modelo', 'sha256', 'dispositivo', 'ms']
  const lineas = filas.filter((f) => f.estado === 'listo').map(({ clave, resultado: r }) => [
    clave, r.probability_pcr.toFixed(6), r.predicted_class === 1 ? 'pCR' : 'sin pCR', r.threshold,
    r.model_version, r.model_sha256, r.device, Math.round(r.latency_ms),
  ].join('\t'))
  return [cabecera.join('\t'), ...lineas].join('\n')
}
