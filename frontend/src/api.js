// Comunicación con el servidor de Python (app.py). Toda la inferencia ocurre allí,
// con el mismo preprocesado que en el entrenamiento; aquí solo se envían y reciben datos.
const FASES = ['PRE', 'EARLY', 'LATE']

async function leer(respuesta) {
  const datos = await respuesta.json().catch(() => ({}))
  if (!respuesta.ok) throw new Error(datos.error || `El servidor respondió con un error (${respuesta.status}).`)
  return datos
}

async function pedir(url, opciones) {
  let respuesta
  try {
    respuesta = await fetch(url, opciones)
  } catch {
    throw new Error('No se puede conectar con el servidor del modelo. Comprueba que app.py está en marcha.')
  }
  return leer(respuesta)
}

export const obtenerModelo = () => pedir('/api/modelo')

// archivos: { PRE: File, EARLY: File, LATE: File }
export function predecir(archivos) {
  const formulario = new FormData()
  for (const fase of FASES) formulario.append(fase, archivos[fase], archivos[fase].name || `${fase}.png`)
  return pedir('/api/predict', { method: 'POST', body: formulario })
}

// Los ejemplos se envían exactamente igual que una subida: así pasan por la misma validación.
export async function archivosDeEjemplo(ejemplo) {
  const archivos = {}
  for (const fase of ejemplo.fases) {
    const blob = await (await fetch(fase.src)).blob()
    archivos[fase.id] = new File([blob], `${ejemplo.id}_${fase.id}.png`, { type: 'image/png' })
  }
  return archivos
}
