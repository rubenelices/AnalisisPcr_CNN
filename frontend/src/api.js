// Comunicación con el servidor de Python (app.py). Toda la inferencia ocurre allí,
// con el mismo preprocesado que en el entrenamiento; aquí solo se envían y reciben datos.
const FASES = ['PRE', 'EARLY', 'LATE']
// Vacío: la API está en el mismo origen (app.py sirve la interfaz). Si la interfaz se aloja en
// otro dominio, VITE_API_URL apunta al servidor del modelo (p. ej. un Hugging Face Space).
const API = (import.meta.env.VITE_API_URL || '').replace(/\/$/, '')

async function leer(respuesta) {
  const datos = await respuesta.json().catch(() => ({}))
  if (!respuesta.ok) throw new Error(datos.error || `El servidor respondió con un error (${respuesta.status}).`)
  return datos
}

async function pedir(url, opciones) {
  let respuesta
  try {
    respuesta = await fetch(API + url, opciones)
  } catch {
    throw new Error('No se puede conectar con el servidor del modelo. Comprueba que app.py está en marcha.')
  }
  return leer(respuesta)
}

export const obtenerModelo = () => pedir('/api/modelo')

// archivos: { PRE: File, EARLY: File, LATE: File }
function formularioFases(archivos) {
  const formulario = new FormData()
  for (const fase of FASES) formulario.append(fase, archivos[fase], archivos[fase].name || `${fase}.png`)
  return formulario
}

export const predecir = (archivos) => pedir('/api/predict', { method: 'POST', body: formularioFases(archivos) })

// Mapa de oclusión del mismo corte: el servidor tapa cada cuadro y vuelve a predecir.
// opciones: { patch: 8 | 16 | 32, fase: '' (las tres) | 'PRE' | 'EARLY' | 'LATE' }
export function mapaOclusion(archivos, { patch = 16, fase = '' } = {}) {
  const formulario = formularioFases(archivos)
  formulario.append('patch', String(patch))
  if (fase) formulario.append('fase', fase)
  return pedir('/api/occlusion', { method: 'POST', body: formulario })
}

// Laboratorio: el servidor aplica los ajustes (aumentos del entrenamiento o sondas sobre las fases) y vuelve
// a predecir. Devuelve la probabilidad y las tres fases tal como las recibe la red.
export function laboratorio(archivos, ajustes) {
  const formulario = formularioFases(archivos)
  for (const [clave, valor] of Object.entries(ajustes)) {
    formulario.append(clave, typeof valor === 'boolean' ? (valor ? '1' : '0') : String(valor))
  }
  return pedir('/api/laboratorio', { method: 'POST', body: formulario })
}

// pCR real del corte según el CSV de datos, o null si el corte no está en él.
export async function etiquetaReal(clave) {
  try {
    const r = await fetch(`${API}/api/etiqueta/${encodeURIComponent(clave)}`)
    return r.ok ? (await r.json()).pcr : null
  } catch { return null }
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
