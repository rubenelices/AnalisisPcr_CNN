import { useEffect, useState } from 'react'

// Cifras y datos de gráficas generados a partir de los resultados guardados:
//   /resultados/resumen.json  ← python -m src.web_resultados
//   /resultados/graficas.json ← python -m src.web_graficas
// Cada fichero se pide una vez y se comparte entre secciones.
const promesas = {}

function useJson(url) {
  const [datos, setDatos] = useState({ estado: 'cargando' })
  useEffect(() => {
    promesas[url] ??= fetch(url).then((r) => (r.ok ? r.json() : Promise.reject(new Error('sin datos'))))
    let vivo = true
    promesas[url].then((d) => vivo && setDatos({ estado: 'listo', ...d }))
      .catch(() => { delete promesas[url]; if (vivo) setDatos({ estado: 'error' }) })
    return () => { vivo = false }
  }, [url])
  return datos
}

export const useResumen = () => useJson('/resultados/resumen.json')
export const useGraficas = () => useJson('/resultados/graficas.json')

// Números con coma decimal, como en el resto del trabajo.
export const fmt = (v, d = 2) => v.toFixed(d).replace('.', ',')
// Probabilidad con 3 decimales, o 4 si con 3 se confundiría con el umbral (p. ej. 0,3249 frente a 0,325).
export const fmtProb = (p, umbral) => (fmt(p, 3) === fmt(umbral, 3) ? fmt(p, 4) : fmt(p, 3))
// Punto de millares también en números de 4 cifras.
export const mil = (n) => String(n).replace(/\B(?=(\d{3})+(?!\d))/g, '.')
