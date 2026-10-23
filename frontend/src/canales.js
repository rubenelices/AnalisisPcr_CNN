import { useEffect, useState } from 'react'

// Calcula en el navegador los tres canales que recibe la red a partir de las fases de un ejemplo:
// PRE, EARLY − PRE y LATE − EARLY. Las restas se muestran con el gris medio como cero (más claro,
// la intensidad sube; más oscuro, baja) y el contraste duplicado para que se vean.
export function useCanales(ejemplo) {
  const [canales, setCanales] = useState(null)
  useEffect(() => {
    let vivo = true
    const cargar = (src) => new Promise((ok, mal) => { const i = new Image(); i.onload = () => ok(i); i.onerror = mal; i.src = src })
    Promise.all(ejemplo.fases.map((f) => cargar(f.src))).then(([pre, early, late]) => {
      const lienzo = document.createElement('canvas')
      lienzo.width = 256; lienzo.height = 256
      const ctx = lienzo.getContext('2d', { willReadFrequently: true })
      const leer = (img) => { ctx.drawImage(img, 0, 0, 256, 256); return ctx.getImageData(0, 0, 256, 256).data }
      const [a, b, c] = [leer(pre), leer(early), leer(late)]
      const pintar = (f) => {
        const out = ctx.createImageData(256, 256)
        for (let k = 0; k < out.data.length; k += 4) {
          const v = f(k)
          out.data[k] = out.data[k + 1] = out.data[k + 2] = v; out.data[k + 3] = 255
        }
        ctx.putImageData(out, 0, 0)
        return lienzo.toDataURL('image/png')
      }
      const resta = (x, y) => (k) => Math.max(0, Math.min(255, 128 + 2 * (x[k] - y[k])))
      if (vivo) setCanales([pintar((k) => a[k]), pintar(resta(b, a)), pintar(resta(c, b))])
    }).catch(() => {})
    return () => { vivo = false }
  }, [ejemplo])
  return canales
}
