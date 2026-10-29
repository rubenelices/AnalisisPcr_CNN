// Comprobación de las tres fases en el navegador, para avisar al momento con mensajes claros.
// El servidor repetirá estas comprobaciones: la validación que cuenta es la suya.
const FASES = ['PRE', 'EARLY', 'LATE']
const MAX_BYTES = 5 * 1024 * 1024

const fasePorNombre = (nombre) => {
  const m = nombre.match(/_(PRE|EARLY|LATE)\.png$/i)
  return m ? m[1].toUpperCase() : null
}

const cargarImagen = (url) => new Promise((ok, fallo) => {
  const img = new Image()
  img.onload = () => ok(img)
  img.onerror = () => fallo(new Error('no se puede leer'))
  img.src = url
})

const pixeles = (img) => {
  const c = document.createElement('canvas')
  c.width = img.naturalWidth; c.height = img.naturalHeight
  const ctx = c.getContext('2d', { willReadFrequently: true })
  ctx.drawImage(img, 0, 0)
  return ctx.getImageData(0, 0, c.width, c.height).data
}

// Paleta inferno (la misma que usa la figura del enunciado), interpolada entre 6 paradas.
const INFERNO = [[0, 0, 4], [66, 10, 104], [147, 38, 103], [221, 81, 58], [252, 165, 10], [252, 255, 164]]
const colorInferno = (t) => {
  const x = Math.min(Math.max(t, 0), 1) * (INFERNO.length - 1)
  const i = Math.min(Math.floor(x), INFERNO.length - 2), f = x - i
  return INFERNO[i].map((v, k) => Math.round(v + (INFERNO[i + 1][k] - v) * f))
}

function mapaRealce(pre, early, ancho, alto) {
  const c = document.createElement('canvas')
  c.width = ancho; c.height = alto
  const ctx = c.getContext('2d')
  const salida = ctx.createImageData(ancho, alto)
  for (let p = 0; p < pre.length; p += 4) {
    const [r, g, b] = colorInferno((early[p] - pre[p]) / 255 / 0.5)
    salida.data[p] = r; salida.data[p + 1] = g; salida.data[p + 2] = b; salida.data[p + 3] = 255
  }
  ctx.putImageData(salida, 0, 0)
  return c.toDataURL('image/png')
}

// Realce EARLY − PRE a partir de dos imágenes ya cargables (URL o data URL), con la misma paleta.
export async function realceDeUrls(pre, early) {
  const [a, b] = await Promise.all([cargarImagen(pre), cargarImagen(early)])
  return mapaRealce(pixeles(a), pixeles(b), a.naturalWidth, a.naturalHeight)
}

export async function validarFases(lista) {
  const archivos = Array.from(lista)
  const errores = []
  if (archivos.length !== 3) {
    return { errores: [`Has elegido ${archivos.length} ${archivos.length === 1 ? 'fichero' : 'ficheros'}. Hacen falta exactamente tres: PRE, EARLY y LATE del mismo corte.`] }
  }
  const porFase = {}
  for (const a of archivos) {
    if (!/\.png$/i.test(a.name) || (a.type && a.type !== 'image/png')) errores.push(`${a.name} no es un PNG.`)
    else if (a.size > MAX_BYTES) errores.push(`${a.name} ocupa más de 5 MB.`)
    const fase = fasePorNombre(a.name)
    if (!fase) errores.push(`${a.name} no indica su fase: el nombre debe terminar en _PRE.png, _EARLY.png o _LATE.png.`)
    else if (porFase[fase]) errores.push(`La fase ${fase} está repetida.`)
    else porFase[fase] = a
  }
  const faltan = FASES.filter((f) => !porFase[f])
  if (faltan.length && !errores.some((e) => e.includes('repetida') || e.includes('no indica'))) {
    errores.push(`Falta la fase ${faltan.join(' y ')}.`)
  }
  if (errores.length) return { errores }

  const bases = FASES.map((f) => porFase[f].name.replace(/_(PRE|EARLY|LATE)\.png$/i, ''))
  if (new Set(bases).size !== 1) {
    return { errores: ['Las tres fases deben ser del mismo corte: sus nombres, sin la fase, no coinciden.'] }
  }

  const fases = [], datos = {}
  for (const f of FASES) {
    const url = URL.createObjectURL(porFase[f])
    let img
    try { img = await cargarImagen(url) } catch {
      return { errores: [`${porFase[f].name} está dañado o no es una imagen válida.`] }
    }
    if (img.naturalWidth !== 256 || img.naturalHeight !== 256) {
      return { errores: [`${porFase[f].name} mide ${img.naturalWidth} × ${img.naturalHeight} píxeles; deben ser 256 × 256.`] }
    }
    const px = pixeles(img)
    let suma = 0, color = false
    for (let p = 0; p < px.length; p += 4) {
      if (px[p] !== px[p + 1] || px[p] !== px[p + 2]) color = true
      suma += px[p]
    }
    if (color) return { errores: [`${porFase[f].name} tiene color: cada fase debe ser una imagen en escala de grises.`] }
    datos[f] = px
    fases.push({ id: f, src: url, media: suma / (px.length / 4) / 255,
      cuando: { PRE: 'antes del contraste', EARLY: 'primera fase tras el contraste', LATE: 'fase tardía' }[f] })
  }
  return {
    errores: [],
    nombre: bases[0],
    archivos: porFase,
    fases,
    realce: mapaRealce(datos.PRE, datos.EARLY, 256, 256),
  }
}

