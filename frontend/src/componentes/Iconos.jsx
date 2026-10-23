// Iconos de trazo fino dibujados a mano (24 × 24), sin dependencias.
const base = {
  width: 18, height: 18, viewBox: '0 0 24 24', fill: 'none', stroke: 'currentColor',
  strokeWidth: 1.8, strokeLinecap: 'round', strokeLinejoin: 'round', 'aria-hidden': true,
}

export const IconoSubir = () => (
  <svg {...base}><path d="M12 16V4" /><path d="m7 9 5-5 5 5" /><path d="M5 20h14" /></svg>
)

export const IconoEjemplo = () => (
  <svg {...base}><rect x="3" y="5" width="7" height="7" rx="1.5" /><rect x="14" y="5" width="7" height="7" rx="1.5" />
    <rect x="3" y="15" width="18" height="4" rx="1.5" /></svg>
)

export const IconoCapas = () => (
  <svg {...base}><path d="m12 3 9 5-9 5-9-5 9-5Z" /><path d="m3 13 9 5 9-5" /></svg>
)

export const IconoAvanzar = () => (
  <svg {...base}><path d="M5 12h14" /><path d="m13 6 6 6-6 6" /></svg>
)
