import { useId } from 'react'

// Logotipo: una pila de mapas de activación (como los volúmenes del diagrama de la red) con un
// filtro 3 × 3 delante; la celda central se ilumina con el mapa de color del realce.
export function Logo({ tamano = 36 }) {
  const id = useId().replace(/:/g, '')
  const celdas = []
  for (let f = 0; f < 3; f++) for (let c = 0; c < 3; c++) celdas.push([c, f])
  return (
    <svg className="logo" width={tamano} height={tamano} viewBox="0 0 40 40" aria-hidden="true">
      <defs>
        <linearGradient id={`realce-${id}`} x1="0" y1="1" x2="1" y2="0">
          <stop offset="0" stopColor="#bc3754" />
          <stop offset="0.55" stopColor="#f98e09" />
          <stop offset="1" stopColor="#fcffa4" />
        </linearGradient>
      </defs>
      <g fill="none" stroke="currentColor" strokeLinejoin="round">
        <rect x="13" y="3" width="24" height="24" rx="3" strokeWidth="1.6" opacity="0.35" />
        <rect x="8" y="8" width="24" height="24" rx="3" strokeWidth="1.8" opacity="0.6" />
      </g>
      <rect x="3" y="13" width="24" height="24" rx="3" fill="#15191e" stroke="currentColor" strokeWidth="2.2" />
      {celdas.map(([c, f]) => (
        <rect key={`${c}${f}`} x={6.4 + c * 6.2} y={16.4 + f * 6.2} width="4.8" height="4.8" rx="1"
          fill={c === 1 && f === 1 ? `url(#realce-${id})` : 'currentColor'} opacity={c === 1 && f === 1 ? 1 : 0.28} />
      ))}
    </svg>
  )
}
