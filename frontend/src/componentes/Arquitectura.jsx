// Las capas reales de E57 (comprobadas contra el modelo: 298.161 parámetros).
export const CAPAS = [
  { id: 'entrada', etapa: 'Entrada', detalle: 'PRE, EARLY − PRE, LATE − EARLY (restas fijas, sin parámetros)', salida: '3 × 256 × 256', params: 0, c: 3, s: 256, color: '#7c8791' },
  { id: 'b1', etapa: 'Bloque 1', detalle: 'Conv 3×3 → MaxPool 2×2 → Conv 3×3, 16 canales', salida: '16 × 128 × 128', params: 2800, c: 16, s: 128, color: '#3d6b96' },
  { id: 'b2', etapa: 'Bloque 2', detalle: 'Conv 3×3 → MaxPool 2×2 → Conv 3×3, 32 canales', salida: '32 × 64 × 64', params: 13952, c: 32, s: 64, color: '#33608c' },
  { id: 'b3', etapa: 'Bloque 3', detalle: 'Conv 3×3 → MaxPool 2×2 → Conv 3×3, 64 canales', salida: '64 × 32 × 32', params: 55552, c: 64, s: 32, color: '#2a5582' },
  { id: 'b4', etapa: 'Bloque 4', detalle: 'Conv 3×3 → MaxPool 2×2 → Conv 3×3, 128 canales', salida: '128 × 16 × 16', params: 221696, c: 128, s: 16, color: '#224a75' },
  { id: 'gap', etapa: 'Resumen', detalle: 'Media espacial global (GAP): un número por canal', salida: '128', params: 0, c: 128, color: '#5d6873' },
  { id: 'cabeza', etapa: 'Cabeza', detalle: 'Lineal 128 → 32, ReLU, dropout 0,3, lineal 32 → 1', salida: '1 logit → probabilidad', params: 4161, c: 32, color: '#b5367a' },
]
