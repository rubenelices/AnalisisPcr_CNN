import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// En desarrollo, las llamadas a /api van al servidor de Python (app.py, puerto 8000).
// En producción, el propio app.py sirve la interfaz compilada y la API en el mismo origen.
export default defineConfig({
  plugins: [react()],
  server: { proxy: { '/api': 'http://127.0.0.1:8000' } },
})
