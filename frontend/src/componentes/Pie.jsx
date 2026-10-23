import { RUTAS } from './Cabecera.jsx'
import { Logo } from './Logo.jsx'
import { fmt, useResumen } from '../datos.js'

export function Pie({ ir, modelo }) {
  const d = useResumen()
  const t = d.estado === 'listo' ? d.test_e57 : null
  return (
    <footer className="pie">
      <div className="pie-columnas">
        <div>
          <p className="pie-nombre"><Logo tamano={28} /> Respuesta al tratamiento neoadyuvante</p>
          <p>Trabajo individual de Aprendizaje Automático, 4.º de Ingeniería Matemática.</p>
          <p className="pie-legal">
            Uso educativo, sin validez clínica. Imágenes y datos: BreastDCEDL (Fridman et al., Scientific Data, 2026),
            licencia CC BY-NC 4.0.
          </p>
        </div>
        <nav aria-label="Secciones (pie)">
          <p className="pie-etiqueta">Secciones</p>
          <ul>
            {RUTAS.map((r) => (
              <li key={r.nombre}><a href={`#/${r.id}`} onClick={(e) => { e.preventDefault(); ir(r.id) }}>{r.nombre}</a></li>
            ))}
          </ul>
        </nav>
        <dl className="pie-datos">
          <div><dt>Modelo en servicio</dt><dd>{modelo.estado === 'listo' ? `${modelo.version.split('@')[0]} · umbral ${fmt(modelo.threshold, 3)}` : 'sin conexión'}</dd></div>
          <div><dt>Pesos (SHA-256)</dt><dd>{modelo.estado === 'listo' ? `${modelo.sha256.slice(0, 16)}…` : '—'}</dd></div>
          <div><dt>Test público, una evaluación</dt><dd>{t ? `AUC ${fmt(t.auc, 3)} · IC95 ${fmt(t.ic95[0], 3)}–${fmt(t.ic95[1], 3)}` : '—'}</dd></div>
        </dl>
      </div>
    </footer>
  )
}
