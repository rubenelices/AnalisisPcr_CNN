import { motion } from 'framer-motion'

// variante: 'principal' | 'secundario' | 'texto'; fondo: 'oscuro' (sala) | 'claro' (papel)
export function Boton({ variante = 'principal', fondo = 'oscuro', icono, children, className = '', ...props }) {
  return (
    <motion.button
      type="button"
      whileTap={{ scale: 0.97 }}
      transition={{ type: 'spring', stiffness: 600, damping: 30 }}
      className={`boton boton-${variante} sobre-${fondo} ${className}`}
      {...props}
    >
      {icono && <span className="boton-icono">{icono}</span>}
      <span>{children}</span>
    </motion.button>
  )
}
