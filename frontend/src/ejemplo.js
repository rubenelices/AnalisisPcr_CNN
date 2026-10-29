// Cortes de ejemplo de train (CC BY-NC 4.0, BreastDCEDL), generados con python -m src.web.web_ejemplos:
// los tres de siempre y uno al azar (semilla fija) por cohorte × pCR. La etiqueta pCR es la pública del train;
// la probabilidad no se guarda aquí: se calcula enviando sus tres fases al servidor, igual que una subida.
import DATOS from './ejemplos.json'

const COHORTES = { spy1: 'I-SPY1', spy2: 'I-SPY2', duke: 'Duke' }

const ejemplo = ({ id, paciente, posicion, cohorte, pcr, medias }) => ({
  id, paciente, posicion, pcr, cohorte: COHORTES[cohorte],
  fases: [
    { id: 'PRE', cuando: 'antes del contraste', src: `/examples/${id}/PRE.png`, media: medias[0] },
    { id: 'EARLY', cuando: 'primera fase tras el contraste', src: `/examples/${id}/EARLY.png`, media: medias[1] },
    { id: 'LATE', cuando: 'fase tardía', src: `/examples/${id}/LATE.png`, media: medias[2] },
  ],
  realce: `/examples/${id}/REALCE.png`,
})

export const EJEMPLOS = DATOS.map(ejemplo)

// Proporción de pacientes con pCR en train (322 de 1.097), para dar contexto a la probabilidad.
export const PREVALENCIA_TRAIN = 0.293
