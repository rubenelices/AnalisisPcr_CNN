// Cortes de ejemplo de train (CC BY-NC 4.0, BreastDCEDL). La probabilidad no se guarda aquí:
// se calcula enviando sus tres fases al servidor, igual que una subida.
const cohortes = { spy1: 'I-SPY1', spy2: 'I-SPY2', duke: 'Duke' }

const ejemplo = (id, paciente, posicion, cohorte, medias) => ({
  id, paciente, posicion, cohorte: cohortes[cohorte],
  fases: [
    { id: 'PRE', cuando: 'antes del contraste', src: `/examples/${id}/PRE.png`, media: medias[0] },
    { id: 'EARLY', cuando: 'primera fase tras el contraste', src: `/examples/${id}/EARLY.png`, media: medias[1] },
    { id: 'LATE', cuando: 'fase tardía', src: `/examples/${id}/LATE.png`, media: medias[2] },
  ],
  realce: `/examples/${id}/REALCE.png`,
})

export const EJEMPLOS = [
  ejemplo('ISPY1_1001_z016', 'ISPY1_1001', 'z016', 'spy1', [0.123, 0.155, 0.164]),
  ejemplo('ISPY2-923968_z035', 'ISPY2-923968', 'z035', 'spy2', [0.124, 0.194, 0.206]),
  ejemplo('Breast_MRI_592_z034', 'Breast_MRI_592', 'z034', 'duke', [0.117, 0.141, 0.159]),
]

// Proporción de pacientes con pCR en train (322 de 1.097), para dar contexto a la probabilidad.
export const PREVALENCIA_TRAIN = 0.293
