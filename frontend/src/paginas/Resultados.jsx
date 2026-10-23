import { motion, useReducedMotion } from 'framer-motion'
import { Barras, COLOR, Figura, Grafico, Horizontal, Linea, Texto } from '../componentes/Graficas.jsx'
import { Contador, Revelar } from '../componentes/Movimiento.jsx'
import { fmt, useGraficas, useResumen } from '../datos.js'

const COHORTES = { spy1: 'I-SPY1', spy2: 'I-SPY2', duke: 'Duke' }
const MIN = 0.35, MAX = 0.9
const x = (v) => `${((v - MIN) / (MAX - MIN)) * 100}%`

function EscalaMedidas({ filas, referencia }) {
  const reducir = useReducedMotion()
  const marcas = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
  return (
    <div className="medidas" role="img" aria-label="AUC del modelo medido de distintas formas, con sus intervalos de confianza">
      <div className="medidas-etiquetas">
        <span className="medidas-hueco" />
        {filas.map((f) => (
          <div key={f.nombre} className={`medida-texto medida-${f.tipo}`}><strong>{f.nombre}</strong><span>{f.detalle}</span></div>
        ))}
      </div>
      <div className="medidas-grafico">
        <div className="medidas-eje">{marcas.map((m) => <span key={m} style={{ left: x(m) }}>{fmt(m, 1)}</span>)}</div>
        <span className="medidas-guia azar" style={{ left: x(0.5) }}><span>azar</span></span>
        <span className="medidas-guia buena" style={{ left: x(referencia) }}><span>buen modelo (teórico)</span></span>
        {filas.map((f, i) => (
          <div key={f.nombre} className={`medida-pista medida-${f.tipo}`}>
            {f.ic && <span className="medida-ic" style={{ left: x(f.ic[0]), width: `calc(${x(f.ic[1])} - ${x(f.ic[0])})` }} />}
            <motion.span className="medida-punto" style={{ left: x(f.valor) }}
              initial={{ opacity: 0, scale: reducir ? 1 : 0.6 }} whileInView={{ opacity: 1, scale: 1 }} viewport={{ once: true }}
              transition={{ delay: reducir ? 0 : 0.15 + i * 0.1, duration: 0.4 }} />
            <span className="medida-valor" style={{ left: x(f.valor) }}>{fmt(f.valor, 3)}</span>
          </div>
        ))}
      </div>
    </div>
  )
}

function Cohortes({ cohortes }) {
  return (
    <div className="cohortes">
      {Object.entries(cohortes).sort((a, b) => b[1].n - a[1].n).map(([c, v], i) => {
        const ancho = (Math.abs(v.auc - 0.5) / 0.25) * 50
        return (
          <div key={c} className="cohorte">
            <span className="cohorte-nombre">{COHORTES[c]}<small>{v.n} pacientes, {v.pcr} con pCR</small></span>
            <span className="cohorte-pista">
              <span className="cohorte-centro" />
              <motion.span className={v.auc >= 0.5 ? 'cohorte-barra positiva' : 'cohorte-barra negativa'}
                style={v.auc >= 0.5 ? { left: '50%', width: `${ancho}%`, originX: 0 } : { right: '50%', width: `${ancho}%`, originX: 1 }}
                initial={{ scaleX: 0 }} whileInView={{ scaleX: 1 }} viewport={{ once: true }} transition={{ duration: 0.8, delay: i * 0.1 }} />
            </span>
            <span className="cohorte-valor">{fmt(v.auc, 2)}</span>
          </div>
        )
      })}
      <div className="cohorte-eje"><span>0,25 · ordena al revés</span><span>0,5 · azar</span><span>0,75</span></div>
    </div>
  )
}

function Kpis({ t, g }) {
  const items = [
    { valor: t.auc, d: 3, nombre: 'AUC por paciente', nota: `IC95 ${fmt(t.ic95[0], 3)}–${fmt(t.ic95[1], 3)}` },
    { valor: g.e57.test.corte.auc, d: 3, nombre: 'AUC por corte', nota: `${g.e57.test.corte.n.toLocaleString('es-ES')} cortes sueltos` },
    { valor: t.sensibilidad, d: 2, nombre: 'Sensibilidad', nota: `${t.matriz.tp} de ${t.pcr} pCR detectadas` },
    { valor: t.especificidad, d: 2, nombre: 'Especificidad', nota: `${t.matriz.tn} de ${t.matriz.tn + t.matriz.fp} sin pCR` },
    { valor: t.precision, d: 2, nombre: 'Precisión', nota: 'cuando dice pCR' },
    { valor: t.balanced_accuracy, d: 2, nombre: 'Exactitud equilibrada', nota: `umbral ${fmt(t.umbral, 3)}` },
  ]
  return (
    <div className="kpis">
      {items.map((k, i) => (
        <Revelar key={k.nombre} className="kpi" orden={i}>
          <p className="kpi-valor"><Contador valor={k.valor} decimales={k.d} /></p>
          <p className="kpi-nombre">{k.nombre}</p>
          <p className="kpi-nota">{k.nota}</p>
        </Revelar>
      ))}
    </div>
  )
}

function ValidacionCruzada({ r, g }) {
  const n = r.cv_p02.normal, w = r.cv_p02.weighted, par = r.cv_p02.pareada
  const datos = []
  n.folds.forEach((f, i) => datos.push({ x: i + 0.8, y: f.exterior, color: COLOR.train }))
  w?.folds.forEach((f, i) => datos.push({ x: i + 1.2, y: f.exterior, color: COLOR.valid }))
  return (
    <>
      <div className="dos-columnas">
        <Figura titulo="AUC de pacientes no vistas, fold a fold"
          leyenda={[{ texto: `BCE normal (conjunta ${fmt(n.auc, 3)})`, color: COLOR.train }, ...(w ? [{ texto: `BCE ponderada (conjunta ${fmt(w.auc, 3)})`, color: COLOR.valid }] : []),
            { texto: 'Azar', color: COLOR.azar, tipo: 'discontinua' }]}
          explicacion={<p>Cada fold deja fuera 219 pacientes que no intervienen ni en el entrenamiento ni en la elección de época y umbral. Las cinco barras de cada color suman las {n.pacientes} pacientes de entrenamiento: es la medida honesta de la receta.</p>}>
          <Grafico x={[0.4, 5.6]} y={[0.4, 0.65]} ticksX={[1, 2, 3, 4, 5]} ticksY={[0.4, 0.45, 0.5, 0.55, 0.6, 0.65]} fmtX={(v) => `fold ${v - 1}`} fmtY={(v) => fmt(v, 2)}
            etiquetaX="" etiquetaY="AUC exterior" descripcion="AUC exterior por fold con BCE normal y ponderada">
            <Horizontal y={0.5} etiqueta="azar" lado="izquierda" />
            <Barras datos={datos.map((d) => ({ ...d, y: Math.max(d.y, 0.4) }))} ancho={0.36} />
            {datos.map((d, i) => <Texto key={i} x={d.x} y={d.y} dy={-6} clase="valor-grafico pequeno">{fmt(d.y, 2)}</Texto>)}
          </Grafico>
        </Figura>
        <div className="texto-columna">
          <h3 className="subtitulo">Normal frente a ponderada</h3>
          {par && w ? (
            <p>
              Sobre las mismas {par.n} pacientes, la BCE ponderada (pos_weight = N0/N1) se queda en AUC {fmt(w.auc, 3)}, el nivel
              del azar, y la normal en {fmt(n.auc, 3)}. La diferencia es {fmt(par.diferencia_auc_normal_menos_ponderada, 3)}, con un
              intervalo bootstrap pareado de {fmt(par.ic95[0], 3)} a {fmt(par.ic95[1], 3)}: incluye el 0, así que no hay diferencia
              demostrada.
            </p>
          ) : <p>La comparación aparecerá al completar los 5 folds de la BCE ponderada.</p>}
          <table className="tabla compacta">
            <thead><tr><th>Fold</th><th className="num">Época</th><th className="num">Interna</th><th className="num">Exterior</th>{w && <th className="num">Ponderada</th>}</tr></thead>
            <tbody>
              {n.folds.map((f, i) => (
                <tr key={f.fold}><td>{f.fold}</td><td className="num">{f.epoca}</td><td className="num">{fmt(f.interna, 3)}</td>
                  <td className="num">{fmt(f.exterior, 3)}</td>{w && <td className="num">{fmt(w.folds[i].exterior, 3)}</td>}</tr>
              ))}
              <tr className="total"><td>Conjunta</td><td /><td className="num">{fmt(n.interna_media, 3)}</td><td className="num">{fmt(n.auc, 3)}</td>{w && <td className="num">{fmt(w.auc, 3)}</td>}</tr>
            </tbody>
          </table>
          <p className="nota-pie">Interna: AUC de la validación interna en la época elegida (optimista). Exterior: pacientes no vistas.</p>
        </div>
      </div>

      {g.p02.normal && (
        <div className="dos-columnas">
          {['normal', 'weighted'].filter((k) => g.p02[k]).map((k) => (
            <Figura key={k} titulo={`AUC por época, 5 folds · BCE ${k === 'normal' ? 'normal' : 'ponderada'}`}
              leyenda={[{ texto: 'Entrenamiento (FIT)', color: COLOR.train }, { texto: 'Validación interna', color: COLOR.valid }, { texto: 'Buen modelo teórico', color: COLOR.ref, tipo: 'puntos' }]}
              explicacion={<p>{k === 'normal'
                ? 'Con lotes de 10 pacientes y aumento sincronizado (P02), la red sí aprende: el AUC de entrenamiento sube con las épocas. El de validación no lo acompaña, y en tres folds la validación eligió la época 1 o 2.'
                : 'Con la pérdida ponderada pasa lo mismo. En dos folds la validación interna eligió una de las tres primeras épocas, antes de que la red aprendiera casi nada.'}</p>}>
              <Grafico x={[1, 30]} y={[0.4, 0.9]} ticksX={[1, 5, 10, 15, 20, 25, 30]} ticksY={[0.4, 0.5, 0.6, 0.7, 0.8, 0.9]} fmtX={(v) => v}
                etiquetaX="Época" etiquetaY="AUC por paciente" descripcion={`AUC por época de los 5 folds con BCE ${k}`}>
                <Horizontal y={0.8} color={COLOR.ref} tipo="puntos" />
                <Horizontal y={0.5} />
                {g.p02[k].map((f, i) => <Linea key={`t${i}`} datos={f.historia.map((h) => [h.epoch, h.fit_auc])} color={COLOR.train} grosor={1.6} retraso={i * 0.08} />)}
                {g.p02[k].map((f, i) => <Linea key={`v${i}`} datos={f.historia.map((h) => [h.epoch, h.inner_auc])} color={COLOR.valid} grosor={1.6} retraso={0.3 + i * 0.08} />)}
              </Grafico>
            </Figura>
          ))}
        </div>
      )}
    </>
  )
}

const REGLAS = (n) => [
  ['Manda el AUC exterior conjunto de los 5 folds', `BCE normal ${fmt(n, 3)}`],
  ['BCE normal salvo que la ponderada sea claramente mejor', 'El intervalo pareado incluye 0: normal'],
  ['P02 sustituye a E57 solo si su AUC conjunta llega a 0,55', `${fmt(n, 3)} < 0,55: se mantiene E57`],
  ['El test se evalúa una sola vez', 'E57 ya tenía su evaluación: no se repite'],
]

const FIGURAS_NOTEBOOK = [
  ['curvas_normal.png', 'Curvas de los 5 folds, BCE normal'],
  ['curvas_ponderada.png', 'Curvas de los 5 folds, BCE ponderada'],
  ['matrices.png', 'Matrices de confusión con tres umbrales'],
  ['roc_pr.png', 'ROC y precisión–recall de la validación cruzada'],
  ['cohortes_folds.png', 'AUC por cohorte y por corte'],
  ['calibracion.png', 'Calibración de la validación cruzada'],
]

export function Resultados() {
  const r = useResumen(), g = useGraficas()
  if (r.estado === 'error' || g.estado === 'error') return <section className="papel-seccion"><h1 className="titulo-pagina">Resultados</h1><p>Faltan los datos: genera python -m src.web_resultados y python -m src.web_graficas.</p></section>
  if (r.estado !== 'listo' || g.estado !== 'listo') return <section className="papel-seccion"><h1 className="titulo-pagina">Resultados</h1><p>Cargando…</p></section>

  const t = r.test_e57, normal = r.cv_p02.normal, ponderada = r.cv_p02.weighted
  const filas = [
    { nombre: 'Pico de validación de E57', detalle: 'fold 0; la época se eligió mirando estas mismas pacientes', valor: r.picos_e57.fold0, tipo: 'optimista' },
    { nombre: 'Media de los picos de E57', detalle: '5 folds; misma forma de elegir', valor: r.picos_e57.media, tipo: 'optimista' },
    normal && { nombre: 'Validación cruzada P02, BCE normal', detalle: `${normal.pacientes} pacientes no vistas`, valor: normal.auc, ic: normal.ic95, tipo: 'honesta' },
    ponderada && { nombre: 'Validación cruzada P02, BCE ponderada', detalle: `las mismas ${ponderada.pacientes} pacientes`, valor: ponderada.auc, ic: ponderada.ic95, tipo: 'honesta' },
    { nombre: 'Test público, E57', detalle: `${t.pacientes} pacientes, evaluado una sola vez`, valor: t.auc, ic: t.ic95, tipo: 'final' },
  ].filter(Boolean)

  return (
    <div className="pagina-resultados">
      <section className="papel-seccion">
        <h1 className="titulo-pagina">Resultados</h1>
        <p className="entradilla">
          El modelo entregado es E57, evaluado una sola vez en las {t.pacientes} pacientes del test público con el umbral
          y la agregación fijados antes. Estas son sus métricas por paciente, junto a cómo se han medido y qué se comparó.
        </p>
        <Kpis t={t} g={g} />
        <p className="nota-pie">El AUC por corte importa para la demostración: en la defensa se evalúa un corte suelto por paciente, no la media de sus cortes.</p>
      </section>

      <section className="papel-seccion">
        <h2 className="titulo-seccion">Cómo se ha medido</h2>
        <p className="parrafo">
          El mismo tipo de modelo da cifras muy distintas según con qué pacientes se mida. Si la época se elige mirando a esas
          mismas pacientes, el resultado sale inflado; con pacientes que no intervinieron en ninguna decisión, las cifras
          coinciden entre sí. Las barras horizontales son el intervalo de confianza del 95 %.
        </p>
        <Revelar><EscalaMedidas filas={filas} referencia={r.referencia.auc} /></Revelar>
      </section>

      <section className="papel-seccion">
        <h2 className="titulo-seccion">Validación cruzada de 5 folds (P02)</h2>
        <p className="parrafo">
          La receta corregida (P02: 10 pacientes por lote y aumento geométrico sincronizado) se evaluó en los 5 folds del
          profesor, una vez con cada pérdida, en la GPU de la universidad. Es la comparación BCE normal frente a ponderada que
          pide el enunciado.
        </p>
        <ValidacionCruzada r={r} g={g} />
      </section>

      {normal && (
        <section className="papel-seccion">
          <h2 className="titulo-seccion">Por cohorte</h2>
          <div className="dos-columnas">
            <Revelar><Cohortes cohortes={normal.cohortes} /></Revelar>
            <p className="parrafo">
              AUC de la validación cruzada (BCE normal) separada por cohorte. En Duke el modelo ordena al revés: sus imágenes
              se adquirieron con otro protocolo y sus cortes se seleccionaron con otro criterio. Lo que la red aprende en I-SPY
              no se traslada, lo que indica que parte de la señal depende de la cohorte y no solo de la biología.
            </p>
          </div>
        </section>
      )}

      <section className="papel-seccion">
        <h2 className="titulo-seccion">Decisión de entrega</h2>
        <p className="parrafo">
          Las reglas para elegir el modelo final se fijaron el 20 de octubre, antes de ver la validación cruzada, y quedaron
          registradas en el repositorio (commit 830ccd2).
        </p>
        <ol className="reglas">
          {REGLAS(normal?.auc ?? 0).map(([regla, resultado], i) => (
            <Revelar as="li" key={regla} orden={i}><span className="regla">{regla}</span><span className="regla-resultado">{resultado}</span></Revelar>
          ))}
        </ol>
        <p className="parrafo"><strong>Resultado: se entrega E57</strong>, con su umbral 0,325 y la media de probabilidades de los cortes de cada paciente.</p>
      </section>

      <section className="papel-seccion">
        <h2 className="titulo-seccion">Conclusiones</h2>
        <ol className="conclusiones">
          <Revelar as="li" orden={0}><strong>El modelo inicial no aprendía.</strong> Con lotes de 2 pacientes, BatchNorm impedía aprender diferencias entre pacientes; con 10 por lote, la red aprende.</Revelar>
          <Revelar as="li" orden={1}><strong>Aprender no es generalizar.</strong> Lo que la red aprende se traslada poco a pacientes nuevas, incluso con aumento de datos; la BCE ponderada no lo mejora.</Revelar>
          <Revelar as="li" orden={2}><strong>La cohorte pesa.</strong> En Duke el modelo ordena al revés: con solo imagen y 1.097 pacientes de tres cohortes, la señal es débil y en parte específica de cada cohorte.</Revelar>
          <Revelar as="li" orden={3}><strong>La medida honesta coincide.</strong> El pico de validación (0,67) prometía más que el test (0,57); la validación cruzada (0,53) lo anticipaba.</Revelar>
        </ol>
      </section>

      <section className="papel-seccion">
        <h2 className="titulo-seccion">Figuras del notebook</h2>
        <p className="parrafo">Las mismas figuras que en el notebook de resultados del repositorio. Se abren a tamaño completo.</p>
        <div className="galeria">
          {FIGURAS_NOTEBOOK.map(([f, t], i) => (
            <Revelar as="a" key={f} orden={i} className="galeria-item" href={`/resultados/${f}`} target="_blank" rel="noreferrer">
              <img src={`/resultados/${f}`} alt={t} loading="lazy" />
              <span>{t}</span>
            </Revelar>
          ))}
        </div>
      </section>
    </div>
  )
}
