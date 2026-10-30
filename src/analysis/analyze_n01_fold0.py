"""Diagnóstico descriptivo ligero de N01: CSV guardados, sin pesos ni imágenes.

No busca umbrales, agregaciones, calibraciones o recetas. No accede a test ni
modifica artefactos científicos. Añade una lectura reproducible de la pareja0.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.metrics import binary_metrics

ROOT = Path(__file__).resolve().parents[2]
RUNS = Path('reports/new_approach/N01_e57_bce/runs')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_pair(root=ROOT):
    root = Path(root)
    metadata = pd.read_csv(root/'breastdcedl/metadata/patients.csv', usecols=['pid', 'dataset', 'split'])
    metadata = metadata.loc[metadata.split.eq('train'), ['pid', 'dataset']]
    result, inputs = {}, {}
    for loss in ('normal', 'weighted'):
        folder = root/RUNS/f'{loss}_fold_0_seed_42'
        paths = [folder/n for n in ('summary.json', 'predictions_outer_patient.csv', 'predictions_outer_cut.csv')]
        inputs.update({str(p.relative_to(root)): sha(p) for p in paths})
        s = json.loads(paths[0].read_text())
        if (s.get('status') != 'complete' or s.get('fold') != 0 or s.get('loss') != loss
                or s.get('scientific_result') is not True or s.get('test_evaluated') is not False):
            raise ValueError('El diagnóstico requiere la pareja científica0 completa y sin test')
        p = pd.read_csv(paths[1])
        if p.patient_id.duplicated().any() or len(p) != s['counts']['outer']['patients']:
            raise ValueError('Una fila por paciente exterior requerida')
        binary_metrics(p.target, p.probability, s['threshold'])
        cuts = pd.read_csv(paths[2])
        if cuts.groupby('patient_id').target.nunique().ne(1).any():
            raise ValueError('Etiquetas contradictorias entre cortes')
        averaged = cuts.groupby('patient_id').agg(probability=('probability', 'mean'), target=('target', 'first'))
        saved = p.set_index('patient_id').sort_index()
        if (set(averaged.index) != set(saved.index)
                or not np.array_equal(averaged.loc[saved.index].target, saved.target)
                or not np.allclose(averaged.loc[saved.index].probability, saved.probability, atol=1e-12, rtol=0)):
            raise ValueError('La media por paciente no reproduce las predicciones guardadas')
        p = p.merge(metadata, left_on='patient_id', right_on='pid', validate='one_to_one', how='left')
        if p.dataset.isna().any():
            raise ValueError('Paciente fuera del train público o sin cohorte')
        p['predicted'] = p.probability.ge(s['threshold'])
        p['correct'] = p.predicted.eq(p.target.astype(bool))
        m = binary_metrics(p.target, p.probability, s['threshold'])
        expected = s['outer_at_inner_threshold']
        if (m['confusion_matrix'] != expected['confusion_matrix']
                or any(not np.isclose(m[k], expected[k], atol=1e-12, rtol=0)
                       for k in ('roc_auc', 'pr_auc', 'brier_score'))):
            raise ValueError('Matriz guardada incompatible con la decisión congelada')
        result[loss] = (s, p)
    a, b = [result[n][0] for n in ('normal', 'weighted')]
    if (a['protocol_sha256'] != b['protocol_sha256']
            or a['initial_state_sha256'] != b['initial_state_sha256'] or a['counts'] != b['counts']):
        raise ValueError('La pareja no comparte protocolo, inicialización y particiones')
    inputs['breastdcedl/metadata/patients.csv'] = sha(root/'breastdcedl/metadata/patients.csv')
    return result, inputs


def paired_errors(a, b):
    if a.patient_id.duplicated().any() or b.patient_id.duplicated().any():
        raise ValueError('Pacientes repetidas')
    a, b = a.set_index('patient_id').sort_index(), b.set_index('patient_id').sort_index()
    if not a.index.equals(b.index) or not np.array_equal(a.target, b.target):
        raise ValueError('Pacientes/etiquetas distintas en la pareja')
    rows = []
    for label, select in [('Todas', np.ones(len(a), dtype=bool)),
                          ('pCR=0', a.target.eq(0)), ('pCR=1', a.target.eq(1))]:
        x, y = a.loc[select, 'correct'], b.loc[select, 'correct']
        rows.append(dict(Grupo=label, Ambas_bien=int((x & y).sum()),
                         Solo_normal_bien=int((x & ~y).sum()), Solo_ponderada_bien=int((~x & y).sum()),
                         Ambas_mal=int((~x & ~y).sum())))
    return rows


def summarize(pair):
    distributions, near, cohorts, calibration = [], [], [], []
    for loss, (s, p) in pair.items():
        for target, group in p.groupby('target'):
            v = group.probability
            distributions.append(dict(Pérdida=loss, Clase=int(target), Pacientes=len(group),
                                      Media=float(v.mean()), Mediana=float(v.median()),
                                      Mínimo=float(v.min()), Máximo=float(v.max())))
        near.append(dict(Pérdida=loss, Umbral=s['threshold'],
                         A_menos_de_0005=int((abs(p.probability-s['threshold'])<=.005).sum()),
                         A_menos_de_001=int((abs(p.probability-s['threshold'])<=.01).sum()),
                         Positivas_a_05=int(p.probability.ge(.5).sum()),
                         Positivas_al_umbral_interno=int(p.predicted.sum())))
        prior = s['counts']['fit']['positive']/s['counts']['fit']['patients']
        calibration.append(dict(Pérdida=loss, Probabilidad_media=float(p.probability.mean()),
                                Frecuencia_pCR=float(p.target.mean()),
                                Brier=binary_metrics(p.target, p.probability)['brier_score'],
                                Brier_constante_FIT=float(np.mean((p.target-prior)**2))))
        for cohort, group in p.groupby('dataset'):
            m = binary_metrics(group.target, group.probability, s['threshold'])
            cohorts.append(dict(Pérdida=loss, Cohorte=cohort, Pacientes=len(group), pCR=int(group.target.sum()),
                                AUC=m['roc_auc'], AP=m['pr_auc'], BA=m['balanced_accuracy'],
                                Sensibilidad=m['sensitivity'], Especificidad=m['specificity'],
                                **{k.upper(): v for k, v in m['confusion_matrix'].items()}))
    return dict(distributions=distributions, near_threshold=near, calibration=calibration, cohorts=cohorts,
                paired_errors=paired_errors(pair['normal'][1], pair['weighted'][1]))


def build(root=ROOT):
    root = Path(root)
    pair, inputs = load_pair(root)
    analysis = summarize(pair)
    out = root/'reports/new_approach/N01_e57_bce/diagnostics_fold0'
    out.mkdir(exist_ok=True)
    (out/'analysis.json').write_text(json.dumps(dict(scope='descriptive_outer_development_fold0_only',
        inputs_sha256=inputs, analysis_source_sha256=sha(Path(__file__)),
        no_model_loading=True, no_images_read=True, test_evaluated=False,
        threshold_selection_performed=False, analysis=analysis), indent=2, ensure_ascii=False, allow_nan=False))
    def table(name):
        frame = pd.DataFrame(analysis[name]).round(5)
        if 'Pérdida' in frame:
            frame['Pérdida'] = frame['Pérdida'].replace({'normal': 'Normal', 'weighted': 'Ponderada'})
        frame = frame.rename(columns={
            'A_menos_de_0005': 'Dentro de ±0,005', 'A_menos_de_001': 'Dentro de ±0,01',
            'Positivas_a_05': 'Positivas a 0,5', 'Positivas_al_umbral_interno': 'Positivas al umbral interno',
            'Probabilidad_media': 'Probabilidad media', 'Frecuencia_pCR': 'Frecuencia pCR',
            'Brier_constante_FIT': 'Brier constante FIT', 'Ambas_bien': 'Ambas bien',
            'Solo_normal_bien': 'Solo normal bien', 'Solo_ponderada_bien': 'Solo ponderada bien',
            'Ambas_mal': 'Ambas mal'})
        return frame.to_markdown(index=False)
    text = '''# Diagnóstico ligero de N01 · primera pareja

Análisis descriptivo de las 219 pacientes exteriores del fold 0, utilizando exclusivamente predicciones guardadas. No se cargan pesos ni imágenes, no se entrena y no se consulta test. Los umbrales se conservan tal como se eligieron en la interna. Este análisis no se utiliza para ajustarlos.

## Separación de las clases

''' + table('distributions') + '''

Las distribuciones se solapan mucho. La media positiva es algo mayor, pero la diferencia es pequeña. La ponderada desplaza las probabilidades hacia 0,5 y separa poco las clases. Mover un umbral no corrige la ordenación débil que refleja el AUC. Estas probabilidades son salidas del modelo y no deben interpretarse como riesgos clínicos calibrados.

## Sensibilidad a cambios pequeños del umbral

Se cuentan las bandas descriptivas ±0,005 y ±0,01 alrededor del umbral congelado. No son nuevos criterios de selección ni se exploran umbrales alternativos buscando el mejor resultado exterior.

''' + table('near_threshold') + '''

En la ponderada, 98 de 219 pacientes están dentro de ±0,005 del umbral interno. Pasar de 0,5 al umbral interno ya fijado cambia 27 predicciones. Esto ayuda a explicar por qué una matriz puede variar mucho con un cambio numéricamente pequeño. No permite escoger ahora el corte que dé mejor matriz exterior.

## Probabilidades y calibración

''' + table('calibration') + '''

El predictor constante utiliza la prevalencia de FIT, no la del exterior. La normal mejora muy ligeramente su Brier; la ponderada es peor. La ponderación cambia el objetivo y puede alterar la calibración. No se ajusta ninguna calibración con estas 219 pacientes. Una transformación estrictamente creciente de los scores conserva su ordenación y por tanto su ROC-AUC; no resolvería por sí sola la discriminación baja.

## ¿Se equivocan con las mismas pacientes?

''' + table('paired_errors') + '''

Las dos aciertan 91 pacientes y fallan 43. La normal acierta 48 que la ponderada falla; la ponderada acierta 37 que la normal falla. Hay errores diferentes, pero eso no demuestra que una combinación mejore ni justifica ajustar un ensemble con este exterior. Las diferencias dependen también de los umbrales internos distintos.

## Resultado por cohorte

''' + table('cohorts') + '''

Las submuestras son pequeñas, especialmente I-SPY1 (21 pacientes). Estas cifras sirven para describir variabilidad, no para elegir cohortes favorables, eliminar pacientes o declarar causas. No se añaden edad, menopausia ni otros datos clínicos al modelo.

## Qué queda aprendido y qué no

La prioridad es conseguir una separación de clases más estable y una evaluación coherente, además de una matriz útil. Las curvas juntas no bastan: también pueden aparecer cuando el modelo discrimina poco. El fold 0 favorece descriptivamente a la normal, pero no confirma que ocurra en otros folds ni permite atribuir el resultado a una causa única.

Se conserva la arquitectura y la receta N01. No se cambian pooling, aumentos, umbral, calibración o variables a partir de este diagnóstico. Cuando vuelva a ser posible entrenar, la pareja del fold 1 permitirá comprobar si el comportamiento se repite. Si se cierra aquí, el resultado se reportará como comparación de un solo fold.

Fuentes: [revisión de la pareja](REVISION_N01_FOLD0.md), [protocolo fijado](PROTOCOLO_N01.md) y [resultados completos](RESULTADOS.ipynb). Respaldo reproducible en `reports/new_approach/N01_e57_bce/diagnostics_fold0/`; regeneración: `python -m src.analysis.analyze_n01_fold0`.

Uso educativo, sin validez clínica.
'''
    target = root/'research/DIAGNOSTICO_N01_FOLD0.md'
    target.write_text(text)
    notebook_path = root/'research/RESULTADOS.ipynb'
    if notebook_path.is_file():
        import nbformat
        from src.analysis.new_approach_results import write_notebook
        nb = nbformat.read(notebook_path, 4)
        nb.cells = [c for c in nb.cells if not c.metadata.get('n01_saved_scores_analysis')]
        cell = nbformat.v4.new_markdown_cell(text)
        cell.metadata.update(n01_saved_scores_analysis=True, new_approach_run='N01_fold0_diagnostic')
        position = next((i+1 for i, c in enumerate(nb.cells) if c.metadata.get('n01_fold0_review')),
                        next(i for i, c in enumerate(nb.cells)
                             if c.metadata.get('new_approach_generated') == 'reference'))
        nb.cells.insert(position, cell)
        nbformat.validate(nb)
        write_notebook(nb, notebook_path)
    return target


if __name__ == '__main__':
    print(build())
