"""Vista de la etapa activa a partir de documentos y salidas ya guardadas.

No importa modelos, carga imágenes del dataset ni ejecuta inferencia. Conserva
las celdas añadidas al notebook para futuros experimentos o notas del estudiante.
"""
import copy
import os
from pathlib import Path
import re
import tempfile

import nbformat

ROOT = Path(__file__).resolve().parents[2]
MARKER = 'new_approach_generated'


def write_notebook(notebook, path):
    """Reemplaza la vista solo tras escribirla por completo; conserva la anterior si falla."""
    path = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile('w', encoding='utf-8', dir=path.parent,
                                         prefix=f'.{path.stem}.', suffix='.ipynb', delete=False) as stream:
            temporary = Path(stream.name)
            nbformat.write(notebook, stream)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _cell(source, section):
    cell = nbformat.v4.new_markdown_cell(source)
    cell.metadata[MARKER] = section
    cell.id = f'new-approach-{section}'
    return cell


def _rebase_links(source):
    """Reubica enlaces Markdown relativos al copiar del dossier E57."""
    pattern = r'(!?\[[^\]]*\]\()([^\s)]+)(\))'

    def replace(match):
        target = match[2]
        if target.startswith(('#', '/', '../')) or re.match(r'[a-zA-Z][\w+.-]*:', target):
            return match[0]
        return f'{match[1]}../experiments/E57_patient_feature_mean_mac/results/{target}{match[3]}'

    return re.sub(pattern, replace, source)


def build(root=ROOT):
    root = Path(root)
    dest = root / 'research'
    reference_path = root / 'experiments/E57_patient_feature_mean_mac/results/RESULTADOS.ipynb'
    state = (dest / 'ESTADO.md').read_text()
    reference = nbformat.read(reference_path, as_version=4)
    nbformat.validate(reference)
    path = dest / 'RESULTADOS.ipynb'
    previous = nbformat.read(path, as_version=4) if path.is_file() else None
    # Solo reemplazamos las secciones que este generador posee explícitamente.
    retained = [copy.deepcopy(c) for c in previous.cells if MARKER not in c.metadata] if previous else []
    cells = [
        _cell('# Nuevo enfoque · resultados y seguimiento\n\n'
              '[Inicio](README.md) · [Plan](PLAN.md) · [Demo](DEMO.md)\n\n'
              'Esta etapa comienza el 20 de octubre de 2026. Las tablas y figuras ya están '
              'mostradas: no hace falta ejecutar el notebook.\n\n'
              '**Referencia: E57, solo imágenes.** La media 0,6224 corresponde a máximos '
              'seleccionados con la misma validación; no es una estimación exterior independiente. '
              'El test 0,5728 pertenece al checkpoint original de época 14.\n\n'
              'El apéndice conserva las 108 épocas de los cinco folds como antecedentes; '
              'no son entrenamientos nuevos ni se han recalculado sus predicciones.', 'intro'),
        _cell(state, 'state'),
        _cell('## Nuevos experimentos de esta etapa\n\n'
              'Consulta el estado de ejecución y las secciones de cada experimento a continuación. '
              'El [protocolo N01](PROTOCOLO_N01.md) define la comparación normal frente a ponderada completa.\n\n'
              'Aquí se incorporarán las hipótesis, protocolos, tablas completas de épocas, '
              'gráficas de pérdida/AUC, matrices y conclusiones de las nuevas ejecuciones. '
              'Las celdas añadidas se conservan al actualizar esta vista.', 'new-runs'),
    ]
    cells.extend(retained)
    cells.append(_cell('## Apéndice · resultados anteriores de E57\n\n'
                       'Copia de consulta del [dossier original](../experiments/E57_patient_feature_mean_mac/results/RESULTADOS.ipynb), '
                       'que permanece intacto. Incluye el único test ya realizado y los cinco folds '
                       'de desarrollo. Los resultados de modelos posteriores no heredan ese test.', 'reference'))
    for index, original in enumerate(reference.cells):
        cell = copy.deepcopy(original)
        cell.metadata[MARKER] = 'reference-e57'
        cell.id = f'reference-e57-{index}'
        if cell.cell_type == 'markdown':
            cell.source = _rebase_links(cell.source)
        cells.append(cell)
    notebook = nbformat.v4.new_notebook(cells=cells)
    notebook.metadata = copy.deepcopy(previous.metadata if previous else reference.metadata)
    notebook.metadata['new_approach'] = {'start_date': '2026-10-20',
                                       'reference': 'experiments/E57_patient_feature_mean_mac/results/RESULTADOS.ipynb',
                                       'mode': 'saved-results-only'}
    nbformat.validate(notebook)
    write_notebook(notebook, path)
    return path


if __name__ == '__main__':
    print(build())
