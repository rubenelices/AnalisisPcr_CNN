"""La vista activa conserva notas/ejecuciones futuras y los históricos originales."""
import hashlib

import nbformat
import pytest

from src.analysis.new_approach_results import MARKER, build, write_notebook


def test_refresh_preserves_user_cells_outputs_and_source(tmp_path):
    source_dir = tmp_path / 'experiments/E57_patient_feature_mean_mac/results'
    dest = tmp_path / 'research'
    source_dir.mkdir(parents=True)
    dest.mkdir()
    (dest / 'ESTADO.md').write_text('Estado original')
    source = source_dir / 'RESULTADOS.ipynb'
    reference = nbformat.v4.new_notebook(cells=[
        nbformat.v4.new_markdown_cell('[Test](TEST.md) · [Guía](../breastdcedl/GUIA.md)'),
        nbformat.v4.new_code_cell('# Figura guardada', outputs=[
            nbformat.v4.new_output('display_data', data={'text/plain': 'Figura existente'})]),
    ])
    nbformat.write(reference, source)
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    path = build(tmp_path)
    notebook = nbformat.read(path, 4)
    note = nbformat.v4.new_markdown_cell('Conclusión y métricas de una ejecución posterior')
    note.metadata['new_approach_run'] = 'N01'
    notebook.cells.insert(3, note)
    nbformat.write(notebook, path)
    (dest / 'ESTADO.md').write_text('Estado actualizado')
    build(tmp_path)
    result = nbformat.read(path, 4)
    nbformat.validate(result)
    assert [c for c in result.cells if MARKER not in c.metadata] == [note]
    assert 'Estado actualizado' == result.cells[1].source
    assert result.cells[-2].source == '[Test](../experiments/E57_patient_feature_mean_mac/results/TEST.md) · [Guía](../breastdcedl/GUIA.md)'
    assert result.cells[-1].outputs == reference.cells[-1].outputs
    assert hashlib.sha256(source.read_bytes()).hexdigest() == before
    # Una segunda actualización no duplica la referencia ni las notas.
    build(tmp_path)
    assert nbformat.read(path, 4) == result


def test_failed_notebook_write_preserves_previous_file_and_cleans_temp(tmp_path, monkeypatch):
    target = tmp_path/'RESULTADOS.ipynb'
    original = nbformat.v4.new_notebook(cells=[nbformat.v4.new_markdown_cell('Nota que debe sobrevivir')])
    nbformat.write(original, target)
    before = target.read_bytes()
    def interrupted(notebook, stream):
        stream.write('archivo incompleto')
        raise OSError('Interrupción simulada durante escritura')
    monkeypatch.setattr(nbformat, 'write', interrupted)
    with pytest.raises(OSError): write_notebook(original, target)
    assert target.read_bytes() == before
    assert list(tmp_path.iterdir()) == [target]
