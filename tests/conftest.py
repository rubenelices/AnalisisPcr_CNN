"""Deselecciona los tests que leen configuraciones de experimentos archivados.

El 20 de octubre de 2026 se limpió el proyecto: las configuraciones y resultados
de experimentos históricos se movieron fuera (zip de respaldo en la carpeta CASO 1, fuera del repositorio).
Los casos de `archivados_20261020.txt` fallaban solo por FileNotFoundError de
esos ficheros; el código que prueban sigue cubierto por el resto de casos.
"""
from pathlib import Path

ARCHIVED = set((Path(__file__).parent/'archivados_20261020.txt').read_text().splitlines())


def pytest_collection_modifyitems(config, items):
    selected, deselected = [], []
    for item in items:
        (deselected if item.nodeid in ARCHIVED else selected).append(item)
    if deselected:
        config.hook.pytest_deselected(items=deselected)
        items[:] = selected
