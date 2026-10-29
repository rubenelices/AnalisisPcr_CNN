"""Codigo fuente del proyecto BreastDCEDL.

Subpaquetes: pipeline (datos y preprocesado), models (arquitecturas), training,
evaluation, analysis (análisis estadísticos), reporting (notebooks y fichas),
web (inferencia y datos de la web) y campaigns (runners de campañas anteriores).
"""
from pathlib import Path


def source_path(name: str) -> Path:
    """Ruta de un módulo del proyecto por su nombre de fichero (p. ej. 'data.py'), en cualquier subpaquete."""
    matches = sorted(Path(__file__).parent.rglob(name))
    if len(matches) != 1:
        raise FileNotFoundError(f'{name}: {len(matches)} coincidencias en src/')
    return matches[0]
