"""N01 · validación cruzada de la receta de E57 (criterios escritos antes: research/N01_CV.md).

Compara las predicciones exteriores de N01 normal (5 folds) con P02 normal y con la línea
base de cohorte, en las 1.097 pacientes. Solo train; nunca toca test.

    python -m src.analysis.analisis_n01_cv       # escribe reports/new_approach/N01_e57_bce/cv/
"""
import json
from pathlib import Path

from src.analysis.analisis_p05 import P02, analizar, pacientes
from src.analysis.analisis_paso0 import metadatos

ROOT = Path(__file__).resolve().parents[2]
N01 = ROOT/'reports/new_approach/N01_e57_bce'
CRITERIO_DOC_SHA = 'ebdfb36b4ed4255ac5080c9b7193de87be9c8f68aabba20840abe2fa8287a7a2'


def renombrar(x):
    """El análisis de P05 llama 'p05' a la receta evaluada; aquí es N01."""
    if isinstance(x, dict):
        return {k.replace('p05', 'n01'): renombrar(v) for k, v in x.items()}
    return [renombrar(v) for v in x] if isinstance(x, list) else x


def main():
    out = N01/'cv'
    out.mkdir(parents=True, exist_ok=True)
    salida, df = analizar(N01/'runs', P02, pacientes(metadatos()))
    salida = renombrar(salida)
    salida.pop('secundario_epoca_fija', None)  # N01 no guarda predicciones por época
    salida.pop('candidato_a_entrega', None)    # N01 no es candidato (ver N01_CV.md)
    salida['criterio_doc_sha256_antes_de_entrenar'] = CRITERIO_DOC_SHA
    salida['criterios'] = {'aprende_mas_alla_de_la_cohorte': salida['criterios']['supera_cohorte']}
    df.rename(columns={'p05': 'n01'}).to_csv(out/'predicciones_exteriores.csv', index=False)
    (out/'resumen.json').write_text(json.dumps(salida, indent=2, ensure_ascii=False))
    print(json.dumps(salida, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
