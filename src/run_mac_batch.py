"""Tanda E32–E41: vista previa por defecto, entrenamiento local solo con --execute.

No descarga modelos/datos ni reanuda o sobrescribe ejecuciones parciales.
Cada hijo usa el mismo Python que esta cola, con un único entrenamiento activo.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from src.experiment_config import load_experiment_config
from src.run_night import execution_state

ROOT = Path(__file__).resolve().parents[1]
BASE = "E19_weight_decay_001_mac"
# Las claves describen diferencias de configuraciones RESUELTAS, no solo de JSON.
SPECS = (
    ("E32_three_blocks_mac", BASE, {"model.channels": [16, 32, 64]},
     "Tres bloques: menos capacidad, último mapa 64×32×32; riesgo de perder patrones complejos."),
    ("E33_one_conv_mac", BASE,
     {"model.convolutions_per_block": 1, "model.pool_position": "after_convolutions"},
     "Una convolución por bloque y luego pool. Dos campos ligados: no puede haber pooling entre dos convoluciones si solo queda una. Menos capacidad y campo receptivo."),
    ("E34_kernel5_compact_mac", "E23_compact", {"model.kernel_size": 5},
     "Kernel 5×5, padding 2, sobre la compacta E23. Mayor contexto local y más parámetros; puede aumentar sobreajuste. Comparar con E23, no atribuir E34−E19 solo al kernel."),
    ("E35_batch32_mac", BASE, {"data.batch_size": 32},
     "Batch 32: otras estadísticas BatchNorm y menos actualizaciones por época; no implica mayor AUC."),
    ("E36_batch64_mac", BASE, {"data.batch_size": 64},
     "Batch 64: gradiente de más cortes y menos pasos. Mayor consumo de memoria unificada; no reducir automáticamente el batch ni desactivar límites MPS."),
    ("E37_rotation5_mac", BASE,
     {"data.augmentation.enabled": True,
      "data.augmentation.max_translation_fraction": 0.0},
     "Solo giro ±5°, sin traslación. Una rejilla bilineal compartida por PRE/EARLY/LATE, solo train; relleno cero e interpolación pueden perder detalle."),
    ("E38_spatial_dropout010_mac", BASE, {"model.spatial_dropout": 0.1},
     "Dropout2d 0,1 al final de cada bloque: anula mapas completos solo al entrenar. Mantiene dropout final 0,3. Puede reducir coadaptación o causar infraajuste."),
    ("E39_dropout_affine_mac", "E38_spatial_dropout010_mac",
     {"data.augmentation.enabled": True,
      "data.augmentation.max_translation_fraction": 0.02},
     "Sobre E38, añade giro ±5° y traslación ±2% sincronizados. Una política geométrica como variable principal; frente a E19 combina DOS decisiones. Comparar también con E38."),
    ("E40_lr0003_mac", BASE, {"training.learning_rate": 0.0003},
     "LR inicial 0,0003 con el mismo Plateau: actualizaciones menores podrían estabilizar entrenamiento, pero quizá necesiten más pasos; mismo presupuesto no asegura convergencia."),
    ("E41_pool_after_mac", BASE, {"model.pool_position": "after_convolutions"},
     "Dos convoluciones antes de cada pool: conserva parámetros y cambia campo receptivo/coste. La segunda convolución opera a mayor resolución, por lo que puede tardar más."),
)
EXPERIMENTS = tuple(spec[0] for spec in SPECS)
MANIFEST = "configs/mac_batch_E32_E41_manifest.json"


def flatten(value: dict, prefix: str = "") -> dict:
    result = {}
    for key, item in value.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(item, dict):
            result.update(flatten(item, name))
        else:
            result[name] = list(item) if isinstance(item, tuple) else item
    return result


def changed_settings(parent, candidate) -> dict:
    left = flatten({key: asdict(getattr(parent, key)) for key in ("data", "model", "training")})
    right = flatten({key: asdict(getattr(candidate, key)) for key in ("data", "model", "training")})
    return {key: value for key, value in right.items() if value != left[key]}


def validate_plan(root: Path) -> dict:
    """Valida TODA la tanda antes de leer imágenes o crear un entrenamiento."""
    manifest = json.loads((root / MANIFEST).read_text(encoding="utf-8"))
    if (manifest.get("schema_version") != 1 or manifest.get("test_evaluated") is not False
            or [row["experiment_id"] for row in manifest["experiments"]] != list(EXPERIMENTS)):
        raise ValueError("Manifiesto de tanda incompatible")
    configs = {}
    for (name, parent_name, changes, _), row in zip(SPECS, manifest["experiments"]):
        path = root / f"configs/experiments/{name}.json"
        parent_path = root / f"configs/experiments/{parent_name}.json"
        expected_paths = {"config": path, "parent": parent_path}
        if row.get("parent_id") != parent_name or row.get("changes") != changes:
            raise ValueError(f"Protocolo cambiado: {name}")
        for label, file in expected_paths.items():
            if hashlib.sha256(file.read_bytes()).hexdigest() != row[f"{label}_sha256"]:
                raise ValueError(f"Configuración modificada después del plan: {name}/{label}")
        candidate, parent = map(load_experiment_config, (path, parent_path))
        if changed_settings(parent, candidate) != changes:
            raise ValueError(f"Cambios no previstos: {name}")
        if (candidate.experiment_id != name
                or candidate.output_dir != f"reports/experiments/{name}/fold_0_seed_42"
                or candidate.checkpoint_dir != f"checkpoints/{name}/fold_0_seed_42"
                or candidate.data.validation_fold != 0 or candidate.training.seed != 42
                or candidate.data.train_patient_fraction != 1
                or candidate.data.train_patients_per_class is not None
                or candidate.data.validation_patients_per_class is not None
                or candidate.training.epochs != 50 or candidate.training.early_stopping_patience != 10
                or candidate.data.num_workers != 0 or candidate.training.mixed_precision):
            raise ValueError(f"Rutas, dispositivo o presupuesto incompatible: {name}")
        configs[name] = (path, candidate)
    return configs


def run_child(command: list[str], root: Path, log_path: Path) -> int:
    # 'x' evita truncar incluso un log huérfano de una ejecución interrumpida.
    with log_path.open("x", encoding="utf-8") as log:
        child = subprocess.Popen(command, cwd=root, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, text=True, encoding="utf-8")
        try:
            for line in child.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            return child.wait()
        except BaseException:
            child.terminate()
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
            raise


def require_device(requested: str):
    import torch
    if requested == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        raise RuntimeError("MPS no disponible en este entorno Python. No se sustituye por CPU.")
    return torch.device(requested)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    parser.add_argument("--only", nargs="+", choices=tuple(name[:3] for name in EXPERIMENTS),
                        help="Elegir, por ejemplo, --only E32 E33 E37 E38 E40")
    args = parser.parse_args(argv)
    selected = args.only or [name[:3] for name in EXPERIMENTS]
    if len(selected) != len(set(selected)):
        parser.error("No repetir experimentos en la cola")
    configs = validate_plan(ROOT)
    log_dir = ROOT / "reports/mac_batch_E32_E41"
    plans = []
    for short in selected:
        name = next(name for name in EXPERIMENTS if name[:3] == short)
        path, config = configs[name]
        state = execution_state(ROOT, path)
        if state == "pending" and (log_dir / f"{name}.log").exists():
            raise FileExistsError(f"Log existente de {name}; revisar, no sobrescribir")
        print(f"{name}: {state}; batch {config.data.batch_size}; máximo 50 épocas / paciencia 10", flush=True)
        plans.append((name, path, state))
    if not args.execute:
        print("VISTA PREVIA: no se han leído imágenes ni entrenado. Añade --execute para empezar.")
        return 0
    require_device(args.device)  # Fallar antes de escribir o cargar datos.
    log_dir.mkdir(parents=True, exist_ok=True)
    lock = log_dir / "active.lock"
    with lock.open("x", encoding="utf-8") as stream:
        stream.write("Cola activa. Tras un cierre abrupto, comprobar procesos antes de retirar el bloqueo.\n")
    try:
        for name, path, state in plans:
            if state == "complete":
                print(f"Omitido: {name} ya completo, sin reentrenar.", flush=True)
                continue
            # Revalidar también el manifiesto por si se editó mientras otro hijo entrenaba.
            validate_plan(ROOT)
            if execution_state(ROOT, path) != "pending":
                raise RuntimeError(f"Estado cambiado: {name}; no sobrescribir")
            command = [sys.executable, "-u", "-m", "src.train", "--config", str(path), "--device", args.device]
            code = run_child(command, ROOT, log_dir / f"{name}.log")
            if code != 0:
                print(f"Cola detenida: {name} terminó con error {code}. Los resultados anteriores se conservan.")
                return code
            if execution_state(ROOT, path) != "complete":
                raise RuntimeError(f"{name}: no produjo artefactos científicos completos")
        print("Tanda terminada. Conservar todos los resultados; test cerrado.")
    finally:
        lock.unlink()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
