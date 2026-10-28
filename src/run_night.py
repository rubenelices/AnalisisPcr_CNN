"""Cola secuencial E21-E23. Vista previa por defecto; --execute entrena.

Sin descargas, nube, reanudaciones implícitas ni sobrescritura. Conserva cada
salida completa de terminal en reports/night_queue/<experimento>.log.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

from src.experiment_config import load_experiment_config, experiment_config_from_dict

EXPERIMENTS = ("E21_group_norm", "E22_average_max", "E23_compact")
ROOT = Path(__file__).resolve().parents[1]


def execution_state(root: Path, config_path: Path) -> str:
    config = load_experiment_config(config_path)
    directory = root / config.output_dir
    checkpoints = root / config.checkpoint_dir
    summary_path = directory / "summary.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text())
        resolved = experiment_config_from_dict(json.loads((directory / "config_resolved.json").read_text()))
        if (resolved == config and summary.get("experiment_id") == config.experiment_id
                and summary.get("status") == "complete"
                and summary.get("scientific_result") is True
                and summary.get("test_evaluated") is False
                and (checkpoints / "best.pt").is_file()):
            return "complete"
        raise ValueError(f"Resumen incompatible para {config.experiment_id}; revisar sin sobrescribir")
    if directory.exists() or checkpoints.exists():
        raise FileExistsError(f"Ejecución parcial {config.experiment_id}; revisar antes de continuar")
    return "pending"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--device", choices=("mps", "cuda", "cpu"), default="mps")
    parser.add_argument("--experiments", nargs="+", choices=EXPERIMENTS, default=list(EXPERIMENTS))
    args = parser.parse_args()
    if len(set(args.experiments)) != len(args.experiments):
        parser.error("No repetir un experimento en la cola")
    plans = []
    for name in args.experiments:
        path = ROOT / f"configs/experiments/{name}.json"
        config = load_experiment_config(path)
        state = execution_state(ROOT, path)
        print(f"{name}: {state}; máximo {config.training.epochs} épocas, paciencia {config.training.early_stopping_patience}", flush=True)
        plans.append((name, path, state))
    if not args.execute:
        print("VISTA PREVIA: no se ha entrenado nada. Añade --execute para iniciar.")
        return 0
    from src.train import choose_device
    choose_device(args.device)  # Fallar antes de escribir si el dispositivo no existe.
    log_dir = ROOT / "reports/night_queue"
    log_dir.mkdir(parents=True, exist_ok=True)
    lock = log_dir / "active.lock"
    # El bloqueo evita dos colas simultáneas; no se borra un bloqueo ajeno.
    with lock.open("x") as stream:
        stream.write("Cola activa. Si hubo un cierre abrupto, comprobar procesos antes de retirar este archivo.\n")
    try:
        for name, path, state in plans:
            if state == "complete":
                print(f"Omitido: {name} ya completo; conserva sus resultados.", flush=True)
                continue
            if execution_state(ROOT, path) != "pending":
                raise RuntimeError("La ejecución cambió mientras se preparaba la cola")
            command = [sys.executable, "-u", "-m", "src.train", "--config", str(path), "--device", args.device]
            with (log_dir / f"{name}.log").open("x", encoding="utf-8") as log:
                child = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE,
                                         stderr=subprocess.STDOUT, text=True, encoding="utf-8")
                try:
                    for line in child.stdout:
                        print(line, end="", flush=True)
                        log.write(line)
                        log.flush()
                    code = child.wait()
                except BaseException:
                    child.terminate()
                    try:
                        child.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        child.kill()
                        child.wait()
                    raise
            if code != 0:
                print(f"Cola detenida: {name} terminó con error {code}. Resultados anteriores conservados.")
                return code
            if execution_state(ROOT, path) != "complete":
                raise RuntimeError(f"{name}: salida sin artefactos completos")
        print("Tanda terminada. No se ha evaluado test. Revisa todos los resultados antes de elegir.")
    finally:
        lock.unlink()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
