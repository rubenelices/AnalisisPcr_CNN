"""Servidor de la web: API de inferencia (``src.inference``) e interfaz.

Arranque: ``python app.py`` carga el modelo de entrega (``DEFAULT_MODEL``). Las variables
``BREAST_PCR_CHECKPOINT`` y ``BREAST_PCR_DECISION`` permiten usar otro sin tocar el código.
Si no hay checkpoint en disco y ``BREAST_PCR_HF_REPO`` está definida (despliegue), los pesos
se descargan de ese repositorio de Hugging Face; el protocolo comprueba su SHA-256 al cargar.
``BREAST_PCR_ALLOWED_ORIGINS`` (lista separada por comas) permite servir la API a una interfaz
alojada en otro dominio.
Si existe ``frontend/dist`` (la interfaz React compilada con ``npm run build``),
se sirve en ``/``; la demo HTML anterior sigue en ``/demo``. El servidor escucha
en localhost por defecto y no guarda las imágenes recibidas.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd
from flask import Flask, jsonify, request, send_from_directory
from werkzeug.exceptions import RequestEntityTooLarge

from src.data import DataValidationError, load_dce_triplet
from src.inference import DCEInferenceEngine, EDUCATIONAL_NOTICE, prediction_as_dict


PHASES = ("PRE", "EARLY", "LATE")
MAX_PHASE_BYTES = 5 * 1024 * 1024
WEB_DIR = Path(__file__).resolve().parent / "web"
FRONTEND_DIR = Path(__file__).resolve().parent / "frontend" / "dist"
# Modelo de entrega: checkpoint de E57 y protocolo con su umbral fijado antes del test.
# Si se cambia el modelo entregado, se actualizan estas dos rutas.
ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = (
    ROOT / "checkpoints/E57_patient_feature_mean_mac/fold_0_seed_42/best.pt",
    ROOT / "modelo/protocolo_e57.json",  # umbral y SHA-256 congelados antes del test
)
HF_FILENAME = "best.pt"
SAMPLES_CSV = ROOT / "breastdcedl" / "metadata" / "samples.csv"


def create_app(
    frontend_dir: Path | None = None,
    default_model: tuple[Path, Path] | None = DEFAULT_MODEL,
    samples_csv: Path | None = SAMPLES_CSV,
) -> Flask:
    app = Flask(__name__, static_folder=None)
    app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024
    app.config["INFERENCE_ENGINE"] = None
    interfaz = frontend_dir if frontend_dir and (frontend_dir / "index.html").is_file() else None

    def model_paths() -> tuple[str, str | None] | None:
        """Las variables de entorno mandan; si no hay, el modelo de entrega si está en disco."""
        checkpoint = os.environ.get("BREAST_PCR_CHECKPOINT")
        if checkpoint:
            return checkpoint, os.environ.get("BREAST_PCR_DECISION")
        if default_model is not None and default_model[0].is_file():
            return str(default_model[0]), str(default_model[1])
        repo = os.environ.get("BREAST_PCR_HF_REPO")
        if repo and default_model is not None:
            from huggingface_hub import hf_hub_download
            return hf_hub_download(repo_id=repo, filename=HF_FILENAME), str(default_model[1])
        return None

    def engine_or_none() -> DCEInferenceEngine | None:
        """Carga el modelo una sola vez y lo reutiliza en todas las peticiones."""
        engine = app.config["INFERENCE_ENGINE"]
        if engine is None:
            paths = model_paths()
            if paths is None:
                return None
            engine = DCEInferenceEngine.from_checkpoint(paths[0], decision_path=paths[1])
            app.config["INFERENCE_ENGINE"] = engine
        return engine

    origenes = {o.strip() for o in os.environ.get("BREAST_PCR_ALLOWED_ORIGINS", "").split(",") if o.strip()}

    @app.after_request
    def cors(respuesta: Any) -> Any:
        """Solo para la API y solo para los orígenes configurados; por defecto, mismo origen."""
        origen = request.headers.get("Origin")
        if origen in origenes and request.path.startswith("/api/"):
            respuesta.headers["Access-Control-Allow-Origin"] = origen
            respuesta.headers["Vary"] = "Origin"
        return respuesta

    @app.get("/")
    def index() -> Any:
        if interfaz is not None:
            return send_from_directory(interfaz, "index.html")
        return send_from_directory(WEB_DIR, "index.html")

    if interfaz is not None:
        # Ficheros de la interfaz compilada (código, estilos, ejemplos, diagrama...).
        # Las rutas explícitas (/api/..., /demo) tienen prioridad sobre esta regla.
        @app.get("/<path:nombre>")
        def estaticos(nombre: str) -> Any:
            return send_from_directory(interfaz, nombre)

    @app.get("/demo")
    def demo() -> Any:
        return send_from_directory(WEB_DIR, "demo.html")

    @app.get("/tailwind.css")
    def tailwind_stylesheet() -> Any:
        return send_from_directory(WEB_DIR, "tailwind.css")

    @app.get("/style.css")
    def stylesheet() -> Any:
        return send_from_directory(WEB_DIR, "style.css")

    @app.get("/site.js")
    def site_script() -> Any:
        return send_from_directory(WEB_DIR, "site.js")

    @app.get("/app.js")
    def script() -> Any:
        return send_from_directory(WEB_DIR, "app.js")

    @app.get("/api/status")
    def status() -> Any:
        configured = bool(app.config["INFERENCE_ENGINE"] or model_paths())
        return jsonify({"model_configured": configured, "notice": EDUCATIONAL_NOTICE})

    @app.get("/api/modelo")
    def modelo() -> Any:
        """Ficha del modelo cargado, para la cabecera y la sección Modelo de la web."""
        try:
            engine = engine_or_none()
        except (ValueError, OSError) as exc:
            return jsonify({"error": f"No se pudo cargar el modelo: {exc}"}), 500
        if engine is None:
            return jsonify({"error": "Aun no hay un modelo entrenado configurado."}), 503
        return jsonify({
            "version": engine.model_version,
            "sha256": engine.model_sha256,
            "threshold": engine.threshold,
            "device": str(engine.device),
            "parameters": sum(p.numel() for p in engine.model.parameters()),
            "notice": EDUCATIONAL_NOTICE,
        })

    @app.get("/api/etiqueta/<sample_id>")
    def etiqueta(sample_id: str) -> Any:
        """pCR real de un corte del CSV público (samples.csv), para que Pruebas compare con la predicción.

        Solo se usa para mostrar acierto o fallo; no interviene en la predicción. Los cortes que no
        están en el CSV (por ejemplo, los de la validación privada) devuelven 404.
        """
        if samples_csv is None or not samples_csv.is_file():
            return jsonify({"error": "No hay CSV de etiquetas."}), 404
        if app.config.get("SAMPLES") is None:
            app.config["SAMPLES"] = pd.read_csv(samples_csv, usecols=["sample_id", "pCR", "split"]).set_index("sample_id")
        tabla = app.config["SAMPLES"]
        if sample_id not in tabla.index:
            return jsonify({"error": "Este corte no está en el CSV de datos."}), 404
        fila = tabla.loc[sample_id]
        return jsonify({"sample_id": sample_id, "pcr": int(fila["pCR"]), "split": str(fila["split"])})

    @app.errorhandler(RequestEntityTooLarge)
    def too_large(_error: RequestEntityTooLarge) -> Any:
        return jsonify({"error": "La carga total supera el limite de 16 MB."}), 413

    def uploaded_triplet() -> Any:
        """Valida las tres fases subidas y las convierte con el preprocesado del entrenamiento."""
        if (set(request.files) != set(PHASES)
                or any(len(request.files.getlist(phase)) != 1 for phase in PHASES)):
            raise DataValidationError("Carga exactamente PRE, EARLY y LATE.")
        sources: dict[str, bytes] = {}
        for phase in PHASES:
            upload = request.files[phase]
            if not upload.filename or not upload.filename.lower().endswith(".png"):
                raise DataValidationError(f"{phase}: el fichero debe ser PNG.")
            data = upload.stream.read(MAX_PHASE_BYTES + 1)
            if len(data) > MAX_PHASE_BYTES:
                raise DataValidationError(f"{phase}: supera el limite de 5 MB.")
            sources[phase] = data
        return load_dce_triplet(sources)

    @app.post("/api/predict")
    def predict() -> Any:
        try:
            image = uploaded_triplet()
            # El dataset puede contener fases realmente idénticas. Se advierte,
            # pero no se modifica ni rechaza un tensor válido para el modelo.
            identical = [f'{PHASES[i]} y {PHASES[j]}' for i, j in ((0, 1), (0, 2), (1, 2))
                         if image[i].equal(image[j])]
            warnings = ([f'Fases con píxeles idénticos: {", ".join(identical)}. '
                         'Comprueba que has seleccionado los archivos previstos.'] if identical else [])

            engine = engine_or_none()
            if engine is None:
                return jsonify({"error": "Aun no hay un modelo entrenado configurado."}), 503
            result = engine.predict_tensor(image)
        except (DataValidationError, ValueError, OSError) as exc:
            return jsonify({"error": str(exc)}), 400
        return jsonify(dict(prediction_as_dict(result), warnings=warnings))

    @app.post("/api/occlusion")
    def occlusion() -> Any:
        """Mapa de oclusión del mismo corte: cuánto cambia la probabilidad al tapar cada cuadrado."""
        try:
            image = uploaded_triplet()
            engine = engine_or_none()
            if engine is None:
                return jsonify({"error": "Aun no hay un modelo entrenado configurado."}), 503
            result = engine.occlusion_tensor(image)
        except (DataValidationError, ValueError, OSError) as exc:
            return jsonify({"error": str(exc)}), 400
        return jsonify({
            "base_probability": result.base_probability,
            "patch": result.patch,
            "size": result.size,
            "deltas": result.deltas,
            "latency_ms": result.latency_ms,
        })

    return app


app = create_app(frontend_dir=FRONTEND_DIR)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, debug=False)
