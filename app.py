"""Servidor de la web: API de inferencia (``src.inference``) e interfaz.

Arranque: ``BREAST_PCR_CHECKPOINT=... BREAST_PCR_DECISION=... python app.py``.
Si existe ``frontend/dist`` (la interfaz React compilada con ``npm run build``),
se sirve en ``/``; la demo HTML anterior sigue en ``/demo``. El servidor escucha
en localhost por defecto y no guarda las imágenes recibidas.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, request, send_from_directory
from werkzeug.exceptions import RequestEntityTooLarge

from src.data import DataValidationError, load_dce_triplet
from src.inference import DCEInferenceEngine, EDUCATIONAL_NOTICE, prediction_as_dict


PHASES = ("PRE", "EARLY", "LATE")
MAX_PHASE_BYTES = 5 * 1024 * 1024
WEB_DIR = Path(__file__).resolve().parent / "web"
FRONTEND_DIR = Path(__file__).resolve().parent / "frontend" / "dist"


def create_app(frontend_dir: Path | None = None) -> Flask:
    app = Flask(__name__, static_folder=None)
    app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024
    app.config["INFERENCE_ENGINE"] = None
    interfaz = frontend_dir if frontend_dir and (frontend_dir / "index.html").is_file() else None

    def engine_or_none() -> DCEInferenceEngine | None:
        """Carga el modelo una sola vez y lo reutiliza en todas las peticiones."""
        engine = app.config["INFERENCE_ENGINE"]
        if engine is None:
            checkpoint = os.environ.get("BREAST_PCR_CHECKPOINT")
            if not checkpoint:
                return None
            engine = DCEInferenceEngine.from_checkpoint(
                checkpoint, decision_path=os.environ.get('BREAST_PCR_DECISION'))
            app.config["INFERENCE_ENGINE"] = engine
        return engine

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
        configured = bool(app.config["INFERENCE_ENGINE"] or os.environ.get("BREAST_PCR_CHECKPOINT"))
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

    @app.errorhandler(RequestEntityTooLarge)
    def too_large(_error: RequestEntityTooLarge) -> Any:
        return jsonify({"error": "La carga total supera el limite de 16 MB."}), 413

    @app.post("/api/predict")
    def predict() -> Any:
        if (set(request.files) != set(PHASES)
                or any(len(request.files.getlist(phase)) != 1 for phase in PHASES)):
            return jsonify({"error": "Carga exactamente PRE, EARLY y LATE."}), 400

        try:
            sources: dict[str, bytes] = {}
            for phase in PHASES:
                upload = request.files[phase]
                if not upload.filename or not upload.filename.lower().endswith(".png"):
                    raise DataValidationError(f"{phase}: el fichero debe ser PNG.")
                data = upload.stream.read(MAX_PHASE_BYTES + 1)
                if len(data) > MAX_PHASE_BYTES:
                    raise DataValidationError(f"{phase}: supera el limite de 5 MB.")
                sources[phase] = data
            image = load_dce_triplet(sources)
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

    return app


app = create_app(frontend_dir=FRONTEND_DIR)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, debug=False)
