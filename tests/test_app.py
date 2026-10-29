"""Contrato HTTP de la pagina HTML y su inferencia por las tres fases."""

from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image
from torch import Tensor

from app import create_app
from src.web.inference import InferenceResult


def _png(value: int) -> bytes:
    stream = io.BytesIO()
    Image.fromarray(np.full((256, 256), value, dtype=np.uint8), mode="L").save(
        stream, format="PNG"
    )
    return stream.getvalue()


def _uploads() -> dict[str, tuple[io.BytesIO, str]]:
    return {
        phase: (io.BytesIO(_png(value)), f"{phase}.png")
        for phase, value in (("PRE", 0), ("EARLY", 127), ("LATE", 255))
    }


class RecordingEngine:
    def __init__(self) -> None:
        self.calls = 0

    def predict_tensor(self, image: Tensor) -> InferenceResult:
        self.calls += 1
        assert image.shape == (3, 256, 256)
        assert image[:, 0, 0].tolist() == pytest.approx([0.0, 127 / 255, 1.0])
        return InferenceResult(0.73, 1, 0.5, "cpu", 2.0, "abc", "test")


def test_project_and_demo_pages_are_served_with_assets_and_notice() -> None:
    client = create_app().test_client()
    project = client.get("/")
    demo = client.get("/demo")
    assert project.status_code == demo.status_code == 200
    assert 'id="arquitectura"' in project.get_data(as_text=True)
    assert "E57 original · test público ya evaluado" in project.get_data(as_text=True)
    assert "298.161 parámetros" in project.get_data(as_text=True)
    assert "N01 no hereda ese test" in project.get_data(as_text=True)
    assert demo.data.count(b'type="file"') == 3
    assert "no tiene validez clínica" in demo.get_data(as_text=True)
    for asset in ("/app.js", "/site.js", "/style.css", "/tailwind.css"):
        assert client.get(asset).status_code == 200


def test_prediction_reuses_engine_for_consecutive_uploads() -> None:
    app = create_app()
    engine = RecordingEngine()
    app.config["INFERENCE_ENGINE"] = engine
    client = app.test_client()
    for _ in range(5):
        response = client.post("/api/predict", data=_uploads())
        assert response.status_code == 200
        assert response.json["probability_pcr"] == 0.73
        assert response.json["predicted_class"] == 1
    assert engine.calls == 5


@pytest.mark.parametrize("name", ["E21_group_norm", "E22_average_max", "E23_compact"])
def test_five_uploads_with_real_new_cnn_match_shared_pipeline(tmp_path, name):
    # PNG sintéticos y pesos aleatorios, no datos/pesos privados ni calidad científica.
    from pathlib import Path
    import torch
    from src.models.architectures import BreastPCRNet
    from src.training.experiment_config import load_experiment_config
    from src.web.inference import DCEInferenceEngine
    root = Path(__file__).resolve().parents[1]
    config = load_experiment_config(root / f"configs/experiments/{name}.json")
    model = BreastPCRNet(config.model).eval()
    checkpoint = tmp_path / "synthetic.pt"
    torch.save({"config": config.to_dict(), "model_state_dict": model.state_dict(),
                "experiment_id": name, "epoch": 0}, checkpoint)
    engine = DCEInferenceEngine.from_checkpoint(checkpoint, device_name="cpu")
    app = create_app()
    app.config["INFERENCE_ENGINE"] = engine
    client = app.test_client()
    for i in range(5):
        sources = {phase: _png(value + i) for phase, value in (("PRE", 5), ("EARLY", 125), ("LATE", 245))}
        expected = engine.predict_sources(sources)
        uploads = {k: (io.BytesIO(v), f"{k}.png") for k, v in sources.items()}
        result = client.post("/api/predict", data=uploads)
        assert result.status_code == 200
        assert result.json["probability_pcr"] == pytest.approx(expected.probability_pcr, abs=1e-7)
        assert result.json["model_sha256"] == expected.model_sha256
        assert app.config["INFERENCE_ENGINE"] is engine and not engine.model.training


def test_invalid_or_missing_phases_do_not_reach_model() -> None:
    app = create_app()
    engine = RecordingEngine()
    app.config["INFERENCE_ENGINE"] = engine
    client = app.test_client()
    missing = _uploads()
    del missing["LATE"]
    assert client.post("/api/predict", data=missing).status_code == 400
    corrupt = _uploads()
    corrupt["EARLY"] = (io.BytesIO(b"not a PNG"), "EARLY.png")
    assert client.post("/api/predict", data=corrupt).status_code == 400
    assert engine.calls == 0


def test_identical_phases_are_valid_warned_and_not_modified():
    class SamePhasesEngine:
        def predict_tensor(self, image):
            assert image[1].equal(image[2])
            assert image[1, 0, 0].item() == pytest.approx(127/255)
            return InferenceResult(.3, 0, .325, 'cpu', 1., 'synthetic', 'synthetic')
    app = create_app(); app.config['INFERENCE_ENGINE'] = SamePhasesEngine()
    data = _uploads(); data['LATE'] = (io.BytesIO(_png(127)), 'LATE.png')
    response = app.test_client().post('/api/predict', data=data)
    assert response.status_code == 200 and response.json['probability_pcr'] == .3
    assert len(response.json['warnings']) == 1
    assert 'EARLY y LATE' in response.json['warnings'][0]


def test_two_files_for_one_phase_are_rejected_instead_of_silently_ignored():
    from werkzeug.datastructures import MultiDict
    app = create_app(); engine = RecordingEngine(); app.config['INFERENCE_ENGINE'] = engine
    data = MultiDict(_uploads())
    data.add('PRE', (io.BytesIO(_png(11)), 'another_PRE.png'))
    assert app.test_client().post('/api/predict', data=data).status_code == 400
    assert engine.calls == 0


@pytest.mark.parametrize('problem', ['dimensions', 'rgb', 'jpeg', 'phase_size', 'total_size'])
def test_invalid_image_shapes_formats_and_sizes_never_reach_model(problem):
    app = create_app(); engine = RecordingEngine(); app.config['INFERENCE_ENGINE'] = engine
    data = _uploads()
    if problem in ('dimensions', 'rgb', 'jpeg'):
        stream = io.BytesIO()
        array = np.zeros((255, 256), dtype=np.uint8) if problem == 'dimensions' else (
            np.zeros((256, 256, 3), dtype=np.uint8) if problem == 'rgb' else np.zeros((256, 256), dtype=np.uint8))
        Image.fromarray(array).save(stream, format='JPEG' if problem == 'jpeg' else 'PNG')
        stream.seek(0)
        data['EARLY'] = (stream, 'EARLY.png')
    else:
        size = 5*1024*1024+1 if problem == 'phase_size' else 17*1024*1024
        data['EARLY'] = (io.BytesIO(b'x'*size), 'EARLY.png')
    response = app.test_client().post('/api/predict', data=data)
    assert response.status_code == (413 if problem == 'total_size' else 400)
    assert 'error' in response.json and engine.calls == 0


def test_missing_checkpoint_is_reported_without_fake_prediction(monkeypatch) -> None:
    monkeypatch.delenv("BREAST_PCR_CHECKPOINT", raising=False)
    client = create_app(default_model=None).test_client()
    assert client.get("/api/status").json["model_configured"] is False
    response = client.post("/api/predict", data=_uploads())
    assert response.status_code == 503
    assert "modelo entrenado" in response.json["error"]


def test_project_architecture_numbers_and_receptive_fields_match_real_e57():
    from pathlib import Path
    import torch
    from torch import nn
    from src.models.patient_feature_model import PatientFeatureNet
    from src.training.train_patient_features import load_spec
    root = Path(__file__).resolve().parents[1]
    _, config = load_spec(root/'configs/patient_features/E57_patient_feature_mean_mac.json')
    model = PatientFeatureNet(config.model)
    html = (root/'web/index.html').read_text()
    script = (root/'web/site.js').read_text()
    number = lambda n: f'{n:,}'.replace(',', '.')
    assert f'{number(sum(p.numel() for p in model.parameters()))} parámetros' in html
    assert f'{number(sum(p.numel() for p in model.head.parameters()))} parámetros' in html
    receptive, jump = 1, 1
    observed = []
    handles = []
    def executed(layer, inputs, output):
        nonlocal receptive, jump
        kernel = layer.kernel_size[0] if isinstance(layer.kernel_size, tuple) else layer.kernel_size
        stride = layer.stride[0] if isinstance(layer.stride, tuple) else layer.stride
        receptive += (kernel-1)*jump
        jump *= stride
    for block in model.encoder.blocks:
        assert f'{number(sum(p.numel() for p in block.parameters()))} parámetros' in html
        for layer in block.modules():
            if isinstance(layer, (nn.Conv2d, nn.MaxPool2d)):
                handles.append(layer.register_forward_hook(executed))
        handles.append(block.register_forward_hook(lambda layer, inputs, output: observed.append(receptive)))
    try:
        with torch.inference_mode(): model.eval()(torch.zeros(1, 3, 32, 32))
    finally:
        for handle in handles: handle.remove()
    assert observed == [8, 22, 50, 106]
    assert all(f'{r} × {r} píxeles' in script for r in observed)


class InfoEngine:
    def __init__(self) -> None:
        import torch
        self.model = torch.nn.Linear(4, 1)  # 5 parámetros
        self.model_version = "E57@epoch-14"
        self.model_sha256 = "a" * 64
        self.threshold = 0.325
        self.device = "cpu"


def test_model_card_reports_the_loaded_model() -> None:
    app = create_app()
    app.config["INFERENCE_ENGINE"] = InfoEngine()
    data = app.test_client().get("/api/modelo").json
    assert data["version"] == "E57@epoch-14" and data["threshold"] == 0.325
    assert data["parameters"] == 5 and data["sha256"] == "a" * 64 and data["device"] == "cpu"
    assert "validez" in data["notice"]


def test_model_card_without_model_is_an_explicit_error(monkeypatch) -> None:
    monkeypatch.delenv("BREAST_PCR_CHECKPOINT", raising=False)
    response = create_app(default_model=None).test_client().get("/api/modelo")
    assert response.status_code == 503 and "modelo" in response.json["error"]


def test_built_interface_is_served_at_root_with_assets_and_examples(tmp_path) -> None:
    (tmp_path / "assets").mkdir()
    (tmp_path / "examples" / "c").mkdir(parents=True)
    (tmp_path / "index.html").write_text("<div id=root></div>")
    (tmp_path / "assets" / "index.js").write_text("console.log(1)")
    (tmp_path / "examples" / "c" / "PRE.png").write_bytes(_png(10))
    (tmp_path / "modelo").mkdir()
    (tmp_path / "modelo" / "arquitectura.svg").write_text("<svg/>")
    client = create_app(frontend_dir=tmp_path).test_client()
    assert "id=root" in client.get("/").get_data(as_text=True)
    assert client.get("/assets/index.js").status_code == 200
    assert client.get("/examples/c/PRE.png").status_code == 200
    assert client.get("/modelo/arquitectura.svg").status_code == 200
    assert client.get("/no-existe.png").status_code == 404
    assert client.get("/api/modelo").status_code in (200, 503)  # la API no queda tapada
    assert client.get("/demo").status_code == 200  # la demo anterior sigue disponible
    assert client.get("/assets/../app.py").status_code == 404


def test_plain_start_uses_the_delivery_model_and_env_overrides_it(monkeypatch, tmp_path) -> None:
    from pathlib import Path
    checkpoint, decision = tmp_path / "best.pt", tmp_path / "protocolo.json"
    checkpoint.write_bytes(b"x")
    monkeypatch.delenv("BREAST_PCR_CHECKPOINT", raising=False)
    monkeypatch.delenv("BREAST_PCR_HF_REPO", raising=False)
    assert create_app(default_model=(checkpoint, decision)).test_client().get("/api/status").json["model_configured"]
    missing = (tmp_path / "no-existe.pt", decision)
    assert not create_app(default_model=missing).test_client().get("/api/status").json["model_configured"]
    monkeypatch.setenv("BREAST_PCR_CHECKPOINT", str(Path("otro.pt")))
    assert create_app(default_model=missing).test_client().get("/api/status").json["model_configured"]


def test_etiqueta_devuelve_pcr_del_csv_y_404_si_no_existe(tmp_path):
    csv = tmp_path/"samples.csv"
    csv.write_text("sample_id,patient_id,split,pCR\nP1_z001,P1,train,1\nP2_z005,P2,test,0\n")
    cliente = create_app(default_model=None, samples_csv=csv).test_client()
    assert cliente.get("/api/etiqueta/P1_z001").get_json() == {"sample_id": "P1_z001", "pcr": 1, "split": "train"}
    assert cliente.get("/api/etiqueta/P2_z005").get_json()["pcr"] == 0
    assert cliente.get("/api/etiqueta/NOEXISTE").status_code == 404


def test_sin_checkpoint_local_descarga_de_hugging_face_solo_si_se_configura(monkeypatch, tmp_path) -> None:
    import sys, types
    descargas = []
    falso = types.ModuleType("huggingface_hub")
    falso.hf_hub_download = lambda repo_id, filename: descargas.append((repo_id, filename)) or str(tmp_path / filename)
    monkeypatch.setitem(sys.modules, "huggingface_hub", falso)
    monkeypatch.delenv("BREAST_PCR_CHECKPOINT", raising=False)
    missing = (tmp_path / "no-existe.pt", tmp_path / "protocolo.json")
    monkeypatch.delenv("BREAST_PCR_HF_REPO", raising=False)
    assert not create_app(default_model=missing).test_client().get("/api/status").json["model_configured"]
    assert descargas == []
    monkeypatch.setenv("BREAST_PCR_HF_REPO", "usuario/modelo")
    assert create_app(default_model=missing).test_client().get("/api/status").json["model_configured"]
    assert descargas == [("usuario/modelo", "best.pt")]


def test_cors_solo_para_la_api_y_los_origenes_configurados(monkeypatch) -> None:
    monkeypatch.setenv("BREAST_PCR_ALLOWED_ORIGINS", "https://demo.example, https://otra.example")
    cliente = create_app(default_model=None).test_client()
    permitido = cliente.get("/api/status", headers={"Origin": "https://demo.example"})
    assert permitido.headers["Access-Control-Allow-Origin"] == "https://demo.example"
    assert "Access-Control-Allow-Origin" not in cliente.get("/api/status", headers={"Origin": "https://mala.example"}).headers
    assert "Access-Control-Allow-Origin" not in cliente.get("/demo", headers={"Origin": "https://demo.example"}).headers
    monkeypatch.delenv("BREAST_PCR_ALLOWED_ORIGINS")
    assert "Access-Control-Allow-Origin" not in create_app(default_model=None).test_client().get(
        "/api/status", headers={"Origin": "https://demo.example"}).headers
