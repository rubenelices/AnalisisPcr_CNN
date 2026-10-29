import hashlib
import io
import json

import numpy as np
import pytest
import torch
from PIL import Image

from app import create_app
from src.web.inference import DCEInferenceEngine
from src.models.patient_feature_model import PatientFeatureNet
from src.training.train_patient_features import load_spec


@pytest.fixture
def release(tmp_path):
    spec, cfg = load_spec('configs/patient_features/E57_patient_feature_mean_mac.json')
    model = PatientFeatureNet(cfg.model).eval()
    checkpoint = tmp_path/'e57.pt'
    torch.save(dict(model_class='PatientFeatureNet_v1', config=spec,
                    epoch=14, model_state_dict=model.state_dict()), checkpoint)
    decision = tmp_path/'decision.json'
    decision.write_text(json.dumps(dict(schema='e57_public_test_frozen_v1',
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        threshold=.325, aggregation='mean_cut_probabilities')))
    return model, checkpoint, decision


def test_e57_restores_real_head_and_frozen_threshold(release):
    model, checkpoint, decision = release
    engine = DCEInferenceEngine.from_checkpoint(checkpoint, device_name='cpu', decision_path=decision)
    x = torch.zeros(3,256,256)
    x[1].fill_(.4);x[2].fill_(.8)
    with torch.inference_mode():expected = torch.sigmoid(model(x.unsqueeze(0))).item()
    result = engine.predict_tensor(x)
    assert isinstance(engine.model, PatientFeatureNet)
    assert result.probability_pcr == pytest.approx(expected, abs=1e-7)
    assert result.threshold == .325
    assert result.predicted_class == int(expected >= .325)
    assert result.model_version == 'E57_patient_feature_mean_mac@epoch-14'


def test_decision_cannot_be_used_for_different_weights(release):
    _, checkpoint, decision = release
    spec = json.loads(decision.read_text());spec['checkpoint_sha256']='wrong'
    decision.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match='no corresponde'):
        DCEInferenceEngine.from_checkpoint(checkpoint, device_name='cpu', decision_path=decision)


def test_threshold_cannot_silently_override_frozen_protocol(release):
    _, checkpoint, decision = release
    with pytest.raises(ValueError, match='contradice'):
        DCEInferenceEngine.from_checkpoint(checkpoint, device_name='cpu', decision_path=decision, threshold=.5)


def test_patient_prediction_is_mean_of_individual_probabilities(release):
    _, checkpoint, decision = release
    engine = DCEInferenceEngine.from_checkpoint(checkpoint, device_name='cpu', decision_path=decision)
    x = torch.stack([torch.zeros(3,256,256),torch.full((3,256,256),.3)])
    with torch.inference_mode():expected = torch.sigmoid(engine.model(x)).mean().item()
    assert engine.predict_patient_tensors(x).probability_pcr == pytest.approx(expected, abs=1e-7)


def test_web_e57_uses_decision_protocol_for_five_consecutive_samples(release, monkeypatch):
    _, checkpoint, decision = release
    monkeypatch.setenv('BREAST_PCR_CHECKPOINT',str(checkpoint))
    monkeypatch.setenv('BREAST_PCR_DECISION',str(decision))
    app = create_app();client = app.test_client()
    for _ in range(5):
        files={}
        for phase, value in [('PRE',20),('EARLY',120),('LATE',200)]:
            buf=io.BytesIO();Image.fromarray(np.full((256,256),value,dtype=np.uint8)).save(buf,format='PNG');buf.seek(0)
            files[phase]=(buf,phase+'.png')
        response=client.post('/api/predict',data=files)
        assert response.status_code==200
        assert response.json['threshold']==.325
        assert response.json['model_version']=='E57_patient_feature_mean_mac@epoch-14'
    assert not app.config['INFERENCE_ENGINE'].model.training
