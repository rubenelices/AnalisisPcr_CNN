import json
import numpy as np
import pandas as pd
import pytest
import torch
from src.training.simple_cnn_comparison import config, make_model, load_checkpoint, patient_result


@pytest.mark.parametrize('representation', ['raw', 'pre_differences'])
def test_exact_topology_shapes_gradient_and_restore(tmp_path, representation):
    torch.set_num_threads(2)
    model = make_model(representation)
    assert sum(p.numel() for p in model.parameters()) == 97809
    assert sum(isinstance(m, torch.nn.Conv2d) for m in model.modules()) == 4
    assert sum(isinstance(m, torch.nn.BatchNorm2d) for m in model.modules()) == 4
    assert sum(isinstance(m, torch.nn.Linear) for m in model.modules()) == 1
    assert [x.shape for x in model.trace_shapes(2) if x.operation.startswith('bloque')] == [
        (2,16,128,128), (2,32,64,64), (2,64,32,32), (2,128,16,16)]
    x = torch.rand(2,3,64,64,requires_grad=True)
    z = model(x)
    torch.nn.functional.binary_cross_entropy_with_logits(z, torch.tensor([0.,1.])).backward()
    assert x.grad.abs().sum()>0
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    model.eval()
    expected = model(x).detach()
    spec = json.loads(json.dumps(config(0, representation)))
    ck = tmp_path/'best.pt'
    torch.save(dict(model_class='BreastPCRNet_97809_v1', config=spec,
                    model_state_dict=model.state_dict(), epoch=3), ck)
    restored, _ = load_checkpoint(ck)
    with torch.no_grad():
        torch.testing.assert_close(restored(x),expected,rtol=0,atol=0)


def test_probability_average_is_not_mean_logits():
    probabilities=np.array([.01,.7,.3,.6])
    cuts=pd.DataFrame(dict(patient_id=['a','a','b','b'],target=[0,0,1,1],probability=probabilities))
    patients, metrics=patient_result(cuts)
    assert np.allclose(patients.probability,[.355,.45])
    assert patients.patient_id.tolist()==['a','b']
    assert metrics['n']==2 and not metrics['precision_defined']
    assert not np.isclose(.355,1/(1+np.exp(-np.log(probabilities[:2]/(1-probabilities[:2])).mean())))


def test_representations_share_initial_weights_and_transform_only_input():
    torch.manual_seed(42);raw=make_model('raw').eval()
    torch.manual_seed(42);diff=make_model('pre_differences').eval()
    assert all(torch.equal(v,diff.state_dict()[k]) for k,v in raw.state_dict().items())
    x=torch.rand(2,3,32,32)
    transformed=torch.stack((x[:,0],x[:,1]-x[:,0],x[:,2]-x[:,1]),dim=1)
    with torch.no_grad():
        torch.testing.assert_close(raw(transformed),diff(x),rtol=0,atol=0)


def test_reject_unknown_representation():
    with pytest.raises(ValueError):make_model('rgb')


def test_balanced_reader_keeps_new_schema_separate(tmp_path):
    from src.reporting.balanced_results import entries
    from src.reporting.clear_results import completed_runs
    folder=tmp_path/'experiments/C16_cnn_97809/raw_fold0/resultados'
    folder.mkdir(parents=True)
    (folder/'config_resolved.json').write_text(json.dumps(config(0,'raw')))
    (folder/'history.csv').write_text('epoch,train_loss\n1,.6\n')
    (folder/'summary.json').write_text(json.dumps(dict(status='complete',scientific_result=True,
        test_evaluated=False,debug_patient_subset=False,experiment_id='raw_fold0')))
    assert len(entries(tmp_path))==1
    assert completed_runs(tmp_path)==[]
