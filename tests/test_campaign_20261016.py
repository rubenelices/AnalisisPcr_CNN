import copy
import json
from pathlib import Path

import nbformat
import pytest
import torch

from src.campaigns import campaign_20261016 as campaign
import pandas as pd


def test_control_changes_only_channels():
    wide=campaign.spec('wide');small=campaign.spec('small',compact=True)
    a=copy.deepcopy(wide);b=copy.deepcopy(small)
    a.pop('experiment_id');b.pop('experiment_id')
    a['model'].pop('channels');b['model'].pop('channels')
    assert a==b
    assert sum(p.numel() for p in campaign.make_model(wide).parameters())==97809
    assert sum(p.numel() for p in campaign.make_model(small).parameters())==24713


def test_patient_sampler_one_cut_per_patient_reproducible_without_global_rng():
    rows=pd.DataFrame([dict(patient_id=f'p{i}',sample_id=f'p{i}_{j}',split='train') for i in range(20) for j in range(10)])
    a=campaign.PatientCutSampler(rows,42);b=campaign.PatientCutSampler(rows,42)
    state=torch.random.get_rng_state().clone();first=list(a);second=list(a)
    assert first==list(b) and second==list(b) and first!=second
    assert len(first)==20 and rows.iloc[first].patient_id.nunique()==20
    assert torch.equal(state,torch.random.get_rng_state())
    rows.loc[0,'split']='test'
    with pytest.raises(ValueError):campaign.PatientCutSampler(rows,42)


def test_horizontal_only_keeps_rng_and_phase_alignment(monkeypatch):
    draws=iter([torch.tensor(.1),torch.tensor(.1)])
    monkeypatch.setattr(torch,'rand',lambda *a,**kw:next(draws))
    x=torch.arange(60.).reshape(3,4,5)
    assert torch.equal(campaign.horizontal_only(x),x.flip(-1))
    with pytest.raises(StopIteration):next(draws)


def test_models_restore_and_accept_original_triplet(tmp_path):
    torch.set_num_threads(2)
    for compact in (False,True):
        s=campaign.spec('roundtrip',compact=compact)
        model=campaign.make_model(s).eval();x=torch.randn(2,3,64,64)
        with torch.no_grad():expected=model(x)
        p=tmp_path/'weights.pt';torch.save(model.state_dict(),p)
        restored=campaign.make_model(json.loads(json.dumps(s))).eval()
        restored.load_state_dict(torch.load(p,weights_only=True))
        with torch.no_grad():torch.testing.assert_close(restored(x),expected,rtol=0,atol=0)
        assert expected.shape==(2,)


def test_deadline_and_stop_prevent_more_work(monkeypatch,tmp_path):
    monkeypatch.setattr(campaign,'ROOT',tmp_path)
    monkeypatch.setattr(campaign,'remaining',lambda:-1)
    with pytest.raises(InterruptedError):campaign.check()
    monkeypatch.setattr(campaign,'remaining',lambda:60)
    campaign.check();(tmp_path/'STOP').touch()
    with pytest.raises(InterruptedError):campaign.check()


def test_repeated_report_preserves_historical_outputs(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    campaign.ROOT.mkdir(parents=True)
    main=campaign.ROOT.parent/'RESULTADOS.ipynb'
    historical=nbformat.v4.new_code_cell('old',execution_count=1,outputs=[nbformat.v4.new_output('stream',name='stdout',text='untouched')])
    nbformat.write(nbformat.v4.new_notebook(cells=[historical]),main)
    campaign.report();campaign.report()
    saved=nbformat.read(main,as_version=4)
    assert saved.cells[1]==historical
    assert sum(c.metadata.get('campaign_overview')==campaign.TAG for c in saved.cells)==1


def test_cv_report_rejects_patient_overlap_across_validation_folds(tmp_path,monkeypatch):
    from src.evaluation import generalization_panels
    monkeypatch.setattr(generalization_panels,'diagnostics',lambda *a:Path('fake.png'))
    monkeypatch.setattr(generalization_panels,'fold_overview',lambda *a:None)
    from src.evaluation.metrics import binary_metrics
    from src.campaigns.campaign_20261015_fast import enrich
    monkeypatch.chdir(tmp_path)
    campaign.ROOT.mkdir(parents=True)
    main=campaign.ROOT.parent/'RESULTADOS.ipynb'
    nbformat.write(nbformat.v4.new_notebook(),main)
    monkeypatch.setattr(campaign,'figure',lambda *a,**k:None)
    monkeypatch.setattr(campaign,'png_cell',lambda *a:nbformat.v4.new_markdown_cell('fake figure'))
    m=enrich(binary_metrics([0,1],[.2,.8],.5));m['precision_defined']=True
    for fold,n in enumerate([219,220,219,220,219]):
        folder=campaign.ROOT/'resultados'/f'C23_un_corte_paciente_normal_fold{fold}';folder.mkdir(parents=True)
        (folder/'config_resolved.json').write_text(json.dumps({'experiment_id':'C23_un_corte_paciente_normal','fold':fold}))
        (folder/'summary.json').write_text(json.dumps({'selection':{'validation_eval':{'patient_metrics':m}}}))
        (folder/'selection.json').write_text(json.dumps({'epoch':1,'train_eval':{'patient_metrics':m},'validation_eval':{'patient_metrics':m}}))
        pd.DataFrame({'epoch':[1],'patient_roc_auc':[1.]}).to_csv(folder/'history.csv',index=False)
        pd.DataFrame({'patient_id':[f'p{fold}_{i}' for i in range(n)],'target':[i%2 for i in range(n)],'probability':[.3]*n}).to_csv(folder/'predictions_validation_patients_best.csv',index=False)
    campaign.report()
    stats=json.loads((campaign.ROOT/'C23_cv_summary.json').read_text())
    assert stats['patients']==1097 and stats['folds']==5 and stats['selection_independent'] is False
    folder=campaign.ROOT/'resultados/C23_un_corte_paciente_normal_fold4'
    frame=pd.read_csv(folder/'predictions_validation_patients_best.csv');frame.loc[0,'patient_id']='p0_0';frame.to_csv(folder/'predictions_validation_patients_best.csv',index=False)
    with pytest.raises(ValueError,match='disjuntas'):campaign.report()
