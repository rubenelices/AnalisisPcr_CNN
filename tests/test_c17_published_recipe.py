import json
from pathlib import Path
import nbformat
import numpy as np
import pandas as pd
import pytest
import torch
from src import train_c17_published_recipe as c17


@pytest.mark.parametrize('draws,dims',[([.9,.9],()),([.1,.9],(-1,)),([.9,.1],(-2,)),([.1,.1],(-2,-1))])
def test_flips_preserve_phase_alignment(monkeypatch,draws,dims):
    values=iter(draws)
    monkeypatch.setattr(torch,'rand',lambda *a,**k:torch.tensor(next(values)))
    pre=torch.arange(20.).reshape(4,5);image=torch.stack([pre,pre+100,pre+200])
    result=c17.synchronized_flips(image)
    expected=torch.flip(image,dims=dims) if dims else image
    assert torch.equal(result,expected)
    assert torch.equal(result[1]-result[0],torch.full_like(pre,100))


def test_exact_recipe_and_reject_accidental_changes(tmp_path):
    path=tmp_path/'config.json';path.write_text(json.dumps(c17.recipe()))
    spec=c17.load_recipe(path)
    assert spec['optimizer']=='Adam' and spec['scheduler'] is None
    assert spec['max_epochs']==40 and spec['patience']==8
    assert spec['weight_decay']==.0001 and spec['losses']==['normal','ponderada']
    spec['learning_rate']=.0003;path.write_text(json.dumps(spec))
    with pytest.raises(ValueError):c17.load_recipe(path)


def test_patient_split_and_cut_weight_train_only():
    rows=pd.DataFrame([dict(patient_id=f'p{i}',pCR=i%2,fold=i%5,split='train',sample_id=f'p{i}_{j}') for i in range(30) for j in range(1 if i%2 else 3)])
    rows=pd.concat([rows,pd.DataFrame([dict(patient_id='test',pCR=1,fold=0,split='test',sample_id='test')])],ignore_index=True)
    tr,va=c17.split_rows(rows)
    assert not set(tr.patient_id)&set(va.patient_id)
    assert 'test' not in set(tr.patient_id)|set(va.patient_id)
    assert c17.positive_weight(tr,'ponderada')==3
    with pytest.raises(ValueError):c17.positive_weight(rows,'ponderada')


def test_evaluation_generator_does_not_consume_global_torch_rng(monkeypatch):
    class Dummy(torch.utils.data.Dataset):
        def __init__(self,rows,root,transform):self.rows=rows;self.transform=transform
        def __len__(self):return len(self.rows)
        def __getitem__(self,i):return dict(image=torch.zeros(3,2,2),target=torch.tensor(0.))
    monkeypatch.setattr(c17,'CachedDataset',Dummy)
    dl=c17.loader(pd.DataFrame({'n':[1,2,3]}),c17.recipe())
    torch.manual_seed(42);before=torch.random.get_rng_state().clone()
    list(dl)
    assert torch.equal(before,torch.random.get_rng_state())


def test_evaluate_keeps_cut_loss_separate_from_patient_metrics(monkeypatch):
    monkeypatch.setattr(c17,'stop_requested',lambda:None)
    p=torch.tensor([.1,.5,.9]);z=torch.logit(p)
    class Model(torch.nn.Module):
        def forward(self,x):return x[:,0,0,0]
    x=torch.zeros(3,3,2,2);x[:,0,0,0]=z
    rows=pd.DataFrame(dict(patient_id=['a','a','b'],sample_id=['a0','a1','b0'],pCR=[0,0,1]))
    batch=dict(image=x,sample_id=rows.sample_id.tolist())
    result,cuts,patients=c17.evaluate(Model(),[batch],rows,torch.device('cpu'),2.)
    expected=torch.nn.functional.binary_cross_entropy_with_logits(z,torch.tensor([0.,0.,1.])).item()
    assert np.isclose(result['cut_bce'],expected)
    assert np.allclose(patients.probability,[.3,.9])
    assert result['patient_metrics']['n']==2 and result['patient_metrics']['roc_auc']==1
    assert result['objective_cut_bce']>result['cut_bce']


def test_prepared_notebook_contains_no_executed_training_cells():
    nb=nbformat.read(c17.ROOT/'EJECUTAR.ipynb',as_version=4)
    assert all(c.execution_count is None and not c.outputs for c in nb.cells if c.cell_type=='code')


def test_new_schema_not_read_as_historical(tmp_path):
    from src.balanced_results import entries
    from src.clear_results import completed_runs
    folder=tmp_path/c17.ROOT/'resultados/normal/fold_0_seed_42';folder.mkdir(parents=True)
    (folder/'history.csv').write_text('epoch,train_loss\n1,.6\n')
    (folder/'config_resolved.json').write_text(json.dumps({'recipe':c17.recipe()}))
    (folder/'summary.json').write_text(json.dumps(dict(status='complete',scientific_result=True,test_evaluated=False,debug_patient_subset=False,experiment_id='C17_normal')))
    assert len(entries(tmp_path))==1 and completed_runs(tmp_path)==[]


def test_pending_results_keep_old_notebook_cells(tmp_path,monkeypatch):
    from src import c17_results as view
    monkeypatch.chdir(tmp_path)
    main=Path('experimentos_equilibrados/RESULTADOS.ipynb');main.parent.mkdir()
    old=nbformat.v4.new_code_cell('historical',execution_count=1,outputs=[nbformat.v4.new_output('stream',name='stdout',text='preserved')])
    nbformat.write(nbformat.v4.new_notebook(cells=[old]),main)
    view.build();view.build()
    saved=nbformat.read(main,as_version=4)
    assert saved.cells[1]==old
    assert sum(c.metadata.get('campaign_overview')==view.TAG for c in saved.cells)==1
    assert len(saved.cells)==4


def test_scientific_entry_requires_technical_preflight_without_training(tmp_path,monkeypatch):
    monkeypatch.setattr(c17,'ROOT',tmp_path/'C17')
    path=tmp_path/'config.json';path.write_text(json.dumps(c17.recipe()))
    calls=[]
    monkeypatch.setattr(c17,'run_one',lambda spec,loss,device,live: calls.append((spec['debug'],loss)))
    c17.run_config(path,'cpu')
    assert calls==[(True,'normal'),(True,'ponderada'),(False,'normal'),(False,'ponderada')]
    assert not (c17.ROOT/'active_process.json').exists()


def test_checkpoint_roundtrip_without_training(tmp_path):
    torch.set_num_threads(2)
    model=c17.make_model('raw').eval()
    x=torch.zeros(2,3,32,32)
    with torch.no_grad():expected=model(x)
    path=tmp_path/'model.pt'
    torch.save(dict(model_class='C17_BreastPCRNet_97809_v1',recipe=json.loads(json.dumps(c17.recipe())),
                    model_state_dict=model.state_dict(),epoch=1),path)
    restored,ck=c17.load_checkpoint(path)
    with torch.no_grad():torch.testing.assert_close(restored(x),expected,rtol=0,atol=0)
    assert ck['epoch']==1 and sum(p.numel() for p in restored.parameters())==97809


def test_six_panel_plot_from_synthetic_reports(tmp_path):
    from src.c17_results import figure
    from src.campaign_20261015_fast import enrich
    from src.metrics import binary_metrics
    from PIL import Image
    folder=tmp_path/'normal/fold0';folder.mkdir(parents=True)
    h=pd.DataFrame(dict(epoch=[1,2],train_eval_loss=[.65,.6],validation_loss=[.66,.62],
                        train_patient_auc=[.55,.62],patient_roc_auc=[.53,.59],mean_epoch_auc=[.53,.56]))
    h.to_csv(folder/'history.csv',index=False)
    p=pd.DataFrame(dict(patient_id=['a','b','c','d'],target=[0,0,1,1],probability=[.2,.4,.3,.7]))
    for name in ('train','validation'):p.to_csv(folder/f'predictions_{name}_patients_best.csv',index=False)
    m=enrich(binary_metrics(p.target,p.probability))
    figure(folder,dict(epoch=2,validation_eval={'patient_metrics':m}))
    with Image.open(folder/'curvas_y_diagnostico.png') as img:
        assert img.size==(2210,1300) and np.asarray(img).std()>0
