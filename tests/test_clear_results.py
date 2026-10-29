import json

from src.reporting.clear_results import completed_runs


def test_only_complete_science_no_smoke_or_test(tmp_path):
    base=tmp_path/'experiments'
    for name,values in [('good',{}),('debug',{'debug_patient_subset':True}),
                        ('test',{'test_evaluated':True}),('partial',{'status':'partial'})]:
        path=base/name/'resultados/fold_0_seed_42';path.mkdir(parents=True)
        s=dict(status='complete',scientific_result=True,debug_patient_subset=False,
               test_evaluated=False,experiment_id=name)
        s.update(values);(path/'summary.json').write_text(json.dumps(s));(path/'history.csv').write_text('epoch\n1\n')
    assert [s['experiment_id'] for _,s in completed_runs(tmp_path)]==['good']


def test_d00_copy_is_not_counted_twice(tmp_path):
    path=tmp_path/'experiments/D00_E58_referencia/resultados/fold_0_seed_42';path.mkdir(parents=True)
    (path/'summary.json').write_text(json.dumps(dict(status='complete',scientific_result=True,
        debug_patient_subset=False,test_evaluated=False,experiment_id='E58')))
    (path/'history.csv').write_text('epoch\n1\n')
    assert completed_runs(tmp_path)==[]


def test_renderer_keeps_all_epochs_and_six_graphs(tmp_path):
    import pandas as pd
    from src.evaluation.metrics import binary_metrics
    from src.reporting.clear_results import show_run
    folder=tmp_path
    predictions=pd.DataFrame(dict(patient_id=['a','b','c','d'],target=[0,0,1,1],probability=[.2,.3,.7,.8]))
    predictions.to_csv(folder/'predictions_patient.csv',index=False)
    pd.DataFrame(dict(epoch=[1,2],train_loss=[.7,.6],validation_loss=[.65,.64],
                      patient_roc_auc=[1.,1.],patient_pr_auc=[1.,1.],learning_rate=[.001,.001],
                      elapsed_seconds=[1.,1.])).to_csv(folder/'history.csv',index=False)
    summary=dict(experiment_id='synthetic',best_epoch=1,
                 validation_metrics={'patient':binary_metrics(predictions.target,predictions.probability,.5)})
    outputs=show_run(folder,summary,as_outputs=True)
    assert len(outputs)==3 and 'image/png' in outputs[2]['data']
    assert 'Loss validación' in outputs[1]['data']['text/html']
    assert outputs[1]['data']['text/html'].count('<tr>')==2  # dos épocas, header lleva estilo
    assert 'patient_id' not in outputs[1]['data']['text/html']
