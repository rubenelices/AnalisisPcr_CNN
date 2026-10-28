import json
from pathlib import Path
import pandas as pd
import pytest
import torch

from src.clinical_features import ClinicalPreprocessor
from src.metrics import binary_metrics
from src.patient_visual_report import validate_patients,threshold_table,save_patient_visual_report
from src.train_multimodal import load_spec


def frame():
    return pd.DataFrame(dict(patient_id=['a','b','c','d'],split=['validation']*4,
        fold=[0]*4,target=[0,1,0,1],probability=[.1,.35,.2,.8]))


def test_threshold_tradeoff_does_not_change_ranking():
    table=threshold_table(frame())
    assert len(table)==19
    assert table.roc_auc.nunique()==1 and table.pr_auc.nunique()==1
    assert table.sensitivity.is_monotonic_decreasing
    assert table.specificity.is_monotonic_increasing
    assert set(['tn','fp','fn','tp','precision','f1','brier_score']).issubset(table)


@pytest.mark.parametrize('edit',['duplicate','test','invalid_probability','fold'])
def test_report_rejects_invalid_or_test(edit):
    f=frame()
    if edit=='duplicate':f.loc[1,'patient_id']='a'
    if edit=='test':f.loc[0,'split']='test'
    if edit=='invalid_probability':f.loc[0,'probability']=2
    if edit=='fold':f.loc[0,'fold']=1
    with pytest.raises(ValueError):validate_patients(f)


def test_ablation_disables_value_and_missing_and_survives_restore():
    p=pd.DataFrame(dict(pid=['a','b'],age=[30,50],HR=[0,1],HER2=[1,0],menopause=[None,1]))
    prep=ClinicalPreprocessor.fit(p,active_columns=['HR','HER2'])
    changed=p.copy(); changed.age=[20,90]; changed.menopause=[0,None]
    assert torch.equal(prep.transform(p),prep.transform(changed))
    assert not prep.transform(p)[:,[0,3,4,7]].any()
    assert torch.equal(prep.transform(p),ClinicalPreprocessor.from_dict(prep.to_dict()).transform(p))
    with pytest.raises(ValueError):ClinicalPreprocessor.fit(p,active_columns=['pCR'])


def test_default_old_checkpoint_schema_unchanged():
    p=pd.DataFrame(dict(pid=['a'],age=[40],HR=[1],HER2=[0],menopause=[1]))
    prep=ClinicalPreprocessor.fit(p)
    assert 'active_columns' not in prep.to_dict()
    assert ClinicalPreprocessor.from_dict(prep.to_dict()).active_columns==('age','HR','HER2','menopause')


def test_new_trials_one_factor_and_unique_paths():
    base=json.loads(Path('configs/multimodal/E58_clinical_fusion_mac.json').read_text())
    paths=[]
    for name,key,value in [('D01_lr0003','learning_rate',.0003),('D02_wd0003','weight_decay',.003)]:
        spec,cfg=load_spec(f'experimentos_datos/{name}/config.json')
        expected=base['experiment']['training']|{key:value}
        assert spec['experiment']['training']==expected
        assert spec['experiment']['model']==base['experiment']['model']
        paths.append(cfg.output_dir)
    assert len(set(paths))==2


def test_report_generates_all_artifacts_without_touching_sources(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    source=Path('reports/example');source.mkdir(parents=True)
    f=frame();f.to_csv(source/'predictions_patient.csv',index=False)
    m=binary_metrics(f.target,f.probability,.5)
    summary=dict(experiment_id='synthetic',best_epoch=1,test_evaluated=False,
                 scientific_result=False,validation_metrics=dict(patient=m))
    (source/'summary.json').write_text(json.dumps(summary))
    pd.DataFrame(dict(epoch=[1,2],patient_roc_auc=[m['roc_auc'],.8],patient_pr_auc=[m['pr_auc'],.6])).to_csv(source/'history.csv',index=False)
    before={p.name:p.read_bytes() for p in source.iterdir()}
    dest=Path('experimentos_datos/test/resultados')
    result=save_patient_visual_report(source,dest)
    assert result['threshold_selected'] is False
    assert len(list(dest.glob('*.png')))==6
    assert (dest/'LEEME_RESULTADOS.md').exists()
    assert before=={p.name:p.read_bytes() for p in source.iterdir()}


def test_legacy_patient_csv_context_verified_against_metadata(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    source=Path('reports/legacy');source.mkdir(parents=True)
    metadata=Path('breastdcedl/metadata');metadata.mkdir(parents=True)
    f=frame(); f.drop(columns=['split','fold']).to_csv(source/'predictions_patient.csv',index=False)
    index=f.rename(columns={'target':'pCR'}).assign(split='train')
    index[['patient_id','split','fold','pCR']].to_csv(metadata/'samples.csv',index=False)
    (source/'config_resolved.json').write_text(json.dumps(dict(schema='patient_features_v1',
        experiment=dict(data=dict(root='breastdcedl',validation_fold=0)))))
    m=binary_metrics(f.target,f.probability,.5)
    (source/'summary.json').write_text(json.dumps(dict(experiment_id='legacy',best_epoch=1,
        model_class='PatientFeatureNet_v1',test_evaluated=False,scientific_result=True,
        debug_patient_subset=False,validation_metrics=dict(patient=m))))
    pd.DataFrame(dict(epoch=[1],patient_roc_auc=[m['roc_auc']],patient_pr_auc=[m['pr_auc']])).to_csv(source/'history.csv',index=False)
    save_patient_visual_report(source)
    index.loc[0,'split']='test'
    index[['patient_id','split','fold','pCR']].to_csv(metadata/'samples.csv',index=False)
    with pytest.raises(ValueError,match='validación completa'):save_patient_visual_report(source)
