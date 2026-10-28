"""Configuraciones y reportes sintéticos; no científicos ni smokes entrenados."""
import json
from pathlib import Path
import nbformat
import pytest
import torch
from src.balanced_results import entries,overview,class_f1,class_table,build
from src.train_multimodal import load_spec
from src.multimodal_model import MultimodalPCRNet

NAMES=('B01_bce_ponderada','B02_aumento_geometrico','B03_silu_cnn','B04_semilla17')

@pytest.mark.parametrize('name',NAMES)
def test_configs_are_basic_independent_and_not_debug(name):
    spec,cfg=load_spec(f'experimentos_equilibrados/{name}/config.json')
    _,ref=load_spec('experimentos_datos/D36_edad_menopausia/config.json')
    assert spec['clinical_input_columns']==spec['active_clinical_columns']==['age','menopause']
    assert cfg.training.epochs==50 and cfg.training.early_stopping_patience==10
    assert cfg.data.train_patients_per_class is None and cfg.data.validation_fold==0
    assert cfg.training.loss==('weighted' if name==NAMES[0] else 'normal')
    assert cfg.model.activation==('silu' if name==NAMES[2] else 'relu')
    assert cfg.training.seed==(17 if name==NAMES[3] else 42)
    assert cfg.data.augmentation.enabled==(name==NAMES[1])
    assert cfg.training.learning_rate==ref.training.learning_rate and cfg.model.channels==ref.model.channels
    assert cfg.output_dir.startswith(f'experimentos_equilibrados/{name}/resultados/')

@pytest.mark.parametrize('name',NAMES)
def test_smokes_prepared_not_scientific(name):
    _,cfg=load_spec(f'experimentos_equilibrados/{name}/smoke.json')
    assert cfg.training.epochs==2 and cfg.data.train_patients_per_class==4
    assert cfg.data.validation_patients_per_class==2 and cfg.output_dir.startswith('reports/smoke/')

def test_silu_architecture_real_shapes_no_training():
    torch.set_num_threads(2)
    _,cfg=load_spec('experimentos_equilibrados/B03_silu_cnn/config.json')
    model=MultimodalPCRNet(cfg.model).eval()
    assert sum(isinstance(m,torch.nn.SiLU) for m in model.image_encoder.modules())==8
    assert isinstance(model.clinical_branch[1],torch.nn.ReLU)
    assert sum(p.numel() for p in model.parameters())==294289
    with torch.inference_mode():z=model(torch.zeros(2,3,64,64),torch.zeros(2,8))
    assert z.shape==(2,) and torch.isfinite(z).all()
    folder=Path('experimentos/B03_silu_cnn')
    assert (folder/'arquitectura.png').is_file() and (folder/'arquitectura.svg').is_file()
    assert 'SiLU' in (folder/'README.md').read_text()

def fixture_config(tmp_path,output=None):
    folder=tmp_path/'experimentos_equilibrados/B01_bce_ponderada';folder.mkdir(parents=True)
    spec=json.loads(Path('experimentos_equilibrados/B01_bce_ponderada/config.json').read_text())
    if output is not None:spec['experiment']['output_dir']=output
    (folder/'config.json').write_text(json.dumps(spec));return spec

def test_pending_notebook_has_no_invented_scientific_results(tmp_path):
    fixture_config(tmp_path)
    items=entries(tmp_path);assert len(items)==2 and all(x['summary'] is None for x in items)
    table=overview(items);assert 'ROC-AUC' not in table
    p=build(tmp_path);nb=nbformat.read(p,as_version=4)
    assert len(nb.cells)==6 and not any('image/png' in o.get('data',{}) for c in nb.cells if c.cell_type=='code' for o in c.outputs)


def test_campaign_snapshot_survives_future_regeneration_without_duplicates(tmp_path):
    fixture_config(tmp_path);path=build(tmp_path);book=nbformat.read(path,as_version=4)
    overview_cell=nbformat.v4.new_markdown_cell('Campaña: resumen conservado');overview_cell.metadata['campaign_overview']='example'
    detail_cell=nbformat.v4.new_markdown_cell('Campaña: resultados conservados');detail_cell.metadata['embedded_campaign']='example'
    book.cells.insert(0,overview_cell);book.cells.append(detail_cell);nbformat.write(book,path)
    for _ in range(2):
        build(tmp_path);restored=nbformat.read(path,as_version=4)
        assert len(restored.cells)==8
        assert restored.cells[0].id==overview_cell.id and restored.cells[-1].id==detail_cell.id
        assert sum(c.metadata.get('embedded_campaign')=='example' for c in restored.cells)==1

def test_partial_not_marked_complete(tmp_path):
    spec=fixture_config(tmp_path);out=tmp_path/spec['experiment']['output_dir'];out.mkdir(parents=True)
    (out/'history.csv').write_text('epoch,train_loss\n1,0.6\n')
    item=entries(tmp_path)[1];assert 'parcial' in item['status'] and item['summary'] is None

def test_extra_folds_same_id_preserved_separately(tmp_path):
    spec=fixture_config(tmp_path)
    extra=tmp_path/'experimentos_equilibrados/B01_bce_ponderada/resultados/fold_1_seed_42'
    extra.mkdir(parents=True);spec['experiment']['data']['validation_fold']=1
    (extra/'config_resolved.json').write_text(json.dumps(spec))
    (extra/'history.csv').write_text('epoch,train_loss\n1,0.6\n')
    items=entries(tmp_path)
    assert len(items)==3 and sum(x['name']=='B01_bce_ponderada' for x in items)==2
    assert len(set(x['directory'] for x in items))==3
    table=overview(items)
    assert set(table[table.Experimento=='B01_bce_ponderada'].Fold)=={0,1}

@pytest.mark.parametrize('output',['../outside','reports/other','/tmp/outside'])
def test_branch_paths_guarded(tmp_path,output):
    fixture_config(tmp_path,output)
    with pytest.raises(ValueError,match='Salida fuera'):entries(tmp_path)

def test_forbidden_clinical_inputs_rejected(tmp_path):
    spec=fixture_config(tmp_path);spec['clinical_input_columns']=['age','HR']
    (tmp_path/'experimentos_equilibrados/B01_bce_ponderada/config.json').write_text(json.dumps(spec))
    with pytest.raises(ValueError,match='sin HR'):entries(tmp_path)

def test_incomplete_final_report_is_not_trusted(tmp_path,monkeypatch):
    spec=fixture_config(tmp_path);out=tmp_path/spec['experiment']['output_dir'];out.mkdir(parents=True)
    (out/'summary.json').write_text(json.dumps(dict(status='complete',experiment_id='B01_bce_ponderada',active_clinical_columns=['age','menopause'])))
    def reject(*args,**kwargs):
        assert kwargs['require_scientific'] is True
        raise ValueError('Informe inconsistente')
    monkeypatch.setattr('src.balanced_results._load_experiment',reject)
    with pytest.raises(ValueError,match='inconsistente'):entries(tmp_path)

def test_f1_classes_not_confused_with_balanced_accuracy():
    m=dict(confusion_matrix=dict(tn=155,fp=0,fn=64,tp=0),f1=0.,precision=0.,sensitivity=0.,specificity=1.)
    negative,macro=class_f1(m)
    assert negative==pytest.approx(310/374) and macro==pytest.approx(155/374)
    table=class_table(m)
    assert table.Clase.tolist()==['no pCR (0)','pCR (1)'] and table.Pacientes.tolist()==[155,64]
    assert table.iloc[0]['Precisión']==pytest.approx(.7078)
