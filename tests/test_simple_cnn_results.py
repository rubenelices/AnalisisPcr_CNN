import nbformat
import pandas as pd
from src.reporting.simple_cnn_results import class_table, embed, TAG, equivalent_losses


def test_empty_positive_precision_marked_undefined():
    rows=class_table(dict(confusion_matrix=dict(tn=12,fp=0,fn=4,tp=0)))
    assert pd.isna(rows.loc[1,'Precision'])
    assert rows.loc[1,'Recall']==0
    assert rows.loc[0,'Precision']==.75


def test_equivalent_bce_and_baseline_fit_only():
    import numpy as np
    fit=pd.DataFrame(dict(target=[0,0,0,1],probability=[.1,.2,.3,.8]))
    inner=pd.DataFrame(dict(target=[0,1,1,1],probability=[.2,.3,.4,.5]))
    table=equivalent_losses(dict(fit=fit,inner=inner)).set_index('Particion')
    expected=-np.mean([np.log(.9),np.log(.8),np.log(.7),np.log(.8)])
    assert np.isclose(table.loc['fit','BCEpacienteEval'],expected)
    assert np.isclose(table.loc['inner','BCEbaselineConstanteFit'],-(np.log(.75)+3*np.log(.25))/4)


def test_embedding_keeps_historical_cells_and_rebases_links(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    main=tmp_path/'experiments/RESULTADOS.ipynb'
    main.parent.mkdir()
    old=nbformat.v4.new_code_cell('print("historical")',execution_count=1,
                                 outputs=[nbformat.v4.new_output('stream',name='stdout',text='historical\n')])
    campaign=nbformat.v4.new_markdown_cell('Previous campaign',metadata=dict(embedded_campaign='campaign_20261015'))
    nbformat.write(nbformat.v4.new_notebook(cells=[old,campaign]),main)
    overview=nbformat.v4.new_markdown_cell('[plan](PLAN.md)')
    detail=nbformat.v4.new_markdown_cell('![matrix](raw_fold0/resultados/diagnostico.png)')
    nb=nbformat.v4.new_notebook(cells=[overview,detail])
    embed(nb,overview);embed(nb,overview)
    restored=nbformat.read(main,as_version=4)
    assert sum(c.metadata.get('campaign_overview')==TAG for c in restored.cells)==1
    assert sum(c.metadata.get('embedded_campaign')==TAG for c in restored.cells)==1
    assert restored.cells[1]==old and restored.cells[2]==campaign
    assert 'C16_cnn_97809/PLAN.md' in restored.cells[0].source
    assert 'C16_cnn_97809/raw_fold0/resultados/diagnostico.png' in restored.cells[-1].source
