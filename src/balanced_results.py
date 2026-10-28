"""Notebook de la línea equilibrada: lee informes, no entrena ni altera fuentes."""
import argparse
import json
import re
from pathlib import Path
import nbformat
import pandas as pd
from src.clear_results import show_run
from src.compare_experiments import _load_experiment

ROOT=Path(__file__).resolve().parents[1]
BRANCH='experimentos_equilibrados'
REFERENCE='experimentos_datos/D36_edad_menopausia/resultados/fold_0_seed_42'

def class_f1(metrics):
    c=metrics['confusion_matrix']
    denom=2*c['tn']+c['fp']+c['fn']
    negative=2*c['tn']/denom if denom else 0.
    return negative,(negative+metrics['f1'])/2

def entries(root=ROOT):
    root=Path(root).resolve()
    specs=[dict(name='D36_edad_menopausia',directory=REFERENCE,role='Referencia original',config=None)]
    for path in sorted((root/BRANCH).glob('B*/config.json')):
        spec=json.loads(path.read_text());cfg=spec['experiment']
        directory=Path(cfg['output_dir'])
        if (directory.is_absolute() or '..' in directory.parts or
                not directory.is_relative_to(BRANCH) or 'resultados' not in directory.parts):
            raise ValueError('Salida fuera de la línea equilibrada')
        if spec.get('clinical_input_columns')!=['age','menopause'] or spec.get('active_clinical_columns')!=['age','menopause']:
            raise ValueError('Esta línea exige edad/menopausia, sin HR/HER2')
        specs.append(dict(name=cfg['experiment_id'],directory=directory.as_posix(),role='Nueva prueba',config=cfg))
    from src.image_result_entries import image_entries
    specs.extend(image_entries(root))
    # Incluir réplicas/folds adicionales aunque compartan el ID de arquitectura.
    covered={x['directory'] for x in specs}
    folders={p.parent for filename in ('summary.json','config_resolved.json','history.csv')
             for p in (root/BRANCH).rglob(filename) if 'resultados' in p.relative_to(root).parts
             and not p.is_relative_to(root/BRANCH/'campana_20261015')
             and not p.is_relative_to(root/BRANCH/'C16_cnn_97809')
             and not p.is_relative_to(root/BRANCH/'C17_receta_publicada')
             and not p.is_relative_to(root/BRANCH/'campana_20261016')}
    for folder in sorted(folders):
        relative=folder.relative_to(root).as_posix()
        if relative in covered:continue
        resolved=folder/'config_resolved.json'
        if not resolved.is_file():raise ValueError('Ejecución adicional sin configuración resuelta')
        spec=json.loads(resolved.read_text())
        if spec.get('schema') != 'multimodal_e19_v1':
            continue  # Otros runners tienen su propia integración/registro.
        cfg=spec['experiment']
        if spec.get('clinical_input_columns')!=['age','menopause'] or spec.get('active_clinical_columns')!=['age','menopause']:
            raise ValueError('Réplica fuera de las entradas clínicas de esta línea')
        specs.append(dict(name=cfg['experiment_id'],directory=relative,role='Réplica/fold adicional',config=cfg))
        covered.add(relative)
    seen=set()
    for item in specs:
        if not re.fullmatch(r'[A-Za-z0-9_-]+',item['name']) or item['directory'] in seen:
            raise ValueError('Identidad inválida/repetida')
        seen.add(item['directory']);folder=root/item['directory'];summary=folder/'summary.json'
        if item.get('image_only'):continue
        item['summary']=None
        item['status']='No completo (parcial/en curso)' if (folder/'history.csv').exists() else 'Pendiente'
        if summary.is_file():
            s=json.loads(summary.read_text())
            if s.get('status')=='complete':
                if (s.get('experiment_id')!=item['name'] or s.get('active_clinical_columns')!=['age','menopause']):
                    raise ValueError('Informe no corresponde a esta línea')
                _,_,provenance=_load_experiment(folder,require_scientific=True)
                item.update(summary=s,status='Completo',config=provenance['config'])
    return sorted(specs,key=lambda x:(x['role']!='Referencia original',x['name'],x['directory']))

def overview(items):
    rows=[]
    for item in items:
        cfg=item.get('config') or {}
        row={'Experimento':item['name'],'Estado':item['status'],'Papel':item['role'],
             'Fold':cfg.get('data',{}).get('validation_fold'),
             'Semilla':cfg.get('training',{}).get('seed')}
        s=item['summary']
        if s:
            m=s['validation_metrics']['patient'];negative,macro=class_f1(m)
            row.update({'Mejor época':s['best_epoch'],'Épocas':s['epochs_completed'],
                'ROC-AUC':m['roc_auc'],'AP':m['pr_auc'],'Brier↓':m['brier_score'],
                'Umbral':m['threshold'],'Precisión pCR':m['precision'],
                'Sensibilidad pCR':m['sensitivity'],'Especificidad':m['specificity'],
                'F1 pCR':m['f1'],'F1 no pCR':negative,'F1 macro':macro,
                'Balanced accuracy':m['balanced_accuracy'],**m['confusion_matrix']})
        rows.append(row)
    return pd.DataFrame(rows).round(4)

def class_table(metrics):
    c=metrics['confusion_matrix'];negative,_=class_f1(metrics)
    precision_negative=c['tn']/(c['tn']+c['fn']) if c['tn']+c['fn'] else 0.
    return pd.DataFrame([dict(Clase='no pCR (0)',Precisión=precision_negative,
        Recall=metrics['specificity'],F1=negative,Pacientes=c['tn']+c['fp']),
        dict(Clase='pCR (1)',Precisión=metrics['precision'],Recall=metrics['sensitivity'],
             F1=metrics['f1'],Pacientes=c['tp']+c['fn'])]).round(4)

def class_outputs(summary):
    m=summary['validation_metrics']['patient'];table=class_table(m)
    c=m['confusion_matrix'];denom=2*(c['tn']+c['fp'])+c['tp']+c['fn']
    baseline=(c['tn']+c['fp'])/denom if denom else 0.
    note=f'Baseline siempre noPCR: F1macro={baseline:.4f}; balancedaccuracy=0,5. Métricas al umbral {m["threshold"]}, sin ajustarlo aquí.'
    return [nbformat.v4.new_output('display_data',data={'text/html':table.to_html(index=False),'text/plain':table.to_string(index=False)}),
            nbformat.v4.new_output('display_data',data={'text/markdown':note,'text/plain':note})]

def display_entry(directory,root=ROOT):
    from IPython.display import display,Markdown,Image
    import matplotlib.pyplot as plt
    item=next(x for x in entries(root) if x['directory']==directory)
    folder=Path(root)/item['directory']
    if item['summary']:
        display(class_table(item['summary']['validation_metrics']['patient']))
        display(Markdown(class_outputs(item['summary'])[1]['data']['text/markdown']))
        show_run(folder,item['summary'])
        matched=folder/'curvas_comparables.png'
        if matched.is_file():
            display(Markdown('**E57: BCE y AUC paciente de train y validación en el mismo modo de evaluación.**'))
            display(Image(filename=str(matched)))
        from src.decision_diagnostics import display_decision
        display_decision(folder)
    elif (folder/'history.csv').is_file():
        h=pd.read_csv(folder/'history.csv')
        display(Markdown('**Sin evaluación final.** Historia parcial/en curso; no elegir ganador con ella.'))
        with pd.option_context('display.max_rows',None):display(h)
        fig,axes=plt.subplots(1,2,figsize=(12,4))
        for col,label in [('train_loss','Train'),('validation_loss','Validación')]:axes[0].plot(h.epoch,h[col],label=label)
        for col,label in [('patient_roc_auc','ROC-AUC'),('patient_pr_auc','AP')]:axes[1].plot(h.epoch,h[col],label=label)
        for ax in axes:ax.legend();ax.set_xlabel('Época');ax.grid(alpha=.2)
        fig.tight_layout();plt.show();plt.close(fig)
    else:display(Markdown('Pendiente: no hay entrenamiento ni resultados que interpretar.'))

def build(root=ROOT):
    root=Path(root).resolve();items=entries(root);target=root/BRANCH/'RESULTADOS.ipynb'
    previous=nbformat.read(target,as_version=4).cells if target.exists() else []
    # Conservar campañas integradas al regenerar la línea general.
    campaign_overviews=[c for c in previous if c.metadata.get('campaign_overview') or c.metadata.get('architecture_decision')]
    campaign_details=[c for c in previous if c.metadata.get('embedded_campaign')]
    grouped_directories={c.metadata['e57_grouped_directory'] for c in campaign_overviews
                         if c.metadata.get('e57_grouped_directory')}
    intro='''# Experimentos equilibrados · resultados y referencias

Objetivo vigente: media ROC-AUC por paciente0,62–0,63 con sobreajuste controlado y matriz útil.
No elegir un pico ni una semilla favorable. Confirmar candidatos en cincofolds.
Incluye la línea clínica básica, D36, E57 original y sus nuevas variantes solo imágenes.
La columna Papel distingue las entradas: no son todas imagen+clínica.
Los originales se leen donde están; no se mueven ni se reentrenan.
Cada completo incluye **todas las épocas y seis paneles**: pérdidas, evoluciónROC/AP,
ROC, precisión-recall, matriz y calibración del mejor checkpoint.
F1 de ambas clases, macro, balanced accuracy y conteos en la tabla inicial.
Cada completo añade distribución de probabilidades y una tabla diagnóstica
de umbrales fijos: sin elegir ni adoptar umbral y sin evaluación independiente.
Umbral0,5 provisional, sin optimizarlo aquí; seleccionar después en validación interna.
No llamar test independiente a estas métricas; no se entrena ni se evalúa test al abrir.
Pendientes y parciales se distinguen de completos. No se inventan curvas finales.
Uso educativo y de investigación, sin validez clínica.
'''
    intro+='\n[Campaña local 15octubre: validación exterior y umbral interno](campana_20261015/RESULTADOS.ipynb) · [Protocolo](campana_20261015/PLAN.md).\n'
    setup='''from pathlib import Path
import sys
from IPython.display import display
root=next(p for p in (Path.cwd(),*Path.cwd().parents) if (p/'src').is_dir() and (p/'AGENTS.md').is_file())
if str(root) not in sys.path:sys.path.insert(0,str(root))
from src.balanced_results import entries,overview,display_entry
display(overview(entries(root)))'''
    cells=[nbformat.v4.new_markdown_cell(intro),nbformat.v4.new_code_cell(setup)]
    table=overview(items)
    cells[1].outputs=[nbformat.v4.new_output('display_data',data={'text/html':table.to_html(index=False),'text/plain':table.to_string(index=False)})]
    for item in items:
        if item['directory'] in grouped_directories:
            continue
        link='../'+item['directory']+'/'
        heading=f"## {item['name']} · {Path(item['directory']).name} · {item['status']}\n\n{item['role']}."
        if item['summary']:heading+=f' [Respaldos técnicos originales]({link}).'
        cells.append(nbformat.v4.new_markdown_cell(heading))
        cell=nbformat.v4.new_code_cell(f"display_entry({item['directory']!r},root)")
        if item['summary']:
            from src.decision_diagnostics import decision_outputs
            cell.outputs=(class_outputs(item['summary'])+show_run(root/item['directory'],item['summary'],as_outputs=True)
                          +decision_outputs(root/item['directory']))
            matched=root/item['directory']/'curvas_comparables.png'
            if matched.is_file():
                import base64
                cell.outputs.append(nbformat.v4.new_output('display_data',data={
                    'text/plain':'E57: train y validación en el mismo modo de evaluación.',
                    'image/png':base64.b64encode(matched.read_bytes()).decode()}))
        else:cell.outputs=[nbformat.v4.new_output('display_data',data={'text/markdown':item['status'],'text/plain':item['status']})]
        cells.append(cell)
    for cell in cells:
        if cell.cell_type=='code':cell.metadata.update(tags=['hide-input'],jupyter={'source_hidden':True})
    nb=nbformat.v4.new_notebook(cells=campaign_overviews+cells+campaign_details)
    nb.metadata.kernelspec={'display_name':'Python3','language':'python','name':'python3'}
    target.parent.mkdir(parents=True,exist_ok=True)
    temporary=target.with_name('RESULTADOS.rebuild.tmp.ipynb');nbformat.write(nb,temporary);temporary.replace(target)
    print(f'Vista equilibrada: {sum(x["summary"] is not None for x in items)} completos; {target}')
    return target

if __name__=='__main__':
    argparse.ArgumentParser(description=__doc__).parse_args();build()
