"""Incorporar ensayos de solo imágenes al notebook sin mover sus originales."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from src.evaluation.metrics import binary_metrics


def image_entries(root):
    root=Path(root)
    registry=root/'experiments/image_experiments.json'
    if not registry.exists():return []
    result=[]
    for relative in json.loads(registry.read_text()):
        path=Path(relative)
        if path.is_absolute() or '..' in path.parts:raise ValueError('Config fuera del proyecto')
        raw=json.loads((root/path).read_text())
        schema=raw.get('schema')
        if schema=='patient_features_v1':
            from src.training.train_patient_features import load_spec
        elif schema=='patient_features_weighted_v1':
            from src.training.train_e57_weighted import load_spec
        elif schema=='patient_features_augmented_v1':
            from src.training.train_e57_augmented import load_spec
        else:raise ValueError('Esquema de imagen no admitido')
        spec,cfg=load_spec(root/path)
        directory=Path(cfg.output_dir)
        if (directory.is_absolute() or '..' in directory.parts or
                not any(directory.is_relative_to(p) for p in ('reports/patient_features','experiments'))):
            raise ValueError('Resultado de imagen fuera del proyecto')
        folder=root/directory
        item=dict(name=cfg.experiment_id,directory=directory.as_posix(),image_only=True,
            role='Solo imágenes · referencia E57' if schema=='patient_features_v1' else 'Solo imágenes · nueva prueba',
            config=spec['experiment'],summary=None,
            status='No completo (parcial/en curso)' if (folder/'history.csv').exists() else 'Pendiente')
        if (folder/'summary.json').exists():
            summary=json.loads((folder/'summary.json').read_text())
            if summary.get('status')=='complete':
                if (summary.get('scientific_result') is not True or summary.get('test_evaluated') is not False
                        or summary.get('debug_patient_subset') is not False
                        or summary.get('experiment_id')!=cfg.experiment_id):
                    raise ValueError('Resultado de imagen no científico o identidad incorrecta')
                if json.loads((folder/'config_resolved.json').read_text())!=spec:
                    raise ValueError('Config guardada distinta de la registrada')
                h=pd.read_csv(folder/'history.csv');p=pd.read_csv(folder/'predictions_patient.csv')
                m=summary['validation_metrics']['patient']
                if p.patient_id.isna().any() or p.patient_id.duplicated().any() or len(p)!=m['n']:
                    raise ValueError('Pacientes inválidas en resultado de imagen')
                if (len(h)!=summary['epochs_completed'] or h.epoch.duplicated().any()
                        or h.epoch.tolist()!=list(range(1,len(h)+1))):
                    raise ValueError('Historial incompleto')
                best=h.loc[h.epoch==summary['best_epoch']]
                if len(best)!=1 or not np.isclose(best.iloc[0].patient_roc_auc,m['roc_auc'],atol=1e-9,rtol=0):
                    raise ValueError('Best no coincide con el resultado')
                actual=binary_metrics(p.target,p.probability,m['threshold'])
                for key in ('roc_auc','pr_auc','brier_score','balanced_accuracy','f1','sensitivity','specificity','precision'):
                    if not np.isclose(actual[key],m[key],atol=1e-9,rtol=0):raise ValueError('Métricas de imagen incoherentes')
                if actual['confusion_matrix']!=m['confusion_matrix']:raise ValueError('Matriz incoherente')
                item.update(summary=summary,status='Completo')
        result.append(item)
    return result
