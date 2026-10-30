"""Modelo de entrega según research/REGLAS_ENTREGA.md, y su ÚNICA evaluación en test.

Dos pasos separados:
    python -m src.evaluation.final_ensemble freeze --variant p02 --gpu   # aplica las reglas a la validación cruzada; no lee test
    python -m src.evaluation.final_ensemble test --device cpu            # evalúa en test UNA vez; se niega a repetir

`freeze` exige que las reglas no hayan cambiado (SHA fijado el 20/10 a las 22:05),
que estén los 5 folds de normal y ponderada, y escribe la decisión con los SHA de
los 5 pesos y el umbral. `test` solo acepta esa decisión, comprueba los SHA y crea
su carpeta de salida en modo exclusivo: si ya existe, no hace nada.
"""
import argparse
import hashlib
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.evaluation.cv_report import VARIANT_DIRS, build as build_cv
from src.models.patient_feature_model import load_feature_checkpoint

ROOT = Path(__file__).resolve().parents[2]
RULES = Path('research/REGLAS_ENTREGA.md')
RULES_SHA = 'ac853cd9c2edc97ea4183590ffadabd825b5201ca969e6b63282eb4c39a16913'
OUT = Path('reports/new_approach/ENTREGA')
MIN_POOLED_AUC = 0.55  # regla 4
FOLDS = range(5)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def choose(summary):
    """Reglas 1–4 sobre el resumen de cv_report. Devuelve (receta, motivo)."""
    for loss in ('normal', 'weighted'):
        if loss not in summary or summary[loss]['n_folds'] != 5:
            raise ValueError(f'Faltan folds de {loss}: relanzarlos antes de decidir (regla 2)')
    pair = summary.get('pareada')
    if pair is None or pair['n'] != summary['normal']['n_pacientes']:
        raise ValueError('Sin comparación pareada sobre las mismas pacientes')
    lo, hi = pair['ic95']
    if lo <= 0 <= hi:
        loss, why = 'normal', 'IC95 pareado incluye 0: normal por defecto (regla 3)'
    else:
        loss = 'normal' if pair['diferencia_auc_normal_menos_ponderada'] > 0 else 'weighted'
        why = f'IC95 pareado excluye 0: gana {loss} (regla 3)'
    auc = summary[loss]['auc_conjunta']
    if auc < MIN_POOLED_AUC:
        return None, f'{why}; AUC conjunta {auc:.3f} < {MIN_POOLED_AUC}: se mantiene E57 (regla 4)'
    return loss, f'{why}; AUC conjunta {auc:.3f} ≥ {MIN_POOLED_AUC}: sustituye a E57 (regla 4)'


def freeze(variant='p02', mode='runs_cuda', root=ROOT):
    root = Path(root)
    if sha(root/RULES) != RULES_SHA:
        raise ValueError('REGLAS_ENTREGA.md cambió después de fijarse: no se decide')
    out = root/OUT
    if (out/'decision.json').exists():
        raise FileExistsError('La decisión ya está congelada; no se rehace')
    _, summary = build_cv(variant, mode, root=root)
    loss, why = choose(summary)
    decision = dict(schema='entrega_v1', rules_sha256=RULES_SHA, variant=variant, mode=mode,
                    frozen_at_utc=datetime.now(timezone.utc).isoformat(), motivo=why,
                    cv={l: dict(auc_conjunta=summary[l]['auc_conjunta'], ic95=summary[l]['auc_conjunta_ic95'],
                                auc_media_folds=summary[l]['auc_media_folds']) for l in ('normal', 'weighted')},
                    pareada=summary['pareada'], test_evaluated=False)
    if loss is None:
        decision.update(entregable='E57_original', nota='E57 ya tiene su test (0,5728); no se evalúa nada más')
    else:
        weights = root/'checkpoints/new_approach'/VARIANT_DIRS[variant]/mode
        runs = root/'reports/new_approach'/VARIANT_DIRS[variant]/mode
        members, thresholds = [], []
        for fold in FOLDS:
            ckpt = weights/f'{loss}_fold_{fold}_seed_42'/'best.pt'
            d = json.loads((runs/f'{loss}_fold_{fold}_seed_42'/'decision_before_outer.json').read_text())
            if sha(ckpt) != d['checkpoint_sha256']:
                raise ValueError(f'best.pt del fold {fold} no coincide con el de su decisión interna')
            members.append(dict(fold=fold, checkpoint=str(ckpt.relative_to(root)), sha256=d['checkpoint_sha256'],
                                epoch=d['best_epoch'], internal_threshold=d['threshold']))
            thresholds.append(d['threshold'])
        decision.update(entregable=f'{variant.upper()}_{loss}_promedio_5_folds', loss=loss, members=members,
                        threshold=float(np.median(thresholds)), threshold_rule='mediana de los 5 umbrales internos (regla 6)',
                        aggregation='media de los 5 modelos por corte; media de cortes por paciente (regla 5)')
    out.mkdir(parents=True, exist_ok=True)
    (out/'decision.json').write_text(json.dumps(decision, indent=2, ensure_ascii=False))
    return decision


class EnsembleModel(nn.Module):
    """Probabilidad por corte = media de las probabilidades de los modelos (no de los logits)."""

    def __init__(self, models):
        super().__init__()
        self.members = nn.ModuleList(models)

    @torch.no_grad()
    def predict_proba(self, image):
        return torch.stack([torch.sigmoid(m(image)) for m in self.members]).mean(0)


def load_ensemble(decision, root=ROOT, device='cpu'):
    models = []
    for m in decision['members']:
        path = Path(root)/m['checkpoint']
        if sha(path) != m['sha256']:
            raise ValueError(f'Pesos del fold {m["fold"]} modificados tras congelar la decisión')
        models.append(load_feature_checkpoint(path, device))
    return EnsembleModel(models).eval()


def evaluate_test(device_name='cpu', root=ROOT):
    from src.pipeline.data import BreastDCEDataset, load_samples
    from src.evaluation.evaluation import EvaluationConfig, build_evaluation_summary
    from src.evaluation.metrics import aggregate_by_patient
    root = Path(root)
    decision = json.loads((root/OUT/'decision.json').read_text())
    if decision.get('rules_sha256') != RULES_SHA or decision.get('entregable', '').startswith('E57'):
        raise ValueError('No hay un entregable nuevo congelado que evaluar')
    if device_name not in ('cpu', 'mps'):
        raise ValueError('Evaluación local en CPU o MPS')
    device = torch.device(device_name)
    out = root/OUT/'test'
    out.mkdir(parents=True, exist_ok=False)  # exclusivo: una sola evaluación de test
    (out/'protocolo_previo.json').write_text(json.dumps(dict(
        decision_sha256=sha(root/OUT/'decision.json'), threshold=decision['threshold'],
        started_at_utc=datetime.now(timezone.utc).isoformat(), device=device_name,
        nota='Única evaluación de test del entregable; nada se ajusta después.'), indent=2, ensure_ascii=False))
    try:
        started = time.perf_counter()
        samples = load_samples(root/'breastdcedl')
        train = samples.loc[samples.split.eq('train')]
        rows = samples.loc[samples.split.eq('test')].copy()
        if (rows.patient_id.nunique() != 176 or len(rows) != 1758 or not rows.fold.eq(-1).all()
                or set(rows.patient_id) & set(train.patient_id)):
            raise ValueError('El test no corresponde a la partición pública esperada')
        loader = DataLoader(BreastDCEDataset(rows, root/'breastdcedl'), batch_size=16, shuffle=False, num_workers=0)
        ensemble = load_ensemble(decision, root, device)
        records = []
        for batch in loader:
            if set(batch['split']) != {'test'}:
                raise ValueError('Solo test público')
            probs = ensemble.predict_proba(batch['image'].to(device)).cpu().tolist()
            records += [dict(sample_id=batch['sample_id'][i], patient_id=batch['patient_id'][i],
                             slice_index=int(batch['slice_index'][i]), target=int(batch['target'][i]),
                             probability=float(p)) for i, p in enumerate(probs)]
            print(f'Test: {len(records)}/{len(rows)} cortes', flush=True)
        cuts = pd.DataFrame(records)
        patients = aggregate_by_patient(cuts.patient_id, cuts.target, cuts.probability)
        if len(patients) != 176 or len(cuts) != 1758:
            raise ValueError('Evaluación incompleta')
        patients['prediction'] = (patients.probability >= decision['threshold']).astype(int)
        cuts.to_csv(out/'predictions_cut.csv', index=False)
        patients.to_csv(out/'predictions_patient.csv', index=False)
        result = build_evaluation_summary(cuts, patients, EvaluationConfig(
            threshold=decision['threshold'], seed=42, bootstrap_repetitions=2000))
        cm = result['patient_metrics_primary']['confusion_matrix']
        den = 2*cm['tn'] + cm['fp'] + cm['fn']
        result['patient_metrics_primary']['f1_macro'] = ((2*cm['tn']/den if den else 0.)
                                                         + result['patient_metrics_primary']['f1'])/2
        result.update(status='complete', test_evaluated=True, entregable=decision['entregable'],
                      selection_using_test=False, elapsed_seconds=time.perf_counter() - started,
                      completed_at_utc=datetime.now(timezone.utc).isoformat(),
                      environment=dict(device=device_name, torch=torch.__version__, python=platform.python_version()))
        (out/'test_evaluation.json').write_text(json.dumps(result, indent=2, ensure_ascii=False))
        m = result['patient_metrics_primary']
        print(f"TEST {decision['entregable']}: AUC {m['roc_auc']:.4f} · umbral {decision['threshold']:.4f} · "
              f"TN{cm['tn']} FP{cm['fp']} FN{cm['fn']} TP{cm['tp']}", flush=True)
        return result
    except BaseException as error:
        (out/'interrupcion.json').write_text(json.dumps(dict(error=repr(error),
            nota='No repetir automáticamente: revisar antes'), indent=2))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    f = sub.add_parser('freeze')
    f.add_argument('--variant', choices=sorted(VARIANT_DIRS), default='p02')
    f.add_argument('--gpu', action='store_true')
    t = sub.add_parser('test')
    t.add_argument('--device', choices=['cpu', 'mps'], default='cpu')
    args = parser.parse_args()
    if args.command == 'freeze':
        d = freeze(args.variant, 'runs_cuda' if args.gpu else 'runs')
        print(f"Entregable: {d['entregable']} · {d['motivo']}" + (f" · umbral {d['threshold']:.4f}" if 'threshold' in d else ''))
    else:
        evaluate_test(args.device)


if __name__ == '__main__':
    main()
