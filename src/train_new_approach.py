"""N01 manual y local: una condición/fold por invocación; nunca evalúa test."""
import argparse
import copy
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.architectures import trainable_parameter_count
from src.data import BreastDCEDataset
from src.learning_rate_schedule import LearningRateSchedule
from src.metrics import aggregate_by_patient
from src.new_approach_protocol import (ROOT, TECH, WEIGHTS, metrics, prepare, sha,
                                       select_threshold, split_rows, verify, write_json)
from src.patient_feature_model import PatientFeatureNet
from src.patient_training import PatientBatchSampler, patient_level_pos_weight
from src.train import _environment, set_reproducibility
from src.train_patient_features import evaluate_features, load_spec


def run_dirs(root, loss, fold, smoke=False):
    if loss not in ('normal', 'weighted') or type(fold) is not int or fold not in range(5):
        raise ValueError('Pérdida/fold inválidos')
    if smoke and fold != 0:
        raise ValueError('El smoke técnico prefijado es fold 0; los cinco splits se verifican en prepare')
    mode = 'smoke' if smoke else 'runs'
    name = f'{loss}_fold_{fold}_seed_42'
    return Path(root)/TECH/mode/name, Path(root)/WEIGHTS/mode/name


@contextmanager
def exclusive_lock(root):
    path = Path(root)/TECH/'execution.lock'
    path.parent.mkdir(parents=True, exist_ok=True)
    # Un lock existente no se elimina automáticamente, incluso si no hay PID activo.
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        with os.fdopen(fd, 'w') as stream:
            stream.write(f'pid={os.getpid()}\n')
        yield
    finally:
        path.unlink()


def make_loader(rows, root, fold, role, training=False):
    if not rows.split.eq('train').all():
        raise ValueError('Un loader N01 no admite test')
    ds = BreastDCEDataset(rows, Path(root)/'breastdcedl')
    generator = torch.Generator().manual_seed(42 if training else 10000+10*fold+role)
    if training:
        sampler = PatientBatchSampler(rows, 2, generator)
        return DataLoader(ds, batch_sampler=sampler, num_workers=0, generator=generator)
    return DataLoader(ds, batch_size=16, shuffle=False, num_workers=0, generator=generator)


def smoke_subset(frames):
    result = {}
    for name, frame in frames.items():
        n = 4 if name == 'fit' else 2
        patients = frame[['patient_id', 'pCR']].drop_duplicates('patient_id').sort_values('patient_id')
        ids = patients.groupby('pCR').head(n).patient_id
        result[name] = frame.loc[frame.patient_id.isin(ids)].reset_index(drop=True)
    return result


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total, count = 0., 0
    for batch in loader:
        optimizer.zero_grad(set_to_none=True)
        z, y, _ = model.forward_patients(batch['image'].to(device),
                                          batch['target'].to(device), batch['patient_id'])
        loss = criterion(z, y)
        if not torch.isfinite(loss):
            raise FloatingPointError('Pérdida no finita; detener sin producir resultado completo')
        loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise FloatingPointError('Gradiente no finito')
        optimizer.step()
        total += float(loss.detach().cpu())*len(y)
        count += len(y)
    if not count:
        raise ValueError('Loader de entrenamiento vacío')
    return total/count


def evaluate(model, loader, device):
    bce, bag_bce, all_metrics, _, cuts, _ = evaluate_features(model, loader, device, .5)
    patients = aggregate_by_patient(cuts.patient_id, cuts.target, cuts.probability)
    p = cuts.probability.to_numpy(float).clip(1e-7, 1-1e-7)
    y = cuts.target.to_numpy(float)
    cut_bce = float(-(y*np.log(p)+(1-y)*np.log1p(-p)).mean())
    return dict(bce=bce, bag_bce_secondary=bag_bce, cut_bce=cut_bce,
                cut_metrics_secondary=all_metrics['cut'],
                metrics=metrics(patients, .5)), patients, cuts


def state_sha(model):
    h = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        h.update(name.encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def write_history(path, history):
    tmp = path.with_suffix('.csv.tmp')
    pd.DataFrame(history).to_csv(tmp, index=False)
    tmp.replace(path)


def save_checkpoint(checkpoint, path):
    tmp = path.with_suffix('.pt.tmp')
    torch.save(checkpoint, tmp)
    tmp.replace(path)


def refresh(root):
    from src.new_approach_n01_results import build
    return build(root)


def run(loss, fold, device_name='mps', smoke=False, root=ROOT):
    root = Path(root)
    manifest, rows = verify(root)
    cfg = manifest['config']
    out, weights = run_dirs(root, loss, fold, smoke)
    if out.exists() or weights.exists():
        raise FileExistsError('Ejecución existente: conservarla; no se sobrescribe ni reanuda')
    if not smoke:
        smoke_out, _ = run_dirs(root, loss, 0, True)
        smoke_path = smoke_out/'summary.json'
        if not smoke_path.is_file():
            raise RuntimeError(f'Falta smoke técnico de la condición {loss} (fold 0)')
        result = json.loads(smoke_path.read_text())
        if (result.get('status') != 'complete' or result.get('scientific_result') is not False
                or result.get('protocol_sha256') != sha(root/TECH/'protocol/manifest.json')):
            raise RuntimeError('Smoke no completo o de otra versión del protocolo')
    if device_name not in ('mps', 'cpu'):
        raise ValueError('Solo CPU/MPS local')
    if device_name == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('MPS no disponible; no se cambia a CPU silenciosamente')
    device = torch.device(device_name)
    frames = split_rows(rows, fold)
    if smoke:
        frames = smoke_subset(frames)
    with exclusive_lock(root):
        # Revalidar después de adquirir el lock: evita cambios mientras se esperaba.
        verify(root)
        if out.exists() or weights.exists():
            raise FileExistsError('Salida creada por otra ejecución')
        set_reproducibility(42)
        torch.set_num_threads(4)
        spec, original_cfg = load_spec(root/cfg['reference'])
        model = PatientFeatureNet(original_cfg.model).to(device)
        if trainable_parameter_count(model) != 298161:
            raise ValueError('La CNN no coincide con E57 original')
        initial_sha = state_sha(model)
        fit_loader = make_loader(frames['fit'], root, fold, 0, training=True)
        fit_eval = make_loader(frames['fit'], root, fold, 1)
        inner_loader = make_loader(frames['inner'], root, fold, 2)
        # No se crea ni itera el loader exterior hasta congelar la decisión final.
        pos_weight = patient_level_pos_weight(frames['fit']) if loss == 'weighted' else 1.
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.001)
        scheduler = LearningRateSchedule(optimizer, original_cfg.training)
        # Identidad propia serializada: no etiquetar pesos N01 como el E57 histórico.
        checkpoint_spec = copy.deepcopy(spec)
        checkpoint_spec['experiment'].update(
            experiment_id=f'N01_{loss}_fold_{fold}_seed_42' + ('_smoke' if smoke else ''),
            output_dir=str(out.relative_to(root)), checkpoint_dir=str(weights.relative_to(root)))
        checkpoint_spec['experiment']['training']['loss'] = loss
        checkpoint_spec['experiment']['data']['validation_fold'] = fold
        out.mkdir(parents=True, exist_ok=False)
        weights.mkdir(parents=True, exist_ok=False)
        base = dict(schema='new_approach_n01_run_v1', loss=loss, fold=fold,
                    scientific_result=not smoke, test_evaluated=False,
                    protocol_sha256=sha(root/TECH/'protocol/manifest.json'),
                    initial_state_sha256=initial_sha, pos_weight=pos_weight,
                    parameter_count=298161, training_pool_stage='features',
                    evaluation_aggregation='mean_cut_probabilities',
                    counts={n: dict(patients=f.patient_id.nunique(), cuts=len(f),
                                    positive=int(f.drop_duplicates('patient_id').pCR.sum()))
                            for n, f in frames.items()},
                    started_utc=datetime.now(timezone.utc).isoformat(),
                    environment=_environment(device))
        write_json(out/'config_resolved.json', dict(protocol=manifest, run=base))
        history, best, best_epoch, stale = [], -float('inf'), 0, 0
        started = time.perf_counter()
        summary = dict(base, status='running', epochs_completed=0)
        write_json(out/'summary.json', summary)
        try:
            for epoch in range(1, (1 if smoke else cfg['epochs'])+1):
                tick = time.perf_counter()
                lr = scheduler.start_epoch(epoch)
                online_loss = train_epoch(model, fit_loader, optimizer, criterion, device)
                fit_result, _, _ = evaluate(model, fit_eval, device)
                inner_result, _, _ = evaluate(model, inner_loader, device)
                score = inner_result['metrics']['roc_auc']
                scheduler.finish_epoch(score)
                history.append(dict(epoch=epoch, train_online_objective=online_loss,
                                    fit_bce=fit_result['bce'], inner_bce=inner_result['bce'],
                                    fit_auc=fit_result['metrics']['roc_auc'], inner_auc=score,
                                    fit_cut_bce=fit_result['cut_bce'], inner_cut_bce=inner_result['cut_bce'],
                                    inner_ap=inner_result['metrics']['pr_auc'], learning_rate=lr,
                                    seconds=time.perf_counter()-tick))
                checkpoint = dict(model_class='PatientFeatureNet_v1', config=checkpoint_spec, epoch=epoch,
                                  model_state_dict=model.state_dict(),
                                  optimizer_state_dict=optimizer.state_dict(),
                                  scheduler_state_dict=scheduler.state_dict(),
                                  n01=dict(protocol_sha256=base['protocol_sha256'], loss=loss,
                                           fold=fold, pos_weight=pos_weight))
                save_checkpoint(checkpoint, weights/'last.pt')
                if score > best:
                    best, best_epoch, stale = score, epoch, 0
                    save_checkpoint(checkpoint, weights/'best.pt')
                else:
                    stale += 1
                write_history(out/'history.csv', history)
                summary.update(epochs_completed=len(history), best_epoch=best_epoch,
                               best_inner_auc=best)
                write_json(out/'summary.json', summary)
                if not smoke:
                    refresh(root)
                print(f'N01 {loss} fold={fold} época={epoch:02d} '
                      f'AUC FIT-eval={history[-1]["fit_auc"]:.4f} '
                      f'AUC INTERNA={score:.4f} BCE FIT={fit_result["bce"]:.4f} '
                      f'BCE INTERNA={inner_result["bce"]:.4f} '
                      f'LR={lr:g} tiempo={history[-1]["seconds"]:.1f}s', flush=True)
                if stale >= cfg['patience']:
                    break
            checkpoint = torch.load(weights/'best.pt', map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'], strict=True)
            fit_result, fit_patients, _ = evaluate(model, fit_eval, device)
            inner_result, inner_patients, inner_cuts = evaluate(model, inner_loader, device)
            threshold, threshold_metrics = select_threshold(inner_patients)
            decision = dict(best_epoch=best_epoch, checkpoint_sha256=sha(weights/'best.pt'),
                            protocol_sha256=base['protocol_sha256'], threshold=threshold,
                            selected_on='inner_only', inner_metrics=threshold_metrics,
                            frozen_utc=datetime.now(timezone.utc).isoformat())
            write_json(out/'decision_before_outer.json', decision)
            inner_patients.to_csv(out/'predictions_inner_patient.csv', index=False)
            inner_cuts.to_csv(out/'predictions_inner_cut.csv', index=False)
            fit_patients.to_csv(out/'predictions_fit_patient.csv', index=False)
            # Primera y única evaluación exterior: la decisión ya existe en disco.
            outer_loader = make_loader(frames['outer'], root, fold, 3)
            outer_result, outer_patients, outer_cuts = evaluate(model, outer_loader, device)
            verify(root)
            outer_patients.to_csv(out/'predictions_outer_patient.csv', index=False)
            outer_cuts.to_csv(out/'predictions_outer_cut.csv', index=False)
            summary.update(status='complete', best_epoch=best_epoch,
                           threshold=threshold, epochs_completed=len(history),
                           fit_at_best=fit_result, inner_at_best=inner_result,
                           outer_at_05=outer_result,
                           outer_at_inner_threshold=metrics(outer_patients, threshold),
                           decision=decision, seconds=time.perf_counter()-started,
                           ended_utc=datetime.now(timezone.utc).isoformat())
            write_json(out/'summary.json', summary)
            print(f'N01 {loss} fold={fold} CERRADO: época={best_epoch}, '
                  f'umbral INTERNO={threshold:.5f}, '
                  f'AUC EXTERIOR desarrollo={outer_result["metrics"]["roc_auc"]:.4f}. '
                  'TEST no evaluado.', flush=True)
        except BaseException as exc:
            summary.update(status='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed',
                           error=f'{type(exc).__name__}: {exc}', epochs_completed=len(history))
            write_json(out/'summary.json', summary)
            raise
        finally:
            if not smoke:
                refresh(root)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('prepare', 'verify', 'report'):
        sub.add_parser(name)
    for name in ('smoke', 'run'):
        p = sub.add_parser(name)
        p.add_argument('--loss', choices=['normal', 'weighted'], required=True)
        p.add_argument('--fold', type=int, choices=range(5), required=True)
        p.add_argument('--device', choices=['mps', 'cpu'], default='mps')
    args = parser.parse_args()
    if args.command == 'prepare':
        manifest = prepare()
        print('N01 congelado sin entrenamiento ni imágenes. Pacientes FIT/interna/exterior:')
        for fold, parts in manifest['splits'].items():
            print(f'fold {fold}: ' + ' / '.join(str(parts[n]['patients']) for n in ('fit', 'inner', 'outer')))
    elif args.command == 'verify':
        verify()
        print('N01 compatible con el protocolo congelado; no se ha entrenado ni evaluado test.')
    elif args.command == 'report':
        print(refresh(ROOT))
    else:
        run(args.loss, args.fold, args.device, smoke=args.command == 'smoke')


if __name__ == '__main__':
    main()
