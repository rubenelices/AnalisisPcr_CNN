"""Contraste: CNN propia de cuatro convoluciones, 97.809 parámetros.

Dos representaciones prefijadas, cinco folds; no búsqueda ni selección exterior.
Los runners y checkpoints de la campaña cerrada permanecen intactos.
"""
from src import source_path
import argparse
from dataclasses import asdict
from datetime import datetime, timezone, timedelta
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import time

import numpy as np
import pandas as pd
import torch
from torch import nn

from src.models.architectures import BaseCNNConfig, BreastPCRNet
from src.campaigns.campaign_20261015_cache import CachedDataset
from src.campaigns.campaign_20261015_fast import nested_split, select_threshold, enrich, write_json
from src.pipeline.data import load_samples
from src.evaluation.metrics import aggregate_by_patient, binary_metrics
from src.training.train import set_reproducibility

ROOT = Path('experiments/C16_cnn_97809')
WEIGHTS = Path('checkpoints/C16_cnn_97809')
SCHEMA = 'simple_cnn_97809_v1'
REPRESENTATIONS = ('raw', 'pre_differences')


def model_config(representation):
    if representation not in REPRESENTATIONS:
        raise ValueError('Representación fuera del contraste prefijado')
    return BaseCNNConfig(convolutions_per_block=1, pool_position='after_convolutions',
                         input_representation=representation)


def make_model(representation):
    model = BreastPCRNet(model_config(representation))
    assert sum(p.numel() for p in model.parameters()) == 97809
    return model


def config(fold, representation, debug=False):
    if fold not in range(5):
        raise ValueError('Fold inválido')
    model_config(representation)
    name = f'{representation}_fold{fold}' + ('_smoke' if debug else '')
    return dict(schema=SCHEMA, experiment_id=name, outer_fold=fold,
                representation=representation, seed=42, debug=debug, device='mps',
                batch_size=16, eval_batch_size=32, max_epochs=2 if debug else 19,
                patience=6, warmup_epochs=3, learning_rate=.001, weight_decay=.001,
                loss='normal', augmentation=None, aggregation='mean_probabilities',
                output_dir=str(ROOT/name/'resultados'), checkpoint_dir=str(WEIGHTS/name),
                model=json.loads(json.dumps(asdict(model_config(representation)))))


def source_hashes():
    sources = ['simple_cnn_comparison.py', 'architectures.py', 'phase_representation.py',
               'campaign_20261015_cache.py', 'campaign_20261015_fast.py', 'data.py',
               'metrics.py', 'train.py']
    return {n: hashlib.sha256(source_path(n).read_bytes()).hexdigest()
            for n in sources}


def batches(rows, training=False, seed=42):
    # Misma caché exacta y mismo orden/generador que referencia16.
    kwargs = dict(batch_size=16 if training else 32, shuffle=training, num_workers=0)
    if training:
        kwargs['generator'] = torch.Generator().manual_seed(seed)
    return torch.utils.data.DataLoader(CachedDataset(rows), **kwargs)


def patient_result(cuts, threshold=.5):
    patients = aggregate_by_patient(cuts.patient_id, cuts.target, cuts.probability)
    metrics = enrich(binary_metrics(patients.target, patients.probability, threshold))
    # binary_metrics conserva su convención histórica precision=0 si no hay positivas;
    # este indicador evita interpretar ese 0 como una precisión estimable.
    metrics['precision_defined'] = bool(metrics['confusion_matrix']['tp'] + metrics['confusion_matrix']['fp'])
    return patients, metrics


@torch.no_grad()
def evaluate(model, rows, device):
    model.eval()
    records = []
    for batch in batches(rows):
        probabilities = torch.sigmoid(model(batch['image'].to(device))).cpu().tolist()
        records.extend(dict(sample_id=sid, patient_id=pid, target=int(y), probability=p)
                       for sid, pid, y, p in zip(batch['sample_id'], batch['patient_id'],
                                                batch['target'].tolist(), probabilities))
    cuts = pd.DataFrame(records)
    if len(cuts) != len(rows) or cuts.sample_id.tolist() != rows.sample_id.tolist():
        raise ValueError('Inferencia desordenada/incompleta')
    patients, metrics = patient_result(cuts)
    p = np.clip(patients.probability.to_numpy(float), 1e-7, 1-1e-7)
    y = patients.target.to_numpy(float)
    loss = float(np.mean(-(y*np.log(p)+(1-y)*np.log1p(-p))))
    return dict(patient_probability_bce=loss, metrics=metrics), patients, cuts


def load_checkpoint(path, device='cpu'):
    ck = torch.load(path, map_location=device, weights_only=False)
    spec = ck['config']
    if spec.get('schema') != SCHEMA or ck.get('model_class') != 'BreastPCRNet_97809_v1':
        raise ValueError('Checkpoint ajeno a este contraste')
    if spec['model'] != json.loads(json.dumps(asdict(model_config(spec['representation'])))):
        raise ValueError('Configuración arquitectónica incoherente')
    model = make_model(spec['representation']).to(device)
    model.load_state_dict(ck['model_state_dict'], strict=True)
    if not all(torch.isfinite(x).all() for x in model.state_dict().values()):
        raise ValueError('Pesos no finitos')
    return model.eval(), ck


def run(spec, deadline):
    out, weights = Path(spec['output_dir']), Path(spec['checkpoint_dir'])
    if spec != config(spec['outer_fold'], spec['representation'], spec['debug']):
        raise ValueError('Config distinta del protocolo prefijado')
    if out.exists() or weights.exists():
        raise FileExistsError('No sobrescribir ni reanudar ejecuciones')
    device = torch.device(spec['device'])
    if not torch.backends.mps.is_available():
        raise RuntimeError('MPS local no disponible; sin fallback a nube')
    torch.set_num_threads(4)
    set_reproducibility(spec['seed'])
    rows = load_samples()
    rows = rows.loc[rows.split.eq('train')].reset_index(drop=True)
    fit, inner, outer = nested_split(rows, spec['outer_fold'])
    if spec['debug']:
        def subset(frame):
            ids = frame[['patient_id', 'pCR']].drop_duplicates().groupby('pCR').head(2).patient_id
            return frame.loc[frame.patient_id.isin(ids)].reset_index(drop=True)
        fit, inner, outer = map(subset, (fit, inner, outer))
    # Comprobar emparejamiento con referencia antes de entrenar; no leer sus métricas.
    reference = Path(f'local/campana_20261015/reference16_fold{spec["outer_fold"]}/resultados/split_manifest.json')
    reference_manifest = json.loads(reference.read_text())
    if not spec['debug']:
        for name, frame in [('fit', fit), ('inner', inner), ('outer', outer)]:
            if set(frame.patient_id) != set(reference_manifest[name]):
                raise ValueError('Pacientes distintas de la referencia')
    out.mkdir(parents=True)
    weights.mkdir(parents=True)
    write_json(out/'config_resolved.json', spec)
    write_json(out/'split_manifest.json', dict(
        **{k: x.patient_id.unique().tolist() for k, x in [('fit', fit), ('inner', inner), ('outer', outer)]},
        test_evaluated=False))
    hashes = source_hashes()
    model = make_model(spec['representation']).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=spec['learning_rate'], weight_decay=spec['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=.5, patience=3)
    train_loader = batches(fit, training=True, seed=spec['seed'])
    history, best, best_monitor, stale = [], -float('inf'), -float('inf'), 0
    start = time.monotonic()
    def check():
        if (ROOT/'STOP').exists() or datetime.now(timezone.utc) >= deadline:
            raise InterruptedError('Parada o límite de seguridad; conservar parcial')
    print('START', spec['experiment_id'], 'patients', *[x.patient_id.nunique() for x in (fit, inner, outer)], flush=True)
    try:
        for epoch in range(1, spec['max_epochs']+1):
            check()
            before = time.monotonic()
            lr = optimizer.param_groups[0]['lr']
            model.train()
            total, count = 0., 0
            for batch in train_loader:
                check()
                optimizer.zero_grad(set_to_none=True)
                logits = model(batch['image'].to(device))
                y = batch['target'].to(device)
                loss = nn.functional.binary_cross_entropy_with_logits(logits, y)
                if not torch.isfinite(loss):
                    raise ValueError('Pérdida no finita')
                loss.backward()
                optimizer.step()
                total += float(loss.detach().cpu())*len(y)
                count += len(y)
            evaluation, _, _ = evaluate(model, inner, device)
            row = dict(epoch=epoch, train_loss=total/count,
                       validation_loss=evaluation['patient_probability_bce'], learning_rate=lr,
                       patient_roc_auc=evaluation['metrics']['roc_auc'],
                       patient_pr_auc=evaluation['metrics']['pr_auc'])
            history.append(row)
            monitor = float(np.mean([x['patient_roc_auc'] for x in history[-3:]]))
            row.update(monitor_inner_auc_ma3=monitor, elapsed_seconds=time.monotonic()-before)
            pd.DataFrame(history).to_csv(out/'history.csv', index=False)
            ck = dict(model_class='BreastPCRNet_97809_v1', config=spec, epoch=epoch,
                      model_state_dict=model.state_dict(), source_sha256=hashes)
            torch.save(ck, weights/'last.pt')
            if epoch >= min(spec['warmup_epochs'], spec['max_epochs']):
                scheduler.step(monitor)
                if row['patient_roc_auc'] > best:
                    best = row['patient_roc_auc']
                    torch.save(ck, weights/'best.pt')
                if monitor > best_monitor:
                    best_monitor, stale = monitor, 0
                else:
                    stale += 1
            print(f'{spec["experiment_id"]} epoch={epoch} loss={row["train_loss"]:.4f} '
                  f'innerAUC={row["patient_roc_auc"]:.4f} AP={row["patient_pr_auc"]:.4f} '
                  f'MA3={monitor:.4f} lr={lr:g} sec={row["elapsed_seconds"]:.1f}', flush=True)
            if stale >= spec['patience']:
                break
        model, selected = load_checkpoint(weights/'best.pt', device)
        result = dict(status='complete', experiment_id=spec['experiment_id'],
                      scientific_result=not spec['debug'], debug_patient_subset=spec['debug'],
                      test_evaluated=False, representation=spec['representation'],
                      outer_fold=spec['outer_fold'], seed=spec['seed'], parameter_count=97809,
                      best_epoch=selected['epoch'], epochs_completed=len(history),
                      source_sha256=hashes, best_inner_smoothed_auc=best_monitor,
                      environment=dict(torch=torch.__version__, python=platform.python_version(),
                                       platform=platform.platform(), device=str(device)))
        for name, frame in [('fit', fit), ('inner', inner)]:
            ev, pred, cuts = evaluate(model, frame, device)
            result[name+'_eval'] = ev
            pred.to_csv(out/f'predictions_{name}.csv', index=False)
            cuts.to_csv(out/f'cut_predictions_{name}.csv', index=False)
        innerpred = pd.read_csv(out/'predictions_inner.csv')
        threshold = select_threshold(innerpred)
        result['selected_threshold'] = threshold
        result['threshold_rule'] = 'F1macro inner only; balanced accuracy, closeness0.5 ties'
        # Arquitectura/representación/épocas/umbral ya congelados antes del exterior.
        ev, pred, cuts = evaluate(model, outer, device)
        result['outer_eval'] = ev
        decision = enrich(binary_metrics(pred.target, pred.probability, threshold))
        decision['precision_defined'] = bool(decision['confusion_matrix']['tp']+decision['confusion_matrix']['fp'])
        result['outer_decision_metrics'] = decision
        result['elapsed_seconds'] = time.monotonic()-start
        result['host_maxrss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        result['mps_current_allocated_bytes'] = torch.mps.current_allocated_memory()
        result['mps_driver_allocated_bytes'] = torch.mps.driver_allocated_memory()
        result['memory_note'] = 'RSS máximo del proceso acumulado; MPS son valores al cierre, no pico.'
        result['checkpoint_sha256'] = hashlib.sha256((weights/'best.pt').read_bytes()).hexdigest()
        pred.to_csv(out/'predictions_outer.csv', index=False)
        cuts.to_csv(out/'cut_predictions_outer.csv', index=False)
        write_json(out/'summary.json', result)
        print('COMPLETE', spec['experiment_id'], 'outerAUC', ev['metrics']['roc_auc'], 'best', selected['epoch'], flush=True)
        return result
    except BaseException as error:
        write_json(out/'partial.json', dict(status='partial', error=str(error), epochs_completed=len(history)))
        raise
    finally:
        del model, optimizer, train_loader
        torch.mps.empty_cache()


def main(smoke=False):
    ROOT.mkdir(parents=True, exist_ok=True)
    lock = ROOT/'active_process.json'
    fd = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'w') as file:
        json.dump(dict(pid=os.getpid(), started_utc=datetime.now(timezone.utc).isoformat()), file)
    try:
        specs = [config(0, 'raw', True)] if smoke else [config(f, r) for r in REPRESENTATIONS for f in range(5)]
        deadline = datetime.now(timezone.utc)+timedelta(hours=3)
        for spec in specs:
            path = ROOT/'configs'/(spec['experiment_id']+'.json')
            path.parent.mkdir(exist_ok=True)
            if path.exists():
                if json.loads(path.read_text()) != spec:
                    raise ValueError('Config existente distinta')
            else:
                write_json(path, spec)
            summary_path = Path(spec['output_dir'])/'summary.json'
            if summary_path.exists():
                old = json.loads(summary_path.read_text())
                if old['source_sha256'] != source_hashes() or old['status'] != 'complete':
                    raise ValueError('No reutilizar resultados con fuentes distintas')
                continue
            run(spec, deadline)
        if not smoke:
            write_json(ROOT/'complete.json', dict(completed_utc=datetime.now(timezone.utc).isoformat(),
                                                experiments=10, test_evaluated=False))
    finally:
        lock.unlink()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke', action='store_true')
    main(parser.parse_args().smoke)
