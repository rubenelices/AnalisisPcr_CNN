"""P01 · lotes con más pacientes para BatchNorm. Manual y local; nunca evalúa test.

Hipótesis: con 2 pacientes por lote, BatchNorm normaliza cada canal usando solo
esas dos pacientes, de modo que la red apenas puede aprender diferencias
absolutas entre pacientes y se queda en la prevalencia (pérdida ≈ 0,605).

Único cambio frente a N01/E57: la composición del lote. Mismas ~20 imágenes por
paso y por época, pero de 10 pacientes × 2 cortes aleatorios en vez de
2 pacientes × 10 cortes. Red, particiones, optimizador, scheduler, criterio de
selección y umbral interno se reutilizan de N01 sin modificar sus fuentes.

Subcomandos:
  overfit --recipe e57|p01   memorizar 40 pacientes FIT (diagnóstico, no ciencia)
  overfit-report             figura comparando ambas recetas de overfit
  smoke/run --loss --fold [--variant p01|p02]   entrenamiento con la partición N01
  report [--variant]         tabla de ejecuciones completas

P02 = P01 + un único cambio: aumento geométrico sincronizado de las tres fases
solo en entrenamiento (volteos H/V y giro/traslación pequeños). Evaluación sin aumento.

P05 = P02 + un único cambio: tasa de aprendizaje 0,0001 (research/P05_LR_BAJO.md).
Guarda además las predicciones exteriores de cada época, solo para el análisis
secundario de época fija; la elección de época y umbral sigue siendo interna.
Retirado sin ejecutar: la tasa 0,0001 ya se probó (E12, E55).

P07 = P02 con semillas 17 y 123 además de la 42 (--seed), para promediar los 3 modelos
de cada fold (research/P07_SEMILLAS.md). La semilla no cambia las particiones.
"""
import argparse
import copy
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Sampler

from src.models.architectures import trainable_parameter_count
from src.pipeline.augmentation import AugmentationConfig, SharedRandomAffine
from src.pipeline.data import BreastDCEDataset
from src.training.learning_rate_schedule import LearningRateSchedule
from src.training.new_approach_protocol import (ROOT, TECH as N01_TECH, metrics, select_threshold,
                                       sha, split_rows, verify, write_json)
from src.models.patient_feature_model import PatientFeatureNet
from src.training.patient_training import PatientBatchSampler, _patient_groups, patient_level_pos_weight
from src.training.train import _environment, set_reproducibility
from src.training.train_new_approach import (evaluate, save_checkpoint, smoke_subset, state_sha,
                                    train_epoch, write_history)
from src.training.train_patient_features import load_spec

VARIANTS = {
    'p01': ('P01_lotes_pacientes', 'new_approach_p01_v1'),
    'p02': ('P02_aumento_geometrico', 'new_approach_p02_v1'),
    'p05': ('P05_lr_bajo', 'new_approach_p05_v1'),
}
TECH = Path('reports/new_approach/P01_lotes_pacientes')  # overfit y lock por defecto
SEEDS = (42, 17, 123)  # 42 es la de siempre; 17 y 123 solo para el ensemble P07


def variant_paths(variant):
    if variant not in VARIANTS:
        raise ValueError('Variante desconocida')
    name = VARIANTS[variant][0]
    return (Path(f'configs/new_approach/{name}.json'), Path(f'reports/new_approach/{name}'),
            Path(f'checkpoints/new_approach/{name}'))
PRIOR_ENTROPY = 0.6053  # BCE del predictor constante con 29,3 % pCR


class PatientCutSampler(Sampler[list[int]]):
    """Lotes de `patients_per_batch` pacientes con `cuts_per_patient` cortes al azar.

    Una época son `passes` recorridos de todas las pacientes; en cada recorrido
    cada paciente aparece una vez con cortes nuevos sin reemplazo. Con 10×2 y
    5 recorridos, una época ve las mismas imágenes que E57 (≈10 por paciente).
    """
    def __init__(self, rows, patients_per_batch, cuts_per_patient, passes, generator):
        for value in (patients_per_batch, cuts_per_patient, passes):
            if type(value) is not int or value < 1:
                raise ValueError('Tamaños del sampler deben ser enteros positivos')
        self.groups = _patient_groups(rows)
        self.patients_per_batch = patients_per_batch
        self.cuts_per_patient = cuts_per_patient
        self.passes = passes
        self.generator = generator

    def __iter__(self):
        for _ in range(self.passes):
            order = torch.randperm(len(self.groups), generator=self.generator).tolist()
            for start in range(0, len(order), self.patients_per_batch):
                batch = []
                for patient in order[start:start + self.patients_per_batch]:
                    cuts = self.groups[patient]
                    pick = torch.randperm(len(cuts), generator=self.generator)[:self.cuts_per_patient]
                    batch.extend(cuts[i] for i in pick.tolist())
                yield batch

    def __len__(self):
        return self.passes * math.ceil(len(self.groups) / self.patients_per_batch)


def load_config(root=ROOT, variant='p01'):
    cfg = json.loads((Path(root)/variant_paths(variant)[0]).read_text())
    if cfg.get('schema') != VARIANTS[variant][1]:
        raise ValueError(f'Configuración {variant} inesperada')
    return cfg


class SyncFlipAffine:
    """Mismo volteo y la misma rejilla afín para PRE, EARLY y LATE (nunca por canal).

    Usa el generador global de torch, sembrado con la semilla 42 (workers=0).
    """
    def __init__(self, cfg):
        self.h = float(cfg['horizontal_flip_probability'])
        self.v = float(cfg['vertical_flip_probability'])
        if not (0 <= self.h <= 1 and 0 <= self.v <= 1):
            raise ValueError('Probabilidades de volteo fuera de [0,1]')
        self.affine = SharedRandomAffine(AugmentationConfig(
            enabled=True, max_rotation_degrees=float(cfg['max_rotation_degrees']),
            max_translation_fraction=float(cfg['max_translation_fraction'])))

    def __call__(self, image):
        if torch.rand(()).item() < self.h:
            image = torch.flip(image, dims=(-1,))
        if torch.rand(()).item() < self.v:
            image = torch.flip(image, dims=(-2,))
        return self.affine(image.contiguous())


def train_transform(cfg):
    aug = cfg.get('augmentation')
    return SyncFlipAffine(aug) if aug else None


def training_loader(rows, root, recipe, cfg, seed=42, transform=None):
    ds = BreastDCEDataset(rows, Path(root)/'breastdcedl', transform=transform)
    generator = torch.Generator().manual_seed(seed)
    if recipe == 'e57':
        sampler = PatientBatchSampler(rows, 2, generator)
    elif recipe == 'p01':
        b = cfg['batch']
        sampler = PatientCutSampler(rows, b['patients_per_batch'], b['cuts_per_patient'],
                                    b['passes_per_epoch'], generator)
    else:
        raise ValueError('Receta desconocida')
    return DataLoader(ds, batch_sampler=sampler, num_workers=0, generator=generator)


def eval_loader(rows, root, seed):
    if not rows.split.eq('train').all():
        raise ValueError('P01 no admite test')
    ds = BreastDCEDataset(rows, Path(root)/'breastdcedl')
    return DataLoader(ds, batch_size=16, shuffle=False, num_workers=0,
                      generator=torch.Generator().manual_seed(seed))


def device_for(name):
    if name not in ('mps', 'cpu', 'cuda'):
        raise ValueError('Dispositivo no admitido')
    if name == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('MPS no disponible; no se cambia a CPU silenciosamente')
    if name == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('GPU (CUDA/ROCm) no disponible; no se cambia a CPU silenciosamente')
    return torch.device(name)


def new_model(root, device, seed=42):
    set_reproducibility(seed)
    torch.set_num_threads(4)
    spec, original_cfg = load_spec(Path(root)/'configs/patient_features/E57_patient_feature_mean_mac.json')
    model = PatientFeatureNet(original_cfg.model).to(device)
    if trainable_parameter_count(model) != 298161:
        raise ValueError('La CNN no coincide con E57 original')
    return spec, original_cfg, model


@contextmanager
def exclusive_lock(root, tech=TECH):
    path = Path(root)/tech/'execution.lock'
    path.parent.mkdir(parents=True, exist_ok=True)
    # Un lock existente no se elimina automáticamente, incluso si no hay PID activo.
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        with os.fdopen(fd, 'w') as stream:
            stream.write(f'pid={os.getpid()}\n')
        yield
    finally:
        path.unlink()


def overfit_subset(fit, n_per_class):
    patients = fit[['patient_id', 'pCR']].drop_duplicates('patient_id').sort_values('patient_id')
    ids = patients.groupby('pCR').sample(n=n_per_class, random_state=42).patient_id
    return fit.loc[fit.patient_id.isin(ids)].reset_index(drop=True)


def overfit(recipe, device_name='mps', root=ROOT):
    """Ambas recetas: mismos 40 pacientes, mismos pasos (20/época) e imágenes por paso."""
    root = Path(root)
    cfg = load_config(root)
    _, rows = verify(root)
    o = cfg['overfit']
    subset = overfit_subset(split_rows(rows, 0)['fit'], o['patients_per_class'])
    out = root/TECH/'overfit'/recipe
    if out.exists():
        raise FileExistsError('Overfit existente: conservarlo; no se sobrescribe')
    device = device_for(device_name)
    with exclusive_lock(root):
        _, _, model = new_model(root, device)
        loader = training_loader(subset, root, recipe, cfg)
        evaluation = eval_loader(subset, root, 20000)
        criterion = nn.BCEWithLogitsLoss()
        # LR constante: aquí solo se pregunta si la receta puede memorizar.
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.001)
        out.mkdir(parents=True)
        base = dict(schema='p01_overfit_v1', recipe=recipe, scientific_result=False,
                    test_evaluated=False, patients=int(subset.patient_id.nunique()),
                    cuts=len(subset), steps_per_epoch=len(loader.batch_sampler),
                    source_sha256=sha(root/'src/training/train_p01.py'),
                    started_utc=datetime.now(timezone.utc).isoformat(),
                    environment=_environment(device))
        history = []
        for epoch in range(1, o['epochs']+1):
            tick = time.perf_counter()
            online = train_epoch(model, loader, optimizer, criterion, device)
            result, patients, _ = evaluate(model, evaluation, device)
            history.append(dict(epoch=epoch, train_online_objective=online,
                                eval_bce=result['bce'], eval_auc=result['metrics']['roc_auc'],
                                eval_cut_auc=result['cut_metrics_secondary']['roc_auc'],
                                prob_std=float(patients.probability.std()),
                                seconds=time.perf_counter()-tick))
            write_history(out/'history.csv', history)
            h = history[-1]
            print(f'OVERFIT {recipe} época={epoch:02d} pérdida online={online:.4f} '
                  f'(constante={PRIOR_ENTROPY}) AUC eval paciente={h["eval_auc"]:.4f} '
                  f'corte={h["eval_cut_auc"]:.4f} std prob={h["prob_std"]:.3f} '
                  f'tiempo={h["seconds"]:.1f}s', flush=True)
        write_json(out/'summary.json', dict(base, status='complete', final=history[-1],
                                            ended_utc=datetime.now(timezone.utc).isoformat()))
    return history


def overfit_report(root=ROOT):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root = Path(root)
    runs = {r: pd.read_csv(root/TECH/'overfit'/r/'history.csv')
            for r in ('e57', 'p01') if (root/TECH/'overfit'/r/'history.csv').is_file()}
    if not runs:
        raise FileNotFoundError('No hay ejecuciones de overfit')
    labels = dict(e57='E57: 2 pacientes × 10 cortes', p01='P01: 10 pacientes × 2 cortes')
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for recipe, h in runs.items():
        axes[0].plot(h.epoch, h.train_online_objective, marker='o', label=labels[recipe])
        axes[1].plot(h.epoch, h.eval_auc, marker='o', label=labels[recipe])
        axes[2].plot(h.epoch, h.prob_std, marker='o', label=labels[recipe])
    axes[0].axhline(PRIOR_ENTROPY, ls='--', c='grey', label='predictor constante')
    axes[1].axhline(.5, ls='--', c='grey')
    for ax, title in zip(axes, ('Pérdida online de train', 'AUC paciente en eval() (mismos 40)',
                                'Dispersión de probabilidades')):
        ax.set_title(title); ax.set_xlabel('época'); ax.grid(alpha=.3); ax.legend()
    fig.suptitle('P01 · ¿Puede cada receta memorizar 40 pacientes? (diagnóstico, no rendimiento)')
    fig.tight_layout()
    path = root/TECH/'overfit'/'comparacion_overfit.png'
    fig.savefig(path, dpi=130)
    return path


def run_dirs(root, loss, fold, smoke, variant='p01', device_name='mps', seed=42):
    if loss not in ('normal', 'weighted') or type(fold) is not int or fold not in range(5):
        raise ValueError('Pérdida/fold inválidos')
    if seed not in SEEDS:
        raise ValueError('Semilla no prevista (P07: 42, 17, 123)')
    if smoke and fold != 0:
        raise ValueError('El smoke técnico prefijado es fold 0')
    name = f'{loss}_fold_{fold}_seed_{seed}'
    # La GPU de la universidad escribe en carpetas propias: no se mezcla con el Mac
    # (CUDA/ROCm y MPS no dan números idénticos con la misma semilla).
    mode = ('smoke' if smoke else 'runs') + ('_cuda' if device_name == 'cuda' else '')
    _, tech, weights = variant_paths(variant)
    return Path(root)/tech/mode/name, Path(root)/weights/mode/name


def run(loss, fold, device_name='mps', smoke=False, root=ROOT, variant='p01', seed=42):
    root = Path(root)
    cfg = load_config(root, variant)
    config_path, tech, _ = variant_paths(variant)
    tag = variant.upper()
    manifest, rows = verify(root)  # mismas particiones y fuentes congeladas que N01
    out, weights = run_dirs(root, loss, fold, smoke, variant, device_name, seed)
    if out.exists() or weights.exists():
        raise FileExistsError('Ejecución existente: conservarla; no se sobrescribe ni reanuda')
    if not smoke:
        smoke_path = run_dirs(root, loss, 0, True, variant, device_name, seed)[0]/'summary.json'
        if not smoke_path.is_file() or json.loads(smoke_path.read_text()).get('status') != 'complete':
            raise RuntimeError(f'Falta smoke técnico completo de la condición {loss} (fold 0)')
    device = device_for(device_name)
    frames = split_rows(rows, fold)
    if smoke:
        frames = smoke_subset(frames)
    with exclusive_lock(root, tech):
        verify(root)
        spec, original_cfg, model = new_model(root, device, seed)
        initial_sha = state_sha(model)
        fit_loader = training_loader(frames['fit'], root, 'p01', cfg, seed=seed,
                                     transform=train_transform(cfg))
        fit_eval = eval_loader(frames['fit'], root, 10000+10*fold+1)
        inner_loader = eval_loader(frames['inner'], root, 10000+10*fold+2)
        pos_weight = patient_level_pos_weight(frames['fit']) if loss == 'weighted' else 1.
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg['learning_rate']),
                                      weight_decay=float(cfg['weight_decay']))
        outer_by_epoch = cfg.get('outer_predictions_per_epoch', False)
        if outer_by_epoch:
            outer_loader = eval_loader(frames['outer'], root, 10000+10*fold+3)
        scheduler = LearningRateSchedule(optimizer, original_cfg.training)
        checkpoint_spec = copy.deepcopy(spec)
        checkpoint_spec['experiment'].update(
            experiment_id=f'{tag}_{loss}_fold_{fold}_seed_{seed}' + ('_smoke' if smoke else ''),
            output_dir=str(out.relative_to(root)), checkpoint_dir=str(weights.relative_to(root)))
        checkpoint_spec['experiment']['training']['loss'] = loss
        checkpoint_spec['experiment']['data']['validation_fold'] = fold
        out.mkdir(parents=True)
        weights.mkdir(parents=True)
        base = dict(schema='new_approach_p01_run_v1', variant=variant, loss=loss, fold=fold, seed=seed,
                    scientific_result=not smoke, test_evaluated=False, batch=cfg['batch'],
                    augmentation=cfg.get('augmentation'),
                    n01_protocol_sha256=sha(root/N01_TECH/'protocol/manifest.json'),
                    source_sha256=sha(root/'src/training/train_p01.py'),
                    config_sha256=sha(root/config_path),
                    initial_state_sha256=initial_sha, pos_weight=pos_weight,
                    parameter_count=298161, steps_per_epoch=len(fit_loader.batch_sampler),
                    evaluation_aggregation='mean_cut_probabilities',
                    counts={n: dict(patients=f.patient_id.nunique(), cuts=len(f),
                                    positive=int(f.drop_duplicates('patient_id').pCR.sum()))
                            for n, f in frames.items()},
                    started_utc=datetime.now(timezone.utc).isoformat(),
                    environment=_environment(device))
        write_json(out/'config_resolved.json', dict(p01=cfg, n01_manifest=manifest, run=base))
        history, best, best_epoch, stale = [], -float('inf'), 0, 0
        started = time.perf_counter()
        summary = dict(base, status='running', epochs_completed=0)
        write_json(out/'summary.json', summary)
        try:
            for epoch in range(1, (1 if smoke else cfg['epochs'])+1):
                tick = time.perf_counter()
                lr = scheduler.start_epoch(epoch)
                online = train_epoch(model, fit_loader, optimizer, criterion, device)
                fit_result, fit_patients, _ = evaluate(model, fit_eval, device)
                inner_result, _, _ = evaluate(model, inner_loader, device)
                score = inner_result['metrics']['roc_auc']
                scheduler.finish_epoch(score)
                history.append(dict(
                    epoch=epoch, train_online_objective=online,
                    fit_bce=fit_result['bce'], inner_bce=inner_result['bce'],
                    fit_auc=fit_result['metrics']['roc_auc'], inner_auc=score,
                    fit_cut_auc=fit_result['cut_metrics_secondary']['roc_auc'],
                    inner_cut_auc=inner_result['cut_metrics_secondary']['roc_auc'],
                    inner_ap=inner_result['metrics']['pr_auc'],
                    fit_prob_std=float(fit_patients.probability.std()),
                    learning_rate=lr, seconds=time.perf_counter()-tick))
                if outer_by_epoch:
                    # Solo para el análisis secundario de época fija: no se imprime ni decide nada.
                    _, epoch_outer, _ = evaluate(model, outer_loader, device)
                    epoch_outer.assign(epoch=epoch).to_csv(
                        out/'predictions_outer_patient_by_epoch.csv', index=False,
                        mode='a', header=epoch == 1)
                checkpoint = dict(model_class='PatientFeatureNet_v1', config=checkpoint_spec,
                                  epoch=epoch, model_state_dict=model.state_dict(),
                                  optimizer_state_dict=optimizer.state_dict(),
                                  scheduler_state_dict=scheduler.state_dict(),
                                  p01=dict(loss=loss, fold=fold, pos_weight=pos_weight,
                                           batch=cfg['batch']))
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
                h = history[-1]
                print(f'{tag} {loss} fold={fold} época={epoch:02d} pérdida online={online:.4f} '
                      f'AUC FIT-eval={h["fit_auc"]:.4f} AUC INTERNA={score:.4f} '
                      f'(corte FIT={h["fit_cut_auc"]:.4f} INTERNA={h["inner_cut_auc"]:.4f}) '
                      f'BCE FIT={h["fit_bce"]:.4f} INTERNA={h["inner_bce"]:.4f} '
                      f'LR={lr:g} tiempo={h["seconds"]:.1f}s', flush=True)
                if stale >= cfg['patience']:
                    break
            checkpoint = torch.load(weights/'best.pt', map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'], strict=True)
            fit_result, fit_patients, _ = evaluate(model, fit_eval, device)
            inner_result, inner_patients, inner_cuts = evaluate(model, inner_loader, device)
            threshold, threshold_metrics = select_threshold(inner_patients)
            decision = dict(best_epoch=best_epoch, checkpoint_sha256=sha(weights/'best.pt'),
                            threshold=threshold, selected_on='inner_only',
                            inner_metrics=threshold_metrics,
                            frozen_utc=datetime.now(timezone.utc).isoformat())
            write_json(out/'decision_before_outer.json', decision)
            inner_patients.to_csv(out/'predictions_inner_patient.csv', index=False)
            inner_cuts.to_csv(out/'predictions_inner_cut.csv', index=False)
            fit_patients.to_csv(out/'predictions_fit_patient.csv', index=False)
            # Primera y única evaluación exterior: la decisión ya existe en disco.
            outer_loader = eval_loader(frames['outer'], root, 10000+10*fold+3)
            outer_result, outer_patients, outer_cuts = evaluate(model, outer_loader, device)
            verify(root)
            outer_patients.to_csv(out/'predictions_outer_patient.csv', index=False)
            outer_cuts.to_csv(out/'predictions_outer_cut.csv', index=False)
            summary.update(status='complete', best_epoch=best_epoch, threshold=threshold,
                           epochs_completed=len(history), fit_at_best=fit_result,
                           inner_at_best=inner_result, outer_at_05=outer_result,
                           outer_at_inner_threshold=metrics(outer_patients, threshold),
                           decision=decision, seconds=time.perf_counter()-started,
                           ended_utc=datetime.now(timezone.utc).isoformat())
            write_json(out/'summary.json', summary)
            print(f'{tag} {loss} fold={fold} CERRADO: época={best_epoch}, umbral INTERNO={threshold:.5f}, '
                  f'AUC EXTERIOR paciente={outer_result["metrics"]["roc_auc"]:.4f}, '
                  f'corte={outer_result["cut_metrics_secondary"]["roc_auc"]:.4f}. TEST no evaluado.',
                  flush=True)
        except BaseException as exc:
            summary.update(status='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed',
                           error=f'{type(exc).__name__}: {exc}', epochs_completed=len(history))
            write_json(out/'summary.json', summary)
            raise
    return summary


def report(root=ROOT, variant='p01', mode='runs'):
    rows = []
    for path in sorted((Path(root)/variant_paths(variant)[1]/mode).glob('*/summary.json')):
        s = json.loads(path.read_text())
        if s.get('status') != 'complete':
            continue
        rows.append(dict(loss=s['loss'], fold=s['fold'], epoca=s['best_epoch'],
                         auc_fit=s['fit_at_best']['metrics']['roc_auc'],
                         auc_interna=s['inner_at_best']['metrics']['roc_auc'],
                         auc_exterior=s['outer_at_05']['metrics']['roc_auc'],
                         auc_exterior_corte=s['outer_at_05']['cut_metrics_secondary']['roc_auc'],
                         ap_exterior=s['outer_at_05']['metrics']['pr_auc'],
                         ba_umbral_interno=s['outer_at_inner_threshold']['balanced_accuracy']))
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('overfit')
    p.add_argument('--recipe', choices=['e57', 'p01'], required=True)
    p.add_argument('--device', choices=['mps', 'cpu'], default='mps')
    sub.add_parser('overfit-report')
    p = sub.add_parser('report')
    p.add_argument('--variant', choices=sorted(VARIANTS), default='p01')
    p.add_argument('--gpu', action='store_true', help='leer las ejecuciones hechas en GPU (runs_cuda)')
    for name in ('smoke', 'run'):
        p = sub.add_parser(name)
        p.add_argument('--variant', choices=sorted(VARIANTS), default='p01')
        p.add_argument('--loss', choices=['normal', 'weighted'], required=True)
        p.add_argument('--fold', type=int, choices=range(5), required=True)
        p.add_argument('--device', choices=['mps', 'cpu', 'cuda'], default='mps')
        p.add_argument('--seed', type=int, choices=SEEDS, default=42)
    args = parser.parse_args()
    if args.command == 'overfit':
        overfit(args.recipe, args.device)
    elif args.command == 'overfit-report':
        print(overfit_report())
    elif args.command == 'report':
        table = report(variant=args.variant, mode='runs_cuda' if args.gpu else 'runs')
        print(table.round(4).to_string(index=False) if len(table) else 'Sin ejecuciones completas')
    else:
        run(args.loss, args.fold, args.device, smoke=args.command == 'smoke', variant=args.variant,
            seed=args.seed)


if __name__ == '__main__':
    main()
