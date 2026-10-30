"""E58: experimento multimodal aislado, nunca usa test ni pesos previos."""
from src import source_path
import argparse
from dataclasses import asdict
import hashlib
import json
import math
import time
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.models.architectures import trainable_parameter_count, parameter_breakdown
from src.pipeline.clinical_features import (CLINICAL_COLUMNS, ClinicalPreprocessor, ClinicalImageDataset,
                                  load_patient_clinical)
from src.pipeline.data import BreastDCEDataset, load_samples, split_by_patient_fold, limit_splits_for_debug, seed_worker
from src.training.experiment_config import experiment_config_from_dict
from src.models.multimodal_model import MultimodalPCRNet
from src.training.train import set_reproducibility, _environment, _history_row, _write_history, build_criterion
from src.training.learning_rate_schedule import LearningRateSchedule
from src.evaluation.metrics import cut_and_patient_metrics
from src.evaluation.evaluation import EvaluationConfig, save_evaluation_artifacts
from src.training.training_report import build_training_diagnostics, save_diagnostics_json, save_training_dashboard
from src.reporting.patient_visual_report import save_patient_visual_report
from src.training.moderate_positive_weight import build_multimodal_criterion, validate_positive_weight_exponent
from src.pipeline.augmentation import build_train_transform


def load_spec(path):
    spec = json.loads(Path(path).read_text())
    if spec.get('schema') not in {'multimodal_e19_v1','clinical_only_e58_v1','multimodal_patient_e58_v1','multimodal_compact_e58_v1','multimodal_basic_regularized_v1'} or spec.get('clinical_columns') != list(CLINICAL_COLUMNS):
        raise ValueError('Esquema clínico no admitido: usar solo cuatro variables prefijadas')
    patient_protocol=spec['schema']=='multimodal_patient_e58_v1'
    cfg = experiment_config_from_dict(spec['experiment'])
    validate_positive_weight_exponent(spec.get('positive_weight_exponent',1.),cfg.training.loss)
    pooling=spec.get('image_pooling','average')
    if type(pooling) is not str or pooling not in {'average','enhancement_weighted'}:
        raise ValueError('image_pooling inválido')
    if spec['schema']=='clinical_only_e58_v1' and pooling!='average':
        raise ValueError('Clínica sola no admite pooling de imágenes')
    hidden=spec.get('fusion_hidden',0)
    normalization=spec.get('feature_normalization','none')
    if type(normalization) is not str or normalization not in {'none','l2_fixed'}:
        raise ValueError('Normalización de features inválida')
    if normalization!='none' and (spec['schema']=='clinical_only_e58_v1' or hidden!=0 or pooling!='average'):
        raise ValueError('Normalización fija exige fusión E58 original')
    occlusion=spec.get('image_occlusion')
    if occlusion is not None:
        from src.pipeline.synchronized_occlusion import SynchronizedOcclusion
        if not isinstance(occlusion,dict) or set(occlusion)!={'probability','side'}:
            raise ValueError('Oclusión exige probability y side explícitos')
        SynchronizedOcclusion(**occlusion)
        if spec['schema']=='clinical_only_e58_v1':
            raise ValueError('Clínica sola no admite oclusión de imágenes')
    if type(hidden) is not int or hidden not in (0,32):
        raise ValueError('fusion_hidden exige 0 o 32')
    if spec['schema']=='clinical_only_e58_v1' and hidden!=0:
        raise ValueError('Clínica sola no admite fusión multimodal')
    ratio=spec.get('image_lr_factor',1.)
    decay_factor=spec.get('image_weight_decay_factor',1.)
    if type(decay_factor) not in (int,float) or not math.isfinite(decay_factor) or decay_factor<=0:
        raise ValueError('image_weight_decay_factor exige número finito positivo')
    if spec['schema']=='clinical_only_e58_v1' and decay_factor!=1:
        raise ValueError('Clínica sola no admite regularización de rama imagen')
    if type(ratio) not in (int,float) or not math.isfinite(ratio) or not 0<ratio<=1:
        raise ValueError('image_lr_factor exige número finito en (0,1]')
    if ratio!=1 and cfg.training.lr_schedule.policy!='plateau':
        raise ValueError('LR por rama exige Plateau para preservar la proporción')
    active=spec.get('active_clinical_columns',list(CLINICAL_COLUMNS))
    basic_clinical = (active == ['age','menopause']
                      and spec.get('clinical_input_columns') == ['age','menopause'])
    if spec['schema']=='clinical_only_e58_v1' and (ratio!=1 or
            (active!=list(CLINICAL_COLUMNS) and not basic_clinical)):
        raise ValueError('Control clínica sola: cuatro variables o entrada básica explícita, LR sin ramas')
    if (not isinstance(active,list) or not active or len(set(active))!=len(active)
            or not set(active).issubset(CLINICAL_COLUMNS)):
        raise ValueError('Lista de variables clínicas activas inválida')
    inputs=spec.get('clinical_input_columns')
    if inputs is not None and (inputs!=active or not isinstance(inputs,list)
            or not set(inputs).issubset({'age','menopause'})
            or spec['schema'] not in {'multimodal_e19_v1','clinical_only_e58_v1','multimodal_basic_regularized_v1'}):
        raise ValueError('Entrada clínica básica exige solo edad/menopausia activas')
    if ((not patient_protocol and (cfg.training.objective != 'cut' or cfg.data.train_patients_per_batch is not None))
            or cfg.training.mixed_precision or cfg.training.loss not in {'normal','weighted'}
            or cfg.training.ema.enabled or cfg.training.sam.enabled or cfg.training.mixup.enabled
            or cfg.training.label_smoothing.enabled
            or cfg.data.train_patient_fraction != 1
            or (cfg.model.input_representation != 'pre_differences'
                and not (spec['schema']=='multimodal_e19_v1'
                         and inputs==['age','menopause']
                         and cfg.model.input_representation=='raw'
                         and pooling=='average' and normalization=='none' and hidden==0))
            or cfg.model.input_normalization != 'none' or cfg.training.monitor != 'patient_roc_auc'):
        raise ValueError('Multimodal exige E19 sin EMA/AMP ni mezcla de objetivos')
    if spec['schema']=='clinical_only_e58_v1' and any((cfg.data.augmentation.enabled,
            cfg.data.intensity_augmentation.enabled,cfg.data.horizontal_flip.enabled)):
        raise ValueError('Clínica sola no admite aumentos de imágenes')
    field=spec.get('image_intensity_field')
    if field is not None:
        from src.pipeline.shared_intensity_field import SharedIntensityField
        if not isinstance(field,dict) or set(field)!={'max_amplitude'}:
            raise ValueError('Campo exige max_amplitude explícita')
        SharedIntensityField(**field)
        if spec['schema']=='clinical_only_e58_v1':
            raise ValueError('Clínica sola no admite campo de imágenes')
        if (normalization!='none' or pooling!='average' or hidden!=0 or occlusion is not None
                or cfg.training.loss!='normal' or ratio!=1 or decay_factor!=1
                or any((cfg.data.augmentation.enabled,cfg.data.intensity_augmentation.enabled,
                        cfg.data.horizontal_flip.enabled))):
            raise ValueError('Campo exige E58 sin otros cambios')
    if patient_protocol:
        if (cfg.data.train_patients_per_batch!=2 or cfg.training.loss!='normal'
                or hidden!=0 or pooling!='average' or normalization!='none'
                or ratio!=1 or decay_factor!=1 or occlusion is not None or field is not None
                or active!=list(CLINICAL_COLUMNS)
                or any((cfg.data.augmentation.enabled,cfg.data.intensity_augmentation.enabled,
                        cfg.data.horizontal_flip.enabled))):
            raise ValueError('Protocolo paciente exige E58 original y dos pacientes completas/lote')
    compact=spec['schema']=='multimodal_compact_e58_v1'
    regularized=spec['schema']=='multimodal_basic_regularized_v1'
    if regularized:
        from src.training.basic_regularized import validate_basic_regularized
        validate_basic_regularized(spec, cfg)
    if (cfg.model.channels==(8,16,32,64)) != (compact or regularized):
        raise ValueError('Anchura compacta exige su esquema propio')
    if compact:
        from src.models.architectures import BaseCNNConfig
        expected=BaseCNNConfig(channels=(8,16,32,64),pool_position='between_convolutions',input_representation='pre_differences')
        if (cfg.model!=expected or hidden!=0 or pooling!='average' or normalization!='none'
                or ratio!=1 or decay_factor!=1 or occlusion is not None or field is not None
                or active!=list(CLINICAL_COLUMNS) or cfg.training.loss!='normal'
                or any((cfg.data.augmentation.enabled,cfg.data.intensity_augmentation.enabled,cfg.data.horizontal_flip.enabled))):
            raise ValueError('Compacta exige solo cambio de anchura sobre E58')
    return spec, cfg


def build_image_datasets(train_rows, validation_rows, cfg):
    """Aumentos SOLO en train; fases juntas antes de diferencias dentro de CNN.

    Los valores desactivados conservan el camino original y el consumo de RNG.
    Evaluación e inferencia nunca reciben transformaciones aleatorias.
    """
    transform=build_train_transform(cfg.data.augmentation,cfg.data.intensity_augmentation,
                                    cfg.data.horizontal_flip)
    return (BreastDCEDataset(train_rows,cfg.data.root,transform=transform),
            BreastDCEDataset(validation_rows,cfg.data.root))


def build_optimizer(model,cfg,spec):
    ratio=spec.get('image_lr_factor',1.)
    decay_factor=spec.get('image_weight_decay_factor',1.)
    if ratio==1 and decay_factor==1:
        # Ruta original E58 sin agrupación adicional ni cambios de RNG.
        return torch.optim.AdamW(model.parameters(),lr=cfg.training.learning_rate,
                                 weight_decay=cfg.training.weight_decay)
    groups=[dict(params=list(model.image_encoder.parameters()),
                 lr=cfg.training.learning_rate*ratio,
                 weight_decay=cfg.training.weight_decay*decay_factor,modality='image'),
            dict(params=[*model.clinical_branch.parameters(),*model.fusion.parameters()],
                 lr=cfg.training.learning_rate,modality='clinical_and_fusion')]
    return torch.optim.AdamW(groups,weight_decay=cfg.training.weight_decay)


def train_epoch(model, loader, optimizer, device, criterion=None, objective='cut'):
    model.train()
    total, count = 0., 0
    for batch in loader:
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch['image'].to(device), batch['clinical'].to(device))
        y = batch['target'].to(device)
        if objective=='cut':
            loss = (nn.functional.binary_cross_entropy_with_logits(logits, y)
                    if criterion is None else criterion(logits,y))
            units=len(y)
        else:
            from src.training.patient_training import objective_loss
            loss,units=objective_loss(logits,y,batch['patient_id'],
                                     criterion if criterion is not None else nn.BCEWithLogitsLoss(),objective)
        loss.backward()
        optimizer.step()
        total += float(loss.detach().cpu()) * units
        count += units
    return total / count


@torch.no_grad()
def evaluate(model, loader, device, threshold, criterion=None, objective='cut'):
    model.eval()
    total, count, rows = 0., 0, []
    for batch in loader:
        y = batch['target'].to(device)
        z = model(batch['image'].to(device), batch['clinical'].to(device))
        if objective=='cut':
            loss = (nn.functional.binary_cross_entropy_with_logits(z, y)
                    if criterion is None else criterion(z,y))
            units=len(y)
        else:
            from src.training.patient_training import objective_loss
            loss,units=objective_loss(z,y,batch['patient_id'],
                                     criterion if criterion is not None else nn.BCEWithLogitsLoss(),objective)
        total += float(loss.cpu()) * units
        count += units
        scores = torch.sigmoid(z).cpu().tolist()
        rows.extend(dict(sample_id=batch['sample_id'][i], patient_id=batch['patient_id'][i],
                         target=float(batch['target'][i]), probability=score,
                         slice_index=int(batch['slice_index'][i]), split=batch['split'][i],
                         fold=int(batch['fold'][i])) for i, score in enumerate(scores))
    frame = pd.DataFrame(rows)
    metrics = cut_and_patient_metrics(frame.patient_id, frame.target, frame.probability,
                                     threshold=threshold, aggregation='mean')
    return total / count, metrics, frame


def run(path, device_name):
    spec, cfg = load_spec(path)
    device = torch.device(device_name)
    if device.type == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('MPS no disponible; sin fallback CPU')
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA/ROCm no disponible')
    if device.type not in {'cpu','mps','cuda'}:
        raise ValueError('Dispositivo inválido')
    out, weights = Path(cfg.output_dir), Path(cfg.checkpoint_dir)
    for folder, prefix in ((out,'reports'),(weights,'checkpoints')):
        allowed=folder.is_relative_to(prefix) or (prefix=='reports'
            and any(folder.is_relative_to(base) for base in ('experiments','experiments'))
            and 'resultados' in folder.parts)
        if folder.is_absolute() or '..' in folder.parts or not allowed:
            raise ValueError('Ruta de salida inválida')
        if folder.exists():
            raise FileExistsError(f'Conservar {folder}; no overwrite/resume')
    set_reproducibility(cfg.training.seed)
    splits = limit_splits_for_debug(split_by_patient_fold(load_samples(cfg.data.root),cfg.data.validation_fold),
        cfg.data.train_patients_per_class,cfg.data.validation_patients_per_class,cfg.training.seed)
    train_ids = splits.train.patient_id.unique().tolist()
    val_ids = splits.validation.patient_id.unique().tolist()
    clinical = (load_patient_clinical(cfg.data.root, train_ids + val_ids)
                if 'clinical_input_columns' not in spec else
                load_patient_clinical(cfg.data.root, train_ids + val_ids,spec['clinical_input_columns']))
    train_clinical = clinical[clinical.pid.isin(train_ids)].copy()
    prep = ClinicalPreprocessor.fit(train_clinical,spec.get('active_clinical_columns',CLINICAL_COLUMNS))
    clinical_only = spec['schema']=='clinical_only_e58_v1'
    if clinical_only:
        from src.models.clinical_only import ClinicalOnlyPCRNet, ClinicalCutDataset
        ds_train = ClinicalCutDataset(splits.train,clinical,prep)
        ds_val = ClinicalCutDataset(splits.validation,clinical,prep)
    else:
        train_images,val_images=build_image_datasets(splits.train,splits.validation,cfg)
        ds_train = ClinicalImageDataset(train_images,clinical,prep)
        ds_val = ClinicalImageDataset(val_images,clinical,prep)
    if spec.get('image_occlusion') is not None:
        from src.pipeline.synchronized_occlusion import SynchronizedOcclusion, OccludedTrainingDataset
        ds_train=OccludedTrainingDataset(ds_train,SynchronizedOcclusion(**spec['image_occlusion']))
    if spec.get('image_intensity_field') is not None:
        from src.pipeline.shared_intensity_field import SharedIntensityField,IntensityFieldTrainingDataset
        ds_train=IntensityFieldTrainingDataset(ds_train,SharedIntensityField(**spec['image_intensity_field']))
    common = dict(batch_size=cfg.data.batch_size,num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory and device.type=='cuda',worker_init_fn=seed_worker,
        persistent_workers=cfg.data.num_workers>0)
    generator = torch.Generator().manual_seed(cfg.training.seed)
    train_loader = DataLoader(ds_train,shuffle=True,generator=generator,**common)
    val_loader = DataLoader(ds_val,shuffle=False,**common)
    if spec['schema']=='multimodal_patient_e58_v1':
        from src.training.multimodal_patient_objective import build_complete_patient_loaders
        train_loader,val_loader=build_complete_patient_loaders(ds_train,ds_val,splits.train,splits.validation,
            patients_per_batch=2,seed=cfg.training.seed,num_workers=cfg.data.num_workers,
            pin_memory=cfg.data.pin_memory and device.type=='cuda')
    model = (ClinicalOnlyPCRNet() if clinical_only else MultimodalPCRNet(
        cfg.model,spec.get('fusion_hidden',0),spec.get('image_pooling','average'),
        spec.get('feature_normalization','none'))).to(device)
    model_class = ('ClinicalOnlyPCRNet_v1' if clinical_only else
                   'MultimodalPCRNet_compact_v1' if spec['schema'] in {'multimodal_compact_e58_v1','multimodal_basic_regularized_v1'} else
                   'MultimodalPCRNet_normalized_v1' if spec.get('feature_normalization')=='l2_fixed' else
                   'MultimodalPCRNet_enhancement_v1' if spec.get('image_pooling')=='enhancement_weighted' else
                   'MultimodalPCRNet_nonlinear_v1' if spec.get('fusion_hidden',0) else 'MultimodalPCRNet_v1')
    criterion,pos_weight=build_multimodal_criterion(cfg,splits.train,device,spec.get('positive_weight_exponent',1.))
    print(f'{cfg.experiment_id} · dispositivo={device} · parámetros={trainable_parameter_count(model)} · '
          f'pacientes train/val={len(train_ids)}/{len(val_ids)} · clínica={list(prep.active_columns)}',flush=True)
    if spec['schema']=='multimodal_patient_e58_v1':
        print(f'Lotes:2pacientes completas; cortes variables. BCE unidad={cfg.training.objective}',flush=True)
    optimizer = build_optimizer(model,cfg,spec)
    scheduler = LearningRateSchedule(optimizer,cfg.training)
    out.mkdir(parents=True)
    weights.mkdir(parents=True)
    environment = _environment(device)
    names = ['clinical_features.py','multimodal_model.py','train_multimodal.py','architectures.py','data.py','patient_visual_report.py','moderate_positive_weight.py']
    if clinical_only: names.append('clinical_only.py')
    if spec['schema']=='multimodal_basic_regularized_v1': names.append('basic_regularized.py')
    if spec['schema']=='multimodal_patient_e58_v1':names.extend(['patient_training.py','multimodal_patient_objective.py'])
    if spec.get('image_occlusion') is not None: names.append('synchronized_occlusion.py')
    if spec.get('image_intensity_field') is not None:names.append('shared_intensity_field.py')
    if spec.get('image_pooling')=='enhancement_weighted': names.append('enhancement_weighted_pool.py')
    if spec.get('feature_normalization')=='l2_fixed':names.append('image_feature_normalization.py')
    if any((cfg.data.augmentation.enabled,cfg.data.intensity_augmentation.enabled,cfg.data.horizontal_flip.enabled)):
        names.extend(['augmentation.py','horizontal_flip.py'])
    environment['source_sha256'] = {n:hashlib.sha256(source_path(n).read_bytes()).hexdigest() for n in names}
    environment['input_config_sha256'] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    (out/'environment.json').write_text(json.dumps(environment,indent=2))
    (out/'config_resolved.json').write_text(json.dumps(spec,indent=2))
    (out/'clinical_preprocessor.json').write_text(json.dumps(prep.to_dict(),indent=2))
    audit = dict(fit_scope='internal_train_only',patients=len(train_clinical),
        columns=list(CLINICAL_COLUMNS),missing_counts=train_clinical[list(CLINICAL_COLUMNS)].isna().sum().to_dict(),
        age_min=float(train_clinical.age.min()),age_max=float(train_clinical.age.max()),
        ignored='pCR/pid como feature/cohorte/split/test/metadatos adquisición/subtipos redundantes/raza',
        note='Missing indicators pueden codificar diferencias de cohorte; interpretar con cautela')
    (out/'clinical_train_audit.json').write_text(json.dumps(audit,indent=2))
    history,best,best_epoch,stale=[], -float('inf'),0,0
    if device.type=='cuda': torch.cuda.reset_peak_memory_stats(device)
    start=time.perf_counter()
    for epoch in range(1,cfg.training.epochs+1):
        epoch_start=time.perf_counter()
        lr=scheduler.start_epoch(epoch)
        used_rates=[g['lr'] for g in optimizer.param_groups]
        loss=train_epoch(model,train_loader,optimizer,device,criterion,objective=cfg.training.objective)
        val_loss,metrics,_=evaluate(model,val_loader,device,cfg.training.threshold,criterion,objective=cfg.training.objective)
        score=metrics['patient']['roc_auc']
        scheduler.finish_epoch(score)
        elapsed=time.perf_counter()-epoch_start
        row=_history_row(epoch,loss,val_loss,lr,elapsed,metrics)
        row.update(image_learning_rate=None if clinical_only else used_rates[0],clinical_learning_rate=used_rates[-1])
        row.update(loss_name=cfg.training.loss,pos_weight=pos_weight,
                   positive_weight_exponent=spec.get('positive_weight_exponent',1.))
        history.append(row)
        _write_history(out/'history.csv',history)
        ck=dict(model_class=model_class,epoch=epoch,config=spec,
            clinical_preprocessor=prep.to_dict(),model_state_dict=model.state_dict(),
            optimizer_state_dict=optimizer.state_dict(),scheduler_state_dict=scheduler.state_dict(),
            validation_metrics=metrics,loss_name=cfg.training.loss,pos_weight=pos_weight,
            positive_weight_exponent=spec.get('positive_weight_exponent',1.))
        torch.save(ck,weights/'last.pt')
        if score>best:
            best,best_epoch,stale=score,epoch,0
            torch.save(ck,weights/'best.pt')
        else: stale+=1
        print(f'epoch={epoch:03d} train_loss={loss:.4f} val_loss={val_loss:.4f} '
              f'patient_roc_auc={score:.4f} patient_pr_auc={metrics["patient"]["pr_auc"]:.4f} '
              f'lr={lr:g} clinical_lr={used_rates[-1]:g} time={elapsed:.1f}s',flush=True)
        if stale>=cfg.training.early_stopping_patience:
            print('Early stopping: validación por paciente sin mejora.',flush=True)
            break
    total=time.perf_counter()-start
    ck=torch.load(weights/'best.pt',map_location=device,weights_only=False)
    model.load_state_dict(ck['model_state_dict'])
    val_loss,metrics,frame=evaluate(model,val_loader,device,cfg.training.threshold,criterion,objective=cfg.training.objective)
    save_evaluation_artifacts(frame,out,EvaluationConfig(threshold=cfg.training.threshold,seed=cfg.training.seed))
    diagnostics=build_training_diagnostics(history,metrics,best_epoch)
    if pos_weight is not None:
        diagnostics['loss_comparison_note']='BCE ponderada: escala no comparable con BCE normal; peso fijado solo en cortes de train interno.'
    save_diagnostics_json(diagnostics,out/'training_diagnostics.json')
    save_training_dashboard(history,metrics,best_epoch,out/'training_curves.png')
    debug=cfg.data.train_patients_per_class is not None or cfg.data.validation_patients_per_class is not None
    summary=dict(status='complete',experiment_id=cfg.experiment_id,scientific_result=not debug,
        debug_patient_subset=debug,test_evaluated=False,model_class=model_class,
        best_epoch=best_epoch,epochs_completed=len(history),best_monitor_value=best,
        monitor=cfg.training.monitor,total_seconds=total,seconds_per_epoch_mean=total/len(history),
        parameter_count=trainable_parameter_count(model),parameter_breakdown=parameter_breakdown(model),
        clinical_columns=list(CLINICAL_COLUMNS),active_clinical_columns=list(prep.active_columns),clinical_feature_count=8,
        clinical_input_columns=spec.get('clinical_input_columns',list(CLINICAL_COLUMNS)),
        image_lr_factor=spec.get('image_lr_factor',1.),loss_name=cfg.training.loss,pos_weight=pos_weight,
        positive_weight_exponent=spec.get('positive_weight_exponent',1.),
        image_weight_decay_factor=spec.get('image_weight_decay_factor',1.),
        image_occlusion=spec.get('image_occlusion'),
        image_intensity_field=spec.get('image_intensity_field'),
        image_pooling=spec.get('image_pooling','average'),
        feature_normalization=spec.get('feature_normalization','none'),
        image_augmentation=dict(affine=asdict(cfg.data.augmentation),
            intensity=asdict(cfg.data.intensity_augmentation),horizontal_flip=asdict(cfg.data.horizontal_flip),
            scope='internal_train_only',synchronized_phases=True),
        clinical_fit_patient_count=prep.train_patient_count,clinical_required_at_inference=True,
        clinical_availability_in_professor_demo='not_confirmed',
        web_checkpoint_loader_compatible=False,loss_unit='cut' if cfg.training.objective=='cut' else 'patient',
        training_objective=cfg.training.objective,train_patients_per_batch=cfg.data.train_patients_per_batch,
        patients=splits.patient_counts(),
        samples=splits.sample_counts(),environment=environment,validation_loss=val_loss,
        validation_metrics=metrics,training_diagnostics=diagnostics,
        gpu_peak_memory_bytes=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None,
        image_information_used=not clinical_only,
        note=('Control clínica sola propio; sin PNG ni CNN; no sustituye modelo obligatorio. ' if clinical_only else
              'CNN propia desde cero + clínica basal supuesta; ensayo exploratorio, no modelo final. ')
             +
             'No atribuir toda mejora a imágenes sin referencia clínica sola.')
    (out/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False))
    save_patient_visual_report(out)
    if not debug:
        # Vista de lectura, sin tocar CSV/pesos ni ejecutar nuevos entrenamientos.
        from src.reporting.clear_results import build
        build()
        if out.is_relative_to('experiments'):
            from src.reporting.balanced_results import build as build_balanced
            build_balanced()
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True)
    p.add_argument('--device',required=True,choices=['cpu','mps','cuda'])
    args=p.parse_args()
    summary=run(args.config,args.device)
    print(json.dumps(dict(experiment_id=summary['experiment_id'],
        best_epoch=summary['best_epoch'],epochs_completed=summary['epochs_completed'],
        patient_roc_auc=summary['validation_metrics']['patient']['roc_auc'],
        patient_pr_auc=summary['validation_metrics']['patient']['pr_auc'],
        total_seconds=summary['total_seconds'],
        full_results=load_spec(args.config)[1].output_dir),indent=2,ensure_ascii=False))
