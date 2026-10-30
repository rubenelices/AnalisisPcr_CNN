"""Una evaluación de test: E57 original, época14, umbral0,325.

Los parámetros están fijados en código antes de leer test. `report` solo vuelve
a presentar los resultados guardados y no recalcula inferencia.
"""
from src import source_path
import argparse
import base64
import hashlib
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import nbformat
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.pipeline.data import BreastDCEDataset, load_samples
from src.evaluation.evaluation import EvaluationConfig, build_evaluation_summary, save_evaluation_figure
from src.evaluation.metrics import aggregate_by_patient
from src.models.patient_feature_model import load_feature_checkpoint

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'reports/final_e57_test_v1'
CHECKPOINT = ROOT / 'checkpoints/E57_patient_feature_mean_mac/fold_0_seed_42/best.pt'
EXPECTED_SHA = 'a1fb12dfae2c3d6b730ae0b1d916727981154116077e40aa64799415b8cfde5f'
THRESHOLD = .325


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_checkpoint():
    if sha(CHECKPOINT) != EXPECTED_SHA:
        raise ValueError('El checkpoint no coincide con la versión fijada antes de test')
    ck = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)
    if ck.get('model_class') != 'PatientFeatureNet_v1' or ck.get('epoch') != 14:
        raise ValueError('Clase o época de checkpoint incorrecta')


@torch.inference_mode()
def collect(model, loader, device):
    records = []
    model.eval()
    for batch in loader:
        if set(batch['split']) != {'test'} or not batch['fold'].eq(-1).all():
            raise ValueError('Solo se admite el test público, sin folds de desarrollo')
        probabilities = torch.sigmoid(model(batch['image'].to(device))).cpu().tolist()
        for index, probability in enumerate(probabilities):
            records.append(dict(sample_id=batch['sample_id'][index], patient_id=batch['patient_id'][index],
                                slice_index=int(batch['slice_index'][index]), target=int(batch['target'][index]),
                                probability=float(probability)))
        print(f'Test: {len(records)}/{len(loader.dataset)} cortes procesados', flush=True)
    cuts = pd.DataFrame(records)
    if cuts.empty or cuts.sample_id.duplicated().any():
        raise ValueError('Predicciones vacías o sample_id repetidas')
    return cuts


def evaluate(device_name='cpu'):
    verify_checkpoint()
    device = torch.device(device_name)
    if device.type == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('MPS no disponible; especificar CPU expresamente')
    if device.type not in {'cpu', 'mps'}:
        raise ValueError('Solo evaluación local')
    # mkdir exclusivo protege la evaluación original frente a repeticiones.
    OUT.mkdir(parents=True, exist_ok=False)
    source_names = ('evaluate_e57_final.py', 'patient_feature_model.py', 'architectures.py',
                    'data.py', 'phase_representation.py', 'evaluation.py', 'metrics.py')
    protocol = dict(schema='e57_public_test_frozen_v1',
                    frozen_at_utc=datetime.now(timezone.utc).isoformat(),
                    checkpoint=str(CHECKPOINT.relative_to(ROOT)), checkpoint_sha256=EXPECTED_SHA,
                    epoch=14, threshold=THRESHOLD, aggregation='mean_cut_probabilities',
                    clinical_inputs=[], augmentation=False, calibration_fitted=False,
                    seed=42, bootstrap_repetitions=2000, batch_size=16,
                    source_sha256={name:sha(source_path(name)) for name in source_names},
                    note='Congelado el 20 de octubre antes de abrir test: una única evaluación, sin selección posterior con test.')
    (OUT/'protocolo_previo.json').write_text(json.dumps(protocol, indent=2, ensure_ascii=False))
    try:
        torch.set_num_threads(4)
        started = time.perf_counter()
        samples = load_samples(ROOT/'breastdcedl')
        train = samples.loc[samples.split.eq('train')]
        rows = samples.loc[samples.split.eq('test')].copy()
        if (rows.patient_id.nunique() != 176 or len(rows) != 1758
                or not rows.fold.eq(-1).all() or set(rows.patient_id)&set(train.patient_id)):
            raise ValueError('El test no corresponde a la partición pública esperada')
        if not rows.groupby('patient_id').pCR.nunique().eq(1).all():
            raise ValueError('Etiquetas inconsistentes por paciente')
        for column in ('path_pre','path_early','path_late'):
            if not all((ROOT/'breastdcedl'/p).is_file() for p in rows[column]):
                raise FileNotFoundError('Faltan imágenes del test público')
        loader = DataLoader(BreastDCEDataset(rows, ROOT/'breastdcedl'), batch_size=16,
                            shuffle=False, num_workers=0,
                            generator=torch.Generator().manual_seed(42))
        model = load_feature_checkpoint(CHECKPOINT, device)
        state = {key:value.detach().cpu().clone() for key,value in model.state_dict().items()}
        if sum(p.numel() for p in model.parameters()) != 298161:
            raise ValueError('Arquitectura incorrecta')
        cuts = collect(model, loader, device)
        patients = aggregate_by_patient(cuts.patient_id, cuts.target, cuts.probability)
        if len(patients) != 176 or len(cuts) != 1758:
            raise ValueError('Evaluación incompleta')
        patients['prediction'] = (patients.probability >= THRESHOLD).astype(int)
        cuts.to_csv(OUT/'predictions_cut.csv', index=False)
        patients.to_csv(OUT/'predictions_patient.csv', index=False)
        if sha(CHECKPOINT) != EXPECTED_SHA or any(not torch.equal(value, model.state_dict()[key].cpu())
                                               for key,value in state.items()):
            raise ValueError('Pesos o buffers modificados durante inferencia')
        cfg = EvaluationConfig(threshold=THRESHOLD, seed=42, bootstrap_repetitions=2000)
        result = build_evaluation_summary(cuts, patients, cfg)
        result.update(scope='public_test_final_single_evaluation', status='complete',
                      test_evaluated=True, model='E57_original_epoch14',
                      checkpoint_sha256=EXPECTED_SHA, selection_using_test=False,
                      training_performed=False, elapsed_seconds=time.perf_counter()-started,
                      completed_at_utc=datetime.now(timezone.utc).isoformat(),
                      environment=dict(device=str(device), torch=torch.__version__,
                                       python=platform.python_version(), threads=4))
        cm = result['patient_metrics_primary']['confusion_matrix']
        negative_f1 = 2 * cm['tn'] / (2 * cm['tn'] + cm['fp'] + cm['fn']) if 2 * cm['tn'] + cm['fp'] + cm['fn'] else 0.
        result['patient_metrics_primary']['f1_macro'] = (negative_f1 + result['patient_metrics_primary']['f1']) / 2
        (OUT/'test_evaluation.json').write_text(json.dumps(result, indent=2, ensure_ascii=False))
        report()
        return result
    except BaseException as error:
        (OUT/'interrupcion.json').write_text(json.dumps(dict(error=repr(error),
                  note='No repetir automáticamente; conservar salida original'), indent=2))
        raise


def report():
    """Representación de resultados guardados; no carga pesos ni imágenes de test."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from src.reporting.e57_stability_results import image_cell
    result = json.loads((OUT/'test_evaluation.json').read_text())
    if result.get('status') != 'complete' or result['checkpoint_sha256'] != EXPECTED_SHA:
        raise ValueError('No hay evaluación final completa')
    patients = pd.read_csv(OUT/'predictions_patient.csv')
    cfg = EvaluationConfig(threshold=THRESHOLD)
    images = ROOT/'experiments/E57_patient_feature_mean_mac/results/figures'
    images.mkdir(parents=True, exist_ok=True)
    picture = images/'TEST_RESULTADOS.png'
    save_evaluation_figure(patients, result, cfg, picture,
                          title='E57 · TEST público · época14 · umbral0,325 fijado antes de test')
    audit = json.loads((ROOT/'local/campana_20261015/analysis/original_reference_eval.json').read_text())
    m = result['patient_metrics_primary']; intervals = result['bootstrap']['intervals']
    labels = {'roc_auc':'ROC-AUC','pr_auc':'AP','sensitivity':'Sensibilidad','specificity':'Especificidad',
              'precision':'Precisión pCR','f1':'F1 pCR','balanced_accuracy':'Balanced accuracy','brier_score':'Brier'}
    table = pd.DataFrame([dict(Métrica=label, Valor=round(m[k],4),
                  IC95=f'{intervals[k]["lower"]:.4f}–{intervals[k]["upper"]:.4f}') for k,label in labels.items()]).to_markdown(index=False)
    c = m['confusion_matrix']
    text = ('# E57 · única evaluación final de TEST\n\n'
            'Modelo original de época14, entrenado con878 pacientes y seleccionado con219 de validación. '
            'Umbral0,325 y media de probabilidades por paciente fijados antes de abrir test. '
            '176pacientes/1.758cortes de test público. Sin entrenamiento ni ajustes con test.\n\n'
            + table + f'\n\nMatriz: **TN{c["tn"]}, FP{c["fp"]}, FN{c["fn"]}, TP{c["tp"]}**.\n\n'
            + 'Los intervalos son bootstrap por paciente, con2.000 remuestreos y semilla42. '
              'La validación seleccionó época y umbral; el test mide este checkpoint fijo, '
              'no la media de máximos de cinco modelos. Cualquier cambio posterior basado '
              'en test convertiría este conjunto en datos de desarrollo.\n\nUso educativo, sin validez clínica.\n')
    (ROOT/'experiments/E57_patient_feature_mean_mac/results/TEST.md').write_text(text)
    fig, ax = plt.subplots(figsize=(8,4.5))
    values = [audit['train_eval']['metrics']['roc_auc'],audit['validation_eval']['metrics']['roc_auc'],m['roc_auc']]
    ax.bar(['Train eval','Validación','TEST'],values,color=['#2f75b5','#ed7134','#424e57'])
    for i,value in enumerate(values):ax.text(i,value+.02,f'{value:.3f}',ha='center',fontweight='bold')
    ax.axhline(.5,color='0.6',ls=':');ax.set(ylim=(0,1),ylabel='ROC-AUC paciente',title='E57 · mismo checkpoint de época14')
    ax.grid(axis='y',alpha=.2);fig.tight_layout()
    bars=images/'TEST_COMPARACION.png';fig.savefig(bars,dpi=150,bbox_inches='tight');plt.close(fig)
    cells=[nbformat.v4.new_markdown_cell(text), image_cell(picture,'E57 · resultados finales TEST'),
           image_cell(bars,'E57 · train/validación/test del mismo checkpoint')]
    nb=nbformat.v4.new_notebook(cells=cells)
    nb.metadata.kernelspec={'display_name':'Python 3','language':'python','name':'python3'}
    nbformat.write(nb,ROOT/'experiments/E57_patient_feature_mean_mac/results/TEST.ipynb')
    from src.reporting.present_e57 import build as build_dossier
    build_dossier()
    for name in ('experiments/RESULTADOS.ipynb',):
        path=ROOT/name
        if not path.exists():continue
        existing=nbformat.read(path,4)
        existing.cells=[cell for cell in existing.cells if cell.metadata.get('campaign_overview')!='E57_TEST_FINAL']
        import copy
        inserted=copy.deepcopy(cells)
        for cell in inserted:cell.metadata['campaign_overview']='E57_TEST_FINAL'
        existing.cells[0:0]=inserted
        nbformat.write(existing,path)
    print('E57 TEST COMPLETO',json.dumps(m,ensure_ascii=False),flush=True)
    return ROOT/'experiments/E57_patient_feature_mean_mac/results/TEST.ipynb'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['evaluate-test','report'])
    parser.add_argument('--device',choices=['cpu','mps'],default='cpu')
    args=parser.parse_args()
    if args.action=='report':print(report())
    else:evaluate(args.device)


if __name__=='__main__':main()
