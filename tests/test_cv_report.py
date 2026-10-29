"""Informe de validación cruzada con 5 folds × 2 pérdidas sintéticos (sin imágenes ni modelos)."""
import json

import numpy as np
import pandas as pd
import pytest

from src.evaluation.cv_report import build, confusion


def fake_runs(root, mode='runs_cuda', losses=('normal', 'weighted'), folds=range(5), seed=0):
    rng = np.random.default_rng(seed)
    base = root/'reports/new_approach/P02_aumento_geometrico'/mode
    patients = [f'P{i:04d}' for i in range(1000)]
    labels = (rng.random(1000) < .3).astype(int)
    signal = labels*0.4 + rng.normal(0, 1, 1000)  # mismas pacientes, señal compartida
    cohorts = {p: ('duke', 'spy1', 'spy2')[i % 3] for i, p in enumerate(patients)}
    for loss in losses:
        for fold in folds:
            out = base/f'{loss}_fold_{fold}_seed_42'
            out.mkdir(parents=True)
            idx = np.arange(fold, 1000, 5)
            n_ep = 20 + fold
            ep = np.arange(1, n_ep + 1)
            h = pd.DataFrame(dict(epoch=ep, fit_bce=.62 - .004*ep, inner_bce=.61 - .001*ep,
                                  fit_auc=.5 + .01*ep, inner_auc=.5 + .005*ep + rng.normal(0, .01, n_ep)))
            h.to_csv(out/'history.csv', index=False)
            best = int(h.inner_auc.idxmax()) + 1
            shift = 0 if loss == 'normal' else .15
            prob = 1/(1 + np.exp(-(signal[idx] - 1 + shift + rng.normal(0, .3, len(idx)))))
            pat = pd.DataFrame(dict(patient_id=np.array(patients)[idx], probability=prob,
                                    target=labels[idx], cuts=10))
            pat.to_csv(out/'predictions_outer_patient.csv', index=False)
            cuts = pat.loc[pat.index.repeat(2)].assign(sample_id=lambda d: d.patient_id + '_z', slice_index=1)
            cuts['probability'] = (cuts.probability + rng.normal(0, .02, len(cuts))).clip(0, 1)
            cuts[['sample_id', 'patient_id', 'slice_index', 'target', 'probability']].to_csv(
                out/'predictions_outer_cut.csv', index=False)
            m = lambda a: dict(metrics=dict(roc_auc=a, pr_auc=.4, brier_score=.2), cut_metrics_secondary=dict(roc_auc=a - .01))
            (out/'decision_before_outer.json').write_text(json.dumps(dict(threshold=.3, best_epoch=best)))
            (out/'summary.json').write_text(json.dumps(dict(
                status='complete', scientific_result=True, test_evaluated=False, loss=loss, fold=fold,
                best_epoch=best, epochs_completed=n_ep, seconds=600., fit_at_best=m(.66),
                inner_at_best=m(.62), outer_at_05=m(.55 + .015*fold - (.02 if loss == 'weighted' else 0)))))
    return cohorts


def test_full_report_builds_every_figure_and_paired_comparison(tmp_path):
    cohorts = fake_runs(tmp_path)
    out, summary = build('p02', 'runs_cuda', root=tmp_path, cohorts=cohorts)
    for name in ('curvas_normal.png', 'curvas_weighted.png', 'matrices.png', 'roc_pr.png',
                 'cohortes_folds.png', 'calibracion.png', 'RESUMEN.md', 'resumen.json'):
        assert (out/name).is_file()
    n = summary['normal']
    assert n['n_folds'] == 5 and n['n_pacientes'] == 1000
    assert n['auc_conjunta_ic95'][0] < n['auc_conjunta'] < n['auc_conjunta_ic95'][1]
    assert set(n['matrices']) == {'umbral interno (por fold)', '0,325', '0,5'}
    c = n['matrices']['0,325']
    assert c['tn'] + c['fp'] + c['fn'] + c['tp'] == 1000
    assert summary['pareada']['n'] == 1000
    assert set(n['cohortes']) == {'duke', 'spy1', 'spy2'}


def test_partial_runs_and_ignores_incomplete(tmp_path):
    cohorts = fake_runs(tmp_path, losses=('normal',), folds=[0, 1])
    bad = tmp_path/'reports/new_approach/P02_aumento_geometrico/runs_cuda/weighted_fold_0_seed_42'
    bad.mkdir()
    (bad/'summary.json').write_text(json.dumps(dict(status='failed')))
    _, summary = build('p02', 'runs_cuda', root=tmp_path, cohorts=cohorts)
    assert set(summary) == {'normal'} and summary['normal']['n_folds'] == 2


def test_refuses_runs_that_touched_test(tmp_path):
    cohorts = fake_runs(tmp_path, losses=('normal',), folds=[0])
    path = tmp_path/'reports/new_approach/P02_aumento_geometrico/runs_cuda/normal_fold_0_seed_42/summary.json'
    s = json.loads(path.read_text()); s['test_evaluated'] = True; path.write_text(json.dumps(s))
    with pytest.raises(ValueError):
        build('p02', 'runs_cuda', root=tmp_path, cohorts=cohorts)


def test_confusion_counts():
    c = confusion([0, 0, 1, 1], [.1, .6, .4, .9], .5)
    assert (c['tn'], c['fp'], c['fn'], c['tp']) == (1, 1, 1, 1)
    assert c['balanced_accuracy'] == .5
