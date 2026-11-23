"""EER and minDCF exactly as in the ASVspoof reference scripts (eval_metrics_DF.py)."""
from __future__ import annotations

import numpy as np


def compute_det_curve(target_scores, nontarget_scores):
    target_scores = np.asarray(target_scores, dtype=np.float64)
    nontarget_scores = np.asarray(nontarget_scores, dtype=np.float64)
    n_scores = target_scores.size + nontarget_scores.size
    all_scores = np.concatenate((target_scores, nontarget_scores))
    labels = np.concatenate((np.ones(target_scores.size), np.zeros(nontarget_scores.size)))
    indices = np.argsort(all_scores, kind='mergesort')
    labels = labels[indices]
    tar_trial_sums = np.cumsum(labels)
    nontarget_trial_sums = nontarget_scores.size - (np.arange(1, n_scores + 1) - tar_trial_sums)
    frr = np.concatenate((np.atleast_1d(0), tar_trial_sums / target_scores.size))
    far = np.concatenate((np.atleast_1d(1), nontarget_trial_sums / nontarget_scores.size))
    thresholds = np.concatenate((np.atleast_1d(all_scores[indices[0]] - 0.001), all_scores[indices]))
    return frr, far, thresholds


def compute_eer(bonafide_scores, spoof_scores):
    """Bona fide = target (higher score = more bona fide). Returns (eer in [0,1], threshold)."""
    frr, far, thr = compute_det_curve(bonafide_scores, spoof_scores)
    i = np.argmin(np.abs(frr - far))
    return float(np.mean((frr[i], far[i]))), float(thr[i])


def compute_min_dcf(bonafide_scores, spoof_scores, p_target=0.05, c_miss=1, c_fa=10):
    """Normalised minDCF with ASVspoof5 default costs."""
    frr, far, _ = compute_det_curve(bonafide_scores, spoof_scores)
    dcf = c_miss * p_target * frr + c_fa * (1 - p_target) * far
    return float(dcf.min() / min(c_miss * p_target, c_fa * (1 - p_target)))


def eer_from_arrays(scores, labels):
    """labels: 1 = bona fide, 0 = spoof."""
    scores, labels = np.asarray(scores), np.asarray(labels)
    return compute_eer(scores[labels == 1], scores[labels == 0])[0]
