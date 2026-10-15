"""Choose a patient-score threshold using calibration labels only."""
import numpy as np
from evaluation import classification_metrics


def select_threshold(y, scores, target_sensitivity=0.90):
    y, scores = np.asarray(y), np.asarray(scores, dtype=float)
    if y.shape != scores.shape or not np.isfinite(scores).all():
        raise ValueError('Expected matching labels and finite scores')
    if set(np.unique(y)) != {0, 1} or not 0 < target_sensitivity <= 1:
        raise ValueError('Both classes and a valid sensitivity target are required')
    # Highest specificity, then highest threshold, among feasible operating points.
    thresholds = np.unique(scores)[::-1]
    patient_predictions = scores[:, None] >= thresholds
    sensitivity = patient_predictions[y == 0].mean(axis=0)
    specificity = 1 - patient_predictions[y == 1].mean(axis=0)
    feasible = np.flatnonzero(sensitivity + 1e-12 >= target_sensitivity)
    best_specificity = specificity[feasible].max()
    index = feasible[np.flatnonzero(specificity[feasible] == best_specificity)[0]]
    threshold = float(thresholds[index])
    pred = np.where(scores >= threshold, 0, 1)
    return threshold, classification_metrics(y, pred, scores)
