"""Patient=0, healthy=1. Larger scores indicate patients."""
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score

def isolation_labels(raw):
    raw = np.asarray(raw)
    if not np.isin(raw, [-1, 1]).all():
        raise ValueError('Expected -1/+1 predictions')
    return (raw == 1).astype(int)

def classification_metrics(y, pred, score):
    if not np.isin(y, [0, 1]).all() or not np.isin(pred, [0, 1]).all():
        raise ValueError('Expected patient=0, healthy=1')
    pp, ph, hp, hh = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    sensitivity, specificity = pp/(pp+ph), hh/(hp+hh)
    return dict(sensitivity=sensitivity, specificity=specificity,
        balanced_accuracy=(sensitivity+specificity)/2,
        f1_patient=f1_score(y,pred,pos_label=0,zero_division=0),
        f1_macro=f1_score(y,pred,average='macro',zero_division=0),
        f1_weighted=f1_score(y,pred,average='weighted',zero_division=0),
        roc_auc=roc_auc_score(np.asarray(y)==0,score),
        patient_as_patient=int(pp),patient_as_healthy=int(ph),
        healthy_as_patient=int(hp),healthy_as_healthy=int(hh))
