"""Check that all five models were evaluated on exactly the same rows."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from evaluation import classification_metrics
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results'/'nested_comparison'

def main():
    raw=pd.read_csv(ROOT/'EEG_data.csv');raw=raw[raw.age.between(18,30)]
    names=[c for c in raw if c.startswith('AB.')]
    groups=pd.util.hash_pandas_object(raw[names],index=False)
    splits=json.loads((OUT/'splits.json').read_text())
    assert len(splits)==10
    for split in splits:
        tr,te=split['train_rows'],split['test_rows']
        assert not set(tr)&set(te)
        assert set(tr)|set(te)==set(raw.index)
        assert not set(groups.loc[tr])&set(groups.loc[te])
    pred=pd.read_csv(OUT/'predictions.csv');metrics=pd.read_csv(OUT/'fold_metrics.csv')
    expected={'Dummy','L1','RF','L1_RF','AE_RF'}
    assert set(pred.model)==expected and len(metrics)==50
    for (repeat,model),g in pred.groupby(['repeat','model']):
        assert len(g)==len(raw) and g.dataset_row.nunique()==len(raw)
        assert set(g.dataset_row)==set(raw.index)
    for (fold,model),g in pred.groupby(['fold','model']):
        assert set(g.dataset_row)==set(splits[fold]['test_rows'])
        truth=(raw.loc[g.dataset_row,'specific.disorder']=='Healthy control').astype(int).to_numpy()
        assert np.array_equal(truth,g.y_true)
        actual=classification_metrics(g.y_true,g.y_pred,g.patient_score)
        saved=metrics[(metrics.fold==fold)&(metrics.model==model)].iloc[0]
        for key,value in actual.items():assert np.isclose(saved[key],value)
    print('Verified 10 outer folds, 5 models, all predictions and metrics; no identical-profile leakage.')

if __name__=='__main__':main()
