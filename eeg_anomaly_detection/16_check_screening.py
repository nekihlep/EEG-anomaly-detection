"""Verify calibration thresholds, disjoint groups and saved test metrics."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from evaluation import classification_metrics
from screening import select_threshold
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results'/'screening_comparison'


def main():
    raw=pd.read_csv(ROOT/'EEG_data.csv');raw=raw[raw.age.between(18,30)]
    names=[c for c in raw if c.startswith('AB.')]
    groups=pd.util.hash_pandas_object(raw[names],index=False)
    splits=json.loads((OUT/'splits.json').read_text())
    old=json.loads((ROOT/'results'/'nested_comparison'/'splits.json').read_text())
    expected_models={'Dummy','L1','RF','L1_RF','AE_RF'}
    pred=pd.read_csv(OUT/'predictions.csv')
    metrics=pd.read_csv(OUT/'fold_metrics.csv')
    assert len(splits)==10 and len(metrics)==100
    for fold,split in enumerate(splits):
        train,cal,test=[split[k] for k in ['train_rows','calibration_rows','test_rows']]
        assert set(train)|set(cal)|set(test)==set(raw.index)
        for a,b in [(train,cal),(train,test),(cal,test)]:
            assert not set(a)&set(b) and not set(groups.loc[a])&set(groups.loc[b])
        assert set(test)==set(old[fold]['test_rows'])
        calibration=pd.read_csv(OUT/f'fold_{fold:02d}_calibration.csv')
        params=json.loads((OUT/f'fold_{fold:02d}_params.json').read_text())
        assert set(calibration.model)==expected_models
        for model in expected_models:
            c=calibration[calibration.model==model]
            assert len(c)==len(cal) and set(c.dataset_row)==set(cal)
            truth=(raw.loc[c.dataset_row,'specific.disorder']=='Healthy control').astype(int)
            assert np.array_equal(truth,c.y_true)
            threshold,cal_metrics=select_threshold(c.y_true,c.patient_score,.9)
            assert np.isclose(threshold,params[model]['threshold'])
            assert cal_metrics['sensitivity']>=.9-1e-12
            for key,value in cal_metrics.items():
                assert np.isclose(value,params[model]['calibration_metrics'][key])
            standard=pred[(pred.fold==fold)&(pred.model==model)&(pred['mode']=='standard')].sort_values('dataset_row')
            screened=pred[(pred.fold==fold)&(pred.model==model)&(pred['mode']=='screening_90')].sort_values('dataset_row')
            assert np.array_equal(standard.patient_score,screened.patient_score)
            assert np.array_equal(screened.y_pred,np.where(screened.patient_score>=threshold,0,1))
    for (fold,model,mode),g in pred.groupby(['fold','model','mode']):
        assert len(g)==len(splits[fold]['test_rows'])
        assert set(g.dataset_row)==set(splits[fold]['test_rows'])
        truth=(raw.loc[g.dataset_row,'specific.disorder']=='Healthy control').astype(int)
        assert np.array_equal(truth,g.y_true)
        actual=classification_metrics(g.y_true,g.y_pred,g.patient_score)
        saved=metrics[(metrics.fold==fold)&(metrics.model==model)&(metrics['mode']==mode)].iloc[0]
        for key,value in actual.items():assert np.isclose(value,saved[key])
    for (_,_,_),g in pred.groupby(['repeat','model','mode']):
        assert len(g)==599 and g.dataset_row.nunique()==599
    print('Verified 10 folds, 5 models, 2 operating rules, calibration-only thresholds and all metrics.')


if __name__=='__main__':main()
