"""Audit trusted saved models. Legacy RF uses first 20 columns, not L1 top20.
AE threshold metrics withheld because the original threshold used test labels.
These are diagnostic results on the existing split, not final paper results.
"""
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from evaluation import classification_metrics, isolation_labels
ROOT = Path(__file__).resolve().parent
DATA = ROOT/'processed_data'
OUT = ROOT/'results'/'audit'

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    x=np.load(DATA/'X_test_scaled.npy'); y=np.load(DATA/'y_test.npy')
    rows=[]
    for name,filename in [('LogReg L1','logreg_l1_cv.pkl'),
        ('RF legacy first20','random_forest_top20.pkl'),
        ('Isolation Forest','iso_forest.pkl'),('Dummy','dummy_model.pkl')]:
        with open(DATA/'models'/filename,'rb') as f:
            model=pickle.load(f)
        inputs=x[:,:20] if name.startswith('RF') else x
        raw=model.predict(inputs)
        if name=='Isolation Forest':
            pred=isolation_labels(raw); score=-model.decision_function(inputs)
        else:
            pred=raw; score=model.predict_proba(inputs)[:,list(model.classes_).index(0)]
        rows.append(dict(model=name,**classification_metrics(y,pred,score)))
        pd.DataFrame(dict(test_row=np.arange(len(y)),y_true=y,y_pred=pred,
            patient_score=score)).to_csv(OUT/filename.replace('.pkl','_predictions.csv'),index=False)
    with open(DATA/'models'/'autoencoder_model.pkl','rb') as f:
        ae=pickle.load(f)
    error=np.mean((x-ae.predict(x))**2,axis=1)
    rows.append(dict(model='AE (uncalibrated)',roc_auc=roc_auc_score(y==0,error)))
    pd.DataFrame(dict(test_row=np.arange(len(y)),y_true=y,patient_score=error)).to_csv(OUT/'ae_scores.csv',index=False)
    result=pd.DataFrame(rows)
    result.to_csv(OUT/'saved_model_metrics.csv',index=False)
    print(result.round(4).to_string(index=False))
    print('AUDIT ONLY: existing split; legacy RF inputs; AE threshold metrics withheld.')

if __name__=='__main__':
    main()
