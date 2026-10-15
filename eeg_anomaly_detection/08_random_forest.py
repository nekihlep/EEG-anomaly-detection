"""Compare RF on all AB features and L1-selected top20 on the same split.
Scaling and feature selection are fitted inside each CV training fold.
The existing holdout has already been inspected: results are exploratory.
"""
from pathlib import Path
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from evaluation import classification_metrics

ROOT = Path(__file__).resolve().parent

def main():
    df = pd.read_csv(ROOT/'EEG_data.csv')
    df = df[df.age.between(18, 30)]
    names = [c for c in df if c.startswith('AB.')]
    x = df[names]
    y = (df['specific.disorder']=='Healthy control').astype(int)
    train, test = train_test_split(np.arange(len(y)),test_size=.2,stratify=y,random_state=42)
    # Same participants as the original preprocessing script.
    assert np.array_equal(y.iloc[test],np.load(ROOT/'processed_data/y_test.npy'))
    out = ROOT/'results'/'rf_comparison'
    out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for label,selection in [('all114','passthrough'),('l1_top20',SelectFromModel(
        LogisticRegressionCV(Cs=10,penalty='l1',solver='liblinear',cv=5,
            scoring='recall_macro',max_iter=5000,random_state=42),
        threshold=-np.inf,max_features=20))]:
        pipeline=Pipeline([('scale',StandardScaler()),('select',selection),
            ('rf',RandomForestClassifier(random_state=42,n_jobs=1,class_weight='balanced'))])
        search=GridSearchCV(pipeline,{'rf__n_estimators':[100,200],
            'rf__max_depth':[6,10,None],'rf__min_samples_split':[2,5]},
            cv=StratifiedKFold(5,shuffle=True,random_state=42),
            scoring='balanced_accuracy',n_jobs=2,error_score='raise')
        search.fit(x.iloc[train],y.iloc[train])
        best=search.best_estimator_
        pred=best.predict(x.iloc[test])
        score=best.predict_proba(x.iloc[test])[:,list(best.classes_).index(0)]
        rows.append(dict(model=label,cv_balanced_accuracy=search.best_score_,
            **classification_metrics(y.iloc[test],pred,score)))
        chosen=np.arange(len(names)) if label=='all114' else np.flatnonzero(best['select'].get_support())
        table=pd.DataFrame({'feature_index':chosen,'feature':np.array(names)[chosen],
            'rf_importance':best['rf'].feature_importances_})
        table['rhythm']=table.feature.str.split('.').str[2]
        table['electrode']=table.feature.str.split('.').str[-1]
        if label=='l1_top20':
            table['l1_abs_coef']=np.abs(best['select'].estimator_.coef_[0])[chosen]
        table.to_csv(out/f'{label}_features.csv',index=False)
        pd.DataFrame(dict(dataset_row=df.index[test],y_true=y.iloc[test].to_numpy(),
            y_pred=pred,patient_score=score)).to_csv(out/f'{label}_predictions.csv',index=False)
        pd.DataFrame(search.cv_results_).to_csv(out/f'{label}_cv.csv',index=False)
        with open(out/f'{label}_pipeline.pkl','wb') as f:
            pickle.dump(best,f)
        (out/f'{label}_params.json').write_text(json.dumps(search.best_params_,indent=2))
        print(label,rows[-1],flush=True)
    pd.DataFrame(rows).to_csv(out/'metrics.csv',index=False)

if __name__=='__main__':
    main()
