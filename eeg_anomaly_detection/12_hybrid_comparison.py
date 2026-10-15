"""AE(114-64-32-64-114) -> RF, same split and RF grid as step 08.
All representation learning happens inside the CV training fold.
"""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','2')
os.environ.setdefault('TF_NUM_INTRAOP_THREADS','1')
os.environ.setdefault('TF_NUM_INTEROP_THREADS','1')
from pathlib import Path
import json,pickle
import numpy as np
import pandas as pd
from joblib import Memory
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split,StratifiedKFold,GridSearchCV
from ae_transformer import AutoencoderFeatures
from evaluation import classification_metrics
ROOT=Path(__file__).resolve().parent

def main():
    df=pd.read_csv(ROOT/'EEG_data.csv')
    df=df[df.age.between(18,30)]
    names=[c for c in df if c.startswith('AB.')]
    x=df[names];y=(df['specific.disorder']=='Healthy control').astype(int)
    train,test=train_test_split(np.arange(len(y)),test_size=.2,stratify=y,random_state=42)
    out=ROOT/'results'/'hybrid_comparison';out.mkdir(parents=True,exist_ok=True)
    pipeline=Pipeline([('scale',StandardScaler()),('ae',AutoencoderFeatures()),
        ('rf',RandomForestClassifier(random_state=42,n_jobs=1,class_weight='balanced'))],
        memory=Memory(ROOT/'initial_project_artifacts'/'cv_cache',verbose=0))
    search=GridSearchCV(pipeline,{'rf__n_estimators':[100,200],
        'rf__max_depth':[6,10,None],'rf__min_samples_split':[2,5]},
        cv=StratifiedKFold(5,shuffle=True,random_state=42),
        scoring='balanced_accuracy',n_jobs=1,error_score='raise',verbose=1)
    search.fit(x.iloc[train],y.iloc[train])
    model=search.best_estimator_
    pred=model.predict(x.iloc[test]);score=model.predict_proba(x.iloc[test])[:,0]
    row=dict(model='AE32_RF',cv_balanced_accuracy=search.best_score_,
        **classification_metrics(y.iloc[test],pred,score))
    pd.DataFrame([row]).to_csv(out/'metrics.csv',index=False)
    pd.DataFrame(dict(dataset_row=df.index[test],y_true=y.iloc[test].to_numpy(),
        y_pred=pred,patient_score=score)).to_csv(out/'predictions.csv',index=False)
    pd.DataFrame(search.cv_results_).to_csv(out/'cv.csv',index=False)
    pd.DataFrame({'epoch':np.arange(1,51),'training_loss':model['ae'].loss_history_}).to_csv(out/'training_loss.csv',index=False)
    (out/'params.json').write_text(json.dumps(search.best_params_,indent=2))
    with open(out/'pipeline.pkl','wb') as f:pickle.dump(model,f)
    print(row,flush=True)

if __name__=='__main__':main()
