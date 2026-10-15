"""Fixed exploratory protocol: repeated nested CV, shared outer folds.
Run from repo root: python eeg_anomaly_detection/13_nested_comparison.py
No external validation: dataset and methods were explored before this protocol.
"""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','3')
os.environ.setdefault('TF_NUM_INTRAOP_THREADS','1')
os.environ.setdefault('TF_NUM_INTEROP_THREADS','1')
from pathlib import Path
import json, hashlib, warnings, platform
import numpy as np
import pandas as pd
import sklearn
from joblib import Memory
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV
from sklearn.exceptions import ConvergenceWarning
from evaluation import classification_metrics
from ae_transformer import AutoencoderFeatures

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results'/'nested_comparison'
METRICS=['sensitivity','specificity','balanced_accuracy','f1_patient','f1_macro','f1_weighted','roc_auc']
C_VALUES=[0.1,1,10]
RF_GRID={'rf__n_estimators':[200],'rf__max_depth':[6,None],'rf__min_samples_leaf':[1,3]}

def make_models(seed, cache):
    l1=LogisticRegression(penalty='l1',solver='liblinear',max_iter=5000,random_state=seed)
    rf=RandomForestClassifier(random_state=seed,n_jobs=1,class_weight='balanced')
    return [
        ('Dummy',DummyClassifier(strategy='most_frequent'),{}),
        ('L1',Pipeline([('scale',StandardScaler()),('l1',l1)],memory=cache),{'l1__C':C_VALUES}),
        ('RF',Pipeline([('scale',StandardScaler()),('rf',rf)],memory=cache),RF_GRID),
        ('L1_RF',Pipeline([('scale',StandardScaler()),('select',SelectFromModel(l1,threshold=-np.inf,max_features=20)),('rf',rf)],memory=cache),dict(RF_GRID,**{'select__estimator__C':C_VALUES})),
        ('AE_RF',Pipeline([('scale',StandardScaler()),('ae',AutoencoderFeatures(random_state=seed)),('rf',rf)],memory=cache),RF_GRID),
    ]

def summarize():
    frames=list(OUT.glob('fold_*_metrics.csv'))
    all_metrics=pd.concat([pd.read_csv(p) for p in frames],ignore_index=True)
    all_metrics.to_csv(OUT/'fold_metrics.csv',index=False)
    all_metrics.groupby('model')[METRICS].agg(['mean','std']).to_csv(OUT/'summary.csv')
    pred=pd.concat([pd.read_csv(p) for p in OUT.glob('fold_*_predictions.csv')],ignore_index=True)
    pred.to_csv(OUT/'predictions.csv',index=False)
    # Within-repeat OOF: one prediction per participant, not independent cohorts.
    pooled=[]
    for (repeat,model),g in pred.groupby(['repeat','model']):
        pooled.append(dict(repeat=repeat,model=model,**classification_metrics(g.y_true,g.y_pred,g.patient_score)))
    pd.DataFrame(pooled).to_csv(OUT/'oof_metrics.csv',index=False)
    print(all_metrics.groupby('model')[METRICS].mean().round(4).to_string(),flush=True)

def main():
    import tensorflow as tf  # Fail early if the environment is incomplete.
    OUT.mkdir(parents=True,exist_ok=True)
    df=pd.read_csv(ROOT/'EEG_data.csv'); df=df[df.age.between(18,30)]
    names=[c for c in df if c.startswith('AB.')]
    x=df[names]; y=(df['specific.disorder']=='Healthy control').astype(int)
    assert len(names)==114 and not x.isna().any().any()
    groups=pd.util.hash_pandas_object(x,index=False).to_numpy()
    splits=[]
    for repeat in range(2):
        splits.extend(StratifiedGroupKFold(5,shuffle=True,random_state=2026+repeat).split(x,y,groups))
    manifest=dict(outer_folds=5,repeats=2,inner_folds=3,outer_seed=2026,grouping='identical 114 AB values',
        C_values=C_VALUES,rf_grid=RF_GRID,selection_score='balanced_accuracy',
        ae=dict(hidden=[64,32,64],output='linear',epochs=50,batch_size=32),
        label_mapping={'patient':0,'healthy':1},
        dataset_sha256=hashlib.sha256((ROOT/'EEG_data.csv').read_bytes()).hexdigest(),
        python=platform.python_version(),sklearn=sklearn.__version__,tensorflow=tf.__version__)
    (OUT/'protocol.json').write_text(json.dumps(manifest,indent=2))
    (OUT/'splits.json').write_text(json.dumps([{'fold':i,'train_rows':df.index[tr].tolist(),'test_rows':df.index[te].tolist()} for i,(tr,te) in enumerate(splits)],indent=2))
    for fold,(train,test) in enumerate(splits):
        metrics_path=OUT/f'fold_{fold:02d}_metrics.csv'
        if metrics_path.exists():
            print('Already complete:',fold,flush=True); continue
        assert not set(train)&set(test)
        seed=2026+fold
        assert not set(groups[train]) & set(groups[test])
        inner=list(StratifiedGroupKFold(3,shuffle=True,random_state=seed).split(x.iloc[train],y.iloc[train],groups[train]))
        for a,b in inner:
            assert not set(groups[train[a]]) & set(groups[train[b]])
        cache=Memory(ROOT/'processed_data'/'nested_cache'/str(fold),verbose=0)
        rows=[];predictions=[];details={};features=[]
        for name,estimator,grid in make_models(seed,cache):
            print(f'Fold {fold+1}/10: {name}',flush=True)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always',ConvergenceWarning)
                warnings.simplefilter('ignore',FutureWarning)
                if grid:
                    search=GridSearchCV(estimator,grid,cv=inner,scoring='balanced_accuracy',n_jobs=1,error_score='raise')
                    search.fit(x.iloc[train],y.iloc[train]);model=search.best_estimator_
                    details[name]={'best_params':search.best_params_,'inner_score':search.best_score_}
                else:
                    model=estimator.fit(x.iloc[train],y.iloc[train]);details[name]={}
            conv=[str(w.message) for w in caught if issubclass(w.category,ConvergenceWarning)]
            if conv:
                (OUT/f'fold_{fold:02d}_warnings.json').write_text(json.dumps(conv))
                raise RuntimeError(f'Convergence warning in fold {fold}, {name}; inspect before proceeding')
            pred=model.predict(x.iloc[test]);score=model.predict_proba(x.iloc[test])[:,list(model.classes_).index(0)]
            rows.append(dict(fold=fold,repeat=fold//5,model=name,**classification_metrics(y.iloc[test],pred,score)))
            predictions.append(pd.DataFrame(dict(fold=fold,repeat=fold//5,model=name,dataset_row=df.index[test],y_true=y.iloc[test].to_numpy(),y_pred=pred,patient_score=score)))
            if name=='L1_RF':
                selected=model['select'];indices=np.flatnonzero(selected.get_support())
                features=[dict(feature=names[i],l1_abs_coef=abs(selected.estimator_.coef_[0,i])) for i in indices]
                details[name]['nonzero_selected']=int(sum(f['l1_abs_coef']>0 for f in features))
            if name=='AE_RF':
                details[name]['training_reconstruction_mse']=model['ae'].reconstruction_mse_
            print('  BA:',round(rows[-1]['balanced_accuracy'],3),flush=True)
        pd.concat(predictions).to_csv(OUT/f'fold_{fold:02d}_predictions.csv',index=False)
        (OUT/f'fold_{fold:02d}_params.json').write_text(json.dumps(details,indent=2))
        pd.DataFrame(features).to_csv(OUT/f'fold_{fold:02d}_features.csv',index=False)
        pd.DataFrame(rows).to_csv(metrics_path,index=False)
        cache.clear(warn=False)
    summarize()

if __name__=='__main__':main()
