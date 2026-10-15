"""Render the computed screening results as two English summary tables."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

ROOT=Path(__file__).resolve().parent


def render_tables(folder=None):
    folder=Path(folder) if folder else ROOT/'results'/'screening_comparison'
    data=pd.read_csv(folder/'fold_metrics.csv')
    metrics=['sensitivity','specificity','balanced_accuracy','roc_auc']
    grouped=data.groupby(['model','mode'])[metrics].agg(['mean','std'])
    labels={'Dummy':'Dummy classifier','L1':'L1 logistic regression','RF':'Random Forest',
            'L1_RF':'L1 selection + Random Forest','AE_RF':'Autoencoder + Random Forest'}
    columns=['Model','Sensitivity','Specificity','Balanced accuracy','ROC AUC']
    fig,axes=plt.subplots(2,1,figsize=(14,6.7))
    for ax,mode,title in zip(axes,['standard','screening_90'],
        ['Standard classification','Screening: calibration sensitivity target ≥ 0.90']):
        rows=[]
        for model in ['Dummy','L1','RF','L1_RF','AE_RF']:
            row=[labels[model]]
            for metric in metrics:
                mean,sd=grouped.loc[(model,mode),metric]
                row.append(f'{mean:.3f} ± {sd:.3f}')
            rows.append(row)
        ax.axis('off')
        ax.set_title(title,fontsize=14,pad=10)
        table=ax.table(cellText=rows,colLabels=columns,cellLoc='center',
                       colWidths=[.34,.165,.165,.165,.165],bbox=[0,.08,1,.84])
        table.auto_set_font_size(False);table.set_fontsize(11)
        for (row,col),cell in table.get_celld().items():
            cell.set_edgecolor('#cccccc')
            if row==0:
                cell.set_facecolor('#e8edf2');cell.get_text().set_fontweight('bold')
            elif col==0:
                cell.get_text().set_ha('left')
    fig.text(.5,.018,'Mean ± SD across 10 outer folds; test sensitivity may differ from the calibration target.',
             ha='center',fontsize=10)
    fig.subplots_adjust(left=.025,right=.975,top=.94,bottom=.06,hspace=.30)
    path=folder/'final_tables.png'
    fig.savefig(path,dpi=200,facecolor='white');plt.close(fig)
    return path


if __name__=='__main__':
    print(render_tables())
