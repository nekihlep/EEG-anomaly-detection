import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns

electrodes = ['FP1','FP2','F7','F3','Fz','F4','F8','T3','C3','Cz','C4','T4','T5','P3','Pz','P4','T6','O1','O2',
              'FPz','AF3','AF4','F5','F1','F2','F6','T7','T8']

coef = np.load('processed_data/models/logreg_coef.npy')

features_real = []
for rhythm, start in zip(['Delta','Theta','Alpha','Beta','HighBeta','Gamma'], [0,28,56,84,112,140]):
    for i, elec in enumerate(electrodes):
        if start+i < 114:
            features_real.append(f"{rhythm}.{elec}")

df_features = pd.DataFrame({
    'feature': features_real[:114],
    'importance': np.abs(coef),
    'rhythm': [f.split('.')[0] for f in features_real[:114]],
    'electrode': [f.split('.')[-1] for f in features_real[:114]]
})

top20 = df_features.nlargest(20, 'importance')[['feature','rhythm','electrode','importance']].round(4)

# Таблицы
os.makedirs('results/tables', exist_ok=True)
top20.to_csv('results/tables/top20_electrodes.csv', index=False)
df_features.to_csv('results/tables/ab_electrodes_full.csv', index=False)

os.makedirs('results/figures', exist_ok=True)

# 1. Ритмы (столбцы)
plt.figure(figsize=(10,6))
rhythm_mean = df_features.groupby('rhythm')['importance'].mean()
plt.bar(rhythm_mean.index, rhythm_mean.values, color=['blue','orange','green','red','purple','brown'])
plt.title('Средняя важность L1 по ритмам EEG')
plt.ylabel('|коэффициент|')
plt.xticks(rotation=45)
for i, v in enumerate(rhythm_mean.values):
    plt.text(i, v+0.5, f'{v:.1f}', ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig('results/figures/01_rhythm_importance.png', dpi=300, bbox_inches='tight')
plt.show()

# 2. ТОП-20 горизонтально
plt.figure(figsize=(12,8))
top20.sort_values('importance').plot(x='feature', y='importance', kind='barh',
                                     legend=False, color='steelblue')
plt.title('ТОП-20 наиболее важных EEG-признаков')
plt.xlabel('|коэффициент|')
plt.tight_layout()
plt.savefig('results/figures/02_top20_features.png', dpi=300, bbox_inches='tight')
plt.show()

# 3. Heatmap ритмы×электроды (только основные 4 ритма)
plt.figure(figsize=(14,10))
pivot_main = df_features[df_features.rhythm.isin(['Delta','Theta','Alpha','Beta'])].pivot_table(
    values='importance', index='electrode', columns='rhythm', aggfunc='mean')
sns.heatmap(pivot_main, annot=True, fmt='.1f', cmap='Reds', cbar_kws={'label': '|коэффициент|'})
plt.title('Важность признаков: Электроды × Основные ритмы')
plt.xlabel('Ритм')
plt.ylabel('Электрод')
plt.tight_layout()
plt.savefig('results/figures/03_electrodes_rhythms.png', dpi=300, bbox_inches='tight')
plt.show()

# 4. ТОП-5 электродов
plt.figure(figsize=(10,6))
electrode_mean = df_features.groupby('electrode')['importance'].mean().nlargest(10)
plt.bar(electrode_mean.index, electrode_mean.values, color='darkgreen')
plt.title('ТОП-10 электродов по важности')
plt.ylabel('|коэффициент|')
plt.xticks(rotation=45)
for i, v in enumerate(electrode_mean.values):
    plt.text(i, v+0.3, f'{v:.1f}', ha='center')
plt.tight_layout()
plt.savefig('results/figures/04_top_electrodes.png', dpi=300, bbox_inches='tight')
plt.show()