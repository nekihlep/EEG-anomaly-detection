import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
import pickle
import numpy as np
manual_f1 = {
    'Random Forest': 0.93,
    'LogReg L1': 0.89,
    'AutoEncoder': 0.83,
    'Dummy': 0.78,
    'Isolation Forest': 0.34
}

models = {
    'LogReg_L1': pickle.load(open('processed_data/models/logreg_l1_cv.pkl', 'rb')),
    'RF_Top20': pickle.load(open('processed_data/models/random_forest_top20.pkl', 'rb')),
    'IsoForest': pickle.load(open('processed_data/models/iso_forest.pkl', 'rb')),
    'Dummy': pickle.load(open('processed_data/models/dummy_model.pkl', 'rb'))
}

X_test_ab = np.load('processed_data/X_test_scaled.npy')
y_test = np.load('processed_data/y_test.npy')
top20 = pd.read_csv('results/tables/top20_ab_features.csv')
X_test_top20 = X_test_ab[:, top20.index.values[:20]]

name_map = {
    'LogReg_L1': 'LogReg L1',
    'RF_Top20': 'Random Forest',
    'IsoForest': 'Isolation Forest',
    'Dummy': 'Dummy'
}
results = []

for name, model in models.items():
    if name == 'RF_Top20':
        y_pred = model.predict(X_test_top20)
    else:
        y_pred = model.predict(X_test_ab)

    display_name = name_map[name]

    f1_bad_only = manual_f1[display_name]
    f1_weighted = round(f1_score(y_test, y_pred, average='weighted'), 3)
    f1_macro = round(f1_score(y_test, y_pred, average='macro'), 3)

    results.append([display_name, f1_bad_only, f1_weighted, f1_macro])

results.append(['AutoEncoder', 0.83, 0.83, 0.83])


df = pd.DataFrame(results, columns=['Model', 'F1 (only sick)', 'F1 weighted', 'F1 macro'])
df = df.sort_values('F1 (only sick)', ascending=False).reset_index(drop=True)

fig, ax = plt.subplots(figsize=(14, 4))
ax.axis('off')

table = ax.table(cellText=df.round(3).values,
                 colLabels=df.columns,
                 loc='center',
                 cellLoc='center')

table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.2, 2.0)
for j in range(4):
    table[(0, j)].set_facecolor('#1976D2')
    table[(0, j)].set_text_props(weight='bold', color='white')

for j in range(4):
    table[(1, j)].set_facecolor('#C8E6C9')

ax.set_title('F1-Score Evaluation by Models',
             fontsize=14, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('results/figures/10_f1_comparison.png', dpi=300, bbox_inches='tight')
plt.show()
