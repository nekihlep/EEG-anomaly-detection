import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV, learning_curve
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
X_train_ab = np.load('processed_data/X_train_scaled.npy')
X_test_ab = np.load('processed_data/X_test_scaled.npy')
X_train_ae = np.load('processed_data/ae_features_train.npy')
X_test_ae  = np.load('processed_data/ae_features_test.npy')
y_train = np.load('processed_data/y_train.npy')
y_test = np.load('processed_data/y_test.npy')


top20 = pd.read_csv('results/tables/top20_ab_features.csv')

X_train_top20 = X_train_ab[:, top20.index.values[:20]]
X_test_top20  = X_test_ab[:, top20.index.values[:20]]

print(f"Размеры: AB(114): {X_train_ab.shape} | ТОП20: {X_train_top20.shape} | AE(32): {X_train_ae.shape}")

rf_params = {
    'n_estimators': [100, 200],
    'max_depth': [6, 10, None],
    'min_samples_split': [2, 5],
    'class_weight': ['balanced']
}

rf = RandomForestClassifier(random_state=42, n_jobs=-1)
grid_rf = GridSearchCV(rf, rf_params, cv=StratifiedKFold(5), scoring='recall_macro', n_jobs=-1,verbose=1)
grid_rf.fit(X_train_top20, y_train)

best_rf = grid_rf.best_estimator_
y_pred_top20 = best_rf.predict(X_test_top20)

importances_rf = best_rf.feature_importances_
feature_names = [f"AB_{i}" for i in top20.index.values[:20]]
rf_importance = pd.DataFrame({
    'feature': feature_names,
    'importance': importances_rf
}).sort_values('importance', ascending=False)
# Learning curves
train_sizes, train_scores, val_scores = learning_curve(
    best_rf, X_train_top20, y_train, cv=5, scoring='f1_weighted',
    train_sizes=np.linspace(0.1, 1.0, 10), n_jobs=-1
)

plt.figure(figsize=(8,6))
plt.plot(train_sizes, train_scores.mean(axis=1), 'o-', label='Train', linewidth=2)
plt.plot(train_sizes, val_scores.mean(axis=1), 'o-', label='Val', linewidth=2)
plt.title('Learning Curve RF ТОП-20 AB', fontsize=14, fontweight='bold')
plt.xlabel('Размер выборки')
plt.ylabel('F1-weighted')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('results/figures/08_learning_curve.png', dpi=300, bbox_inches='tight')
plt.show()
train_final = train_scores.mean(axis=1)[-1]
val_final = val_scores.mean(axis=1)[-1]
gap = train_final - val_final

print("\n LEARNING CURVE — ЧИСЛА:")
print(f"Train F1 (финал):  {train_final:.3f}")
print(f"Val F1 (финал):    {val_final:.3f}")
print(f"ЗАЗОР Train-Val:   {gap:.3f}")
print(f"Стабилизация Val:  {val_final:.3f} при {train_sizes[-1]:.0f} примерах")

# 3. Confusion Matrix
plt.figure(figsize=(8,6))
cm = confusion_matrix(y_test, y_pred_top20)
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds',
            xticklabels=['Больной','Здоровый'],
            yticklabels=['Больной','Здоровый'],
            cbar_kws={'label': 'Количество'})
plt.title('RF ТОП-20: Матрица ошибок (Test)', fontsize=14, fontweight='bold')
plt.ylabel('Истинные')
plt.xlabel('Предсказанные')
plt.tight_layout()
plt.savefig('results/figures/08_confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()
# 8. Сохранение
pickle.dump(best_rf, open('processed_data/models/rf_top20.pkl', 'wb'))
rf_importance.to_csv('results/tables/rf_top20_importance.csv', index=False)
