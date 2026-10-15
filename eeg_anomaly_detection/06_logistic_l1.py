import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.linear_model import Ridge, Lasso
import seaborn as sns
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import (precision_score, recall_score, f1_score,
                           classification_report, confusion_matrix)
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

X_train = np.load('initial_project_artifacts/X_train_scaled.npy')
X_test = np.load('initial_project_artifacts/X_test_scaled.npy')
y_train = np.load('initial_project_artifacts/y_train.npy')
y_test = np.load('initial_project_artifacts/y_test.npy')

logreg_l1_cv = LogisticRegressionCV(
    Cs=10,
    penalty='l1',
    solver='liblinear',
    cv=5,
    scoring='recall_macro',
    max_iter=1000,
    random_state=42,
    n_jobs=-1,
    verbose=1
)

logreg_l1_cv.fit(X_train, y_train)
y_pred = logreg_l1_cv.predict(X_test)

print("The best C:", logreg_l1_cv.C_[0])
print("Active features:", np.sum(logreg_l1_cv.coef_ != 0))
# L1
plt.figure(figsize=(10,6))
importance = np.abs(logreg_l1_cv.coef_[0])
plt.bar(range(len(importance)), importance)
plt.title(f'L1 Importance (active: {np.sum(importance>0)}/114 features)')
plt.xlabel('Spectral Feature Index')
plt.savefig('results/figures/logreg_l1_importance.png', dpi=300)
plt.show()

# Confusion Matrix
plt.figure(figsize=(8,6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds',
            xticklabels=['Patient', 'Healthy'],
            yticklabels=['Patient', 'Healthy'])
plt.title('Logistic L1 (macro-recall tuning)')
plt.savefig('results/figures/logreg_l1_cm.png', dpi=300)
plt.show()

import pickle
import os
os.makedirs('initial_project_artifacts/models', exist_ok=True)
pickle.dump(logreg_l1_cv, open('initial_project_artifacts/models/logreg_l1_cv.pkl', 'wb'))
np.save('initial_project_artifacts/models/logreg_coef.npy', logreg_l1_cv.coef_[0])




