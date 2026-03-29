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

X_train = np.load('processed_data/X_train_scaled.npy')
X_test = np.load('processed_data/X_test_scaled.npy')
y_train = np.load('processed_data/y_train.npy')
y_test = np.load('processed_data/y_test.npy')

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
plt.title(f'L1 Importance (active: {np.sum(importance>0)}/32 features)')
plt.xlabel('AE Feature Index')
plt.savefig('results/figures/logreg_l1_importance.png', dpi=300)
plt.show()

# Confusion Matrix
plt.figure(figsize=(8,6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds',
            xticklabels=['Patient', 'Healthy'],
            yticklabels=['Patient', 'Healthy'])
plt.title('Logistic L1 (F1-optimal)')
plt.savefig('results/figures/logreg_l1_cm.png', dpi=300)
plt.show()

import pickle
import os
os.makedirs('processed_data/models', exist_ok=True)
pickle.dump(logreg_l1_cv, open('processed_data/models/logreg_l1_cv.pkl', 'wb'))
np.save('processed_data/models/logreg_coef.npy', logreg_l1_cv.coef_[0])




