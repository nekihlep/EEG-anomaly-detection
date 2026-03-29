import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from sklearn.metrics import f1_score
import os
import pickle
X_train = np.load('processed_data/X_train_scaled.npy')
X_test = np.load('processed_data/X_test_scaled.npy')
y_train = np.load('processed_data/y_train.npy')
y_test = np.load('processed_data/y_test.npy')

# Healthy -- train
X_healthy_train = X_train[y_train == 1]
print(f"Healthy train: {X_healthy_train.shape[0]} из {X_train.shape[0]}")

#  Isolation Forest fits on healthy
iso_forest = IsolationForest(
    contamination=0.5,
    random_state=42,
    n_estimators=200
)
iso_forest.fit(X_healthy_train)

predictions_test = iso_forest.predict(X_test)
y_pred_binary = (predictions_test == -1).astype(int)
# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_binary)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Patient', 'Healthy'],
            yticklabels=['Patient', 'Healthy'])
plt.title('Isolation Forest: Confusion Matrix')
plt.ylabel('True')
plt.xlabel('Predicted')
plt.savefig('results/figures/isolation_forest_cm.png', dpi=300)
plt.show()

pickle.dump(iso_forest, open('processed_data/models/iso_forest.pkl', 'wb'))