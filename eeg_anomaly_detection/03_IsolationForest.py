import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import os
X_train = np.load('processed_data/X_train_scaled.npy')
X_test = np.load('processed_data/X_test_scaled.npy')
y_train = np.load('processed_data/y_train.npy')
y_test = np.load('processed_data/y_test.npy')

# 2. Healthy из train (one-class!)
X_healthy_train = X_train[y_train == 1]
print(f"Healthy train: {X_healthy_train.shape[0]} из {X_train.shape[0]}")

# 3. Isolation Forest (обучаем ТОЛЬКО на здоровых!)
iso_forest = IsolationForest(
    contamination=0.5,
    random_state=42,
    n_estimators=200
)
iso_forest.fit(X_healthy_train)

# 4. Предсказания на test
predictions_test = iso_forest.predict(X_test)
y_pred_binary = (predictions_test == -1).astype(int)

# 5. Метрики
print("\n=== РЕЗУЛЬТАТЫ ===")
print(classification_report(y_test, y_pred_binary, target_names=['Больной', 'Здоровый']))

# 6. Confusion Matrix
cm = confusion_matrix(y_test, y_pred_binary)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Больной', 'Здоровый'],
            yticklabels=['Больной', 'Здоровый'])
plt.title('Isolation Forest: Confusion Matrix')
plt.ylabel('True')
plt.xlabel('Predicted')
plt.savefig('results/figures/isolation_forest_cm.png', dpi=300)
plt.show()

print("График: results/figures/isolation_forest_cm.png")
