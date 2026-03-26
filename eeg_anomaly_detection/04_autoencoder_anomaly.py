import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import (precision_score, recall_score, f1_score,
                           classification_report, confusion_matrix)
import seaborn as sns

# 1. Загрузка
X_train = np.load('processed_data/X_train_scaled.npy')
X_test = np.load('processed_data/X_test_scaled.npy')
y_train = np.load('processed_data/y_train.npy')
y_test = np.load('processed_data/y_test.npy')

# 2. Healthy из train
X_healthy_train = X_train[y_train == 1]
print(f"Healthy для обучения: {X_healthy_train.shape[0]} пациентов")

# 3. Автоэнкодер
autoencoder = MLPRegressor(
    hidden_layer_sizes=(64, 32, 16, 32, 64),
    activation='relu', solver='adam', max_iter=500,
    batch_size=32, random_state=42
)
autoencoder.fit(X_healthy_train, X_healthy_train)

# 4. Reconstruction error
X_test_rec = autoencoder.predict(X_test)
mse_test = np.mean(np.power(X_test - X_test_rec, 2), axis=1)

healthy_mse = mse_test[y_test == 1]
threshold = np.percentile(healthy_mse, 40)
y_pred = (mse_test <= threshold).astype(int)
print(f"Порог: {threshold:.3f} (40-й процентиль)")

# 6. Метрики
print(classification_report(y_test, y_pred, target_names=['Больной', 'Здоровый']))

# 7. Heatmap
plt.figure(figsize=(8,6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds',
            xticklabels=['Больной','Здоровый'],
            yticklabels=['Больной','Здоровый'])
plt.title('Autoencoder: Confusion Matrix (порог 40%)')
plt.ylabel('Истинные')
plt.xlabel('Предсказанные')
plt.savefig('results/figures/autoencoder_cm.png', dpi=300)
plt.show()
