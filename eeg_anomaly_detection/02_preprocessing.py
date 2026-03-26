import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import os
import warnings
warnings.filterwarnings('ignore')
# 1. Загрузка данных
df = pd.read_csv('EEG_data.csv')
missing = df.isnull().sum()
missing_with_values = missing[missing > 0]
print(f"Колонки с пропусками: {len(missing_with_values)}")
print(missing_with_values)
df = df.loc[:, ~df.columns.str.contains('Unnamed')]
#Исправление пропусков для education и IQ
if 'education' in df.columns:
    # Образование - категориальная переменная, заполняем модой
    education_mode = df['education'].mode()[0]
    df['education'] = df['education'].fillna(education_mode)
    print(f"education: заполнено {missing_with_values.get('education', 0)} пропусков модой = {education_mode}")

if 'IQ' in df.columns:
    # IQ - числовая переменная, заполняем медианой
    iq_median = df['IQ'].median()
    df['IQ'] = df['IQ'].fillna(iq_median)
# 2. Целевая переменная
df['is_healthy'] = (df['specific.disorder'] == 'Healthy control').astype(int)
print("Баланс классов:", df['is_healthy'].value_counts(normalize=True).round(3))

# 3. Фильтр по возрасту 18-30
df_young = df[(df['age'] >= 18) & (df['age'] <= 30)].copy()
print(f"После фильтра 18-30: {df_young.shape}")

# 4. ПРИЗНАКИ - AB ритмы (114 штук!)
features_ab = [col for col in df_young.columns if col.startswith('AB.')]
print(f"EEG-признаки (AB): {len(features_ab)}")

X = df_young[features_ab].copy()
y = df_young['is_healthy']

print("Пропуски в EEG-признаках:", X.isnull().sum().sum())
print("Доля пропусков:", X.isnull().sum().sum() / (X.shape[0] * X.shape[1]))

# 5. Разбиение train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 6. Стандартизация
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Train: {X_train_scaled.shape[0]} samp, {X_train_scaled.shape[1]} feat")
print(f"Test:  {X_test_scaled.shape[0]} samp, {X_test_scaled.shape[1]} feat")
print("Баланс train:", np.bincount(y_train) / len(y_train))
print("Баланс test: ", np.bincount(y_test) / len(y_test))

# 7. Корреляционная матрица
print("\nКорреляции между ритмами (сумма по каналам):")
bands = {
    'Delta': features_ab[:18],
    'Theta': features_ab[18:36],
    'Alpha': features_ab[36:54],
    'Beta': features_ab[54:72]
}

df_bands = pd.DataFrame()
for name, cols in bands.items():
    df_bands[name] = X_train[cols].sum(axis=1)

corr_matrix = df_bands.corr()
plt.figure(figsize=(6,5))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Корреляции EEG ритмов')
plt.savefig('results/figures/band_correlations.png', dpi=300)
plt.show()
os.makedirs('processed_data', exist_ok=True)

np.save('processed_data/X_train_scaled.npy', X_train_scaled)
np.save('processed_data/X_test_scaled.npy', X_test_scaled)
np.save('processed_data/y_train.npy', y_train.values)
np.save('processed_data/y_test.npy', y_test.values)
np.save('processed_data/scaler.npy', scaler)