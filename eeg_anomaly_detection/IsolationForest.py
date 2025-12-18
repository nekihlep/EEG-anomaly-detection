import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')
#Data
df = pd.read_csv ('EEG_data.csv')
print (df.head(5))
healthy_count = len(df[df['main.disorder'] == 'Healthy control'])
anomaly_count = len(df[df['main.disorder'] != 'Healthy control'])
print(f"Healthy people : {healthy_count}" )
print(f"Not healthy: {anomaly_count}")
total_patients = len(df)
real_contamination = anomaly_count / total_patients
print(f"Recommended contamination: {real_contamination:.3f}")

# Labels
eeg_features = []
for col in df.columns:
    if col.startswith('AB.'):
        eeg_features.append(col)
print(f"\nFound {len(eeg_features)} EEG features")
#Groups
#Learning group: healthy
X_healthy = df[df['main.disorder'] == 'Healthy control'][eeg_features]
# Test group : all
X_all = df[eeg_features]
#Normalization
scaler = StandardScaler()
X_healthy_scaled = scaler.fit_transform(X_healthy)
X_all_scaled = scaler.transform(X_all)
# IsolationForest
model = IsolationForest(contamination=0.5, random_state=42, n_estimators=200)
model.fit(X_healthy_scaled)
predictions = model.predict(X_all_scaled)
df['predicted_isolation'] = predictions
df['predicted_status'] = np.where(
    df['predicted_isolation'] == -1,
    'ANOMALY_DETECTED',
    'NORMAL_EEG'
)
df['predicted_binary'] = np.where(predictions == -1, 1, 0)
print("\n Prediction Format:")
print("="*50)
print("-1 = Anomaly (IsolationForest)")
print(" 1 = Normal (IsolationForest)")
print(" 0 = Normal (Binary)")
print(" 1 = Anomaly (Binary)")

#Scale of results
df['true_binary'] = (df['main.disorder'] != 'Healthy control').astype(int)
true_labels = df['true_binary']
pred_labels = df['predicted_binary']
accuracy = accuracy_score(true_labels, pred_labels)
precision = precision_score(true_labels, pred_labels)  # Точность
recall = recall_score(true_labels, pred_labels)
print("\n📈 Metrics:")
print("="*40)
print(f"Accuracy:  {accuracy:.2%}")
print(f"Precision: {precision:.2%} ")
print(f"Recall:     {recall:.2%}  ")
