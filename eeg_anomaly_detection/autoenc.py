import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
import warnings

warnings.filterwarnings('ignore')

# 1. Load data
df = pd.read_csv('EEG_data.csv')
print(f"Data: {df.shape[0]} patients, {df.shape[1]} features")

# EXPLORATORY ANALYSIS

print("Age statistics:")
print(f"Average age: {df['age'].mean():.1f} years")
print(f"Min age: {df['age'].min()} years")
print(f"Max age: {df['age'].max()} years")
print(f"Standard deviation: {df['age'].std():.1f} years")

# By diagnosis
print("\nAge by diagnosis (top-5):")
for diagnosis in df['specific.disorder'].value_counts().head().index:
    age_data = df[df['specific.disorder'] == diagnosis]['age']
    print(f"{diagnosis:30s}: {age_data.mean():5.1f} ± {age_data.std():4.1f} years (n={len(age_data)})")

# By gender
print("\nAge by gender:")
print(f"Men: {df[df['sex'] == 'M']['age'].mean():.1f} ± {df[df['sex'] == 'M']['age'].std():.1f} years")
print(f"Women: {df[df['sex'] == 'F']['age'].mean():.1f} ± {df[df['sex'] == 'F']['age'].std():.1f} years")

plt.figure(figsize=(10, 6))
n, bins, patches = plt.hist (df['age'],color = 'red',bins = 8,ec = 'black')
plt.title('Age Distribution ', fontsize=14)
plt.xlabel('Age', fontsize=12)
plt.ylabel('Count', fontsize=12)

for count, patch in zip(n, patches):
    height = count
    x = patch.get_x() + patch.get_width() / 2
    y = height

    if height > 0:
        plt.text(x, y + 0.5, f'{int(height)}',
                 ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('results/age_distrib.png')
plt.show()

plt.figure(figsize=(10, 6))
sex_counts = df['sex'].value_counts()
bars = plt.bar(sex_counts.index,
               sex_counts.values,
               color=['blue', 'pink'],
               alpha=0.7)
plt.title('Sex Distribution ', fontsize=14)
plt.xlabel('Sex', fontsize=12)
plt.ylabel('Count', fontsize=12)

for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + 0.5,
             f'{int(height)}', ha='center', va='bottom',
             fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig('results/sex_distrib.png')
plt.show()
# 1. Variance
variance = df['age'].var()
print(f"Age variance: {variance:.2f}")

# 2. Standard deviation
std_dev = df['age'].std()
print(f"Standard deviation: {std_dev:.2f} years")

# 3. Coefficient of variation
mean_age = df['age'].mean()
cv = (std_dev / mean_age) * 100 if mean_age > 0 else 0
print(f"Coefficient of variation: {cv:.1f}%")

total = len(df)
male_total = (df['sex'] == 'M').sum()
female_total = (df['sex'] == 'F').sum()

print(f"Total patients: {total}")
print(f"Men: {male_total} ({male_total / total * 100:.1f}%)")
print(f"Women: {female_total} ({female_total / total * 100:.1f}%)")

# By diagnosis
print("\nDISTRIBUTION BY DIAGNOSIS:")
print("-" * 50)

for diagnosis in df['specific.disorder'].value_counts().head(8).index:
    diag_df = df[df['specific.disorder'] == diagnosis]
    diag_total = len(diag_df)
    male = (diag_df['sex'] == 'M').sum()
    female = (diag_df['sex'] == 'F').sum()

    print(f"{diagnosis:35s}: M={male:3d} ({male / diag_total * 100:4.1f}%), " +
          f"F={female:3d} ({female / diag_total * 100:4.1f}%), " +
          f"Total={diag_total:3d}")

# Create age groups
age_bins = [18, 30, 50, 72]
age_labels = ['18-30', '31-50', '51+']

df['age_group'] = pd.cut(df['age'], bins=age_bins, labels=age_labels, include_lowest=True)

# 2. Distribution by groups
print("=" * 60)
print("DISTRIBUTION BY AGE GROUPS:")
print("=" * 60)

for group in age_labels:
    group_data = df[df['age_group'] == group]
    total = len(group_data)
    healthy = (group_data['main.disorder'] == 'Healthy control').sum()
    sick = total - healthy

    print(f"{group:5s}: Total={total:3d} patients")
    print(f"       Healthy: {healthy:3d} ({healthy / total * 100:5.1f}%)")
    print(f"       Sick:    {sick:3d} ({sick / total * 100:5.1f}%)")

# 2. Data preparation
df['is_healthy'] = (df['main.disorder'] == 'Healthy control').astype(int)
eeg_features = [col for col in df.columns if col.startswith('AB.')]

X = df[eeg_features].values
y = df['is_healthy'].values
X_healthy = X[y == 1]
X_all = X

# 3. Normalization
scaler = StandardScaler()
X_healthy_scaled = scaler.fit_transform(X_healthy)  # (X - mean) / standard_deviation
X_all_scaled = scaler.transform(X_all)

# 4. Autoencoder
print("\nTraining autoencoder...")
autoencoder = MLPRegressor(
    hidden_layer_sizes=(64, 32, 16, 32, 64),
    activation='relu',
    solver='adam',  # "How the neural network learns from errors"
    max_iter=500,  # "Maximum training steps"
    batch_size=32,  # "How many patients to process at once"
    random_state=42,
    verbose=False  # "Show training progress?"
)
autoencoder.fit(X_healthy_scaled, X_healthy_scaled)

# 5. Anomaly detection
X_rec = autoencoder.predict(X_all_scaled)
mse = np.mean(np.power(X_all_scaled - X_rec, 2), axis=1)  # axis=1 = average by ROW (0 = column)
df['rec_error'] = mse

# Threshold (90% of healthy)
healthy_errors = mse[y == 1]
threshold = np.percentile(healthy_errors, 90)
df['predicted_healthy'] = (mse <= threshold).astype(int)

# 6. Evaluation
accuracy = accuracy_score(y, df['predicted_healthy'])
precision = precision_score(y, df['predicted_healthy'], zero_division=0)
recall = recall_score(y, df['predicted_healthy'])
f1 = f1_score(y, df['predicted_healthy'])

print(f"\nResults:")
print(f"Accuracy:  {accuracy:.2%}")
print(f"Precision: {precision:.2%}")
print(f"Recall:    {recall:.2%}")
print(f"F1-Score:  {f1:.2%}")

# FIRST GRAPH - Patients
plt.figure(figsize=(12, 6))

# Get data
first_healthy_idx = np.where(y == 1)[0][0]
second_healthy_idx = np.where(y == 1)[0][1]
first_sick_idx = np.where(y == 0)[0][0]
second_sick_idx = np.where(y == 0)[0][545]
third_sick_idx = np.where(y == 0)[0][117]

healthy_eeg = X[first_healthy_idx]
healthy_eeg_2 = X[second_healthy_idx]
sick_eeg = X[first_sick_idx]
sick_eeg_2 = X[second_sick_idx]
sick_eeg_3 = X[third_sick_idx]

healthy_diagnosis = df.iloc[first_healthy_idx]['specific.disorder']
healthy_diagnosis_2 = df.iloc[second_healthy_idx]['specific.disorder']
sick_diagnosis = df.iloc[first_sick_idx]['specific.disorder']
sick_diagnosis_2 = df.iloc[second_sick_idx]['specific.disorder']
sick_diagnosis_3 = df.iloc[third_sick_idx]['specific.disorder']

# Draw first graph
n_features = 85
x_positions = range(n_features)

plt.plot(x_positions, healthy_eeg[:n_features],
         'go-', linewidth=2, markersize=6,
         label=f'Healthy ({healthy_diagnosis})')
plt.plot(x_positions, healthy_eeg_2[:n_features],
         'g*-', linewidth=2, markersize=8,
         label=f'Healthy ({healthy_diagnosis_2})')

plt.plot(x_positions, sick_eeg[:n_features],
         color='red', marker='o', linewidth=2, markersize=6,
         label=f'Sick ({sick_diagnosis})')
plt.plot(x_positions, sick_eeg_2[:n_features],
         color='magenta', marker='o', linewidth=2, markersize=6,
         label=f'Sick 2 ({sick_diagnosis_2})')
plt.plot(x_positions, sick_eeg_3[:n_features],
         color='pink', marker='o', linewidth=2, markersize=6,
         label=f'Sick 3 ({sick_diagnosis_3})')

plt.xlabel('EEG features (85 out of 114)', fontsize=12)
plt.ylabel('Feature value', fontsize=12)
plt.title('Comparison of EEG feagures in healthy and sick patients', fontsize=14, pad=20)
plt.legend(fontsize=10, loc='upper right')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('results/Comparison_EEG_feat.png')
plt.show()

# SECOND GRAPH - Metrics
plt.figure(figsize=(10, 6))

metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
values = [0.8889, 0.4722, 0.8947, 0.6182]
colors = ['blue', 'orange', 'green', 'red']

bars = plt.bar(metrics, values, color=colors)
plt.ylim(0, 1)
plt.ylabel('Value', fontsize=12)
plt.title('Autoencoder quality metrics', fontsize=14, pad=20)
plt.grid(True, alpha=0.3)

for bar, val in zip(bars, values):
    plt.text(bar.get_x() + bar.get_width() / 2, val + 0.02,
             f'{val:.1%}', ha='center', fontsize=11)

plt.tight_layout()
plt.savefig('results/Autoenc_metrics.png')
plt.show()
plt.close()


def plot_spectral_profile(patient_idx):
    # Group features by frequency bands
    freq_bands = {
        'Delta': [col for col in df.columns if 'AB.A.delta' in col],
        'Theta': [col for col in df.columns if 'AB.B.theta' in col],
        'Alpha': [col for col in df.columns if 'AB.C.alpha' in col],
        'Beta': [col for col in df.columns if 'AB.D.beta' in col],
        'High Beta': [col for col in df.columns if 'AB.E.highbeta' in col],
        'Gamma': [col for col in df.columns if 'AB.F.gamma' in col]
    }

    avg_powers = {}
    for band, features in freq_bands.items():
        avg_powers[band] = df.iloc[patient_idx][features].mean()

    # Visualization
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    bands = list(avg_powers.keys())
    powers = list(avg_powers.values())
    bars = plt.bar(bands, powers, color=['blue', 'green', 'red', 'orange', 'purple', 'brown'])
    plt.ylabel('Average power (μV²)')
    plt.title(f'Spectral profile\n{df.iloc[patient_idx]["specific.disorder"]}')
    plt.xticks(rotation=45)

    # Add values
    for bar, val in zip(bars, powers):
        plt.text(bar.get_x() + bar.get_width() / 2, val,
                 f'{val:.2f}', ha='center', va='bottom')


# PREPARE DATA FOR NORM CREATION
df_young = df[df['age'] <= 30].copy()


def calculate_relative_power(row):
    freq_bands = {
        'Delta': [col for col in df.columns if 'AB.A.delta' in col],
        'Theta': [col for col in df.columns if 'AB.B.theta' in col],
        'Alpha': [col for col in df.columns if 'AB.C.alpha' in col],
        'Beta': [col for col in df.columns if 'AB.D.beta' in col],
        'High Beta': [col for col in df.columns if 'AB.E.highbeta' in col],
        'Gamma': [col for col in df.columns if 'AB.F.gamma' in col]
    }

    total_power = 0
    for band_features in freq_bands.values():
        total_power += row[band_features].sum()

    relative_powers = {}
    for band_name, band_features in freq_bands.items():
        band_power = row[band_features].sum()
        relative_powers[f'{band_name}_rel'] = (band_power / total_power * 100) if total_power > 0 else 0

    return pd.Series(relative_powers)


print("\nCalculating relative power...")
relative_powers_df = df_young.apply(calculate_relative_power, axis=1)
df_young = pd.concat([df_young, relative_powers_df], axis=1)

relative_features = [col for col in df_young.columns if '_rel' in col]
print(f"Created {len(relative_features)} relative features:")
for feat in relative_features:
    print(f"  {feat}")

healthy_norm_rel = df_young[df_young['is_healthy'] == 1]
print(f"\nHealthy for norm: {len(healthy_norm_rel)} people")

norm_stats_rel = {}
for feature in relative_features:
    values = healthy_norm_rel[feature].dropna()
    if len(values) > 10:
        norm_stats_rel[feature] = {
            'mean': values.mean(),
            'std': values.std(),
            'n': len(values),
            'ci_95_lower': values.mean() - 1.96 * values.std() / np.sqrt(len(values)),
            'ci_95_upper': values.mean() + 1.96 * values.std() / np.sqrt(len(values))
        }

print(f"\nCreated norm for {len(norm_stats_rel)} relative features")


def simple_diagnosis_analysis(diagnosis_name):
    patients = df_young[df_young['specific.disorder'] == diagnosis_name]

    if len(patients) < 10:
        print(f"Too few patients: {len(patients)}")
        return

    print(f"Patients: {len(patients)}")
    print("\nAverage relative power (%):")
    print(f"{'Band':<12} {'Healthy':<10} {'Patients':<10} {'Diff':<10} {'%':<8}")
    print("-" * 60)

    for feature in relative_features:
        healthy_mean = float(healthy_norm_rel[feature].mean())
        patient_mean = float(patients[feature].mean())
        diff = patient_mean - healthy_mean

        if healthy_mean != 0:
            pct_diff = (diff / healthy_mean) * 100
        else:
            pct_diff = 0

        band = feature.split('_')[0]
        direction = "↑" if diff > 0 else "↓"

        if abs(pct_diff) > 15:
            marker = "⚠️ "
        elif abs(pct_diff) > 10:
            marker = "• "
        else:
            marker = "  "

        print(f"{marker}{band:<10} {healthy_mean:<10.1f} {patient_mean:<10.1f} " +
              f"{diff:<10.1f} {pct_diff:+.1f}% {direction}")


print("RELATIVE POWER ANALYSIS")

for diagnosis in ['Depressive disorder', 'Schizophrenia', 'Alcohol use disorder', 'Acute stress disorder',
                  'Panic disorder', 'Behavioral addiction disorder', 'Obsessive compulsitve disorder',
                  'Social anxiety disorder', 'Bipolar disorder']:
    patient_count = (df_young['specific.disorder'] == diagnosis).sum()
    if patient_count >= 20:
        print(f"\n>>> Analyzing {diagnosis} ({patient_count} patients)")
        simple_diagnosis_analysis(diagnosis)
    else:
        print(f"\n>>> Skipping {diagnosis} (only {patient_count} patients)")
