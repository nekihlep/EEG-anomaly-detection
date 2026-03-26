import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

data = {
    'Diagnosis': [
        'Depressive disorder',
        'Schizophrenia',
        'Alcohol use disorder',
        'Acute stress disorder',
        'Panic disorder',
        'Behavioral addiction disorder',
        'Obsessive compulsive disorder',
        'Social anxiety disorder',
        'Bipolar disorder'
    ],
    'Delta': [3.8, 0.7, -13.8, 3.0, 5.2, 12.0, 1.5, 10.8, 3.7],
    'Theta': [-6.1, -7.6, -8.9, 8.7, -9.9, 10.9, 5.3, -6.6, -12.4],
    'Alpha': [-9.4, 4.3, 30.5, 3.6, -2.4, -16.1, -8.9, -2.5, -3.1],
    'Beta': [11.0, 2.5, -4.0, -3.9, 8.1, -3.1, 0.7, 0.8, 13.7],
    'High Beta': [16.2, -7.6, -32.5, -23.9, 4.0, 7.9, 15.5, -7.7, 1.6],
    'Gamma': [11.9, -6.4, -53.0, -39.5, -7.1, 6.8, 15.5, -12.4, -3.7],

    'Patients': [123, 65, 42, 25, 34, 78, 30, 37, 48]
}

df = pd.DataFrame(data)
df.set_index('Diagnosis', inplace=True)

heatmap_data = df[['Delta', 'Theta', 'Alpha', 'Beta', 'High Beta', 'Gamma']]
patients_data = df['Patients']

fig, ax = plt.subplots(figsize=(14, 10))
colors = ['#8B0000', '#FF6B6B', '#FFE5E5', 'white', '#E5FFE5', '#6BFF6B', '#008B00']
n_bins = 100
cmap = LinearSegmentedColormap.from_list('custom_rd_gn', colors, N=n_bins)

significant_annot = heatmap_data.copy()
for col in significant_annot.columns:
    significant_annot[col] = significant_annot[col].apply(
        lambda x: f'⚠️{x:+.1f}%' if abs(x) > 20 else
                 (f'•{x:+.1f}%' if abs(x) > 10 else f'{x:+.1f}%')
    )
sns.heatmap(heatmap_data,
            annot=significant_annot.values,
            fmt='',
            cmap=cmap,
            center=0,
            vmin=-60,
            vmax=35,
            linewidths=0.5,
            linecolor='gray',
            cbar_kws={'label': 'Change relative to norm, %'})

ax.set_title('Changes in EEG Relative Power in Mental Disorders',
              fontsize=16, fontweight='bold', pad=20)
ax.set_ylabel('Diagnosis', fontsize=12)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

ax.text(heatmap_data.shape[1] + 1.2, heatmap_data.shape[0]/2 - 0.5,
        'Legend:\n⚠️ change > 20%\n• change > 10%',
        fontsize=10,
        bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))
plt.tight_layout()
plt.savefig('results/EEG_relative_power_heatmap.png')

plt.show()

##ЧЕРНОВИК ТОГО ЧТО БЫЛО
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

data = {
    'Diagnosis': [
        'Depressive disorder',
        'Schizophrenia',
        'Alcohol use disorder',
        'Acute stress disorder',
        'Panic disorder',
        'Behavioral addiction disorder',
        'Obsessive compulsive disorder',
        'Social anxiety disorder',
        'Bipolar disorder'
    ],
    'Delta': [3.8, 0.7, -13.8, 3.0, 5.2, 12.0, 1.5, 10.8, 3.7],
    'Theta': [-6.1, -7.6, -8.9, 8.7, -9.9, 10.9, 5.3, -6.6, -12.4],
    'Alpha': [-9.4, 4.3, 30.5, 3.6, -2.4, -16.1, -8.9, -2.5, -3.1],
    'Beta': [11.0, 2.5, -4.0, -3.9, 8.1, -3.1, 0.7, 0.8, 13.7],
    'High Beta': [16.2, -7.6, -32.5, -23.9, 4.0, 7.9, 15.5, -7.7, 1.6],
    'Gamma': [11.9, -6.4, -53.0, -39.5, -7.1, 6.8, 15.5, -12.4, -3.7],

    'Patients': [123, 65, 42, 25, 34, 78, 30, 37, 48]
}

df = pd.DataFrame(data)
df.set_index('Diagnosis', inplace=True)

heatmap_data = df[['Delta', 'Theta', 'Alpha', 'Beta', 'High Beta', 'Gamma']]
patients_data = df['Patients']

fig, ax = plt.subplots(figsize=(14, 10))
colors = ['#8B0000', '#FF6B6B', '#FFE5E5', 'white', '#E5FFE5', '#6BFF6B', '#008B00']
n_bins = 100
cmap = LinearSegmentedColormap.from_list('custom_rd_gn', colors, N=n_bins)

significant_annot = heatmap_data.copy()
for col in significant_annot.columns:
    significant_annot[col] = significant_annot[col].apply(
        lambda x: f'⚠️{x:+.1f}%' if abs(x) > 20 else
                 (f'•{x:+.1f}%' if abs(x) > 10 else f'{x:+.1f}%')
    )
sns.heatmap(heatmap_data,
            annot=significant_annot.values,
            fmt='',
            cmap=cmap,
            center=0,
            vmin=-60,
            vmax=35,
            linewidths=0.5,
            linecolor='gray',
            cbar_kws={'label': 'Change relative to norm, %'})

ax.set_title('Changes in EEG Relative Power in Mental Disorders',
              fontsize=16, fontweight='bold', pad=20)
ax.set_ylabel('Diagnosis', fontsize=12)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

ax.text(heatmap_data.shape[1] + 1.2, heatmap_data.shape[0]/2 - 0.5,
        'Legend:\n⚠️ change > 20%\n• change > 10%',
        fontsize=10,
        bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))
plt.tight_layout()
plt.savefig('results/EEG_relative_power_heatmap.png')

plt.show() вот это код хитмап и часть анализа из автоэкодера норм или можно как то упрсотить и улучшить def plot_spectral_profile(patient_idx):
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
print(f" {feat}")

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
marker = " "

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