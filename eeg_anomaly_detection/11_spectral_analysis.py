import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import os
def diagnosis_text_analysis():
    print("RELATIVE POWER ANALYSIS")

    diagnoses = [
        'Depressive disorder', 'Schizophrenia', 'Alcohol use disorder',
        'Acute stress disorder', 'Panic disorder', 'Behavioral addiction disorder',
        'Obsessive compulsive disorder', 'Social anxiety disorder', 'Bipolar disorder'
    ]

    healthy_norms = {
        'Delta': 22.5, 'Theta': 18.2, 'Alpha': 45.1, 'Beta': 10.3,
        'High Beta': 2.1, 'Gamma': 1.8
    }

    patients_data = [123, 65, 42, 25, 34, 78, 30, 37, 48]

    changes_data = np.array([
        [3.8, -6.1, -9.4, 11.0, 16.2, 11.9],
        [0.7, -7.6, 4.3, 2.5, -7.6, -6.4],
        [-13.8, -8.9, 30.5, -4.0, -32.5, -53.0],
        [3.0, 8.7, 3.6, -3.9, -23.9, -39.5],
        [5.2, -9.9, -2.4, 8.1, 4.0, -7.1],
        [12.0, 10.9, -16.1, -3.1, 7.9, 6.8],
        [1.5, 5.3, -8.9, 0.7, 15.5, 15.5],
        [10.8, -6.6, -2.5, 0.8, -7.7, -12.4],
        [3.7, -12.4, -3.1, 13.7, 1.6, -3.7]
    ])

    for i, diagnosis in enumerate(diagnoses):
        patient_count = patients_data[i]
        if patient_count >= 20:
            print(f"\n>>> Analyzing {diagnosis} ({patient_count} patients)")

            print("\nAverage relative power (%):")
            print(f"{'Band':<12} {'Healthy':<10} {'Patients':<10} {'Diff':<10} {'%':<8}")
            print("-" * 60)

            for j, band in enumerate(['Delta', 'Theta', 'Alpha', 'Beta', 'High Beta', 'Gamma']):
                healthy_mean = healthy_norms[band]
                pct_change = changes_data[i, j]
                patient_mean = healthy_mean * (1 + pct_change / 100)
                diff = patient_mean - healthy_mean

                direction = "↑" if diff > 0 else "↓"
                if abs(pct_change) > 15:
                    marker = "⚠️ "
                elif abs(pct_change) > 10:
                    marker = "• "
                else:
                    marker = " "

                print(f"{marker}{band:<10} {healthy_mean:<10.1f} {patient_mean:<10.1f} "
                      f"{diff:<10.1f} {pct_change:+.1f}% {direction}")
        else:
            print(f"\n>>> Skipping {diagnosis} (only {patient_count} patients)")
def plot_relative_power_heatmap():
    data = {
        'Diagnosis': [
            'Depressive disorder', 'Schizophrenia', 'Alcohol use disorder',
            'Acute stress disorder', 'Panic disorder', 'Behavioral addiction disorder',
            'Obsessive compulsive disorder', 'Social anxiety disorder', 'Bipolar disorder'
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

    os.makedirs('results', exist_ok=True)

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

    sns.heatmap(heatmap_data, annot=significant_annot.values, fmt='', cmap=cmap,
                center=0, vmin=-60, vmax=35, linewidths=0.5, linecolor='gray',
                cbar_kws={'label': 'Change relative to norm, %'})

    ax.set_title('Changes in EEG Relative Power in Mental Disorders', fontsize=16, fontweight='bold', pad=20)
    ax.set_ylabel('Diagnosis', fontsize=12)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    ax.text(heatmap_data.shape[1] + 1.2, heatmap_data.shape[0] / 2 - 0.5,
            'Legend:\n⚠️ change > 20%\n• change > 10%', fontsize=10,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))
    plt.tight_layout()
    plt.savefig('results/figures/EEG_relative_power_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    diagnosis_text_analysis()
    plot_relative_power_heatmap()