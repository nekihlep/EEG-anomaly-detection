import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os

warnings.filterwarnings('ignore')
os.makedirs('results/figures', exist_ok=True)
sns.set_style("whitegrid")

# Load data
df = pd.read_csv('EEG_data.csv')
df['is_healthy'] = (df['specific.disorder'] == 'Healthy control').astype(int)

print(f"Dataset: {df.shape[0]} patients")

# Age groups
age_bins = [18, 30, 50, 72]
age_labels = ['18-30', '31-50', '51+']
df['age_group'] = pd.cut(df['age'], bins=age_bins, labels=age_labels, include_lowest=True)

age_groups = age_labels
group_data = df.groupby('age_group')['is_healthy'].agg(['count', 'sum']).round(0).astype(int)
healthy_counts = group_data['sum'].values
sick_counts = group_data['count'] - group_data['sum']
print("Age groups:", dict(zip(age_groups, zip(healthy_counts, sick_counts))))

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Age distribution histogram
axes[0,0].hist(df['age'], bins=15, color='steelblue', alpha=0.7, edgecolor='black')
axes[0,0].set_title('Age Distribution')
axes[0,0].grid(True, alpha=0.3)

# 2. Gender pie chart
sex_counts = df['sex'].value_counts()
axes[0,1].pie(sex_counts.values, labels=sex_counts.index, autopct='%1.1f%%', colors=['#3498db', '#e74c3c'])
axes[0,1].set_title('Gender Distribution')

# 3. Stacked bar by age groups
x = np.arange(len(age_groups))
axes[1,0].bar(x, healthy_counts, label='Healthy', color='green', alpha=0.8)
axes[1,0].bar(x, sick_counts, bottom=healthy_counts, label='Patients', color='red', alpha=0.8)
axes[1,0].set_xticks(x)
axes[1,0].set_xticklabels(age_groups)
axes[1,0].set_title('Distribution by Age Groups')
axes[1,0].legend()

# 4. Healthy percentage by age group
percentages = [h/(h+s)*100 for h,s in zip(healthy_counts, sick_counts)]
axes[1,1].bar(age_groups, percentages, color='orange', alpha=0.8)
axes[1,1].set_ylabel('Healthy %, %')
axes[1,1].set_title('Healthy Proportion by Age Group')
for i, pct in enumerate(percentages):
    axes[1,1].text(i, pct+1, f'{pct:.1f}%', ha='center')

plt.suptitle('Demographic Analysis', fontsize=16)
plt.tight_layout()
plt.savefig('results/figures/demographic_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# Top-5 diagnoses
top_diag = df['specific.disorder'].value_counts().head(5)
plt.figure(figsize=(10,6))
sns.barplot(x=top_diag.values, y=top_diag.index, palette='viridis')
plt.title('Top-5 Diagnoses')
plt.savefig('results/figures/top_diagnoses.png', dpi=300)
plt.show()