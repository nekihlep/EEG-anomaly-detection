import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os

warnings.filterwarnings('ignore')
os.makedirs('results/figures', exist_ok=True)
sns.set_style("whitegrid")

# Загрузка
df = pd.read_csv('EEG_data.csv')
df['is_healthy'] = (df['specific.disorder'] == 'Healthy control').astype(int)

print(f"Данные: {df.shape[0]} пациентов")

# Возрастные группы
age_bins = [18, 30, 50, 72]
age_labels = ['18-30', '31-50', '51+']
df['age_group'] = pd.cut(df['age'], bins=age_bins, labels=age_labels, include_lowest=True)

# ✅ ИСПРАВЛЕНИЕ: вычисляем группы
age_groups = age_labels
group_data = df.groupby('age_group')['is_healthy'].agg(['count', 'sum']).round(0).astype(int)
healthy_counts = group_data['sum'].values
sick_counts = group_data['count'] - group_data['sum']
print("Возрастные группы:", dict(zip(age_groups, zip(healthy_counts, sick_counts))))

# 4 графика в одном figsize
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Гистограмма возраста
axes[0,0].hist(df['age'], bins=15, color='steelblue', alpha=0.7, edgecolor='black')
axes[0,0].set_title('Распределение возраста')
axes[0,0].grid(True, alpha=0.3)

# 2. Пол (упрощённо)
sex_counts = df['sex'].value_counts()
axes[0,1].pie(sex_counts.values, labels=sex_counts.index, autopct='%1.1f%%', colors=['#3498db', '#e74c3c'])
axes[0,1].set_title('Доля по полу')

# 3. Stacked bar по возрастным группам
x = np.arange(len(age_groups))
axes[1,0].bar(x, healthy_counts, label='Здоровые', color='green', alpha=0.8)
axes[1,0].bar(x, sick_counts, bottom=healthy_counts, label='Больные', color='red', alpha=0.8)
axes[1,0].set_xticks(x)
axes[1,0].set_xticklabels(age_groups)
axes[1,0].set_title('По возрастным группам')
axes[1,0].legend()

# 4. Доля здоровых (%)
percentages = [h/(h+s)*100 for h,s in zip(healthy_counts, sick_counts)]
axes[1,1].bar(age_groups, percentages, color='orange', alpha=0.8)
axes[1,1].set_ylabel('Доля здоровых, %')
axes[1,1].set_title('Доля здоровых по группам')
for i, pct in enumerate(percentages):
    axes[1,1].text(i, pct+1, f'{pct:.1f}%', ha='center')

plt.suptitle('Демографический анализ', fontsize=16)
plt.tight_layout()
plt.savefig('results/figures/demographic_analysis.png', dpi=300, bbox_inches='tight')
plt.show()
# Диагнозы (top-5)
top_diag = df['specific.disorder'].value_counts().head(5)
plt.figure(figsize=(10,6))
sns.barplot(x=top_diag.values, y=top_diag.index, palette='viridis')
plt.title('Топ-5 диагнозов')
plt.savefig('results/figures/top_diagnoses.png', dpi=300)
plt.show()