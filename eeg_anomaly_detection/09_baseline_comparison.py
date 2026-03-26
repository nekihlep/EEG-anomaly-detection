from sklearn.dummy import DummyClassifier
from sklearn.metrics import classification_report, confusion_matrix
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')
X_test = np.load('processed_data/X_test_scaled.npy')
y_test = np.load('processed_data/y_test.npy')

dummy = DummyClassifier(strategy='most_frequent')
dummy.fit(X_test, y_test)

y_pred_dummy = dummy.predict(X_test)

print("\nDummyClassifier (most_frequent):")
print(classification_report(y_test, y_pred_dummy, target_names=['Больной','Здоровый']))
# 6. Confusion Matrix
cm = confusion_matrix(y_test, y_pred_dummy)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Больной', 'Здоровый'],
            yticklabels=['Больной', 'Здоровый'])
plt.title('DummyClassifier: Confusion Matrix')
plt.ylabel('True')
plt.xlabel('Predicted')
plt.savefig('results/figures/dummyclass_cm.png', dpi=300)
plt.show()