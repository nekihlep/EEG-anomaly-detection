# Screening comparison

Mean ± SD across ten outer folds. SD describes dependent-fold variability, not confidence intervals.
screening_90 thresholds target sensitivity ≥ 0.90 on calibration data; test sensitivity may differ.

| Model | Operating rule | Sensitivity | Specificity | Balanced accuracy | Patient F1 | Macro F1 | Weighted F1 | ROC AUC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Dummy classifier | standard | 1.000 ± 0.000 | 0.000 ± 0.000 | 0.500 ± 0.000 | 0.925 ± 0.002 | 0.462 ± 0.001 | 0.795 ± 0.004 | 0.500 ± 0.000 |
| L1 logistic regression | standard | 0.898 ± 0.057 | 0.178 ± 0.165 | 0.538 ± 0.073 | 0.883 ± 0.028 | 0.528 ± 0.078 | 0.784 ± 0.033 | 0.658 ± 0.057 |
| Random Forest | standard | 0.939 ± 0.032 | 0.125 ± 0.112 | 0.532 ± 0.054 | 0.902 ± 0.017 | 0.528 ± 0.063 | 0.797 ± 0.025 | 0.632 ± 0.058 |
| L1 selection + Random Forest | standard | 0.937 ± 0.044 | 0.119 ± 0.084 | 0.528 ± 0.025 | 0.900 ± 0.018 | 0.522 ± 0.040 | 0.794 ± 0.013 | 0.637 ± 0.052 |
| Autoencoder + Random Forest | standard | 0.952 ± 0.032 | 0.089 ± 0.075 | 0.521 ± 0.027 | 0.906 ± 0.013 | 0.512 ± 0.039 | 0.796 ± 0.010 | 0.620 ± 0.043 |
| Dummy classifier | screening_90 | 1.000 ± 0.000 | 0.000 ± 0.000 | 0.500 ± 0.000 | 0.925 ± 0.002 | 0.462 ± 0.001 | 0.795 ± 0.004 | 0.500 ± 0.000 |
| L1 logistic regression | screening_90 | 0.884 ± 0.043 | 0.203 ± 0.122 | 0.543 ± 0.068 | 0.877 ± 0.028 | 0.543 ± 0.072 | 0.783 ± 0.038 | 0.658 ± 0.057 |
| Random Forest | screening_90 | 0.890 ± 0.050 | 0.191 ± 0.128 | 0.541 ± 0.066 | 0.880 ± 0.029 | 0.540 ± 0.061 | 0.784 ± 0.034 | 0.632 ± 0.058 |
| L1 selection + Random Forest | screening_90 | 0.896 ± 0.049 | 0.208 ± 0.101 | 0.552 ± 0.030 | 0.884 ± 0.021 | 0.549 ± 0.032 | 0.790 ± 0.013 | 0.637 ± 0.052 |
| Autoencoder + Random Forest | screening_90 | 0.892 ± 0.055 | 0.201 ± 0.157 | 0.547 ± 0.062 | 0.882 ± 0.023 | 0.540 ± 0.060 | 0.786 ± 0.022 | 0.620 ± 0.043 |
