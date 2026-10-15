# Screening results

Thresholds targeted sensitivity >= 0.90 on calibration data only. Test metrics below are means across ten outer folds.

| Model | Test sensitivity | Test specificity | Balanced accuracy | ROC AUC |
|---|---:|---:|---:|---:|
| Dummy classifier | 1.000 | 0.000 | 0.500 | 0.500 |
| L1 logistic regression | 0.884 | 0.203 | 0.543 | 0.658 |
| Random Forest | 0.890 | 0.191 | 0.541 | 0.632 |
| L1 selection + Random Forest | 0.896 | 0.208 | 0.552 | 0.637 |
| Autoencoder + Random Forest | 0.892 | 0.201 | 0.547 | 0.620 |

L1 selection + RF has the highest mean screening specificity (0.208) and balanced accuracy (0.552).
The differences among learned models are small and variable; statistical superiority is not established.
Calibration sensitivity met the target in every model/fold; mean external sensitivity is approximately 0.884–0.896.
Learned models exclude approximately 19–21% of healthy controls from positive screening, whereas Dummy excludes none.
These screening rules still miss approximately 10–12% of patients on average and flag most healthy controls.

ROC AUC is identical between operating rules for each fitted model. Differences from the earlier nested experiment
are due to retraining with a smaller fitting partition (calibration data were reserved), not to threshold adjustment.
The same outer test records are retained. The first-paper IF/AE models are not included in this experiment,
so improvements over that paper remain descriptive and not a controlled head-to-head comparison.

Validation passed: calibration-only thresholds, disjoint rows and identical-profile groups, all predictions and 100 metric records.
