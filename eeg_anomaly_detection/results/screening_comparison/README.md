# Sensitivity-oriented screening evaluation

This extends the five-model comparison with calibration-only threshold selection.
The same ten outer test folds are retained. Each outer training partition is
split into approximately 75% fitting and 25% calibration data using grouped
stratification. Overall proportions are approximately 60/20/20, not exact.
Identical spectral profiles stay together in every partition.

Hyperparameters, scaling, feature selection and autoencoder fitting use only
the fitting partition. The calibration partition is never used for those steps.
The RF and L1 search grids and AE architecture remain fixed as in step 13.

For each model, choose the threshold with highest calibration specificity
subject to calibration sensitivity >= 0.90. Ties favor the highest threshold.
Patient prediction means patient_score >= threshold. Apply this threshold
unchanged to the outer test partition. Test sensitivity may be below 0.90.
This target is exploratory and is not a clinical standard.

Both standard predictions and calibrated screening predictions are evaluated
for the same newly fitted model. Previous standard results are preserved in
nested_comparison; the new models use fewer fitting records, so those earlier
values are not directly interchangeable with this experiment.

The original first-paper IF/AE models are not re-evaluated here. Matching the
sensitivity target alone does not make comparisons with that paper controlled.
No external validation is performed. SD across dependent folds is descriptive.

Run: python eeg_anomaly_detection/main.py screening
Individual folds: python eeg_anomaly_detection/15_screening_comparison.py --folds 0,2,4,6,8
Outputs include calibration scores, selected thresholds, test predictions and
English tables. Existing completed folds can be resumed under the same protocol.
