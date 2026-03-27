import subprocess
steps = [
    "01_eda.py",
    "02_preprocessing.py",
    "03_isolation_forest.py",
    "04_autoencoder_anomaly.py",
    "05_autoencoder_features.py",
    "07_logistic_l1.py",
    "08_random_forest.py",
    "09_baseline_comparison.py",
    "10_model_evaluation.py",
    "11_error_analysis.py",
    "12_spectral_analysis.py"
]

for step in steps:
    print(f"Running {step}...")
    subprocess.run(["python", step])
