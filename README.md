EEG Anomaly Detection is an machine learning pipeline designed to identify abnormal brain activity patterns in EEG (electroencephalogram) signals. The system learns from healthy patients, then detects deviations in spectral power across 6 frequency bands, enabling binary classification of patients as "healthy" or "anomalous". Post-detection statistical analysis reveals disorder-specific spectral signatures for 9+ mental health conditions.

Core Concept: "If it doesn't look like a healthy brain, flag it — then figure out how it's different."

## ✨ Key Features
- Automated pipeline for loading and preprocessing a real clinical EEG dataset  
- Unsupervised anomaly detection using **autoencoder** and **Isolation Forest**  
- Empirical EEG **norm** built from healthy participants (18–30 years)  
- Relative power analysis across six standard frequency bands  
- Visualization of diagnostic **spectral patterns** (heatmaps / plots)

## 📂 Data & Problem Setting

[Dataset: open **Kaggle** dataset “EEG Psychiatric Disorders Dataset”] (https://www.kaggle.com/datasets/shashwatwork/eeg-psychiatric-disorders-dataset)
EEG features:
  - Absolute power (**AB**) in six bands: delta, theta, alpha, beta, high beta, gamma  
  - 114 numerical features per subject (AB × electrodes / regions)  
Task:
  - Binary: **healthy vs pathology** (anomaly detection, training only on healthy)  
  - Descriptive: comparison of **relative power** patterns across diagnoses  

> Coherence (COH) features are present in the dataset but not yet used in this version of the project.

## 🧮 Machine Learning Pipeline

### Unsupervised Anomaly Detection

 **Scaling & Norm**
  - Standardization parameters (mean, std) are computed on healthy participants 18–30  
  - The same transformation is applied to all subjects (healthy + patients)  
  - Healthy group defines an empirical EEG **norm**

  **Autoencoder (MLP)**
  - Symmetric feedforward network trained to reconstruct standardized EEG features of healthy controls  
  - Reconstruction error (MSE) used as an **anomaly score**  
  - Threshold: 90‑th percentile of errors in healthy group → binary label  
    `predicted_healthy` / `predicted_pathology`

 **Isolation Forest**
  - Tree‑based anomaly detector trained on the same healthy subset  
  - Used as a baseline model for comparison under strong class imbalance  

## 📊 Spectral Analysis & Visualisation

- Empirical norm is computed for each spectral feature on healthy 18–30 group  
- For each subject and each band, **relative deviation** is computed as percent difference from norm  
- Diagnostic groups (e.g. depression, schizophrenia, alcohol use disorder, OCD, bipolar disorder, etc.) are compared by:
   Mean relative power deviations in delta–gamma bands  
   Heatmaps showing patterns of increases/decreases vs healthy norm  

>This allows exploring candidate EEG biomarkers and shared vs distinct spectral patterns across disorders.

## 🛠 Tech Stack

**Language:** Python  
**Core libraries:**  
  - Data: `pandas`, `numpy`  
  - ML: `scikit-learn` 
  - Visualisation: `matplotlib` / `seaborn` (heatmaps, plots)  

## ⚡ Quick Start

1. Clone repository
git clone https://github.com/nekihlep/EEG-anomaly-detection.git
2. Create environment (optional, recommended)
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
3. Install dependencies
pip install -r requirements.txt
4. Run main file autoenc.py
   
>It's more convenient to do it in PyCharm

## 🚀 Future Directions

- **Data expansion**: COH features, raw EEG, new datasets
- **Clinical application**: Therapy monitoring (pre/post-treatment spectral dynamics)
- **Model improvements**: Larger samples, deep learning, multi-class classification

