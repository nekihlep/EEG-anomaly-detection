## 🧠 EEG-based Psychiatric Screening Pipeline

**EEG-anomaly-detection** is a hybrid machine learning pipeline for automated screening of psychiatric disorders. The system combines unsupervised representation learning (Autoencoders) to extract latent features and ensemble learning (Random Forest) for high-accuracy pathology classification.

**Core Concept:**  
*"Establish the neurological norm, compress high-dimensional brain activity into clinical signatures, and detect pathologies with world-class accuracy."*

---

## ✨ Key Features

- **Hybrid Architecture:** Integration of Unsupervised Anomaly Detection (Autoencoders) and Supervised Classification (Random Forest)  
- **Representation Learning:** Extraction of 32 nonlinear features from the bottleneck layer of a neural network (TensorFlow/Keras)  
- **Biomarker Discovery:** Identification of clinically relevant rhythms (Theta, Alpha) and brain regions (F3, Fz) using L1 regularization  
- **High Performance:** Achieved F1-score of **0.934** on the test set  
- **Advanced Visualization:** Interactive heatmaps of spectral deviations for 9+ diagnoses  

---

## 📂 Data & Problem Setting

- **Dataset:** EEG Psychiatric Disorders Dataset (Kaggle)  
- **Input:** 114 spectral features (absolute power across 6 frequency bands: Delta–Gamma)  
- **Empirical Norm:** Constructed from healthy individuals aged 18–30 to detect pathological deviations  
- **Classes:** Healthy Controls vs. Patients (Depression, Schizophrenia, AUD, OCD, etc.)  

---

## 🧮 Machine Learning Pipeline (10 Stages)

The project represents a modular research workflow that transforms raw EEG data into interpretable clinical biomarkers:

1. **Exploratory Data Analysis (01_eda.py)**  
   Analysis of demographics and diagnosis distribution. Identified dominance of depression and schizophrenia cases  

2. **Preprocessing (02_preprocessing.py)**  
   Data cleaning and stratified 80/20 split  
   StandardScaler used for neural network preparation  

3. **Baseline Anomaly Detection (03_isolation_forest.py)**  
   Isolation Forest as baseline anomaly detector (**F1 = 0.34**)  

4. **Neural Anomaly Detection (04_autoencoder_anomaly.py)**  
   Empirical norm built using MLP autoencoder trained on healthy subjects  
   Detection based on reconstruction error  

5. **Representation Learning (05_autoencoder_features.py)**  
   Compression of 114 spectral features into 32 nonlinear latent features via bottleneck layer  

6. **Biomarker Discovery (06_feature_importance.py)**  
   Feature importance analysis revealed dominance of:  
   - Theta rhythm (importance: 3.5)  
   - Frontal electrodes F3 (13.8) and Fz (12.8)  

7. **Logistic L1 Classification (07_logistic_l1.py)**  
   Lasso-regularized classification for feature selection  
   Achieved **F1 = 0.89** for patients  

8. **Ensemble Classification (08_random_forest.py)**  
   Random Forest trained on TOP-20 features using GridSearchCV  
   Best result:  
   - **F1-score = 0.934 (patients)**  
   - **0.862 (weighted)**  

9. **Baseline Comparison (09_baseline_comparison.py)**  
   Comparative evaluation of all approaches  
   Confirmed superiority of Random Forest over classical DL and anomaly detection  

10. **Model Evaluation (10_model_evaluation.py)**  
    Final evaluation including ROC/PR curves and normalized confusion matrices  

---

## 📊 Key Insights & Results

- **SOTA Accuracy:** Random Forest significantly outperforms baseline models  
  (Isolation Forest F1 = 0.34, Dummy F1 = 0.78)  

- **Theta Dominance:**  
  Most informative features are Theta rhythms in frontal regions (F3, Fz), consistent with clinical neuroscience findings  

- **Model Stability:**  
  Learning curves show strong convergence with a Train–Validation gap of only **0.17**, indicating excellent generalization  

---

## 🛠 Tech Stack

- **Language:** Python 3.12+  
- **Deep Learning:** Keras (Autoencoders)  
- **Machine Learning:** Scikit-Learn (Random Forest, Logistic L1, Isolation Forest)  
- **Analysis:** Pandas, NumPy, SciPy  
- **Visualization:** Matplotlib, Seaborn (custom heatmaps)  

---

## ⚡️ Quick Start

```bash
# Clone repository
git clone https://github.com/nekihlep/EEG-anomaly-detection.git

# Install dependencies
pip install tensorflow-cpu scikit-learn pandas seaborn

# Run the complete pipeline
python main.py
```
Note: It is recommended to run the project in PyCharm for correct handling of data paths.

## 🚀 Future Directions

- **Connectivity Analysis:**
Incorporating coherence (COH) features to model interactions between brain regions
-**Graph Neural Networks (GNN):**
Transition to graph-based models where EEG electrodes are nodes (relevant for EEML 2026)
-**Real-time Screening:**
Development of a lightweight classifier for real-time clinical deployment
