# Financial Fraud Detection Support System (CFS + C4.5)

This repository hosts the code, pipeline, and interactive decision support system dashboard for a B.Sc. Research Project demonstrating optimized Credit Card Fraud Detection using **Correlation-Based Feature Selection (CFS)** and a **C4.5 Decision Tree classifier**.

The interactive web dashboard is deployed at: **[https://financialfraud69.streamlit.app/](https://financialfraud69.streamlit.app/)**

---

## Project Overview

In real-world financial systems, raw transaction datasets are highly dimensional and heavily imbalanced (e.g., standard ULB benchmark with only 0.17% fraud instances). This project implements an optimized machine learning pipeline that:
1. **Reduces Dimensionality**: Applies Correlation-Based Feature Selection (CFS) to select stable features across stratified folds, reducing input dimensions by **53.3%** (from 30 down to 14 features).
2. **Handles Imbalance**: Employs SMOTE (Synthetic Minority Over-sampling Technique) strictly on the training partitions to balance class distributions.
3. **Optimizes Inference**: Trains a C4.5 equivalent Decision Tree model to extract human-readable auditing rules, ensuring fast, compliance-friendly decision-making.

### Performance Highlights (10-Fold Cross-Validation)
- **Accuracy**: 96.44%
- **Recall (Sensitivity)**: 86.98%
- **AUC-ROC**: 95.47%
- **Precision**: 4.20% (low precision is expected due to the severe 578:1 transaction imbalance; our high recall prioritized intercepting actual fraud cases).

---

## File Structure

- `requirements.txt`: Python package dependencies.
- `retrain.py`: Unified pipeline orchestrator. Runs fold preprocessing, CFS feature selection, fold evaluation, model serialization, and metrics generation.
- `app.py`: Streamlit dashboard code. Loads the serialized model and runs live, transient, zero-storage transaction risk assessments.
- `verify_dataset.py`: Initial data audit script that validates dataset dimensions and target class distributions.
- `run_pipeline.py`: Code for Phase 2: Stratified 10-Fold generation with SMOTE and MinMaxScaler.
- `run_cfs.py`: Code for Phase 3: Correlation-based Feature Selection.
- `train_c45.py`: Code for Phase 4 & 5: C4.5 Decision Tree training and performance evaluation.
- `save_model.py`: Code for Phase 6: Final model training and serialization.
- `output/`: Folder containing generated evaluation metrics CSVs, training params logs, and the visual ROC curve plot.
- `output/models/c45_fraud_model.pkl`: Serialized production model bundle containing the classifier, scaler, and features list.

---

## Installation & Local Execution

### 1. Clone the repository
```bash
git clone https://github.com/Davey117/financial_fraud_detection-CFS-C4.5-
cd financial_fraud_detection-CFS-C4.5-
```

### 2. Set up raw dataset
Ensure the Kaggle ULB Credit Card Fraud dataset is downloaded, and place the `creditcard.csv` file inside the `data/` subdirectory:
```
data/creditcard.csv
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Execute retraining pipeline (Optional)
To regenerate files in the `output/` directory, compute metrics, and recreate the model binary, run:
```bash
python retrain.py
```

### 5. Run the Streamlit web dashboard locally
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser.

---

## Features of the Dashboard

- **Real-Time Transaction Monitoring**: Use sidebar inputs to stage transaction risk parameters and evaluate fraud probabilities instantly.
- **Preconfigured Templates**: Quick-click buttons to load either a typical **Legitimate** transaction profile or a high-risk **Fraudulent** transaction profile (engineered to trigger decision tree splits).
- **Interactive Evaluation Comparison**: Visualized side-by-side grouped bar chart comparing the C4.5 Baseline model (30 features) vs the Proposed CFS + C4.5 model (14 features).
- **ROC Curve Display**: Showcases the Receiver Operating Characteristic curve representing the validation out-of-fold predictions.
- **Compliance Rules**: Provides direct inspection of the first three tiers of the canonical decision tree rules for compliance audit transparency.
