import os
import gc
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve
)

def run_pipeline(csv_path, output_dir):
    print("=" * 60)
    print("[PHASE 2] STARTING STRATIFIED 10-FOLD GENERATION WITH SMOTE")
    print("=" * 60)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    df = pd.read_csv(csv_path)
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    smote = SMOTE(k_neighbors=5, random_state=42)
    scaler = MinMaxScaler()
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        print(f"--> Processing Fold/Epoch {fold + 1} of 10...")
        
        X_train, X_test = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        X_train[['Amount', 'Time']] = scaler.fit_transform(X_train[['Amount', 'Time']])
        X_test[['Amount', 'Time']] = scaler.transform(X_test[['Amount', 'Time']])
        
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
        
        train_fold_df = pd.concat([X_train_res, y_train_res], axis=1)
        test_fold_df = pd.concat([X_test, y_test], axis=1)
        
        train_fold_df.to_csv(os.path.join(output_dir, f'fold_{fold+1}_train.csv'), index=False)
        test_fold_df.to_csv(os.path.join(output_dir, f'fold_{fold+1}_test.csv'), index=False)
        
        del X_train, X_test, y_train, y_test, X_train_res, y_train_res, train_fold_df, test_fold_df
        gc.collect()
        
    print("\n[SUCCESS] Preprocessing completed! Folds exported to:", output_dir)
    print("=" * 60)

def run_cfs(processed_dir):
    print("=" * 60)
    print("[PHASE 3] EXECUTING CORRELATION-BASED FEATURE SELECTION")
    print("=" * 60)
    
    feature_vote_counts = {}
    
    for fold in range(1, 11):
        train_path = os.path.join(processed_dir, f'fold_{fold}_train.csv')
        df = pd.read_csv(train_path)
        
        X = df.drop('Class', axis=1)
        y = df['Class']
        
        correlations_with_target = X.corrwith(y).abs()
        base_features = correlations_with_target[correlations_with_target > 0.05].index.tolist()
        corr_matrix = X[base_features].corr().abs()
        
        selected_features = list(base_features)
        for i in range(len(base_features)):
            for j in range(i + 1, len(base_features)):
                f1, f2 = base_features[i], base_features[j]
                if f1 in selected_features and f2 in selected_features:
                    if corr_matrix.loc[f1, f2] > 0.7:
                        if correlations_with_target[f1] > correlations_with_target[f2]:
                            selected_features.remove(f2)
                        else:
                            selected_features.remove(f1)
                            
        print(f"--> Fold {fold} Optimized Feature Count: {len(selected_features)} features selected.")
        
        for feat in selected_features:
            feature_vote_counts[feat] = feature_vote_counts.get(feat, 0) + 1
            
    print("\n" + "="*40)
    print("FINAL FEATURE SELECTION STABILITY STACK:")
    print("="*40)
    sorted_votes = sorted(feature_vote_counts.items(), key=lambda x: x[1], reverse=True)
    for feat, votes in sorted_votes:
        print(f"Feature {feat}: Selected in {votes}/10 Folds")
        
    stable_features = [feat for feat, votes in sorted_votes if votes >= 8]
    print(f"\n[RECOMMENDED SYSTEM VARIABLES FOR C4.5]: {stable_features}")
    print("=" * 60)
    return stable_features

def train_and_evaluate(processed_dir, selected_features):
    print("=" * 60)
    print("[PHASE 4 & 5] TRAINING C4.5 DECISION TREE & PERFORMANCE EVALUATION")
    print("=" * 60)
    
    fold_results = []
    all_y_true = []
    all_y_proba = []
    
    for fold in range(1, 11):
        train_df = pd.read_csv(os.path.join(processed_dir, f'fold_{fold}_train.csv'))
        test_df = pd.read_csv(os.path.join(processed_dir, f'fold_{fold}_test.csv'))
        
        X_train = train_df[selected_features]
        y_train = train_df['Class']
        X_test = test_df[selected_features]
        y_test = test_df['Class']
        
        clf = DecisionTreeClassifier(
            criterion='entropy',
            min_samples_leaf=2,
            ccp_alpha=0.001,
            random_state=42
        )
        
        clf.fit(X_train, y_train)
        
        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)[:, 1]
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_proba)
        
        # Save results per epoch (fold)
        fold_results.append({
            'Epoch': fold,
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1_Score': f1,
            'AUC_ROC': auc
        })
        
        all_y_true.extend(y_test.tolist())
        all_y_proba.extend(y_proba.tolist())
        
        print(f"Epoch/Fold {fold:2d} -> Acc: {acc:.4f} | Prec: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")
        
    # 1. Save epoch/fold metrics to CSV
    df_folds = pd.DataFrame(fold_results)
    df_folds.to_csv('output/fold_metrics.csv', index=False)
    print("\n[SAVED] Fold-by-fold (epoch 1-10) results saved to: output/fold_metrics.csv")
    
    # 2. Save aggregate metrics to CSV (Accuracy, F1, Precision, Recall, AUC-ROC)
    agg_metrics = {
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC'],
        'Mean': [
            df_folds['Accuracy'].mean(),
            df_folds['Precision'].mean(),
            df_folds['Recall'].mean(),
            df_folds['F1_Score'].mean(),
            df_folds['AUC_ROC'].mean()
        ],
        'Std': [
            df_folds['Accuracy'].std(),
            df_folds['Precision'].std(),
            df_folds['Recall'].std(),
            df_folds['F1_Score'].std(),
            df_folds['AUC_ROC'].std()
        ]
    }
    df_agg = pd.DataFrame(agg_metrics)
    df_agg.to_csv('output/evaluation_metrics.csv', index=False)
    print("[SAVED] Overall aggregate evaluation metrics saved to: output/evaluation_metrics.csv")
    
    # 3. Calculate and Save ROC Curve Data (FPR, TPR, Thresholds)
    fpr, tpr, thresholds = roc_curve(all_y_true, all_y_proba)
    df_roc = pd.DataFrame({
        'FPR': fpr,
        'TPR': tpr,
        'Threshold': thresholds
    })
    df_roc.to_csv('output/roc_curve_data.csv', index=False)
    print("[SAVED] ROC curve data coordinates saved to: output/roc_curve_data.csv")
    
    # 4. Draw and Save the ROC Curve Plot
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkred', lw=2, label=f'C4.5 Decision Tree (AUC = {roc_auc_score(all_y_true, all_y_proba):.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig('output/roc_curve.png', dpi=300)
    plt.close()
    print("[SAVED] ROC curve visual plot saved to: output/roc_curve.png")
    
    print("=" * 60)

def train_and_serialize_final_model(csv_path, selected_features, model_output_path):
    print("=" * 60)
    print("SERIALIZING FINAL COMPLIANCE MODEL FOR STREAMLIT DEPLOYMENT")
    print("=" * 60)
    
    df = pd.read_csv(csv_path)
    X = df[selected_features]
    y = df['Class']
    
    scaler = MinMaxScaler()
    X = X.copy()
    X[['Time']] = scaler.fit_transform(X[['Time']])
    
    print("[1/3] Balancing full dataset via SMOTE...")
    smote = SMOTE(k_neighbors=5, random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    
    print("[2/3] Training final C4.5 induction tree...")
    clf = DecisionTreeClassifier(
        criterion='entropy',
        min_samples_leaf=2,
        ccp_alpha=0.001,
        random_state=42
    )
    clf.fit(X_resampled, y_resampled)
    
    print("[3/3] Saving model artifacts...")
    if not os.path.exists(os.path.dirname(model_output_path)):
        os.makedirs(os.path.dirname(model_output_path))
        
    artifact_bundle = {
        'model': clf,
        'scaler': scaler,
        'features': selected_features
    }
    joblib.dump(artifact_bundle, model_output_path)
    print(f"\n[SUCCESS] Model artifact bundle successfully saved to: {model_output_path}")
    print("=" * 60)
    
    # Save training params to JSON
    training_params = {
        'model_type': 'DecisionTreeClassifier (C4.5 equivalent)',
        'criterion': 'entropy',
        'min_samples_leaf': 2,
        'ccp_alpha': 0.001,
        'random_state': 42,
        'smote_k_neighbors': 5,
        'scaler': 'MinMaxScaler',
        'scaling_variables': ['Time'],
        'selected_features': selected_features,
        'number_of_folds': 10
    }
    with open('output/training_params.json', 'w') as f:
        json.dump(training_params, f, indent=4)
    print("[SAVED] Training parameters saved to: output/training_params.json")
    print("=" * 60)

if __name__ == "__main__":
    # Create output directories
    if not os.path.exists('output'):
        os.makedirs('output')
        
    raw_data_path = "data/creditcard.csv"
    processed_folds_dir = "data/processed_folds"
    model_output_path = "output/models/c45_fraud_model.pkl"
    
    # 1. Run Fold Preprocessing
    run_pipeline(raw_data_path, processed_folds_dir)
    
    # 2. Run CFS Feature Selection
    stable_features = run_cfs(processed_folds_dir)
    
    # 3. Train & Evaluate Fold-by-Fold, exporting all metric CSVs & ROC plot
    train_and_evaluate(processed_folds_dir, stable_features)
    
    # 4. Train final model & export serialized artifacts & params JSON
    train_and_serialize_final_model(raw_data_path, stable_features, model_output_path)
