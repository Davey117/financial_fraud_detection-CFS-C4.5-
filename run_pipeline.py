import os
import gc
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE

def generate_memory_safe_folds(csv_path, output_dir):
    print("=" * 60)
    print("[PHASE 2] STARTING STRATIFIED 10-FOLD GENERATION WITH SMOTE")
    print("=" * 60)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Read the data efficiently
    df = pd.read_csv(csv_path)
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    # Configure Stratified K-Fold and SMOTE parameters per research scope
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    smote = SMOTE(k_neighbors=5, random_state=42)
    scaler = MinMaxScaler()
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        print(f"--> Processing Fold {fold + 1} of 10...")
        
        # Isolate the fold indices dynamically
        X_train, X_test = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Min-Max Normalise 'Amount' and 'Time' explicitly to prevent leakage
        X_train[['Amount', 'Time']] = scaler.fit_transform(X_train[['Amount', 'Time']])
        X_test[['Amount', 'Time']] = scaler.transform(X_test[['Amount', 'Time']])
        
        # Synthesize minority instances strictly on the training partition
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
        
        # Package processed indices back into single files for WEKA compatibility
        train_fold_df = pd.concat([X_train_res, y_train_res], axis=1)
        test_fold_df = pd.concat([X_test, y_test], axis=1)
        
        # Save to disk
        train_fold_df.to_csv(os.path.join(output_dir, f'fold_{fold+1}_train.csv'), index=False)
        test_fold_df.to_csv(os.path.join(output_dir, f'fold_{fold+1}_test.csv'), index=False)
        
        # Enforce strict memory reclamation for an 8GB RAM threshold
        del X_train, X_test, y_train, y_test, X_train_res, y_train_res, train_fold_df, test_fold_df
        gc.collect()
        
    print("\n[SUCCESS] Preprocessing completed! Folds exported to:", output_dir)
    print("=" * 60)

if __name__ == "__main__":
    generate_memory_safe_folds("data/creditcard.csv", "data/processed_folds")