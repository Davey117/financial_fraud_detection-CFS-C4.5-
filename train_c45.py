import os
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def train_and_evaluate_c45(processed_dir, selected_features):
    print("=" * 60)
    print("[PHASE 4 & 5] TRAINING C4.5 DECISION TREE & PERFORMANCE EVALUATION")
    print("=" * 60)
    
    # Storage lists for fold-level metrics
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []
    auc_rocs = []
    
    # We will hold onto one tree instance to inspect the generated compliance rules
    sample_tree = None
    
    for fold in range(1, 11):
        # Load the split data partitions
        train_df = pd.read_csv(os.path.join(processed_dir, f'fold_{fold}_train.csv'))
        test_df = pd.read_csv(os.path.join(processed_dir, f'fold_{fold}_test.csv'))
        
        # Filter features to the CFS optimized subset
        X_train = train_df[selected_features]
        y_train = train_df['Class']
        X_test = test_df[selected_features]
        y_test = test_df['Class']
        
        # Instantiate tree mirroring standard C4.5 (J48) parameter constraints
        # min_samples_leaf=2, criterion='entropy' (Shannon Information Gain Ratio approximation)
        clf = DecisionTreeClassifier(
            criterion='entropy',
            min_samples_leaf=2,
            ccp_alpha=0.001, # Enables cost-complexity pruning to mirror C4.5 reduced-error pruning
            random_state=42
        )
        
        # Train model
        clf.fit(X_train, y_train)
        sample_tree = clf
        
        # Predict on un-leaked validation fold
        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)[:, 1]
        
        # Compute metrics
        accuracies.append(accuracy_score(y_test, y_pred))
        precisions.append(precision_score(y_test, y_pred, zero_division=0))
        recalls.append(recall_score(y_test, y_pred))
        f1_scores.append(f1_score(y_test, y_pred, zero_division=0))
        auc_rocs.append(roc_auc_score(y_test, y_proba))
        
        print(f"Fold {fold:2d} -> Acc: {accuracies[-1]:.4f} | Prec: {precisions[-1]:.4f} | Recall: {recalls[-1]:.4f} | F1: {f1_scores[-1]:.4f}")

    print("\n" + "="*50)
    print("FINAL 10-FOLD AGGREGATE STATISTICAL METRICS (MEAN +/- SD)")
    print("="*50)
    print(f"Accuracy    : {np.mean(accuracies)*100:.2f}% +/- {np.std(accuracies)*100:.2f}%")
    print(f"Precision   : {np.mean(precisions):.4f} +/- {np.std(precisions):.4f}")
    print(f"Recall (Sens): {np.mean(recalls):.4f} +/- {np.std(recalls):.4f}")
    print(f"F1-Score    : {np.mean(f1_scores):.4f} +/- {np.std(f1_scores):.4f}")
    print(f"AUC-ROC     : {np.mean(auc_rocs):.4f} +/- {np.std(auc_rocs):.4f}")
    print("=" * 50)
    
    # Export human-readable decision rules for the audit/compliance analysis
    print("\n[EXCERPT] HUMAN-READABLE COMPLIANCE AUDIT RULES FROM INDUCTION TREE:")
    tree_rules = export_text(sample_tree, feature_names=selected_features, max_depth=3)
    print(tree_rules)

if __name__ == "__main__":
    cfs_features = ['Time', 'V6', 'V8', 'V13', 'V14', 'V18', 'V19', 'V20', 'V21', 'V24', 'V27', 'V28', 'V25', 'V26']
    train_and_evaluate_c45("data/processed_folds", cfs_features)