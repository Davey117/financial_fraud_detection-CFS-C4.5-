import os
import pandas as pd
import numpy as np

def compute_cfs_for_folds(processed_dir):
    print("=" * 60)
    print("[PHASE 3] EXECUTING CORRELATION-BASED FEATURE SELECTION")
    print("=" * 60)
    
    # We will track which features get picked most across all 10 folds
    feature_vote_counts = {}
    
    for fold in range(1, 11):
        train_path = os.path.join(processed_dir, f'fold_{fold}_train.csv')
        df = pd.read_csv(train_path)
        
        X = df.drop('Class', axis=1)
        y = df['Class']
        
        # Calculate absolute Pearson correlation between features and target class
        correlations_with_target = X.corrwith(y).abs()
        
        # Filter out features with extremely low correlation to the target class first
        # This acts as our forward search baseline helper
        base_features = correlations_with_target[correlations_with_target > 0.05].index.tolist()
        
        # Calculate inter-feature correlation matrix to remove redundancy
        corr_matrix = X[base_features].corr().abs()
        
        # Heuristic Elimination: If two features are highly correlated with each other (>0.7), 
        # drop the one that has a lower correlation with the target class.
        selected_features = list(base_features)
        for i in range(len(base_features)):
            for j in range(i + 1, len(base_features)):
                f1, f2 = base_features[i], base_features[j]
                if f1 in selected_features and f2 in selected_features:
                    if corr_matrix.loc[f1, f2] > 0.7:
                        # Drop the one weaker against target class
                        if correlations_with_target[f1] > correlations_with_target[f2]:
                            selected_features.remove(f2)
                        else:
                            selected_features.remove(f1)
                            
        print(f"--> Fold {fold} Optimized Feature Count: {len(selected_features)} features selected.")
        
        # Count votes for our final research report analysis
        for feat in selected_features:
            feature_vote_counts[feat] = feature_vote_counts.get(feat, 0) + 1
            
    print("\n" + "="*40)
    print("FINAL FEATURE SELECTION STABILITY STACK:")
    print("="*40)
    # Sort features by how consistently they were selected across all 10 folds
    sorted_votes = sorted(feature_vote_counts.items(), key=lambda x: x[1], reverse=True)
    for feat, votes in sorted_votes:
        print(f"Feature {feat}: Selected in {votes}/10 Folds")
        
    return [feat for feat, votes in sorted_votes if votes >= 8]

if __name__ == "__main__":
    stable_features = compute_cfs_for_folds("data/processed_folds")
    print(f"\n[RECOMMENDED SYSTEM VARIABLES FOR C4.5]: {stable_features}")