import os
import joblib
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MinMaxScaler

def train_and_serialize_final_model(csv_path, selected_features, model_output_path):
    print("=" * 60)
    print("SERIAlIZING FINAL COMPLIANCE MODEL FOR STREAMLIT DEPLOYMENT")
    print("=" * 60)
    
    # 1. Load data
    df = pd.read_csv(csv_path)
    X = df[selected_features]
    y = df['Class']
    
    # 2. Apply scaling strictly to the 'Time' variable (Amount wasn't selected by CFS)
    # We save the scaler too so the Streamlit app can normalize inputs exactly the same way
    scaler = MinMaxScaler()
    X = X.copy()
    X[['Time']] = scaler.fit_transform(X[['Time']])
    
    # 3. Apply SMOTE globally for the final model deployment pool
    print("[1/3] Balancing full dataset via SMOTE...")
    smote = SMOTE(k_neighbors=5, random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    
    # 4. Train the production C4.5 equivalent tree
    print("[2/3] Training final C4.5 induction tree...")
    clf = DecisionTreeClassifier(
        criterion='entropy',
        min_samples_leaf=2,
        ccp_alpha=0.001,
        random_state=42
    )
    clf.fit(X_resampled, y_resampled)
    
    # 5. Save artifacts to disk
    print("[3/3] Saving model artifacts...")
    if not os.path.exists('output/models'):
        os.makedirs('output/models')
        
    # We save both the classifier and the scaler as a dictionary bundle
    artifact_bundle = {
        'model': clf,
        'scaler': scaler,
        'features': selected_features
    }
    joblib.dump(artifact_bundle, model_output_path)
    print(f"\n[SUCCESS] Model artifact bundle successfully saved to: {model_output_path}")
    print("=" * 60)

if __name__ == "__main__":
    cfs_features = ['Time', 'V6', 'V8', 'V13', 'V14', 'V18', 'V19', 'V20', 'V21', 'V24', 'V27', 'V28', 'V25', 'V26']
    train_and_serialize_final_model("data/creditcard.csv", cfs_features, "output/models/c45_fraud_model.pkl")