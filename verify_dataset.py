import pandas as pd
import os

def verify_raw_data(file_path):
    print("=" * 50)
    print("NIGERIAN ACADEMIC RESEARCH PORTAL: DATA VERIFICATION")
    print("=" * 50)
    
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"[ERROR] File not found at {file_path}. Please check your path.")
        return
    
    # Load dataset memory-efficiently
    print("[1/4] Loading creditcard.csv into memory...")
    df = pd.read_csv(file_path)
    
    # 1. Verify Structural Dimensions
    rows, cols = df.shape
    print(f"[2/4] Structural Dimensions:")
    print(f"      - Total Observed Transactions (Rows): {rows:,}")
    print(f"      - Total Features (Columns): {cols}")
    
    # 2. Check Completeness (Missing Values)
    missing_count = df.isnull().sum().sum()
    print(f"[3/4] Completeness Assessment:")
    print(f"      - Missing/Null Values Found: {missing_count}")
    
    # 3. Class Distribution Profile
    class_counts = df['Class'].value_counts()
    legit_count = class_counts.get(0, 0)
    fraud_count = class_counts.get(1, 0)
    fraud_percentage = (fraud_count / rows) * 100
    
    print(f"[4/4] Target Class Distribution:")
    print(f"      - Legitimate Transactions (Class 0): {legit_count:,}")
    print(f"      - Fraudulent Transactions (Class 1): {fraud_count:,}")
    print(f"      - Fraud Evasion Ratio (Imbalance): {fraud_percentage:.3f}%")
    print("=" * 50)
    
    # Cross-reference with standard benchmark expectations
    if rows == 284807 and cols == 31:
        print("[SUCCESS] Data integrity verified. This matches the standard ULB benchmark.")
    else:
        print("[WARNING] Dataset dimensions deviate from standard publication parameters.")

if __name__ == "__main__":
    # Adjust this path if your file is inside a 'data' subfolder
    verify_raw_data("data/creditcard.csv")