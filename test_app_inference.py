import os
import joblib
import pandas as pd
import numpy as np
import tensorflow as tf
from pytorch_tabnet.tab_model import TabNetClassifier

print("--- Testing App Inference Pipeline with 3 Core Models (DNN, Wide & Deep, TabNet) ---")

def engineer_features(df):
    df = df.copy()
    services = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    for s in services:
        if s not in df.columns:
            df[s] = 'No'
            
    df['TotalServices'] = (df[services] == 'Yes').sum(axis=1)
    df['SecurityServices'] = (df[['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport']] == 'Yes').sum(axis=1)
    df['StreamingServices'] = (df[['StreamingTV', 'StreamingMovies']] == 'Yes').sum(axis=1)
    df['NoSecurityCount'] = (df[['OnlineSecurity', 'TechSupport']] == 'No').sum(axis=1)
    
    if 'tenure' in df.columns:
        df['TenureYears'] = df['tenure'] / 12.0
        df['TenureSq'] = (df['tenure'] / 72.0) ** 2
        df['IsNewCustomer'] = (df['tenure'] <= 12).astype(int)
        df['IsLongTerm'] = (df['tenure'] >= 48).astype(int)
    
    if 'MonthlyCharges' in df.columns and 'TotalCharges' in df.columns:
        df['MonthlyToTotalRatio'] = df['MonthlyCharges'] / (df['TotalCharges'] + 1.0)
        df['ExpectedTotalCharges'] = df['tenure'] * df['MonthlyCharges']
        df['ChargeDiff'] = df['TotalCharges'] - df['ExpectedTotalCharges']
        df['AvgCostPerService'] = df['MonthlyCharges'] / (df['TotalServices'] + 1.0)
        df['LogTotalCharges'] = np.log1p(df['TotalCharges'])
        df['LogMonthlyCharges'] = np.log1p(df['MonthlyCharges'])
    
    if 'Contract' in df.columns and 'PaymentMethod' in df.columns:
        df['Contract_Payment'] = df['Contract'].astype(str) + "_" + df['PaymentMethod'].astype(str)
    if 'InternetService' in df.columns and 'TechSupport' in df.columns:
        df['Internet_TechSupport'] = df['InternetService'].astype(str) + "_" + df['TechSupport'].astype(str)
    if 'Contract' in df.columns and 'InternetService' in df.columns:
        df['Contract_Internet'] = df['Contract'].astype(str) + "_" + df['InternetService'].astype(str)

    if 'Contract' in df.columns:
        df['IsMonthToMonth'] = (df['Contract'] == 'Month-to-month').astype(int)
    if 'PaymentMethod' in df.columns:
        df['IsElectronicCheck'] = (df['PaymentMethod'] == 'Electronic check').astype(int)
    if 'IsMonthToMonth' in df.columns and 'IsElectronicCheck' in df.columns:
        df['HighRiskCombo'] = (df['IsMonthToMonth'] & df['IsElectronicCheck']).astype(int)

    return df

feature_cols = joblib.load("saved_models/feature_columns.pkl")
scaler = joblib.load("saved_models/scaler.pkl")

dnn_model = tf.keras.models.load_model("saved_models/dnn_model.keras")
wd_model = tf.keras.models.load_model("saved_models/wide_deep_model.keras")

tabnet_model = TabNetClassifier()
tabnet_model.load_model("saved_models/tabnet_model.zip")

sample_cust = {
    'gender': 'Male',
    'SeniorCitizen': 0,
    'Partner': 'Yes',
    'Dependents': 'No',
    'tenure': 8,
    'PhoneService': 'Yes',
    'MultipleLines': 'No',
    'InternetService': 'Fiber optic',
    'OnlineSecurity': 'No',
    'OnlineBackup': 'Yes',
    'DeviceProtection': 'No',
    'TechSupport': 'No',
    'StreamingTV': 'Yes',
    'StreamingMovies': 'Yes',
    'Contract': 'Month-to-month',
    'PaperlessBilling': 'Yes',
    'PaymentMethod': 'Electronic check',
    'MonthlyCharges': 85.50,
    'TotalCharges': 684.00
}

cust_df = pd.DataFrame([sample_cust])
cust_eng = engineer_features(cust_df)
cust_encoded = pd.get_dummies(cust_eng, drop_first=True)
cust_encoded = cust_encoded.reindex(columns=feature_cols, fill_value=0)

print(f"Encoded feature shape: {cust_encoded.shape}")
assert cust_encoded.shape[1] == len(feature_cols), "Feature column count mismatch!"

cust_scaled = scaler.transform(cust_encoded)
cust_tab = cust_encoded.astype(int).values

dnn_prob = float(dnn_model.predict(cust_scaled, verbose=0)[0][0])
wd_prob = float(wd_model.predict(cust_scaled, verbose=0)[0][0])
tabnet_prob = float(tabnet_model.predict_proba(cust_tab)[0][1])

ens_prob = float(0.50 * dnn_prob + 0.50 * wd_prob)

print(f"DNN Prob:      {dnn_prob*100:.2f}%")
print(f"Wide & Deep:   {wd_prob*100:.2f}%")
print(f"TabNet Prob:   {tabnet_prob*100:.2f}%")
print(f"Ensemble Prob: {ens_prob*100:.2f}%")

assert 0.0 <= dnn_prob <= 1.0
assert 0.0 <= wd_prob <= 1.0
assert 0.0 <= tabnet_prob <= 1.0

print("[SUCCESS] Full inference pipeline test with 3 Core Models PASSED!")
