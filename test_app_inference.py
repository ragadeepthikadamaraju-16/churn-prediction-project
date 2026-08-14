import os
import joblib
import pandas as pd
import numpy as np
import tensorflow as tf
from pytorch_tabnet.tab_model import TabNetClassifier

print("--- Testing App Inference Pipeline ---")

feature_cols = joblib.load("saved_models/feature_columns.pkl")
scaler = joblib.load("saved_models/scaler.pkl")

dnn_model = tf.keras.models.load_model("saved_models/dnn_model.keras")
wd_model = tf.keras.models.load_model("saved_models/wide_deep_model.keras")

tabnet_model = TabNetClassifier()
tabnet_model.load_model("saved_models/tabnet_model.zip")

# Sample customer with all 19 features
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
cust_encoded = pd.get_dummies(cust_df, drop_first=True)
cust_encoded = cust_encoded.reindex(columns=feature_cols, fill_value=0)

print(f"Encoded feature shape: {cust_encoded.shape}")

cust_scaled = scaler.transform(cust_encoded)
cust_tab = cust_encoded.astype(int).values

dnn_prob = float(dnn_model.predict(cust_scaled, verbose=0)[0][0])
wd_prob = float(wd_model.predict(cust_scaled, verbose=0)[0][0])
tabnet_prob = float(tabnet_model.predict_proba(cust_tab)[0][1])
ens_prob = float(np.mean([dnn_prob, wd_prob, tabnet_prob]))

print(f"DNN Prob: {dnn_prob*100:.2f}%")
print(f"Wide & Deep Prob: {wd_prob*100:.2f}%")
print(f"TabNet Prob: {tabnet_prob*100:.2f}%")
print(f"Ensemble Prob: {ens_prob*100:.2f}%")

assert 0.0 <= dnn_prob <= 1.0, "DNN prob out of range"
assert 0.0 <= wd_prob <= 1.0, "Wide & Deep prob out of range"
assert 0.0 <= tabnet_prob <= 1.0, "TabNet prob out of range"

print("✅ Full 19-feature preprocessing and inference test PASSED!")
