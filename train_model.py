# train_model_adaboost.py

# =============================================================================
# 1. IMPORT LIBRARY ===========================================================
import pandas as pd
import numpy as np
import time
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import AdaBoostClassifier

# =============================================================================
# 2. LOAD DATA LATIH & UJI ====================================================
df_train = pd.read_excel('DSStuntingFIX80.xlsx')
df_test  = pd.read_excel('DSStuntingFIX20.xlsx')

print(f"Data latih : {df_train.shape[0]} baris")
print(f"Data uji   : {df_test.shape[0]} baris")

# =============================================================================
# 3. PISAHKAN FITUR & LABEL ===================================================
# Gunakan Z-score WHO sebagai fitur sah
X_train = df_train.drop(columns=['Status'])
y_train = df_train['Status']

X_test = df_test.drop(columns=['Status'])
y_test = df_test['Status']

# =============================================================================
# 4. STANDARISASI FITUR =======================================================
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std  = scaler.transform(X_test)

# =============================================================================
# 5. TRAINING MODEL ADABOOST ==================================================
ada_model = AdaBoostClassifier(
    n_estimators=200,
    learning_rate=0.1,
    random_state=42
)

start_train = time.perf_counter()
ada_model.fit(X_train_std, y_train)
end_train = time.perf_counter()
training_time = end_train - start_train

# =============================================================================
# 6. PREDIKSI & EVALUASI ======================================================
start_test = time.perf_counter()
y_pred = ada_model.predict(X_test_std)
end_test = time.perf_counter()
testing_time = end_test - start_test

acc = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
conf_mat = confusion_matrix(y_test, y_pred)

print("=== Evaluasi Model AdaBoost ===")
print(f"Akurasi         : {acc:.4f}")
print(f"Training time   : {training_time:.4f} detik")
print(f"Testing time    : {testing_time:.4f} detik")
print("Confusion Matrix:\n", conf_mat)
print(report)

# =============================================================================
# 7. SIMPAN MODEL & HASIL PREDIKSI ============================================
# Simpan model & scaler
joblib.dump(ada_model, 'adaboost_model.pkl')
joblib.dump(scaler, 'standarscaler_stunting.pkl')

# Simpan hasil prediksi ke Excel
df_hasil = df_test.copy()
df_hasil['Prediksi'] = y_pred
df_hasil['Evaluasi'] = np.where(df_hasil['Status'] == df_hasil['Prediksi'], 'Benar', 'Salah')
df_hasil.to_excel('hasil_prediksi_adaboost.xlsx', index=False)

print("\n✅ Model dan hasil prediksi telah disimpan ke file.")
