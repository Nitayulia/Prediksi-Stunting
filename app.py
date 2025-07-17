# app.py – Aplikasi Prediksi Stunting + Riwayat
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sqlite3
import io
import matplotlib.pyplot as plt
from datetime import datetime

# === LOAD MODEL & SCALER =============================
model = joblib.load("adaboost_model.pkl")
scaler = joblib.load("standarscaler_stunting.pkl")
who_df = pd.read_csv("WHO_PBU_TBU_RESMI.csv")


# === DEFINISI FUNGSI Z-SCORE =========================
def hitung_zscore(jk, usia, tinggi):
    try:
        row = who_df[(who_df['Jenis Kelamin'] == jk) & (who_df['Usia (bulan)'] == usia)]
        if row.empty or pd.isna(tinggi):
            return np.nan
        median = row.iloc[0]['Median']
        sd = row.iloc[0]['SD']
        return (tinggi - median) / sd
    except:
        return np.nan


# === FUNGSI SIMPAN RIWAYAT ==========================
def simpan_ke_history(jk, usia, tinggi, z, faktor_risiko, status_prediksi, status_akhir):
    conn = sqlite3.connect('history_stunting.db')
    c = conn.cursor()
    c.execute('''
              CREATE TABLE IF NOT EXISTS history
              (
                  id
                  INTEGER
                  PRIMARY
                  KEY
                  AUTOINCREMENT,
                  timestamp
                  TEXT,
                  jenis_kelamin
                  TEXT,
                  usia_bulan
                  INTEGER,
                  tinggi_anak
                  REAL,
                  z_score
                  REAL,
                  faktor_risiko
                  TEXT,
                  prediksi_status
                  TEXT,
                  status_akhir
                  TEXT
              )
              ''')
    c.execute('''
              INSERT INTO history (timestamp, jenis_kelamin, usia_bulan, tinggi_anak, z_score, faktor_risiko,
                                   prediksi_status, status_akhir)
              VALUES (?, ?, ?, ?, ?, ?, ?, ?)
              ''', (
                  datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                  jk, usia, tinggi, z, faktor_risiko, status_prediksi, status_akhir
              ))
    conn.commit()
    conn.close()


# === PREDIKSI & RISIKO ===============================
def prediksi_dan_risiko(data):
    data['Z-score WHO'] = data.apply(
        lambda row: hitung_zscore(row['Jenis Kelamin'], row['Usia (bulan)'], row['Tinggi Anak']), axis=1)
    if data['Jenis Kelamin'].dtype == object:
        data['Jenis Kelamin'] = data['Jenis Kelamin'].map({'L': 1, 'P': 0})
    data['Z-score WHO'] = data['Z-score WHO'].fillna(-2.0)

    fitur_model = ['Jenis Kelamin', 'Usia (bulan)', 'Tinggi Anak', 'JKN', 'AIR BERSIH', 'JAMBAN',
                   'MEROKOK(KELUARGA)', 'PENY PENYERTA', 'KEK SAAT KEHAMILAN', 'Z-score WHO']
    X = scaler.transform(data[fitur_model])
    prediksi = model.predict(X)
    data['Prediksi'] = np.where(prediksi == 1, 'Stunting', 'Normal')

    def hitung_risiko(row):
        risiko = 0
        if row['JKN'] == 0: risiko += 1
        if row['AIR BERSIH'] == 0: risiko += 1
        if row['JAMBAN'] == 0: risiko += 1
        if row['MEROKOK(KELUARGA)'] == 1: risiko += 1
        if row['PENY PENYERTA'] == 1: risiko += 1
        if row['KEK SAAT KEHAMILAN'] == 1: risiko += 1
        return risiko

    data['Risiko Tambahan'] = data.apply(hitung_risiko, axis=1)
    data['Status Akhir'] = data.apply(lambda row:
                                      'Stunting Risiko Tinggi' if row['Prediksi'] == 'Stunting' and row[
                                          'Risiko Tambahan'] >= 3 else
                                      'Berisiko Stunting' if row['Prediksi'] == 'Normal' and row[
                                          'Risiko Tambahan'] >= 2 else
                                      row['Prediksi'], axis=1)
    return data


# === PENJELASAN STATUS ===============================
def tampilkan_penjelasan(row):
    status = row['Status Akhir']
    risiko = row['Risiko Tambahan']
    if status == "Normal":
        if risiko == 0:
            st.info("🟢 Anak dalam kondisi normal tanpa faktor risiko tambahan.")
        else:
            st.info(f"🟡 Anak normal dengan {risiko} faktor risiko tambahan.")
    elif status == "Berisiko Stunting":
        st.warning(f"🟠 Anak normal tapi berisiko (faktor risiko: {risiko}). Perlu intervensi.")
    elif status == "Stunting Risiko Tinggi":
        st.error("🔴 Anak stunting dan berisiko tinggi. Butuh penanganan serius.")
    elif status == "Stunting":
        st.warning("🔴 Anak mengalami stunting berdasarkan z-score WHO.")


# === UI STREAMLIT ====================================
st.set_page_config(page_title="Prediksi Stunting Anak", layout="wide")
st.title("📊 Prediksi Status Stunting pada Anak")

menu = st.sidebar.radio("Pilih Menu", ["Input Manual", "Upload Excel", "Riwayat Data"])
hasil = None

# === INPUT MANUAL ====================================
if menu == "Input Manual":
    st.subheader("📝 Input Data Anak")
    with st.form("manual_form"):
        jk = st.selectbox("Jenis Kelamin", options=["L", "P"])
        usia = st.number_input("Usia (bulan)", 0, 60, step=1)
        tinggi = st.number_input("Tinggi Anak (cm)", 30.0, 130.0, step=0.1)
        jkn = st.selectbox("JKN", [1, 0])
        air = st.selectbox("Air Bersih", [1, 0])
        jamban = st.selectbox("Jamban", [1, 0])
        rokok = st.selectbox("Merokok (Keluarga)", [1, 0])
        penyerta = st.selectbox("Penyakit Penyerta", [1, 0])
        kek = st.selectbox("KEK saat Kehamilan", [1, 0])
        submitted = st.form_submit_button("Prediksi")

    if submitted:
        df_input = pd.DataFrame([{
            'Jenis Kelamin': jk,
            'Usia (bulan)': usia,
            'Tinggi Anak': tinggi,
            'JKN': jkn,
            'AIR BERSIH': air,
            'JAMBAN': jamban,
            'MEROKOK(KELUARGA)': rokok,
            'PENY PENYERTA': penyerta,
            'KEK SAAT KEHAMILAN': kek
        }])
        hasil = prediksi_dan_risiko(df_input)
        st.success("✅ Hasil Prediksi")
        st.dataframe(hasil)
        tampilkan_penjelasan(hasil.iloc[0])
        baris = hasil.iloc[0]
        faktor_risiko = f"{jkn},{air},{jamban},{rokok},{penyerta},{kek}"
        simpan_ke_history(jk, usia, tinggi, baris['Z-score WHO'], faktor_risiko, baris['Prediksi'],
                          baris['Status Akhir'])

# === UPLOAD EXCEL ====================================
elif menu == "Upload Excel":
    st.subheader("📤 Upload Data Excel")
    file = st.file_uploader("Upload file Excel", type=["xlsx"])
    if file:
        df = pd.read_excel(file)
        st.write("📋 Data yang Diupload:")
        st.dataframe(df)

        nan_rows = df[df.isnull().any(axis=1)]
        valid_rows = df.dropna()

        st.subheader("🧪 Pemeriksaan Nilai Kosong")
        st.write(f"🔍 Baris tidak valid: {len(nan_rows)}")
        if not nan_rows.empty:
            st.warning("Baris dengan nilai kosong:")
            st.dataframe(nan_rows.style.applymap(lambda v: 'background-color: #ffcccc' if pd.isnull(v) else ''))

        if not valid_rows.empty:
            hasil = prediksi_dan_risiko(valid_rows.copy())
            st.success("✅ Hasil Prediksi")
            st.dataframe(hasil)
            for _, row in hasil.iterrows():
                faktor = f"{row['JKN']},{row['AIR BERSIH']},{row['JAMBAN']},{row['MEROKOK(KELUARGA)']},{row['PENY PENYERTA']},{row['KEK SAAT KEHAMILAN']}"
                simpan_ke_history(row['Jenis Kelamin'], row['Usia (bulan)'], row['Tinggi Anak'], row['Z-score WHO'],
                                  faktor, row['Prediksi'], row['Status Akhir'])

            excel_output = io.BytesIO()
            hasil.to_excel(excel_output, index=False, engine='openpyxl')
            st.download_button(
                label="📥 Download Hasil Excel",
                data=excel_output.getvalue(),
                file_name="hasil_prediksi_stunting.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

        else:
            st.error("🚫 Tidak ada data yang bisa diproses.")

# === RIWAYAT =========================================
elif menu == "Riwayat Data":
    st.title("📂 Riwayat Data Input Prediksi")
    conn = sqlite3.connect('history_stunting.db')
    df = pd.read_sql_query("SELECT * FROM history", conn)
    conn.close()

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    t_awal = st.date_input("Dari Tanggal", value=pd.to_datetime("2025-01-01"))
    t_akhir = st.date_input("Sampai Tanggal", value=pd.to_datetime("today"))
    df_filtered = df[(df['timestamp'].dt.date >= t_awal) & (df['timestamp'].dt.date <= t_akhir)]

    st.dataframe(df_filtered)

    id_hapus = st.text_input("Masukkan ID untuk dihapus")
    if st.button("Hapus Data"):
        if id_hapus:
            conn = sqlite3.connect('history_stunting.db')
            c = conn.cursor()
            c.execute("DELETE FROM history WHERE id = ?", (id_hapus,))
            conn.commit()
            conn.close()
            st.success("✅ Data berhasil dihapus")
            st.rerun()

    if st.button("Hapus Semua Data dalam Rentang"):
        conn = sqlite3.connect('history_stunting.db')
        c = conn.cursor()
        c.execute("DELETE FROM history WHERE timestamp BETWEEN ? AND ?", (
            t_awal.strftime("%Y-%m-%d 00:00:00"),
            t_akhir.strftime("%Y-%m-%d 23:59:59")
        ))
        conn.commit()
        conn.close()
        st.success("✅ Data dalam rentang berhasil dihapus")
        st.rerun()

    if st.button("📥 Unduh Excel"):
        excel_io = io.BytesIO()
        df_filtered.to_excel(excel_io, index=False)
        st.download_button("Klik untuk Unduh", data=excel_io.getvalue(), file_name="riwayat_stunting.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# === VISUALISASI =====================================
if hasil is not None and not hasil.empty:
    st.subheader("📊 Distribusi Status Akhir")
    status_counts = hasil['Status Akhir'].value_counts().sort_index()
    fig, ax = plt.subplots()
    ax.pie(status_counts, labels=status_counts.index, autopct='%1.1f%%', startangle=90)
    ax.set_title("Distribusi Status Akhir Prediksi")
    st.pyplot(fig)
