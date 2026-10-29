import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import zscore
import os
import joblib

# ==========================================
# KONFIGURASI HALAMAN & PATH DATASET
# ==========================================
st.set_page_config(
    page_title="Dashboard Stunting Balita",
    page_icon="👶",
    layout="wide"
)

DEFAULT_CSV_NAME = "Growth standard comparison between WHO and Indonesian Growth Chart-Determinants Data.csv"
MODEL_BUNDLE_NAME = "stunting_model_bundle.joblib"

def get_csv_path():
    if '__file__' in globals():
        base_dir = os.path.dirname(os.path.abspath(__file__))
    else:
        base_dir = os.getcwd()
    
    path = os.path.join(base_dir, DEFAULT_CSV_NAME)
    return path if os.path.exists(path) else None

# Mapping
LABEL_MAP = {0: "Sangat Pendek", 1: "Pendek (Stunting)", 2: "Normal"}
STATUS_SINGKAT = {
    0: "Stunting Berat",
    1: "Stunting",
    2: "Normal"
}
RAW_LABEL_MAP = {1: "Sangat Pendek", 2: "Pendek (Stunting)", 3: "Normal"}

# ==========================================
# CSS KUSTOM (UI MODERN & MENARIK)
# ==========================================
st.markdown("""
<style>
    :root {
        --ink: #172b4d;
        --muted: #66758c;
        --line: #e5eaf2;
        --brand: #176b70;
        --brand-dark: #10565b;
        --surface: #ffffff;
        --soft: #f4f8fb;
    }

    .stApp {
        background: linear-gradient(180deg, #f5f9fc 0%, #ffffff 42%);
    }
    [data-testid="stHeader"] { background: rgba(255,255,255,0.75); }
    .block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1440px; }

    .hero {
        padding: 1.8rem 2rem;
        border-radius: 22px;
        background: linear-gradient(120deg, #123f55 0%, #176b70 58%, #2b8b86 100%);
        color: #ffffff;
        margin-bottom: 1.2rem;
        box-shadow: 0 12px 30px rgba(23, 75, 91, 0.14);
    }
    .hero-kicker {
        color: #c8f3ed;
        text-transform: uppercase;
        letter-spacing: 1.7px;
        font-size: .72rem;
        font-weight: 800;
        margin-bottom: .55rem;
    }
    .main-title { font-size: clamp(1.65rem, 3vw, 2.45rem); font-weight: 800; line-height: 1.15; margin: 0; color: #ffffff; }
    .sub-title { color: #e2f3f4; font-size: .98rem; margin-top: .65rem; margin-bottom: 0; line-height: 1.6; }
    .section-note { color: var(--muted); font-size: .92rem; line-height: 1.6; margin-top: -.35rem; }


    [data-testid="stTabs"] [data-baseweb="tab-list"] {
        gap: .5rem; background: #edf3f6; padding: .4rem; border-radius: 14px;
    }
    [data-testid="stTabs"] button[role="tab"] {
        border-radius: 10px; padding: .65rem 1rem; font-weight: 700; color: #52647a;
    }
    [data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
        background: #ffffff; color: var(--brand-dark); box-shadow: 0 2px 8px rgba(25,48,77,.08);
    }
    [data-testid="stVerticalBlockBorderWrapper"] {
        border-color: var(--line) !important; border-radius: 18px !important;
        background: rgba(255,255,255,.88);
    }
    h2, h3, h4 { color: var(--ink); letter-spacing: -.02em; }
    .stCaption, [data-testid="stCaptionContainer"] { color: var(--muted); }
    div[data-testid="stFormSubmitButton"] button, .stButton>button {
        background: var(--brand) !important; color: #ffffff !important;
        border: 0 !important; border-radius: 11px !important; font-weight: 750 !important;
        min-height: 3rem !important; transition: all .18s ease;
    }
    div[data-testid="stFormSubmitButton"] button:hover, .stButton>button:hover {
        background: var(--brand-dark) !important; box-shadow: 0 5px 14px rgba(23,107,112,.2) !important;
        transform: translateY(-1px);
    }
    [data-testid="stNumberInput"] input, [data-testid="stSelectbox"] div,
    [data-testid="stSlider"] { border-radius: 10px; }
    .result-box {
        padding: 1.8rem; border-radius: 18px; margin: 1.2rem 0;
        text-align: center; box-shadow: 0 8px 24px rgba(20,40,65,.06);
        animation: fadeIn .35s ease-out;
    }
    .result-normal { background: #eaf8f3; border: 1px solid #8ad8bf; }
    .result-pendek { background: #fff7e6; border: 1px solid #f2ce79; }
    .result-sangat { background: #fff0ef; border: 1px solid #f3aaa5; }
    .result-title { font-size: clamp(1.35rem, 2.5vw, 1.8rem); font-weight: 850; margin-bottom: .45rem; color: #172b4d; }
    .result-desc { font-size: .98rem; color: #52647a; margin: 0; line-height: 1.65; }
    .rec-box {
        background: #ffffff; padding: 1.35rem 1.5rem; border-radius: 16px; margin-top: 1rem;
        box-shadow: 0 5px 18px rgba(25,48,77,.045); border: 1px solid var(--line);
        border-left: 5px solid #176b70; animation: fadeIn .45s ease-out;
    }
    .rec-title { margin-top: 0; font-size: 1.05rem; font-weight: 800; display: flex; align-items: center; gap: 8px; }
    .rec-list { margin-bottom: 0; color: #405168; font-size: .94rem; line-height: 1.8; padding-left: 1.2rem; }
    .model-card {
        padding: 1.3rem; border-radius: 16px; border: 1px solid var(--line);
        background: #ffffff; box-shadow: 0 5px 18px rgba(25,48,77,.04); min-height: 175px;
        display: flex; flex-direction: column; justify-content: space-between;
    }
    .model-card h4 { margin-top: 0; font-size: 1.02rem; color: #405168; margin-bottom: .7rem; }
    .conf-value { font-size: 1.35rem; font-weight: 850; color: var(--ink); margin: .35rem 0; }
    .progress-bg { background: #e9eef4; border-radius: 999px; height: 8px; width: 100%; overflow: hidden; margin-top: 1rem; }
    .progress-fill { height: 100%; border-radius: 999px; transition: width .5s ease; }
    .badge { display: inline-block; padding: .35rem .75rem; border-radius: 999px; font-size: .78rem; font-weight: 750; margin-top: .55rem; }
    .badge-ok { background: #dff7ed; color: #08775e; }
    .badge-danger { background: #fff0dc; color: #9a5700; }
    .disclaimer { padding: .9rem 1rem; border-radius: 12px; background: #f1f6fa; border: 1px solid #dce7ef; color: #52647a; font-size: .84rem; line-height: 1.6; }
    @keyframes fadeIn { from { opacity: 0; transform: translateY(7px); } to { opacity: 1; transform: translateY(0); } }

    @media (max-width: 760px) {
        .block-container { padding-left: 1rem; padding-right: 1rem; }
        .hero { padding: 1.35rem; border-radius: 17px; }
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# FUNGSI DATA & MODEL
# ==========================================
@st.cache_data
def load_and_clean_data(csv_path):
    if not csv_path:
        return None, None, True
    df = pd.read_csv(csv_path)
    df_clean = df.copy()
    if "maternal_MUAC_cm" in df_clean.columns:
        z = zscore(df_clean["maternal_MUAC_cm"])
        df_clean = df_clean[np.abs(z) <= 3]
    if "birthweight_child_gram" in df_clean.columns:
        z = zscore(df_clean["birthweight_child_gram"])
        df_clean = df_clean[np.abs(z) <= 3]
    return df, df_clean, False

def get_model_bundle_path():
    """Locate the pretrained model bundle beside this Streamlit script."""
    if '__file__' in globals():
        base_dir = os.path.dirname(os.path.abspath(__file__))
    else:
        base_dir = os.getcwd()
    path = os.path.join(base_dir, MODEL_BUNDLE_NAME)
    return path if os.path.exists(path) else None


@st.cache_resource
def load_model_bundle(model_path):
    """Load the fitted estimators and evaluation results prepared from the notebook."""
    if not model_path or not os.path.exists(model_path):
        return None
    return joblib.load(model_path)

def confidence_for(model, proba, pred):
    classes = list(model.classes_)
    return proba[classes.index(pred)] * 100 if pred in classes else 0.0

# ==========================================
# HEADER & LOAD DATA
# ==========================================
st.markdown("""
<div class="hero">
    <div class="hero-kicker">Dashboard Kesehatan • Analisis Antropometri</div>
    <div class="main-title">Pemantauan Pertumbuhan Balita</div>
    <div class="sub-title">Eksplorasi data dan prediksi awal status tinggi badan menurut umur menggunakan Decision Tree dan Gaussian Naïve Bayes.</div>
</div>
""", unsafe_allow_html=True)

csv_path = get_csv_path()
df_raw, df_clean, is_missing_csv = load_and_clean_data(csv_path)
model_path = get_model_bundle_path()
model_bundle = load_model_bundle(model_path)

if is_missing_csv:
    st.error("⚠️ **Dataset Tidak Ditemukan!** Pastikan file CSV berada di folder yang sama dengan file Python ini.")
if model_bundle is None:
    st.error("⚠️ **File model belum ditemukan.** Pastikan `stunting_model_bundle.joblib` berada di folder yang sama dengan file Python ini.")

st.markdown("---")

# ==========================================
# TABS
# ==========================================
tab_prediksi, tab_evaluasi, tab_data = st.tabs(["🩺 Prediksi", "📈 Evaluasi Model", "🗂️ Data"])

# ==========================================
# TAB 1: PREDIKSI
# ==========================================
with tab_prediksi:
    st.subheader("Formulir pengukuran")
    st.markdown('<p class="section-note">Lengkapi data balita dan ibu. Pastikan angka sesuai hasil pengukuran sebelum menekan tombol prediksi.</p>', unsafe_allow_html=True)

    with st.container(border=True):
        with st.form("prediction_form"):
            col_anak, col_ibu = st.columns(2, gap="large")

            with col_anak:
                st.markdown("##### 🧒 Data Balita & Kelahiran")
                age_child_months = st.selectbox(
                    "Umur Balita (bulan)",
                    options=list(range(60)),
                    index=24,
                    format_func=lambda umur: f"{umur} bulan",
                    help="Pilih umur balita dari 0 sampai 59 bulan.",
                    key="umur_balita_dropdown_v5",
                )
                height_cm = st.number_input("Tinggi / Panjang Badan (cm)", 40.0, 120.0, 82.0, 0.5, format="%.1f")
                weight_kg = st.number_input("Berat Badan Saat Ini (kg)", 2.0, 25.0, 10.5, 0.1, format="%.1f")
                birthweight_child_gram = st.number_input("Berat Badan Saat Lahir (gram)", 1000, 5000, 3000, 50)
                
                if birthweight_child_gram >= 2500:
                    st.markdown('<div class="badge badge-ok">✅ Normal (≥ 2.500 g)</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="badge badge-danger">⚠️ BBLR (< 2.500 g)</div>', unsafe_allow_html=True)

            with col_ibu:
                st.markdown("##### 👩 Data Ibu")
                maternal_height_cm = st.number_input("Tinggi Badan Ibu (cm)", 130.0, 185.0, 150.5, 0.5, format="%.1f")
                maternal_MUAC_cm = st.number_input("LiLA / MUAC Ibu (cm)", 15.0, 40.0, 23.4, 0.1, format="%.1f")
                maternal_age_year = st.number_input("Umur Ibu (tahun)", 15, 50, 29, 1)

            submitted = st.form_submit_button("🔍 Prediksi Sekarang", type="primary", use_container_width=True)

    if submitted:
        if is_missing_csv:
            st.error("Dataset belum tersedia.")
        elif model_bundle is None:
            st.error("Model terlatih belum tersedia. Letakkan file model bundle di folder aplikasi.")
        else:
            feature_cols = model_bundle["feature_cols"]
            input_df = pd.DataFrame([{
                "maternal_height_cm": maternal_height_cm, "age_child_months": age_child_months,
                "birthweight_child_gram": birthweight_child_gram, "maternal_MUAC_cm": maternal_MUAC_cm,
                "weight_kg": weight_kg, "height_cm": height_cm, "maternal_age_year": maternal_age_year
            }])

            with st.spinner("Menghitung prediksi..."):
                # Model sudah dilatih berdasarkan alur notebook; di sini hanya memuat
                # scaler tersimpan dan menjalankan prediksi, tanpa melatih ulang.
                art = model_bundle
                input_scaled = art["scaler"].transform(input_df)

            pred_dtc = int(art["dtc"].predict(input_scaled)[0])
            conf_dtc = confidence_for(art["dtc"], art["dtc"].predict_proba(input_scaled)[0], pred_dtc)
            
            pred_gnb = int(art["gnb"].predict(input_scaled)[0])
            conf_gnb = confidence_for(art["gnb"], art["gnb"].predict_proba(input_scaled)[0], pred_gnb)

            # --- HASIL PREDIKSI UTAMA ---
            st.markdown("### Hasil prediksi")
            st.markdown('<p class="section-note">Hasil berikut merupakan estimasi model untuk membantu penapisan awal, bukan diagnosis medis.</p>', unsafe_allow_html=True)
            worst = min(pred_dtc, pred_gnb)
            both_normal = pred_dtc == 2 and pred_gnb == 2

            if both_normal:
                box_class, icon, headline = "result-normal", "✅", "Tidak Stunting (Normal)"
                rec_color, rec_icon, rec_title = "#10b981", "🌟", "Rekomendasi Tindak Lanjut (Normal)"
                rec_items = [
                    "<b>Apresiasi:</b> Berikan pujian kepada ibu dan anak atas pencapaian pertumbuhan yang baik.",
                    "<b>Edukasi Gizi (PMBA):</b> Lanjutkan pemberian MP-ASI bergizi seimbang serta ASI hingga usia 2 tahun.",
                    "<b>Pemantauan Rutin:</b> Tetap datang ke Posyandu setiap bulan untuk memantau pertumbuhan (BB & TB)."
                ]
            elif worst == 1:
                box_class, icon, headline = "result-pendek", "⚠️", "Terindikasi Stunting (Pendek)"
                rec_color, rec_icon, rec_title = "#f59e0b", "🏥", "Rekomendasi Tindak Lanjut (Waspada)"
                rec_items = [
                    "<b>Segera Konsultasi:</b> Bawa balita ke Posyandu atau Puskesmas terdekat untuk pemeriksaan lanjutan.",
                    "<b>Intervensi Gizi:</b> Ikuti program Pemberian Makanan Tambahan (PMT) dan edukasi pola asuh.",
                    "<b>Pantau Ketat:</b> Lakukan penimbangan rutin dan catat perkembangan di Buku KIA."
                ]
            else:
                box_class, icon, headline = "result-sangat", "🚨", "Terindikasi Stunting Berat"
                rec_color, rec_icon, rec_title = "#ef4444", "🚨", "Rekomendasi Tindak Lanjut (Darurat)"
                rec_items = [
                    "<b>Rujukan Medis:</b> Segera bawa balita ke dokter spesialis anak atau rumah sakit terdekat.",
                    "<b>Penanganan Intensif:</b> Diskusikan dengan ahli gizi untuk rencana diet khusus (F-100 / RUTF jika diperlukan).",
                    "<b>Pendampingan:</b> Minta pendampingan kader posyandu untuk pemantauan harian."
                ]

            # Render HTML (Menggunakan div bukan p untuk menghindari bug </p>)
            st.markdown(f"""
            <div class="result-box {box_class}">
                <div class="result-title">{icon} {headline}</div>
                <div class="result-desc">
                    Balita usia <b>{age_child_months} bulan</b> diprediksi <b>{headline}</b> berdasarkan pengukuran antropometri.
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Peringatan jika model berbeda pendapat
            if pred_dtc != pred_gnb:
                st.warning(f"**Perhatian:** Kedua model memberikan hasil yang berbeda. Sistem mengambil keputusan paling aman (waspada) yaitu **{LABEL_MAP[worst]}**.")

            # Render Box Rekomendasi
            list_html = "".join([f"<li>{item}</li>" for item in rec_items])
            st.markdown(f"""
            <div class="rec-box" style="border-left-color: {rec_color};">
                <h4 class="rec-title" style="color: {rec_color};">{rec_icon} {rec_title}</h4>
                <ul class="rec-list">
                    {list_html}
                </ul>
            </div>
            """, unsafe_allow_html=True)

            st.markdown('<div class="disclaimer">Catatan: persentase keyakinan merupakan probabilitas yang dikeluarkan model, bukan jaminan bahwa prediksi benar. Jika hasil mengkhawatirkan atau pengukuran diragukan, konsultasikan dengan tenaga kesehatan.</div>', unsafe_allow_html=True)

            # --- DETAIL PER MODEL ---
            cc1, cc2 = st.columns(2)
            with cc1:
                st.markdown(f"""
                <div class="model-card">
                    <div>
                        <h4>🌳 Decision Tree</h4>
                        <div class="conf-value">{LABEL_MAP[pred_dtc]}</div>
                        <div style="color:#6b7280; font-size:0.9rem;">Status: {STATUS_SINGKAT[pred_dtc]}</div>
                    </div>
                    <div>
                        <div class="progress-bg">
                            <div class="progress-fill" style="width: {conf_dtc}%; background: #3b82f6;"></div>
                        </div>
                        <div style="text-align: right; font-size: 0.8rem; color: #6b7280; margin-top: 5px;">Keyakinan: {conf_dtc:.1f}%</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
            with cc2:
                st.markdown(f"""
                <div class="model-card">
                    <div>
                        <h4>📈 Gaussian Naïve Bayes</h4>
                        <div class="conf-value">{LABEL_MAP[pred_gnb]}</div>
                        <div style="color:#6b7280; font-size:0.9rem;">Status: {STATUS_SINGKAT[pred_gnb]}</div>
                    </div>
                    <div>
                        <div class="progress-bg">
                            <div class="progress-fill" style="width: {conf_gnb}%; background: #8b5cf6;"></div>
                        </div>
                        <div style="text-align: right; font-size: 0.8rem; color: #6b7280; margin-top: 5px;">Keyakinan: {conf_gnb:.1f}%</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

# ==========================================
# TAB 2: EVALUASI MODEL
# ==========================================
with tab_evaluasi:
    st.subheader("Perbandingan kinerja model")
    st.markdown('<p class="section-note">Hasil evaluasi menggunakan model terlatih dengan parameter terbaik yang tercatat di notebook dan pembagian data uji 20% (random_state=42). Periksa juga confusion matrix untuk melihat prediksi pada tiap kelas.</p>', unsafe_allow_html=True)
    if is_missing_csv:
        st.warning("Dataset belum dimuat.")
    elif model_bundle is None:
        st.warning("File model terlatih belum tersedia.")
    else:
        art = model_bundle

        c1, c2, c3 = st.columns(3)
        c1.metric("🧹 Data Bersih", f"{len(df_clean)} baris", f"Train {art['n_train']} | Test {art['n_test']}")
        c2.metric("🌳 Akurasi Decision Tree", f"{art['dtc_acc']*100:.2f}%")
        c3.metric("📈 Akurasi Naïve Bayes", f"{art['gnb_acc']*100:.2f}%")

        st.markdown("---")
        st.markdown("##### Perbandingan Akurasi Model")
        chart_df = pd.DataFrame({
            "Model": ["Decision Tree", "Gaussian NB"],
            "Akurasi (%)": [art["dtc_acc"] * 100, art["gnb_acc"] * 100]
        }).set_index("Model")
        st.bar_chart(chart_df, use_container_width=True, height=280)

        st.markdown("---")
        labels = ["Sangat Pendek", "Pendek", "Normal"]
        cm_col1, cm_col2 = st.columns(2)
        with cm_col1:
            st.markdown("##### 🌳 Confusion Matrix — Decision Tree")
            df_cm1 = pd.DataFrame(art["dtc_cm"], index=[f"Aktual: {l}" for l in labels], columns=[f"Prediksi: {l}" for l in labels])
            st.dataframe(df_cm1, use_container_width=True)
        with cm_col2:
            st.markdown("##### 📈 Confusion Matrix — Gaussian NB")
            df_cm2 = pd.DataFrame(art["gnb_cm"], index=[f"Aktual: {l}" for l in labels], columns=[f"Prediksi: {l}" for l in labels])
            st.dataframe(df_cm2, use_container_width=True)

# ==========================================
# TAB 3: DATASET
# ==========================================
with tab_data:
    st.subheader("Eksplorasi dataset")
    st.markdown('<p class="section-note">Gunakan bagian ini untuk melihat distribusi kelas target dan meninjau contoh data setelah pembersihan.</p>', unsafe_allow_html=True)
    if is_missing_csv:
        st.write("Dataset belum ditemukan.")
    else:
        st.write(f"Menampilkan **{len(df_clean)} baris data bersih** dari total **{len(df_raw)} baris** awal.")
        
        st.markdown("##### Distribusi Kelas Target (`Indonesian_height_age`)")
        dist = df_clean["Indonesian_height_age"].value_counts().sort_index()
        dist_df = pd.DataFrame({
            "Kelas": [RAW_LABEL_MAP.get(int(k), f"Kelas {k}") for k in dist.index], 
            "Jumlah": dist.values
        }).set_index("Kelas")
        
        st.bar_chart(dist_df, use_container_width=True, height=260)
        
        st.markdown("##### Data Bersih (Preview)")
        st.dataframe(df_clean, use_container_width=True, height=400)