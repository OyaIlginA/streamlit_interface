import streamlit as st
import pandas as pd
from pathlib import Path
import pydicom
import numpy as np
from PIL import Image
import os
import zipfile
from huggingface_hub import HfApi, hf_hub_download

st.set_page_config(page_title="TBI Doktor Etiketleme Paneli", layout="wide")

st.title("🧠 TBI / CQ500 Kesit İnceleme ve Etiketleme Paneli")

# ── HUGGING FACE AYARLARI ────────────────────────────────────────
# Kendi Hugging Face kullanıcı adı ve dataset adını buraya yazmalısın!
HF_REPO_ID = "oyailgin/CQ500_export300"  # Örn: "oyailgin/tbi-cq500-slices"
HF_TOKEN = st.secrets.get("HF_TOKEN", "token") # Veya Streamlit secrets'tan alır

LOCAL_DATA_DIR = Path("extracted_data")
ANNOTATION_FILE = "doctor_annotations.csv"
READS_CSV_PATH = LOCAL_DATA_DIR / "cq500_exact_300_reads.csv"

@st.cache_resource
def download_and_extract_data():
    """Hugging Face'teki private dataset'ten zip'i indirip çözer"""
    if not LOCAL_DATA_DIR.exists():
        os.makedirs(LOCAL_DATA_DIR, exist_ok=True)
        st.info("🔄 Veriler Hugging Face'ten indiriliyor, lütfen bekleyin...")
        try:
            # Hugging Face'ten zip dosyasını indir (Dataset'e yüklediğin zip dosyasının adı neyse onu yaz)
            zip_path = hf_hub_download(
                repo_id=HF_REPO_ID, 
                filename="TBI_Exact_300_Export_20260813_1341.zip", # <-- Yüklediğin ZIP dosyasının adı!
                repo_type="dataset", 
                token=HF_TOKEN
            )
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(LOCAL_DATA_DIR)
            st.success("✅ Veriler başarıyla yüklendi!")
        except Exception as e:
            st.error(f"Veri indirilirken hata oluştu: {e}")

download_and_extract_data()

# ── DICOM GÖRSELLEŞTİRME ────────────────────────────────────────
def dcm_to_image(dcm_path):
    try:
        ds = pydicom.dcmread(str(dcm_path))
        arr = ds.pixel_array.astype(float)
        
        # Orijinal DICOM Hounsfield Unit (HU) değerlerine dönüştürme (Slope ve Intercept varsa)
        slope = getattr(ds, 'RescaleSlope', 1)
        intercept = getattr(ds, 'RescaleIntercept', 0)
        arr = arr * slope + intercept
        
        # Tıbbi Beyin Penceresi (Brain Window): Level = 40, Width = 80
        # Bu aralık beyin parankimini, ventrikülleri ve kanamaları net bir şekilde ortaya çıkarır.
        window_center = 40.0
        window_width = 80.0
        
        min_val = window_center - (window_width / 2.0)
        max_val = window_center + (window_width / 2.0)
        
        # Pikselleri bu pencereye göre kırp (Clipping)
        arr = np.clip(arr, min_val, max_val)
        
        # 0 ile 255 arasına normalize et (Görselleştirme için)
        arr = ((arr - min_val) / (max_val - min_val)) * 255.0
        arr = arr.astype(np.uint8)
        
        return Image.fromarray(arr)
    except Exception as e:
        return None

# Klasör yapısını bulma (selected_slices nerede kaldıysa)
slices_root = LOCAL_DATA_DIR / "selected_slices"
if not slices_root.exists():
    # Eğer zip direkt klasörleri çıkardıysa arayalım
    sub_dirs = [d for d in LOCAL_DATA_DIR.iterdir() if d.is_dir()]
    if sub_dirs:
        slices_root = sub_dirs[0] / "selected_slices"

if slices_root.exists():
    patients = sorted([p.name for p in slices_root.iterdir() if p.is_dir()])
else:
    patients = []

if not patients:
    st.warning("⚠️ Hasta klasörleri bulunamadı. Lütfen HF repo adı ve zip dosya adını kontrol edin.")
else:
    selected_patient = st.sidebar.selectbox("📂 Hasta Seçin", patients)
    doctor_name = st.sidebar.text_input("👨‍⚕️ Doktor Adı / ID", value="Dr. Uzman")

    st.subheader(f"Seçilen Hasta: {selected_patient}")

    patient_folder = slices_root / selected_patient
    dcm_files = sorted(list(patient_folder.glob("*.dcm")))

    if dcm_files:
        st.markdown("🔍 **SSA Algoritması Tarafından Seçilen Kesitler (DICOM):**")
        cols = st.columns(len(dcm_files) if len(dcm_files) <= 6 else 6)
        
        for idx, dcm_file in enumerate(dcm_files):
            col_idx = idx % 6
            with cols[col_idx]:
                pil_img = dcm_to_image(dcm_file)
                if pil_img:
                    st.image(pil_img, caption=f"Kesit {idx+1}", use_column_width=True)
                else:
                    st.error(f"Okunamadı")
        
        st.markdown("---")

        # Form Alanı
        with st.form("annotation_form"):
            st.subheader("📋 Klinik Değerlendirme Formu")
            
            c1, c2 = st.columns(2)
            with c1:
                midline_shift_mm = st.number_input("Midline Shift (mm)", min_value=0.0, max_value=50.0, value=0.0, step=0.1)
                mls_positive = st.checkbox("Orta Hat Kayması Var mı? (≥5mm)")
            with c2:
                basal_status = st.selectbox(
                    "Bazal Sistern Durumu",
                    ["Normal (0)", "Sıkışmış / Compressed (1)", "Yok / Absent (2)"]
                )
            
            notes = st.text_area("Klinik Notlar / Açıklamalar")
            submitted = st.form_submit_button("💾 İşaretlemeyi Kaydet ve HF'ye Gönder")
            
            if submitted:
                # 1. Doktorun o anki girdileri
                new_record = {
                    "doctor_name": doctor_name,
                    "patient_id": selected_patient,
                    "midline_shift_mm": midline_shift_mm,
                    "mls_positive": int(mls_positive),
                    "basal_status": basal_status,
                    "notes": notes,
                    "timestamp": str(pd.Timestamp.now())
                }
                
                # 2. reads.csv içinden bu hastaya ait orijinal satırı bul ve birleştir
                if READS_CSV_PATH.exists():
                    df_reads = pd.read_csv(READS_CSV_PATH)
                    # Hasta ID sütununun adını kendi csv'ne göre düzenleyebilirsin (örn: 'PatientID' veya 'id')
                    patient_meta = df_reads[df_reads['name'].astype(str) == str(selected_patient)]
                    
                    if not patient_meta.empty:
                        # Meta verileri ile doktor verilerini sözlük (dict) olarak birleştir
                        meta_dict = patient_meta.iloc[0].to_dict()
                        new_record.update(meta_dict) # Orijinal veriler de satıra eklendi!
                
                df_new = pd.DataFrame([new_record])
                local_csv = Path(ANNOTATION_FILE)
                
                if local_csv.exists():
                    df_old = pd.read_csv(local_csv)
                    # Aynı hastanın aynı doktor tarafından tekrar kaydı varsa eskisini günçelle veya alta ekle
                    df_combined = pd.concat([df_old, df_new], ignore_index=True)
                else:
                    df_combined = df_new
                    
                df_combined.to_csv(local_csv, index=False)
                
                # Hugging Face Dataset'e güncellemeyi gönder
                try:
                    api = HfApi(token=HF_TOKEN)
                    api.upload_file(
                        path_or_fileobj=str(local_csv),
                        path_in_repo=ANNOTATION_FILE,
                        repo_id=HF_REPO_ID,
                        repo_type="dataset",
                        commit_message=f"Doktor notu eklendi: {selected_patient}"
                    )
                    st.success(f"✅ {selected_patient} için orijinal verilerle birleştirilen kayıt başarıyla HF'ye yüklendi!")
                except Exception as e:
                    st.error(f"Kayıt yapıldı ancak HF'ye yüklenirken hata oluştu: {e}")