import streamlit as st
import pandas as pd
from pathlib import Path
import pydicom
import traceback
import numpy as np
from PIL import Image
import os
import zipfile
from huggingface_hub import HfApi, hf_hub_download
from streamlit_image_coordinates import streamlit_image_coordinates

st.set_page_config(
    page_title="TBI Doktor Etiketleme Paneli", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── HUGGING FACE & DOSYA AYARLARI ────────────────────────────────
HF_REPO_ID = "oyailgin/CQ500_export300"
HF_TOKEN = st.secrets.get("HF_TOKEN", "token")

LOCAL_DATA_DIR = Path("extracted_data")
ANNOTATION_FILE = "doctor_annotations.csv"
READS_CSV_PATH = LOCAL_DATA_DIR / "cq500_exact_300_reads.csv"


@st.cache_resource
def download_and_extract_data():
    """Hugging Face'teki private dataset'ten tüm kesitleri içeren zip'i indirip çözer"""
    if not LOCAL_DATA_DIR.exists():
        os.makedirs(LOCAL_DATA_DIR, exist_ok=True)
    
    st.info("🔄 Veriler Hugging Face'ten indiriliyor ve hazırlanıyor, lütfen bekleyin...")
    try:
        # Tüm kesitleri içeren güncel zip dosyasını indir
        zip_path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename="all_slices_complete.zip",
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

        slope = getattr(ds, 'RescaleSlope', 1)
        intercept = getattr(ds, 'RescaleIntercept', 0)
        arr = arr * slope + intercept

        # Tıbbi Beyin Penceresi (Brain Window): Level = 40, Width = 80
        window_center = 40.0
        window_width = 80.0

        min_val = window_center - (window_width / 2.0)
        max_val = window_center + (window_width / 2.0)

        arr = np.clip(arr, min_val, max_val)
        arr = ((arr - min_val) / (max_val - min_val)) * 255.0
        arr = arr.astype(np.uint8)

        return Image.fromarray(arr)
    except Exception as e:
        print(f"--- DICOM OKUMA HATASI ---")
        print(f"Dosya: {dcm_path}")
        print(f"Hata mesajı: {str(e)}")
        return None


# Klasör yapısını bulma
slices_root = LOCAL_DATA_DIR / "selected_slices"
if not slices_root.exists():
    sub_dirs = [d for d in LOCAL_DATA_DIR.iterdir() if d.is_dir()]
    if sub_dirs:
        slices_root = sub_dirs[0] / "selected_slices"

if slices_root.exists():
    patients = sorted([p.name for p in slices_root.iterdir() if p.is_dir()])
else:
    patients = []

# ── YAN MENÜ (SIDEBAR) ────────────────────────────────────────
st.sidebar.title("🎛️ Kontrol Paneli")
doctor_name = st.sidebar.text_input("👨‍⚕️ Doktor Adı / ID", value="Dr. Uzman")

if not patients:
    st.warning("⚠️ Hasta klasörleri bulunamadı. Lütfen HF repo adı ve zip dosya adını kontrol edin.")
else:
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📂 Hasta Gezinme")

    # Session State ile İleri / Geri Index Takibi
    if "patient_index" not in st.session_state:
        st.session_state.patient_index = 0

    col_prev, col_next = st.sidebar.columns(2)
    with col_prev:
        if st.button("⬅️ Önceki"):
            if st.session_state.patient_index > 0:
                st.session_state.patient_index -= 1
                st.rerun()
    with col_next:
        if st.button("Sonraki ➡️"):
            if st.session_state.patient_index < len(patients) - 1:
                st.session_state.patient_index += 1
                st.rerun()

    # Dropdown ve butonların senkronizasyonu
    selected_patient = st.sidebar.selectbox(
        "Hasta Seçin", 
        patients, 
        index=st.session_state.patient_index,
        key="selected_patient_selectbox"
    )
    st.session_state.patient_index = patients.index(selected_patient)

    # İlerleme Bilgisi
    st.sidebar.markdown("---")
    st.sidebar.info(f"📊 **Toplam Hasta:** {len(patients)} | **Aktif Sıra:** {st.session_state.patient_index + 1}")

    # ── ANA BAŞLIK VE BİLGİ ────────────────────────────────────────
    st.title(f"🧠 Vaka İnceleme: Hasta {selected_patient}")
    st.info(f"👤 **Aktif Sorumlu:** {doctor_name} | **Seçilen Vaka ID:** {selected_patient}")

    patient_folder = slices_root / selected_patient
    dcm_files = sorted(list(patient_folder.glob("*.dcm")))

    if not dcm_files:
        st.error(f"❌ {selected_patient} klasöründe hiç `.dcm` kesiti bulunamadı!")
    else:
        # ── MODERN SEKME (TAB) YAPISI ──────────────────────────────
        tab1, tab2, tab3 = st.tabs([
            "🖼️ Tüm Kesitler (Görselleştirme)", 
            "📍 Midline Shift & Klinik Form", 
            "📜 Hasta Geçmişi & Orijinal Tanılar"
        ])

        # --- TAB 1: TÜM KESİTLER ---
        with tab1:
            st.subheader(f"Hasta {selected_patient} — Tüm DICOM Kesitleri ({len(dcm_files)} Kesit)")
            st.markdown("Aşağıda ilgili hastaya ait tüm taranmış kesitler yer almaktadır.")
            
            # 4'lü sütunlar halinde tüm kesitleri grid olarak göster
            grid_cols = st.columns(4)
            for idx, dcm_file in enumerate(dcm_files):
                c_idx = idx % 4
                with grid_cols[c_idx]:
                    pil_img = dcm_to_image(dcm_file)
                    if pil_img:
                        st.image(pil_img, caption=f"Kesit {idx+1}", use_container_width=True)
                    else:
                        st.error(f"Kesit {idx+1} okunamadı")

        # --- TAB 2: MIDLINE SHIFT & KLİNİK FORM ---
        with tab2:
            st.subheader("📍 Orta Hat Kayması (Midline Shift) İşaretleme")
            st.markdown("İşaretleme yapmak istediğiniz kesiti seçin ve beyin orta hattı için **3 nokta** belirleyin.")

            slice_names = [f"Kesit {i + 1}" for i in range(len(dcm_files))]
            selected_slice_idx = st.selectbox(
                "İşaretleme yapılacak kesiti seçin", range(len(dcm_files)),
                format_func=lambda i: slice_names[i],
                key="ann_slice_select"
            )

            points_key = f"points_{selected_patient}_{selected_slice_idx}"
            if points_key not in st.session_state:
                st.session_state[points_key] = []

            mark_img = dcm_to_image(dcm_files[selected_slice_idx])

            if mark_img:
                mark_img_rgb = mark_img.convert("RGB")
                draw_arr = np.array(mark_img_rgb)

                # Önceden işaretlenmiş noktaları kırmızı kareciklerle çiz
                for (px, py) in st.session_state[points_key]:
                    y0, y1 = max(0, py - 3), min(draw_arr.shape[0], py + 3)
                    x0, x1 = max(0, px - 3), min(draw_arr.shape[1], px + 3)
                    draw_arr[y0:y1, x0:x1] = [255, 0, 0]

                display_img = Image.fromarray(draw_arr)
                coords = streamlit_image_coordinates(display_img, key=f"click_{points_key}")

                if coords is not None:
                    new_point = (coords["x"], coords["y"])
                    if not st.session_state[points_key] or st.session_state[points_key][-1] != new_point:
                        if len(st.session_state[points_key]) >= 3:
                            st.session_state[points_key] = []
                        st.session_state[points_key].append(new_point)
                        st.rerun()

                st.caption(f"İşaretlenen nokta sayısı: {len(st.session_state[points_key])}/3 (3. noktada liste sıfırlanır)")
                if st.button("🔄 Noktaları Temizle"):
                    st.session_state[points_key] = []
                    st.rerun()

            st.markdown("---")

            # Klinik Değerlendirme Formu
            with st.form("annotation_form"):
                st.subheader("📋 Klinik Değerlendirme ve Kayıt Formu")

                points = st.session_state.get(points_key, [])
                if len(points) == 3:
                    st.success(f"✅ 3 Nokta Hazır: {points[0]} , {points[1]} , {points[2]}")
                else:
                    st.warning("⚠️ Kaydetmeden önce yukarıda tam olarak 3 nokta işaretlemelisiniz.")

                basal_status = st.selectbox(
                    "Bazal Sistern Durumu",
                    ["Normal (0)", "Sıkışmış / Compressed (1)", "Yok / Absent (2)"]
                )

                notes = st.text_area("Klinik Notlar / Açıklamalar")
                submitted = st.form_submit_button("💾 İşaretlemeyi Kaydet ve HF'ye Gönder")

                if submitted:
                    if len(points) != 3:
                        st.error("Kaydetmeden önce tam olarak 3 nokta işaretlemelisiniz.")
                        st.stop()

                    new_record = {
                        "doctor_name": doctor_name,
                        "patient_id": selected_patient,
                        "slice_idx": selected_slice_idx,
                        "point1_x": points[0][0],
                        "point1_y": points[0][1],
                        "point2_x": points[1][0],
                        "point2_y": points[1][1],
                        "point3_x": points[2][0],
                        "point3_y": points[2][1],
                        "basal_status": basal_status,
                        "notes": notes,
                        "timestamp": str(pd.Timestamp.now())
                    }

                    # reads.csv içinden orijinal meta verileri birleştir
                    if READS_CSV_PATH.exists():
                        df_reads = pd.read_csv(READS_CSV_PATH)
                        # Sütun adını esnek aratalım ('name' veya 'patient_id')
                        name_col = 'name' if 'name' in df_reads.columns else 'patient_id'
                        patient_meta = df_reads[df_reads[name_col].astype(str) == str(selected_patient)]

                        if not patient_meta.empty:
                            meta_dict = patient_meta.iloc[0].to_dict()
                            new_record.update(meta_dict)

                    df_new = pd.DataFrame([new_record])
                    local_csv = Path(ANNOTATION_FILE)

                    if local_csv.exists():
                        df_old = pd.read_csv(local_csv)
                        for col in new_record.keys():
                            if col not in df_old.columns:
                                df_old[col] = None

                        mask = (df_old["patient_id"].astype(str) == str(selected_patient)) & \
                               (df_old["slice_idx"] == selected_slice_idx)

                        if mask.any():
                            for col, val in new_record.items():
                                df_old.loc[mask, col] = val
                            df_combined = df_old
                        else:
                            df_combined = pd.concat([df_old, df_new], ignore_index=True)
                    else:
                        df_combined = df_new

                    df_combined.to_csv(local_csv, index=False)

                    # Hugging Face Dataset'e güncellemeyi yükle
                    try:
                        api = HfApi(token=HF_TOKEN)
                        api.upload_file(
                            path_or_fileobj=str(local_csv),
                            path_in_repo=ANNOTATION_FILE,
                            repo_id=HF_REPO_ID,
                            repo_type="dataset",
                            commit_message=f"Doktor notu eklendi: Hasta {selected_patient}"
                        )
                        st.success(f"✅ {selected_patient} için kayıt başarıyla HF'ye yüklendi!")
                        st.session_state[points_key] = []
                    except Exception as e:
                        st.error(f"Kayıt yapıldı ancak HF'ye yüklenirken hata oluştu: {e}")

        # --- TAB 3: HASTA GEÇMİŞİ & ORİJİNAL TANILAR ---
        with tab3:
            st.subheader(f"📜 Hasta {selected_patient} - Orijinal Tanı ve Geçmiş Etiketler")
            
            # 1. Orijinal Read / Tanı Bilgileri
            st.markdown("#### 🏥 Orijinal Veriseti Tanıları (Reads)")
            if READS_CSV_PATH.exists():
                df_reads = pd.read_csv(READS_CSV_PATH)
                name_col = 'name' if 'name' in df_reads.columns else 'patient_id'
                p_meta = df_reads[df_reads[name_col].astype(str) == str(selected_patient)]
                
                if not p_meta.empty:
                    st.dataframe(p_meta, use_container_width=True)
                else:
                    st.info("ℹ️ Bu hasta için `reads.csv` içinde eşleşen satır bulunamadı.")
            else:
                st.warning("⚠️ `reads.csv` dosyası dizinde bulunamadı.")

            st.markdown("---")

            # 2. Daha Önce Bu Hastaya Girilmiş Doktor Notları
            st.markdown("#### 👨‍⚕️ Bu Vaka İçin Daha Önce Girilen Doktor Etiketleri")
            local_csv = Path(ANNOTATION_FILE)
            if local_csv.exists():
                df_existing = pd.read_csv(local_csv)
                patient_history = df_existing[df_existing["patient_id"].astype(str) == str(selected_patient)]
                
                if not patient_history.empty:
                    display_cols = [c for c in ["doctor_name", "timestamp", "slice_idx", "basal_status", "notes"] if c in patient_history.columns]
                    st.dataframe(patient_history[display_cols], use_container_width=True)
                else:
                    st.info("ℹ️ Bu hasta için henüz başka bir doktor kaydı veya etiketleme girilmemiş.")
            else:
                st.info("ℹ️ Henüz sistemde yerel `doctor_annotations.csv` kaydı yok.")