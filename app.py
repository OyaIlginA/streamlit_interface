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

st.set_page_config(page_title="TBI Doktor Etiketleme Paneli", layout="wide")

st.title("🧠 TBI / CQ500 Kesit İnceleme ve Etiketleme Paneli")

# ── HUGGING FACE AYARLARI ────────────────────────────────────────
HF_REPO_ID = "oyailgin/CQ500_export300"
HF_TOKEN = st.secrets.get("HF_TOKEN", "token")

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
            zip_path = hf_hub_download(
                repo_id=HF_REPO_ID,
                filename="selected_slices.zip",
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
        print(f"--- HATA DETAYI ---")
        print(f"Dosya: {dcm_path}")
        print(f"Hata mesajı: {str(e)}")
        traceback.print_exc() # Tüm hata izini konsola dök


# Klasör yapısını bulma (selected_slices nerede kaldıysa)
slices_root = LOCAL_DATA_DIR / "selected_slices"
if not slices_root.exists():
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
                    st.image(pil_img, caption=f"Kesit {idx+1}", use_container_width=True)
                else:
                    st.error("Okunamadı")

        st.markdown("---")

        # ── MIDLINE SHIFT İÇİN NOKTA İŞARETLEME (3 NOKTA) ────────
        st.markdown("📍 **Midline Shift İşaretleme (3 nokta seçin)**")

        slice_names = [f"Kesit {i + 1}" for i in range(len(dcm_files))]
        selected_slice_idx = st.selectbox(
            "İşaretleme yapılacak kesiti seçin", range(len(dcm_files)),
            format_func=lambda i: slice_names[i]
        )

        points_key = f"points_{selected_patient}_{selected_slice_idx}"
        if points_key not in st.session_state:
            st.session_state[points_key] = []

        mark_img = dcm_to_image(dcm_files[selected_slice_idx])

        if mark_img:
            mark_img_rgb = mark_img.convert("RGB")
            draw_arr = np.array(mark_img_rgb)

            # önceden işaretlenmiş noktaları görselde göster
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

            st.caption(f"İşaretlenen nokta sayısı: {len(st.session_state[points_key])}/3")
            if st.button("🔄 Noktaları Temizle"):
                st.session_state[points_key] = []
                st.rerun()

        st.markdown("---")

        # ── FORM ALANI ────────────────────────────────────────────
        with st.form("annotation_form"):
            st.subheader("📋 Klinik Değerlendirme Formu")

            points = st.session_state.get(points_key, [])
            if len(points) == 3:
                st.success(f"Noktalar: {points[0]} , {points[1]} , {points[2]}")
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

                # reads.csv içinden bu hastaya ait orijinal satırı bul ve birleştir
                if READS_CSV_PATH.exists():
                    df_reads = pd.read_csv(READS_CSV_PATH)
                    patient_meta = df_reads[df_reads['name'].astype(str) == str(selected_patient)]

                    if not patient_meta.empty:
                        meta_dict = patient_meta.iloc[0].to_dict()
                        new_record.update(meta_dict)

                df_new = pd.DataFrame([new_record])
                local_csv = Path(ANNOTATION_FILE)

                if local_csv.exists():
                    df_old = pd.read_csv(local_csv)

                    # eski CSV'de olmayan sütunları ekle (geriye dönük uyumluluk)
                    for col in new_record.keys():
                        if col not in df_old.columns:
                            df_old[col] = None

                    # aynı hasta + aynı kesit için upsert (üzerine yaz)
                    mask = (df_old["patient_id"] == selected_patient) & \
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
                    st.session_state[points_key] = []
                except Exception as e:
                    st.error(f"Kayıt yapıldı ancak HF'ye yüklenirken hata oluştu: {e}")