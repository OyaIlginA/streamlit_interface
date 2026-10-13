from __future__ import annotations

import os
import traceback
import zipfile
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import pydicom
import streamlit as st
from PIL import Image, ImageDraw
from huggingface_hub import HfApi, hf_hub_download
from streamlit_image_coordinates import streamlit_image_coordinates


# =============================================================================
# CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="TBI | İnceleme Paneli",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

HF_REPO_ID = "oyailgin/CQ500_export300"
ZIP_FILENAME = "cq500_dual_slices.zip"
LOCAL_DATA_DIR = Path("extracted_data")
ANNOTATION_FILE = Path("doctor_annotations.csv")
READS_CSV_PATH = LOCAL_DATA_DIR / "cq500_exact_300_reads.csv"

try:
    HF_TOKEN = st.secrets.get("HF_TOKEN", os.getenv("HF_TOKEN", ""))
except Exception:
    HF_TOKEN = os.getenv("HF_TOKEN", "")


# =============================================================================
# VISUAL DESIGN
# =============================================================================
st.markdown(
    """
    <style>
      :root { color-scheme: light; }
      .block-container { padding-top: 1.4rem; padding-bottom: 3rem; max-width: 1600px; }
      [data-testid="stSidebar"] { background: #f4f7fb; border-right: 1px solid #e4eaf2; }
      [data-testid="stSidebar"] > div { padding-top: 1.5rem; }
      h1, h2, h3 { letter-spacing: -0.025em; }
      h1 { color: #172b4d; }
      div[data-testid="stMetric"] {
        background: #ffffff; border: 1px solid #e5eaf1; padding: 14px 16px;
        border-radius: 14px; box-shadow: 0 2px 8px rgba(24, 43, 77, .035);
      }
      div[data-testid="stVerticalBlockBorderWrapper"] {
        border-color: #e5eaf1 !important; border-radius: 14px !important;
      }
      .eyebrow { color: #64748b; font-size: .78rem; font-weight: 700;
                 letter-spacing: .08em; text-transform: uppercase; }
      .muted { color: #64748b; }
      div.stButton > button { border-radius: 10px; min-height: 2.55rem; font-weight: 600; }
      div.stFormSubmitButton > button { border-radius: 10px; min-height: 2.8rem; font-weight: 700; }
      [data-testid="stImage"] img { border-radius: 10px; }
      .section-note { color: #64748b; font-size: .92rem; margin-top: -.35rem; }
      @media (max-width: 800px) {
        .block-container { padding-left: 1rem; padding-right: 1rem; }
      }
    </style>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# DATA LOADING
# =============================================================================
@st.cache_resource(show_spinner=False)
def download_and_extract_data() -> tuple[bool, str]:
    """Download and extract the configured dataset once per app process."""
    LOCAL_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # If a usable dataset is already present locally, do not redownload it.
    if any(LOCAL_DATA_DIR.glob("*/ssa_slices")) or any(LOCAL_DATA_DIR.glob("*/all_slices")):
        return True, "Yerel veri klasörü kullanılıyor."

    if not HF_TOKEN:
        return False, (
            "HF_TOKEN bulunamadı. Hugging Face erişimi için .streamlit/secrets.toml "
            "dosyasına HF_TOKEN ekleyin veya HF_TOKEN ortam değişkenini tanımlayın."
        )

    try:
        zip_path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=ZIP_FILENAME,
            repo_type="dataset",
            token=HF_TOKEN,
        )
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            # Prevent paths in an archive from writing outside the data directory.
            root = LOCAL_DATA_DIR.resolve()
            for member in zip_ref.infolist():
                target = (LOCAL_DATA_DIR / member.filename).resolve()
                if target != root and root not in target.parents:
                    raise ValueError("ZIP dosyasında güvenli olmayan bir dosya yolu bulundu.")
            zip_ref.extractall(LOCAL_DATA_DIR)
        return True, "Veriler başarıyla indirildi."
    except Exception as exc:
        return False, f"Veriler indirilemedi: {exc}"


@st.cache_data(show_spinner=False)
def dcm_to_array(dcm_path: str) -> np.ndarray | None:
    """Convert a DICOM image to the app's existing brain-window grayscale view."""
    try:
        ds = pydicom.dcmread(dcm_path)
        arr = ds.pixel_array.astype(np.float32)
        slope = getattr(ds, "RescaleSlope", 1)
        intercept = getattr(ds, "RescaleIntercept", 0)
        arr = arr * slope + intercept

        # Preserve the existing display window: center 40, width 80.
        window_center, window_width = 40.0, 80.0
        min_val = window_center - window_width / 2.0
        max_val = window_center + window_width / 2.0
        arr = np.clip(arr, min_val, max_val)
        arr = ((arr - min_val) / (max_val - min_val)) * 255.0
        return arr.astype(np.uint8)
    except Exception:
        print(f"DICOM okunamadı: {dcm_path}")
        traceback.print_exc()
        return None


def dcm_to_image(dcm_path: str | Path) -> Image.Image | None:
    arr = dcm_to_array(str(dcm_path))
    return Image.fromarray(arr) if arr is not None else None


def get_patient_directories(root: Path) -> tuple[Path, list[str]]:
    """Locate patient folders while supporting the common nested ZIP layouts."""
    if not root.exists():
        return root, []

    candidates = [root]
    candidates.extend(p for p in root.iterdir() if p.is_dir() and p.name != "__MACOSX")

    for base in candidates:
        patient_dirs = sorted(
            p.name for p in base.iterdir()
            if p.is_dir() and p.name != "__MACOSX"
            and (p.name.startswith("CQ500") or (p / "ssa_slices").exists()
                 or (p / "all_slices").exists())
        )
        if patient_dirs:
            return base, patient_dirs

    # Fallback: preserve the prior app's broad folder discovery behavior.
    dirs = sorted(p.name for p in root.iterdir() if p.is_dir() and p.name != "__MACOSX")
    return root, dirs


def find_slice_folder(patient_folder: Path, preferred: str) -> Path:
    preferred_path = patient_folder / preferred
    if preferred_path.exists():
        return preferred_path
    return patient_folder


def get_dicom_files(folder: Path) -> list[Path]:
    return sorted(folder.glob("*.dcm"), key=lambda p: p.name.lower())


def patient_point_key(patient_id: str, slice_idx: int) -> str:
    return f"points::{patient_id}::{slice_idx}"


def safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def sync_patient_index(patient_ids: list[str]) -> None:
    if not patient_ids:
        st.session_state.patient_index = 0
        return
    st.session_state.patient_index = max(
        0, min(int(st.session_state.get("patient_index", 0)), len(patient_ids) - 1)
    )


# =============================================================================
# STARTUP
# =============================================================================
with st.spinner("Veri klasörü kontrol ediliyor..."):
    data_ok, data_message = download_and_extract_data()

if not data_ok:
    st.warning(data_message)
elif data_message:
    st.toast(data_message, icon="📂")

data_root, patients = get_patient_directories(LOCAL_DATA_DIR)

# The source CSV may be inside the extracted data directory.
if not READS_CSV_PATH.exists():
    matches = list(LOCAL_DATA_DIR.rglob("cq500_exact_300_reads.csv"))
    if matches:
        READS_CSV_PATH = matches[0]

st.title("🧠 TBI İnceleme Paneli")
st.caption("CQ500 · DICOM kesit inceleme, işaretleme ve klinik değerlendirme")

if not patients:
    st.error(
        "Hasta klasörleri bulunamadı. Hugging Face depo adını, ZIP dosya adını "
        "ve çıkarılan klasör yapısını kontrol edin."
    )
    st.stop()


# =============================================================================
# SIDEBAR: PATIENT NAVIGATION
# =============================================================================
sync_patient_index(patients)

with st.sidebar:
    st.markdown("## 🩺 Hasta gezinme")
    st.caption("İncelemek istediğiniz hastayı seçin.")

    patient_options = patients
    current_patient = patient_options[st.session_state.patient_index]

    chosen_patient = st.selectbox(
        "Hasta",
        patient_options,
        index=st.session_state.patient_index,
        key="patient_picker",
        label_visibility="collapsed",
    )
    new_index = patient_options.index(chosen_patient)
    if new_index != st.session_state.patient_index:
        st.session_state.patient_index = new_index
        st.session_state.all_slice_index = 0
        st.rerun()

    nav_prev, nav_next = st.columns(2)
    with nav_prev:
        if st.button("← Önceki", use_container_width=True, disabled=st.session_state.patient_index == 0):
            st.session_state.patient_index -= 1
            st.session_state.all_slice_index = 0
            st.rerun()
    with nav_next:
        if st.button(
            "Sonraki →",
            use_container_width=True,
            disabled=st.session_state.patient_index >= len(patients) - 1,
        ):
            st.session_state.patient_index += 1
            st.session_state.all_slice_index = 0
            st.rerun()

    st.divider()
    doctor_name = st.text_input("Doktor adı / ID", value=st.session_state.get("doctor_name", "Dr. Uzman"))
    st.session_state.doctor_name = doctor_name

    st.divider()
    st.markdown("#### Kısayol bilgisi")
    st.caption("1. Izgaradan bir kesit seçin.")
    st.caption("2. Görüntü üzerine üç nokta yerleştirin.")
    st.caption("3. Klinik değerlendirmeyi kaydedin.")


selected_patient = patients[st.session_state.patient_index]
patient_folder = data_root / selected_patient

# Support an additional nested root layout if the selected patient is not under data_root.
if not patient_folder.exists():
    patient_folder = LOCAL_DATA_DIR / selected_patient

ssa_folder = find_slice_folder(patient_folder, "ssa_slices")
ssa_files = get_dicom_files(ssa_folder)
all_slices_folder = find_slice_folder(patient_folder, "all_slices")
all_dcm_files = get_dicom_files(all_slices_folder)

# If the selected directory contains nested patient folders, search one level down.
if not ssa_files:
    nested = sorted(patient_folder.glob("*/ssa_slices/*.dcm"))
    if nested:
        ssa_files = nested
if not all_dcm_files:
    nested_all = sorted(patient_folder.glob("*/all_slices/*.dcm"))
    if nested_all:
        all_dcm_files = nested_all


# =============================================================================
# PATIENT OVERVIEW + HISTORY
# =============================================================================
df_existing = safe_read_csv(ANNOTATION_FILE)
if df_existing.empty:
    # Preserve compatibility if the CSV was saved alongside extracted data.
    candidate_csv = LOCAL_DATA_DIR / ANNOTATION_FILE.name
    if candidate_csv.exists():
        df_existing = safe_read_csv(candidate_csv)

patient_history = pd.DataFrame()
if not df_existing.empty and "patient_id" in df_existing.columns:
    patient_history = df_existing[
        df_existing["patient_id"].astype(str) == str(selected_patient)
    ]

saved_count = len(patient_history)
total_slices = len(ssa_files)

header_col, status_col = st.columns([2.3, 1])
with header_col:
    st.subheader(f"Seçili hasta: {selected_patient}")
    st.markdown(
        '<div class="section-note">Hasta verisi, işaretleme ve kayıt işlemleri</div>',
        unsafe_allow_html=True,
    )
with status_col:
    st.metric("Kayıtlı değerlendirme", saved_count)

metric1, metric2, metric3 = st.columns(3)
metric1.metric("Hasta", f"{st.session_state.patient_index + 1} / {len(patients)}")
metric2.metric("SSA kesiti", total_slices)
metric3.metric("Tüm kesit", len(all_dcm_files))

with st.expander("📜 Bu hastanın geçmiş etiketleri", expanded=False):
    if not patient_history.empty:
        display_cols = [
            c for c in ["doctor_name", "timestamp", "slice_idx", "basal_status", "notes"]
            if c in patient_history.columns
        ]
        st.dataframe(
            patient_history[display_cols],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("Bu hasta için henüz kayıtlı bir değerlendirme bulunmuyor.")


# =============================================================================
# SSA GRID: CLICK A THUMBNAIL TO OPEN IT FOR EDITING
# =============================================================================
st.markdown("## 🧩 SSA kesitleri")
st.caption("Bir kesit kartındaki “Bu kesiti düzenle” düğmesine basarak büyük görüntüleme alanında açın.")

if not ssa_files:
    st.warning(f"{selected_patient} için SSA DICOM kesiti bulunamadı.")
    st.info("Klasör yapısının hasta/ssa_slices/*.dcm biçiminde olduğunu kontrol edin.")
    st.stop()

selection_key = f"selected_ssa_slice::{selected_patient}"
if selection_key not in st.session_state:
    st.session_state[selection_key] = 0
selected_slice_idx = int(st.session_state[selection_key])
selected_slice_idx = max(0, min(selected_slice_idx, len(ssa_files) - 1))

grid_columns = st.selectbox(
    "Izgara sütun sayısı",
    options=[2, 3, 4, 5, 6],
    index=2,
    key="ssa_grid_columns",
    help="Ekranınıza göre kesit kartlarının genişliğini ayarlayın.",
)

for start in range(0, len(ssa_files), grid_columns):
    row_files = ssa_files[start : start + grid_columns]
    columns = st.columns(grid_columns, gap="medium")
    for offset, dcm_path in enumerate(row_files):
        idx = start + offset
        with columns[offset]:
            with st.container(border=True):
                image = dcm_to_image(dcm_path)
                if image is not None:
                    st.image(image, use_container_width=True)
                else:
                    st.warning("Görüntü okunamadı.")
                if idx == selected_slice_idx:
                    st.success(f"✓ Kesit {idx + 1} · Düzenleniyor")
                else:
                    st.markdown(f"**Kesit {idx + 1}**")
                st.caption(dcm_path.name)
                if st.button(
                    "Düzenle" if idx != selected_slice_idx else "Seçili kesit",
                    key=f"choose_ssa_{selected_patient}_{idx}",
                    use_container_width=True,
                    disabled=(idx == selected_slice_idx),
                ):
                    st.session_state[selection_key] = idx
                    st.rerun()


# =============================================================================
# SELECTED SLICE EDITOR
# =============================================================================
st.divider()
st.markdown("## 🎯 Kesit düzenleme")
st.caption("Seçili görüntü üzerine tıklayarak üç işaretleme noktası ekleyin.")

selected_slice_idx = int(st.session_state[selection_key])
selected_slice_idx = max(0, min(selected_slice_idx, len(ssa_files) - 1))
selected_dcm = ssa_files[selected_slice_idx]
points_key = patient_point_key(selected_patient, selected_slice_idx)
if points_key not in st.session_state:
    st.session_state[points_key] = []

left, right = st.columns([1.65, 1], gap="large")

with left:
    st.markdown(f"### Kesit {selected_slice_idx + 1} / {len(ssa_files)}")
    st.caption(selected_dcm.name)

    mark_img = dcm_to_image(selected_dcm)
    if mark_img is not None:
        draw_arr = np.array(mark_img.convert("RGB"))
        draw_img = Image.fromarray(draw_arr)
        draw = ImageDraw.Draw(draw_img)
        for point_no, (px, py) in enumerate(st.session_state[points_key], start=1):
            radius = max(4, round(min(draw_img.size) * 0.012))
            draw.ellipse((px - radius, py - radius, px + radius, py + radius), fill=(245, 65, 75), outline=(255, 255, 255), width=2)
            draw.text((px + radius + 3, py - radius - 3), str(point_no), fill=(255, 255, 255), stroke_width=2, stroke_fill=(20, 20, 20))

        # Use a stable key per patient and slice to retain the click-widget state.
        click_version_key = f"image_click_version::{selected_patient}::{selected_slice_idx}"
        if click_version_key not in st.session_state:
            st.session_state[click_version_key] = 0
        coords = streamlit_image_coordinates(
            draw_img,
            key=f"image_click::{selected_patient}::{selected_slice_idx}::{st.session_state[click_version_key]}",
        )
        if coords is not None:
            new_point = (int(coords["x"]), int(coords["y"]))
            current_points = st.session_state[points_key]
            if not current_points or current_points[-1] != new_point:
                if len(current_points) >= 3:
                    st.session_state[points_key] = []
                    current_points = st.session_state[points_key]
                current_points.append(new_point)
                st.rerun()
    else:
        st.error("Seçilen DICOM görüntüsü okunamadı.")

with right:
    st.markdown("### İşaretleme durumu")
    current_points = st.session_state[points_key]
    for i in range(3):
        if i < len(current_points):
            x, y = current_points[i]
            st.success(f"Nokta {i + 1} · x={x}, y={y}")
        else:
            st.info(f"Nokta {i + 1} · Bekleniyor")

    st.progress(len(current_points) / 3, text=f"{len(current_points)} / 3 nokta işaretlendi")
    if st.button("↺ Noktaları temizle", key=f"clear_points_{selected_patient}_{selected_slice_idx}", use_container_width=True):
        st.session_state[points_key] = []
        click_version_key = f"image_click_version::{selected_patient}::{selected_slice_idx}"
        st.session_state[click_version_key] = st.session_state.get(click_version_key, 0) + 1
        st.rerun()

    st.markdown("---")
    st.markdown("### Klinik değerlendirme")
    with st.form(f"annotation_form::{selected_patient}::{selected_slice_idx}"):
        basal_options = [
            "Normal (0)",
            "Sıkışmış / Compressed (1)",
            "Yok / Absent (2)",
        ]
        basal_status = st.selectbox("Bazal sistern durumu", basal_options)
        notes = st.text_area(
            "Klinik notlar / açıklamalar",
            placeholder="İsteğe bağlı klinik notunuzu girin...",
            height=130,
        )
        submitted = st.form_submit_button(
            "💾 İşaretlemeyi kaydet ve HF'ye gönder",
            use_container_width=True,
            type="primary",
        )

    if submitted:
        points = st.session_state[points_key]
        if len(points) != 3:
            st.error("Kaydetmeden önce görüntü üzerinde tam olarak 3 nokta işaretlemelisiniz.")
        else:
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
                "timestamp": str(pd.Timestamp.now()),
            }

            if READS_CSV_PATH.exists():
                try:
                    df_reads = pd.read_csv(READS_CSV_PATH)
                    if "name" in df_reads.columns:
                        patient_meta = df_reads[df_reads["name"].astype(str) == str(selected_patient)]
                        if not patient_meta.empty:
                            new_record.update(patient_meta.iloc[0].to_dict())
                except Exception as exc:
                    st.warning(f"Ek hasta metaverisi okunamadı: {exc}")

            df_new = pd.DataFrame([new_record])
            local_csv = ANNOTATION_FILE
            df_old = safe_read_csv(local_csv)

            if not df_old.empty:
                # Ensure older CSV files gain newly introduced columns without losing old data.
                for col in new_record:
                    if col not in df_old.columns:
                        df_old[col] = None
                mask = (
                    df_old["patient_id"].astype(str).eq(str(selected_patient))
                    & pd.to_numeric(df_old["slice_idx"], errors="coerce").eq(selected_slice_idx)
                )
                if mask.any():
                    for col, value in new_record.items():
                        df_old.loc[mask, col] = value
                    df_combined = df_old
                else:
                    df_combined = pd.concat([df_old, df_new], ignore_index=True)
            else:
                df_combined = df_new

            try:
                df_combined.to_csv(local_csv, index=False)
                st.success("Kayıt yerel CSV dosyasına yazıldı.")
            except Exception as exc:
                st.error(f"CSV dosyasına kayıt yapılamadı: {exc}")
                st.stop()

            if not HF_TOKEN:
                st.warning(
                    "Yerel kayıt tamamlandı ancak HF_TOKEN tanımlı olmadığı için Hugging Face'e "
                    "yüklenmedi. .streamlit/secrets.toml dosyasına HF_TOKEN ekleyin."
                )
            else:
                try:
                    api = HfApi(token=HF_TOKEN)
                    api.upload_file(
                        path_or_fileobj=str(local_csv),
                        path_in_repo=ANNOTATION_FILE.name,
                        repo_id=HF_REPO_ID,
                        repo_type="dataset",
                        commit_message=f"Doktor notu eklendi: {selected_patient}",
                    )
                    st.success("Kayıt Hugging Face'e başarıyla yüklendi.")
                    st.session_state[points_key] = []
                    click_version_key = f"image_click_version::{selected_patient}::{selected_slice_idx}"
                    st.session_state[click_version_key] = st.session_state.get(click_version_key, 0) + 1
                    st.rerun()
                except Exception as exc:
                    st.error(
                        "Kayıt yerel CSV dosyasına yazıldı ancak Hugging Face yüklemesi başarısız oldu. "
                        f"Hata: {exc}"
                    )


# =============================================================================
# ALL SLICES VIEWER
# =============================================================================
st.divider()
with st.expander("🩻 Tüm kesitler gezinme paneli", expanded=False):
    if not all_dcm_files:
        st.warning(f"{selected_patient} için all_slices klasöründe DICOM dosyası bulunamadı.")
    else:
        all_index_key = f"all_slice_index::{selected_patient}"
        if all_index_key not in st.session_state:
            st.session_state[all_index_key] = 0
        st.session_state[all_index_key] = max(
            0, min(st.session_state[all_index_key], len(all_dcm_files) - 1)
        )
        all_idx = st.session_state[all_index_key]

        st.caption(f"{selected_patient} · {len(all_dcm_files)} toplam kesit")
        all_prev, all_info, all_next = st.columns([1, 2, 1])
        with all_prev:
            if st.button("← Önceki kesit", key=f"all_prev_{selected_patient}", use_container_width=True, disabled=all_idx == 0):
                st.session_state[all_index_key] -= 1
                st.rerun()
        with all_info:
            st.markdown(
                f"<div style='text-align:center;padding:.55rem 0;font-weight:700'>"
                f"Kesit {all_idx + 1} / {len(all_dcm_files)}</div>",
                unsafe_allow_html=True,
            )
        with all_next:
            if st.button("Sonraki kesit →", key=f"all_next_{selected_patient}", use_container_width=True, disabled=all_idx >= len(all_dcm_files) - 1):
                st.session_state[all_index_key] += 1
                st.rerun()

        all_image = dcm_to_image(all_dcm_files[all_idx])
        if all_image is not None:
            image_col1, image_col2, image_col3 = st.columns([1, 2, 1])
            with image_col2:
                st.image(all_image, caption=all_dcm_files[all_idx].name, use_container_width=True)

st.caption("TBI İnceleme Paneli · Görsel düzenleme, mevcut iş akışını koruyacak şekilde tasarlanmıştır.")
