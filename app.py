"""
TBI / CQ500 — Doktor Etiketleme Paneli
Kesit başına: Midline shift (var/yok) + 3 anatomik nokta (yalnızca "Var" ise) + Bazal sistern durumu.
Akış: Izgara görünümü → kesiti büyüt → işaretle → kaydet.
"""
from __future__ import annotations

import math
import os
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
import streamlit as st
from huggingface_hub import HfApi, hf_hub_download
from PIL import Image, ImageDraw
from streamlit_image_coordinates import streamlit_image_coordinates

# ══════════════════════════════════════════════════════════════════
# AYARLAR
# ══════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="TBI Etiketleme Paneli",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

HF_REPO_ID = "oyailgin/CQ500_export300"
ZIP_FILENAME = "cq500_dual_slices.zip"
ANNOTATION_FILE = "doctor_annotations.csv"

LOCAL_DATA_DIR = Path("extracted_data")
ANNOTATION_PATH = Path(ANNOTATION_FILE)


def _get_token() -> str | None:
    try:
        tok = st.secrets.get("HF_TOKEN")
    except Exception:
        tok = None
    return tok or os.environ.get("HF_TOKEN")


HF_TOKEN = _get_token()

WINDOW_PRESETS = {
    "Beyin (40 / 80)": (40, 80),
    "Subdural (75 / 215)": (75, 215),
    "İnme (40 / 40)": (40, 40),
    "Kemik (600 / 2800)": (600, 2800),
    "Özel": None,
}

MIDLINE_OPTIONS = ["Yok (0)", "Var (1)"]
MIDLINE_NO, MIDLINE_YES = MIDLINE_OPTIONS
BASAL_OPTIONS = ["Normal (0)", "Sıkışmış / Compressed (1)", "Yok / Absent (2)"]

STEPS = [
    ("Falx — Ön", "Orta hattın ön (anterior) ucu"),
    ("Falx — Arka", "Orta hattın arka (posterior) ucu"),
    ("Kayan Yapı", "Orta hattan sapmış yapı"),
]
POINT_RGB = [(239, 68, 68), (34, 197, 94), (59, 130, 246)]
POINT_HEX = ["#ef4444", "#22c55e", "#3b82f6"]

CORE_COLS = [
    "doctor_name", "patient_id", "slice_source", "slice_file", "slice_idx",
    "midline_shift_status",
    "falx_ant_x", "falx_ant_y", "falx_post_x", "falx_post_y",
    "shift_point_x", "shift_point_y", "basal_status", "notes", "timestamp",
]

# ══════════════════════════════════════════════════════════════════
# STİL
# ══════════════════════════════════════════════════════════════════
st.markdown(
    """
<style>
.block-container {padding-top: 1.4rem; max-width: 1500px;}
#MainMenu, footer {visibility: hidden;}

.hero {
    padding: 1.1rem 1.4rem; border-radius: 16px; margin-bottom: 1rem;
    background: linear-gradient(120deg, rgba(99,102,241,.18), rgba(14,165,233,.14));
    border: 1px solid rgba(128,128,128,.25);
}
.hero h1 {margin: 0; font-size: 1.55rem; padding: 0;}
.hero p  {margin: .25rem 0 0 0; opacity: .75; font-size: .92rem;}

.chip {
    display: inline-block; padding: .15rem .65rem; border-radius: 999px;
    font-size: .78rem; margin-right: .4rem; border: 1px solid rgba(128,128,128,.35);
    background: rgba(128,128,128,.10);
}
.chip.ok  {background: rgba(34,197,94,.15);  border-color: rgba(34,197,94,.5);}
.chip.warn{background: rgba(234,179,8,.15);  border-color: rgba(234,179,8,.5);}
.chip.red {background: rgba(239,68,68,.15);  border-color: rgba(239,68,68,.5);}

.step {
    display: flex; gap: .75rem; align-items: center; padding: .55rem .75rem;
    border-radius: 12px; margin-bottom: .45rem;
    border: 1px solid rgba(128,128,128,.22);
}
.step small {opacity: .7;}
.step.todo  {opacity: .5;}
.step.active{border-color: rgba(99,102,241,.9); background: rgba(99,102,241,.10);}
.step.done  {background: rgba(34,197,94,.08);}
.step .dot {
    width: 28px; height: 28px; border-radius: 50%; flex: none;
    display: flex; align-items: center; justify-content: center;
    color: white; font-weight: 700; font-size: .85rem; background: var(--c);
}
</style>
""",
    unsafe_allow_html=True,
)

# ══════════════════════════════════════════════════════════════════
# VERİ: İNDİRME & KEŞİF
# ══════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="📥 Veri seti Hugging Face'ten indiriliyor…")
def ensure_data() -> str | None:
    """Zip'i yalnızca bir kez indirip çıkarır. Hata varsa mesajı döner."""
    marker = LOCAL_DATA_DIR / ".ready"
    if marker.exists():
        return None
    try:
        LOCAL_DATA_DIR.mkdir(parents=True, exist_ok=True)
        zip_path = hf_hub_download(
            repo_id=HF_REPO_ID, filename=ZIP_FILENAME, repo_type="dataset", token=HF_TOKEN
        )
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(LOCAL_DATA_DIR)
        marker.touch()
        return None
    except Exception as e:  # noqa: BLE001
        return str(e)


@st.cache_resource(show_spinner=False)
def pull_annotations_once() -> None:
    """Uygulama ilk açıldığında Hub'daki güncel annotation dosyasını getirir."""
    if ANNOTATION_PATH.exists():
        return
    try:
        hf_hub_download(
            repo_id=HF_REPO_ID, filename=ANNOTATION_FILE, repo_type="dataset",
            token=HF_TOKEN, local_dir=".",
        )
    except Exception:  # noqa: BLE001  (dosya henüz yok olabilir)
        pass


def _has_dicom(d: Path) -> bool:
    return next(d.rglob("*.dcm"), None) is not None


@st.cache_data(show_spinner=False)
def discover_patients(base: str) -> tuple[str, list[str]]:
    """Zip içindeki klasör yapısı ne olursa olsun hasta klasörlerini bulur."""
    base_p = Path(base)
    skip = ("_", ".")
    top = sorted(p for p in base_p.iterdir() if p.is_dir() and not p.name.startswith(skip))
    for root in [base_p, *top]:
        pats = sorted(
            p.name for p in root.iterdir()
            if p.is_dir() and not p.name.startswith(skip) and _has_dicom(p)
        )
        if pats:
            return str(root), pats
    return str(base_p), []


def _natural_key(p: Path):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", p.name)]


@st.cache_data(show_spinner=False)
def list_slices(root: str, patient: str, source: str) -> list[str]:
    pf = Path(root) / patient
    sub = pf / ("ssa_slices" if source == "ssa" else "all_slices")
    folder = sub if sub.exists() else pf
    return [str(p) for p in sorted(folder.glob("*.dcm"), key=_natural_key)]


@st.cache_data(show_spinner=False)
def load_reads_meta() -> pd.DataFrame:
    for p in LOCAL_DATA_DIR.rglob("*reads*.csv"):
        try:
            return pd.read_csv(p)
        except Exception:  # noqa: BLE001
            continue
    return pd.DataFrame()


def patient_meta(patient: str) -> dict:
    df = load_reads_meta()
    if df.empty or "name" not in df.columns:
        return {}
    row = df[df["name"].astype(str) == str(patient)]
    return row.iloc[0].to_dict() if not row.empty else {}


# ══════════════════════════════════════════════════════════════════
# DICOM
# ══════════════════════════════════════════════════════════════════
@st.cache_data(max_entries=256, show_spinner=False)
def load_slice(path: str) -> tuple[np.ndarray, float | None]:
    """HU dizisi + piksel aralığı (mm). Pencereleme ayrı yapılır → ucuz."""
    ds = pydicom.dcmread(path)
    arr = ds.pixel_array.astype(np.float32)
    arr = arr * float(getattr(ds, "RescaleSlope", 1)) + float(getattr(ds, "RescaleIntercept", 0))
    ps = getattr(ds, "PixelSpacing", None)
    return arr, (float(ps[0]) if ps else None)


def apply_window(arr: np.ndarray, center: float, width: float) -> Image.Image:
    lo, hi = center - width / 2.0, center + width / 2.0
    out = np.clip((arr - lo) / max(hi - lo, 1e-6), 0, 1) * 255.0
    return Image.fromarray(out.astype(np.uint8))


@st.cache_data(max_entries=600, show_spinner=False)
def thumbnail(path: str, center: float, width: float, size: int = 360) -> Image.Image | None:
    try:
        arr, _ = load_slice(path)
        img = apply_window(arr, center, width)
        img.thumbnail((size, size))
        return img
    except Exception:  # noqa: BLE001
        return None


# ══════════════════════════════════════════════════════════════════
# ANNOTATION DEPOSU
# ══════════════════════════════════════════════════════════════════
def annotations_df() -> pd.DataFrame:
    df = pd.DataFrame()
    if ANNOTATION_PATH.exists():
        try:
            df = pd.read_csv(ANNOTATION_PATH)
        except Exception:  # noqa: BLE001
            df = pd.DataFrame()
    for c in CORE_COLS:
        if c not in df.columns:
            df[c] = None
    return df


def find_existing(df: pd.DataFrame, doctor: str, patient: str, source: str, fname: str):
    if df.empty:
        return None
    m = (
        (df["doctor_name"].astype(str) == doctor)
        & (df["patient_id"].astype(str) == patient)
        & (df["slice_source"].astype(str) == source)
        & (df["slice_file"].astype(str) == fname)
    )
    return df[m].iloc[0] if m.any() else None


def choice_from_row(row, col: str, options: list[str]) -> str | None:
    if row is None:
        return None
    v = row[col]
    return v if isinstance(v, str) and v in options else None


def points_from_row(row) -> list[tuple[float, float]]:
    try:
        keys = [("falx_ant_x", "falx_ant_y"), ("falx_post_x", "falx_post_y"),
                ("shift_point_x", "shift_point_y")]
        pts = [(float(row[a]), float(row[b])) for a, b in keys]
        return [] if any(math.isnan(v) for p in pts for v in p) else pts
    except Exception:  # noqa: BLE001
        return []


def shift_metrics(pts, spacing):
    """Kayan noktanın falx doğrusuna dik uzaklığı (piksel / mm)."""
    (ax, ay), (px, py), (sx, sy) = pts
    dx, dy = px - ax, py - ay
    length2 = dx * dx + dy * dy
    if length2 == 0:
        return None
    t = ((sx - ax) * dx + (sy - ay) * dy) / length2
    fx, fy = ax + t * dx, ay + t * dy
    dist = math.hypot(sx - fx, sy - fy)
    return {"foot": (fx, fy), "px": dist, "mm": dist * spacing if spacing else None}


# ══════════════════════════════════════════════════════════════════
# CALLBACK'LER
# ══════════════════════════════════════════════════════════════════
def reset_slice():
    """Hasta / kaynak değişince ızgaraya dön."""
    st.session_state.cur_slice = 1
    st.session_state.detail_open = False


def go_patient(patients: list[str], delta: int):
    cur = patients.index(st.session_state.patient)
    st.session_state.patient = patients[(cur + delta) % len(patients)]
    reset_slice()


def go_next_pending(patients: list[str], done: set[str]):
    cur = patients.index(st.session_state.patient)
    for step in range(1, len(patients) + 1):
        cand = patients[(cur + step) % len(patients)]
        if cand not in done:
            st.session_state.patient = cand
            reset_slice()
            return
    st.toast("🎉 Tüm hastalar için en az bir kayıt var.")


def open_slice(slice_no: int):
    st.session_state.cur_slice = slice_no
    st.session_state.detail_open = True


def close_detail(page_key: str, per_page: int):
    st.session_state.detail_open = False
    st.session_state[page_key] = (st.session_state.cur_slice - 1) // per_page + 1


def step_slice(delta: int, n: int):
    st.session_state.cur_slice = min(max(st.session_state.cur_slice + delta, 1), n)


def sync_slider():
    st.session_state.cur_slice = st.session_state.slice_slider


def undo_point(key: str):
    if st.session_state.get(key):
        st.session_state[key].pop()


def clear_points(key: str):
    st.session_state[key] = []


def save_annotation(ctx: dict):
    pts = st.session_state.get(ctx["pts_key"], [])
    doctor = ctx["doctor"].strip()
    midline = st.session_state.get(ctx["midline_key"])
    basal = st.session_state.get(ctx["basal_key"])
    has_shift = midline == MIDLINE_YES

    if not doctor or midline is None or basal is None or (has_shift and len(pts) != 3):
        st.session_state.flash = ("error", "Zorunlu alanlar eksik.")
        return

    m = shift_metrics(pts, ctx["spacing"]) if has_shift else None
    record = {
        "doctor_name": doctor,
        "patient_id": ctx["patient"],
        "slice_source": ctx["source"],
        "slice_file": ctx["fname"],
        "slice_idx": ctx["idx"],
        "img_w": ctx["img_w"],
        "img_h": ctx["img_h"],
        "pixel_spacing_mm": ctx["spacing"],
        "midline_shift_status": midline,
        # Midline "Yok" ise nokta alanları bilinçli olarak boş bırakılır
        "falx_ant_x": pts[0][0] if has_shift else None,
        "falx_ant_y": pts[0][1] if has_shift else None,
        "falx_post_x": pts[1][0] if has_shift else None,
        "falx_post_y": pts[1][1] if has_shift else None,
        "shift_point_x": pts[2][0] if has_shift else None,
        "shift_point_y": pts[2][1] if has_shift else None,
        "midline_shift_px": round(m["px"], 2) if m else None,
        "midline_shift_mm": round(m["mm"], 2) if m and m["mm"] is not None else None,
        "basal_status": basal,
        "notes": st.session_state[ctx["notes_key"]],
        "timestamp": str(pd.Timestamp.now()),
    }
    for k, v in patient_meta(ctx["patient"]).items():
        record.setdefault(k, v)

    # Başka doktorların kayıtlarını ezmemek için önce Hub'daki en güncel dosyayı çek
    try:
        hf_hub_download(
            repo_id=HF_REPO_ID, filename=ANNOTATION_FILE, repo_type="dataset",
            token=HF_TOKEN, local_dir=".", force_download=True,
        )
    except Exception:  # noqa: BLE001
        pass

    df_old = annotations_df()
    for c in record:
        if c not in df_old.columns:
            df_old[c] = None
    same = (
        (df_old["doctor_name"].astype(str) == doctor)
        & (df_old["patient_id"].astype(str) == ctx["patient"])
        & (df_old["slice_source"].astype(str) == ctx["source"])
        & (df_old["slice_file"].astype(str) == ctx["fname"])
    )
    df_all = pd.concat([df_old[~same], pd.DataFrame([record])], ignore_index=True)
    df_all.to_csv(ANNOTATION_PATH, index=False)

    try:
        HfApi(token=HF_TOKEN).upload_file(
            path_or_fileobj=str(ANNOTATION_PATH),
            path_in_repo=ANNOTATION_FILE,
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            commit_message=f"{doctor}: {ctx['patient']} / {ctx['fname']}",
        )
        st.session_state.flash = ("success", f"Kaydedildi ve Hub'a gönderildi: {ctx['patient']} · {ctx['fname']}")
    except Exception as e:  # noqa: BLE001
        st.session_state.flash = ("warning", f"Yerel kayıt yapıldı, Hub'a yüklenemedi: {e}")

    if ctx["advance"] and st.session_state.cur_slice < ctx["n"]:
        st.session_state.cur_slice += 1


def save_batch_annotations(ctx: dict):
    """Midline (SSA) ve bazal (tüm kesitler) seçimlerini tek seferde kaydet."""
    doctor = ctx["doctor"].strip()
    if not doctor:
        st.session_state.flash = ("error", "Doktor adı zorunludur.")
        return

    ssa_by_name = {Path(path).name: i for i, path in enumerate(ctx["ssa_files"])}
    records = []
    now = str(pd.Timestamp.now())
    for idx, path in enumerate(ctx["all_files"]):
        fname = Path(path).name
        midline = None
        pts = []
        source = "all"
        slice_idx = idx
        if fname in ssa_by_name:
            source = "ssa"
            slice_idx = ssa_by_name[fname]
            midline = st.session_state[ctx["midline_key"](fname)]
            pts = st.session_state[ctx["pts_key"](fname)]

        record = {
            "doctor_name": doctor,
            "patient_id": ctx["patient"],
            "slice_source": source,
            "slice_file": fname,
            "slice_idx": slice_idx,
            "midline_shift_status": midline,
            "falx_ant_x": pts[0][0] if len(pts) == 3 else None,
            "falx_ant_y": pts[0][1] if len(pts) == 3 else None,
            "falx_post_x": pts[1][0] if len(pts) == 3 else None,
            "falx_post_y": pts[1][1] if len(pts) == 3 else None,
            "shift_point_x": pts[2][0] if len(pts) == 3 else None,
            "shift_point_y": pts[2][1] if len(pts) == 3 else None,
            "basal_status": st.session_state[ctx["basal_key"](fname)],
            "notes": st.session_state[ctx["notes_key"]],
            "timestamp": now,
        }
        record.update({k: v for k, v in patient_meta(ctx["patient"]).items() if k not in record})
        records.append(record)

    all_names = {Path(path).name for path in ctx["all_files"]}
    for idx, path in enumerate(ctx["ssa_files"]):
        fname = Path(path).name
        if fname in all_names:
            continue
        pts = st.session_state[ctx["pts_key"](fname)]
        record = {
            "doctor_name": doctor,
            "patient_id": ctx["patient"],
            "slice_source": "ssa",
            "slice_file": fname,
            "slice_idx": idx,
            "midline_shift_status": st.session_state[ctx["midline_key"](fname)],
            "falx_ant_x": pts[0][0] if len(pts) == 3 else None,
            "falx_ant_y": pts[0][1] if len(pts) == 3 else None,
            "falx_post_x": pts[1][0] if len(pts) == 3 else None,
            "falx_post_y": pts[1][1] if len(pts) == 3 else None,
            "shift_point_x": pts[2][0] if len(pts) == 3 else None,
            "shift_point_y": pts[2][1] if len(pts) == 3 else None,
            "basal_status": None,
            "notes": st.session_state[ctx["notes_key"]],
            "timestamp": now,
        }
        record.update({k: v for k, v in patient_meta(ctx["patient"]).items() if k not in record})
        records.append(record)

    try:
        hf_hub_download(
            repo_id=HF_REPO_ID, filename=ANNOTATION_FILE, repo_type="dataset",
            token=HF_TOKEN, local_dir=".", force_download=True,
        )
    except Exception:  # noqa: BLE001
        pass

    df_old = annotations_df()
    for record in records:
        for col in record:
            if col not in df_old.columns:
                df_old[col] = None
    saved_names = all_names | {Path(path).name for path in ctx["ssa_files"]}
    replace = (
        (df_old["doctor_name"].astype(str) == doctor)
        & (df_old["patient_id"].astype(str) == ctx["patient"])
        & (df_old["slice_file"].astype(str).isin(saved_names))
    )
    df_all = pd.concat([df_old[~replace], pd.DataFrame(records)], ignore_index=True)
    df_all.to_csv(ANNOTATION_PATH, index=False)

    try:
        HfApi(token=HF_TOKEN).upload_file(
            path_or_fileobj=str(ANNOTATION_PATH), path_in_repo=ANNOTATION_FILE,
            repo_id=HF_REPO_ID, repo_type="dataset",
            commit_message=f"{doctor}: {ctx['patient']} toplu kesit etiketleri",
        )
        st.session_state.flash = ("success", f"{ctx['patient']} için tüm seçimler kaydedildi ve Hub'a gönderildi.")
    except Exception as e:  # noqa: BLE001
        st.session_state.flash = ("warning", f"Yerel kayıt yapıldı, Hub'a yüklenemedi: {e}")


# ══════════════════════════════════════════════════════════════════
# GÖRSELLEŞTİRME
# ══════════════════════════════════════════════════════════════════
def draw_overlay(img: Image.Image, pts, scale: float, metrics) -> Image.Image:
    rgb = img.convert("RGB")
    rgb = rgb.resize((round(rgb.width * scale), round(rgb.height * scale)), Image.LANCZOS)
    d = ImageDraw.Draw(rgb)
    sp = [(x * scale, y * scale) for x, y in pts]

    if len(sp) >= 2:
        d.line([sp[0], sp[1]], fill=(250, 204, 21), width=2)
    if len(sp) == 3 and metrics:
        foot = (metrics["foot"][0] * scale, metrics["foot"][1] * scale)
        d.line([sp[2], foot], fill=(34, 211, 238), width=2)

    for i, (x, y) in enumerate(sp):
        r = 7
        d.ellipse([x - r, y - r, x + r, y + r], fill=POINT_RGB[i], outline=(255, 255, 255), width=2)
        d.text((x + 11, y - 7), str(i + 1), fill=POINT_RGB[i])
    return rgb


def stepper_html(n_pts: int) -> str:
    out = []
    for i, (title, desc) in enumerate(STEPS):
        state = "done" if i < n_pts else ("active" if i == n_pts else "todo")
        mark = "✓" if state == "done" else str(i + 1)
        out.append(
            f'<div class="step {state}"><span class="dot" style="--c:{POINT_HEX[i]}">{mark}</span>'
            f"<div><b>{title}</b><br><small>{desc}</small></div></div>"
        )
    return "".join(out)


# ══════════════════════════════════════════════════════════════════
# GÖRÜNÜM 1: IZGARA
# ══════════════════════════════════════════════════════════════════
def render_grid(files, patient, source, doctor, df_ann, win, grid_cols):
    n = len(files)
    mine = df_ann[
        (df_ann["doctor_name"].astype(str) == doctor) & (df_ann["patient_id"].astype(str) == patient)
    ]
    status = dict(zip(mine["slice_file"].astype(str), mine["midline_shift_status"]))
    marked = sum(Path(f).name in status for f in files)

    per_page = grid_cols * 3
    pages = max(1, math.ceil(n / per_page))
    page_key = f"grid_page::{patient}::{source}"

    head_l, head_r = st.columns([4, 1])
    head_l.markdown(
        f'<span class="chip">{n} kesit</span>'
        f'<span class="chip {"ok" if marked == n else "warn"}">{marked}/{n} işaretli</span>'
        f"&nbsp; <small>Bir kesiti büyütmek ve işaretlemek için <b>🔍 Aç</b> düğmesine basın.</small>",
        unsafe_allow_html=True,
    )
    if pages > 1:
        page = head_r.number_input("Sayfa", 1, pages, 1, key=page_key)
    else:
        page = 1

    start = (page - 1) * per_page
    cols = st.columns(grid_cols)
    for i, f in enumerate(files[start:start + per_page]):
        gi = start + i
        name = Path(f).name
        with cols[i % grid_cols]:
            with st.container(border=True):
                img = thumbnail(f, *win)
                if img is not None:
                    st.image(img, use_container_width=True)
                else:
                    st.error("Okunamadı")

                if name in status:
                    ms = status[name]
                    tag = "Shift: Var" if ms == MIDLINE_YES else ("Shift: Yok" if ms == MIDLINE_NO else "kayıtlı")
                    st.caption(f"Kesit {gi + 1} · ✓ {tag}")
                else:
                    st.caption(f"Kesit {gi + 1}")
                st.button("🔍 Aç", key=f"open::{patient}::{source}::{gi}", use_container_width=True,
                          on_click=open_slice, args=(gi + 1,))


# ══════════════════════════════════════════════════════════════════
# GÖRÜNÜM 2: BÜYÜK KESİT + İŞARETLEME
# ══════════════════════════════════════════════════════════════════
def render_detail(files, patient, source, doctor, df_ann, win, disp_w, advance, grid_cols):
    n = len(files)
    cur = st.session_state.cur_slice
    idx = cur - 1
    path = files[idx]
    fname = Path(path).name
    page_key = f"grid_page::{patient}::{source}"

    # ── üst çubuk: ızgaraya dön + kesit gezinme ──
    h_back, h_prev, h_mid, h_next = st.columns([2, 1, 7, 1])
    h_back.button("⬅️ Izgaraya dön", use_container_width=True,
                  on_click=close_detail, args=(page_key, grid_cols * 3))
    h_prev.button("◀", key="prev_slice", use_container_width=True, disabled=cur <= 1,
                  on_click=step_slice, args=(-1, n))
    if n > 1:
        st.session_state.slice_slider = cur
        h_mid.slider("Kesit", 1, n, key="slice_slider", on_change=sync_slider,
                     label_visibility="collapsed")
    h_next.button("▶", key="next_slice", use_container_width=True, disabled=cur >= n,
                  on_click=step_slice, args=(1, n))

    try:
        arr, spacing = load_slice(path)
    except Exception as e:  # noqa: BLE001
        st.error(f"DICOM okunamadı ({fname}): {e}")
        return
    base_img = apply_window(arr, *win)
    img_w, img_h = base_img.size

    # ── kesit başına durum ──
    state_key = f"{patient}::{source}::{fname}"
    pts_key = f"pts::{state_key}"
    midline_key, basal_key, notes_key = (f"midline::{state_key}", f"basal::{state_key}", f"notes::{state_key}")
    existing = find_existing(df_ann, doctor, patient, source, fname)

    if pts_key not in st.session_state:
        st.session_state[pts_key] = points_from_row(existing) if existing is not None else []
    if midline_key not in st.session_state:
        st.session_state[midline_key] = choice_from_row(existing, "midline_shift_status", MIDLINE_OPTIONS)
    if basal_key not in st.session_state:
        st.session_state[basal_key] = choice_from_row(existing, "basal_status", BASAL_OPTIONS)
    if notes_key not in st.session_state:
        nt = existing["notes"] if existing is not None else ""
        st.session_state[notes_key] = "" if pd.isna(nt) else str(nt)

    pts: list = st.session_state[pts_key]
    midline = st.session_state[midline_key]
    marking_on = midline != MIDLINE_NO          # "Yok" seçiliyse nokta işaretleme kapalı
    has_shift = midline == MIDLINE_YES
    metrics = shift_metrics(pts, spacing) if (has_shift and len(pts) == 3) else None

    left, right = st.columns([3, 2], gap="large")

    # ── büyük görüntü + tıklama ──
    with left:
        scale = disp_w / img_w
        show_pts = pts if marking_on else []
        disp_img = draw_overlay(base_img, show_pts, scale, metrics)
        coords = streamlit_image_coordinates(disp_img, key=f"click::{state_key}")

        if coords is not None:
            last_key = f"lastclick::{state_key}"
            if coords.get("unix_time") != st.session_state.get(last_key):
                st.session_state[last_key] = coords.get("unix_time")
                if marking_on and len(pts) < 3:
                    factor = img_w / (coords.get("width") or disp_img.width)
                    pts.append((round(coords["x"] * factor, 1), round(coords["y"] * factor, 1)))
                    st.rerun()

        saved = '<span class="chip ok">✓ kayıtlı</span>' if existing is not None else \
                '<span class="chip warn">kayıt yok</span>'
        st.markdown(
            f'<span class="chip">{patient}</span><span class="chip">{fname}</span>'
            f'<span class="chip">Kesit {cur}/{n}</span><span class="chip">{img_w}×{img_h}</span>{saved}',
            unsafe_allow_html=True,
        )

    # ── sağ panel ──
    with right:
        with st.container(border=True):
            st.markdown("**🧭 Midline Shift Durumu**")
            st.radio("Midline shift", MIDLINE_OPTIONS, index=None, key=midline_key,
                     horizontal=True, label_visibility="collapsed")

            if midline == MIDLINE_NO:
                st.info("Midline shift yok olarak işaretlendi — nokta işaretlemeye gerek yok.")
            else:
                st.markdown(stepper_html(len(pts)), unsafe_allow_html=True)
                if midline is None:
                    st.caption("Önce midline shift durumunu seçin. Noktaları şimdiden işaretleyebilirsiniz.")
                c1, c2 = st.columns(2)
                c1.button("↩️ Geri al", use_container_width=True, disabled=not pts,
                          on_click=undo_point, args=(pts_key,))
                c2.button("🗑️ Temizle", use_container_width=True, disabled=not pts,
                          on_click=clear_points, args=(pts_key,))
                if metrics:
                    val = f"{metrics['mm']:.1f} mm" if metrics["mm"] is not None else f"{metrics['px']:.0f} px"
                    st.metric("Tahmini midline shift", val,
                              help="Kayan noktanın falx doğrusuna dik uzaklığı (otomatik hesaplanır).")

        with st.container(border=True):
            st.markdown("**📋 Klinik Değerlendirme**")
            st.radio("Bazal sistern durumu", BASAL_OPTIONS, index=None, key=basal_key)
            st.text_area("Klinik notlar", key=notes_key, height=100, placeholder="Gözlem, şüphe, yorum…")

            missing = []
            if not doctor.strip():
                missing.append("doktor adı")
            if midline is None:
                missing.append("midline durumu")
            if has_shift and len(pts) != 3:
                missing.append(f"{3 - len(pts)} nokta")
            if st.session_state[basal_key] is None:
                missing.append("bazal sistern durumu")

            ctx = dict(
                pts_key=pts_key, midline_key=midline_key, basal_key=basal_key, notes_key=notes_key,
                doctor=doctor, patient=patient, source=source, fname=fname, idx=idx, n=n,
                img_w=img_w, img_h=img_h, spacing=spacing, advance=advance,
            )
            st.button(
                "💾 Kaydet" + (" ve sonraki kesite geç" if advance else ""),
                type="primary", use_container_width=True, disabled=bool(missing),
                on_click=save_annotation, args=(ctx,),
            )
            if missing:
                st.caption("Eksik: " + ", ".join(missing))


# ══════════════════════════════════════════════════════════════════
# SEKMELER
# ══════════════════════════════════════════════════════════════════
def render_slices_tab(root, patient, doctor, df_ann, win, disp_w, advance, grid_cols, ssa_n, all_n):
    flash = st.session_state.pop("flash", None)
    if flash:
        {"success": st.success, "warning": st.warning, "error": st.error}[flash[0]](flash[1])

    ssa_files = list_slices(root, patient, "ssa")
    all_files = list_slices(root, patient, "all")
    if not ssa_files and not all_files:
        st.warning("Bu hasta için DICOM kesiti bulunamadı.")
        return
    render_batch_selection(patient, doctor, df_ann, win, disp_w, grid_cols, ssa_files, all_files)


def activate_batch_slice(key: str, fname: str):
    st.session_state[key] = fname


def render_selection_grid(files, patient, section, win, grid_cols, active_key):
    """Kesitlerin tamamını ızgarada göster; seçilen kesiti aşağıda büyüt."""
    cols = st.columns(grid_cols)
    for idx, path in enumerate(files):
        fname = Path(path).name
        with cols[idx % grid_cols]:
            with st.container(border=True):
                img = thumbnail(path, *win)
                if img is not None:
                    st.image(img, use_container_width=True)
                else:
                    st.error("Görüntü okunamadı")
                st.caption(f"Kesit {idx + 1} · {fname}")
                selected = st.session_state.get(active_key) == fname
                st.button(
                    "✓ Seçili kesit" if selected else "🔍 Büyüt ve seç",
                    key=f"{section}-activate::{patient}::{fname}",
                    type="primary" if selected else "secondary",
                    use_container_width=True,
                    on_click=activate_batch_slice,
                    args=(active_key, fname),
                )


def render_batch_selection(patient, doctor, df_ann, win, disp_w, grid_cols, ssa_files, all_files):
    """SSA midline ve tüm kesit bazal etiketlerini tek hasta akışında al."""
    flash = st.session_state.pop("flash", None)
    if flash:
        {"success": st.success, "warning": st.warning, "error": st.error}[flash[0]](flash[1])

    state_base = f"batch::{patient}"
    ssa_names = {Path(path).name for path in ssa_files}
    all_names = {Path(path).name for path in all_files}
    for path in ssa_files:
        fname = Path(path).name
        key = f"{state_base}::midline::{fname}"
        if key not in st.session_state:
            row = find_existing(df_ann, doctor, patient, "ssa", fname)
            st.session_state[key] = choice_from_row(row, "midline_shift_status", MIDLINE_OPTIONS)
        pkey = f"{state_base}::points::{fname}"
        if pkey not in st.session_state:
            row = find_existing(df_ann, doctor, patient, "ssa", fname)
            st.session_state[pkey] = points_from_row(row) if row is not None else []
    for path in all_files:
        fname = Path(path).name
        key = f"{state_base}::basal::{fname}"
        if key not in st.session_state:
            matching = df_ann[
                (df_ann["doctor_name"].astype(str) == doctor)
                & (df_ann["patient_id"].astype(str) == patient)
                & (df_ann["slice_file"].astype(str) == fname)
            ]
            st.session_state[key] = choice_from_row(
                matching.iloc[0] if not matching.empty else None, "basal_status", BASAL_OPTIONS
            )

    mid_active_key = f"{state_base}::active_midline"
    basal_active_key = f"{state_base}::active_basal"
    if st.session_state.get(mid_active_key) not in ssa_names:
        st.session_state[mid_active_key] = Path(ssa_files[0]).name if ssa_files else None
    if st.session_state.get(basal_active_key) not in all_names:
        st.session_state[basal_active_key] = Path(all_files[0]).name if all_files else None

    mid_done = sum(
        st.session_state[f"{state_base}::midline::{fname}"] == MIDLINE_NO
        or (
            st.session_state[f"{state_base}::midline::{fname}"] == MIDLINE_YES
            and len(st.session_state[f"{state_base}::points::{fname}"]) == 3
        )
        for fname in ssa_names
    )
    basal_done = sum(st.session_state[f"{state_base}::basal::{fname}"] in BASAL_OPTIONS for fname in all_names)

    st.markdown("### 🧭 Midline Shift · SSA seçili kesitler")
    st.caption(f"Yalnızca SSA ile belirlenmiş {len(ssa_files)} kesit gösterilir. Her kesitte durumu seçin; 'Var' ise büyütülmüş görüntüde 3 noktayı işaretleyin.")
    st.progress(mid_done / max(len(ssa_files), 1), text=f"Midline tamamlandı: {mid_done}/{len(ssa_files)}")
    render_selection_grid(ssa_files, patient, "midline", win, grid_cols, mid_active_key)

    mid_fname = st.session_state.get(mid_active_key)
    if mid_fname:
        mid_path = next((p for p in ssa_files if Path(p).name == mid_fname), None)
        if mid_path:
            st.markdown(f"#### Midline seçimi · {mid_fname}")
            try:
                arr, spacing = load_slice(mid_path)
                base_img = apply_window(arr, *win)
                img_w, img_h = base_img.size
                mid_key = f"{state_base}::midline::{mid_fname}"
                pts_key = f"{state_base}::points::{mid_fname}"
                status_col, image_col = st.columns([1, 2], gap="large")
                with status_col:
                    st.radio("Midline shift durumu", MIDLINE_OPTIONS, key=mid_key, index=None)
                    if st.session_state[mid_key] == MIDLINE_YES:
                        pts = st.session_state[pts_key]
                        st.markdown(stepper_html(len(pts)), unsafe_allow_html=True)
                        c1, c2 = st.columns(2)
                        c1.button("↩️ Geri al", key=f"undo::{mid_fname}", disabled=not pts,
                                  on_click=undo_point, args=(pts_key,))
                        c2.button("🗑️ Temizle", key=f"clear::{mid_fname}", disabled=not pts,
                                  on_click=clear_points, args=(pts_key,))
                    elif st.session_state[mid_key] == MIDLINE_NO:
                        st.success("Bu SSA kesitte nokta işaretlemesi gerekmiyor.")
                if st.session_state[mid_key] == MIDLINE_YES:
                    with image_col:
                        scale = min(disp_w / img_w, 1.0)
                        metrics = shift_metrics(st.session_state[pts_key], spacing) if len(st.session_state[pts_key]) == 3 else None
                        disp_img = draw_overlay(base_img, st.session_state[pts_key], scale, metrics)
                        coords = streamlit_image_coordinates(disp_img, key=f"batch-click::{patient}::{mid_fname}")
                        if coords is not None:
                            click_key = f"batch-lastclick::{patient}::{mid_fname}"
                            if coords.get("unix_time") != st.session_state.get(click_key):
                                st.session_state[click_key] = coords.get("unix_time")
                                if len(st.session_state[pts_key]) < 3:
                                    factor = img_w / (coords.get("width") or disp_img.width)
                                    st.session_state[pts_key].append((round(coords["x"] * factor, 1), round(coords["y"] * factor, 1)))
                                    st.rerun()
                        st.caption(f"{img_w}×{img_h} px · Noktaları görüntüye tıklayarak sırayla girin.")
                        if metrics:
                            value = f"{metrics['mm']:.1f} mm" if metrics["mm"] is not None else f"{metrics['px']:.0f} px"
                            st.metric("Tahmini midline shift", value)
            except Exception as e:  # noqa: BLE001
                st.error(f"DICOM okunamadı ({mid_fname}): {e}")

    st.markdown("---")
    st.markdown("### 📋 Bazal Sistern · tüm kesitler")
    st.caption(f"Tüm {len(all_files)} kesit aynı anda gösterilir. Bir kesite basınca görüntü büyür; seçim paneli büyütülmüş görüntünün altında açılır.")
    st.progress(basal_done / max(len(all_files), 1), text=f"Bazal sistern tamamlandı: {basal_done}/{len(all_files)}")
    render_selection_grid(all_files, patient, "basal", win, grid_cols, basal_active_key)

    basal_fname = st.session_state.get(basal_active_key)
    if basal_fname:
        basal_path = next((p for p in all_files if Path(p).name == basal_fname), None)
        if basal_path:
            st.markdown(f"#### Bazal sistern seçimi · {basal_fname}")
            try:
                arr, _ = load_slice(basal_path)
                img = apply_window(arr, *win)
                img.thumbnail((disp_w, disp_w))
                st.image(img, caption=basal_fname, width=img.width)
            except Exception as e:  # noqa: BLE001
                st.error(f"DICOM okunamadı ({basal_fname}): {e}")
            st.radio("Bazal sistern durumu", BASAL_OPTIONS,
                     key=f"{state_base}::basal::{basal_fname}", index=None, horizontal=True)

    all_complete = bool(ssa_files and all_files) and mid_done == len(ssa_files) and basal_done == len(all_files)
    st.markdown("---")
    if all_complete:
        st.markdown("### 📝 Not ve kayıt")
        notes_key = f"{state_base}::notes"
        if notes_key not in st.session_state:
            prior_notes = df_ann[
                (df_ann["doctor_name"].astype(str) == doctor)
                & (df_ann["patient_id"].astype(str) == patient)
            ]["notes"]
            note = prior_notes.dropna().astype(str)
            st.session_state[notes_key] = note.iloc[-1] if not note.empty else ""
        st.text_area("Tüm kesitler için ortak not", key=notes_key, height=110,
                     placeholder="Gözlem, şüphe, yorum…")
        batch_ctx = {
            "patient": patient,
            "doctor": doctor,
            "ssa_files": ssa_files,
            "all_files": all_files,
            "midline_key": lambda fname: f"{state_base}::midline::{fname}",
            "pts_key": lambda fname: f"{state_base}::points::{fname}",
            "basal_key": lambda fname: f"{state_base}::basal::{fname}",
            "notes_key": notes_key,
        }
        st.button("💾 Tüm seçimleri kaydet", type="primary", use_container_width=True,
                  disabled=not doctor.strip(), on_click=save_batch_annotations, args=(batch_ctx,))
    else:
        st.info("Not ekleme ve toplu kayıt, SSA midline noktaları/durumları ile tüm kesitlerin bazal sistern seçimleri tamamlandıktan sonra açılır.")


def render_history_tab(patient, df_ann):
    mine = df_ann[df_ann["patient_id"].astype(str) == patient].dropna(how="all", axis=1)
    st.markdown(f"**{patient}** için kayıtlar")
    if mine.empty:
        st.info("Bu hasta için henüz kayıt yok.")
    else:
        cols = [c for c in ["doctor_name", "slice_source", "slice_file", "midline_shift_status",
                            "midline_shift_mm", "midline_shift_px", "basal_status", "notes", "timestamp"]
                if c in mine.columns]
        st.dataframe(mine[cols], use_container_width=True, hide_index=True)

    st.markdown("---")
    valid = df_ann.dropna(subset=["patient_id"])
    c1, c2, c3 = st.columns(3)
    c1.metric("Toplam kayıt", len(valid))
    c2.metric("İşaretlenen hasta", valid["patient_id"].nunique())
    c3.metric("Doktor sayısı", valid["doctor_name"].nunique())
    if not valid.empty:
        st.download_button("⬇️ Tüm kayıtları CSV indir", valid.to_csv(index=False).encode("utf-8"),
                           file_name=ANNOTATION_FILE, mime="text/csv")


# ══════════════════════════════════════════════════════════════════
# ANA AKIŞ
# ══════════════════════════════════════════════════════════════════
st.markdown(
    '<div class="hero"><h1>🧠 TBI / CQ500 Etiketleme Paneli</h1>'
    "<p>Midline shift (var/yok + 3 nokta) · bazal sistern değerlendirmesi · Hugging Face senkronizasyonu</p></div>",
    unsafe_allow_html=True,
)

err = ensure_data()
if err:
    st.error(f"Veri indirilemedi: {err}")
    if st.button("🔄 Yeniden dene"):
        ensure_data.clear()
        st.rerun()
    st.stop()

pull_annotations_once()
root, patients = discover_patients(str(LOCAL_DATA_DIR))
if not patients:
    st.warning("Hasta klasörü bulunamadı. HF repo adını ve zip yapısını kontrol edin.")
    st.stop()

# ── kenar çubuğu ──
with st.sidebar:
    st.markdown("### 👨‍⚕️ Doktor")
    doctor = st.text_input("Doktor adı / ID", value="Dr. Uzman", label_visibility="collapsed")

    df_ann = annotations_df()
    my = df_ann[df_ann["doctor_name"].astype(str) == doctor]
    done_patients = set(my["patient_id"].dropna().astype(str))

    if st.session_state.get("patient") not in patients:
        st.session_state.patient = patients[0]

    st.markdown("### 📂 Hasta")
    p1, p2 = st.columns(2)
    p1.button("◀ Önceki", use_container_width=True, on_click=go_patient, args=(patients, -1))
    p2.button("Sonraki ▶", use_container_width=True, on_click=go_patient, args=(patients, +1))
    st.selectbox(
        "Hasta", patients, key="patient", on_change=reset_slice, label_visibility="collapsed",
        format_func=lambda p: f"{'✅' if p in done_patients else '⬜'}  {p}",
    )
    st.button("⏭️ Sonraki işaretlenmemiş hasta", use_container_width=True,
              on_click=go_next_pending, args=(patients, done_patients))

    st.progress(len(done_patients) / len(patients),
                text=f"İlerleme: {len(done_patients)}/{len(patients)} hasta")

    st.markdown("### 🎚️ Görüntü")
    preset = st.selectbox("Pencere", list(WINDOW_PRESETS), index=0)
    if WINDOW_PRESETS[preset] is None:
        wc = st.slider("Window Level", -200, 1000, 40)
        ww = st.slider("Window Width", 1, 4000, 80)
    else:
        wc, ww = WINDOW_PRESETS[preset]
    grid_cols = st.slider("Izgara sütun sayısı", 3, 6, 4)
    disp_w = st.slider("Büyük görünüm genişliği (px)", 450, 1000, 720, 20)
    advance = st.toggle("Kaydedince sonraki kesite geç", value=True)

patient = st.session_state.patient
ssa_files = list_slices(root, patient, "ssa")
all_files = list_slices(root, patient, "all")
ssa_n = len(ssa_files)
all_n = len(all_files) if all_files != ssa_files or Path(root, patient, "all_slices").exists() else 0

st.markdown(
    f"#### {patient} "
    f'<span class="chip">{patients.index(patient) + 1}/{len(patients)}</span>'
    f'<span class="chip {"ok" if patient in done_patients else "warn"}">'
    f'{"işaretlendi" if patient in done_patients else "bekliyor"}</span>',
    unsafe_allow_html=True,
)

tab_slices, tab_hist = st.tabs(["🩻 Kesitler", "📜 Kayıtlar"])
with tab_slices:
    render_slices_tab(root, patient, doctor, df_ann, (wc, ww), disp_w, advance, grid_cols, ssa_n, all_n)
with tab_hist:
    render_history_tab(patient, df_ann)