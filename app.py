# ══════════════════════════════════════════════════════════════════
# DEĞİŞİKLİK 1 — CSS: <style> bloğunun İÇİNE (</style>'dan hemen önce) ekleyin
# ══════════════════════════════════════════════════════════════════
"""
.sticky-anchor {display: none;}
div[data-testid="stColumn"]:has(.sticky-anchor),
div[data-testid="column"]:has(.sticky-anchor) {
    position: sticky; top: 3.75rem; align-self: flex-start;
}
"""

# ══════════════════════════════════════════════════════════════════
# DEĞİŞİKLİK 2 — Dosyada "def _batch_key" satırından
# "st.button("💾 Tüm seçimleri kaydet" ... )" satırına kadar olan bölümün
# (yani _batch_key, _set_active_slice, _activate_midline_slice,
#  _render_all_slice_cards ve _render_batch_annotation) TAMAMINI
# aşağıdaki kodla değiştirin. render_slices_tab ve diğer her şey aynı kalır.
# ══════════════════════════════════════════════════════════════════

def _batch_key(patient: str, category: str, fname: str) -> str:
    return f"batch::{patient}::{category}::{fname}"


def _set_active_slice(key: str, fname: str):
    st.session_state[key] = fname


@st.cache_data(show_spinner=False)
def slice_header_meta(path: str) -> tuple[float | None, int | None, int | None]:
    """Piksel verisini okumadan spacing / genişlik / yükseklik döner (hızlı)."""
    ds = pydicom.dcmread(path, stop_before_pixels=True)
    ps = getattr(ds, "PixelSpacing", None)
    return (
        float(ps[0]) if ps else None,
        int(getattr(ds, "Columns", 0)) or None,
        int(getattr(ds, "Rows", 0)) or None,
    )


def stepper_inline_html(n_pts: int) -> str:
    """Tek satırlık kompakt nokta adımları (sağ panelde yer kazandırır)."""
    parts = []
    for i, (title, _) in enumerate(STEPS):
        state = "ok" if i < n_pts else ("warn" if i == n_pts else "")
        mark = "✓" if i < n_pts else str(i + 1)
        parts.append(f'<span class="chip {state}" style="border-color:{POINT_HEX[i]}">{mark} {title}</span>')
    return "".join(parts)


def _render_all_slice_cards(files, patient, category, active_key, win, grid_cols, done=frozenset()):
    """Verilen kesitlerin hepsini aynı anda gösterir; tıklanan kesit aktif olur."""
    cols = st.columns(grid_cols)
    for idx, path in enumerate(files):
        fname = Path(path).name
        with cols[idx % grid_cols]:
            with st.container(border=True):
                img = thumbnail(path, *win, size=150)
                if img is None:
                    st.error("Görüntü okunamadı")
                else:
                    st.image(img, use_container_width=True)
                st.caption(f"Kesit {idx + 1}" + (" · ✓" if fname in done else ""))
                is_active = st.session_state.get(active_key) == fname
                st.button(
                    "✓ Seçili" if is_active else "🔍 Büyüt",
                    key=f"batch-open::{patient}::{category}::{idx}",
                    type="primary" if is_active else "secondary",
                    use_container_width=True,
                    on_click=_set_active_slice,
                    args=(active_key, fname),
                )


def _render_batch_annotation(patient, doctor, df_ann, win, disp_w, grid_cols, ssa_files, all_files):
    """Midline: SSA kesitleri + 3 nokta. Bazal sistern: tüm kesitler + tek seçim. En altta not."""
    LIST_COLS = 3  # sol paneldeki küçük resim sütun sayısı

    ssa_names = [Path(path).name for path in ssa_files]
    all_names = [Path(path).name for path in all_files]
    unique_names = list(dict.fromkeys(ssa_names + all_names))

    # ── durum başlangıçları (kayıtlı veri varsa onu yükle) ──
    midline_status_key = f"batch::{patient}::midline-status"
    if midline_status_key not in st.session_state:
        prior_midline = df_ann[
            (df_ann["doctor_name"].astype(str) == doctor)
            & (df_ann["patient_id"].astype(str) == patient)
            & df_ann["midline_shift_status"].astype(str).isin(MIDLINE_OPTIONS)
        ]
        st.session_state[midline_status_key] = (
            prior_midline.iloc[-1]["midline_shift_status"] if not prior_midline.empty else None
        )

    for fname in ssa_names:
        points_key = _batch_key(patient, "points", fname)
        if points_key not in st.session_state:
            saved = find_existing(df_ann, doctor, patient, "ssa", fname)
            if saved is None:
                saved = find_existing(df_ann, doctor, patient, "all", fname)
            st.session_state[points_key] = points_from_row(saved) if saved is not None else []

    basal_key = f"batch::{patient}::basal-status"
    if basal_key not in st.session_state:
        prior_basal = df_ann[
            (df_ann["doctor_name"].astype(str) == doctor)
            & (df_ann["patient_id"].astype(str) == patient)
            & df_ann["basal_status"].astype(str).isin(BASAL_OPTIONS)
        ]
        st.session_state[basal_key] = prior_basal.iloc[-1]["basal_status"] if not prior_basal.empty else None

    mid_active_key = f"batch::{patient}::active-midline"
    basal_active_key = f"batch::{patient}::active-basal"
    if ssa_names and st.session_state.get(mid_active_key) not in ssa_names:
        st.session_state[mid_active_key] = ssa_names[0]
    if all_names and st.session_state.get(basal_active_key) not in all_names:
        st.session_state[basal_active_key] = all_names[0]

    midline_status = st.session_state[midline_status_key]
    done_midline = {
        f for f in ssa_names
        if len(st.session_state[_batch_key(patient, "points", f)]) == 3
    }
    midline_complete = (
        len(ssa_names) if midline_status == MIDLINE_NO
        else len(done_midline) if midline_status == MIDLINE_YES
        else 0
    )

    # ══ 1) MIDLINE SHIFT ══════════════════════════════════════════
    st.markdown("### 🧭 Midline Shift · SSA kesitleri")
    st.caption("Var/Yok bir kez seçilir ve tüm SSA kesitlerine uygulanır. 'Var' ise soldan bir kesite "
               "basın; kesit büyür ve her kesitte 3 nokta sırayla işaretlenir.")
    st.radio("Midline shift durumu (tüm SSA kesitleri için)", MIDLINE_OPTIONS,
             key=midline_status_key, index=None, horizontal=True)

    if midline_status == MIDLINE_YES:
        if not ssa_files:
            st.warning("Bu hasta için SSA kesiti bulunamadı.")
        else:
            st.progress(midline_complete / len(ssa_names),
                        text=f"Üç noktası tamamlanan SSA kesiti: {midline_complete}/{len(ssa_names)}")
            list_col, view_col = st.columns([2, 3], gap="large")

            with list_col:
                _render_all_slice_cards(ssa_files, patient, "midline", mid_active_key, win,
                                        LIST_COLS, done=done_midline)

            with view_col:
                st.markdown('<span class="sticky-anchor"></span>', unsafe_allow_html=True)
                active = st.session_state[mid_active_key]
                path = next(p for p in ssa_files if Path(p).name == active)
                points_key = _batch_key(patient, "points", active)
                try:
                    arr, spacing = load_slice(path)
                    base_img = apply_window(arr, *win)
                    img_w, img_h = base_img.size
                    st.session_state[f"batch::{patient}::meta::{active}"] = {
                        "spacing": spacing, "img_w": img_w, "img_h": img_h,
                    }
                    pts = st.session_state[points_key]

                    st.markdown(
                        f'<span class="chip">Kesit {ssa_names.index(active) + 1}/{len(ssa_names)}</span>'
                        + stepper_inline_html(len(pts)),
                        unsafe_allow_html=True,
                    )
                    undo_col, clear_col = st.columns(2)
                    undo_col.button("↩️ Geri al", key=f"batch-undo::{patient}::{active}",
                                    disabled=not pts, use_container_width=True,
                                    on_click=undo_point, args=(points_key,))
                    clear_col.button("🗑️ Temizle", key=f"batch-clear::{patient}::{active}",
                                     disabled=not pts, use_container_width=True,
                                     on_click=clear_points, args=(points_key,))

                    scale = min(min(disp_w, 640) / img_w, 1.0)
                    metrics = shift_metrics(pts, spacing) if len(pts) == 3 else None
                    display_img = draw_overlay(base_img, pts, scale, metrics)
                    coords = streamlit_image_coordinates(display_img, key=f"batch-click::{patient}::{active}")

                    if coords is not None:
                        click_key = f"batch-lastclick::{patient}::{active}"
                        if coords.get("unix_time") != st.session_state.get(click_key):
                            st.session_state[click_key] = coords.get("unix_time")
                            if len(pts) < 3:
                                factor = img_w / (coords.get("width") or display_img.width)
                                pts.append((round(coords["x"] * factor, 1), round(coords["y"] * factor, 1)))
                                if len(pts) == 3:
                                    # 3. nokta bitince sıradaki eksik SSA kesitine otomatik geç
                                    nxt = next(
                                        (f for f in ssa_names
                                         if f != active
                                         and len(st.session_state[_batch_key(patient, "points", f)]) < 3),
                                        None,
                                    )
                                    if nxt:
                                        st.session_state[mid_active_key] = nxt
                                st.rerun()

                    if len(pts) < 3:
                        st.caption(f"{img_w}×{img_h} px · Görüntüye tıklayarak {3 - len(pts)} nokta daha girin.")
                    else:
                        value = f"{metrics['mm']:.1f} mm" if metrics and metrics["mm"] is not None \
                            else (f"{metrics['px']:.0f} px" if metrics else "-")
                        st.caption(f"{img_w}×{img_h} px · ✓ 3 nokta tamam · Tahmini shift: {value}")
                except Exception as e:  # noqa: BLE001
                    st.error(f"DICOM okunamadı ({active}): {e}")

    elif midline_status == MIDLINE_NO:
        st.success("Midline shift yok olarak tüm SSA kesitlerine uygulandı; nokta seçimi gerekmiyor.")

    # ══ 2) BAZAL SİSTERN ══════════════════════════════════════════
    st.markdown("---")
    st.markdown("### 📋 Bazal Sistern · tüm kesitler")
    st.caption("Soldan bir kesite basın, sağda büyüsün. Seçim en altta tek sefer yapılır ve tüm kesitlere uygulanır.")

    if all_files:
        list_col, view_col = st.columns([2, 3], gap="large")
        with list_col:
            _render_all_slice_cards(all_files, patient, "basal", basal_active_key, win, LIST_COLS)
        with view_col:
            st.markdown('<span class="sticky-anchor"></span>', unsafe_allow_html=True)
            active_basal = st.session_state[basal_active_key]
            path = next(p for p in all_files if Path(p).name == active_basal)
            try:
                arr, _ = load_slice(path)
                img = apply_window(arr, *win)
                img.thumbnail((disp_w, disp_w))
                st.image(img, caption=f"Kesit {all_names.index(active_basal) + 1}/{len(all_names)} · {active_basal}",
                         use_container_width=True)
            except Exception as e:  # noqa: BLE001
                st.error(f"DICOM okunamadı ({active_basal}): {e}")
    else:
        st.warning("Bu hasta için kesit bulunamadı.")

    # tek bir bazal sistern seçimi — en altta
    st.markdown("#### Bazal sistern durumu (tüm kesitlere uygulanır)")
    st.radio("Bazal sistern durumu (tüm kesitlere uygulanır)", BASAL_OPTIONS,
             key=basal_key, index=None, label_visibility="collapsed")
    basal_status = st.session_state[basal_key]
    basal_complete = int(basal_status in BASAL_OPTIONS)

    # ══ 3) NOT + KAYIT (her şey tamamlanınca) ═════════════════════
    all_complete = bool(ssa_names and all_names) and midline_complete == len(ssa_names) and basal_complete == 1
    st.markdown("---")
    if not all_complete:
        st.info("Not alanı; midline durumu (Var ise her SSA kesitinde 3 nokta) ve bazal sistern seçimi "
                "tamamlandıktan sonra burada açılır.")
        return

    st.markdown("### 📝 Not ve kayıt")
    notes_key = f"batch::{patient}::notes"
    if notes_key not in st.session_state:
        prior = df_ann[
            (df_ann["doctor_name"].astype(str) == doctor)
            & (df_ann["patient_id"].astype(str) == patient)
        ]["notes"].dropna().astype(str)
        st.session_state[notes_key] = prior.iloc[-1] if not prior.empty else ""
    st.text_area("Tüm seçimler için ortak not", key=notes_key, height=110,
                 placeholder="Gözlem, şüphe, yorum…")

    def save_batch():
        if not doctor.strip():
            st.session_state.flash = ("error", "Doktor adı zorunludur.")
            return
        records = []
        basal_value = st.session_state[basal_key]
        file_by_name = {Path(p).name: p for p in ssa_files + all_files}
        ssa_name_set = set(ssa_names)
        ssa_index = {Path(path).name: idx for idx, path in enumerate(ssa_files)}
        all_index = {Path(path).name: idx for idx, path in enumerate(all_files)}
        for fname in unique_names:
            source = "ssa" if fname in ssa_name_set else "all"
            slice_idx = ssa_index[fname] if fname in ssa_index else all_index[fname]
            has_midline = fname in ssa_name_set
            pts = st.session_state[_batch_key(patient, "points", fname)] if has_midline else []
            midline = st.session_state[midline_status_key] if has_midline else None

            # Kesit hiç açılmadıysa (ör. "Yok" seçildi ya da kayıttan yüklendi) metadata'yı başlıktan oku
            metadata = st.session_state.get(f"batch::{patient}::meta::{fname}")
            if not metadata and has_midline:
                try:
                    sp, w, h = slice_header_meta(file_by_name[fname])
                    metadata = {"spacing": sp, "img_w": w, "img_h": h}
                except Exception:  # noqa: BLE001
                    metadata = {}
            metadata = metadata or {}
            spacing, img_w, img_h = metadata.get("spacing"), metadata.get("img_w"), metadata.get("img_h")

            use_pts = midline == MIDLINE_YES and len(pts) == 3
            metrics = shift_metrics(pts, spacing) if use_pts else None
            record = {
                "doctor_name": doctor.strip(), "patient_id": patient,
                "slice_source": source, "slice_file": fname, "slice_idx": slice_idx,
                "img_w": img_w, "img_h": img_h, "pixel_spacing_mm": spacing,
                "midline_shift_status": midline,
                "falx_ant_x": pts[0][0] if use_pts else None,
                "falx_ant_y": pts[0][1] if use_pts else None,
                "falx_post_x": pts[1][0] if use_pts else None,
                "falx_post_y": pts[1][1] if use_pts else None,
                "shift_point_x": pts[2][0] if use_pts else None,
                "shift_point_y": pts[2][1] if use_pts else None,
                "midline_shift_px": round(metrics["px"], 2) if metrics else None,
                "midline_shift_mm": round(metrics["mm"], 2) if metrics and metrics["mm"] is not None else None,
                "basal_status": basal_value,
                "notes": st.session_state[notes_key], "timestamp": str(pd.Timestamp.now()),
            }
            record.update({k: v for k, v in patient_meta(patient).items() if k not in record})
            records.append(record)

        try:
            hf_hub_download(repo_id=HF_REPO_ID, filename=ANNOTATION_FILE, repo_type="dataset",
                            token=HF_TOKEN, local_dir=".", force_download=True)
        except Exception:  # noqa: BLE001
            pass
        old = annotations_df()
        for record in records:
            for col in record:
                if col not in old.columns:
                    old[col] = None
        remove = (
            (old["doctor_name"].astype(str) == doctor.strip())
            & (old["patient_id"].astype(str) == patient)
            & old["slice_file"].astype(str).isin(set(unique_names))
        )
        updated = pd.concat([old[~remove], pd.DataFrame(records)], ignore_index=True)
        updated.to_csv(ANNOTATION_PATH, index=False)
        try:
            HfApi(token=HF_TOKEN).upload_file(
                path_or_fileobj=str(ANNOTATION_PATH), path_in_repo=ANNOTATION_FILE,
                repo_id=HF_REPO_ID, repo_type="dataset",
                commit_message=f"{doctor.strip()}: {patient} toplu kesit etiketleri",
            )
            st.session_state.flash = ("success", f"{patient} için tüm seçimler kaydedildi ve Hub'a gönderildi.")
        except Exception as e:  # noqa: BLE001
            st.session_state.flash = ("warning", f"Yerel kayıt yapıldı, Hub'a yüklenemedi: {e}")

    st.button("💾 Tüm seçimleri kaydet", type="primary", use_container_width=True,
              disabled=not doctor.strip(), on_click=save_batch)