"""Streamlit GUI for pydic: chessboard stereo calibration, single-camera
2D-DIC, and two-camera stereo-DIC, all driven from pydic.pipeline.

Run with:
    streamlit run gui/app.py
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pydic.calibration import StereoCalibration, calibrate_stereo, preview_chessboard_detection
from pydic.io_utils import load_gray, load_sequence, save_fields_csv, save_points3d_csv
from pydic.pipeline import Dic2D, StereoDic
from pydic.visualization import plot_points_3d, plot_scalar_field, plot_vector_field

st.set_page_config(page_title="pydic - Digital Image Correlation", page_icon="\U0001F4D0", layout="wide")

st.markdown(
    """
    <style>
    #MainMenu, footer {visibility: hidden; height: 0;}
    .block-container {padding-top: 1.5rem; padding-bottom: 2rem; max-width: 1200px;}
    div[data-testid="stMetricValue"] {font-size: 1.4rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------- utilities

def _natural_key(name: str):
    digits = "".join(ch for ch in Path(name).stem if ch.isdigit())
    return (int(digits) if digits else 0, name)


def _persist_uploads(files, workdir: str) -> list[str]:
    """Write uploaded files to disk (needed by calibration, which reads by
    path) and return paths sorted the same way as a real image sequence."""
    files_sorted = sorted(files, key=lambda f: _natural_key(f.name))
    paths = []
    for f in files_sorted:
        p = os.path.join(workdir, f.name)
        with open(p, "wb") as out:
            out.write(f.getbuffer())
        paths.append(p)
    return paths


def image_source_picker(label: str, key: str, multiple: bool) -> list[str]:
    """A small reusable widget: choose 'Upload files' or 'Local folder path'
    and return a sorted list of image paths (uploaded files are written to a
    session-scoped temp dir so downstream code always just deals in paths)."""
    mode = st.radio(f"{label} source", ["Upload files", "Local folder path"], key=f"{key}_mode", horizontal=True)
    workdir = st.session_state.setdefault("_workdir", tempfile.mkdtemp(prefix="pydic_gui_"))

    if mode == "Upload files":
        files = st.file_uploader(
            label, accept_multiple_files=multiple, key=f"{key}_upload",
            type=["png", "jpg", "jpeg", "tif", "tiff", "bmp"],
        )
        if not files:
            return []
        files = files if multiple else [files]
        sub = os.path.join(workdir, key)
        os.makedirs(sub, exist_ok=True)
        return _persist_uploads(files, sub)
    else:
        folder = st.text_input(f"{label} folder", key=f"{key}_folder")
        pattern = st.text_input(f"{label} filename pattern", value="*.tif", key=f"{key}_pattern")
        if not folder:
            return []
        paths = load_sequence(folder, pattern)
        st.caption(f"{len(paths)} file(s) matched")
        return paths


def zip_dir(directory: str) -> bytes:
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
        archive_base = tmp.name[:-4]
    shutil.make_archive(archive_base, "zip", directory)
    data = Path(archive_base + ".zip").read_bytes()
    os.remove(archive_base + ".zip")
    return data


def _corner_overlay_thumbnail(image_bgr: np.ndarray, pattern_size, corners, found: bool, max_dim: int = 320) -> np.ndarray:
    vis = image_bgr.copy()
    if found and corners is not None:
        cv2.drawChessboardCorners(vis, pattern_size, corners, found)
    h, w = vis.shape[:2]
    scale = max_dim / max(h, w)
    vis = cv2.resize(vis, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)


def chessboard_diagnostics(paths, pattern_size, key: str):
    """Show a pass/fail grid with corner overlays so a failed calibration is
    debuggable (wrong board size, glare, blur, board partly out of frame,
    image too large for the old detector, ...) instead of a bare error."""
    if not paths:
        return
    if not st.button("Preview corner detection", key=f"{key}_preview_btn"):
        return
    with st.spinner(f"Running detection on {len(paths)} image(s)..."):
        results = preview_chessboard_detection(paths, pattern_size)
    n_found = sum(r["found"] for r in results)
    st.write(f"**{n_found}/{len(results)}** image(s) detected with a {pattern_size[0]}x{pattern_size[1]} inner-corner pattern.")
    cols = st.columns(4)
    for i, r in enumerate(results):
        with cols[i % 4]:
            name = Path(r["path"]).name
            if r["image"] is None:
                st.error(f"{name}: could not read file")
                continue
            thumb = _corner_overlay_thumbnail(r["image"], pattern_size, r["corners"], r["found"])
            st.image(thumb, caption=f"{'OK' if r['found'] else 'FAILED'}: {name}")


def roi_bbox_picker(image: np.ndarray, key: str):
    h, w = image.shape[:2]
    st.image(image, caption="Reference image", clamp=True, width="stretch")
    use_roi = st.checkbox("Restrict to a rectangular ROI", key=f"{key}_use_roi")
    if not use_roi:
        return None
    c1, c2 = st.columns(2)
    x0, x1 = c1.slider("x range", 0, w, (0, w), key=f"{key}_x")
    y0, y1 = c2.slider("y range", 0, h, (0, h), key=f"{key}_y")
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


# --------------------------------------------------------------------- tabs

def about_tab():
    st.title("pydic - Digital Image Correlation")
    st.markdown(
        """
This app runs the `pydic` library end to end:

1. **Stereo Calibration** - calibrate a two-camera rig from chessboard images.
2. **2D DIC** - single-camera planar displacement + strain over an image sequence.
3. **Stereo DIC** - two-camera 3D displacement + surface strain over an image sequence.

Algorithms: IC-GN subpixel subset correlation, reliability-guided (RG-DIC)
full-field propagation, epipolar stereo matching + triangulation, and local
least-squares strain (2D and 3D-surface Green-Lagrange). See the project
README for the underlying references.

**Tips for good results**
- Speckle pattern: high-contrast, isotropic, non-repeating, ~3-5 px per speckle.
- Subset radius: big enough to contain several speckles (try 12-25 px);
  grid step: usually half to a full subset diameter.
- If tracking fails on frame 1, check the disparity range (stereo) or use a
  seed point for large motion (currently exposed via the Python API only).
        """
    )


def calibration_tab():
    st.header("Stereo Calibration")
    st.caption(
        "Each left/right image pair must show the SAME chessboard pose, "
        "captured simultaneously by camera 1 (left) and camera 2 (right)."
    )

    loaded = st.file_uploader("Or load an existing calibration JSON", type=["json"], key="calib_load")
    if loaded is not None:
        # StereoCalibration.load() reads from a path; persist the upload first.
        tmp_path = os.path.join(tempfile.mkdtemp(), "calib.json")
        with open(tmp_path, "wb") as f:
            f.write(loaded.getbuffer())
        st.session_state["stereo_calib"] = StereoCalibration.load(tmp_path)
        st.success("Calibration loaded from file.")

    left_paths = image_source_picker("Left camera chessboard images", "calib_left", multiple=True)
    right_paths = image_source_picker("Right camera chessboard images", "calib_right", multiple=True)

    c1, c2, c3 = st.columns(3)
    cols = c1.number_input("Inner corners (columns)", min_value=2, value=9)
    rows = c2.number_input("Inner corners (rows)", min_value=2, value=6)
    square_size = c3.number_input("Square size (mm)", min_value=0.01, value=20.0)

    with st.expander("Preview corner detection (debug a failed calibration here)"):
        st.caption(
            "Run this before calibrating if you're not sure the board size is "
            "right, or after a failed calibration to see exactly which images "
            "were rejected and why (wrong corner count, glare, blur, board "
            "partly out of frame, ...)."
        )
        pc1, pc2 = st.columns(2)
        with pc1:
            st.write("Left camera")
            chessboard_diagnostics(left_paths, (int(cols), int(rows)), "calib_left_diag")
        with pc2:
            st.write("Right camera")
            chessboard_diagnostics(right_paths, (int(cols), int(rows)), "calib_right_diag")

    if st.button("Run stereo calibration", disabled=not (left_paths and right_paths)):
        if len(left_paths) != len(right_paths):
            st.error(f"Left ({len(left_paths)}) and right ({len(right_paths)}) image counts differ.")
        else:
            with st.spinner("Detecting chessboards and calibrating..."):
                try:
                    calib = calibrate_stereo(left_paths, right_paths, (int(cols), int(rows)), float(square_size))
                except RuntimeError as e:
                    st.error(f"{e} Use the 'Preview corner detection' panel above to see which images failed and why.")
                    calib = None
            if calib is not None:
                st.session_state["stereo_calib"] = calib
                st.success("Calibration succeeded.")

    calib: StereoCalibration | None = st.session_state.get("stereo_calib")
    if calib is not None:
        st.subheader("Calibration result")
        m1, m2, m3 = st.columns(3)
        m1.metric("Stereo RMS (px)", f"{calib.rms_error:.4f}")
        m2.metric("Cam1 RMS (px)", f"{calib.cam1.rms_error:.4f}")
        m3.metric("Cam2 RMS (px)", f"{calib.cam2.rms_error:.4f}")
        st.write("Baseline T (mm):", calib.T.ravel().tolist())

        tmp_json = os.path.join(tempfile.mkdtemp(), "stereo_calib.json")
        calib.save(tmp_json)
        st.download_button(
            "Download calibration JSON", data=Path(tmp_json).read_bytes(),
            file_name="stereo_calib.json", mime="application/json",
        )


def dic2d_tab():
    st.header("2D DIC (single camera)")

    ref_paths = image_source_picker("Reference image", "d2_ref", multiple=False)
    seq_paths = image_source_picker("Deformed image sequence (ordered)", "d2_seq", multiple=True)

    c1, c2, c3, c4 = st.columns(4)
    subset_radius = c1.number_input("Subset radius (px)", min_value=3, value=15, key="d2_radius")
    grid_step = c2.number_input("Grid step (px)", min_value=1, value=10, key="d2_step")
    zncc_thr = c3.number_input("ZNCC threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.05, key="d2_zncc")
    strain_hops = c4.number_input("Strain window (grid hops)", min_value=1, value=2, key="d2_hops")

    roi = None
    if ref_paths:
        roi = roi_bbox_picker(load_gray(ref_paths[0]), "d2_roi")

    if st.button("Run 2D DIC", disabled=not (ref_paths and seq_paths)):
        with st.spinner(f"Tracking {len(seq_paths)} frame(s)..."):
            dic = Dic2D(subset_radius=int(subset_radius), grid_step=int(grid_step), zncc_threshold=float(zncc_thr))
            dic.set_reference(ref_paths[0], roi_polygon=roi)
            results = dic.run_sequence(seq_paths)
        st.session_state["dic2d"] = dic
        st.session_state["dic2d_results"] = results
        st.success(f"Done: {len(dic.points)} points x {len(results)} frame(s).")

    dic: Dic2D | None = st.session_state.get("dic2d")
    results = st.session_state.get("dic2d_results")
    if dic is not None and results:
        frame_idx = (
            st.slider("Frame", 1, len(results), 1, key="d2_frame") - 1 if len(results) > 1 else 0
        )
        frame = results[frame_idx]
        strain = dic.compute_strain(frame, window_hops=int(strain_hops))

        st.metric("Converged fraction", f"{frame.valid.mean()*100:.1f}%")

        field = st.selectbox(
            "Field to display", ["displacement (u, v)", "exx", "eyy", "exy", "major principal", "von Mises"],
            key="d2_field",
        )
        fig, ax = plt.subplots(figsize=(6, 5))
        if field == "displacement (u, v)":
            plot_vector_field(frame.points, frame.u, frame.v, valid=frame.valid, ax=ax)
        else:
            values = {
                "exx": strain.exx, "eyy": strain.eyy, "exy": strain.exy,
                "major principal": strain.major_principal, "von Mises": strain.von_mises,
            }[field]
            plot_scalar_field(frame.points, values, valid=strain.valid, title=field, ax=ax)
        st.pyplot(fig)

        out_dir = os.path.join(tempfile.mkdtemp(), "dic2d_export")
        os.makedirs(out_dir, exist_ok=True)
        for t, fr in enumerate(results, start=1):
            s = dic.compute_strain(fr, window_hops=int(strain_hops))
            save_fields_csv(
                os.path.join(out_dir, f"frame_{t:04d}.csv"), fr.points,
                {"u": fr.u, "v": fr.v, "zncc": fr.zncc, "valid": fr.valid.astype(int),
                 "exx": s.exx, "eyy": s.eyy, "exy": s.exy},
            )
        st.download_button(
            "Download all frames (CSV, zipped)", data=zip_dir(out_dir),
            file_name="dic2d_results.zip", mime="application/zip",
        )


def stereo_dic_tab():
    st.header("Stereo DIC (two cameras)")

    calib: StereoCalibration | None = st.session_state.get("stereo_calib")
    if calib is None:
        st.warning("Run or load a stereo calibration in the 'Stereo Calibration' tab first.")
        return
    st.caption(f"Using loaded calibration (stereo RMS {calib.rms_error:.4f} px).")

    ref1 = image_source_picker("Reference image - camera 1 (left)", "s_ref1", multiple=False)
    ref2 = image_source_picker("Reference image - camera 2 (right)", "s_ref2", multiple=False)
    seq1 = image_source_picker("Deformed sequence - camera 1", "s_seq1", multiple=True)
    seq2 = image_source_picker("Deformed sequence - camera 2", "s_seq2", multiple=True)

    c1, c2, c3 = st.columns(3)
    subset_radius = c1.number_input("Subset radius (px)", min_value=3, value=15, key="s_radius")
    grid_step = c2.number_input("Grid step (px)", min_value=1, value=10, key="s_step")
    zncc_thr = c3.number_input("ZNCC threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.05, key="s_zncc")

    c4, c5, c6 = st.columns(3)
    disp_min = c4.number_input("Disparity range min (px)", value=-150.0, key="s_dmin")
    disp_max = c5.number_input("Disparity range max (px)", value=150.0, key="s_dmax")
    epi_band = c6.number_input("Epipolar search band (px)", min_value=1.0, value=6.0, key="s_band")

    roi = None
    if ref1:
        roi = roi_bbox_picker(load_gray(ref1[0]), "s_roi")

    if st.button("Run stereo DIC", disabled=not (ref1 and ref2 and seq1 and seq2)):
        if len(seq1) != len(seq2):
            st.error(f"Camera-1 ({len(seq1)}) and camera-2 ({len(seq2)}) sequence lengths differ.")
        else:
            with st.spinner(f"Matching + tracking {len(seq1)} frame(s)..."):
                dic = StereoDic(
                    calib, subset_radius=int(subset_radius), grid_step=int(grid_step),
                    zncc_threshold=float(zncc_thr), epipolar_band=float(epi_band),
                    disparity_range=(float(disp_min), float(disp_max)),
                )
                dic.set_reference(ref1[0], ref2[0], roi_polygon=roi)
                frame_results = dic.run_sequence(seq1, seq2)
            st.session_state["stereo_dic"] = dic
            st.session_state["stereo_dic_results"] = frame_results
            st.success(f"Done: {len(dic.points1)} points x {len(frame_results)} frame(s).")

    dic: StereoDic | None = st.session_state.get("stereo_dic")
    frame_results = st.session_state.get("stereo_dic_results")
    if dic is not None and frame_results:
        frame_idx = (
            st.slider("Frame", 1, len(frame_results), 1, key="s_frame") - 1
            if len(frame_results) > 1 else 0
        )
        fr = frame_results[frame_idx]
        strain = dic.compute_strain(fr)

        st.metric("Valid fraction", f"{fr.valid.mean()*100:.1f}%")

        field = st.selectbox(
            "Field to display",
            ["3D displacement magnitude", "exx", "eyy", "exy", "major principal", "von Mises"],
            key="s_field",
        )
        fig = plt.figure(figsize=(6, 5))
        if field == "3D displacement magnitude":
            ax = fig.add_subplot(projection="3d")
            mag = np.linalg.norm(fr.displacement_3d, axis=1)
            plot_points_3d(fr.points_cur_3d, values=mag, valid=fr.valid, title="|displacement| (mm)", ax=ax)
        else:
            ax = fig.add_subplot()
            values = {
                "exx": strain.exx, "eyy": strain.eyy, "exy": strain.exy,
                "major principal": strain.major_principal, "von Mises": strain.von_mises,
            }[field]
            plot_scalar_field(dic.points1, values, valid=strain.valid, title=f"{field} (camera-1 view)", ax=ax)
        st.pyplot(fig)

        out_dir = os.path.join(tempfile.mkdtemp(), "stereo_dic_export")
        os.makedirs(out_dir, exist_ok=True)
        for t, f in enumerate(frame_results, start=1):
            s = dic.compute_strain(f)
            d = f.displacement_3d
            save_points3d_csv(
                os.path.join(out_dir, f"frame_{t:04d}.csv"), f.points_ref_3d,
                {"dX": d[:, 0], "dY": d[:, 1], "dZ": d[:, 2],
                 "exx": s.exx, "eyy": s.eyy, "exy": s.exy, "valid": f.valid.astype(int)},
            )
        st.download_button(
            "Download all frames (CSV, zipped)", data=zip_dir(out_dir),
            file_name="stereo_dic_results.zip", mime="application/zip",
        )


def main():
    tabs = st.tabs(["About", "Stereo Calibration", "2D DIC", "Stereo DIC"])
    with tabs[0]:
        about_tab()
    with tabs[1]:
        calibration_tab()
    with tabs[2]:
        dic2d_tab()
    with tabs[3]:
        stereo_dic_tab()


if __name__ == "__main__":
    main()
