# pydic

Subset-based 2D and stereo (3D) Digital Image Correlation in Python, with a
`pydic.pipeline` library API and a Streamlit GUI (`gui/app.py`) on top of it.

## Run the GUI

```bash
pip install -r requirements.txt
streamlit run gui/app.py
```

This opens in your browser (default `http://localhost:8501`). It has four
tabs: **About**, **Stereo Calibration** (chessboard calibration, save/load
JSON), **2D DIC** (single camera), and **Stereo DIC** (two cameras, gated on
having a calibration loaded). Each DIC tab lets you point at images either
by uploading files or by typing a local folder path + filename pattern, set
subset radius / grid step / thresholds, run, page through frames, view
displacement/strain fields, and download all frames as a zipped CSV.

**Selecting the ROI**: check "Restrict to a rectangular ROI" to get an
interactive crop box you drag directly on the reference image (requires
`streamlit-cropper`, in requirements.txt) — a preview of exactly what's
inside the box is shown below it, so you can confirm it actually covers the
speckled/textured region before running. This matters: DIC has nothing to
correlate against a flat, dark, or untextured area, and a run over such a
region will "succeed" while reporting ~0 displacement everywhere, which is
easy to mistake for a bug rather than a targeting problem. A run shows a
live progress bar (points resolved so far) rather than a bare spinner, and
if the result comes back with near-zero convergence or near-zero
displacement, the app tells you directly instead of leaving you to guess.

If you'd rather not launch a browser app, the same functionality is
available as plain Python (below) or via the `examples/*.py` CLI scripts.

## Why another DIC implementation

Most of the accuracy problems people hit with free DIC tools come down to
two things: a weak subpixel correlation engine (nearest-neighbor / plain
template matching instead of a real subpixel optimizer), and no principled
way to turn a noisy displacement field into strain. This library implements
the same core algorithms used by the serious (often commercial) tools:

- **IC-GN subpixel correlation** — Inverse-Compositional Gauss-Newton
  optimization of a first-order (affine) subset shape function against a
  bicubic-spline-interpolated deformed image, using the zero-mean
  normalized sum-of-squared-differences (ZNSSD) criterion (robust to
  linear brightness/contrast changes). This is the algorithm behind
  Ncorr/VIC-2D/VIC-3D-class accuracy (see Pan, Li & Xie, *Exp. Mech.* 2013).
  Interpolation is done with small *local* splines built per subset rather
  than one spline over the whole frame — scipy's spline evaluation cost
  scales with the whole image's resolution regardless of how few points you
  query, so on a multi-megapixel camera photo a naive global spline can make
  correlation take minutes; local interpolation keeps it fast independent
  of the surrounding image's resolution.
- **Reliability-guided propagation (RG-DIC)** — subsets are correlated
  starting from a seed and grown outward, always expanding next from the
  most-reliable (highest ZNCC) converged point, using its warp as the
  initial guess for its neighbors (Pan, *Applied Optics* 2009). This is far
  more robust to large or non-uniform deformation than independently
  searching every grid point.
- **Local least-squares strain** — strain at each point is fit from a
  small neighborhood of displacement/point-cloud data rather than
  differentiated point-to-point, which is what keeps strain fields usable
  instead of dominated by correlation noise.
- **Proper stereo geometry** — epipolar-constrained correspondence search
  at the reference frame, triangulation via undistorted normalized camera
  coordinates (not naive rectification), and independent-but-synchronized
  temporal tracking per camera (always against the fixed reference image,
  so there's no drift), triangulated at every frame to get 3D
  displacement and a local-surface Green-Lagrange strain tensor.

Every piece of math above is validated against synthetic ground truth in
`tests/` (see below) — you can run the tests to confirm the whole pipeline
before pointing it at real images.

## Install

```bash
pip install -r requirements.txt
# or: pip install -e .
```

## Quick start — single camera (2D-DIC)

```python
from pydic.pipeline import Dic2D

dic = Dic2D(subset_radius=15, grid_step=10)
dic.set_reference("images/frame_0000.tif")          # optionally roi_polygon=[(x,y), ...]
results = dic.run_sequence(["images/frame_0001.tif", "images/frame_0002.tif", ...])

for frame in results:
    strain = dic.compute_strain(frame)               # kind="green-lagrange" or "engineering"
    # frame.points (N,2), frame.u, frame.v, frame.valid
    # strain.exx, strain.eyy, strain.exy, strain.major_principal, strain.von_mises
```

See `examples/run_2d_dic.py` for a CLI wrapper that writes per-frame CSVs.

## Quick start — two cameras (stereo/3D-DIC)

1. **Calibrate** with a chessboard (same board pose, captured simultaneously
   by both cameras, for each calibration image):

```bash
python3 examples/calibrate_stereo.py calib/left calib/right \
    --cols 9 --rows 6 --square-size 20.0 --out stereo_calib.json
```

`--cols`/`--rows` are *inner* corners (a 10x7-square board has 9x6 inner
corners). `--square-size` is the physical square size (mm) — this fixes the
scale of every downstream 3D measurement, so measure it carefully.

Left/right images are matched by the numeric ID in their filename
(`L10.png` pairs with `R10.png`) via `match_stereo_pairs_by_filename`, not
by list position — so upload/glob order can't silently mismatch a pair.
Files with no matching number on the other side, or duplicate numbers, are
reported rather than guessed at.

**Calibration quality gate**: `calibrate_stereo` doesn't report success
just because OpenCV returned without an exception. By default it raises
`CalibrationQualityError` if the stereo RMS, any individual pair's
reprojection error, or the number of valid views fails a threshold
(`max_rms_error`, `max_pair_reproj_error`, `min_valid_views`) — the
rejected `StereoCalibration` and its full diagnostics are still attached to
the exception (`.calibration`, `.report`) for inspection. `StereoDic`
independently refuses a `calibration.quality_ok == False` object unless you
pass `allow_poor_quality_calibration=True`, so a bad calibration can't
accidentally end up driving a 3D-DIC run just because it was loaded from a
file saved earlier. The CLI's `--force` flag and the GUI's "use this
calibration anyway" checkbox are the explicit opt-outs.

Every calibration's `.report` (`StereoCalibrationReport`) carries per-pair
diagnostics: which images were used vs. rejected and why, whether each
image's corner order was flipped during canonicalization, and per-pair
left/right reprojection error — printed by the CLI and shown as a table in
the GUI's Stereo Calibration tab.

**180-degree corner-ordering ambiguity**: a plain checkerboard looks
identical rotated 180 degrees, so a detector can legally label either of
two diagonally-opposite corners as index 0 in any given photo — perfectly
fine for that camera's own mono calibration (labeling doesn't affect
within-camera reprojection), but if the left and right image of one pose
disagree, that pose's point correspondences are scrambled and can dominate
the joint stereo RMS. This is resolved for every image automatically
(`_canonicalize_corner_order` in `pydic/calibration.py`) using a property
of the physical board, not the photo: the checkerboard square immediately
next to corner 0 is a fixed color, and comparing it *against its neighbor
in the same photo* — rather than a fixed absolute brightness threshold —
keeps the decision correct even when left/right exposure, gain, or
lighting differ, which a threshold-based check would get wrong.

**If calibration reports too few valid views**: chessboard detection runs on
a downscaled copy of each image internally (very high-resolution camera
photos — tens of megapixels — can make the classical OpenCV detector miss
the board entirely or hang), with a fallback to the modern, more robust
`findChessboardCornersSB` detector, so this should be rare. If it still
happens, use the GUI's **Preview corner detection** panel (Stereo
Calibration tab) — it runs detection on every image and shows a thumbnail
with a pass/fail label and the detected corner overlay, so you can see
directly whether the issue is a wrong `--cols`/`--rows` count, glare, blur,
or the board partly out of frame, instead of just an error count.

2. **Run stereo-DIC**:

```python
from pydic.calibration import StereoCalibration
from pydic.pipeline import StereoDic

calib = StereoCalibration.load("stereo_calib.json")
dic = StereoDic(calib, subset_radius=15, grid_step=10)
dic.set_reference("left/0000.tif", "right/0000.tif")   # optionally roi_polygon=[...]

frame_results = dic.run_sequence(
    ["left/0001.tif", "left/0002.tif", ...],
    ["right/0001.tif", "right/0002.tif", ...],
)

for fr in frame_results:
    strain = dic.compute_strain(fr)
    # fr.points_ref_3d, fr.points_cur_3d, fr.displacement_3d (N,3), fr.valid
    # strain.exx, strain.eyy, strain.exy in each point's local surface frame
```

See `examples/run_stereo_dic.py` for a CLI wrapper.

### If matching/tracking struggles

- **Large baseline / disparity**: widen `StereoDic(..., disparity_range=(min, max))`
  (pixels) to cover the true disparity at your working distance.
- **Large deformation on frame 1**: `Dic2D.run_sequence(..., seed_xy=(x, y))`
  or `StereoDic`'s equivalent lets you bootstrap a coarse initial guess from
  template matching instead of assuming near-zero motion.
- **Speckle pattern quality** matters more than any algorithm setting: aim
  for high-contrast, isotropic, non-repeating speckles at 3-5 px per
  speckle at your imaging resolution. A poor pattern will make *any* DIC
  code — this one included — perform worse than expected.

## Package layout

| Module | Responsibility |
|---|---|
| `calibration.py` | Chessboard mono/stereo calibration (OpenCV), save/load |
| `subset.py` | Affine subset shape function, inverse-compositional warp composition |
| `correlation.py` | Bicubic interpolation, IC-GN solver, template-match initial guess |
| `fields.py` | Grid generation, ROI masks, grid neighbor graph |
| `tracking2d.py` | Reliability-guided propagation, frame-to-frame sequence tracking |
| `strain.py` | Local least-squares strain (2D field and 3D surface) |
| `stereo.py` | Epipolar matching, triangulation, stereo sequence tracking |
| `pipeline.py` | `Dic2D` / `StereoDic` — the high-level API a GUI should call |
| `io_utils.py` | Image loading, ROI polygon picker, CSV/JSON I/O |
| `visualization.py` | Matplotlib scatter/quiver/3D plots for quick inspection |

## Extending or replacing the GUI

`pydic.pipeline.Dic2D` and `StereoDic` are the integration point the shipped
Streamlit app (`gui/app.py`) is built on: they take images/paths in and
return plain NumPy arrays out (no plotting or file I/O side effects), so any
other GUI toolkit (Qt/Tk/web) just needs to call `set_reference` /
`run_sequence` / `compute_strain` from its own event handlers and render the
returned arrays. `visualization.py` has ready-made Matplotlib renderers you
can reuse or use as a reference. The current ROI picker is a rectangle
(`gui/app.py`'s slider-based bbox); `io_utils.select_roi_polygon` shows the
interactive-polygon pattern if you want to add free-form ROI drawing (e.g.
via the `streamlit-drawable-canvas` package, or a canvas widget in Qt/Tk).

## Running the tests

```bash
python3 -m pytest tests/ -v
```

These validate the math against known ground truth (no camera or real
images required): subpixel correlation accuracy, RG-DIC + 2D strain against
a known affine deformation, triangulation/epipolar geometry, 3D surface
strain against a known stretch, and a full synthetic two-camera pipeline
(perspective cameras + calibration + matching + tracking + strain) checked
against the analytically known answer.
