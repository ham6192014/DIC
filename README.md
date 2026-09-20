# pydic

Subset-based 2D and stereo (3D) Digital Image Correlation in Python, built as
a library first — a GUI can be layered on top of `pydic.pipeline`.

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

## Building a GUI on top

`pydic.pipeline.Dic2D` and `StereoDic` are the intended integration point:
they take images/paths in and return plain NumPy arrays out (no plotting or
file I/O side effects), so a GUI just needs to call `set_reference` /
`run_sequence` / `compute_strain` from button handlers and render the
returned arrays (`visualization.py` has ready-made Matplotlib renderers you
can embed in a Qt/Tk canvas, or use as a reference for a custom renderer).
`io_utils.select_roi_polygon` shows the interactive-picker pattern if you
want a similar click-to-select ROI in your own canvas.

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
