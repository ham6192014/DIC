import cv2
import numpy as np
import pytest

from pydic.calibration import (
    CalibrationQualityError,
    StereoCalibration,
    _canonicalize_corner_order,
    calibrate_stereo,
    find_chessboard_corners,
    match_stereo_pairs_by_filename,
)
from pydic.pipeline import StereoDic


def _flat_checkerboard(cols: int, rows: int, square_px: int = 220, margin_mult: int = 2) -> np.ndarray:
    """A plain top-down (no perspective) checkerboard render, for tests that
    care about in-image rotation/exposure rather than 3D pose."""
    margin = square_px * margin_mult
    board_w = (cols + 1) * square_px + 2 * margin
    board_h = (rows + 1) * square_px + 2 * margin
    img = np.full((board_h, board_w), 255, dtype=np.uint8)
    for r in range(rows + 1):
        for c in range(cols + 1):
            if (r + c) % 2 == 0:
                continue
            y0, x0 = margin + r * square_px, margin + c * square_px
            img[y0:y0 + square_px, x0:x0 + square_px] = 0
    return img


def _render_chessboard(rvec, tvec, K, D, img_size, cols, rows, square):
    img = np.full((img_size[1], img_size[0]), 255, dtype=np.uint8)
    for r in range(rows + 1):
        for c in range(cols + 1):
            if (r + c) % 2 == 0:
                continue
            pts = np.array(
                [[c * square, r * square, 0], [(c + 1) * square, r * square, 0],
                 [(c + 1) * square, (r + 1) * square, 0], [c * square, (r + 1) * square, 0]],
                dtype=np.float32,
            )
            proj, _ = cv2.projectPoints(pts, rvec, tvec, K, D)
            cv2.fillConvexPoly(img, proj.reshape(-1, 2).astype(np.int32), 0)
    return img


def test_find_chessboard_corners_on_large_image():
    # A high-megapixel image (comparable to a real ~47MB camera photo) is
    # exactly the case that used to fail: the classical detector with
    # CALIB_CB_FAST_CHECK, run at full resolution, would miss the board or
    # take a very long time. Detection must succeed quickly here.
    cols, rows, square = 9, 6, 25.0
    K = np.array([[3200.0, 0, 2000], [0, 3200.0, 1500], [0, 0, 1]])
    D = np.zeros(5)
    img = _render_chessboard(
        np.array([0.15, 0.1, 0.05], dtype=np.float32),
        np.array([-50, -30, 900], dtype=np.float32),
        K, D, (4000, 3000), cols, rows, square,
    )
    bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    found, corners, _flipped = find_chessboard_corners(bgr, (cols, rows))
    assert found
    assert corners is not None
    assert len(corners) == cols * rows


COLS, ROWS, SQUARE = 9, 6, 25.0
_K = np.array([[3200.0, 0, 2000], [0, 3200.0, 1500], [0, 0, 1]])
_D = np.zeros(5)
_IMG_SIZE = (4000, 3000)
_TRUE_BASELINE_X = -150.0


def _build_synthetic_stereo_dataset(tmp_path, n_pairs: int = 6, seed: int = 0, id_offset: int = 0):
    """A synthetic multi-pose stereo chessboard dataset with a known ground
    truth baseline, written to tmp_path/left/L{i}.png and .../right/R{i}.png.

    Returns (left_paths, right_paths) in natural (0..n_pairs-1 + id_offset) order.
    """
    R2 = cv2.Rodrigues(np.array([0.0, 0.15, 0.0]))[0]
    T2 = np.array([_TRUE_BASELINE_X, 0.0, 0.0])

    rng = np.random.default_rng(seed)
    left_dir = tmp_path / "left"
    right_dir = tmp_path / "right"
    left_dir.mkdir(exist_ok=True)
    right_dir.mkdir(exist_ok=True)

    left_paths, right_paths = [], []
    for i in range(n_pairs):
        rvec1 = (rng.uniform(-0.3, 0.3, 3) + np.array([0.15, 0.1, 0.05])).astype(np.float32)
        tvec1 = np.array([-100 + i * 10, -60 + i * 5, 900 + i * 20], dtype=np.float32)
        left_img = _render_chessboard(rvec1, tvec1, _K, _D, _IMG_SIZE, COLS, ROWS, SQUARE)

        R1, _ = cv2.Rodrigues(rvec1)
        Rc2 = R2 @ R1
        Tc2 = (R2 @ tvec1.reshape(3, 1) + T2.reshape(3, 1)).ravel()
        rvec2, _ = cv2.Rodrigues(Rc2)
        right_img = _render_chessboard(rvec2, Tc2.astype(np.float32), _K, _D, _IMG_SIZE, COLS, ROWS, SQUARE)

        nid = i + id_offset
        lp = str(left_dir / f"L{nid}.png")
        rp = str(right_dir / f"R{nid}.png")
        cv2.imwrite(lp, left_img)
        cv2.imwrite(rp, right_img)
        left_paths.append(lp)
        right_paths.append(rp)

    return left_paths, right_paths


def test_calibrate_stereo_recovers_known_baseline(tmp_path):
    left_paths, right_paths = _build_synthetic_stereo_dataset(tmp_path, n_pairs=6)
    calib = calibrate_stereo(left_paths, right_paths, (COLS, ROWS), SQUARE)
    assert calib.quality_ok
    assert calib.rms_error < 1.0
    assert abs(calib.T.ravel()[0] - _TRUE_BASELINE_X) < 5.0
    # per-pair diagnostics were populated for every accepted view
    assert len(calib.report.used_pairs) == 6
    assert all(p.left_reproj_error_px is not None for p in calib.report.used_pairs)
    assert not calib.report.rejected_pairs


def test_canonicalization_resolves_180_degree_labeling_ambiguity():
    # A plain checkerboard is visually symmetric under 180-degree rotation,
    # so a detector can legally label either of two diagonally-opposite
    # corners as index 0. This is exactly the failure mode behind a real
    # bug report: two individually-fine mono calibrations (each doesn't
    # care about labeling) but a stereo RMS of ~150px, because one image of
    # a pose got the "opposite" labeling from its pair. Simulate that here:
    # detect once normally, once with the raw order reversed (standing in
    # for the other camera picking the opposite corner), and confirm
    # canonicalizing both converges on the identical physical labeling.
    cols, rows = COLS, ROWS
    img = _flat_checkerboard(cols, rows)

    found, raw = cv2.findChessboardCornersSB(
        img, (cols, rows), flags=cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
    )
    assert found
    raw = cv2.cornerSubPix(
        img, raw.astype(np.float32), (11, 11), (-1, -1),
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-4),
    )
    flipped_raw = raw[::-1].copy()

    canon_normal, was_flipped_normal = _canonicalize_corner_order(img, raw, (cols, rows))
    canon_flipped, was_flipped_flipped = _canonicalize_corner_order(img, flipped_raw, (cols, rows))
    assert np.allclose(canon_normal, canon_flipped)
    assert was_flipped_normal != was_flipped_flipped


def test_canonicalization_consistent_across_different_board_orientations():
    # Requirement: robust regardless of how the board is rotated/oriented in
    # frame, not just the pure-180-degree case. Detect the same physical
    # board photographed at several different in-image rotations (a stereo
    # rig's two cameras rarely see the operator-held board at exactly the
    # same apparent angle) and confirm every detection resolves to the same
    # physical corner-0, by mapping each detection back into the original
    # (unrotated) frame and checking they all agree.
    cols, rows = COLS, ROWS
    board = _flat_checkerboard(cols, rows)
    h, w = board.shape
    center = (w / 2.0, h / 2.0)

    reference = None
    for angle in [0, 15, -25, 90 + 12, 180, 217]:
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(board, M, (w, h), borderValue=255, flags=cv2.INTER_LINEAR)
        found, corners, _flipped = find_chessboard_corners(
            cv2.cvtColor(rotated, cv2.COLOR_GRAY2BGR), (cols, rows)
        )
        assert found, f"detection failed at rotation {angle} degrees"

        M_inv = cv2.invertAffineTransform(M)
        pts = corners.reshape(-1, 2)
        pts_h = np.hstack([pts, np.ones((len(pts), 1))])
        unrotated = (M_inv @ pts_h.T).T

        if reference is None:
            reference = unrotated
        else:
            max_diff = np.max(np.abs(unrotated - reference))
            assert max_diff < 3.0, (
                f"rotation {angle} degrees disagreed with the reference orientation "
                f"by {max_diff:.2f}px after un-rotating -- corner-0 labeling is not "
                f"orientation-invariant"
            )


def test_canonicalization_consistent_across_different_exposures():
    # Requirement: don't rely on the board appearing "upright" or any fixed
    # absolute brightness. Two photos of the identical pose taken at very
    # different exposures (dark: black~40/white~140; bright: black~150/
    # white~250 -- both straddle the naive 128 midpoint in OPPOSITE ways)
    # must still canonicalize to the same physical labeling.
    cols, rows = COLS, ROWS
    board = _flat_checkerboard(cols, rows)
    dark = np.where(board < 128, 40, 140).astype(np.uint8)
    bright = np.where(board < 128, 150, 250).astype(np.uint8)

    found_d, corners_d, _ = find_chessboard_corners(cv2.cvtColor(dark, cv2.COLOR_GRAY2BGR), (cols, rows))
    found_b, corners_b, _ = find_chessboard_corners(cv2.cvtColor(bright, cv2.COLOR_GRAY2BGR), (cols, rows))
    assert found_d and found_b
    assert np.max(np.abs(corners_d - corners_b)) < 1.0


def test_end_to_end_calibration_survives_a_raw_detector_flip(tmp_path, monkeypatch):
    # The real bug report: calibrate_stereo must produce a good calibration
    # even when the underlying OpenCV detector arbitrarily picks the
    # opposite starting corner for one specific real image (simulated here
    # by intercepting the actual cv2 call and reversing its result for one
    # image only) -- this is what "L10.png and R13.png have corner indices
    # reversed" looks like at the detector level. This exercises the full
    # pipeline (find_chessboard_corners's internal canonicalization, called
    # from inside calibrate_stereo), not just the canonicalization function
    # in isolation.
    left_paths, right_paths = _build_synthetic_stereo_dataset(tmp_path, n_pairs=8)

    real_sb = cv2.findChessboardCornersSB
    call_count = {"n": 0}
    flip_at_call = 6  # lands on some specific image among the 16 (8 left + 8 right)

    def flipping_sb(image, pattern_size, flags=0):
        call_count["n"] += 1
        found, corners = real_sb(image, pattern_size, flags=flags)
        if found and call_count["n"] == flip_at_call:
            corners = corners[::-1].copy()
        return found, corners

    monkeypatch.setattr(cv2, "findChessboardCornersSB", flipping_sb)

    calib = calibrate_stereo(left_paths, right_paths, (COLS, ROWS), SQUARE)
    assert call_count["n"] > 0
    assert calib.quality_ok
    assert calib.rms_error < 1.0
    assert abs(calib.T.ravel()[0] - _TRUE_BASELINE_X) < 5.0


def test_filename_matching_ignores_input_order():
    left = ["L1.png", "L2.png", "L3.png", "L10.png", "L17.png"]
    right = ["R17.png", "R3.png", "R1.png", "R10.png", "R2.png"]  # deliberately shuffled
    m = match_stereo_pairs_by_filename(left, right)
    assert [nid for nid, _, _ in m.pairs] == [1, 2, 3, 10, 17]
    for nid, lp, rp in m.pairs:
        assert lp == f"L{nid}.png"
        assert rp == f"R{nid}.png"
    assert not m.unmatched_left and not m.unmatched_right


def test_filename_matching_flags_unmatched_and_duplicates():
    left = ["L1.png", "L2.png", "L2_dup.png"]
    right = ["R1.png", "R99.png"]
    m = match_stereo_pairs_by_filename(left, right)
    assert m.pairs == [(1, "L1.png", "R1.png")]
    assert m.unmatched_left == ["L2.png", "L2_dup.png"] or set(m.unmatched_left) == {"L2.png", "L2_dup.png"}
    assert m.unmatched_right == ["R99.png"]
    assert 2 in m.duplicate_left_ids


def test_calibrate_stereo_pairs_by_filename_not_list_order(tmp_path):
    # Integration-level version of test_filename_matching_ignores_input_order:
    # feed calibrate_stereo the right-image list in a shuffled order and
    # confirm it still calibrates correctly, proving calibrate_stereo
    # actually uses filename-based pairing rather than zipping by position.
    left_paths, right_paths = _build_synthetic_stereo_dataset(tmp_path, n_pairs=6)
    shuffled_right = list(reversed(right_paths))
    assert shuffled_right != right_paths

    calib = calibrate_stereo(left_paths, shuffled_right, (COLS, ROWS), SQUARE)
    assert calib.quality_ok
    assert calib.rms_error < 1.0
    assert abs(calib.T.ravel()[0] - _TRUE_BASELINE_X) < 5.0


def test_calibration_quality_gate_rejects_too_few_views(tmp_path):
    left_paths, right_paths = _build_synthetic_stereo_dataset(tmp_path, n_pairs=3)
    with pytest.raises(CalibrationQualityError) as excinfo:
        calibrate_stereo(left_paths, right_paths, (COLS, ROWS), SQUARE, min_valid_views=5)
    assert "valid" in str(excinfo.value).lower()
    assert excinfo.value.report is not None


def test_calibration_quality_gate_rejects_high_rms(tmp_path, monkeypatch):
    # Force a bad calibration by corrupting one pair's right-image corners
    # with random noise (simulating a badly-detected/blurred view that
    # slipped past detection) and confirm the quality gate catches the
    # resulting high reprojection error rather than returning it silently.
    left_paths, right_paths = _build_synthetic_stereo_dataset(tmp_path, n_pairs=8)

    import pydic.calibration as calib_mod
    real_find = calib_mod.find_chessboard_corners
    call_count = {"n": 0}

    def noisy_find(image, pattern_size, *args, **kwargs):
        call_count["n"] += 1
        found, corners, flipped = real_find(image, pattern_size, *args, **kwargs)
        if found and call_count["n"] == 3:
            rng = np.random.default_rng(1)
            corners = corners + rng.uniform(-40, 40, corners.shape).astype(np.float32)
        return found, corners, flipped

    monkeypatch.setattr(calib_mod, "find_chessboard_corners", noisy_find)

    with pytest.raises(CalibrationQualityError) as excinfo:
        calibrate_stereo(left_paths, right_paths, (COLS, ROWS), SQUARE, max_pair_reproj_error=2.0)
    bad_calib = excinfo.value.calibration
    assert bad_calib is not None
    assert not bad_calib.quality_ok
    assert bad_calib.quality_issues


def test_calibration_quality_gate_can_be_bypassed_for_inspection(tmp_path):
    left_paths, right_paths = _build_synthetic_stereo_dataset(tmp_path, n_pairs=3)
    calib = calibrate_stereo(
        left_paths, right_paths, (COLS, ROWS), SQUARE,
        min_valid_views=5, raise_on_poor_quality=False,
    )
    assert not calib.quality_ok
    assert calib.quality_issues


def test_stereo_dic_refuses_poor_quality_calibration(tmp_path):
    left_paths, right_paths = _build_synthetic_stereo_dataset(tmp_path, n_pairs=3)
    calib = calibrate_stereo(
        left_paths, right_paths, (COLS, ROWS), SQUARE,
        min_valid_views=5, raise_on_poor_quality=False,
    )
    assert not calib.quality_ok
    with pytest.raises(ValueError):
        StereoDic(calib)
    # explicit override still works, for a user who understands the risk
    StereoDic(calib, allow_poor_quality_calibration=True)
