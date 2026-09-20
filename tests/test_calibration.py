import cv2
import numpy as np

from pydic.calibration import calibrate_stereo, find_chessboard_corners


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
    found, corners = find_chessboard_corners(bgr, (cols, rows))
    assert found
    assert corners is not None
    assert len(corners) == cols * rows


def test_calibrate_stereo_recovers_known_baseline(tmp_path):
    cols, rows, square = 9, 6, 25.0
    K = np.array([[3200.0, 0, 2000], [0, 3200.0, 1500], [0, 0, 1]])
    D = np.zeros(5)
    img_size = (4000, 3000)
    R2 = cv2.Rodrigues(np.array([0.0, 0.15, 0.0]))[0]
    T2 = np.array([-150.0, 0.0, 0.0])

    rng = np.random.default_rng(0)
    left_dir = tmp_path / "left"
    right_dir = tmp_path / "right"
    left_dir.mkdir()
    right_dir.mkdir()

    left_paths, right_paths = [], []
    for i in range(6):
        rvec1 = (rng.uniform(-0.3, 0.3, 3) + np.array([0.15, 0.1, 0.05])).astype(np.float32)
        tvec1 = np.array([-100 + i * 10, -60 + i * 5, 900 + i * 20], dtype=np.float32)
        left_img = _render_chessboard(rvec1, tvec1, K, D, img_size, cols, rows, square)

        R1, _ = cv2.Rodrigues(rvec1)
        Rc2 = R2 @ R1
        Tc2 = (R2 @ tvec1.reshape(3, 1) + T2.reshape(3, 1)).ravel()
        rvec2, _ = cv2.Rodrigues(Rc2)
        right_img = _render_chessboard(rvec2, Tc2.astype(np.float32), K, D, img_size, cols, rows, square)

        lp = str(left_dir / f"L{i}.png")
        rp = str(right_dir / f"R{i}.png")
        cv2.imwrite(lp, left_img)
        cv2.imwrite(rp, right_img)
        left_paths.append(lp)
        right_paths.append(rp)

    calib = calibrate_stereo(left_paths, right_paths, (cols, rows), square)
    assert calib.rms_error < 1.0
    assert abs(calib.T.ravel()[0] - (-150.0)) < 5.0
