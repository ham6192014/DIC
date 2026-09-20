"""Image loading, ROI selection and CSV export helpers."""
from __future__ import annotations

import csv
import glob
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

ImageLike = Union[str, "os.PathLike[str]", np.ndarray]


def load_gray(image: ImageLike) -> np.ndarray:
    """Accept a path or an already-loaded array; always return grayscale uint8/float."""
    if isinstance(image, np.ndarray):
        if image.ndim == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image
    img = cv2.imread(str(image), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"could not read image: {image}")
    return img


def load_sequence(folder: str, pattern: str = "*.tif") -> List[str]:
    """Naturally-sorted list of image paths matching pattern in folder."""
    paths = glob.glob(os.path.join(folder, pattern))

    def _key(p: str):
        stem = Path(p).stem
        digits = "".join(ch for ch in stem if ch.isdigit())
        return (int(digits) if digits else 0, stem)

    return sorted(paths, key=_key)


def select_roi_polygon(image: ImageLike) -> List[Tuple[float, float]]:
    """Interactively click a polygon ROI on `image`; double-click/close to finish.

    Requires an interactive matplotlib backend (not for headless/CI use).
    """
    import matplotlib.pyplot as plt

    gray = load_gray(image)
    fig, ax = plt.subplots()
    ax.imshow(gray, cmap="gray")
    ax.set_title("Click ROI polygon vertices, then close the window")
    pts = plt.ginput(n=-1, timeout=0)
    plt.close(fig)
    return pts


def save_fields_csv(path: str, points: np.ndarray, fields: Dict[str, np.ndarray]) -> None:
    """Write a CSV with columns x, y, then one column per entry in `fields`."""
    names = list(fields.keys())
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y", *names])
        for i in range(len(points)):
            row = [points[i, 0], points[i, 1]] + [fields[name][i] for name in names]
            writer.writerow(row)


def save_points3d_csv(path: str, points3d: np.ndarray, fields: Dict[str, np.ndarray]) -> None:
    names = list(fields.keys())
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["X", "Y", "Z", *names])
        for i in range(len(points3d)):
            row = list(points3d[i]) + [fields[name][i] for name in names]
            writer.writerow(row)


def save_json(path: str, data: dict) -> None:
    Path(path).write_text(json.dumps(data, indent=2))


def load_json(path: str) -> dict:
    return json.loads(Path(path).read_text())
