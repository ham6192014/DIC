"""Validation of compute_strain_3d_surface against known ground truth,
including the required rigid-body-motion cases that must return exactly
zero Green-Lagrange strain.

Root cause of the bug this fixes (see strain.py's _local_tangent_basis /
compute_strain_3d_surface docstring): the deformed neighborhood used to be
projected onto the *reference* tangent plane instead of its own. For a
surface subjected to an out-of-plane rigid rotation, that foreshortens the
projected deformed geometry (like a tilted card's shadow), producing a
fictitious strain of 0.5*(cos^2(theta) - 1) for a rotation of angle theta —
exactly -0.125 at theta=30 degrees, matching a real bug report.
"""
from __future__ import annotations

import numpy as np

from pydic.fields import build_neighbors, generate_grid
from pydic.strain import compute_strain_3d_surface


def _flat_grid(step=20, size=(400, 400), subset_radius=15):
    points2d, index_of = generate_grid(size, step=step, subset_radius=subset_radius)
    neighbors = build_neighbors(index_of)
    n = len(points2d)
    ref3d = np.column_stack([points2d[:, 0], points2d[:, 1], np.zeros(n)])
    valid = np.ones(n, dtype=bool)
    return ref3d, neighbors, valid


def _rotation_matrix(axis: str, degrees: float) -> np.ndarray:
    theta = np.radians(degrees)
    c, s = np.cos(theta), np.sin(theta)
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    if axis == "z":
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    raise ValueError(axis)


def _assert_zero_strain(strain, atol=1e-9):
    assert np.all(strain.valid)
    assert np.nanmax(np.abs(strain.exx)) < atol, f"exx not zero: max={np.nanmax(np.abs(strain.exx))}"
    assert np.nanmax(np.abs(strain.eyy)) < atol, f"eyy not zero: max={np.nanmax(np.abs(strain.eyy))}"
    assert np.nanmax(np.abs(strain.exy)) < atol, f"exy not zero: max={np.nanmax(np.abs(strain.exy))}"


def test_zero_deformation_gives_zero_strain():
    ref3d, neighbors, valid = _flat_grid()
    cur3d = ref3d.copy()
    strain = compute_strain_3d_surface(ref3d, cur3d, valid, neighbors, window_hops=2)
    _assert_zero_strain(strain)


def test_rigid_translation_gives_zero_strain():
    ref3d, neighbors, valid = _flat_grid()
    cur3d = ref3d + np.array([12.3, -7.6, 4.1])  # arbitrary 3D translation
    strain = compute_strain_3d_surface(ref3d, cur3d, valid, neighbors, window_hops=2)
    _assert_zero_strain(strain)


def test_in_plane_rigid_rotation_gives_zero_strain():
    ref3d, neighbors, valid = _flat_grid()
    center = ref3d.mean(axis=0)
    R = _rotation_matrix("z", 25.0)  # rotation about the surface normal
    cur3d = (ref3d - center) @ R.T + center
    strain = compute_strain_3d_surface(ref3d, cur3d, valid, neighbors, window_hops=2)
    _assert_zero_strain(strain, atol=1e-7)


def test_out_of_plane_rigid_rotation_gives_zero_strain():
    # This is the exact reported bug: a flat surface rotated 30 degrees
    # out of its own plane used to produce a fictitious strain of -0.125
    # (Green-Lagrange of a rigid rotation must be exactly zero).
    ref3d, neighbors, valid = _flat_grid()
    center = ref3d.mean(axis=0)
    R = _rotation_matrix("x", 30.0)  # tilts the plane out of XY
    cur3d = (ref3d - center) @ R.T + center
    strain = compute_strain_3d_surface(ref3d, cur3d, valid, neighbors, window_hops=2)
    _assert_zero_strain(strain, atol=1e-6)


def test_out_of_plane_rotation_at_several_angles_gives_zero_strain():
    ref3d, neighbors, valid = _flat_grid()
    center = ref3d.mean(axis=0)
    for angle in [5.0, 15.0, 30.0, 45.0, 60.0, 80.0]:
        R = _rotation_matrix("y", angle)
        cur3d = (ref3d - center) @ R.T + center
        strain = compute_strain_3d_surface(ref3d, cur3d, valid, neighbors, window_hops=2)
        assert np.nanmax(np.abs(strain.exx)) < 1e-6, f"angle={angle}: exx not zero"
        assert np.nanmax(np.abs(strain.eyy)) < 1e-6, f"angle={angle}: eyy not zero"
        assert np.nanmax(np.abs(strain.exy)) < 1e-6, f"angle={angle}: exy not zero"


def test_known_uniaxial_strain_matches_analytical_green_lagrange():
    ref3d, neighbors, valid = _flat_grid()
    center = ref3d.mean(axis=0)
    lam = 1.10  # 10% stretch along X
    F_true = np.diag([lam, 1.0, 1.0])
    cur3d = (ref3d - center) @ F_true.T + center

    strain = compute_strain_3d_surface(ref3d, cur3d, valid, neighbors, window_hops=2)
    exx_true = 0.5 * (lam ** 2 - 1)
    assert np.allclose(strain.exx[strain.valid], exx_true, atol=1e-9)
    assert np.allclose(strain.eyy[strain.valid], 0.0, atol=1e-9)
    assert np.allclose(strain.exy[strain.valid], 0.0, atol=1e-9)


def test_known_shear_deformation_matches_analytical_green_lagrange():
    ref3d, neighbors, valid = _flat_grid()
    center = ref3d.mean(axis=0)
    gamma = 0.08
    F_true = np.array([[1.0, gamma, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    cur3d = (ref3d - center) @ F_true.T + center

    strain = compute_strain_3d_surface(ref3d, cur3d, valid, neighbors, window_hops=2)
    exx_true = 0.0
    eyy_true = 0.5 * gamma ** 2
    exy_true = 0.5 * gamma
    assert np.allclose(strain.exx[strain.valid], exx_true, atol=1e-9)
    assert np.allclose(strain.eyy[strain.valid], eyy_true, atol=1e-9)
    assert np.allclose(strain.exy[strain.valid], exy_true, atol=1e-9)


def test_strain_not_computed_across_missing_points():
    # A window with too few valid neighbors (simulating a hole/discontinuity
    # in the point cloud, e.g. a crack or a rejected-match region) must not
    # be filled in with a computed value.
    ref3d, neighbors, valid = _flat_grid()
    cur3d = ref3d.copy()
    valid = valid.copy()
    # Knock out most of the field, leaving isolated points with too few
    # valid neighbors to fit a local plane.
    rng = np.random.default_rng(0)
    keep = rng.choice(len(valid), size=3, replace=False)
    sparse_valid = np.zeros_like(valid)
    sparse_valid[keep] = True

    strain = compute_strain_3d_surface(ref3d, cur3d, sparse_valid, neighbors, window_hops=2, min_points=5)
    assert not np.any(strain.valid), "strain must not be reported where there aren't enough real neighbors"
    assert np.all(np.isnan(strain.exx[~strain.valid]))
