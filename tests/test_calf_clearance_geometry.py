"""Check the geometric proxy without starting a MuJoCo environment."""

import pytest

from go1_benchmark.fixed_velocity_env import _capsule_floor_clearance


def test_calf2_capsule_clearance_uses_lower_transformed_endpoint():
    np = pytest.importorskip("numpy")
    endpoints = np.asarray(((0.02, 0.0, -0.13), (0.0, 0.0, -0.2)))
    body_height = np.asarray((0.3, 0.3, 0.3))
    rotation_z_rows = np.asarray(
        ((0.0, 0.0, 1.0), (0.0, 0.0, -1.0), (1.0, 0.0, 0.0))
    )

    clearance = _capsule_floor_clearance(
        body_height, rotation_z_rows, endpoints, radius=0.01, floor_height=0.02
    )

    np.testing.assert_allclose(clearance, (0.07, 0.40, 0.27), atol=1e-12)
