"""Configuration, pulse and optional actual-native-MJPC integration checks."""

import copy
import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from go1_benchmark.evaluate_mjpc import force_at, json_finite, walking_check
from go1_benchmark.mjpc_model import TERMS, task_xml, validate_config

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def config():
    return json.loads((ROOT / "configs/mjpc_go1.json").read_text())


def test_defaults_and_sensor_order(config):
    validate_config(config)
    root = ET.fromstring(
        task_xml('<mujoco><include file="robot.xml"/></mujoco>', config)
    )
    assert root[0].tag == "sensor"
    assert [s.attrib["name"] for s in root[0]] == list(TERMS)
    assert sum(int(s.attrib["dim"]) for s in root[0]) == 58
    params = [
        n.attrib["name"]
        for n in root.find("custom")
        if n.attrib["name"].startswith("residual_")
    ]
    assert params == [
        "residual_vx",
        "residual_vy",
        "residual_height",
        "residual_period",
        "residual_swing_fraction",
        "residual_swing_height",
        "residual_calf_margin",
    ]


@pytest.mark.parametrize(
    "key,value",
    [
        ("samples", 129),
        ("samples", 1),
        ("spline_points", 37),
        ("threads", 0),
        ("horizon_s", 0.012),
        ("interpolation", "bad"),
    ],
)
def test_invalid_planner(config, key, value):
    invalid = copy.deepcopy(config)
    invalid["planner"][key] = value
    with pytest.raises(ValueError):
        validate_config(invalid)


def test_pulse_area_and_bounds():
    dt = 0.004
    for shape, impulse in (("triangular", 15), ("rectangular", 30)):
        area = sum(
            force_at(5 + (i + 0.5) * dt, 150, 5, 0.2, shape) * dt for i in range(50)
        )
        assert area == pytest.approx(impulse)
        assert force_at(4.99, 150, 5, 0.2, shape) == 0
        assert force_at(5.21, 150, 5, 0.2, shape) == 0


def test_nonfinite_failure_report():
    assert json_finite({"rmse": float("nan"), "values": [float("inf"), 1.0]}) == {
        "rmse": None,
        "values": [None, 1.0],
    }


@pytest.mark.parametrize(
    "key,value",
    [("solver_iterations", 0), ("solver_iterations", True), ("disable_warmstart", 1)],
)
def test_invalid_solver_settings(config, key, value):
    config["environment"][key] = value
    with pytest.raises(ValueError):
        validate_config(config)


def test_extended_terms_optional(config):
    config["task"]["weights"].update({"SupportForce": 300, "NonfootCollision": 100})
    config["task"]["gait_startup_s"] = 1.0
    validate_config(config)
    config["task"]["gait_startup_s"] = -1
    with pytest.raises(ValueError):
        validate_config(config)


def test_walking_not_just_speed():
    summary = {
        "nominal_pass": True,
        "gait": {
            "steady_contact_phase_match_fraction": 0.9,
            "steady_exactly_one_swing_fraction": 0.7,
            "steady_all_feet_airborne_fraction": 0.0,
        },
    }
    assert walking_check(summary)
    summary["gait"]["steady_all_feet_airborne_fraction"] = 0.02
    assert not walking_check(summary)
    summary["nominal_pass"] = False
    assert not walking_check(summary)


def test_actual_native_planner_and_residual(config, tmp_path):
    library = os.environ.get("GO1_MJPC_LIBRARY")
    if not library:
        pytest.skip("Set GO1_MJPC_LIBRARY after compiling to test real MJPC")
    import mujoco
    import numpy as np

    from go1_benchmark.mjpc_controller import MJPCController
    from go1_benchmark.mjpc_model import make_model, reference_residual

    model = make_model(config)
    model_path = tmp_path / "model.mjb"
    mujoco.mj_saveModel(model, str(model_path))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    with MJPCController(Path(library), model_path, config) as controller:
        for phase in (0, 0.13, 0.41, 0.79):
            data.qpos[:] = model.key("home").qpos
            mujoco.mju_axisAngle2Quat(data.qpos[3:7], np.array([0.0, 0.0, 1.0]), 0.1)
            data.qpos[7:] += np.linspace(-0.03, 0.03, model.nu)
            data.time = phase
            data.qvel[:] = np.linspace(-0.1, 0.1, model.nv)
            data.ctrl[:] = model.key("home").ctrl + 0.01
            mujoco.mj_forward(model, data)
            np.testing.assert_allclose(
                controller.residual(data),
                reference_residual(model, data, config),
                atol=1e-10,
            )
        action, cost = controller.action(data)
        assert np.isfinite(action).all() and np.isfinite(cost)
        assert np.all(action >= model.actuator_ctrlrange[:, 0])
        assert np.all(action <= model.actuator_ctrlrange[:, 1])
        data.qpos[:] = model.key("home").qpos
        data.qpos[2] = 0.13
        data.qvel[:] = 0
        data.ctrl[:] = model.key("home").ctrl
        mujoco.mj_forward(model, data)
        np.testing.assert_allclose(
            controller.residual(data),
            reference_residual(model, data, config),
            atol=1e-10,
        )
        assert controller.residual(data)[-1] > 0
