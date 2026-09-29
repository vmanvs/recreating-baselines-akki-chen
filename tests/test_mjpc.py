"""Configuration, pulse and optional actual-native-MJPC integration checks."""

import copy
import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from go1_benchmark.evaluate_mjpc import force_at, json_finite
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
    assert sum(int(s.attrib["dim"]) for s in root[0]) == 40
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
