"""Go1 model and task specification for the actual MJPC sampling planner.

Does not import JAX or allocate accelerator memory. User sensors are cost
metadata only; they do not change the Playground full-collision physics.
"""

from __future__ import annotations

import importlib.util
import math
import xml.etree.ElementTree as ET
from pathlib import Path

MJPC_REVISION = "ff572a21e7c2bf9fda62e1862a758da7e9a8719b"
FEET = ("FR", "FL", "RR", "RL")
TERMS = {
    "Height": 1,
    "Velocity": 2,
    "Upright": 3,
    "Heading": 2,
    "Effort": 12,
    "Posture": 12,
    "Gait": 4,
    "CalfClearance": 4,
}


def validate_config(config):
    env, planner, task = (config[key] for key in ("environment", "planner", "task"))
    numeric = [
        env["simulation_dt_s"],
        env["control_dt_s"],
        env["kp"],
        planner["horizon_s"],
        task["target_height_m"],
        task["gait_period_s"],
        task["swing_height_m"],
        task["calf_clearance_margin_m"],
    ]
    if any(not math.isfinite(float(x)) or float(x) <= 0 for x in numeric):
        raise ValueError("Times, gains and lengths must be positive and finite")
    if not math.isfinite(env["kd"]) or env["kd"] < 0:
        raise ValueError("kd must be nonnegative and finite")
    for key in ("samples", "spline_points", "iterations_per_control", "threads"):
        value = planner[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"planner.{key} must be a positive integer")
    if not 2 <= planner["samples"] <= 128:
        raise ValueError("Upstream sampling accepts 2..128 candidates here")
    if not 2 <= planner["spline_points"] <= 36:
        raise ValueError("Use 2..36 spline points")
    if planner["interpolation"] not in ("zero", "linear", "cubic"):
        raise ValueError("Invalid spline interpolation")
    if not 0 <= planner["exploration_fraction"] <= 1:
        raise ValueError("exploration_fraction must be in [0, 1]")
    ratio = env["control_dt_s"] / env["simulation_dt_s"]
    horizon_ratio = planner["horizon_s"] / env["simulation_dt_s"]
    if not math.isclose(ratio, round(ratio)):
        raise ValueError("Control dt must be an integer multiple of physics dt")
    if not math.isclose(horizon_ratio, round(horizon_ratio)):
        raise ValueError("Horizon must be an integer multiple of physics dt")
    if planner["horizon_s"] < env["control_dt_s"] or round(horizon_ratio) + 1 > 512:
        raise ValueError("Horizon must cover a control interval and fit 512 states")
    command = env["command_m_s_rad_s"]
    if len(command) != 3 or not all(math.isfinite(x) for x in command):
        raise ValueError("Expected finite vx, vy, yaw command")
    if command[2] != 0:
        raise ValueError("This straight-walk task requires zero yaw command")
    if env["evaluation_collisions"] != "full":
        raise ValueError("MJPC benchmark requires full collisions")
    if len(task["swing_order"]) != 4 or set(task["swing_order"]) != set(FEET):
        raise ValueError("swing_order must contain each foot exactly once")
    if not 0 < task["swing_fraction"] <= 0.25:
        raise ValueError("swing_fraction must be in (0, .25]")
    if set(task["weights"]) != set(TERMS) or any(
        not math.isfinite(x) or x < 0 for x in task["weights"].values()
    ):
        raise ValueError("Expected all eight nonnegative, finite cost weights")


def task_xml(scene_xml: str, config: dict) -> str:
    """Prepend residual sensors before the included native sensor declarations."""
    validate_config(config)
    root = ET.fromstring(scene_xml)
    sensor = ET.Element("sensor")
    for name, dim in TERMS.items():
        weight = config["task"]["weights"][name]
        ET.SubElement(
            sensor,
            "user",
            name=name,
            dim=str(dim),
            needstage="acc",
            user=f"0 {weight} 0 {max(weight, 100)}",
        )
    root.insert(0, sensor)
    custom = ET.SubElement(root, "custom")
    p, t, e = config["planner"], config["task"], config["environment"]
    values = {
        "agent_horizon": p["horizon_s"],
        "sampling_trajectories": p["samples"],
        "sampling_spline_points": p["spline_points"],
        "sampling_representation": ("zero", "linear", "cubic").index(
            p["interpolation"]
        ),
        "sampling_exploration": p["exploration_fraction"],
        # Order is an ABI with go1_task.cc. Keep exactly seven residual numerics.
        "residual_vx": e["command_m_s_rad_s"][0],
        "residual_vy": e["command_m_s_rad_s"][1],
        "residual_height": t["target_height_m"],
        "residual_period": t["gait_period_s"],
        "residual_swing_fraction": t["swing_fraction"],
        "residual_swing_height": t["swing_height_m"],
        "residual_calf_margin": t["calf_clearance_margin_m"],
    }
    for name, value in values.items():
        ET.SubElement(custom, "numeric", name=name, data=str(value))
    offsets = [t["swing_order"].index(foot) / 4 for foot in FEET]
    ET.SubElement(
        custom, "numeric", name="go1_gait_offsets", data=" ".join(map(str, offsets))
    )
    return ET.tostring(root, encoding="unicode")


def make_model(config, menagerie_root: Path | None = None):
    import mujoco

    spec = importlib.util.find_spec("mujoco_playground")
    if spec is None:
        raise RuntimeError(
            "Install the pinned Playground dependency: uv sync --extra rl"
        )
    package = Path(next(iter(spec.submodule_search_locations)))
    xml_dir = package / "_src/locomotion/go1/xmls"
    if menagerie_root is None:
        candidates = [
            package / "external_deps/mujoco_menagerie",
            Path.cwd() / "mujoco_menagerie",
        ]
        menagerie_root = next(
            (p for p in candidates if (p / "unitree_go1/assets").is_dir()), None
        )
    if menagerie_root is None or not (menagerie_root / "unitree_go1/assets").is_dir():
        raise FileNotFoundError(
            "Go1 mesh assets missing. Supply --menagerie-root pointing to a MuJoCo Menagerie checkout, or use the assets already downloaded by PPO."
        )
    assets = {}
    for directory in (
        xml_dir,
        xml_dir / "assets",
        menagerie_root / "unitree_go1",
        menagerie_root / "unitree_go1/assets",
    ):
        if directory.is_dir():
            for path in directory.iterdir():
                if path.is_file():
                    assets[path.name] = path.read_bytes()
    scene = (xml_dir / "scene_mjx_fullcollisions_flat_terrain.xml").read_text()
    model = mujoco.MjModel.from_xml_string(task_xml(scene, config), assets=assets)
    env = config["environment"]
    model.opt.timestep = env["simulation_dt_s"]
    model.opt.ccd_iterations = 20
    model.dof_damping[6:] = env["kd"]
    model.actuator_gainprm[:, 0] = env["kp"]
    model.actuator_biasprm[:, 1] = -env["kp"]
    validate_model(model)
    return model


def validate_model(model):
    import mujoco

    if (model.nq, model.nv, model.nu, model.na) != (19, 18, 12, 0):
        raise ValueError("Unexpected Go1 dimensions")
    for name in ("trunk", *(f"{foot}_calf" for foot in FEET)):
        model.body(name)
    for foot in FEET:
        model.site(foot)
        model.geom(foot)
    model.geom("floor")
    model.sensor("local_linvel")
    model.key("home")
    for i, (name, dim) in enumerate(TERMS.items()):
        if (
            model.sensor_type[i] != mujoco.mjtSensor.mjSENS_USER
            or model.sensor(i).name != name
            or model.sensor_dim[i] != dim
        ):
            raise ValueError(
                "Residual sensors must be first, with matching names and dimensions"
            )
    for i, foot in enumerate(FEET):
        for j, suffix in enumerate(("hip", "thigh", "calf")):
            joint = model.joint(f"{foot}_{suffix}_joint").id
            if model.actuator_trnid[3 * i + j, 0] != joint:
                raise ValueError("Unexpected actuator order")


def calf_clearance(model, data):
    import numpy as np

    endpoints = np.array(((0.02, 0, -0.13), (0, 0, -0.2)))
    floor = model.geom("floor").pos[2]
    return np.array(
        [
            (
                data.body(f"{foot}_calf").xpos[2]
                + endpoints @ data.body(f"{foot}_calf").xmat.reshape(3, 3)[2]
            ).min()
            - 0.01
            - floor
            for foot in FEET
        ]
    )


def reference_residual(model, data, config):
    """Independent Python reference used to verify the native residual ABI."""
    import numpy as np

    t, e = config["task"], config["environment"]
    rot = data.body("trunk").xmat.reshape(3, 3)
    floor = model.geom("floor").pos[2]
    phase = data.time / t["gait_period_s"] % 1
    offsets = np.array([t["swing_order"].index(f) / 4 for f in FEET])
    p = (phase - offsets) % 1
    swing = np.where(
        p < t["swing_fraction"],
        t["swing_height_m"] * np.sin(np.pi * p / t["swing_fraction"]),
        0,
    )
    limits = np.maximum(np.abs(model.actuator_forcerange).max(axis=1), 1)
    return np.concatenate(
        (
            [data.body("trunk").xpos[2] - floor - t["target_height_m"]],
            data.sensor("local_linvel").data[:2] - np.array(e["command_m_s_rad_s"][:2]),
            rot[:, 2] - [0, 0, 1],
            rot[:2, 0] - [1, 0],
            data.actuator_force / limits,
            data.qpos[7:] - model.key("home").qpos[7:],
            np.array(
                [data.site(f).xpos[2] - floor - model.geom(f).size[0] for f in FEET]
            )
            - swing,
            np.maximum(t["calf_clearance_margin_m"] - calf_clearance(model, data), 0),
        )
    )
