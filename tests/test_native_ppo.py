"""Check the native actor contract against pinned upstream observations."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from go1_benchmark.native_ppo import policy_observation


@pytest.mark.parametrize(
    "directory",
    ["ppo-seed-000", "ppo-gait-aware-seed-0", "ppo-gait-calf-clearance-seed-000"],
)
def test_upstream_observation_and_common_physics(directory):
    import jax
    import jax.numpy as jp
    import mujoco

    from go1_benchmark.fixed_velocity_env import make_fixed_velocity_env
    from go1_benchmark.mjpc_model import make_model

    root = Path(__file__).resolve().parents[1]
    environment = json.loads((root / directory / "manifest.json").read_text())[
        "environment"
    ]
    env = make_fixed_velocity_env(
        command=tuple(environment["command_m_s_rad_s"]),
        full_collisions=True,
        noise_level=0,
        randomize_initial_state=False,
        settings=environment,
    )
    common = make_model(json.loads((root / "configs/mjpc_go1.json").read_text()))
    for field in (
        "body_mass",
        "body_inertia",
        "geom_friction",
        "geom_contype",
        "geom_conaffinity",
        "dof_damping",
        "actuator_gainprm",
        "actuator_biasprm",
        "actuator_ctrlrange",
        "actuator_forcerange",
        "qpos0",
        "jnt_range",
    ):
        np.testing.assert_array_equal(
            getattr(env.mj_model, field), getattr(common, field)
        )
    for field in (
        "timestep",
        "integrator",
        "solver",
        "iterations",
        "ls_iterations",
        "disableflags",
        "cone",
        "gravity",
    ):
        np.testing.assert_array_equal(
            getattr(env.mj_model.opt, field), getattr(common.opt, field)
        )
    data = mujoco.MjData(env.mj_model)
    rng = np.random.default_rng(921)
    for t in (0, 0.18, 0.42, 0.81):
        mujoco.mj_resetDataKeyframe(env.mj_model, data, env.mj_model.key("home").id)
        data.time = t
        data.qpos[7:] += rng.normal(0, 0.02, 12)
        data.qvel[:] = rng.normal(0, 0.1, 18)
        mujoco.mj_forward(env.mj_model, data)
        last_action = rng.normal(0, 0.1, 12).astype(np.float32)
        proxy = SimpleNamespace(
            qpos=jp.asarray(data.qpos),
            qvel=jp.asarray(data.qvel),
            time=jp.asarray(t),
            site_xmat=jp.asarray(data.site_xmat.reshape(-1, 3, 3)),
            sensordata=jp.asarray(data.sensordata),
            actuator_force=jp.asarray(data.actuator_force),
            xfrc_applied=jp.asarray(data.xfrc_applied),
        )
        info = {
            "rng": jax.random.PRNGKey(0),
            "last_act": jp.asarray(last_action),
            "command": jp.asarray(environment["command_m_s_rad_s"]),
            "last_contact": jp.zeros(4),
            "feet_air_time": jp.zeros(4),
            "steps_since_last_pert": jp.asarray(0),
            "steps_until_next_pert": jp.asarray(1000),
        }
        upstream = np.asarray(env._get_obs(proxy, info)["state"])
        actual = policy_observation(env.mj_model, data, environment, last_action)
        np.testing.assert_allclose(actual, upstream, rtol=2e-6, atol=2e-6)


def test_action_history_preserves_upstream_update_order(monkeypatch):
    from go1_benchmark import native_ppo
    from go1_benchmark.native_ppo import NativePPO

    policy = NativePPO.__new__(NativePPO)
    policy.model = SimpleNamespace(key=lambda name: SimpleNamespace(qpos=np.zeros(19)))
    policy.environment = {"action_scale_rad": 0.5}
    policy.jp = np
    policy.state_size = 12
    policy.actor = lambda obs: np.ones(12) * (len(observed) + 1)
    observed = []

    def observation(model, data, environment, previous):
        observed.append(previous.copy())
        return previous

    monkeypatch.setattr(native_ppo, "policy_observation", observation)
    policy.reset()
    raw_actions = []
    for _ in range(4):
        targets, raw = policy.action(None)
        raw_actions.append(raw)
        np.testing.assert_allclose(targets, raw * 0.5)
    np.testing.assert_array_equal(observed[0], np.zeros(12))
    np.testing.assert_array_equal(observed[1], np.zeros(12))
    np.testing.assert_array_equal(observed[2], raw_actions[0])
    np.testing.assert_array_equal(observed[3], raw_actions[1])
