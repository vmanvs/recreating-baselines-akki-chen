"""Load all frozen actors and validate a short native integration trace."""

import json
from pathlib import Path

import mujoco
import numpy as np

from go1_benchmark.mjpc_model import make_model
from go1_benchmark.native_ppo import NativePPO

config = json.loads(Path("configs/mjpc_go1.json").read_text())
model = make_model(config)
for directory in (
    "ppo-seed-000",
    "ppo-gait-aware-seed-0",
    "ppo-gait-calf-clearance-seed-000",
):
    policy = NativePPO(Path(directory), model, config["environment"])
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    mujoco.mj_forward(model, data)
    for _ in range(5):
        target, raw = policy.action(data)
        assert target.shape == (12,) and np.isfinite(target).all()
        assert np.max(np.abs(raw)) <= 1.00001
        data.ctrl[:] = target
        for _ in range(5):
            mujoco.mj_step(model, data)
            mujoco.mj_forward(model, data)
    print(
        directory,
        policy.checkpoint.name,
        policy.provenance["checkpoint_sha256"],
        float(data.time),
        flush=True,
    )
