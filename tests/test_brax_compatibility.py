"""Exercise real PPO initialization and updates when RL dependencies are installed."""

import math

import pytest


def test_ppo_initialization_and_updates():
    pytest.importorskip("brax")
    from brax.envs.fast import Fast
    from brax.training.agents.ppo import train as ppo

    steps = []
    _, _, metrics = ppo.train(
        environment=Fast(),
        num_timesteps=16,
        episode_length=4,
        num_envs=2,
        num_eval_envs=2,
        batch_size=2,
        num_minibatches=1,
        unroll_length=2,
        num_updates_per_batch=1,
        num_evals=2,
        max_devices_per_host=1,
        progress_fn=lambda step, metrics: steps.append(int(step)),
    )
    assert steps[-1] >= 16
    assert math.isfinite(float(metrics["eval/episode_reward"]))
