import json
from pathlib import Path

import pytest

from go1_benchmark.evaluate_mjpc import force_at, pulse_impulse
from go1_benchmark.paper_dataset import trials


@pytest.mark.parametrize("shape,scale", [("rectangular", 1), ("triangular", 0.5)])
@pytest.mark.parametrize("force", [0, 50, 100, 150])
def test_impulse(shape, scale, force):
    area = sum(
        force_at(5 + (i + 0.5) * 0.004, force, 5, 0.2, shape) * 0.004 for i in range(50)
    )
    assert pulse_impulse(force, 0.2, shape) == pytest.approx(force * 0.2 * scale)
    assert area == pytest.approx(pulse_impulse(force, 0.2, shape))


@pytest.mark.parametrize(
    "force,duration,shape",
    [
        (-1, 0.2, "triangular"),
        (10, 0, "triangular"),
        (10, 0.2, "unknown"),
        (float("nan"), 0.2, "triangular"),
    ],
)
def test_invalid_impulse(force, duration, shape):
    with pytest.raises(ValueError):
        pulse_impulse(force, duration, shape)


def test_triangular_plan():
    root = Path(__file__).resolve().parents[1]
    protocol = json.loads((root / "configs/paper_triangular_protocol.json").read_text())
    plan = list(trials(protocol))
    assert len(plan) == len({t["id"] for t in plan}) == 36
    assert all(t["force_n"] > 0 and t["duration_s"] == 12 for t in plan)
    assert protocol["pulse_shape"] == "triangular"
