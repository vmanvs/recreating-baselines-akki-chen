"""Metric definitions shared by every controller backend.

The functions use plain Python sequences so the evaluation contract can be
tested before MuJoCo, JAX, or controller-specific dependencies are installed.
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import pairwise
from math import sqrt


def disturbance_impulse(force_n: float, duration_s: float) -> float:
    """Return the impulse of a constant force in newton-seconds."""
    if duration_s < 0:
        raise ValueError("duration_s must be non-negative")
    return abs(float(force_n)) * float(duration_s)


def root_mean_square_error(
    observed: Sequence[float], target: float | Sequence[float]
) -> float:
    """Compute RMSE against a scalar or sample-aligned target."""
    if not observed:
        raise ValueError("observed must contain at least one sample")
    if isinstance(target, Sequence):
        if len(observed) != len(target):
            raise ValueError("observed and target must have the same length")
        errors = (float(x) - float(y) for x, y in zip(observed, target))
    else:
        errors = (float(x) - float(target) for x in observed)
    return sqrt(sum(error * error for error in errors) / len(observed))


def positive_mechanical_work(
    torques: Sequence[Sequence[float]],
    joint_velocities: Sequence[Sequence[float]],
    dt_s: float,
) -> float:
    """Integrate positive actuator mechanical power.

    This matches the paper's convention: negative joint power is set to zero,
    modelling no useful regenerative energy recovery.
    """
    if dt_s <= 0:
        raise ValueError("dt_s must be positive")
    if len(torques) != len(joint_velocities):
        raise ValueError("torques and joint_velocities must align in time")

    work = 0.0
    for torque_row, velocity_row in zip(torques, joint_velocities):
        if len(torque_row) != len(velocity_row):
            raise ValueError("torque and velocity rows must have equal width")
        work += sum(
            max(0.0, float(torque) * float(velocity)) * dt_s
            for torque, velocity in zip(torque_row, velocity_row)
        )
    return work


def cost_of_transport(
    torques: Sequence[Sequence[float]],
    joint_velocities: Sequence[Sequence[float]],
    dt_s: float,
    mass_kg: float,
    distance_m: float,
    gravity_m_s2: float = 9.81,
) -> float:
    """Compute positive-work mechanical cost of transport, E/(m*g*d)."""
    if mass_kg <= 0:
        raise ValueError("mass_kg must be positive")
    if distance_m <= 0:
        raise ValueError("distance_m must be positive")
    if gravity_m_s2 <= 0:
        raise ValueError("gravity_m_s2 must be positive")
    work_j = positive_mechanical_work(torques, joint_velocities, dt_s)
    return work_j / (mass_kg * gravity_m_s2 * distance_m)


def recovery_time(
    times_s: Sequence[float],
    errors: Sequence[float],
    disturbance_end_s: float,
    tolerance: float,
    dwell_s: float,
) -> float | None:
    """Return the first post-disturbance time that remains within tolerance.

    Recovery requires all samples over the following ``dwell_s`` interval to
    remain within tolerance. ``None`` means recovery was not demonstrated in
    the recorded window.
    """
    if len(times_s) != len(errors) or not times_s:
        raise ValueError("times_s and errors must be non-empty and aligned")
    if tolerance < 0 or dwell_s < 0:
        raise ValueError("tolerance and dwell_s must be non-negative")
    if any(t1 >= t2 for t1, t2 in pairwise(times_s)):
        raise ValueError("times_s must be strictly increasing")

    for start_index, (time_s, error) in enumerate(zip(times_s, errors)):
        if time_s < disturbance_end_s or abs(float(error)) > tolerance:
            continue
        dwell_end = time_s + dwell_s
        covered = False
        valid = True
        for check_time, check_error in zip(
            times_s[start_index:], errors[start_index:]
        ):
            if abs(float(check_error)) > tolerance:
                valid = False
                break
            # Decimal sample periods such as 0.1 are not exact binary floats.
            if check_time + 1e-12 >= dwell_end:
                covered = True
                break
        if valid and covered:
            return time_s - disturbance_end_s
    return None
