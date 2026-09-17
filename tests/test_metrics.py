import unittest

from go1_benchmark.metrics import (
    cost_of_transport,
    disturbance_impulse,
    positive_mechanical_work,
    recovery_time,
    root_mean_square_error,
)


class MetricTests(unittest.TestCase):
    def test_paper_displayed_push_is_30_newton_seconds(self):
        self.assertAlmostEqual(disturbance_impulse(150.0, 0.2), 30.0)

    def test_impulse_uses_force_magnitude(self):
        self.assertAlmostEqual(disturbance_impulse(-150.0, 0.2), 30.0)

    def test_rmse(self):
        self.assertAlmostEqual(root_mean_square_error([0.4, 0.5, 0.6], 0.5),
                               (0.02 / 3.0) ** 0.5)

    def test_positive_work_clips_regeneration(self):
        torques = [[2.0, -3.0], [1.0, 2.0]]
        velocities = [[4.0, 2.0], [-5.0, 3.0]]
        self.assertAlmostEqual(positive_mechanical_work(torques, velocities, 0.1),
                               1.4)

    def test_cost_of_transport(self):
        cot = cost_of_transport([[10.0]], [[2.0]], 1.0, 10.0, 2.0)
        self.assertAlmostEqual(cot, 20.0 / (10.0 * 9.81 * 2.0))

    def test_recovery_requires_dwell(self):
        times = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        errors = [0.0, 2.0, 0.1, 0.8, 0.1, 0.1, 0.1]
        self.assertAlmostEqual(
            recovery_time(times, errors, 0.1, tolerance=0.2, dwell_s=0.2),
            0.3,
        )

    def test_no_demonstrated_recovery_returns_none(self):
        self.assertIsNone(
            recovery_time([0.0, 0.1, 0.2], [2.0, 0.1, 0.1], 0.0, 0.2, 0.2)
        )


if __name__ == "__main__":
    unittest.main()
