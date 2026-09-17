from go1_benchmark.train_ppo import _format_progress


def test_progress_shows_current_and_total_steps():
    assert _format_progress(10_000_000, 200_000_000) == (
        "steps=10,000,000/200,000,000 (10.00M/200.00M, 5.0%)"
    )


def test_progress_uses_custom_budget():
    assert _format_progress(250_000, 1_000_000) == (
        "steps=250,000/1,000,000 (0.25M/1.00M, 25.0%)"
    )


def test_progress_starts_at_zero():
    assert _format_progress(0, 200_000_000).endswith("0.0%)")


def test_progress_preserves_batch_overshoot():
    assert _format_progress(1_100_000, 1_000_000) == (
        "steps=1,100,000/1,000,000 (1.10M/1.00M, 110.0%)"
    )
