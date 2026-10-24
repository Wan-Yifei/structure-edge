"""The two pieces of book_pressure.py that are easy to get silently wrong.

The SQL and the archive need a database; these do not. What they cover
is the arithmetic that decides whether the headline numbers mean
anything: the leave-one-day-out fit must actually hold a day out, and
the bootstrap must resample DAYS rather than rows -- minutes inside a
session are anything but independent, and a row-wise interval would come
back several times too narrow and make a null look like a finding.
"""
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from backtest.book_pressure import day_bootstrap, loo_r2  # noqa: E402


def days(n_days, per_day):
    return np.repeat(np.arange(n_days), per_day)


class TestLooR2:
    def test_a_real_relation_scores_positive(self):
        rng = np.random.default_rng(1)
        d = days(6, 200)
        x = rng.normal(size=len(d))
        y = 0.8 * x + rng.normal(scale=0.4, size=len(d))
        assert loo_r2(x, y, d) > 0.5

    def test_noise_does_not(self):
        """The number that matters. An in-sample R² is positive by
        construction on noise; this one must not be."""
        rng = np.random.default_rng(2)
        d = days(6, 200)
        x = rng.normal(size=len(d))
        y = rng.normal(size=len(d))
        assert loo_r2(x, y, d) < 0.01

    def test_a_relation_that_holds_only_within_days_is_not_rewarded(self):
        """Each day gets its own slope, with no consistent sign across
        days. Fitted on the pooled rows this looks like signal; held out
        a day at a time it is not."""
        rng = np.random.default_rng(3)
        xs, ys, ds = [], [], []
        for i in range(8):
            x = rng.normal(size=150)
            sign = 1.0 if i % 2 == 0 else -1.0
            ys.append(sign * x + rng.normal(scale=0.3, size=150))
            xs.append(x)
            ds.append(np.full(150, i))
        r2 = loo_r2(np.concatenate(xs), np.concatenate(ys),
                    np.concatenate(ds))
        assert r2 < 0.05, f"day-specific slopes scored {r2:.3f}"

    def test_one_day_of_data_yields_nothing_rather_than_a_number(self):
        d = days(1, 300)
        rng = np.random.default_rng(4)
        assert np.isnan(loo_r2(rng.normal(size=300), rng.normal(size=300), d))


class TestDayBootstrap:
    def test_the_interval_covers_zero_on_noise(self):
        rng = np.random.default_rng(5)
        d = days(10, 300)
        x = rng.normal(size=len(d))
        y = rng.normal(size=len(d))
        obs, lo, hi = day_bootstrap(x, y, d, reps=600)
        assert lo < 0 < hi

    def test_it_is_wider_than_resampling_rows_would_be(self):
        """The reason it resamples days.

        x carries a day-level level of its own, so "high x" largely means
        "certain days"; y carries a day-level shock. The high-minus-low
        difference is then driven by WHICH DAYS landed on each side, and
        its real uncertainty is a ten-day uncertainty. Resampling rows
        keeps every day present in about its original weight and reports
        an interval far too narrow -- which is how a null gets published
        as a finding.

        A first version of this test gave x no day structure. The day
        shock then fell equally on both halves and cancelled in the
        difference, so row and day intervals agreed -- correctly. The
        clustering has to reach the statistic before it can widen it."""
        rng = np.random.default_rng(6)
        n_days, per = 10, 300
        d = days(n_days, per)
        x = (np.repeat(rng.normal(scale=1.5, size=n_days), per)
             + rng.normal(scale=0.3, size=n_days * per))
        shock = np.repeat(rng.normal(scale=1.5, size=n_days), per)
        y = shock + rng.normal(scale=0.3, size=len(d))
        _, lo, hi = day_bootstrap(x, y, d, reps=800)
        width_days = hi - lo

        rng2 = np.random.default_rng(7)
        cut = np.median(x)
        stats = []
        for _ in range(800):
            idx = rng2.integers(0, len(d), len(d))
            xs, ys = x[idx], y[idx]
            c = np.median(xs)
            stats.append(ys[xs > c].mean() - ys[xs <= c].mean())
        width_rows = np.percentile(stats, 97.5) - np.percentile(stats, 2.5)
        assert width_days > width_rows * 1.5, (
            f"day interval {width_days:.3f} vs row interval "
            f"{width_rows:.3f} -- clustering is not being carried")

    def test_the_observed_difference_is_not_itself_resampled(self):
        """The point estimate has to come from the data, not from the
        bootstrap's mean -- otherwise it drifts with the seed."""
        rng = np.random.default_rng(8)
        d = days(6, 200)
        x = rng.normal(size=len(d))
        y = 2.0 * (x > 0) + rng.normal(scale=0.1, size=len(d))
        o1, _, _ = day_bootstrap(x, y, d, reps=200, seed=1)
        o2, _, _ = day_bootstrap(x, y, d, reps=200, seed=99)
        assert o1 == o2
        assert 1.5 < o1 < 2.5
