"""Unit tests for the toolkit. Run with:  python -m unittest -v   (or pytest)."""
import tempfile
import unittest
from pathlib import Path

from stats_toolkit import (mean, median, mode, variance, std_dev, quartiles,
                           iqr_outliers, load_column, StatisticsAnalyzer)


class TestCentralTendency(unittest.TestCase):
    def test_mean_of_known_values(self):
        self.assertEqual(mean([2, 4, 6]), 4)

    def test_median_even_and_odd_lengths(self):
        self.assertEqual(median([3, 1, 2]), 2)          # odd, unsorted input
        self.assertEqual(median([4, 1, 3, 2]), 2.5)     # even: average of middle two

    def test_mode_with_ties(self):
        self.assertEqual(mode([1, 1, 2, 2, 3]), [1, 2])   # two modes
        self.assertEqual(mode([5, 5, 7]), [5])            # one mode
        self.assertEqual(mode([1, 2, 3]), [])             # every value once -> none

    def test_empty_input_raises(self):
        for func in (mean, median, mode, quartiles, iqr_outliers):
            with self.assertRaises(ValueError):
                func([])


class TestSpread(unittest.TestCase):
    DATA = [2, 4, 4, 4, 5, 5, 7, 9]   # mean 5, sum of squares 32

    def test_sample_vs_population_variance(self):
        self.assertAlmostEqual(variance(self.DATA, sample=False), 4.0)
        self.assertAlmostEqual(variance(self.DATA, sample=True), 32 / 7)
        self.assertAlmostEqual(std_dev(self.DATA, sample=False), 2.0)

    def test_std_dev_single_value(self):
        self.assertEqual(std_dev([5], sample=False), 0.0)
        with self.assertRaises(ValueError):
            std_dev([5], sample=True)          # n - 1 would be zero


class TestQuartiles(unittest.TestCase):
    def test_quartiles_linear_method(self):
        # positions 2.25, 4.5, 6.75 in 1..10 -> 3.25, 5.5, 7.75
        self.assertEqual(quartiles(list(range(1, 11))), (3.25, 5.5, 7.75))


class TestOutliers(unittest.TestCase):
    def test_no_outliers_in_uniform_data(self):
        result = iqr_outliers(list(range(1, 11)))
        self.assertEqual(result["outliers"], [])
        self.assertEqual(result["iqr"], 4.5)

    def test_detects_obvious_outlier(self):
        result = iqr_outliers([10, 11, 12, 13, 14, 15, 16, 1000])
        self.assertEqual(result["outliers"], [1000])
        self.assertEqual(result["lower_fence"], 6.5)
        self.assertEqual(result["upper_fence"], 20.5)

    def test_detects_low_outlier(self):
        self.assertEqual(iqr_outliers([-500, 10, 11, 12, 13, 14, 15, 16])["outliers"], [-500])


class TestLoading(unittest.TestCase):
    def test_invalid_and_missing_values_are_rejected(self):
        csv_text = ("date,agent_id,amount\n"
                    "d1,A,100\n"       # row 1 valid
                    "d2,A,\n"          # row 2 blank
                    "d3,A,-\n"         # row 3 placeholder
                    "d4,A,n/a\n"       # row 4 placeholder
                    "d5,A,abc\n"       # row 5 not a number
                    "d6,A,-50\n"       # row 6 negative
                    'd7,A,"1,200"\n'   # row 7 repaired thousands separator
                    "d8,A,200\n")      # row 8 valid
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "data.csv"
            path.write_text(csv_text, encoding="utf-8")
            values, rejected = load_column(path, "amount")
        self.assertEqual(values, [100.0, 1200.0, 200.0])
        self.assertEqual([n for n, _ in rejected], [2, 3, 4, 5, 6])
        self.assertEqual(rejected[0], (2, ""))


class TestAnalyzer(unittest.TestCase):
    def test_without_outliers_returns_new_analyzer(self):
        original = StatisticsAnalyzer([10, 11, 12, 13, 14, 15, 16, 1000])
        cleaned = original.without_outliers()
        self.assertEqual(len(original.values), 8)      # original untouched
        self.assertEqual(len(cleaned.values), 7)
        self.assertEqual(cleaned.summary()["max"], 16)

    def test_summary_matches_functions(self):
        data = [3, 1, 4, 1, 5, 9, 2, 6]
        s = StatisticsAnalyzer(data).summary()
        self.assertEqual(s["mean"], mean(data))
        self.assertEqual(s["median"], median(data))
        self.assertEqual(s["mode"], [1])


if __name__ == "__main__":
    unittest.main()
