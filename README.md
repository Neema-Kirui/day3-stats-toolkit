# Statistical Analysis Toolkit: Wakala Plus M-Pesa Agents

## Overview
Wakala Plus's risk team watches daily deposit totals per agent. This project is a small statistics toolkit written in plain Python (mean, median, mode, variance, standard deviation, quartiles and IQR outlier detection), wrapped in a `StatisticsAnalyzer` class. NumPy is used only afterwards to prove the results right. The toolkit is covered by 13 unit tests.

## Project structure
```
day3-stats-toolkit/
├── stats_toolkit.py
├── test_stats_toolkit.py
├── data/agent_deposits.csv
├── output/stats_report.txt
├── requirements.txt          # numpy (verification only), pytest if used
└── README.md
```

## How to run
From inside the `day3-stats-toolkit` folder:

1. Install NumPy (only needed for the verification block): `python -m pip install -r requirements.txt`
2. Run the report: `python stats_toolkit.py`. It prints the report and saves it to `output/stats_report.txt`.
3. Run the tests: `python -m unittest -v` (or `pytest`).

The tests and the statistics need only the standard library. If NumPy is not available, the script prints that verification was skipped and does **not** overwrite an existing `output/stats_report.txt`, because the verification is part of the report. (NumPy is blocked by a security policy on my own laptop, so the saved report was generated where NumPy is available.)

## Approach
- **Task 1, mean, median, mode:** all use plain loops and sorting. Median averages the two middle values for an even count. `mode` returns a sorted list of every value with the highest count. **If every value occurs exactly once, there is no mode and it returns an empty list** (shown as `none` in the report).
- **Task 2, variance and standard deviation:** both support `sample=True` (divide by n - 1) and `sample=False` (divide by n). A sample of one value raises a clear `ValueError`, because n - 1 would be zero.
- **Task 3, loading:** `load_column` returns the valid floats and a list of `(row_number, raw_value)` pairs. The report also shows the reason for each rejection.
- **Task 4, quartiles and outliers:** `quartiles` and `iqr_outliers` as described below.
- **Task 5, class:** `StatisticsAnalyzer` has `from_csv`, `summary()`, `without_outliers()` (returns a new analyzer) and `report()`. The data is sorted once in `__init__`, and the summary and fences are cached, so nothing is re-sorted on each call.
- **Task 6, NumPy:** `compare_with_numpy` is the only place NumPy is imported. It compares with `numpy.isclose` (a tolerance), never `==`.
- **Task 7, report:** printed and saved to `output/stats_report.txt`, with all data and outliers-removed side by side.
- **Task 8, tests:** 13 small, hand-checkable tests with no dependence on the dataset.

## My choices
**Sample vs population (n - 1 vs n).** I use the **sample** formulas. The 40 agent-days are a small slice of everything Wakala Plus's agents could deposit, and the risk team wants to say something about agents in general, not only about these 40 days. Dividing by n - 1 corrects for the fact that the mean is estimated from the same data. Population versions are available with `sample=False`.

**Quartile method.** Linear interpolation between closest ranks: the position of quantile p is `p * (n - 1)` (zero-based), and the value is interpolated between the two neighbouring sorted values. This is Hyndman & Fan type 7, the default in NumPy and Excel's `QUARTILE.INC`. My NumPy comparison uses `method="linear"`, which is the same method. Other methods (for example "median of the halves") can give slightly different quartiles, so the comparison would show MISMATCH if the methods did not agree.

**Outlier rule.** A value is an outlier if it is below `Q1 - 1.5 x IQR` or above `Q3 + 1.5 x IQR` (k = 1.5 by default and adjustable).

**What counts as invalid.** A value is rejected, with a reason, if it is:
- blank (rows 6 and 28)
- a placeholder such as `-` or `n/a` (rows 9 and 13)
- not a valid number, such as `12,40O` (row 24, which has the letter O in place of a zero)
- zero, negative or not finite (none in this file, but they are rejected, because a deposit total must be positive)

One value is **repaired**: a correctly formatted thousands separator like `12,400` becomes 12400, because the meaning is certain. I do **not** repair `12,40O`. It looks like a typo for 12,400, but guessing would put an invented number into a risk analysis.

Row numbers are data rows, where 1 is the first row after the header. In this file, row N is the N-th day, so row 6 is 2026-09-06.

## Findings
- **Data:** 40 rows read, 35 valid, 5 rejected (rows 6, 9, 13, 24, 28).
- **All data:** mean KES 22,400.00, median 18,805.47, sample standard deviation 18,111.32.
- **Outliers removed (4 values):** mean falls to 18,349.65 and standard deviation to 3,180.63. The median barely changes (18,805.47 to 18,506.72), which shows why the median is more robust than the mean here.
- **No mode:** the deposit totals are recorded to the cent, and no value repeats, so there is no mode in either version of the data.
- **Outlier fences:** lower 8,530.19 and upper 28,445.41. The outliers are 96,500.00 (AG-104, 2026-09-20), 1,200.00 (AG-104, 2026-10-02), 88,400.00 (AG-104, 2026-10-06) and 29,060.99 (AG-101, 2026-10-07).

**For the risk team:** four days fall outside the fences, and three belong to agent AG-104 in Nakuru: two extreme highs and one extreme low. The fourth, AG-101's 29,060.99, is only just above the upper fence. The very high days could mean fraud, a large one-off deposit or an extra digit typed by mistake, and the very low day could mean a float shortage, an app outage or a missing digit. AG-104's records should be reviewed first.

## NumPy verification
Mean, median, variance (ddof=1), standard deviation and Q1/Q3 (method=linear) all show **MATCH**. There were no mismatches to explain. If variance had mismatched, the first thing to check would be `ddof` (NumPy defaults to 0, population), and for quartiles, the interpolation method.

## Assumptions
- Deposit totals must be positive, so zero and negative values are treated as invalid.
- Only the `daily_deposits_kes` column is analysed. All agents are pooled together, so the statistics describe the whole network, not individual agents.
- The IQR rule flags values that are unusual, not necessarily wrong. Whether they are errors or real events is a judgement for the risk team.
- The risk-team notes in the report are written for this dataset and would need updating if the data changes.
