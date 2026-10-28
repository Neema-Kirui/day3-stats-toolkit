"""
Day 3 - Statistical Analysis Toolkit
Squad Python Specialization - Data Science, AI & Analytics

RULE: every calculation is implemented with plain Python.
NumPy is imported ONLY in compare_with_numpy(), to verify the results.
The statistics module is not used.

Choices (also explained in the README):
  - Variance / std dev: sample formula (n - 1) by default, population (n) on request.
  - Quartile method: linear interpolation between closest ranks, position p * (n - 1)
    (Hyndman & Fan type 7, the NumPy default, method="linear").
  - Mode: every value that occurs most often; an empty list when every value occurs once.
  - Outliers: values below Q1 - k*IQR or above Q3 + k*IQR (k = 1.5 by default).
"""
import csv
import math
import re
import subprocess
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "data" / "agent_deposits.csv"
REPORT_PATH = BASE_DIR / "output" / "stats_report.txt"
COLUMN = "daily_deposits_kes"

PLACEHOLDERS = {"-", "--", "n/a", "na", "null", "none", "nan", "?"}
THOUSANDS = re.compile(r"^\d{1,3}(,\d{3})+(\.\d+)?$")


# ------------------------------------------------------------ loading (task 3)
def _parse_value(raw: str) -> tuple[float | None, str | None]:
    """Return (value, None) if raw is a trustworthy number, else (None, reason)."""
    text = (raw or "").strip()
    if text == "":
        return None, "missing value"
    if text.lower() in PLACEHOLDERS:
        return None, "placeholder, not a number"
    if THOUSANDS.match(text):          # "12,400" is safe to repair
        text = text.replace(",", "")
    if any(ch not in "0123456789.-+" for ch in text):
        return None, "not a valid number (contains other characters)"
    try:
        number = float(text)
    except ValueError:
        return None, "not a valid number"
    if not math.isfinite(number):
        return None, "not a finite number"
    if number <= 0:
        return None, "zero or negative deposit"
    return number, None


def _read_column(path: Path, column: str) -> dict:
    """Read one column, keeping labels (date, agent) for each valid value."""
    values, labels, rejected = [], [], []
    rows_read = 0
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if column not in (reader.fieldnames or []):
            raise ValueError(f"column {column!r} not found; columns are {reader.fieldnames}")
        for row_number, row in enumerate(reader, start=1):
            rows_read += 1
            number, reason = _parse_value(row[column])
            if number is None:
                rejected.append((row_number, row[column], reason))
            else:
                values.append(number)
                labels.append((row.get("date", ""), row.get("agent_id", "")))
    return {"values": values, "labels": labels, "rejected": rejected, "rows_read": rows_read}


def load_column(path: Path, column: str) -> tuple[list[float], list[tuple[int, str]]]:
    """Return (valid_values, rejected) where rejected holds (row_number, raw_value) pairs.

    row_number is the data row (1 = first row after the header).
    """
    data = _read_column(path, column)
    return data["values"], [(n, raw) for n, raw, _ in data["rejected"]]


# ---------------------------------------------------- sorted-data helpers (internal)
def _percentile_sorted(s: list[float], p: float) -> float:
    position = p * (len(s) - 1)
    low = int(math.floor(position))
    high = min(low + 1, len(s) - 1)
    return s[low] + (s[high] - s[low]) * (position - low)


def _median_sorted(s: list[float]) -> float:
    middle = len(s) // 2
    if len(s) % 2 == 1:
        return s[middle]
    return (s[middle - 1] + s[middle]) / 2


def _quartiles_sorted(s: list[float]) -> tuple[float, float, float]:
    return _percentile_sorted(s, 0.25), _median_sorted(s), _percentile_sorted(s, 0.75)


def _require(values: list[float], name: str, minimum: int = 1) -> None:
    if len(values) < minimum:
        raise ValueError(f"{name}() needs at least {minimum} value(s), got {len(values)}")


# ------------------------------------------------------- statistics (tasks 1, 2, 4)
def mean(values: list[float]) -> float:
    _require(values, "mean")
    return sum(values) / len(values)


def median(values: list[float]) -> float:
    _require(values, "median")
    return _median_sorted(sorted(values))


def mode(values: list[float]) -> list[float]:
    """Every value that occurs most often, ascending.

    If every value occurs exactly once there is no mode, so an empty list is returned.
    """
    _require(values, "mode")
    counts = {}
    for v in values:
        counts[v] = counts.get(v, 0) + 1
    highest = max(counts.values())
    if highest == 1:
        return []
    return sorted(v for v, c in counts.items() if c == highest)


def variance(values: list[float], sample: bool = True) -> float:
    """Sample variance (n - 1) by default; population variance (n) when sample is False."""
    _require(values, "variance", 2 if sample else 1)
    centre = mean(values)
    total = sum((v - centre) ** 2 for v in values)
    return total / (len(values) - 1 if sample else len(values))


def std_dev(values: list[float], sample: bool = True) -> float:
    return math.sqrt(variance(values, sample))


def quartiles(values: list[float]) -> tuple[float, float, float]:
    """Return (Q1, Q2, Q3) using linear interpolation at position p * (n - 1)."""
    _require(values, "quartiles")
    return _quartiles_sorted(sorted(values))


def _outlier_info(s: list[float], k: float) -> dict:
    q1, _, q3 = _quartiles_sorted(s)
    iqr = q3 - q1
    return {"q1": q1, "q3": q3, "iqr": iqr,
            "lower_fence": q1 - k * iqr, "upper_fence": q3 + k * iqr}


def iqr_outliers(values: list[float], k: float = 1.5) -> dict:
    """Return the IQR, the lower and upper fences, and the values outside them."""
    _require(values, "iqr_outliers")
    if k < 0:
        raise ValueError("k must not be negative")
    info = _outlier_info(sorted(values), k)
    info["outliers"] = [v for v in values
                        if v < info["lower_fence"] or v > info["upper_fence"]]
    return info


# ------------------------------------------------------------ the class (task 5)
class StatisticsAnalyzer:
    """Wraps one numeric column and exposes every statistic above as a method.

    The data is sorted once in __init__; summary() and the outlier fences are cached.
    """

    def __init__(self, values: list[float], name: str = "values", labels=None,
                 rows_read: int | None = None, rejected=None, k: float = 1.5) -> None:
        _require(values, "StatisticsAnalyzer")
        self.name = name
        self.values = list(values)
        self.labels = list(labels) if labels is not None else [("", "")] * len(values)
        self.rows_read = rows_read if rows_read is not None else len(values)
        self.rejected = list(rejected) if rejected is not None else []
        self.k = k
        self._sorted = sorted(self.values)      # sorted once, reused everywhere
        self._summary = None
        self._fences = None

    @classmethod
    def from_csv(cls, path: Path, column: str) -> "StatisticsAnalyzer":
        data = _read_column(path, column)
        return cls(data["values"], name=column, labels=data["labels"],
                   rows_read=data["rows_read"], rejected=data["rejected"])

    def fences(self) -> dict:
        if self._fences is None:
            self._fences = _outlier_info(self._sorted, self.k)
        return self._fences

    def outlier_indices(self) -> list[int]:
        f = self.fences()
        return [i for i, v in enumerate(self.values)
                if v < f["lower_fence"] or v > f["upper_fence"]]

    def summary(self) -> dict:
        if self._summary is None:
            s = self._sorted
            n = len(s)
            q1, q2, q3 = _quartiles_sorted(s)
            avg = sum(s) / n
            var = sum((v - avg) ** 2 for v in s) / (n - 1) if n >= 2 else None
            self._summary = {
                "count": n, "mean": avg, "median": q2, "mode": mode(self.values),
                "variance": var, "std_dev": math.sqrt(var) if var is not None else None,
                "min": s[0], "max": s[-1], "q1": q1, "q3": q3, "iqr": q3 - q1,
            }
        return self._summary

    def without_outliers(self) -> "StatisticsAnalyzer":
        drop = set(self.outlier_indices())
        keep = [i for i in range(len(self.values)) if i not in drop]
        return StatisticsAnalyzer([self.values[i] for i in keep], name=self.name,
                                  labels=[self.labels[i] for i in keep],
                                  rows_read=self.rows_read, rejected=self.rejected, k=self.k)

    def report(self) -> str:
        clean = self.without_outliers()
        a, b = self.summary(), clean.summary()

        def num(x):
            return "n/a" if x is None else f"{x:,.2f}"

        def row(label, x, y):
            return f"{label:<20}{x:<24}{y}"

        def modes(m):
            return "none" if not m else ", ".join(f"{v:,.2f}" for v in m)

        lines = [f"=== Statistics report: {self.name} ===",
                 f"Rows read         : {self.rows_read}",
                 f"Valid values      : {len(self.values)}",
                 f"Rejected values   : {len(self.rejected)}"]
        for n, raw, reason in self.rejected:
            lines.append(f"  row {n}: '{raw}' ({reason})")
        lines += ["", row("", "all data", "outliers removed"),
                  row("count", a["count"], b["count"]),
                  row("mean", num(a["mean"]), num(b["mean"])),
                  row("median", num(a["median"]), num(b["median"])),
                  row("mode", modes(a["mode"]), modes(b["mode"])),
                  row("variance (sample)", num(a["variance"]), num(b["variance"])),
                  row("std dev (sample)", num(a["std_dev"]), num(b["std_dev"])),
                  row("min / max", f"{num(a['min'])} / {num(a['max'])}",
                      f"{num(b['min'])} / {num(b['max'])}"),
                  row("Q1 / Q3", f"{num(a['q1'])} / {num(a['q3'])}",
                      f"{num(b['q1'])} / {num(b['q3'])}"),
                  row("IQR", num(a["iqr"]), num(b["iqr"])), ""]
        f = self.fences()
        lines.append(f"Outlier fences      : lower {f['lower_fence']:,.2f}, upper {f['upper_fence']:,.2f}")
        found = [f"{self.values[i]:,.2f} ({self.labels[i][0]}, {self.labels[i][1]})"
                 for i in self.outlier_indices()]
        lines.append("Outliers            : " + (", ".join(found) if found else "none"))
        return "\n".join(lines)


# ------------------------------------------------------------- numpy check (task 6)
def compare_with_numpy(analyzer: StatisticsAnalyzer) -> dict[str, bool]:
    """Import numpy HERE, recompute each statistic and return {name: matches}.

    Uses ddof=1 (sample) and method="linear" for quartiles, matching the toolkit.
    Tolerance-based comparison, never ==.
    """
    import numpy as np

    arr = np.array(analyzer.values, dtype=float)
    s = analyzer.summary()
    q1, q3 = np.percentile(arr, [25, 75], method="linear")

    def close(a, b):
        return bool(np.isclose(a, b, rtol=1e-9, atol=1e-9))

    return {
        "mean": close(s["mean"], np.mean(arr)),
        "median": close(s["median"], np.median(arr)),
        "variance": close(s["variance"], np.var(arr, ddof=1)),
        "std_dev": close(s["std_dev"], np.std(arr, ddof=1)),
        "quartiles": close(s["q1"], q1) and close(s["q3"], q3),
    }


def _numpy_block(analyzer: StatisticsAnalyzer) -> tuple[list[str], bool]:
    try:
        results = compare_with_numpy(analyzer)
    except (ImportError, OSError) as e:
        return [f"NumPy is not available here, verification skipped ({e})"], False

    def word(ok):
        return "MATCH" if ok else "MISMATCH"

    return ([f"mean      {word(results['mean'])}",
             f"median    {word(results['median'])}",
             f"variance  {word(results['variance'])}  (ddof=1)",
             f"std dev   {word(results['std_dev'])}",
             f"Q1 / Q3   {word(results['quartiles'])}  (method=linear)"], True)


def _test_block() -> list[str]:
    """Run the unit tests in a separate process and return the summary lines."""
    start = time.time()
    try:
        proc = subprocess.run([sys.executable, "-m", "unittest", "test_stats_toolkit"],
                              cwd=BASE_DIR, capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.SubprocessError) as e:
        return [f"Tests could not be run here ({e})"]
    lines = [ln for ln in proc.stderr.strip().splitlines()
             if ln.startswith("Ran ") or ln.startswith(("OK", "FAILED"))]
    return lines or [f"Tests could not be run: {proc.stderr.strip()[:200]}"]


RISK_NOTES = (
    "Notes for the risk team: Four days fall outside the fences, and three of them belong to "
    "agent AG-104 in Nakuru: two extreme highs (96,500 on 2026-09-20 and 88,400 on 2026-10-06) and one "
    "extreme low (1,200 on 2026-10-02), while the fourth, 29,060.99 for AG-101 on 2026-10-07, is only just "
    "above the upper fence. "
    "The very high days could point to fraud, a large one-off customer deposit or an extra digit typed by "
    "mistake, and the very low day could point to a float shortage, an app outage or a missing digit. "
    "AG-104's records should be reviewed first, while the AG-101 day needs only a quick check."
)


if __name__ == "__main__":
    analyzer = StatisticsAnalyzer.from_csv(DATA_PATH, COLUMN)
    numpy_lines, numpy_ok = _numpy_block(analyzer)
    parts = [analyzer.report(), "", "=== NumPy verification ===", *numpy_lines,
             "", "=== Unit tests ===", *_test_block(), "", RISK_NOTES]
    text = "\n".join(parts)
    print(text)
    if numpy_ok:
        REPORT_PATH.parent.mkdir(exist_ok=True)
        REPORT_PATH.write_text(text + "\n", encoding="utf-8")
        print(f"\nReport saved to {REPORT_PATH}")
    else:
        print("\nReport NOT saved: NumPy verification is part of the report, so the existing "
              "output/stats_report.txt was left untouched.")
