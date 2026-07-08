# OutlierAwareSeries

A small numeric data type for Python that **flags** statistical outliers
instead of silently deleting them.

Most cleaning pipelines do this:

```python
df = df[z_score < 3]   # outliers are now gone forever
```

`OutlierAwareSeries` keeps every value, but tags which ones are outliers,
so you can:

- audit *what* got flagged and *why* (method + threshold)
- compare "with outliers" vs "without outliers" stats side by side
- hand the data to a teammate without silently deciding for them what
  counts as noise

## Install

```bash
pip install outlier-aware-series          # once published to PyPI
# or, from source:
pip install -e .
```

Optional extras:

```bash
pip install outlier-aware-series[pandas]  # for .to_pandas()
pip install outlier-aware-series[plot]    # for .plot()
```

## Usage

```python
from outlier_aware_series import OutlierAwareSeries

s = OutlierAwareSeries([1, 2, 2, 3, 2, 100], method="zscore", threshold=2)

s.outliers()          # [100]
s.clean()             # [1, 2, 2, 3, 2]  (view only -- nothing deleted)
s.data                # [1, 2, 2, 3, 2, 100]  (original, untouched)
s.flags                # [False, False, False, False, False, True]
s.summary()
# {'n': 6, 'n_outliers': 1, 'pct_outliers': 16.67, 'method': 'zscore',
#  'threshold': 2, 'mean_all': 18.33, 'mean_clean': 2.0, ...}

for value, is_outlier in s:
    print(value, "outlier" if is_outlier else "normal")

# Compare detection methods without mutating the series
s.flags_with("iqr", threshold=1.5)

# Switch method/threshold in place
s.recompute(method="iqr", threshold=1.5)

# pandas / matplotlib interop (optional)
s.to_pandas()   # DataFrame with 'value' and 'is_outlier' columns
s.plot()        # scatter plot, outliers highlighted in red
```

### A warning about small datasets and z-score

Plain z-score has a hard mathematical ceiling: for `n` data points, no
single value can ever produce a z-score higher than `(n-1)/sqrt(n)`. On
small datasets, a single extreme value can inflate the mean and standard
deviation so much that it hides from its own detection.

```python
OutlierAwareSeries([10, 12, 11, 13, 12, 5000], method="zscore", threshold=3)
# UserWarning: With only 6 data points, a z-score can never exceed 2.04.
# Your threshold of 3 can NEVER flag any outlier, regardless of how
# extreme it is. Lower the threshold, or use method='modified_zscore'
# or method='iqr' instead.
```

If you see this warning, switch to `modified_zscore` or `iqr`:

```python
OutlierAwareSeries([10, 12, 11, 13, 12, 5000], method="modified_zscore", threshold=3.5).outliers()
# [5000]
```

## Detection methods

| Method            | How it works                                                          |
|-------------------|------------------------------------------------------------------------|
| `zscore`          | Flags values more than `threshold` standard deviations from the mean. Has a known ceiling on small datasets — see above. |
| `iqr`             | Flags values outside `Q1 - threshold*IQR` .. `Q3 + threshold*IQR`     |
| `modified_zscore` | Robust alternative using median and MAD (Median Absolute Deviation) instead of mean/stdev. Not affected by the small-sample ceiling that affects plain `zscore`. Common threshold: `3.5`. |

## Why not just use pandas + a boolean mask?

You can! `OutlierAwareSeries` is just a small, self-contained object that
makes the "keep flags glued to the data" pattern the default instead of
something you have to remember to do by hand every time.

## Running tests

```bash
pip install -e ".[dev]"
pytest
```

## License

MIT
