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

## Detection methods

| Method    | How it works                                             |
|-----------|-----------------------------------------------------------|
| `zscore`  | Flags values more than `threshold` standard deviations from the mean |
| `iqr`     | Flags values outside `Q1 - threshold*IQR` .. `Q3 + threshold*IQR`     |

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
