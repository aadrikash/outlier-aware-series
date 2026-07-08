"""
OutlierAwareSeries
===================

A numeric collection that flags statistical outliers as metadata instead
of silently dropping them, so you never lose visibility into your raw data.

Example
-------
>>> from outlier_aware_series import OutlierAwareSeries
>>> s = OutlierAwareSeries([1, 2, 2, 3, 2, 100], method="zscore", threshold=2)
>>> s.outliers()
[100]
>>> s.clean()
[1, 2, 2, 3, 2]
>>> s.summary()["n_outliers"]
1
"""

from __future__ import annotations

import statistics
from typing import Iterable, Iterator, List, Optional, Sequence, Tuple, Union

Number = Union[int, float]


class OutlierAwareSeries:
    """A sequence of numbers that tracks which values are statistical
    outliers without removing them.

    Parameters
    ----------
    data : Iterable[float]
        The numeric values to wrap.
    method : str, default "zscore"
        Outlier-detection method. One of ``"zscore"`` or ``"iqr"``.
    threshold : float, default 3.0
        For ``"zscore"``: number of standard deviations from the mean.
        For ``"iqr"``: multiplier applied to the interquartile range.

    Notes
    -----
    Outlier flags are computed once at construction time (or whenever
    :meth:`recompute` is called) and stored alongside the data, so the
    values themselves are never discarded.
    """

    VALID_METHODS = ("zscore", "iqr")

    def __init__(
        self,
        data: Iterable[Number],
        method: str = "zscore",
        threshold: float = 3.0,
    ) -> None:
        self.data: List[Number] = list(data)
        if len(self.data) == 0:
            raise ValueError("OutlierAwareSeries cannot be empty")
        if not all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in self.data):
            raise TypeError("OutlierAwareSeries only supports numeric values")

        self.method = method
        self.threshold = threshold
        self._flags: List[bool] = []
        self.recompute(method=method, threshold=threshold)

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------
    def recompute(self, method: Optional[str] = None, threshold: Optional[float] = None) -> "OutlierAwareSeries":
        """Recompute outlier flags, optionally with a new method/threshold.

        Returns ``self`` so calls can be chained.
        """
        if method is not None:
            self.method = method
        if threshold is not None:
            self.threshold = threshold

        if self.method not in self.VALID_METHODS:
            raise ValueError(f"Unknown method '{self.method}'. Choose from {self.VALID_METHODS}")

        if self.method == "zscore":
            self._flags = self._flags_zscore(self.data, self.threshold)
        elif self.method == "iqr":
            self._flags = self._flags_iqr(self.data, self.threshold)

        return self

    @staticmethod
    def _flags_zscore(data: Sequence[Number], threshold: float) -> List[bool]:
        if len(data) < 2:
            return [False] * len(data)
        mean = statistics.mean(data)
        std = statistics.pstdev(data)
        if std == 0:
            return [False] * len(data)
        return [abs((x - mean) / std) > threshold for x in data]

    @staticmethod
    def _flags_iqr(data: Sequence[Number], threshold: float) -> List[bool]:
        if len(data) < 4:
            return [False] * len(data)
        q1, _, q3 = statistics.quantiles(data, n=4)
        iqr = q3 - q1
        if iqr == 0:
            return [False] * len(data)
        lower = q1 - threshold * iqr
        upper = q3 + threshold * iqr
        return [x < lower or x > upper for x in data]

    def flags_with(self, method: str, threshold: float) -> List[bool]:
        """Compute flags with an alternate method/threshold *without*
        mutating this series. Useful for comparing methods side by side.
        """
        if method == "zscore":
            return self._flags_zscore(self.data, threshold)
        elif method == "iqr":
            return self._flags_iqr(self.data, threshold)
        raise ValueError(f"Unknown method '{method}'. Choose from {self.VALID_METHODS}")

    # ------------------------------------------------------------------
    # Access
    # ------------------------------------------------------------------
    @property
    def flags(self) -> List[bool]:
        """Boolean list, True where the corresponding value is an outlier."""
        return list(self._flags)

    def clean(self) -> List[Number]:
        """Return values with outliers excluded (nothing is deleted from
        the underlying series)."""
        return [x for x, f in zip(self.data, self._flags) if not f]

    def outliers(self) -> List[Number]:
        """Return just the values flagged as outliers."""
        return [x for x, f in zip(self.data, self._flags) if f]

    def indices_of_outliers(self) -> List[int]:
        """Positions (into the original data) of the flagged values."""
        return [i for i, f in enumerate(self._flags) if f]

    def summary(self) -> dict:
        """Quick stats comparing the full data to the outlier-free subset."""
        clean = self.clean()
        return {
            "n": len(self.data),
            "n_outliers": sum(self._flags),
            "pct_outliers": round(100 * sum(self._flags) / len(self.data), 2),
            "method": self.method,
            "threshold": self.threshold,
            "mean_all": statistics.mean(self.data),
            "mean_clean": statistics.mean(clean) if clean else None,
            "stdev_all": statistics.pstdev(self.data) if len(self.data) > 1 else 0.0,
            "stdev_clean": statistics.pstdev(clean) if len(clean) > 1 else 0.0,
        }

    # ------------------------------------------------------------------
    # Interop
    # ------------------------------------------------------------------
    def to_pandas(self):
        """Return a pandas DataFrame with 'value' and 'is_outlier' columns.

        Requires pandas to be installed.
        """
        try:
            import pandas as pd
        except ImportError as e:
            raise ImportError(
                "to_pandas() requires pandas. Install it with `pip install pandas`."
            ) from e
        return pd.DataFrame({"value": self.data, "is_outlier": self._flags})

    def plot(self, ax=None):
        """Scatter plot of the series with outliers highlighted in red.

        Requires matplotlib to be installed. Returns the matplotlib Axes.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError as e:
            raise ImportError(
                "plot() requires matplotlib. Install it with `pip install matplotlib`."
            ) from e

        if ax is None:
            _, ax = plt.subplots()

        xs = list(range(len(self.data)))
        colors = ["red" if f else "steelblue" for f in self._flags]
        ax.scatter(xs, self.data, c=colors)
        ax.set_xlabel("index")
        ax.set_ylabel("value")
        ax.set_title(f"OutlierAwareSeries ({self.method}, threshold={self.threshold})")
        return ax

    # ------------------------------------------------------------------
    # Dunder methods — makes it feel like a native Python collection
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.data)

    def __iter__(self) -> Iterator[Tuple[Number, bool]]:
        return iter(zip(self.data, self._flags))

    def __getitem__(self, index: int) -> Tuple[Number, bool]:
        return self.data[index], self._flags[index]

    def __eq__(self, other) -> bool:
        if not isinstance(other, OutlierAwareSeries):
            return NotImplemented
        return self.data == other.data and self.method == other.method and self.threshold == other.threshold

    def __repr__(self) -> str:
        return (
            f"OutlierAwareSeries(n={len(self.data)}, "
            f"n_outliers={sum(self._flags)}, method={self.method!r}, threshold={self.threshold})"
        )
