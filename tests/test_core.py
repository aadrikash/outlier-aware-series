import pytest

from outlier_aware_series import OutlierAwareSeries


# ----------------------------------------------------------------------
# Construction
# ----------------------------------------------------------------------
def test_empty_data_raises():
    with pytest.raises(ValueError):
        OutlierAwareSeries([])


def test_non_numeric_raises():
    with pytest.raises(TypeError):
        OutlierAwareSeries([1, 2, "three"])


def test_bool_rejected_as_numeric():
    with pytest.raises(TypeError):
        OutlierAwareSeries([1, 2, True])


def test_invalid_method_raises():
    with pytest.raises(ValueError):
        OutlierAwareSeries([1, 2, 3, 4], method="bogus")


# ----------------------------------------------------------------------
# Z-score detection
# ----------------------------------------------------------------------
def test_zscore_detects_obvious_outlier():
    s = OutlierAwareSeries([1, 2, 2, 3, 2, 100], method="zscore", threshold=2)
    assert s.outliers() == [100]
    assert s.clean() == [1, 2, 2, 3, 2]


def test_zscore_no_outliers_in_uniform_data():
    with pytest.warns(UserWarning, match="can never exceed"):
        s = OutlierAwareSeries([5, 5, 5, 5], method="zscore")
    assert s.outliers() == []
    assert s.clean() == [5, 5, 5, 5]


def test_zscore_single_value_no_crash():
    s = OutlierAwareSeries([42], method="zscore")
    assert s.flags == [False]


# ----------------------------------------------------------------------
# IQR detection
# ----------------------------------------------------------------------
def test_iqr_detects_outlier():
    s = OutlierAwareSeries([1, 2, 3, 4, 5, 6, 100], method="iqr", threshold=1.5)
    assert 100 in s.outliers()


def test_iqr_short_series_no_flags():
    # Fewer than 4 points -> can't compute quartiles meaningfully
    s = OutlierAwareSeries([1, 2, 3], method="iqr")
    assert s.flags == [False, False, False]


# ----------------------------------------------------------------------
# Nothing is ever deleted
# ----------------------------------------------------------------------
def test_original_data_preserved():
    original = [1, 2, 2, 3, 2, 100]
    s = OutlierAwareSeries(original, method="zscore", threshold=2)
    assert s.data == original
    assert len(s) == len(original)


# ----------------------------------------------------------------------
# Summary stats
# ----------------------------------------------------------------------
def test_summary_reports_counts_and_means():
    s = OutlierAwareSeries([1, 2, 2, 3, 2, 100], method="zscore", threshold=2)
    summary = s.summary()
    assert summary["n"] == 6
    assert summary["n_outliers"] == 1
    assert summary["mean_clean"] < summary["mean_all"]


# ----------------------------------------------------------------------
# Comparing methods without mutating state
# ----------------------------------------------------------------------
def test_flags_with_does_not_mutate():
    with pytest.warns(UserWarning, match="can never exceed"):
        s = OutlierAwareSeries([1, 2, 2, 3, 2, 100], method="zscore", threshold=3)
    before = s.flags
    alt_flags = s.flags_with("iqr", 1.5)
    assert s.flags == before  # unchanged
    assert isinstance(alt_flags, list)

def test_recompute_changes_state_and_returns_self():
    with pytest.warns(UserWarning, match="can never exceed"):
        s = OutlierAwareSeries([1, 2, 2, 3, 2, 100], method="zscore", threshold=10)
    assert s.outliers() == []  # threshold too high to flag anything
    result = s.recompute(threshold=2)
    assert result is s
    assert s.outliers() == [100]

# ----------------------------------------------------------------------
# Dunder / collection-like behavior
# ----------------------------------------------------------------------
def test_len_and_getitem():
    with pytest.warns(UserWarning, match="can never exceed"):
        s = OutlierAwareSeries([10, 20, 30])
    assert len(s) == 3
    value, is_outlier = s[0]
    assert value == 10
    assert isinstance(is_outlier, bool)

def test_iteration_yields_value_flag_pairs():
    s = OutlierAwareSeries([1, 2, 2, 3, 2, 100], method="zscore", threshold=2)
    pairs = list(s)
    assert pairs[-1] == (100, True)
    assert pairs[0] == (1, False)


def test_equality():
    with pytest.warns(UserWarning, match="can never exceed"):
        a = OutlierAwareSeries([1, 2, 3], method="zscore", threshold=3)
    with pytest.warns(UserWarning, match="can never exceed"):
        b = OutlierAwareSeries([1, 2, 3], method="zscore", threshold=3)
    c = OutlierAwareSeries([1, 2, 3], method="iqr", threshold=3)
    assert a == b
    assert a != c

def test_repr_contains_key_info():
    s = OutlierAwareSeries([1, 2, 2, 3, 2, 100], method="zscore", threshold=2)
    r = repr(s)
    assert "n=6" in r
    assert "n_outliers=1" in r


# ----------------------------------------------------------------------
# Interop (pandas optional)
# ----------------------------------------------------------------------
def test_to_pandas_if_available():
    pd = pytest.importorskip("pandas")
    s = OutlierAwareSeries([1, 2, 2, 3, 2, 100], method="zscore", threshold=2)
    df = s.to_pandas()
    assert list(df.columns) == ["value", "is_outlier"]
    assert len(df) == 6
    assert df["is_outlier"].sum() == 1
