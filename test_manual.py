import warnings
warnings.simplefilter("always")

from outlier_aware_series import OutlierAwareSeries


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ----------------------------------------------------------------------
section("1. The original failing case — does modified_zscore fix it?")
data = [10, 12, 11, 13, 12, 5000]

s_zscore = OutlierAwareSeries(data, method="zscore", threshold=3)
print("zscore outliers:        ", s_zscore.outliers())

s_mad = OutlierAwareSeries(data, method="modified_zscore", threshold=3.5)
print("modified_zscore outliers:", s_mad.outliers())


# ----------------------------------------------------------------------
section("2. Same masking test, but with TWO outliers")
data2 = [10, 12, 11, 13, 12, 500, 480]

print("zscore:         ", OutlierAwareSeries(data2, method="zscore", threshold=3).outliers())
print("iqr:            ", OutlierAwareSeries(data2, method="iqr", threshold=1.5).outliers())
print("modified_zscore:", OutlierAwareSeries(data2, method="modified_zscore", threshold=3.5).outliers())


# ----------------------------------------------------------------------
section("3. All three methods side-by-side on the same data")
data3 = [88, 92, 85, 90, 91, 89, 15]

for method, threshold in [("zscore", 3), ("iqr", 1.5), ("modified_zscore", 3.5)]:
    s = OutlierAwareSeries(data3, method=method, threshold=threshold)
    print(f"{method:16s} (threshold={threshold}): {s.outliers()}")


# ----------------------------------------------------------------------
section("4. Negative number outliers")
data4 = [50, 52, 51, 53, 52, -400]

for method, threshold in [("zscore", 3), ("iqr", 1.5), ("modified_zscore", 3.5)]:
    s = OutlierAwareSeries(data4, method=method, threshold=threshold)
    print(f"{method:16s}: {s.outliers()}")


# ----------------------------------------------------------------------
section("5. All identical values (should never crash, zero outliers)")
data5 = [5, 5, 5, 5, 5]

for method in ["zscore", "iqr", "modified_zscore"]:
    s = OutlierAwareSeries(data5, method=method)
    print(f"{method:16s}: flags={s.flags}")


# ----------------------------------------------------------------------
section("6. Single value (should never crash)")
data6 = [42]

for method in ["zscore", "iqr", "modified_zscore"]:
    s = OutlierAwareSeries(data6, method=method)
    print(f"{method:16s}: flags={s.flags}")


# ----------------------------------------------------------------------
section("7. Does the warning fire correctly at the exact ceiling?")
# For n=6, ceiling is (6-1)/sqrt(6) = 2.041...
data7 = [1, 2, 3, 4, 5, 6]

print("threshold=2.0 (below ceiling, should NOT warn):")
OutlierAwareSeries(data7, method="zscore", threshold=2.0)

print("\nthreshold=2.05 (above ceiling, SHOULD warn):")
OutlierAwareSeries(data7, method="zscore", threshold=2.05)


# ----------------------------------------------------------------------
section("8. Large dataset performance + correctness check")
import random
import time

random.seed(42)
big_data = [random.gauss(50, 5) for _ in range(10000)]
big_data.append(1000)  # inject one obvious outlier

start = time.time()
s = OutlierAwareSeries(big_data, method="modified_zscore", threshold=3.5)
elapsed = time.time() - start

print(f"n={len(big_data)}, took {elapsed:.4f}s")
print("outliers found:", s.outliers())
print("summary:", s.summary())


# ----------------------------------------------------------------------
section("9. flags_with() comparison without mutating state")
s = OutlierAwareSeries([1, 2, 2, 3, 2, 100], method="zscore", threshold=3)
print("original method/threshold:", s.method, s.threshold)
print("zscore(2) via flags_with:        ", s.flags_with("zscore", 2))
print("iqr(1.5) via flags_with:         ", s.flags_with("iqr", 1.5))
print("modified_zscore(3.5) flags_with: ", s.flags_with("modified_zscore", 3.5))
print("original still unchanged:", s.method, s.threshold, s.outliers())


# ----------------------------------------------------------------------
section("10. recompute() switching methods on the same object")
s = OutlierAwareSeries([10, 12, 11, 13, 12, 5000], method="zscore", threshold=3)
print("as zscore:        ", s.outliers())
s.recompute(method="modified_zscore", threshold=3.5)
print("after recompute to modified_zscore:", s.outliers())


print("\n\nDONE — review the output above.")