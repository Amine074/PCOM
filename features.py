"""Shared code for train.py and predict.py:
read the CSV, pick the data of ONE package, and build the list of features."""
import csv
import io
import math
import re
from datetime import datetime, timezone

SEVERITIES = ("low", "moderate", "high", "critical", "unknown")
KNOWN = SEVERITIES[:4]
TARGET_COLUMNS = ["target_ghsa_%s_count_90d" % s for s in SEVERITIES]
MISSING = {"", "null", "none", "nan", "na", "n/a"}
NAN = float("nan")
PREDICTION_DATE = datetime(2026, 7, 1, tzinfo=timezone.utc)  # M0 = June 2026 in the record schema

# Used by the simple fallback rule, and if model/meta.json is missing.
DEFAULT_META = {
    "weight_recent": 0.6,
    "global_mean": {"low": 0.0714, "moderate": 0.5, "high": 0.3571,
                    "critical": 0.0, "unknown": 0.0},
}


# ---------------------------------------------------------------- reading the CSV
def is_missing(value):
    return value is None or str(value).strip().lower() in MISSING


def to_number(value):
    """Text -> number, or None if the value is missing or not a valid number >= 0."""
    if is_missing(value):
        return None
    try:
        x = float(str(value).strip())
    except ValueError:
        return None
    return x if 0 <= x < float("inf") else None


def read_rows(path):
    """Read the CSV into a list of dicts. Accepts , or ; and a BOM."""
    with open(path, newline="", encoding="utf-8-sig") as f:
        text = f.read()
    first_line = text.split("\n", 1)[0]
    delimiter = ";" if first_line.count(";") > first_line.count(",") else ","
    rows = []
    for row in csv.DictReader(io.StringIO(text), delimiter=delimiter):
        rows.append({(k or "").strip(): v for k, v in row.items()})
    return rows


def package_name(purl):
    return (purl or "").rsplit("@", 1)[0]


def version_key(purl):
    """Semantic version from the PURL as numbers: pkg:npm/a@1.10.2 -> (1, 10, 2)."""
    m = re.search(r"@(\d+)\.(\d+)\.(\d+)", purl or "")
    return tuple(int(x) for x in m.groups()) if m else (-1, -1, -1)


def merge_rows(rows):
    """The data of one package as ONE dict: the newest version (highest semver) wins;
    a missing value is taken from the next older version."""
    rows = sorted(rows, key=lambda r: version_key(r.get("purl")), reverse=True)
    merged = {}
    for row in rows:
        for column, value in row.items():
            if column not in merged or is_missing(merged[column]):
                merged[column] = value
    return merged


# ---------------------------------------------------------------- features
def num(rec, column):
    x = to_number(rec.get(column))
    return NAN if x is None else x


def month_sum(rec, name, months, zero_if_missing=False):
    """Sum of <name>_M<i> over the given months. NaN (or 0) if every value is missing."""
    values = [to_number(rec.get("%s_M%d" % (name, m))) for m in months]
    values = [v for v in values if v is not None]
    if not values:
        return 0.0 if zero_if_missing else NAN
    return float(sum(values))


def age_in_days(text):
    """Days between the package creation and the prediction date (NaN if unknown)."""
    try:
        t = datetime.fromisoformat(str(text).strip().replace("Z", "+00:00"))
        if t.tzinfo is None:
            t = t.replace(tzinfo=timezone.utc)
        return (PREDICTION_DATE - t).total_seconds() / 86400.0
    except Exception:
        return NAN


def recent_rate(rec, severity):
    """Advisories per 90 days, from the last 12 months (used by the simple rule)."""
    return month_sum(rec, "ghsa_%s_count" % severity, range(12), True) * 3.0 / 12.0


def feature_dict(rec):
    """All features of one package. Unknown values are NaN (XGBoost can handle NaN)."""
    f = {}
    windows = {"3m": range(0, 3), "6m": range(0, 6), "12m": range(0, 12)}
    for sev in KNOWN + ("all",):
        sevs = SEVERITIES if sev == "all" else (sev,)
        for label, months in windows.items():
            f["ghsa_%s_%s" % (sev, label)] = sum(
                month_sum(rec, "ghsa_%s_count" % s, months, True) for s in sevs)
        f["ghsa_%s_lifetime" % sev] = sum(to_number(rec.get("ghsa_%s_count" % s)) or 0.0
                                          for s in sevs)
    f["downloads_log_m0"] = math.log1p(num(rec, "package_downloads_M0"))
    f["downloads_log_12m"] = math.log1p(month_sum(rec, "package_downloads", range(12)))
    f["downloads_trend"] = (math.log1p(month_sum(rec, "package_downloads", range(0, 3)))
                            - math.log1p(month_sum(rec, "package_downloads", range(9, 12))))
    f["releases_3m"] = month_sum(rec, "package_stable_release_count", range(0, 3))
    f["releases_12m"] = month_sum(rec, "package_stable_release_count", range(12))
    for name in ("issues_opened", "issues_closed", "pull_requests_opened",
                 "pull_requests_merged", "pull_requests_closed_unmerged"):
        f["%s_6m" % name] = month_sum(rec, name, range(0, 6))
    f["issues_open_now"] = num(rec, "issues_open_at_month_end_M0")
    f["pull_requests_open_now"] = num(rec, "pull_requests_open_at_month_end_M0")
    for column in ("dependency_count", "optional_dependency_count", "peer_dependency_count",
                   "dev_dependency_count", "bundled_dependency_count", "file_count"):
        f[column] = num(rec, column)
    f["unpacked_size_log"] = math.log1p(num(rec, "unpacked_size"))
    f["package_age_days"] = age_in_days(rec.get("package_created_at") or rec.get("created_at"))
    return f


FEATURE_NAMES = list(feature_dict({}).keys())


def feature_vector(rec):
    f = feature_dict(rec)
    return [f[name] for name in FEATURE_NAMES]


# ---------------------------------------------------------------- simple fallback rule
def rule_prediction(rec, meta):
    """Expected new advisories per severity: mix of the package's own rate in the last
    12 months and the average of the training packages."""
    w = meta["weight_recent"]
    return {sev: w * recent_rate(rec, sev) + (1.0 - w) * meta["global_mean"][sev]
            for sev in SEVERITIES}
