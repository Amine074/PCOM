"""Create one machine-learning row per npm package from version-level records."""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Tuple

import pandas as pd
from packaging.version import Version, InvalidVersion

TARGETS = [
    "target_ghsa_low_count_90d",
    "target_ghsa_moderate_count_90d",
    "target_ghsa_high_count_90d",
    "target_ghsa_critical_count_90d",
    "target_ghsa_unknown_count_90d",
]
AS_OF_DATE = pd.Timestamp("2026-07-01", tz="UTC")

# Other names that could mean "purl" in a CSV file.
PURL_ALIASES = ["package_url", "package_purl", "pkg_purl", "purl_id", "package", "id"]


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Clean column names so small naming differences do not break the code."""
    df = df.copy()
    # remove BOM / spaces and use lower case
    df.columns = [str(c).replace("﻿", "").strip().lower() for c in df.columns]

    renames = {}
    for c in df.columns:
        new = c
        # old names -> names of the record schema
        new = re.sub(r"^(issues_(?:opened|closed|reopened))_count_m(\d+)$", r"\1_m\2", new)
        new = re.sub(r"^package_downloads_days_observed_", "package_download_days_observed_", new)
        if new == "created_at":
            new = "package_created_at"
        # the schema writes the month as upper case M0 ... M11
        new = re.sub(r"_m(\d+)$", lambda m: "_M" + m.group(1), new)
        if new != c:
            renames[c] = new
    df = df.rename(columns=renames)

    # purl under another name
    if "purl" not in df.columns:
        for alt in PURL_ALIASES:
            if alt in df.columns and df[alt].astype(str).str.startswith("pkg:").any():
                df = df.rename(columns={alt: "purl"})
                break
    # the schema names its target columns in lower case already; make sure of it
    return df


def read_csv_flexible(path: str | Path) -> pd.DataFrame:
    """Read a CSV with comma or semicolon, with or without BOM."""
    df = pd.read_csv(path, encoding="utf-8-sig")
    if df.shape[1] == 1:  # probably another separator
        df = pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig")
    return normalize_columns(df)


def package_name_from_purl(purl: str) -> str:
    """Return npm package name from pkg:npm/name@version PURL."""
    if not isinstance(purl, str):
        raise ValueError(f"Invalid npm PURL: {purl!r}")
    purl = purl.strip().replace("%40", "@")  # scoped packages: pkg:npm/%40scope/name@1.0.0
    if not purl.startswith("pkg:npm/"):
        raise ValueError(f"Invalid npm PURL: {purl!r}")
    value = purl[len("pkg:npm/"):]
    if "@" not in value.lstrip("@"):
        raise ValueError(f"PURL has no version: {purl!r}")
    return value.rsplit("@", 1)[0]


def stable_version_from_purl(purl: str):
    """Parse stable semver; return None for prereleases or invalid versions."""
    try:
        name = package_name_from_purl(purl)
        raw_version = purl.strip().rsplit("@", 1)[1]
        parsed = Version(raw_version)
        if parsed.is_prerelease or parsed.is_devrelease:
            return name, None
        return name, parsed
    except (ValueError, InvalidVersion):
        return None, None


def newest_row_without_purl(df: pd.DataFrame) -> pd.DataFrame:
    """Input without purl: all rows are one package. Without a version number we take the
    row with the newest published_at (the last row if there is no date)."""
    if df.empty:
        raise ValueError("Input CSV has no records.")
    idx = df.index[-1]
    if "published_at" in df.columns:
        dates = pd.to_datetime(df["published_at"], errors="coerce", utc=True)
        if dates.notna().any():
            idx = dates.idxmax()
    row = df.loc[idx].to_dict()
    row["package_name"] = "unknown"
    return add_date_features(pd.DataFrame([row]))


def make_package_dataset(input_csv: str | Path,
                         require_targets: bool = True) -> Tuple[pd.DataFrame, list[str]]:
    """One row per package. For training, require_targets=True.
    For predict.py use require_targets=False (the input needs no target columns)."""
    df = read_csv_flexible(input_csv)
    # The files of the prediction rounds have no purl column. They contain the versions of
    # exactly one package, so all rows belong together.
    if "purl" not in df.columns and not require_targets:
        return newest_row_without_purl(df), TARGETS
    required = {"purl", *(TARGETS if require_targets else [])}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(
            f"Input CSV is missing required columns: {missing}. "
            f"First columns found in the file: {list(df.columns[:8])}"
        )

    parsed = df["purl"].apply(stable_version_from_purl)
    df["_package"] = parsed.apply(lambda x: x[0])
    df["_version"] = parsed.apply(lambda x: x[1])
    df = df[df["_version"].notna()].copy()
    if df.empty:
        raise ValueError("No stable npm package versions found in input CSV.")

    rows = []
    inconsistent = []
    for package, group in df.groupby("_package", sort=True):
        # The package-level target should be identical across all versions.
        if require_targets:
            target_values = group[TARGETS].apply(pd.to_numeric, errors="coerce")
            if (target_values.nunique(dropna=False) > 1).any():
                inconsistent.append(package)
                continue

        latest_idx = max(group.index, key=lambda idx: group.loc[idx, "_version"])
        row = group.loc[latest_idx].drop(labels=["_package", "_version"]).to_dict()
        row["package_name"] = package
        rows.append(row)

    if inconsistent:
        preview = ", ".join(inconsistent[:10])
        raise ValueError(
            "Targets are not identical across versions for these packages: " + preview
            + (" ..." if len(inconsistent) > 10 else "")
            + ". Check the builder's target calculation before training."
        )

    result = pd.DataFrame(rows).reset_index(drop=True)
    result = add_date_features(result)
    return result, TARGETS


def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for source, output in [
        ("package_created_at", "package_age_days_at_cutoff"),
        ("published_at", "version_age_days_at_cutoff"),
    ]:
        if source in df.columns:
            dates = pd.to_datetime(df[source], errors="coerce", utc=True)
            df[output] = (AS_OF_DATE - dates).dt.total_seconds() / 86400.0
    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    excluded = set(TARGETS) | {"purl", "package_name", "package_created_at", "published_at"}
    candidates = [c for c in df.columns if c not in excluded]
    numeric = df[candidates].apply(pd.to_numeric, errors="coerce")
    # Remove columns that contain no usable values or no variation.
    return [c for c in numeric.columns if numeric[c].notna().any() and numeric[c].nunique(dropna=True) > 1]


def main():
    parser = argparse.ArgumentParser(description="Create one feature/target row per npm package.")
    parser.add_argument("input", nargs="?", default="training_data.csv")
    parser.add_argument("--output", default="package_training_data.csv")
    args = parser.parse_args()

    packages, _ = make_package_dataset(args.input)
    packages.to_csv(args.output, index=False)
    print(f"Saved {len(packages)} package rows to {args.output}")
    print(f"Selected latest stable version for {packages['package_name'].nunique()} packages.")
    print(f"Target columns: {', '.join(TARGETS)}")


if __name__ == "__main__":
    main()
