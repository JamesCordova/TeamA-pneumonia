#!/usr/bin/env python
"""
Prepare modeling-ready data from the IRAS source.

Steps:
  1. (Optional) Download raw data from PostgreSQL → data/raw/iras_data_raw.csv
  2. Aggregate cases by department × age_group × week
  3. Enforce a regular 7-day frequency, filling the 3 ISO week-53 gaps
     (2004-W53, 2009-W53, 2015-W53) via linear interpolation
  4. Save to data/processed/iras_weekly_clean.csv

Usage:
    python scripts/prepare_data.py                  # download + clean
    python scripts/prepare_data.py --no_download    # clean only (raw already exists)
    python scripts/prepare_data.py --force          # overwrite existing processed file
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from pneumonia.config import DATA_RAW_PATH, DATA_PROCESSED_PATH
from pneumonia.data.download_raw_data import download_raw_data
from pneumonia.models.utils import handle_missing_values
from pneumonia.utils import setup_logger

logger = setup_logger(__name__)

RAW_PATH       = Path(DATA_RAW_PATH) / "iras_data_raw.csv"
PROCESSED_PATH = Path(DATA_PROCESSED_PATH) / "iras_weekly_clean.csv"

AGE_GROUPS = {
    "under5": "pneumonia_under5",
    "60plus": "pneumonia_60plus",
}


def _load_raw() -> pd.DataFrame:
    if not RAW_PATH.exists():
        raise FileNotFoundError(
            f"Raw file not found: {RAW_PATH}\n"
            "Run without --no_download to fetch it from PostgreSQL."
        )
    logger.info(f"Reading raw file: {RAW_PATH}")
    df = pd.read_csv(RAW_PATH)
    df.columns = [c.strip().lower() for c in df.columns]
    df["week_start"] = pd.to_datetime(df["week_start"], errors="coerce")
    df = df.dropna(subset=["week_start"])
    df["department"] = df["department"].astype(str).str.strip().str.upper()
    return df


def _build_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate to dept × age_group × week and fill frequency gaps."""
    records = []
    departments = sorted(df["department"].unique())

    for age_group, case_col in AGE_GROUPS.items():
        if case_col not in df.columns:
            logger.warning(f"Column '{case_col}' not found — skipping {age_group}")
            continue

        df[case_col] = pd.to_numeric(df[case_col], errors="coerce").fillna(0)

        for dept in departments:
            ts = (
                df[df["department"] == dept]
                .groupby("week_start")[case_col]
                .sum()
                .sort_index()
            )

            # Enforce regular 7-day frequency (fills ISO week-53 gaps with NaN)
            ts = ts.asfreq("7D")
            if ts.isna().any():
                n = int(ts.isna().sum())
                logger.info(f"  {dept}/{age_group}: interpolating {n} gap(s)")
                ts = handle_missing_values(ts, method="interpolate")

            for date, val in ts.items():
                records.append({
                    "week_start": date,
                    "department": dept,
                    "age_group":  age_group,
                    "cases":      float(val),
                })

    clean = pd.DataFrame(records)
    clean["week_start"] = pd.to_datetime(clean["week_start"])
    clean = clean.sort_values(["department", "age_group", "week_start"]).reset_index(drop=True)
    return clean


def prepare(no_download: bool = False, force: bool = False) -> Path:
    # --- guard: skip if already done ---
    if PROCESSED_PATH.exists() and not force:
        print(
            f"Processed file already exists: {PROCESSED_PATH}\n"
            "Use --force to regenerate."
        )
        logger.info("Processed file up to date — nothing to do.")
        return PROCESSED_PATH

    # --- step 1: download (optional) ---
    if not no_download:
        print("Downloading raw data from PostgreSQL...")
        download_raw_data()
    else:
        print(f"Skipping download — using existing raw file: {RAW_PATH}")

    # --- step 2: load raw ---
    df = _load_raw()
    depts = df["department"].nunique()
    print(f"Raw file loaded: {len(df):,} rows, {depts} departments")

    # --- step 3: aggregate + clean ---
    print("Aggregating and filling gaps...")
    clean = _build_clean(df)

    gaps_filled = len(clean) - df.groupby(
        ["department", "week_start"]
    )["pneumonia_under5"].sum().reset_index().shape[0] * len(AGE_GROUPS)

    # --- step 4: save ---
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(PROCESSED_PATH, index=False)

    weeks = clean["week_start"].nunique()
    print(
        f"\nProcessed file saved: {PROCESSED_PATH}\n"
        f"  Departments : {clean['department'].nunique()}\n"
        f"  Age groups  : {clean['age_group'].nunique()}\n"
        f"  Weeks       : {weeks}\n"
        f"  Total rows  : {len(clean):,}"
    )
    logger.info(f"Saved processed data: {PROCESSED_PATH} ({len(clean)} rows)")
    return PROCESSED_PATH


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--no_download", action="store_true",
        help="Skip PostgreSQL download; process the existing raw CSV only.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Regenerate the processed file even if it already exists.",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable verbose logging.",
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger("pneumonia").setLevel(logging.DEBUG)

    prepare(no_download=args.no_download, force=args.force)


if __name__ == "__main__":
    main()
