"""
One-time conversion utility: forecasts.csv / svix.csv -> Parquet, with the
`timestamp` column properly typed as Datetime (not left as a string).

WHY THIS MATTERS: generate_pnl.py needs to read only a narrow date window out
of a multi-year forecasts.csv / svix.csv. CSV is a row-oriented text format --
even with a lazy, filtered scan, every byte of the file still has to be read
and tokenized before a filter can decide whether to keep or discard each row,
so peak memory during that scan scales with the FULL FILE SIZE regardless of
how selective your filter is. Parquet is columnar and stores per-row-group
min/max statistics; when `timestamp` is natively typed as Datetime, a
`pl.scan_parquet(...).filter(pl.col('timestamp') >= ...)` can skip entire
row-groups without ever reading their bytes. Measured on a 684 MB synthetic
CSV filtered down to a 6-month window out of 2.5 years of history:

    CSV  scan+filter peak RSS:  ~2045 MB
    Parquet (typed) peak RSS:    ~185 MB      (~11x less)

Run this ONCE after (re)generating forecasts.csv / svix.csv:

    python csv_to_parquet.py --target_path ./data_centre

Then generate_pnl.py will automatically prefer the .parquet versions if
present (falling back to the slower CSV path with a printed note if not).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl


def convert_forecasts(csv_path: Path, parquet_path: Path) -> None:
    print(f"Converting {csv_path} -> {parquet_path} ...")
    (
        pl.scan_csv(str(csv_path), try_parse_dates=False)
        .with_columns(pl.col("timestamp").str.to_datetime(time_zone="UTC", strict=False))
        .sink_parquet(str(parquet_path))
    )
    print(f"  done -- {parquet_path.stat().st_size / 1024**2:.1f} MB "
          f"(was {csv_path.stat().st_size / 1024**2:.1f} MB as CSV)")


def convert_svix(csv_path: Path, parquet_path: Path) -> None:
    print(f"Converting {csv_path} -> {parquet_path} ...")
    (
        pl.scan_csv(str(csv_path), try_parse_dates=False)
        .with_columns(pl.col("timestamp").str.to_datetime(time_zone="UTC", strict=False))
        .sink_parquet(str(parquet_path))
    )
    print(f"  done -- {parquet_path.stat().st_size / 1024**2:.1f} MB "
          f"(was {csv_path.stat().st_size / 1024**2:.1f} MB as CSV)")


def convert_spot_features(csv_path: Path, parquet_path: Path) -> None:
    print(f"Converting {csv_path} -> {parquet_path} ...")
    (
        pl.scan_csv(str(csv_path), try_parse_dates=False)
        .with_columns(pl.col("timestamp").str.to_datetime(time_zone="UTC", strict=False))
        .sink_parquet(str(parquet_path))
    )
    print(f"  done -- {parquet_path.stat().st_size / 1024**2:.1f} MB "
          f"(was {csv_path.stat().st_size / 1024**2:.1f} MB as CSV)")



def convert_option_features(csv_path: Path, parquet_path: Path) -> None:
    print(f"Converting {csv_path} -> {parquet_path} ...")
    (
        pl.scan_csv(str(csv_path), try_parse_dates=False)
        .with_columns(pl.col("timestamp").str.to_datetime(time_zone="UTC", strict=False))
        .sink_parquet(str(parquet_path))
    )
    print(f"  done -- {parquet_path.stat().st_size / 1024**2:.1f} MB "
          f"(was {csv_path.stat().st_size / 1024**2:.1f} MB as CSV)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target_path", type=str, default="./data_centre")
    args = parser.parse_args()
    base = Path(args.target_path) / "crypto_option_spot"
    forecasts_csv = base / "forecasts.csv"
    if forecasts_csv.exists():
        convert_forecasts(forecasts_csv, base / "forecasts.parquet")
    else:
        print(f"NOTE: {forecasts_csv} not found, skipping.")
    svix_csv = base / "svix.csv"
    if svix_csv.exists():
        convert_svix(svix_csv, base / "svix.parquet")
    else:
        print(f"NOTE: {svix_csv} not found, skipping.")
    base = Path(args.target_path)
    spot_features_csv = base / "spot_features.csv"
    option_features_csv = base / "option_features.csv"
    if spot_features_csv.exists():
        convert_forecasts(spot_features_csv, base / "spot_features.parquet")
    else:
        print(f"NOTE: {spot_features_csv} not found, skipping.")
    if option_features_csv.exists():
        convert_svix(option_features_csv, base / "option_features.parquet")
    else:
        print(f"NOTE: {option_features_csv} not found, skipping.")
    print("\nDone. Re-run generate_pnl.py -- it will automatically prefer the .parquet "
          "files if present.")