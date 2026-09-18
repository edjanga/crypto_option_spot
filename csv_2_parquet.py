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
    base = Path(args.target_path)
    forecasts_csv = base / "crypto_option_spot/forecasts.csv"
    if forecasts_csv.exists():
        convert_forecasts(forecasts_csv, base / "crypto_option_spot/forecasts.parquet")
    else:
        print(f"NOTE: {forecasts_csv} not found, skipping.")
    # spot_features_csv = base / "spot_features.csv"
    # option_features_csv = base / "option_features.csv"
    # if spot_features_csv.exists():
    #     convert_spot_features(spot_features_csv, base / "spot_features.parquet")
    # else:
    #     print(f"NOTE: {spot_features_csv} not found, skipping.")
    # if option_features_csv.exists():
    #     convert_option_features(option_features_csv, base / "option_features.parquet")
    # else:
    #     print(f"NOTE: {option_features_csv} not found, skipping.")
    print("\nDone. Re-run generate_pnl3.py -- it will automatically prefer the .parquet "
          "files if present.")