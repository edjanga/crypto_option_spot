#!/bin/bash
clear
source ../venv/bin/activate
for target in RV returns; do
  for freq in 15min 30min 1h; do
      python3 ../generate_results.py --start_date=2023-01-01 --end_date=2025-07-01 --L=3M \
      --regression=elastic --target=$target --to_csv=1 --freq=$freq
    done
  done
deactivate