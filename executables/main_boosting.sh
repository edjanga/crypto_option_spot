#!/bin/bash
clear
source ../venv/bin/activate
L=3M
for D in 1D 1W 1M; do
    for freq in 1h 30min 15min; do
        for target in RV returns; do
            for options in 1 0; do
              python3 ../generate_results.py --start_date=2023-01-01 --end_date=2025-07-01 --L=$L --freq=$freq \
              --regression=lightgbm --target=$target --options=$options --D=$D --to_csv=0
            done
        done
    done
done
deactivate