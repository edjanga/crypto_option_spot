#!/bin/bash
clear
source ../venv/bin/activate
L=3M
for D in 1M 1W 1D; do
    for freq in 15min 30min 1h; do
        for target in RV returns; do
            for options in 0 1; do
              python3 ../generate_results.py --start_date=2023-01-01 --end_date=2025-04-01 --L=$L --freq=$freq \
              --regression=lstm --target=$target --options=$options --D=$D --to_csv=0
            done
        done
    done
done
deactivate