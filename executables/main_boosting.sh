#!/bin/bash
clear
source ../venv/bin/activate
L=3M
for D in 1M 1W 1D; do
    for freq in 1h 30min 15min; do
        for target in returns RV; do
            for options in 1 0; do
              python3 ../generate_results.py --start_date=2023-01-01 --end_date=2025-04-01 --L=$L --freq=$freq \
              --regression=lightgbm --target=$target --options=$options --D=$D --to_csv=0
            done
        done
    done
done
deactivate