#!/bin/bash
clear
source ../venv/bin/activate
L=3M
for D in {1M,1W,1D}
  do
    for target in {RV,returns}
      do
        for freq in {15min,30min,1h}
          do
          for options in {1,0}
            do
              python3 ../generate_results.py --regression=lightgbm --L=$L --D=$D --options=$options --target=$target \
              --freq=$freq --start_date=2023-01-01 --end_date=2025-04-01
            done
        done
      done
  done
deactivate