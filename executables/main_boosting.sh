#!/bin/bash
clear
source ../venv/bin/activate
L=1Y
for D in {1D,1W,1M}
  do
    for target in {returns,RV}
      do
        for freq in {15min,30min,1h}
          do
            for options in {0,1}
              do
                python3 ../generate_results.py --regression=lightgbm --L=$L --D=$D --options=$options --target=$target \
                --freq=$freq
              done
          done
      done
  done
deactivate