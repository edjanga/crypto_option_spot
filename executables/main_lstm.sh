#!/bin/bash
clear
source ../venv/bin/activate
L=3M
#for D in 1M 1W 1D; do
#    for freq in 15min 30min 1h; do
#        for target in RV returns; do
#            for options in 1 0; do
#              python3 ../generate_results.py --start_date=2024-09-01 --end_date=2025-07-01 --L=$L --freq=$freq \
#              --regression=lstm --target=$target --options=$options --D=$D --to_csv=0
#            done
#        done
#    done
#done
MAX_JOBS=2
run_job() {
    local L=$1
    local freq=$2
    local target=$3
    local options=$4
    local D=$5
    python3 ../generate_results.py --start_date=2025-01-01 --end_date=2025-07-01 --L=$L --freq=$freq \
    --regression=lstm --target=$target --options=$options --D=$D --to_csv=0
}
export -f run_job
for D in 1D 1W 1M; do
    for freq in 1h 30min 15min; do
        for target in RV returns; do
            for options in 1 0; do
                while [ $(jobs -r | wc -l) -ge $MAX_JOBS ]; do
                    sleep 0.1
                done
                run_job "$L" "$freq" "$target" "$options" "$D" &
            done
        done
    done
done
wait
echo "All jobs completed!"
deactivate