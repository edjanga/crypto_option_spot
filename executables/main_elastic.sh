##!/bin/bash
#clear
#source ../venv/bin/activate
#L=3M
#for D in 1M 1W 1D; do
#    for freq in 1h 30min 15min; do
#        for target in returns RV; do
#            for options in 1 0; do
#              python3 ../generate_results.py --start_date=2023-01-01 --end_date=2025-07-01 --L=$L --freq=$freq \
#              --regression=elastic --target=$target --options=$options --D=$D --to_csv=0
#            done
#        done
#    done
#done
#MAX_JOBS=4
#run_job() {
#    local L=$1
#    local freq=$2
#    local target=$3
#    local options=$4
#    local D=$5
#    python3 ../generate_results.py --start_date=2025-01-01 --end_date=2025-07-01 --L=$L --freq=$freq \
#    --regression=elastic --target=$target --options=$options --D=$D --to_csv=0
#}
#export -f run_job
#for D in 1D 1W 1M; do
#    for freq in 1h 30min 15min; do
#        for target in RV returns; do
#            for options in 1 0; do
#                while [ $(jobs -r | wc -l) -ge $MAX_JOBS ]; do
#                    sleep 0.1
#                done
#                run_job "$L" "$freq" "$target" "$options" "$D" &
#            done
#        done
#    done
#done
#wait
#echo "All jobs completed!"
#deactivate
L=3M
MAX_JOBS=2
run_job() {
    local L=$1
    local freq=$2
    local target=$3
    local options=$4
    local D=$5
    local start_date=$6
    local end_date=$7

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Running: D=$D freq=$freq target=$target options=$options dates=$start_date to $end_date"

    python3 ../generate_results.py \
        --start_date=$start_date \
        --end_date=$end_date \
        --L=$L \
        --freq=$freq \
        --regression=elastic \
        --target=$target \
        --options=$options \
        --D=$D \
        --to_csv=0
}
export -f run_job
date_ranges=(
    "2023-01-01 2024-01-01"
    "2024-01-01 2025-01-01"
    "2025-01-01 2025-07-01"
)
# Process each date range sequentially
for date_range in "${date_ranges[@]}"; do
    read -r start_date end_date <<< "$date_range"
    echo ""
    echo "=========================================="
    echo "Processing date range: $start_date to $end_date"
    echo "=========================================="
    # Run all parameter combinations in parallel for this date range
    for D in 1D 1W 1M; do
        for freq in 1h 30min 15min; do
            for target in RV returns; do
                for options in 1 0; do
                    while [ $(jobs -r | wc -l) -ge $MAX_JOBS ]; do
                        sleep 0.1
                    done
                    run_job "$L" "$freq" "$target" "$options" "$D" "$start_date" "$end_date" &
                done
            done
        done
    done
    echo "Waiting for all jobs in date range $start_date to $end_date to complete..."
    wait
    echo "✓ Completed date range: $start_date to $end_date"
done
echo ""
echo "=========================================="
echo "All jobs completed successfully!"
echo "=========================================="
deactivate