#!/bin/bash
clear
source ../venv/bin/activate

# Set the maximum number of concurrent jobs
MAX_JOBS=4  # Adjust based on your system's capabilities

# Function to wait for jobs to complete when we hit the limit
wait_for_jobs() {
    while [ $(jobs -r | wc -l) -ge $MAX_JOBS ]; do
        sleep 1
    done
}

# Function to run a single job
run_job() {
    local L=$1
    local D=$2
    local freq=$3
    local target=$4
    local options=$5

    echo "Starting job: L=$L, D=$D, freq=$freq, target=$target, options=$options"
    python3 ../generate_results.py \
        --regression=lasso \
        --L=$L \
        --D=$D \
        --options=$options \
        --target=$target \
        --freq=$freq \
        --start_date=2023-01-01 \
        --end_date=2025-04-01 \
        --to_csv=0 \
        2>&1 | while IFS= read -r line; do
            echo "[L=$L,D=$D,freq=$freq,target=$target,options=$options] $line"
        done

    echo "Completed job: L=$L, D=$D, freq=$freq, target=$target, options=$options"
}

L=3M
total_jobs=0

# Count total jobs for progress tracking
for D in 1M 1W 1D; do
    for freq in 1h 30min 15min; do
        for target in RV returns; do
            for options in 1 0; do
                ((total_jobs++))
            done
        done
    done
done

echo "Total jobs to run: $total_jobs"
echo "Maximum concurrent jobs: $MAX_JOBS"
echo "Starting parallel execution..."
echo

current_job=0

# Run all combinations in parallel
for D in 1M 1W 1D; do
    for freq in 1h 30min 15min; do
        for target in RV returns; do
            for options in 1 0; do
                ((current_job++))
                echo "Queuing job $current_job/$total_jobs"

                # Wait if we've reached the maximum number of concurrent jobs
                wait_for_jobs

                # Run the job in the background
                run_job "$L" "$D" "$freq" "$target" "$options" &
            done
        done
    done
done

# Wait for all remaining jobs to complete
echo
echo "All jobs queued. Waiting for completion..."
wait

echo
echo "All jobs completed successfully!"
deactivate