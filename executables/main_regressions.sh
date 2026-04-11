#!/bin/sh
clear
source ../venv/bin/activate
--target_path=../data_centre
for regression in ml dl; do
  for freq in 1h 30min 15min; do
    ####################################################  1M  ##########################################################
    python3 ../generate_results.py --start_date=2023-01-01 --end_date=2025-07-01 --L=3M --regression=$regression \
      --target=RV --to_csv=1 --freq=$freq --target_path=$target_path --options=1 --D=1M --cpu_count=10
    python3 ../generate_results.py --start_date=2023-01-01 --end_date=2025-07-01 --L=3M --regression=$regression \
    --target=RV --to_csv=1 --freq=$freq --target_path=$target_path --options=0 --D=1M --cpu_count=10
    python3 ../generate_results.py --start_date=2023-01-01 --end_date=2025-07-01 --L=3M --regression=$regression \
      --target=returns --to_csv=1 --freq=$freq --target_path=$target_path --options=1 --D=1M --cpu_count=10
    python3 ../generate_results.py --start_date=2023-01-01 --end_date=2025-07-01 --L=3M --regression=$regression \
    --target=returns --to_csv=1 --freq=$freq --target_path=$target_path --options=0 --D=1M --cpu_count=10
    ####################################################  1W  ##########################################################
    python3 ../generate_results.py --start_date=2023-01-01 --end_date=2025-07-01 --L=3M --regression=$regression \
      --target=RV --to_csv=1 --freq=$freq --target_path=$target_path --options=1 --D=1W --cpu_count=10
    python3 ../generate_results.py --start_date=2023-01-01 --end_date=2025-07-01 --L=3M --regression=$regression \
    --target=RV --to_csv=1 --freq=$freq --target_path=$target_path --options=0 --D=1W --cpu_count=10
    python3 ../generate_results.py --start_date=2023-01-01 --end_date=2025-07-01 --L=3M --regression=$regression \
      --target=returns --to_csv=1 --freq=$freq --target_path=$target_path --options=1 --D=1W --cpu_count=10
    python3 ../generate_results.py --start_date=2023-01-01 --end_date=2025-07-01 --L=3M --regression=$regression \
    --target=returns --to_csv=1 --freq=$freq --target_path=$target_path --options=0 --D=1W --cpu_count=10
    ####################################################  1D  ##########################################################
    python3 ../generate_results.py --start_date=2023-01-01 --end_date=2025-07-01 --L=3M --regression=$regression \
      --target=RV --to_csv=1 --freq=$freq --target_path=$target_path --options=1 --D=1D --cpu_count=10
    python3 ../generate_results.py --start_date=2023-01-01 --end_date=2025-07-01 --L=3M --regression=$regression \
    --target=RV --to_csv=1 --freq=$freq --target_path=$target_path --options=0 --D=1D --cpu_count=10
    python3 ../generate_results.py --start_date=2023-01-01 --end_date=2025-07-01 --L=3M --regression=$regression \
      --target=returns --to_csv=1 --freq=$freq --target_path=$target_path --options=1 --D=1D --cpu_count=10
    python3 ../generate_results.py --start_date=2023-01-01 --end_date=2025-07-01 --L=3M --regression=$regression \
    --target=returns --to_csv=1 --freq=$freq --target_path=$target_path --options=0 --D=1D --cpu_count=10
  done
done

deactivate