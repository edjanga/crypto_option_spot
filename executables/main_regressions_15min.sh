#!/bin/sh
clear
source ../venv/bin/activate
freq=15min
####################################################  1M  ##############################################################
python3 ../generate_results.py --start_date=2023-01-01 --end_date=2025-07-01 --L=3M --target=RV --to_csv=1 \
--freq=$freq --options=1 --D=1M --cpu_count=1
python3 ../generate_results.py --start_date=2023-01-01 --end_date=2025-07-01 --L=3M --target=RV --to_csv=1 \
--freq=$freq --options=0 --D=1M --cpu_count=2
python3 ../generate_results.py --start_date=2023-01-01 --end_date=2025-07-01 --L=3M --target=returns --to_csv=1 \
--freq=$freq --options=1 --D=1M --cpu_count=2
python3 ../generate_results.py --start_date=2023-01-01 --end_date=2025-07-01 --L=3M --target=returns --to_csv=1 \
--freq=$freq --options=0 --D=1M --cpu_count=2
####################################################  1W  ##############################################################
python3 ../generate_results.py --start_date=2023-01-01 --end_date=2025-07-01 --L=3M --target=RV --to_csv=1 \
--freq=$freq --options=1 --D=1W --cpu_count=2
python3 ../generate_results.py --start_date=2023-01-01 --end_date=2025-07-01 --L=3M --target=RV --to_csv=1 \
--freq=$freq --options=0 --D=1W --cpu_count=2
python3 ../generate_results.py --start_date=2023-01-01 --end_date=2025-07-01 --L=3M --target=returns --to_csv=1 \
--freq=$freq --options=1 --D=1W --cpu_count=2
python3 ../generate_results.py --start_date=2023-01-01 --end_date=2025-07-01 --L=3M \
--target=returns --to_csv=1 --freq=$freq --options=0 --D=1W --cpu_count=2
####################################################  1D  ##############################################################
python3 ../generate_results.py --start_date=2023-01-01 --end_date=2025-07-01 --L=3M --target=RV --to_csv=1 \
--freq=$freq --options=1 --D=1D --cpu_count=2
python3 ../generate_results.py --start_date=2023-01-01 --end_date=2025-07-01 --L=3M \
--target=RV --to_csv=1 --freq=$freq --options=0 --D=1D --cpu_count=2
python3 ../generate_results.py --start_date=2023-01-01 --end_date=2025-07-01 --L=3M --target=returns --to_csv=1 \
--freq=$freq --options=1 --D=1D --cpu_count=2
python3 ../generate_results.py --start_date=2023-01-01 --end_date=2025-07-01 --L=3M --target=returns --to_csv=1 \
--freq=$freq --options=0 --D=1D --cpu_count=2
deactivate