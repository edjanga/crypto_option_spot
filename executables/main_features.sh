#!/bin/bash
clear
source ../venv/bin/activate
python3 ../generate_features.py --options=1 --freq=5min --D=1M --start_date=2023-01-01 --end_date=2025-07-01 \
--options=0
python3 ../generate_features.py --options=1 --freq=5min --D=1M --start_date=2023-01-01 --end_date=2025-07-01 \
--options=1
deactivate
