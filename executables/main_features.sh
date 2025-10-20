#!/bin/bash
clear
source ../venv/bin/activate
python3 ../generate_features.py --D=$D --options=1 --freq=5min --D=1M --start_date=2023-01-01 --end_date=2025-07-01
deactivate
