#!/bin/bash
clear
source ../venv/bin/activate
python3 ../data_centre/data.py --start_date=2023-01-01 --end_date=2025-07-01 --type=spot --freq=1m
python3 ../data_centre/data.py --start_date=2023-01-01 --end_date=2025-07-01
deactivate
