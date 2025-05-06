#!/bin/bash
clear
source ../venv/bin/activate
python3 ../data_centre/data.py --start_date=2025-01-01 --end_date=2025-04-01 --chunksize=100000
deactivate
