#!/bin/bash
clear
source ../venv/bin/activate
python3 ../generate_tables.py --start_date=2023-01-01 --end_date=2025-07-01
deactivate