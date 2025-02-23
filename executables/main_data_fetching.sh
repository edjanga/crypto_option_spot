#!/bin/bash
clear
source ../venv/bin/activate
python3 ../data_centre/data.py --start_date=2022-08-02 --end_date=2022-08-03 --multithreading=0 --chunksize=1000 \
--freq=1h
deactivate