#!/bin/bash
clear
source ../venv/bin/activate
python3 ../generate_preliminary_portfolios.py --freq=1h --signal_filter=0 --target=RV
deactivate