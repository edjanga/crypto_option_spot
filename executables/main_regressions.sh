#!/bin/sh
#SBATCH -J option_forecasting_gaussgpu01
#SBATCH --clusters=srf_gpu_01
#SBATCH --partition=standard-gpu
#SBATCH --time=24:00:00
##SBATCH --ntasks=1
# SBATCH --cpus-per-task=15
#SBATCH --mem=100GB
#SBATCH --nodes=1
#SBATCH --nodelist=gaussgpu01.cpu.stats.ox.ac.uk
#SBATCH --gres=gpu:1
#SBATCH --output="./slurm_%j.out"
#SBATCH --error="./slurm_%j.err"
#SBATCH --mail-user=emmanuel.djanga@stats.ox.ac.uk
#SBATCH --mail-type=ALL
clear
source ../venv/bin/activate
for regression in ml dl; do
  for freq in 1h 30min 15min; do
    ####################################################  1M  ##########################################################
    python3 ../generate_results.py --start_date=2023-01-01 --end_date=2025-07-01 --L=3M --regression=$regression \
      --target=RV --to_csv=1 --freq=$freq --target_path=../data_centre --options=1 --D=1M --cpu_count=10
    python3 ../generate_results.py --start_date=2023-01-01 --end_date=2025-07-01 --L=3M --regression=$regression \
    --target=RV --to_csv=1 --freq=$freq --target_path=../data_centre --options=0 --D=1M --cpu_count=10
    python3 ../generate_results.py --start_date=2023-01-01 --end_date=2025-07-01 --L=3M --regression=$regression \
      --target=returns --to_csv=1 --freq=$freq --target_path=../data_centre --options=1 --D=1M --cpu_count=10
    python3 ../generate_results.py --start_date=2023-01-01 --end_date=2025-07-01 --L=3M --regression=$regression \
    --target=returns --to_csv=1 --freq=$freq --target_path=../data_centre --options=0 --D=1M --cpu_count=10
    ####################################################  1W  ##########################################################
    python3 ../generate_results.py --start_date=2023-01-01 --end_date=2025-07-01 --L=3M --regression=$regression \
      --target=RV --to_csv=1 --freq=$freq --target_path=../data_centre --options=1 --D=1W --cpu_count=10
    python3 ../generate_results.py --start_date=2023-01-01 --end_date=2025-07-01 --L=3M --regression=$regression \
    --target=RV --to_csv=1 --freq=$freq --target_path=../data_centre --options=0 --D=1W --cpu_count=10
    python3 ../generate_results.py --start_date=2023-01-01 --end_date=2025-07-01 --L=3M --regression=$regression \
      --target=returns --to_csv=1 --freq=$freq --target_path=../data_centre --options=1 --D=1W --cpu_count=10
    python3 ../generate_results.py --start_date=2023-01-01 --end_date=2025-07-01 --L=3M --regression=$regression \
    --target=returns --to_csv=1 --freq=$freq --target_path=../data_centre --options=0 --D=1W --cpu_count=10
    ####################################################  1D  ##########################################################
    python3 ../generate_results.py --start_date=2023-01-01 --end_date=2025-07-01 --L=3M --regression=$regression \
      --target=RV --to_csv=1 --freq=$freq --target_path=../data_centre --options=1 --D=1D --cpu_count=10
    python3 ../generate_results.py --start_date=2023-01-01 --end_date=2025-07-01 --L=3M --regression=$regression \
    --target=RV --to_csv=1 --freq=$freq --target_path=../data_centre --options=0 --D=1D --cpu_count=10
    python3 ../generate_results.py --start_date=2023-01-01 --end_date=2025-07-01 --L=3M --regression=$regression \
      --target=returns --to_csv=1 --freq=$freq --target_path=../data_centre --options=1 --D=1D --cpu_count=10
    python3 ../generate_results.py --start_date=2023-01-01 --end_date=2025-07-01 --L=3M --regression=$regression \
    --target=returns --to_csv=1 --freq=$freq --target_path=../data_centre --options=0 --D=1D --cpu_count=10
  done
done

deactivate