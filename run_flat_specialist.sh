#!/bin/bash
#SBATCH --job-name=FlatSpecialist
#SBATCH --output=logs/flat_specialist_%j.out
#SBATCH --error=logs/flat_specialist_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --partition=academic
#SBATCH --account=micro-515

source .venv/bin/activate
mkdir -p logs

python train_flat_specialist.py \
    --n_gen     1000 \
    --pop_size  256  \
    --n_repeats 3    \
    --n_steps   1000 \
    --out_dir   results/flat_specialist_cmaes/ \
    --warm_start_dir results_git/Paul_best_flat/ \
