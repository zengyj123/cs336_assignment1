#!/bin/bash
#SBATCH --job-name=tok_exp
#SBATCH --partition=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=tok_exp_%j.log

source ~/miniconda3/etc/profile.d/conda.sh
conda activate cs336
cd $SLURM_SUBMIT_DIR
python -m cs336_basics.tokenizer_experiments
