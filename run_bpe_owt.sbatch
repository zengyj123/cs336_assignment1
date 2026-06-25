#!/bin/bash
#SBATCH --job-name=bpe_owt
#SBATCH --partition=normal
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=bpe_owt_%j.log

export PATH="$HOME/.local/bin:$PATH"
cd $SLURM_SUBMIT_DIR
uv run --no-project --with regex python -m cs336_basics.run_tinystories data/owt_train.txt 32000 owt
