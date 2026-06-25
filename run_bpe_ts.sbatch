#!/bin/bash
#SBATCH --job-name=bpe_ts
#SBATCH --partition=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=bpe_ts_%j.log

export PATH="$HOME/.local/bin:$PATH"
cd $SLURM_SUBMIT_DIR
uv run --no-project --with regex python -m cs336_basics.run_tinystories data/TinyStoriesV2-GPT4-train.txt 10000 ts
