#!/bin/bash
#SBATCH --job-name=fullNormRun4
#SBATCH --mail-type=ALL
#SBATCH --mail-user=afu3@uw.edu
#SBATCH --partition=ckpt-all
#SBATCH --gres=gpu:a100:1
#SBATCH --account=stf
#SBATCH --time=4:00:00
#SBATCH --mem-per-gpu=64G
#SBATCH --cpus-per-gpu=5
#SBATCH --output=log/%x%j.out    # Logs go to log/llama2_run1_JOBID.out

# Load necessary modules
module load cuda/11.8  # Adjust CUDA version as needed

# Activate your environment
source ~/.bashrc
conda activate leaderboard_generation

# Run your job
python tdm_llm_normalization.py   --gold_tdm_path leaderboard-generation/tdm_annotations.json   --tdm_output_path output/deepseek/deepseek_run1-deepseek-05.05.2025-20_51_01/  --prompt_file prompts.json

