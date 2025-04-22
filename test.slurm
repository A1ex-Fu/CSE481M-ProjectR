#!/bin/bash
#SBATCH --job-name=llama2_run1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=afu3@uw.edu
#SBATCH --partition=gpu-l40       # Or gpu-2080ti / gpu-l40s if you prefer
#SBATCH --gres=gpu:1
#SBATCH --account=stf
#SBATCH --time=4:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=log/%x_%j.out    # Logs go to log/llama2_run1_JOBID.out

# Load necessary modules
module load cuda/11.8  # Adjust CUDA version as needed

# Activate your environment
source ~/.bashrc
conda activate leaderboard_generation

# Run your job
python tdm_extraction.py \
  --env_file_path config/env.json \
  --exp_id llama2_run1 \
  --processed_docs_path processed_docsllama2_exp-21.04.2025-14_39_44/processed_docs.pkl \
  --papers_path paperDataset \
  --prompt_file prompts.json \
  --output_path output/llama2_run1/ \
  --model_type llama2 \
  --model_version 7b \
  --deployment_name local \
  --model_path "meta-llama/Llama-2-7b-hf" \
  --max_new_tokens 1024 \
  --seed 42 \
  --is_preprocessed_doc

