#!/bin/bash
#SBATCH --job-name=llama70_run1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=afu3@uw.edu
#SBATCH --partition=ckpt-all
#SBATCH --gres=gpu:a100:1
#SBATCH --account=stf
#SBATCH --time=4:00:00
#SBATCH --mem-per-gpu=32G
#SBATCH --cpus-per-gpu=5
#SBATCH --output=log/%x_%j.out    # Logs go to log/llama2_run1_JOBID.out

# Load necessary modules
module load cuda/11.8  # Adjust CUDA version as needed

# Activate your environment
source ~/.bashrc
conda activate leaderboard_generation

# Run your job
python tdm_extraction.py \
  --env_file_path config/env.json \
  --exp_id llama70_run1 \
  --processed_docs_path processedPapers10-29.04.2025-15_12_41/processed_docs.pkl \
  --papers_path paperDataset \
  --prompt_file prompts.json \
  --output_path output/llama70_run1/ \
  --model_type llama-2-chat-70b \
  --model_version "" \
  --deployment_name local \
  --model_path "meta-llama/Llama-2-70b-hf" \
  --max_new_tokens 4096 \
  --seed 0 \
  --is_preprocessed_doc

