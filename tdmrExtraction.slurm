#!/bin/bash
#SBATCH -A stf                     # Account name
#SBATCH -p ckpt-all                # Partition name
#SBATCH -t 4:00:00                 # Time limit (hh:mm:ss)
#SBATCH -N 1                       # Number of nodes
#SBATCH --gpus=a100:1             # Request 1 A100 GPU
#SBATCH --mem-per-gpu=64G         # Memory per GPU
#SBATCH --cpus-per-gpu=5          # CPUs per GPU

#SBATCH --job-name=llama2_7b_run10 # Job name
#SBATCH --mail-type=ALL           # Send email on all events
#SBATCH --mail-user=afu3@uw.edu   # Your email address
#SBATCH --output=log/%x_%j.out    # Logs go to log/llama2_run1_JOBID.out

# Load necessary modules
module load cuda/11.8  # Adjust CUDA version as needed

# Activate your environment
source ~/.bashrc
conda activate leaderboard_generation

# Run your job

python tdm_extraction.py \
  --env_file_path config/env.json \
  --exp_id llama2_7b_run2 \
  --processed_docs_path processedPapers10-29.04.2025-15_12_41/processed_docs.pkl \
  --papers_path paperDataset \
  --prompt_file prompts.json \
  --output_path output/llama2_7b_run7/ \
  --model_type llama2 \
  --model_version 7b \
  --deployment_name local \
  --model_path "meta-llama/llama-2-7b-hf/" \
  --max_new_tokens 1024 \
  --seed 42 \
  --is_preprocessed_doc
