#!/bin/bash
#SBATCH -A stf                     # Account name
#SBATCH -p compute                 # (or whatever your CPU partition is called)
#SBATCH -t 4:00:00                 # Time limit (hh:mm:ss)
#SBATCH -N 1                       # Number of nodes
#SBATCH --ntasks=1                 # total MPI tasks
#SBATCH --cpus-per-task=8          # cores for this job
#SBATCH --mem=64G                  # total RAM for the job

#SBATCH --job-name=deepseek_tdmr # Job name
#SBATCH --mail-type=ALL           # Send email on all events
#SBATCH --mail-user=xllegion@uw.edu   # Your email address
#SBATCH --output=log/%x_%j.out    # Logs go to log/llama2_run1_JOBID.out

# Load necessary modules
module load cuda/11.8  # Adjust CUDA version as needed

# Activate your environment
source ~/.bashrc
conda activate leaderboard_generation

# Run your job

python tdm_extraction.py \
  --env_file_path config/env.json \
  --exp_id deepseek_run1 \
  --processed_docs_path processedPapers10-29.04.2025-15_12_41/processed_docs.pkl \
  --papers_path paperDataset \
  --prompt_file prompts.json \
  --output_path output/deepseek/ \
  --model_type deepseek \
  --model_version r1 \
  --deployment_name local \
  --model_path "" \
  --is_few_shot \
  --max_new_tokens 1024 \
  --seed 42 \
  --is_preprocessed_doc
