#!/bin/bash
#SBATCH -A stf                     # Account name
#SBATCH -p gpu-l40                 # (or whatever your CPU partition is called)
#SBATCH -t 4:00:00                 # Time limit (hh:mm:ss)
#SBATCH -N 1                       # Number of nodes
#SBATCH --ntasks=1                 # total MPI tasks
#SBATCH --cpus-per-task=8          # cores for this job
#SBATCH --mem=64G                  # total RAM for the job

#SBATCH --job-name=eval_tdmr # Job name
#SBATCH --mail-type=ALL           # Send email on all events
#SBATCH --mail-user=afu3@uw.edu   # Your email address
#SBATCH --output=log/%x_%j.out    # Logs go to log/llama2_run1_JOBID.out

# Load necessary modules
module load cuda/11.8  # Adjust CUDA version as needed

# Activate your environment
source ~/.bashrc
conda activate leaderboard_generation

# Run your job
python tdm_eval.py \
  --gold_data_path leaderboard-generation/tdm_annotations.json \
  --normalized_tdm_output_path normalization/normalized_output.json \
  --eval_results_path tdm_eval_results/avg_results.json \
  --eval_values_path tdm_eval_results/individual_scores.json



python leaderboard_eval.py \
  --gold_leaderboards_file leaderboard-generation/leaderboards.json \
  --masked_leaderboards_file "" \
  --normalized_tdm_output_path normalization/normalized_output.json \
  --eval_results_file leaderboard_eval_results/avg_results.json \
  --eval_values_file leaderboard_eval_results/individual_scores.json



python tdm_embedding_normalization.py \
  --gold_tdm_path leaderboard-generation/tdm_annotations.json \
  --tdm_output_path output/deepseek/deepseek_run1-deepseek-05.05.2025-20_51_01/