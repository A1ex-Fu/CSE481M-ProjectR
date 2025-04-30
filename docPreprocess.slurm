#!/bin/bash
#SBATCH --job-name=my_ckpt_job        # Name of the job
#SBATCH --partition=ckpt              # Use the ckpt partition
#SBATCH --nodes=1                     # Number of nodes
#SBATCH --ntasks=1                    # Number of tasks (processes)
#SBATCH --cpus-per-task=4             # Number of CPUs per task
#SBATCH --mem=8G                      # Memory per node
#SBATCH --time=4:00:00                # Time limit (hh:mm:ss)
#SBATCH --output=logs/%x-%j.out       # Standard output log
#SBATCH --error=logs/%x-%j.err        # Standard error log
#SBATCH --mail-type=ALL          # Email notifications 
#SBATCH --mail-user=afu3@uw.edu # Replace with your UW email

# Load necessary modules
module load cuda/11.8  # Adjust CUDA version as needed

# Activate your environment
source ~/.bashrc
conda activate leaderboard_generation

# Run your job
python doc_preprocessV2.py --process_id "10" --papers_path paperDataset/ --output_path processedPapers
