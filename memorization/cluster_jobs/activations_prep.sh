#!/bin/bash
# This script is a Slurm job submission script for running a Python script on a cluster.


# ---------------- Slurm job configuration ---------------- #
# #SBATCH is a directive for Slurm to configure the job.
# The lines below are Slurm directives that specify job parameters.

#SBATCH --job-name lid_estimations       # Name of the job (shows up in squeue)
#SBATCH --output logs/output.txt       # Stdout log file (in logs/ directory) (anything your script prints, like print()) will go to logs/output_<jobid>.txt
#SBATCH --error logs/error.txt         # Stderr log file (errors go here) (any errors or exceptions) go to logs/error_<jobid>.txt

#SBATCH --partition gpu22              # Partition to submit to (e.g., gpu24 is H100, 80GB VRAM)
#SBATCH --gres gpu:1                  # Request 1 GPU
#SBATCH --mem 100G                      # Amount of RAM to allocate
#SBATCH --cpus-per-task 4              # Number of CPU cores to allocate
#SBATCH --time 0-03:00:00              # Max wall time for the job (2 days)
#SBATCH --nodes 1                      # Number of nodes (machines)
#SBATCH --ntasks 1                     # Number of tasks (normally 1 for single GPU jobs)

# ---------------- Setup runtime environment ---------------- #

# Ensure shell is properly initialized (for mamba)
source ~/.bashrc

# Activate mamba (replace 'myenv' with your actual environment name)
source /BS/data_mani_compress/work/miniforge3/etc/profile.d/conda.sh
conda activate var_mem

# Load W&B secrets (make sure this file is secure and not shared)
source /BS/data_mani_compress/work/VAR/.wandb_secrets.sh

# ---------------- Run your code ---------------- #

# Move to your working directory (adjust this to where your code lives)
cd /BS/data_mani_compress/work/VAR/memorization/

# Determine run number from Slurm array index if present,
# otherwise use optional positional argument.
# - Array usage: sbatch --array=0-9 memorization/cluster_jobs/activations_prep.sh
# - Positional fallback: sbatch memorization/cluster_jobs/activations_prep.sh 3
# ${VAR:-DEFAULT}
RUN_NUMBER=${SLURM_ARRAY_TASK_ID:-${1:-0}}

python activation_prep.py --run_number "$RUN_NUMBER"