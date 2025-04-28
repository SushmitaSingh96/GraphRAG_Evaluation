#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e
set -o pipefail

# Check for mode argument
if [ -z "$1" ]; then
    echo "Usage: $0 [index|tune|query|baseline]"
    exit 1
fi

MODE=$1

case "$MODE" in
    index)
        JOB_NAME="GraphRAG_Indexing"
        LOG_DIR="batch_logs/logs/indexing"
        ;;
    tune)
        JOB_NAME="GraphRAG_Tuning"
        LOG_DIR="batch_logs/logs/tuning"
        ;;
    query)
        JOB_NAME="GraphRAG_Querying"
        LOG_DIR="batch_logs/logs/querying"
        ;;
    baseline)
        JOB_NAME="GraphRAG_Baseline"
        LOG_DIR="batch_logs/logs/baseline"
        ;;
    *)
        echo "Invalid mode: $MODE"
        echo "Usage: $0 [index|tune|query|baseline]"
        exit 1
        ;;
esac

mkdir -p "$(dirname "$LOG_DIR")"

#SBATCH --job-name=${JOB_NAME}
#SBATCH --output=${LOG_DIR}_%j.out   # Output file for the job log
#SBATCH --error=${LOG_DIR}_%j.err    # File to log errors
#SBATCH --gres=gpu:a100:1 -C a100_80 # Request 1 A100 GPU with 80GB memory      
#SBATCH --time=12:00:00              # Request 12 hours of runtime
#SBATCH --ntasks=1                   # Run a single task
#SBATCH --cpus-per-task=16           # Request 16 CPU cores
#SBATCH --partition=a100             # Use the A100 partition

# Activate conda environment
source /apps/jupyterhub/jh3.1.1-py3.11/etc/profile.d/conda.sh
conda activate pytorch-2.3.0

# Set proxy if needed
export http_proxy="http://proxy:80"
export https_proxy="http://proxy:80"
export no_proxy="localhost,127.0.0.1"
export NO_PROXY="localhost,127.0.0.1" 

echo "Starting $MODE job at $(date)"
echo "Running on node(s): $SLURM_NODELIST"
echo "Using $SLURM_CPUS_PER_TASK CPU cores and GPU: $SLURM_GPUS"

# Run the corresponding Python script
if [ "$MODE" == "index" ]; then
    python path/to/scripts/indexing.py
elif [ "$MODE" == "tune" ]; then
    python path/to/scripts/prompt_tune.py
elif [ "$MODE" == "query" ]; then
    python path/to/scripts/query_graphrag.py
elif [ "$MODE" == "baseline" ]; then
    python path/to/scripts/baseline.py
fi

echo "$MODE job completed successfully at $(date)"
