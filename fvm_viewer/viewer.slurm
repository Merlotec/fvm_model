#!/bin/bash
#SBATCH --job-name=fvm_viewer
#SBATCH --account=NIKIFORAKIS-SL3-CPU
#SBATCH --partition=cclake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=03:00:00
#SBATCH --output=/home/%u/fvm_viewer_%j.out

# DATA_DIR and PORT must be set by the caller (via --export or environment).
: "${DATA_DIR:?DATA_DIR must be set (pass via --export DATA_DIR=...)}"
: "${PORT:=8050}"

echo "Node:  $(hostname)"
echo "Data:  $DATA_DIR"
echo "Port:  $PORT"

source ~/rds/hpc-work/activate.sh
python ~/rds/hpc-work/flsim/fvm_model/fvm_viewer/viewer.py "$DATA_DIR" \
    --host 0.0.0.0 \
    --port "$PORT"
