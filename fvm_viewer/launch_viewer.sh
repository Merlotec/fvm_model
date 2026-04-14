#!/usr/bin/env bash
# Launch the FVM viewer on CSD3 and open an SSH tunnel automatically.
#
# Usage:
#   ./launch_viewer.sh <data_dir> [port]
#
# Example:
#   ./launch_viewer.sh /rds/user/nk624/hpc-work/artefacts/fvm_gen_datasets 8050

set -euo pipefail

# ---- configuration ----
REMOTE="csd3"
SLURM_SCRIPT="/home/nk624/rds/hpc-work/flsim/fvm_model/fvm_viewer/viewer.slurm"
# -----------------------

DATA_DIR="${1:?Usage: $0 <data_dir> [port]}"
PORT="${2:-8050}"

# ---- submit ----
echo "Submitting viewer job..."
JOBID=$(ssh "$REMOTE" "DATA_DIR='$DATA_DIR' PORT='$PORT' sbatch --parsable --export=ALL '$SLURM_SCRIPT'")
echo "Job ID: $JOBID"

# ---- wait for RUNNING ----
echo -n "Waiting for job to start"
while true; do
    INFO=$(ssh "$REMOTE" "squeue -j '$JOBID' -h -o '%T %N' 2>/dev/null" || true)
    STATE=$(awk '{print $1}' <<< "$INFO")
    NODE=$(awk '{print $2}' <<< "$INFO")

    case "$STATE" in
        RUNNING)
            echo "  running on $NODE"
            break
            ;;
        FAILED|CANCELLED|TIMEOUT|"")
            echo ""
            echo "Job ended unexpectedly (state: '${STATE:-gone}'). Check ~/fvm_viewer_${JOBID}.out"
            exit 1
            ;;
        *)
            echo -n "."
            sleep 5
            ;;
    esac
done

# ---- wait for server to respond ----
echo -n "Waiting for server"
for _ in $(seq 1 24); do
    sleep 5
    CODE=$(ssh "$REMOTE" "curl -s -o /dev/null -w '%{http_code}' http://${NODE}:${PORT}/ 2>/dev/null" || echo "000")
    if [ "$CODE" = "200" ]; then
        echo "  ready!"
        break
    fi
    echo -n "."
done

# ---- open tunnel ----
echo ""
echo "Tunnel: localhost:${PORT}  ->  ${NODE}:${PORT}  via ${REMOTE}"
echo "Open:   http://localhost:${PORT}"
echo "(Ctrl-C to close the tunnel and end the job)"
echo ""

# Cancel the SLURM job when the tunnel closes
trap "echo 'Cancelling job $JOBID...'; ssh '$REMOTE' 'scancel $JOBID'" EXIT

ssh -o ControlMaster=no -o ControlPath=none -NL "${PORT}:${NODE}:${PORT}" "$REMOTE"
