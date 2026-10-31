#!/bin/bash
# One job script, three stages, selected by $STAGE.  Submitted by
# scripts/submit_isambard_pair.sh -- see that file for the DAG and the knobs.
#
#   STAGE=plan      draw everything once + generate the meshes (serial, gmsh)
#   STAGE=generate  solve one shard of one arm ($ARM, $SLURM_ARRAY_TASK_ID)
#   STAGE=finish    reconcile the two arms, then verify the pair
#
#SBATCH --job-name=fvm_pair
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=%x_%A_%a.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

: "${OUT:?OUT must be set (dataset root)}"
: "${STAGE:?STAGE must be plan|generate|finish}"
SOLVER_DIR="${SOLVER_DIR:-$(cd .. && pwd)/fvm_solver}"
VENV="${VENV:-$SOLVER_DIR/.venv/$(uname -m)}"
PY="$VENV/bin/python"

export PATH="$HOME/.local/bin:$PATH"
export MPLBACKEND=Agg                  # no display on a compute node
export FVM_SOLVER_DIR="$SOLVER_DIR"    # run_gen._import_solver reads this
export FVM_DEVICE="${FVM_DEVICE:-cpu}" # ConfigEllipse.device defaults to cpu

echo "host=$(hostname)  stage=$STAGE  arch=$(uname -m)  device=$FVM_DEVICE"
echo "out=$OUT"
echo "solver=$SOLVER_DIR"

# ---------------------------------------------------------------------------
# Environment.  Built ONLY in the plan stage: plan is serial and every other
# stage depends on it, so there is no concurrent-creation race to guard.
# Separate venvs per architecture because Isambard-AI is aarch64 and MACS x86_64.
# ---------------------------------------------------------------------------
if [ "$STAGE" = plan ] && [ ! -f "$VENV/.installed" ]; then
    echo "== building venv at $VENV"
    uv venv --seed --python=3.12 "$VENV"
    uv pip install --python "$PY" "torch==2.11.0" \
        --index-url https://download.pytorch.org/whl/cu128
    grep -vE '^torch' "$SOLVER_DIR/requirements.txt" > "$VENV/requirements-rest.txt"
    uv pip install --python "$PY" -r "$VENV/requirements-rest.txt"
    touch "$VENV/.installed"
fi
[ -x "$PY" ] || { echo "no interpreter at $PY — did the plan stage run?" >&2; exit 1; }

if [ "$FVM_DEVICE" != cpu ]; then
    nvidia-smi --list-gpus || true
fi
"$PY" -c "import torch; print('torch', torch.__version__,
      'cuda', torch.cuda.is_available())" 2>/dev/null || true

GP=fvm_gen_tests/gen_pair.py

case "$STAGE" in
plan)
    : "${CONFIG:?CONFIG must be set}" "${TRAJ:?TRAJ must be set}"
    # Everything random happens here: contexts, ICs, dt and the meshes.  The
    # generate stage then does no sampling at all, which is why its shards need
    # no coordination and the corpus does not depend on how it was split.
    "$PY" "$GP" plan --config "$CONFIG" --out "$OUT" \
        --trajectories "$TRAJ" \
        --segments "${SEGMENTS:-4}" \
        --meshes "${MESHES:-8}" \
        --save-t-scope "${SAVE_T_SCOPE:-fixed}" \
        ${CONTEXTS:+--contexts "$CONTEXTS"} \
        ${SAVE_T:+--save-t "$SAVE_T"} \
        ${SEED:+--seed "$SEED"} \
        ${FORCE:+--force}
    ;;
generate)
    : "${ARM:?ARM must be switching|frozen}" "${NSHARDS:?NSHARDS must be set}"
    # Exits non-zero if any trajectory diverged.  That is NOT masked: a shard
    # that died for a real reason (broken venv, missing plan, OOM, timeout)
    # must not report success, or the only symptom would be verify failing
    # hours later with an unexplained count mismatch.  Nothing is gated on it
    # either -- the finish stage depends with afterany precisely because
    # divergence is expected -- so sacct stays a usable record of which shards
    # had trouble.  Re-running the array retries them and skips what is done.
    "$PY" "$GP" generate --out "$OUT" --arm "$ARM" \
        --shard "${SLURM_ARRAY_TASK_ID:-0}" --num-shards "$NSHARDS"
    ;;
finish)
    # reconcile makes the pair matched; verify proves it.  verify's exit code is
    # the pipeline's result, so a non-zero finish job means DO NOT TRAIN ON THIS.
    "$PY" "$GP" reconcile --out "$OUT"
    "$PY" "$GP" verify    --out "$OUT"
    ;;
*)
    echo "STAGE must be plan|generate|finish, got '$STAGE'" >&2; exit 2;;
esac
echo "stage $STAGE done"
