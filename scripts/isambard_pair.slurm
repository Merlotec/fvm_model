#!/bin/bash
# One job script, three stages, selected by $STAGE.  Submitted by
# scripts/submit_isambard_pair.sh -- see that file for the DAG and the knobs.
#
#   STAGE=plan      draw everything once + generate the meshes (serial, gmsh)
#   STAGE=generate  solve one shard of one arm ($ARM, $SLURM_ARRAY_TASK_ID)
#   STAGE=finish    reconcile the two arms, then verify the pair
#   STAGE=env       build the venv and exit -- run this on a LOGIN NODE first
#
#SBATCH --job-name=fvm_pair
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=%x_%A_%a.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

: "${OUT:?OUT must be set (dataset root)}"
: "${STAGE:?STAGE must be env|plan|generate|finish}"
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
# Environment.  Separate venvs per architecture: Isambard-AI is aarch64 and
# MACS x86_64.
#
# Build this on a LOGIN NODE (STAGE=env), not inside a job.  Compute nodes
# usually have no outbound internet, and requirements.txt needs some: it carries
# a git+https dependency (cprint), which fvm_equation.py and fvm_mesh.py both
# import.  `uv pip install -r` is one transaction, so a blocked git fetch takes
# numpy and everything else down with it -- that is how a venv ends up holding
# torch but no numpy.
# ---------------------------------------------------------------------------
build_env() {
    echo "== building venv at $VENV  (arch $(uname -m))"
    command -v uv >/dev/null 2>&1 || {
        echo "uv is not on PATH. Install it, or set VENV to an existing venv." >&2
        return 1; }
    uv venv --seed --python=3.12 "$VENV"
    # Three installs rather than one, so a failure names what failed instead of
    # silently taking the rest of the file with it.
    echo "-- torch"
    uv pip install --python "$PY" "torch==2.11.0" \
        --index-url https://download.pytorch.org/whl/cu128
    echo "-- PyPI packages"
    grep -vE '^[[:space:]]*(#|$|torch|git\+)' "$SOLVER_DIR/requirements.txt" \
        > "$VENV/req-pypi.txt"
    uv pip install --python "$PY" -r "$VENV/req-pypi.txt"
    echo "-- git packages (needs outbound https; run this on a login node)"
    grep -E '^[[:space:]]*git\+' "$SOLVER_DIR/requirements.txt" \
        > "$VENV/req-git.txt" || true
    if [ -s "$VENV/req-git.txt" ]; then
        uv pip install --python "$PY" -r "$VENV/req-git.txt" || {
            echo "" >&2
            echo "The git dependency failed. If this node has no outbound https," >&2
            echo "clone it where you do and install from the clone, e.g.:" >&2
            echo "  git clone https://github.com/Maccy-Z/c_print.git ~/c_print" >&2
            echo "  $VENV/bin/python -m pip install ~/c_print" >&2
            return 1; }
    fi
    # .installed is written ONLY once every module the solver needs actually
    # imports, so a half-built venv can never be mistaken for a finished one.
    "$PY" "$SLURM_SUBMIT_DIR_CHK/scripts/check_env.py" || return 1
    touch "$VENV/.installed"
    echo "== venv ready: $VENV"
}

SLURM_SUBMIT_DIR_CHK="${SLURM_SUBMIT_DIR:-$PWD}"

if [ "$STAGE" = env ]; then
    build_env || exit 1
    echo "stage env done"; exit 0
fi

if [ ! -f "$VENV/.installed" ]; then
    if [ "$STAGE" = plan ]; then
        echo "no ready venv at $VENV — attempting to build it here."
        echo "(prefer: STAGE=env bash scripts/isambard_pair.slurm, on a login node)"
        build_env || exit 1
    else
        echo "venv at $VENV is not ready (no .installed marker)." >&2
        echo "Build it on a LOGIN NODE first:" >&2
        echo "  OUT=$OUT STAGE=env bash scripts/isambard_pair.slurm" >&2
        exit 1
    fi
fi
[ -x "$PY" ] || { echo "no interpreter at $PY" >&2; exit 1; }

# Only the solving stage cares about a GPU; reconcile/verify run on a CPU node,
# where reporting "No devices found" is just misleading noise.
if [ "$STAGE" = generate ]; then
    if [ "$FVM_DEVICE" != cpu ]; then
        nvidia-smi --list-gpus || true
        "$PY" -c "import torch; assert torch.cuda.is_available(), \
            'FVM_DEVICE=$FVM_DEVICE but torch sees no CUDA device'" || exit 1
    fi
    "$PY" -c "import torch; print('torch', torch.__version__,
          'cuda', torch.cuda.is_available())"
fi

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
    echo "STAGE must be env|plan|generate|finish, got '$STAGE'" >&2; exit 2;;
esac
echo "stage $STAGE done"
