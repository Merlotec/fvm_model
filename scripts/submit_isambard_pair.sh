#!/bin/bash
# Submit the whole matched-pair generation pipeline on Isambard as a SLURM DAG.
#
#   plan ──┬── generate switching  (array 0..N-1) ──┬── finish (reconcile+verify)
#          └── generate frozen     (array 0..N-1) ──┘
#
# The two arms run concurrently; both wait for plan because plan is what draws
# the contexts, ICs, dt and meshes they share.  finish waits for BOTH arrays
# with `afterany`, NOT afterok: a generate shard exits non-zero when a
# trajectory diverges, which is expected and is exactly what reconcile is for.
# afterok there would leave the pipeline permanently queued.
#
# Run from the fvm_model repo root on a login node:
#
#   OUT=$HOME/data/pair_A TRAJ=128 ./scripts/submit_isambard_pair.sh
#
# Knobs (all optional unless marked):
#   OUT*          dataset root (also where plan.json lands)
#   TRAJ*         trajectories per arm.  Plan ~5% more than you need: some
#                 contexts are too stiff for some ICs and diverge, and
#                 reconcile drops those from both arms.
#   CONFIG        distributions to draw from   [fvm_gen/gen_alternating.json]
#   SEGMENTS      segments per trajectory      [4]
#   CONTEXTS      size of the shared context pool [= TRAJ]
#   MESHES        distinct geometries          [8]
#   SAVE_T_SCOPE  fixed | trajectory | segment [fixed]
#   SAVE_T        the fixed dt                 [centre of the config's spec]
#   SEED          plan seed                    [fresh entropy, recorded]
#   NSHARDS       array tasks per arm          [9]  (should divide TRAJ)
#   THROTTLE      max concurrent array tasks per arm (SLURM %n) [unset]
#   FVM_DEVICE    cpu | cuda                   [cpu]
#   GPUS          GPUs per generate task       [0 on cpu, else 1]
#   CPUS          CPUs per task                [8]
#   ACCOUNT       --account                    [cluster default]
#   PARTITION     --partition                  [cluster default]
#   T_PLAN        plan walltime                [02:00:00]
#   T_GEN         per-shard walltime           [04:00:00]
#   T_FIN         finish walltime              [00:30:00]
#   SOLVER_DIR    fvm_solver checkout          [../fvm_solver]
#   VENV          python env                   [$SOLVER_DIR/.venv/$(uname -m)]
#   SKIP_ENV_CHECK=1  submit without checking the venv is built
#   FORCE=1       let plan replace an existing plan.json
#   DRY=1         print the sbatch commands and exit
#
# Settings files, so none of this has to be retyped.  They LAYER:
#
#   ENV_FILE=<path>[:<path>...]  explicit, sourced first (a missing one is an error)
#   LOCAL_ENV=<path>             implicit  [scripts/isambard_pair.env]; LOCAL_ENV= skips it
#
# Precedence, highest first:
#
#   real environment  >  ENV_FILE  >  LOCAL_ENV
#
# The normal invocation uses BOTH -- the experiment from ENV_FILE, the machine
# from LOCAL_ENV, which is picked up automatically:
#
#   ENV_FILE=scripts/experiments/pair_A.env ./scripts/submit_isambard_pair.sh
#
# Every file assigns with `: "${VAR:=value}"` rather than `VAR=value`, which is
# what makes the layering work and lets a one-off override still win:
#
#   OUT=$HOME/data/other ENV_FILE=scripts/experiments/pair_A.env ./scripts/...

#
# What belongs where:
#
#   machine-specific, DO NOT COMMIT (scripts/isambard_pair.env, gitignored):
#       OUT, ACCOUNT, PARTITION, SOLVER_DIR, FVM_DEVICE, CPUS, GPUS, T_*
#   experiment definition, DO COMMIT (e.g. scripts/experiments/pair_A.env):
#       TRAJ, SEGMENTS, CONTEXTS, MESHES, SAVE_T_SCOPE, SAVE_T, SEED
#
# Commit the experiment file: it is what makes the corpus reproducible, and a
# gitignored .env would lose it.  Set SEED there explicitly -- left unset, plan
# draws fresh entropy, and the file then no longer describes what was built.
#
#   ENV_FILE=scripts/experiments/pair_A.env ./scripts/submit_isambard_pair.sh
#
# Note plan.json already records TRAJ/SEGMENTS/CONTEXTS/MESHES/save_t/seed once
# plan has run, and IT is the authority from then on: the generate and finish
# stages read the plan and need only OUT.  Changing the experiment variables
# after planning does not change an existing corpus (and plan refuses to
# overwrite without FORCE), so the stale values would just mislead.
#
# Resuming: re-run this script with the same OUT and FORCE unset.  plan refuses
# to overwrite, and each generate shard skips trajectories already complete on
# disk, so a time-limited or failed array can simply be submitted again.
set -euo pipefail

# Settings files are LAYERED, not replaced.  Every file assigns with
# `: "${VAR:=...}"`, so the first value seen for a variable wins, giving:
#
#     real environment  >  ENV_FILE (explicit)  >  LOCAL_ENV (implicit)
#
# ENV_FILE goes first because naming it on the command line is the more
# deliberate act; LOCAL_ENV then fills in whatever is still unset.  In practice
# the two barely overlap -- one holds the experiment, the other the machine.
# ENV_FILE may list several files, colon-separated, sourced left to right.
LOCAL_ENV="${LOCAL_ENV-scripts/isambard_pair.env}"
_sourced=""
_source_env() {
    local f="$1"
    case ":$_sourced:" in *":$f:"*) return 0 ;; esac   # never twice
    if [ -f "$f" ]; then
        echo "settings  : $f"
        set -a; . "$f"; set +a
        _sourced="$_sourced:$f"
    elif [ -n "${2:-}" ]; then
        echo "no settings file at $f" >&2; exit 1
    fi
}
if [ -n "${ENV_FILE:-}" ]; then
    _old_ifs=$IFS; IFS=:
    for _f in $ENV_FILE; do IFS=$_old_ifs; _source_env "$_f" required; IFS=:; done
    IFS=$_old_ifs
fi
if [ -n "$LOCAL_ENV" ]; then
    _source_env "$LOCAL_ENV" "${ENV_FILE_REQUIRED:-}"
fi

: "${OUT:?set OUT to the dataset root (env, or a settings file)}"
: "${TRAJ:?set TRAJ to the trajectories per arm (env, or a settings file)}"
CONFIG="${CONFIG:-fvm_gen/gen_alternating.json}"
NSHARDS="${NSHARDS:-9}"   # divide TRAJ; see isambard_pair.env.example
FVM_DEVICE="${FVM_DEVICE:-cpu}"
CPUS="${CPUS:-8}"
if [ "$FVM_DEVICE" = cpu ]; then GPUS="${GPUS:-0}"; else GPUS="${GPUS:-1}"; fi
SLURM_SCRIPT="scripts/isambard_pair.slurm"
export SOLVER_DIR="${SOLVER_DIR:-$(cd .. && pwd)/fvm_solver}"

[ -f "$CONFIG" ]       || { echo "no config at $CONFIG (run from the repo root?)" >&2; exit 1; }
[ -f "$SLURM_SCRIPT" ] || { echo "no $SLURM_SCRIPT (run from the repo root?)" >&2; exit 1; }
if [ "$NSHARDS" -gt "$TRAJ" ]; then
    echo "NSHARDS ($NSHARDS) > TRAJ ($TRAJ): some shards would get no work" >&2; exit 1
fi

# NB if-blocks, not `[ test ] && arr+=(...)`: that AND-list returns non-zero
# when the test is false, which under `set -e` kills the script.
# Pre-flight the venv here, not in 20 queued jobs.  requirements.txt has a
# git+https dependency (cprint, imported by the solver), compute nodes usually
# have no outbound internet, and `uv pip install -r` is one transaction -- so a
# venv built inside a job can end up with torch and no numpy, and every job then
# dies the same way hours later.
VENV="${VENV:-$SOLVER_DIR/.venv/$(uname -m)}"
if [ ! -f "$VENV/.installed" ] && [ -z "${SKIP_ENV_CHECK:-}" ]; then
    echo "No ready venv at $VENV" >&2
    echo >&2
    echo "Build it on this login node first (it needs outbound https):" >&2
    echo "  OUT=$OUT STAGE=env bash $SLURM_SCRIPT" >&2
    echo >&2
    echo "If the generate partition is a DIFFERENT architecture from this login" >&2
    echo "node (e.g. an x86 MACS partition from an aarch64 login node), build it" >&2
    echo "in a short interactive job on that partition instead, because the venv" >&2
    echo "path is per-\$(uname -m):" >&2
    echo "  srun --partition=<p> --pty env OUT=$OUT STAGE=env bash $SLURM_SCRIPT" >&2
    echo >&2
    echo "Then re-run this script. (SKIP_ENV_CHECK=1 submits anyway.)" >&2
    exit 1
fi
echo "venv      : $VENV"

COMMON=(--parsable)   # each job adds its own --export=ALL,STAGE=...
if [ -n "${ACCOUNT:-}" ];   then COMMON+=(--account="$ACCOUNT"); fi
if [ -n "${PARTITION:-}" ]; then COMMON+=(--partition="$PARTITION"); fi
GPUARG=()
if [ "$GPUS" -gt 0 ]; then GPUARG=(--gpus="$GPUS"); fi
ARRAY="0-$((NSHARDS - 1))"
if [ -n "${THROTTLE:-}" ]; then ARRAY="$ARRAY%$THROTTLE"; fi

# Exported to every stage via --export=ALL.
export OUT CONFIG TRAJ NSHARDS FVM_DEVICE
export SEGMENTS="${SEGMENTS:-4}" MESHES="${MESHES:-8}" SAVE_T_SCOPE="${SAVE_T_SCOPE:-fixed}"
export CONTEXTS="${CONTEXTS:-}" SAVE_T="${SAVE_T:-}" SEED="${SEED:-}" FORCE="${FORCE:-}"
export VENV

run() {           # run <description> <sbatch args...>
    local what="$1"; shift
    if [ -n "${DRY:-}" ]; then
        printf '  %-18s sbatch %s\n' "$what" "$*" >&2   # stderr: stdout is the job id
        echo "DRY-${what// /_}"; return
    fi
    sbatch "$@"
}

echo "pair root : $OUT"
echo "plan      : $TRAJ trajectories x $SEGMENTS segments, dt scope $SAVE_T_SCOPE"
echo "generate  : $NSHARDS shards per arm, device $FVM_DEVICE, ${GPUS} gpu(s)/task"
echo

JID_PLAN=$(run "plan" "${COMMON[@]}" --job-name=pair_plan \
    --cpus-per-task="$CPUS" --time="${T_PLAN:-02:00:00}" \
    --export=ALL,STAGE=plan "$SLURM_SCRIPT")
echo "  plan              $JID_PLAN"

JIDS=()
for arm in switching frozen; do
    jid=$(run "generate $arm" "${COMMON[@]}" --job-name="pair_$arm" \
        --cpus-per-task="$CPUS" "${GPUARG[@]}" --time="${T_GEN:-04:00:00}" \
        --array="$ARRAY" --dependency=afterok:"$JID_PLAN" \
        --export=ALL,STAGE=generate,ARM="$arm" "$SLURM_SCRIPT")
    echo "  generate $arm  $jid  (array $ARRAY, after plan)"
    JIDS+=("$jid")
done

# afterany, deliberately -- see the header.
DEP="afterany:${JIDS[0]}:${JIDS[1]}"
JID_FIN=$(run "finish" "${COMMON[@]}" --job-name=pair_finish \
    --cpus-per-task=2 --time="${T_FIN:-00:30:00}" \
    --dependency="$DEP" --export=ALL,STAGE=finish "$SLURM_SCRIPT")
echo "  finish            $JID_FIN  (after both arrays, afterany)"
echo
echo "Watch:   squeue -u \$USER"
echo "Result:  the finish job's exit code is the verdict — non-zero means the"
echo "         pair is NOT matched, so do not train on it. Read pair_finish_*.out."
