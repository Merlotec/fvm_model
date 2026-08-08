"""
Alternating-context rollout generator for the FVM solver.

One trajectory = one continuous physical rollout on one mesh, during which the hidden
physics context (viscosity model/params, thermal conductivity, gamma, ...) is RESAMPLED
every segment.  A segment is steps_per_segment SAVED FRAMES long (default 30), at a
per-segment frame interval save_t drawn from save_t_spec (default U(0.01, 0.2)) — so
segment sim-duration is steps_per_segment * save_t and both the context AND the frame
rate alternate.  save_t is recorded in each segment's params.json.  Each context segment
is written to its own run directory (params.json = that segment's context, frames
restart at t=0), so one trajectory produces a CHAIN of dataset directories, each fully
compatible with the existing mesh_<uid>/run_*/ structure — viewers and loaders need no
changes.

Continuity across a switch is enforced in PRIMITIVES (Vx, Vy, rho, T): the fields the
model observes carry no visible seam, only the dynamics change.  (Conserved energy may
step when C_v/gamma switch — that is the point: the context is hidden and must be
inferred from how the same-looking state evolves.)

The freestream/IC draw (ic_param_specs) is sampled ONCE per trajectory and held fixed
across all its segments — the boundary forcing is visible, so letting it switch would
leak the context switch.  Lineage (traj_id, segment_id, t_offset) lives in ic.json,
which system-grouping ignores, so params.json keeps exactly the grid-mode semantics:
context only.

    FVM_DEVICE=cuda python fvm_model/fvm_gen/gen_alternating.py fvm_model/fvm_gen/gen_alternating.json

    # plan the chains + write params.json/ic.json/manifest WITHOUT running the solver:
    python fvm_model/fvm_gen/gen_alternating.py fvm_model/fvm_gen/gen_alternating.json --dry-run
"""

import argparse
import json
import os
import pickle
import secrets
import sys
from dataclasses import dataclass, field
from typing import Any, Optional, Union

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import run_gen as RG   # reuse solver import, sampling, override plumbing


@dataclass
class AltConfig:
    problem: "Union[str, list]" = "ellipse"  # "ellipse" | "nozzle" | list to mix per mesh
    n_meshes: int = 8                 # distinct collider geometries
    trajs_per_mesh: int = 16          # alternating trajectories per geometry
    n_segments: int = 4               # context switches per trajectory (segments in the chain)
    seed: int = 42

    # A segment is steps_per_segment saved frames at a per-segment save_t drawn from
    # save_t_spec — the frame rate alternates along with the context, and the segment's
    # sim-duration is steps_per_segment * save_t (end_t lands exactly on the last frame
    # since it is a multiple of save_t by construction).
    steps_per_segment: int = 30
    save_t_spec: dict = field(default_factory=lambda: {
        "dist": "uniform", "low": 0.01, "high": 0.2})

    # Solver run controls (applied to every segment).
    n_iter: int = 200000              # per-segment iteration safety cap (stop is end_t;
                                      # 30 steps at save_t=0.2 is ~6 s sim time)
    print_i: int = 500
    compile: bool = True

    # Mesh sizing (obstacle/cell scale); passed straight to the config.
    min_A: Optional[float] = None
    max_A: Optional[float] = None
    lnscale: Optional[float] = None

    # Collider count is geometry, so it is drawn per mesh (inclusive range).
    min_colliders: int = 1
    max_colliders: int = 4

    # context = hidden physics, resampled EVERY SEGMENT (what alternates).
    # ic      = freestream/BC, sampled ONCE PER TRAJECTORY (held fixed).
    context_param_specs: dict[str, dict] = field(default_factory=dict)
    ic_param_specs: dict[str, dict] = field(default_factory=dict)
    phys_overrides: dict[str, Any] = field(default_factory=dict)

    output_subdir: str = "fvm_gen_alternating"


def _handoff_state(prims: torch.Tensor, phy) -> torch.Tensor:
    """Convert carried-over primitives to conserved state under NEW-segment physics.

    Continuity is enforced in primitives, not conserved variables: the observable
    fields must not jump at a context switch, otherwise the switch is visible in a
    single frame and nothing needs to be inferred.
    """
    V, rho, T = prims[:, :2], prims[:, 2:3], prims[:, 3:]
    momentum, rho, Q = phy.primatives_to_state(V, rho, T)
    return torch.cat([momentum, rho, Q], dim=-1)


def run_alternating(gen: AltConfig, dry_run: bool = False, out_dir: Optional[str] = None):
    if not gen.context_param_specs:
        raise SystemExit(
            "context_param_specs is empty: with nothing to resample there is no context "
            "to alternate — every segment would run identical physics."
        )
    if not gen.ic_param_specs:
        # Without IC sampling every trajectory starts from the config-default
        # freestream (v_n_inf=5.5 along +x) — the dataset would have a single
        # boundary condition, always left-to-right.  Silent and easy to miss.
        raise SystemExit(
            "ic_param_specs is empty: every trajectory would start from the identical "
            "default freestream (5.5 along +x).\n"
            "  Sample the starting BCs like the grid gen does, e.g.\n"
            '    "ic_param_specs": {"rho_inf": {...}, "T_inf": {...}, '
            '"v_n_inf": {...}, "aoa_deg": {...}}\n'
            "  (a fixed BC on purpose: use a degenerate spec like "
            '{"values": [5.5]}).'
        )
    if gen.n_segments < 2:
        raise SystemExit(f"n_segments={gen.n_segments}: an alternating trajectory needs "
                         f"at least 2 segments (one switch).")
    if gen.steps_per_segment < 1:
        raise SystemExit(f"steps_per_segment={gen.steps_per_segment}: must be >= 1.")

    rng = np.random.default_rng(gen.seed)
    out_root = os.path.abspath(out_dir) if out_dir else os.path.join(
        RG._DEFAULT_DATA_DIR, gen.output_subdir)
    problems = gen.problem if isinstance(gen.problem, list) else [gen.problem]

    device = mesh_over = None
    if not dry_run:
        RG._import_solver()
        np.random.seed(gen.seed)
        torch.manual_seed(gen.seed)
        device = os.environ.get("FVM_DEVICE")
        mesh_over = {k: v for k, v in {"min_A": gen.min_A, "max_A": gen.max_A,
                                       "lnscale": gen.lnscale, "device": device}.items()
                     if v is not None}

    os.makedirs(out_root, exist_ok=True)
    print(f"Output root: {out_root}   device: {device or '-'}   "
          f"problems: {problems}   dry_run: {dry_run}")

    manifest = {"problems": problems, "mode": "alternating",
                "n_meshes": gen.n_meshes, "trajs_per_mesh": gen.trajs_per_mesh,
                "n_segments": gen.n_segments,
                "steps_per_segment": gen.steps_per_segment,
                "save_t_spec": gen.save_t_spec, "runs": []}

    for m in range(gen.n_meshes):
        mesh_uid = secrets.token_hex(4)
        mesh_dir = out_root if gen.n_meshes == 1 else os.path.join(out_root, f"mesh_{mesh_uid}")

        # ---- geometry, with retries (same failure mode as run_gen) --------------
        base_cfg = mesh = edge_tag = bound_edgs = None
        problem, n_colliders = None, 0
        if not dry_run:
            for attempt in range(RG.MAX_MESH_TRIES):
                problem = str(rng.choice(problems))
                n_colliders = int(rng.integers(gen.min_colliders, gen.max_colliders + 1))
                mesh_cfg_over = dict(mesh_over or {})
                if problem == "ellipse":
                    mesh_cfg_over["n_colliders"] = n_colliders
                base_cfg = RG.apply_overrides(RG._make_base_cfg(problem), mesh_cfg_over)
                try:
                    Xs, tri_idx, all_edgs, bc_edge_mask, edge_tag, bound_edgs = \
                        RG._S["generate_mesh"](base_cfg)
                    mesh = RG._S["FVMMesh2D"](Xs, tri_idx, all_edgs, bc_edge_mask,
                                              device=base_cfg.device)
                    break
                except Exception as e:
                    print(f"    !! mesh gen failed (attempt {attempt + 1}/{RG.MAX_MESH_TRIES}, "
                          f"problem={problem}): {type(e).__name__}: {e}")
                    mesh = None
            if mesh is None:
                print(f"[mesh {m + 1}/{gen.n_meshes}] SKIPPED — all {RG.MAX_MESH_TRIES} "
                      f"mesh-gen attempts failed")
                continue
            os.makedirs(mesh_dir, exist_ok=True)
            print(f"[mesh {m + 1}/{gen.n_meshes}] problem={problem} — collider geometry "
                  f"({mesh_uid}, n_colliders={n_colliders})")
            with open(os.path.join(mesh_dir, "shared_mesh.pkl"), "wb") as f:
                pickle.dump({"mesh": mesh, "edge_tag": edge_tag, "bound_edgs": bound_edgs}, f)
        else:
            problem = str(rng.choice(problems))
            n_colliders = int(rng.integers(gen.min_colliders, gen.max_colliders + 1))
            os.makedirs(mesh_dir, exist_ok=True)
            print(f"[mesh {m + 1}/{gen.n_meshes}] problem={problem} ({mesh_uid})")

        # ---- trajectories -------------------------------------------------------
        for traj in range(gen.trajs_per_mesh):
            traj_uid = secrets.token_hex(4)
            ic_params = RG._sample_specs(gen.ic_param_specs, rng)
            # Draw the whole chain up front so a dry run plans exactly what a real
            # run executes.  Per segment: a context draw AND a frame-interval draw —
            # segment duration is steps_per_segment * save_t, a multiple of save_t by
            # construction, so exact_interval lands the last frame exactly on end_t.
            segments = []
            for seg in range(gen.n_segments):
                ctx = RG._sample_specs(gen.context_param_specs, rng)
                save_t = float(RG._sample_one("save_t", gen.save_t_spec, rng))
                segments.append((ctx, save_t, gen.steps_per_segment * save_t))

            print(f"  [mesh {m + 1}/{gen.n_meshes} | traj {traj + 1}/{gen.trajs_per_mesh}] "
                  f"({traj_uid}) save_t: "
                  + ", ".join(f"{st:.3f}" for _, st, _ in segments)
                  + "  durations: " + ", ".join(f"{d:.2f}s" for _, _, d in segments))

            prims_prev = None     # primitives carried across the context switch
            t_offset = 0.0
            for seg, (ctx, save_t, dur) in enumerate(segments):
                run_uid = secrets.token_hex(4)
                run_dir = os.path.join(mesh_dir, f"run_a{traj:03d}_s{seg:02d}_{run_uid}")
                os.makedirs(run_dir, exist_ok=True)

                # params.json: context (+ constants), same semantics as grid mode —
                # it defines the system this segment belongs to.  save_t lives here
                # because the frame spacing changes what one transition means: the
                # same physics at a different save_t is a different prediction task.
                record = {**ctx, "save_t": save_t, "problem": problem,
                          "mesh_uid": mesh_uid, "n_colliders": n_colliders}
                with open(os.path.join(run_dir, "params.json"), "w") as f:
                    json.dump(record, f, indent=2)
                # ic.json: shared freestream + chain lineage (grouping ignores this).
                with open(os.path.join(run_dir, "ic.json"), "w") as f:
                    json.dump({**ic_params, "traj_id": traj, "traj_uid": traj_uid,
                               "segment_id": seg, "t_offset": round(t_offset, 6),
                               "duration": round(dur, 6)}, f, indent=2)

                manifest["runs"].append({"mesh": mesh_uid, "problem": problem,
                                         "run": run_uid, "n_colliders": n_colliders,
                                         "traj_uid": traj_uid, "traj_id": traj,
                                         "segment_id": seg, "t_offset": round(t_offset, 6),
                                         "save_t": save_t, "duration": round(dur, 6),
                                         **ctx, **ic_params})
                t_offset += dur

                desc = ", ".join(f"{k}={v}" if isinstance(v, str) else f"{k}={v:.3g}"
                                 for k, v in ctx.items())
                print(f"    [seg {seg + 1}/{gen.n_segments}] save_t={save_t:.3f} "
                      f"({dur:.2f}s)  {desc}")
                if dry_run:
                    continue

                overrides = {**ctx, **ic_params, **gen.phys_overrides,
                             "plot": False, "exact_interval": True,
                             "save_t": save_t, "end_t": dur, "n_iter": gen.n_iter,
                             "print_i": gen.print_i, "compile": gen.compile,
                             "save_dir": run_dir}
                cfg = RG.apply_overrides(base_cfg, overrides)
                phy = RG._S["FluidConstitution2D"](cfg, dim=2)
                bc_tags, us_default = RG._init_conds(cfg, mesh, edge_tag, bound_edgs, phy)
                # Segment 0 starts from the standard (no-slip-consistent) IC; later
                # segments continue the rollout from the previous segment's final
                # fields, re-conserved under THIS segment's physics.
                us_init = us_default if prims_prev is None else _handoff_state(prims_prev, phy)

                solver = RG._S["FVMEquation"](cfg, phy, mesh, cfg.N_comp, bc_tags,
                                              us_init=us_init)
                try:
                    solver.solve()
                except Exception as e:
                    # No valid final state to continue from — abort this chain, keep
                    # the frames already written (they are valid up to the failure).
                    print(f"    !! segment failed ({type(e).__name__}: {e}) — "
                          f"aborting trajectory {traj_uid}")
                    break

                # Expected frame count: t=0 plus steps_per_segment saves.  Fewer
                # means the n_iter cap fired first; the chain timing would silently
                # drift, so stop the trajectory rather than continue mid-segment.
                n_expect = gen.steps_per_segment + 1
                n_have = len([f for f in os.listdir(run_dir)
                              if f.startswith("t_") and f.endswith(".npz")])
                if n_have < n_expect:
                    print(f"    !! segment stopped early ({n_have}/{n_expect} frames — "
                          f"n_iter cap?) — aborting trajectory {traj_uid}")
                    break

                prims_prev = solver.cells.get_values()[0].detach().clone()

    with open(os.path.join(out_root, "sweep_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nDone. {len(manifest['runs'])} segment dir(s) planned across "
          f"{gen.n_meshes} mesh(es). Manifest: {os.path.join(out_root, 'sweep_manifest.json')}")


def gen_from_file(path: str) -> AltConfig:
    with open(path) as f:
        data = json.load(f)
    data = {k: v for k, v in data.items() if not k.startswith("_")}
    return AltConfig(**data)


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Alternating-context FVM rollout generator: one continuous rollout, "
                    "context + frame interval (save_t) resampled every segment of "
                    "steps_per_segment saved frames, one run dir per context segment.")
    p.add_argument("config", help="path to a gen_alternating.json config")
    p.add_argument("--dry-run", action="store_true",
                   help="plan the chains + write params.json/ic.json/manifest without "
                        "running the solver")
    p.add_argument("--out-dir", default=None,
                   help="write the dataset under this directory instead of "
                        "data/<output_subdir>")
    args = p.parse_args()
    run_alternating(gen_from_file(args.config), dry_run=args.dry_run, out_dir=args.out_dir)
