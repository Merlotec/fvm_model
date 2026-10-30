"""
Automated report/data-generation pipeline: checkpoint -> rollouts -> figures.

For every configured rollout this script:
  1. runs hfm/infer.py (dynamics + optional refiner checkpoint) on ONE run dir,
     with the inference length inferred from the image request:
         n_predict = stride * (FRAMES_PER_IMAGE - 1)
     (the filmstrip's first column is rollout t=0, the last seed frame, so the
     last column t = stride*(N-1) is the furthest prediction needed);
  2. renders one filmstrip PNG per requested field — byte-identical to the web
     viewer's compare-mode "Frames" download (same code, fvm_viewer/strip_export):
     Ground truth / rollout rows plus the refiner-residual row when a refiner is given,
     columns t-indexed exactly like the eval logs;
  3. copies the per-step metrics CSV written by infer.py
     (t, MAE, MAE_refined, MAE_persist, loss, persist, ratio, ratio_refined);
  4. writes figure.tex — a directly \\input-able LaTeX figure with a pgfplots
     MAE-vs-t graph (base + refined + persistence baseline, read from the
     CSV) above the rollout image(s).

MODELS LAYER: the config lists models, each with its own checkpoints and a tag;
EVERY model runs EVERY rollout.  Outputs land in  <out_root>/<tag>_<run_id>/
(run_id = the run directory's basename; its uid keeps rollouts from colliding,
the tag keeps models apart):
    metrics.csv   <field>.png ...   figure.tex

Raw rollout frames are NOT kept by default.  Set "keep_frames": true to retain a
frames/ dir (standard viewer format — enables --skip-infer re-rendering and
interactive inspection:  python fvm_viewer/viewer.py <real_parent> -c .../frames).

EMBEDDING IN A PAPER: copy the output folder(s) next to your main .tex (or into
figures/), add to the preamble
    \\usepackage{pgfplots} \\usepackage{graphicx} \\pgfplotsset{compat=1.18}
then per figure
    \\input{<tag>_<run_id>/figure.tex}
If the folders live in a subdirectory, set once before the first \\input:
    \\def\\rolloutroot{figures/}

Config (JSON):
{
  "out_root":         "/path/report_out",
  "frames_per_image": 6,                            // GLOBAL: columns per filmstrip
  "refine_steps":     6,                            // default, overridable per model
  "infer_args":       ["--val-only"],               // optional extra infer.py args
  "models": [
    { "tag": "hfm37k",                              // folder prefix: <tag>_<run_id>
      "dynamics_ckpt": "/path/train_step037500.pt",
      "refiner_ckpt":  "/path/refiner_a.pt" },      // optional; omit -> no refined row/cols
    { "tag": "hfm27k-noref",
      "dynamics_ckpt": "/path/train_step027000.pt" }
  ],
  "rollouts": [
    { "run_dir": "/path/data/mesh_x/run_a000_s01_ab12cd34",
      "fields":  ["rho", "Vx"],                     // one image per field ("rho" ok too)
      "stride":  2,                                 // shared by all fields of this rollout
      "include_context": true,                      // per-rollout (falls back to the
                          //   top-level value if set there): prepend the model's context
                          //   frames above the Ground truth row, cut off by a heavy rule.
      "include_refiner_detail": false,              // per-rollout, same fallback: show
                          //   the refiner-residual row.  Default true, i.e. shown
                          //   whenever a refiner actually ran; no effect without one.
      "include_energy": true,                       // per-rollout, same fallback: write
                          //   energy.csv (mean kinetic/internal/total energy density per
                          //   rollout step, ground truth vs rollout vs refined) and add an
                          //   energy-vs-step panel beside the MAE graph.  Built for the
                          //   energy-dip probe datasets (fvm_gen_tests/gen_energy_dip.py:
                          //   seed frame artificially dipped, context untouched -- recovery
                          //   toward the GT level shows the context encoder carries the
                          //   macro-state), but works on any rollout as a drift check.
                          //   Dip runs are detected via params.json and get --seq-start 0
                          //   automatically so the dipped frame is the rollout seed.
      "include_mae": false,                         // per-rollout, same fallback: DEFAULT
                          //   TRUE.  Set false to drop the MAE-vs-step graph (e.g. an
                          //   energy-only figure).  With both panels on they sit side by
                          //   side; with one, it takes the full width; with neither, the
                          //   figure is the filmstrip alone.  metrics.csv is still written.
      "include_refiner_mae": true,                  // per-rollout, same fallback: add each
                          //   model's REFINED MAE to the combined figure's MAE axis, on the
                          //   same graph as its base curve -- same colour and dash (colour
                          //   still means "model"), hollow markers and a lighter stroke for
                          //   the refined one.  Needs a refiner on EVERY model, else the
                          //   axis keeps base curves only and says so.  Default false.
                          //   The PER-MODEL figure always plots base and refined together
                          //   and needs no flag.  (separate_refiner_mae is accepted as an
                          //   older name for this key.)
      "include_base_rollout": true,                 // per-rollout, same fallback: run the
                          //   model a SECOND time with the refiner not loaded at all, and
                          //   add that full rollout as an extra filmstrip row ("... , no
                          //   refiner").  The control for refine_feedback: it shows what
                          //   the base model does on its own, from its own predictions.
                          //   Only runs WITH refine_feedback -- without it the refiner is
                          //   output-only, so the existing rollout row already is the
                          //   un-refined prediction and a second pass would duplicate it
                          //   (a note says so).  Default false.  With "keep_frames" the
                          //   control frames land in frames_norefine/.
      "individual_rollouts": true,                  // per-rollout, same fallback:
                          //   ALSO write a folder per model (<tag>_<run_id>/ with its
                          //   own images, figure.tex and metrics.csv).  DEFAULT FALSE --
                          //   only the single combined <run_id>/ figure is kept, since
                          //   that is the usual deliverable and one folder per model per
                          //   rollout is a lot of files nobody opens.  Inference still
                          //   runs per model either way (the combined figure stacks
                          //   their rows); the folders are just removed once it has
                          //   taken the CSVs it needs.  Forced ON when there is no
                          //   combined figure to fall back on: with comparison_context
                          //   set, or with untagged models (where <tag>_<run_id> and
                          //   <run_id> are the same path).  NOTE "keep_frames" output
                          //   lives in those folders and goes with them.
      "comparison_context": "/path/data/fvm_val_carreau",  // CONTEXT COMPARISON.
                          //   Path to a MIRROR corpus (fvm_gen_tests/gen_mirror.py:
                          //   same meshes, same ICs, same run names, one physics field
                          //   changed).  The model is run TWICE from the same seed
                          //   frame -- once on this trajectory's own history, once on
                          //   the mirror's -- and the figure becomes:
                          //     context (true) / context (mirror)   [above the rule]
                          //     Ground truth
                          //     rollout, true context
                          //     rollout, mirror context
                          //   with BOTH MAE curves on the graph (metrics.csv and
                          //   metrics_ctxswap.csv).  It answers "is the context
                          //   steering the prediction?": identical rows mean it is not.
                          //   The rows are already spent on one model's true-vs-mirror
                          //   contrast, so the cross-model COMBINED figure is skipped
                          //   and you get one figure per model.  Per-rollout only.
                          //   Needs a real inference pass -> skipped under --skip-infer.
                          //   (context_swap is accepted as an older name.)
      "refine_feedback": false,                     // CLOSE THE LOOP: the refined frame
                          //   becomes the autoregressive state instead of the base
                          //   prediction (infer.py --refine-feedback).  DEFAULT TRUE --
                          //   set false only to run the open-loop rollout.  Precedence
                          //   rollout > model > top level, so a single model can opt out.
                          //   No effect without a refiner_ckpt.  Being the default, it is
                          //   NOT mentioned in the caption; include_base_rollout adds the
                          //   open-loop control as an extra filmstrip row instead.
      "postfix": "s2",                              // optional: appended to the output
                          //   folder (<tag>_<run_id>_s2 / <run_id>_s2) so the SAME run
                          //   can appear several times -- different stride, fields or
                          //   infer args -- without overwriting itself.  Lowercased.
      "description": "a shear-thinning flow past three colliders" }
                          // ^ optional: replaces the run id as the caption's
                          //   subject ("Model comparison on <description>: ...").
                          //   Inserted VERBATIM, so LaTeX works ($\\mu$, \\emph{});
                          //   escape any literal _ & % # yourself.
  ]
}

A config WITHOUT "models" (checkpoints at the top level) still works and behaves
as one untagged model with unprefixed output folders.

Usage:
    python scripts/report_pipeline.py report.json
    python scripts/report_pipeline.py report.json --skip-infer   # re-render from existing frames/
"""

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

import numpy as np

_HERE = Path(__file__).resolve().parent
_HFM_DIR = _HERE.parents[1] / 'hfm'
sys.path.insert(0, str(_HERE.parent / 'fvm_viewer'))

from strip_export import (FIELD_NAMES, field_tex, build_renderer,
                          find_timestep_files, load_step, load_gen_frame,
                          load_gen_refined, closest_idx, _strip_indices,
                          _strip_png_bytes, fluid_mask)


def _refine_feedback_set(cfg: dict, model: dict, roll_fb: Optional[bool]) -> bool:
    """Precedence for refine_feedback: rollout > model > top level.

    Says nothing about whether a refiner exists to feed back — callers gate on
    that themselves, so the "set but no refiner" case can still be warned about
    rather than silently resolving to False.
    """
    if roll_fb is not None:
        return bool(roll_fb)
    return bool(model.get('refine_feedback', cfg.get('refine_feedback', True)))


def run_inference(cfg: dict, model: dict, run_dir: Path, n_predict: int,
                  work_dir: Path, refine_feedback: Optional[bool] = None,
                  no_refiner: bool = False,
                  context_run: Optional[str] = None) -> Path:
    """Invoke hfm/infer.py for one (model, run); returns the dir with its outputs.

    Per-model keys (dynamics_ckpt, refiner_ckpt, refine_steps, refine_feedback,
    infer_args) override the same keys at the top level of the config.

    refine_feedback: pass a bool to force it (the per-rollout setting, which wins
    over both), or None to resolve it model -> top-level as with every other key.
    """
    def opt(key, default=None):
        return model.get(key, cfg.get(key, default))

    hfm_dir = Path(opt('hfm_dir', _HFM_DIR)).resolve()
    mesh_dir = run_dir.parent          # dir holding shared_mesh.pkl + the run
    # Paths are passed ABSOLUTE: infer.py runs with cwd=hfm_dir, so a relative
    # out-dir/checkpoint from the config would silently resolve under the hfm
    # repo while this script looks for the results relative to ITS cwd.
    cmd = [sys.executable, str(hfm_dir / 'infer.py'),
           '--checkpoint', str(Path(opt('dynamics_ckpt')).resolve()),
           '--data-dir',   str(mesh_dir.resolve()),
           '--out-dir',    str(work_dir.resolve()),
           '--n-predict',  str(n_predict),
           '--runs',       run_dir.name,
           '--no-images']
    fb = _refine_feedback_set(cfg, model, refine_feedback)
    if no_refiner:
        # Control rollout: the refiner is not loaded at all, so the AR state is
        # the base model's own output at every step.  Only distinguishable from
        # the normal pass when refine_feedback is on -- otherwise the refiner
        # never touches the loop and the two runs coincide.
        fb = False
    elif opt('refiner_ckpt'):
        cmd += ['--refine', str(Path(opt('refiner_ckpt')).resolve()),
                '--refine-steps', str(opt('refine_steps', 6))]
        # Closes the loop: the REFINED frame becomes the autoregressive state
        # instead of the base prediction.  Off by default in infer.py because the
        # base was never trained on refined inputs, so this is a genuinely
        # different rollout, not a display option — hence it belongs in the
        # caption too (see write_figure_tex/write_combined_tex).
        if fb:
            cmd += ['--refine-feedback']
    elif fb:
        print('  [warn] refine_feedback is set but this model has no '
              '"refiner_ckpt" — nothing to feed back, ignoring.')
    if context_run:
        cmd += ['--context-run', str(Path(context_run).resolve())]
    extra = list(opt('infer_args', []))
    # Energy-dip probe runs (fvm_gen_tests/gen_energy_dip.py) perturb ONE frame,
    # placed so that it is the AR seed when the window starts at frame 0.  Without
    # this, infer.py defaults to a mid-run window and the dipped frame would sit
    # unused in the untouched early frames.  An explicit --seq-start in infer_args
    # wins.
    dip = load_energy_dip(run_dir)
    if dip is not None and '--seq-start' not in extra:
        cmd += ['--seq-start', '0']
        print(f'  [energy-dip] {run_dir.name} carries an artificial dip at frame '
              f'{dip.get("frame_index")} ({dip.get("frame")}): pinning '
              f'--seq-start 0 so that frame is the rollout seed')
        if '--val-only' in extra:
            print('  [warn] --val-only may filter out this synthetic run '
                  '(its name hashes to its own split bucket)')
    cmd += extra
    print(f'  $ {" ".join(cmd)}')
    subprocess.run(cmd, check=True, cwd=str(hfm_dir))
    return work_dir


def _context_row(gen_files: list, n_seed: int, fi: int, n_cols: int):
    """The model's input frames for one field: chronological, RIGHT-aligned.

    Seeds are the first n_seed frames and the LAST of them is rollout t=0 (already
    column 0 of the rollout), so the context proper is seeds[:-1].  Time increases
    left to right and the NEWEST frame sits in the rightmost slot, so it lands
    nearest the rollout it precedes; when there are more context frames than
    columns the OLDEST (leftmost) are dropped, and when there are fewer the spare
    slots on the left stay blank.
    """
    ctx = list(range(max(n_seed - 1, 0)))[-n_cols:]          # chronological, newest last
    row = [load_gen_frame(gen_files[i])[1][fi] for i in ctx]
    return [None] * (n_cols - len(row)) + row


def _rollout_label(tag: str = '', no_refiner: bool = False,
                   note: str = '') -> str:
    """Row label for a model's rollout row: "Model (tag); qualifier".

    Shared by the per-model and the combined filmstrip so the two figures can
    never label the same row differently.  Every row on the strip is a rollout,
    so saying so in each label only crowds out the part that differs; the tag and
    the qualifier are the whole content.  Without a tag it degrades to "Model".
    """
    base = f'Model ({tag})' if tag else 'Model'
    for q in (['no refiner'] if no_refiner else []) + ([note] if note else []):
        base += f'; {q}'
    return base


def render_strips(run_dir: Path, frames_dir: Path, fields: list, stride: int,
                  n_frames: int, out_dir: Path, include_context: bool = False,
                  include_refiner_detail: bool = True, tag: str = '',
                  base_frames_dir: Optional[Path] = None,
                  swap_frames_dir: Optional[Path] = None,
                  swap_label: str = '', write_images: bool = True):
    # swap_frames_dir is the SAME model on the SAME seed frame, conditioned on
    # another corpus's history.  Its own seed frames are the context the encoder
    # actually saw (infer.py writes ctx_frames, not frames_gt), so they render as
    # the second context row -- the comparison is only readable with both on
    # screen.
    """One filmstrip per field.  Returns (col_dt, rows_by_field, step_ids, mask).

    Same renderer as the web viewer's compare download, with ONE difference: the
    columns are numbered 0,1,2,... consecutively rather than by rollout step, since
    the figure caption carries the spacing (the returned col_dt) instead.

    The rows are handed back so the caller can stack every model's rollout row
    under one shared Ground truth row for the combined figure, without re-rendering.
    """
    gen_files = find_timestep_files(str(frames_dir))
    real_files = find_timestep_files(str(run_dir))
    # Optional second rollout of the SAME model with the refiner switched off
    # entirely.  Both runs share the seed frames and the prediction count, so the
    # frame lists line up index for index; if they somehow do not, drop the row
    # rather than render columns that are silently off by one.
    base_files = find_timestep_files(str(base_frames_dir)) if base_frames_dir else []
    if base_files and len(base_files) != len(gen_files):
        print(f'  [warn] no-refiner rollout has {len(base_files)} frames vs '
              f'{len(gen_files)}; dropping that row')
        base_files = []
    # Context-swap rollout: same seed frame and same geometry, conditioned on a
    # different run's history.  Same index alignment requirement as above.
    swap_files = find_timestep_files(str(swap_frames_dir)) if swap_frames_dir else []
    if swap_files and len(swap_files) != len(gen_files):
        print(f'  [warn] context-swap rollout has {len(swap_files)} frames vs '
              f'{len(gen_files)}; dropping that row')
        swap_files = []
    n_seed = 0
    for f in gen_files:                      # seeds are always at the front
        if load_gen_frame(f)[2]:
            n_seed += 1
        else:
            break
    t_off = max(n_seed - 1, 0)

    # renderer at the generated resolution, same as the compare viewer
    _, g0, _ = load_gen_frame(gen_files[0])
    renderer = build_renderer(str(run_dir), (g0.shape[1], g0.shape[2]))
    # Fluid mask straight from the renderer's interior index — the same source
    # hfm's build_pixel_mask uses.  Deriving it from the values instead (nonzero,
    # or finite) would misclassify any fluid cell that is legitimately 0, which
    # velocity components hit routinely.
    value_mask = fluid_mask(renderer, g0.shape)

    idxs = _strip_indices(t_off, stride, n_frames, len(gen_files))
    # Real rollout step of each column (0, stride, 2*stride, ...), not 0,1,2,...
    step_ids = [i - t_off for i in idxs]
    # The model's OWN timestep: the gap between CONSECUTIVE rollout frames, which
    # is what one forward pass advances.  Not the column spacing (stride x this),
    # which is only a display choice.
    all_ts = [load_gen_frame(f)[0] for f in gen_files]
    model_dt = None
    if len(all_ts) > t_off + 2:
        model_dt = all_ts[t_off + 2] - all_ts[t_off + 1]
    elif len(all_ts) > 1:
        model_dt = all_ts[-1] - all_ts[-2]
    col_dt = model_dt
    if len(idxs) < n_frames:
        print(f'  [note] only {len(idxs)}/{n_frames} columns fit '
              f'({len(gen_files)} frames at stride {stride})')

    n_ctx_avail = max(n_seed - 1, 0)
    written, rows_by_field = [], {}
    for field in fields:
        fi = FIELD_NAMES.index(field)
        real_row, gen_row, detail_row = [], [], []
        base_row = [] if base_files else None
        swap_row = [] if swap_files else None
        any_refined = False
        for i in idxs:
            t, gg, _ = load_gen_frame(gen_files[i])
            gen_row.append(gg[fi])
            _, rcp = load_step(real_files[closest_idx(real_files, t)])
            real_row.append(renderer.render_cell_smooth(rcp).numpy()[fi])
            rg = load_gen_refined(gen_files[i])
            detail_row.append(rg[fi] - gg[fi] if rg is not None else None)
            any_refined = any_refined or rg is not None
            if base_row is not None:
                base_row.append(load_gen_frame(base_files[i])[1][fi])
            if swap_row is not None:
                swap_row.append(load_gen_frame(swap_files[i])[1][fi])

        kwargs = {}
        if any_refined and include_refiner_detail:
            present = [d for d in detail_row if d is not None]
            # Same masking rationale as the main range: a residual is only
            # meaningful where there is fluid.
            def _amax(d):
                v = d[value_mask] if (value_mask is not None
                                      and value_mask.shape == d.shape) else d
                return float(np.abs(v).max()) if v.size else 0.0
            kwargs = dict(detail_row=detail_row,
                          detail_maxabs=max((_amax(d) for d in present),
                                            default=1.0) or 1.0)
        if include_context or swap_files:
            n_col = len(step_ids)
            if swap_files:
                # Both histories, in the order the caption reads them.
                kwargs['context_rows'] = [
                    ('Context encoder input; true (this trajectory)',
                     _context_row(gen_files, n_seed, fi, n_col)),
                    (f'Context encoder input; mirror ({swap_label})'
                     if swap_label else 'Context encoder input; mirror',
                     _context_row(swap_files, n_seed, fi, n_col)),
                ]
            else:
                kwargs['context_row'] = _context_row(gen_files, n_seed, fi, n_col)
                # Terse on the image; the explanation lives in the figure caption.
                kwargs['context_label'] = 'Context encoder input (sequential frames)'
        strip = [real_row, gen_row]
        labels = ['Ground truth', _rollout_label(tag)]
        if base_row is not None:
            strip.append(base_row)
            labels.append(_rollout_label(tag, no_refiner=True))
        if swap_row is not None:
            # Relabel the pair so the contrast is the row labels, not a footnote.
            labels[1] = _rollout_label(tag, note='true context')
            strip.append(swap_row)
            labels.append(_rollout_label(tag, note='mirror context'))
        # The rows are always built (the combined figure stacks them); the PNG is
        # only encoded when this model gets its own figure.
        if write_images:
            png = _strip_png_bytes(strip, field, step_ids, row_labels=labels,
                                   value_mask=value_mask, **kwargs)
            out = out_dir / f'{field}.png'
            out.write_bytes(png)
            written.append(out)
            print(f'  image → {out}')
        rows_by_field[field] = (real_row, gen_row, base_row, swap_row,
                                kwargs.get('context_row'),
                                kwargs.get('context_label', ''))
    if swap_files:
        print('  [context comparison] 2 context rows + true/mirror rollout rows')
    return col_dt, rows_by_field, step_ids, value_mask


def _grid_energies(grid: np.ndarray, cv: float,
                   mask: np.ndarray) -> "tuple[float, float]":
    """(mean kinetic, mean internal) energy density of one [4, H, W] physical grid
    over the fluid `mask`.  Channels follow FIELD_NAMES: Vx, Vy, rho, T."""
    vx, vy, rho, T = grid[0], grid[1], grid[2], grid[3]
    ke = 0.5 * rho * (vx * vx + vy * vy)
    ie = rho * cv * T
    return float(ke[mask].mean()), float(ie[mask].mean())


def write_energy_csv(run_dir: Path, frames_dir: Path,
                     out_dir: Path) -> "Optional[dict]":
    """energy.csv: spatial mean energy densities per rollout step, GT vs rollout.

    Covers EVERY frame infer.py wrote (not just the filmstrip columns): the seed
    frames land at steps <= 0 (the context, where rollout == GT by construction)
    and the predictions at steps >= 1.  This is the drift instrument: with an
    energy-dip run the plot shows the context level (negative steps), the dip
    (step 0), and whether the rollout recovers to the GT level or keeps the dip.

    Columns: step, sim_t, E_real, E_gen, E_refined, KE_real, KE_gen, IE_real,
    IE_gen.  E = KE + IE with KE = 0.5*rho*|v|^2 and IE = rho*C_v*T (C_v from
    params.json; 1.0 when absent, which only rescales the internal term).
    E_refined is nan for frames without refiner output (seeds, or no refiner);
    the pgfplots axis uses `unbounded coords=jump` so those points are skipped.

    Returns {'refined': bool} or None when there is nothing to measure.
    """
    gen_files = find_timestep_files(str(frames_dir))
    real_files = find_timestep_files(str(run_dir))
    if not gen_files or not real_files:
        return None
    cv = load_fluid_params(run_dir).get('C_v')
    if not isinstance(cv, (int, float)) or not cv > 0:
        cv = 1.0
    _, g0, _ = load_gen_frame(gen_files[0])
    renderer = build_renderer(str(run_dir), (g0.shape[1], g0.shape[2]))
    n_seed = 0
    for f in gen_files:                      # seeds are always at the front
        if load_gen_frame(f)[2]:
            n_seed += 1
        else:
            break
    t_off = max(n_seed - 1, 0)

    lines = ['step,sim_t,E_real,E_gen,E_refined,KE_real,KE_gen,IE_real,IE_gen']
    any_refined = False
    mask = None
    for i, gf in enumerate(gen_files):
        t, grid, _ = load_gen_frame(gf)
        _, rcp = load_step(real_files[closest_idx(real_files, t)])
        real = renderer.render_cell_smooth(rcp).numpy()
        if mask is None:
            # Fluid mask from the rendered GT (NaN outside the mesh/colliders);
            # applied to BOTH grids so the means average the same pixels.
            mask = np.isfinite(real[0])
        ke_r, ie_r = _grid_energies(real, cv, mask)
        ke_g, ie_g = _grid_energies(grid, cv, mask)
        rg = load_gen_refined(gf)
        if rg is not None:
            ke_f, ie_f = _grid_energies(rg, cv, mask)
            e_ref = f'{ke_f + ie_f:.8g}'
            any_refined = True
        else:
            e_ref = 'nan'
        lines.append(f'{i - t_off},{t:.6g},{ke_r + ie_r:.8g},{ke_g + ie_g:.8g},'
                     f'{e_ref},{ke_r:.8g},{ke_g:.8g},{ie_r:.8g},{ie_g:.8g}')
    out = out_dir / 'energy.csv'
    out.write_text('\n'.join(lines) + '\n')
    print(f'  energy → {out}')
    return {'refined': any_refined}


# Bookkeeping keys in params.json / ic.json that are not fluid parameters.
# energy_dip is fvm_gen_tests/gen_energy_dip.py's perturbation record: it drives
# --seq-start pinning and the dip caption, and must not leak into the
# fluid-parameter list.
# save_t is excluded because the caption already opens with the model's timestep,
# measured from the generated frames as col_dt -- the same quantity.  Listing it
# again under "Fluid parameters" printed $\Delta t$ twice in one caption, and it
# is a sampling parameter rather than a property of the fluid anyway.
_NON_FLUID_KEYS = {'problem', 'mesh_uid', 'n_colliders', 'context_id', 'ic_id',
                   'traj_id', 'traj_uid', 'segment_id', 't_offset', 'duration',
                   'energy_dip', 'save_t'}


def load_energy_dip(run_dir: Path) -> Optional[dict]:
    """The run's artificial-perturbation record (gen_energy_dip.py), or None."""
    p = run_dir / 'params.json'
    try:
        dip = json.loads(p.read_text()).get('energy_dip')
        return dip if isinstance(dip, dict) else None
    except Exception:
        return None

# LaTeX (math-mode) symbol per parameter, in caption order; unknown keys fall
# back to \texttt{key}.
_PARAM_TEX = [
    ('visc_model',       None),                      # textual, handled specially
    ('visc_n',           r'n'),
    ('visc_min_factor',  r'f_{\min}'),
    ('visc_gamma_scale', r'\dot\gamma_0'),
    ('viscosity',        r'\mu'),
    ('visc_bulk',        r'\mu_b'),
    ('thermal_cond',     r'\kappa'),
    ('C_v',              r'C_v'),
    ('gamma',            r'\gamma'),
    ('rho_inf',          r'\rho_\infty'),
    ('T_inf',            r'T_\infty'),
    ('v_n_inf',          r'|v_\infty|'),
    ('aoa_deg',          r'\alpha_\infty'),
]

# Math-mode unit suffix appended to the value, per parameter.  Only for
# quantities whose number is meaningless without it; everything else is in
# solver-normalised units and stays bare.
_PARAM_UNIT = {'aoa_deg': r'^\circ'}


def _tex_num(v) -> str:
    """3-sig-fig number for a caption, with e-notation as \\times 10^{k}."""
    s = f'{v:.3g}' if isinstance(v, float) else str(v)
    if 'e' in s:
        mant, exp = s.split('e')
        return rf'{mant} \times 10^{{{int(exp)}}}'
    return s


def load_fluid_params(run_dir: Path) -> dict:
    """The run's fluid parameters: params.json merged with ic.json (grid/alternating
    datasets keep the freestream there), minus bookkeeping keys."""
    params = {}
    for name in ('params.json', 'ic.json'):
        p = run_dir / name
        if p.exists():
            params.update(json.loads(p.read_text()))
    return {k: v for k, v in params.items() if k not in _NON_FLUID_KEYS}


def _fluid_caption(params: dict) -> str:
    """'Fluid parameters: Carreau ($n = 1.4$, ...); $\\mu = ...$; ...' or ''. """
    if not params:
        return ''
    params = dict(params)
    model = params.pop('visc_model', None)
    if model == 'Newtonian':
        # shape parameters are ignored by the solver for a Newtonian draw
        for k in ('visc_n', 'visc_min_factor', 'visc_gamma_scale'):
            params.pop(k, None)
    parts = []
    if model is not None:
        shape = [rf'${_PARAM_TEX[i][1]} = {_tex_num(params.pop(key))}$'
                 for i, key in ((1, 'visc_n'), (2, 'visc_min_factor'),
                                (3, 'visc_gamma_scale')) if key in params]
        parts.append(model + (f' ({", ".join(shape)})' if shape else ''))
    order = {k: i for i, (k, _) in enumerate(_PARAM_TEX)}
    sym = dict(_PARAM_TEX)
    for k in sorted(params, key=lambda k: order.get(k, len(order))):
        label = sym.get(k) or rf'\texttt{{{k.replace("_", chr(92) + "_")}}}'
        parts.append(rf'${label} = {_tex_num(params[k])}{_PARAM_UNIT.get(k, "")}$')
    return ' Fluid parameters: ' + '; '.join(parts) + '.'


def _dip_caption(dip: "Optional[dict]") -> str:
    """Caption clause for an energy-dip probe run (gen_energy_dip.py), or ''.

    Kept (tersely) whenever the metadata is present, with or without the energy
    panel: the ground-truth row's step-0 column IS the perturbed frame, so
    dropping this would leave a figure that misrepresents its own data.
    """
    if not dip:
        return ''
    what = {'total': 'total', 'kinetic': 'kinetic',
            'thermal': 'internal'}.get(dip.get('mode', 'total'),
                                       str(dip.get('mode')))
    try:
        factor = rf'${_tex_num(float(dip["factor"]))}\times$'
    except (KeyError, TypeError, ValueError):
        factor = 'a fraction of'
    return (f' Seed frame (step 0) artificially scaled to {factor} its true '
            f'{what} energy; context and ground truth unmodified.')


# Energy-vs-step caption clause, shared by both figure writers.  Plain-LaTeX
# maths only (\frac, |...|): figure.tex must compile without amsmath.
_ENERGY_CAPTION = (r' Energy panel: mean density $\langle \rho C_v T + '
                   r'\frac{1}{2}\rho|v|^2 \rangle$.')


# Side-by-side panels sit flush: \hfill would push them to opposite margins and
# spend the whole slack as a gap down the middle.  The trailing % eats the
# newline so the only separation left is each axis's own label margin.  pgfplots'
# `width` already INCLUDES the labels and ticks, so two 0.49\linewidth panels
# really do fill the line.
# A thin \hspace, not \hfill: \hfill would spend all the leftover line width as
# one gap down the middle.  The trailing % eats the newline so the only spacing is
# the one stated here.
_PANEL_JOIN = '\\hspace{8pt}%\n'
_PANEL_W2 = r'0.47\linewidth'      # each, when two panels share the line
_PANEL_W1 = r'0.8\linewidth'       # a single panel
# ONE height for every panel.  pgfplots' `height` covers the axis plus its labels,
# so equal values only give equal plot boxes while the label furniture matches --
# which it does here (both axes carry one xlabel line and one rotated ylabel).
# Keeping it a single constant is what stops the two drifting apart again: the
# combined figure used to ask for 4.8cm while the energy axis stayed at 4.6.
_PANEL_H = '4.6cm'

def _mae_axis(width: str, plots: str, height: str = _PANEL_H) -> str:
    """One MAE-vs-step axis.  Base and refined curves always share it: they are
    the same quantity and separate axes autoscale independently, which hides the
    very difference the pair is there to show."""
    return rf"""\begin{{axis}}[
    width={width}, height={height},
    xlabel={{rollout step $t$}}, ylabel={{MAE}},
    grid=major, grid style={{black!12}},
    axis line style={{black!60}},
    tick label style={{font=\small}}, label style={{font=\small}},
    legend style={{font=\small, draw=black!30, at={{(0.03,0.97)}}, anchor=north west}},
    legend cell align=left, ymin=0,
]
{plots}\end{{axis}}"""


def _energy_axis(width: str, plots: str, labels_right: bool = False) -> str:
    """One pgfplots axis reading energy CSVs.  `unbounded coords=jump` skips the
    nan rows (E_refined on seed frames / refiner-less runs).

    labels_right: this axis sits to the RIGHT of the MAE axis, so its ylabel and
    tick labels go on its outer edge.  Left-hand labels would otherwise land in
    the space between the two plot boxes and reopen the gap that butting the
    pictures together just closed.
    """
    side = ('\n    yticklabel pos=right, ylabel near ticks,'
            if labels_right else '')
    return rf"""\begin{{axis}}[
    width={width}, height={_PANEL_H},
    xlabel={{rollout step $t$}}, ylabel={{mean energy density}},{side}
    grid=major, grid style={{black!12}},
    axis line style={{black!60}},
    tick label style={{font=\small}}, label style={{font=\small}},
    legend style={{font=\small, draw=black!30, at={{(0.03,0.03)}}, anchor=south west}},
    legend cell align=left, unbounded coords=jump,
]
{plots}\end{{axis}}"""


# Categorical colours for the multi-model plot.  Validated (dataviz
# validate_palette.js, light surface): lightness band, chroma floor, normal-vision
# separation and contrast all PASS; worst adjacent CVD pair sits at dE 6.3
# (protan), which is only legal WITH secondary encoding -- hence the per-model
# dash pattern + marker below, so the lines stay separable in greyscale and for
# colour-blind readers, not by hue alone.
_MODEL_COLOURS = ['2563EB', 'E8710A', '15803D', '7C3AED', 'B91C1C']
_MODEL_DASH = ['solid', 'dashed', 'dotted', 'dashdotted', 'densely dashed']
_MODEL_MARK = ['*', 'square*', 'triangle*', 'diamond*', 'pentagon*']


def write_combined_tex(out_dir: Path, run_id: str, fields: list, tags: list,
                       col_dt: Optional[float] = None,
                       fluid_params: Optional[dict] = None,
                       has_persist: bool = False, desc: Optional[str] = None,
                       has_energy: bool = False,
                       energy_dip: Optional[dict] = None, has_mae: bool = True,
                       has_refined: bool = False,
                       separate_refiner: bool = False,
                       has_base_rollout: bool = False):
    """Combined figure: every model's MAE curve on one axis, above a filmstrip
    whose rows are Ground truth followed by one row per model.

    Reads each model's metrics_<tag>.csv, copied into this folder so it is
    self-contained and can be dropped into a paper on its own.  The filmstrip
    never carries refiner residual rows -- this figure compares MODELS -- and
    the MAE axis plots base MAE only, UNLESS separate_refiner adds each model's
    refined curve to the SAME axis.
    """
    folder = out_dir.name
    # The rollout's "description" replaces the raw run id in the caption.  It is
    # inserted VERBATIM so it can carry LaTeX (math, emphasis); escape any literal
    # _ & % # yourself.  Without one we fall back to the run id in \texttt.
    subject = desc if desc else rf'run \texttt{{{run_id.replace("_", chr(92) + "_")}}}'
    # The image is self-describing — labelled rows, a labelled context block above
    # a heavy rule, a shared colour bar, a plot legend — so the caption states
    # only what the reader CANNOT see: the subject, the timestep, the physics, and
    # any way the data itself was altered.
    # ...and the extra rollout row from a refiner-free pass, which is otherwise
    # indistinguishable from a duplicated row.
    base_txt = (' (each model with and without refiner active)'
                if has_base_rollout else '')
    energy_txt = _ENERGY_CAPTION if has_energy else ''
    dip_txt = _dip_caption(energy_dip)
    fluid_txt = _fluid_caption(fluid_params or {})
    fields_txt = ', '.join(field_tex(f) for f in fields)
    # Delta t is the MODEL's step, not the gap between displayed columns.
    dt_txt = (rf', $\Delta t = {_tex_num(col_dt)}$' if col_dt else '')

    show_refined = bool(has_mae and has_refined and separate_refiner)
    colours = []

    def _mae_body() -> str:
        """Every model's MAE curve on ONE axis, plus the persistence baseline.

        With show_refined each model contributes a SECOND curve (MAE_refined) in
        the same colour and dash, distinguished by hollow markers and a lighter
        stroke: base and refined belong on the same axis (that comparison is the
        point) while colour still means "model", not "with/without refiner".
        """
        parts = []
        for i, tag in enumerate(tags):
            k = i % len(_MODEL_COLOURS)
            label = tag.replace('_', chr(92) + '_')
            parts.append(
                rf'\addplot [mdlc{i}, thick, {_MODEL_DASH[k]}, mark={_MODEL_MARK[k]}, '
                rf'mark size=1.2pt]' '\n'
                rf'    table [x=t, y=MAE, col sep=comma] '
                rf'{{\rolloutdir/metrics_{tag.lower()}.csv}};' '\n'
                rf'\addlegendentry{{{label}}}')
            if show_refined:
                parts.append(
                    rf'\addplot [mdlc{i}, semithick, {_MODEL_DASH[k]}, '
                    rf'mark={_MODEL_MARK[k]}, mark size=1.2pt, '
                    rf'mark options={{fill=white}}, opacity=0.8]' '\n'
                    rf'    table [x=t, y=MAE_refined, col sep=comma] '
                    rf'{{\rolloutdir/metrics_{tag.lower()}.csv}};' '\n'
                    rf'\addlegendentry{{{label} $+$ refiner}}')
        if has_persist:
            parts.append(r'\addplot [mdlpersist, thick, densely dotted, no marks]'
                         '\n'
                         r'    table [x=t, y=MAE_persist, col sep=comma] '
                         rf'{{\rolloutdir/metrics_{tags[0].lower()}.csv}};' '\n'
                         r'\addlegendentry{persistence}')
        return '\n'.join(parts) + '\n'

    for i, tag in enumerate(tags):
        k = i % len(_MODEL_COLOURS)
        if i >= len(_MODEL_COLOURS):
            print(f'  [note] >{len(_MODEL_COLOURS)} models: colours reused '
                  f'(dash/marker still differ)')
        colours.append(rf'\definecolor{{mdlc{i}}}{{HTML}}{{{_MODEL_COLOURS[k]}}}')
    if has_persist:
        colours.append(r'\definecolor{mdlpersist}{HTML}{6B7280}')

    # Energy panel: one E_gen curve per model in the SAME colour/dash as its MAE
    # curve (so the panel needs no model legend of its own), plus the shared
    # ground truth.  Reads each model's energy_<tag>.csv copied next to the
    # metrics CSVs.  With the MAE panel off it stands alone and DOES need its own
    # model legend, since there is no other panel carrying one.
    n_panels = (1 if has_mae else 0) + (1 if has_energy else 0)
    axis_w = _PANEL_W2 if n_panels == 2 else _PANEL_W1
    energy_panel = ''
    if has_energy:
        eplots = ('\\addplot [black!60, thick, densely dashed, no marks]\n'
                  '    table [x=step, y=E_real, col sep=comma] '
                  f'{{\\rolloutdir/energy_{tags[0].lower()}.csv}};\n'
                  '\\addlegendentry{ground truth}\n')
        for i, tag in enumerate(tags):
            k = i % len(_MODEL_COLOURS)
            eplots += (f'\\addplot [mdlc{i}, thick, {_MODEL_DASH[k]}, '
                       f'mark={_MODEL_MARK[k]}, mark size=1.2pt'
                       + (', forget plot' if has_mae else '') + ']\n'
                       '    table [x=step, y=E_gen, col sep=comma] '
                       f'{{\\rolloutdir/energy_{tag.lower()}.csv}};\n')
            if not has_mae:
                eplots += (r'\addlegendentry{'
                           + tag.replace('_', chr(92) + '_') + '}\n')
        energy_panel = ((_PANEL_JOIN if has_mae else '')
                        + '\\begin{tikzpicture}\n'
                        + _energy_axis(axis_w, eplots, labels_right=has_mae)
                        + '\n\\end{tikzpicture}')

    mae_panel = ''
    if has_mae:
        mae_panel = ('\\begin{tikzpicture}\n'
                     + _mae_axis(axis_w, _mae_body())
                     + '\n\\end{tikzpicture}')
    panels = mae_panel + energy_panel
    panels = panels + '\\\\[6pt]\n' if panels else ''
    guard = (f'metrics_{tags[0].lower()}.csv' if has_mae
             else f'energy_{tags[0].lower()}.csv' if has_energy
             else f'{fields[0]}.png')
    graphics = '\n'.join(
        rf'\includegraphics[width=\linewidth]{{\rolloutdir/{f}.png}}\\[2pt]'
        for f in fields)
    tex = rf"""% Auto-generated by scripts/report_pipeline.py -- do not edit by hand.
% Combined figure: all models on one rollout.  Preamble needs:
%   \usepackage{{pgfplots}} \usepackage{{graphicx}} \pgfplotsset{{compat=1.18}}
% Set \def\rolloutroot{{<subdir>/}} (TRAILING SLASH) before \input if the folder
% is not next to the main .tex.
\providecommand{{\rolloutroot}}{{}}
\def\rolloutdir{{\rolloutroot {folder}}}
{chr(10).join(colours)}
\IfFileExists{{\rolloutdir/{guard}}}{{%
\begin{{figure}}[tbp]
\centering
{panels}{graphics}
\caption{{Model comparison of {fields_txt} on {subject}{base_txt}{dt_txt}.{energy_txt}{dip_txt}{fluid_txt}}}
\label{{fig:rollout-combined-{folder}}}
\end{{figure}}
}}{{%
\begin{{center}}\fbox{{\parbox{{0.92\linewidth}}{{\textbf{{[report\_pipeline] missing
figure data:}} could not find \texttt{{\detokenize{{{folder}/{guard}}}}} (searched
via \texttt{{\textbackslash rolloutroot}}). Place the \texttt{{\detokenize{{{folder}}}}}
folder next to the main .tex, or set
\texttt{{\textbackslash def\textbackslash rolloutroot\{{<subdir>/\}}}} -- note: the
macro to set is \texttt{{\textbackslash rolloutroot}}, NOT
\texttt{{\textbackslash rolloutdir}} (the latter is overridden by this file).}}}}
\end{{center}}
}}
"""
    (out_dir / 'figure.tex').write_text(tex)
    print(f'  latex -> {out_dir / "figure.tex"}')


def write_figure_tex(out_dir: Path, run_id: str, fields: list, has_refined: bool,
                     tag: str = '', fluid_params: Optional[dict] = None,
                     has_persist: bool = False, col_dt: Optional[float] = None,
                     desc: Optional[str] = None,
                     has_energy: bool = False, energy_refined: bool = False,
                     energy_dip: Optional[dict] = None, has_mae: bool = True,
                     has_ctxswap: bool = False, has_base_rollout: bool = False):
    """Directly \\input-able figure: MAE-vs-t pgfplots graph + rollout image(s).

    Paths are relative to \\rolloutdir (defaults to the folder name), so drop the
    whole <run_id> folder next to the main .tex and \\input{<run_id>/figure.tex}.
    Requires \\usepackage{pgfplots} + \\usepackage{graphicx} in the preamble.
    """
    folder = out_dir.name              # <tag>_<run_id> (or run_id without models)
    # See write_combined_tex: inserted verbatim, falls back to the run id.
    subject = desc if desc else rf'run \texttt{{{run_id.replace("_", chr(92) + "_")}}}'
    model_txt = (f" under model \\texttt{{{tag.replace('_', chr(92) + '_')}}}"
                 if tag else '')
    # The strip carries a second rollout row from a pass with the refiner never
    # loaded; unlabelled, two near-identical rows under one model name read as a
    # duplicate rather than a controlled comparison.
    if has_base_rollout:
        model_txt += ' (with and without refiner active)'
    # See write_combined_tex: the caption states only what the image cannot.
    energy_txt = _ENERGY_CAPTION if has_energy else ''
    ctx_cmp_txt = (' The two rollouts are the same model from the same seed frame, '
                   'differing only in which trajectory the context encoder was given.'
                   if has_ctxswap else '')
    dip_txt = _dip_caption(energy_dip)
    fluid_txt = _fluid_caption(fluid_params or {})
    fields_txt = ', '.join(field_tex(f) for f in fields)
    # col_dt is the MODEL's own timestep (one forward pass), the physically
    # meaningful quantity; the column spacing is stride x that and is only a
    # display choice, already readable off the column numbers.
    dt_txt = (rf', $\Delta t = {_tex_num(col_dt)}$' if col_dt else '')
    # With a context comparison the two curves ARE the comparison, so the legend
    # names the conditioning rather than the model.
    base_plot = ('\\addplot [color1mae, thick, mark=*, mark size=1.2pt]\n'
                 '    table [x=t, y=MAE, col sep=comma] {\\rolloutdir/metrics.csv};\n'
                 + ('\\addlegendentry{true context}\n' if has_ctxswap
                    else '\\addlegendentry{base model}\n'))
    ctxswap_plot = ('\\addplot [color4mae, thick, dashdotted, mark=diamond*, '
                    'mark size=1.3pt]\n'
                    '    table [x=t, y=MAE, col sep=comma] '
                    '{\\rolloutdir/metrics_ctxswap.csv};\n'
                    '\\addlegendentry{mirror context}\n') if has_ctxswap else ''
    refined_plot = ''
    persist_plot = ''
    if has_persist:
        # Persistence MAE: the "copy the seed frame" baseline, dashed grey so it
        # reads as a reference rather than a third model.
        persist_plot = (
            r'\addplot [color3mae, thick, densely dotted, mark=triangle*, mark size=1.3pt]'
            '\n    table [x=t, y=MAE_persist, col sep=comma] {\\rolloutdir/metrics.csv};'
            '\n\\addlegendentry{persistence}\n')
    if has_refined:
        refined_plot = (
            r'\addplot [color2mae, thick, dashed, mark=square*, mark size=1.2pt]'
            '\n    table [x=t, y=MAE_refined, col sep=comma] {\\rolloutdir/metrics.csv};'
            '\n\\addlegendentry{with refiner}\n')
    graphics = '\n'.join(
        rf'\includegraphics[width=\linewidth]{{\rolloutdir/{f}.png}}\\[2pt]'
        for f in fields)

    # One panel gets the full 0.8\linewidth, two sit side by side at 0.46: the
    # figure is already a tall float and stacking graphs would push it over
    # \textheight in narrow classes.  Both can be off, leaving a filmstrip-only
    # figure.  Base and refined MAE always share ONE axis -- comparing them is
    # the whole point, and separate axes autoscale independently, which flattens
    # the difference.
    n_panels = (1 if has_mae else 0) + (1 if has_energy else 0)
    axis_w = _PANEL_W2 if n_panels == 2 else _PANEL_W1
    boxes = []
    if has_mae:
        boxes.append(_mae_axis(axis_w, base_plot + ctxswap_plot
                               + refined_plot + persist_plot))
    if has_energy:
        eplots = (
            '\\addplot [black!60, thick, densely dashed, mark=o, mark size=1.1pt]\n'
            '    table [x=step, y=E_real, col sep=comma] {\\rolloutdir/energy.csv};\n'
            '\\addlegendentry{ground truth}\n'
            '\\addplot [color1mae, thick, mark=*, mark size=1.2pt]\n'
            '    table [x=step, y=E_gen, col sep=comma] {\\rolloutdir/energy.csv};\n'
            '\\addlegendentry{rollout}\n')
        if energy_refined:
            eplots += (
                '\\addplot [color2mae, thick, dashed, mark=square*, mark size=1.2pt]\n'
                '    table [x=step, y=E_refined, col sep=comma] {\\rolloutdir/energy.csv};\n'
                '\\addlegendentry{with refiner}\n')
        boxes.append(_energy_axis(axis_w, eplots, labels_right=has_mae))
    panels = _PANEL_JOIN.join('\\begin{tikzpicture}\n' + b + '\n\\end{tikzpicture}'
                              for b in boxes)
    panels = panels + '\\\\[6pt]\n' if panels else ''
    # The \IfFileExists guard must test a file this figure actually reads, or a
    # misplaced folder would slip through silently.
    guard = ('metrics.csv' if has_mae
             else 'energy.csv' if has_energy else f'{fields[0]}.png')
    tex = rf"""% Auto-generated by scripts/report_pipeline.py — do not edit by hand.
% Preamble needs: \usepackage{{pgfplots}} \usepackage{{graphicx}} \pgfplotsset{{compat=1.18}}
% \rolloutroot (optional) = path prefix from the MAIN .tex to where the rollout
% folders live; default empty = folders sit next to the main .tex.  Set once with
% \def\rolloutroot{{figures/}} before the first \input.  (\rolloutdir itself is
% \def'd HERE, unconditionally, so multiple figure.tex files can be \input into
% one document without leaking each other's paths.)
\providecommand{{\rolloutroot}}{{}}
\def\rolloutdir{{\rolloutroot {folder}}}
\definecolor{{color1mae}}{{HTML}}{{2563EB}}
\definecolor{{color2mae}}{{HTML}}{{E8710A}}
\definecolor{{color3mae}}{{HTML}}{{6B7280}}
\definecolor{{color4mae}}{{HTML}}{{7C3AED}}
% Guard: a wrong \rolloutroot (or a misplaced folder) must be VISIBLE in the PDF,
% not a silently-empty figure with errors buried in the .log.
\IfFileExists{{\rolloutdir/{guard}}}{{%
% [tbp]: the filmstrip makes this a TALL float; top-only placement ([t]) can
% exceed \topfraction in narrow-text-block classes (e.g. elsarticle), and LaTeX
% then silently defers the figure to the END of the document.  Allowing float
% pages (p) keeps it near its \input.
\begin{{figure}}[tbp]
\centering
{panels}{graphics}
\caption{{Autoregressive rollout of {fields_txt} on {subject}{model_txt}{dt_txt}.{ctx_cmp_txt}{energy_txt}{dip_txt}{fluid_txt}}}
\label{{fig:rollout-{folder}}}
\end{{figure}}
}}{{%
\begin{{center}}\fbox{{\parbox{{0.92\linewidth}}{{\textbf{{[report\_pipeline] missing
figure data:}} could not find \texttt{{\detokenize{{{folder}/{guard}}}}} (searched
via \texttt{{\textbackslash rolloutroot}}). Place the \texttt{{\detokenize{{{folder}}}}}
folder next to the main .tex, or set
\texttt{{\textbackslash def\textbackslash rolloutroot\{{<subdir>/\}}}} — note: the
macro to set is \texttt{{\textbackslash rolloutroot}}, NOT
\texttt{{\textbackslash rolloutdir}} (the latter is overridden by this file).}}}}
\end{{center}}
}}
"""
    (out_dir / 'figure.tex').write_text(tex)
    print(f'  latex → {out_dir / "figure.tex"}')


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('config', help='pipeline config JSON (see module docstring)')
    ap.add_argument('--skip-infer', action='store_true',
                    help='reuse each rollout\'s existing frames/ + metrics.csv and only '
                         're-render images + latex')
    args = ap.parse_args()

    cfg = json.loads(Path(args.config).read_text())
    # Resolve against THIS invocation's cwd immediately — relative paths must never
    # reach the infer.py subprocess (whose cwd is the hfm repo).
    out_root = Path(cfg['out_root']).resolve()
    n_frames = int(cfg['frames_per_image'])
    out_root.mkdir(parents=True, exist_ok=True)

    # Models layer: each model = its own checkpoints + tag; every model runs every
    # rollout, into <tag>_<run_id>.  A config without "models" (checkpoints at the
    # top level) behaves as a single untagged model, exactly as before.
    models = cfg.get('models')
    if not models:
        models = [{'tag': cfg.get('tag', '')}]
    # Tags are LOWERCASED everywhere (folder names, \rolloutdir baked into
    # figure.tex, labels, captions): a case mismatch between the config tag and
    # the on-disk folder compiles fine on macOS (case-insensitive FS) but breaks
    # on Linux/Overleaf, so one canonical case removes the trap entirely.
    seen_tags = [m.get('tag', '').lower() for m in models]
    if len(models) > 1 and len(set(seen_tags)) != len(seen_tags):
        raise SystemExit(f'model tags must be unique (case-insensitive), got {seen_tags}')

    # Two rollouts that resolve to the same folder would silently overwrite each
    # other's images and CSVs, leaving a report that looks complete but isn't.
    # Catch it before any inference runs.
    seen = {}
    for roll in cfg['rollouts']:
        rid = Path(roll['run_dir']).name
        p = str(roll.get('postfix', '')).strip().strip('_').lower()
        p = ('_' + re.sub(r'[^a-z0-9._-]+', '-', p)) if p else ''
        key = f'{rid}{p}'
        if key in seen:
            raise SystemExit(
                f'two rollouts both resolve to output folder "{key}"\n'
                f'  first : {seen[key]}\n  second: {roll}\n'
                f'  Give at least one of them a distinct "postfix" '
                f'(e.g. "postfix": "s3") so they do not overwrite each other.')
        seen[key] = roll

    n_total = len(models) * len(cfg['rollouts'])
    job = 0
    failed = []
    # ROLLOUT outer, MODEL inner: every model's rows for one rollout must be in
    # hand at the same time to stack them into the combined figure.
    for roll in cfg['rollouts']:
        run_dir = Path(roll['run_dir']).resolve()
        fields = roll['fields'] if isinstance(roll['fields'], list) else [roll['fields']]
        stride = int(roll['stride'])
        unknown = [f for f in fields if f not in FIELD_NAMES]
        if unknown:
            raise SystemExit(f'unknown field(s) {unknown}; valid: {FIELD_NAMES}')
        run_id = run_dir.name
        # Optional postfix so the SAME run can appear several times in one report
        # (different stride, fields, infer args...) without the later entries
        # overwriting the earlier ones.  Lowercased for the same reason tags are:
        # a case-only difference resolves on macOS and breaks on Linux/Overleaf.
        pf = str(roll.get('postfix', '')).strip().strip('_').lower()
        pf = ('_' + re.sub(r'[^a-z0-9._-]+', '-', pf)) if pf else ''
        folder_id = f'{run_id}{pf}'
        desc = roll.get('description')     # optional prose subject for captions
        # Per-rollout, falling back to the top-level value: one report usually
        # wants the same setting everywhere, but individual figures often need to
        # differ (e.g. show the context once, not on every panel).
        def flag(name, default):
            return bool(roll.get(name, cfg.get(name, default)))
        inc_ctx = flag('include_context', False)          # opt-in
        inc_ref = flag('include_refiner_detail', True)    # on when a refiner ran
        inc_energy = flag('include_energy', False)        # opt-in: energy panel
        inc_mae = flag('include_mae', True)               # opt-out: MAE panel
        # Refiner curves on the combined figure's MAE axis (the per-model
        # figure always shows both).  Old key kept working.
        sep_ref = flag('include_refiner_mae',
                       bool(roll.get('separate_refiner_mae',
                                     cfg.get('separate_refiner_mae', False))))
        # Control rollout with the refiner entirely absent.  Only meaningful when
        # the refiner is IN the loop -- without feedback the displayed rollout row
        # already is the un-refined prediction and a second pass would duplicate it.
        inc_base = flag('include_base_rollout', False)
        # Per-model output folders (<tag>_<run_id>/) on top of the combined one.
        # Off by default: the usual deliverable is the single stacked figure, and
        # one folder per model per rollout is a lot of files nobody opens.
        inc_indiv = flag('individual_rollouts', False)
        # CONTEXT COMPARISON: a mirror corpus whose history conditions a SECOND
        # rollout of the same model from the same seed frame.  Per-rollout only --
        # it names a specific corpus, so a top-level default is meaningless.
        ctx_swap = roll.get('comparison_context', roll.get('context_swap'))
        if ctx_swap and not inc_indiv:
            # A context comparison suppresses the combined figure, so without the
            # per-model ones there would be no output at all.
            print('  [note] comparison_context needs per-model figures; '
                  'individual_rollouts forced on.')
            inc_indiv = True
        if not inc_indiv and not any(m.get('tag') for m in models):
            # Untagged single model: <tag>_<run_id> IS <run_id>, so there is no
            # separate combined folder to fall back on.
            print('  [note] no model tags, so there is no separate combined '
                  'folder; individual_rollouts forced on.')
            inc_indiv = True
        if not (inc_mae or inc_energy):
            print('  [note] include_mae and include_energy are both off: '
                  'the figure is the filmstrip alone.')
        # Perturbation record (fvm_gen_tests/gen_energy_dip.py): read once per
        # rollout; drives the caption whether or not the energy panel is on,
        # because the GT row's step-0 column IS the perturbed frame.
        dip_info = load_energy_dip(run_dir)
        # None (not merely False) when unset, so run_inference can fall back to
        # the per-model / top-level value; an explicit false here still wins.
        roll_fb = roll.get('refine_feedback')
        roll_fb = None if roll_fb is None else bool(roll_fb)
        # Accumulated across models for the combined figure.
        combo_rows = {f: [] for f in fields}      # field -> [(tag_raw, gen_row)]
        combo_real = {}                           # field -> real_row (shared)
        combo_ctx = {}                            # field -> (context_row, label)
        combo_tags, combo_dt, combo_persist = [], None, False
        combo_steps = None                        # real rollout step per column
        combo_energy = []                         # (tag, energy.csv) per model
        combo_refined = []                        # MAE_refined present, per model
        combo_base = []                           # no-refiner row present, per model
        combo_mask = None                         # fluid mask (identical per model)
        model_dirs = []                           # <tag>_<run_id> dirs, per model
        for model in models:
            tag_raw = model.get('tag', '')     # original form, caption only
            tag = tag_raw.lower()              # folders / \rolloutdir / labels
            # Feedback is only real if there is a refiner to feed back; keep the
            # caption honest rather than announcing a loop that never closed.
            eff_fb = (_refine_feedback_set(cfg, model, roll_fb)
                      and bool(model.get('refiner_ckpt') or cfg.get('refiner_ckpt')))
            job += 1
            out_dir = out_root / (f'{tag}_{folder_id}' if tag else folder_id)
            out_dir.mkdir(parents=True, exist_ok=True)
            n_predict = stride * (n_frames - 1)
            print(f'[{job}/{n_total}] {out_dir.name}: fields={fields} '
                  f'stride={stride} -> n_predict={n_predict}')

            frames_dir = out_dir / 'frames'
            base_dir = out_dir / 'frames_norefine'
            if not args.skip_infer:
                work = out_dir / '_infer_work'
                if work.exists():
                    shutil.rmtree(work)
                run_inference(cfg, model, run_dir, n_predict, work,
                              refine_feedback=roll_fb)
                # infer.py silently SKIPS runs that are too short to hold
                # n_context + 1 + n_predict frames (it prints "[skip]" and exits 0),
                # in which case there is no viewer output to collect.  Explain and
                # move on rather than dying on the missing directory.
                produced = work / 'viewer' / run_id
                if not produced.exists() or not (work / run_id / 'metrics.csv').exists():
                    n_have = len(list(run_dir.glob('t_*.npz')))
                    print(f'  !! SKIPPED: no rollout output found at {produced}.\n'
                          f'     If infer.py printed a "[skip]" line above, the run is '
                          f'too short: it holds {n_have} frames but needs n_context + 1 '
                          f'+ n_predict (n_predict={n_predict} from '
                          f'stride*(frames_per_image-1)) — lower "stride"/'
                          f'"frames_per_image" or pick a longer run.\n'
                          f'     Otherwise check the infer.py output above for the '
                          f'actual failure.')
                    failed.append(out_dir.name)
                    shutil.rmtree(work, ignore_errors=True)
                    try:                       # drop the empty output folder
                        out_dir.rmdir()
                    except OSError:
                        pass                   # keeps earlier successful artifacts
                    continue
                shutil.copy(work / run_id / 'metrics.csv', out_dir / 'metrics.csv')
                # Optional control pass: same model, same window, refiner never
                # loaded.  Skipped when it would be a duplicate of the row above.
                base_work = base_produced = None
                if inc_base and eff_fb:
                    base_work = out_dir / '_infer_work_norefine'
                    if base_work.exists():
                        shutil.rmtree(base_work)
                    print('  [base rollout] second pass with the refiner disabled')
                    run_inference(cfg, model, run_dir, n_predict, base_work,
                                  no_refiner=True)
                    base_produced = base_work / 'viewer' / run_id
                    if not base_produced.exists():
                        print('  [warn] no-refiner pass produced no frames; '
                              'skipping that row')
                        base_produced = None
                elif inc_base:
                    print('  [note] include_base_rollout skipped: without '
                          'refine_feedback the rollout row already is the '
                          'un-refined prediction.')
                swap_work = swap_produced = None
                if ctx_swap:
                    swap_work = out_dir / '_infer_work_ctxswap'
                    if swap_work.exists():
                        shutil.rmtree(swap_work)
                    print(f'  [context swap] second pass conditioned on {ctx_swap}')
                    run_inference(cfg, model, run_dir, n_predict, swap_work,
                                  refine_feedback=roll_fb, context_run=ctx_swap)
                    swap_produced = swap_work / 'viewer' / run_id
                    if not swap_produced.exists():
                        print('  [warn] context-swap pass produced no frames '
                              '(donor missing, or too short) — skipping that row')
                        swap_produced = None
                    else:
                        shutil.copy(swap_work / run_id / 'metrics.csv',
                                    out_dir / 'metrics_ctxswap.csv')
                # Render straight from the work dir; the raw frames are NOT part of
                # the report output (only image/CSV/tex are).  keep_frames: true in
                # the config retains a frames/ copy for later interactive viewing
                # and for --skip-infer re-rendering.
                col_dt, strip_rows, step_ids, value_mask = render_strips(
                    run_dir, produced, fields, stride, n_frames, out_dir,
                    include_context=inc_ctx, include_refiner_detail=inc_ref,
                    tag=tag_raw, base_frames_dir=base_produced,
                    swap_frames_dir=swap_produced,
                    swap_label=Path(ctx_swap).name if ctx_swap else '',
                    write_images=inc_indiv)
                # Before the work dir is deleted: the energy CSV reads every
                # frame infer.py wrote, not just the filmstrip columns.
                energy_info = (write_energy_csv(run_dir, produced, out_dir)
                               if inc_energy else None)
                if cfg.get('keep_frames', False):
                    if frames_dir.exists():
                        shutil.rmtree(frames_dir)
                    shutil.move(str(produced), str(frames_dir))
                    if base_produced is not None:
                        if base_dir.exists():
                            shutil.rmtree(base_dir)
                        shutil.move(str(base_produced), str(base_dir))
                elif frames_dir.exists():      # stale copy from an older run
                    shutil.rmtree(frames_dir)
                if not cfg.get('keep_frames', False) and base_dir.exists():
                    shutil.rmtree(base_dir)
                shutil.rmtree(work)
                if base_work is not None:
                    shutil.rmtree(base_work, ignore_errors=True)
                if swap_work is not None:
                    shutil.rmtree(swap_work, ignore_errors=True)
            else:
                swap_produced = None
                if not frames_dir.exists():
                    raise SystemExit(
                        f'--skip-infer needs {frames_dir}, which only exists when the '
                        f'pipeline ran with "keep_frames": true. Re-run without '
                        f'--skip-infer (or set keep_frames).')
                col_dt, strip_rows, step_ids, value_mask = render_strips(
                    run_dir, frames_dir, fields, stride, n_frames, out_dir,
                    include_context=inc_ctx, include_refiner_detail=inc_ref,
                    tag=tag_raw,
                    base_frames_dir=base_dir if (inc_base and base_dir.exists())
                    else None, write_images=inc_indiv)
                if ctx_swap:
                    print('  [note] --skip-infer: the context-swap row needs a '
                          'fresh inference pass and is omitted.')
                energy_info = (write_energy_csv(run_dir, frames_dir, out_dir)
                               if inc_energy else None)
            # refined columns exist iff a refiner actually ran — trust the CSV
            rows = (out_dir / 'metrics.csv').read_text().splitlines()
            header = rows[0].split(',') if rows else []
            def _col(name):
                """Value of `name` in the first data row, '' when the column is
                absent — CSVs from an older infer.py have no MAE_persist."""
                if name not in header or len(rows) < 2:
                    return ''
                cells = rows[1].split(',')
                i = header.index(name)
                return cells[i] if i < len(cells) else ''
            swap_ok = (out_dir / 'metrics_ctxswap.csv').exists()
            base_ok = (base_produced is not None if not args.skip_infer
                       else bool(inc_base and base_dir.exists()))
            if inc_indiv:
                write_figure_tex(out_dir, run_id, fields,
                                 has_refined=_col('MAE_refined') != '',
                                 tag=tag_raw,
                                 fluid_params=load_fluid_params(run_dir),
                                 has_persist=_col('MAE_persist') != '',
                                 col_dt=col_dt, desc=desc, has_ctxswap=swap_ok,
                                 has_energy=energy_info is not None,
                                 energy_refined=bool(energy_info
                                                     and energy_info['refined']),
                                 energy_dip=dip_info, has_mae=inc_mae,
                                 has_base_rollout=base_ok)
            if energy_info is not None:
                combo_energy.append((tag_raw or 'model', out_dir / 'energy.csv'))
            combo_refined.append(_col('MAE_refined') != '')
            combo_base.append(base_ok)

            # Stash this model's rows/metrics for the combined figure.
            for f, (real_row, gen_row, base_row, swap_row,
                    ctx_row, ctx_lab) in strip_rows.items():
                combo_real.setdefault(f, real_row)
                combo_ctx.setdefault(f, (ctx_row, ctx_lab))
                # Raw tag (possibly empty) — _rollout_label supplies the wording,
                # so the bar text is built in exactly one place.
                combo_rows[f].append((tag_raw, False, gen_row))
                if base_row is not None:
                    combo_rows[f].append((tag_raw, True, base_row))
            combo_tags.append((tag_raw or 'model', out_dir / 'metrics.csv'))
            model_dirs.append(out_dir)
            combo_mask = combo_mask if combo_mask is not None else value_mask
            combo_dt = combo_dt or col_dt
            combo_steps = combo_steps or step_ids
            combo_persist = combo_persist or (_col('MAE_persist') != '')

        # ---- combined figure: Real on top, one row per model below ----
        # A context comparison already spends the figure's rows on true-vs-mirror
        # for ONE model; stacking several models on top would put two different
        # contrasts in one panel and make neither readable.  One figure per model.
        if ctx_swap:
            print('  [note] comparison_context set: per-model figures only, no '
                  'cross-model combined figure.')
            continue
        # Folder is the BARE run id (no tag prefix) since it spans every model.
        if not combo_tags:
            continue
        comb_dir = out_root / folder_id
        comb_dir.mkdir(parents=True, exist_ok=True)
        for label, csv in combo_tags:
            shutil.copy(csv, comb_dir / f'metrics_{label.lower()}.csv')
        # Energy panel only when EVERY model produced a CSV — a partial panel
        # would invite reading a missing curve as a flat one.
        comb_energy = inc_energy and len(combo_energy) == len(combo_tags)
        if comb_energy:
            for label, csv in combo_energy:
                shutil.copy(csv, comb_dir / f'energy_{label.lower()}.csv')
        elif inc_energy:
            print('  [note] energy panel dropped from the combined figure: '
                  f'{len(combo_energy)}/{len(combo_tags)} models produced '
                  'energy.csv (per-model figures still carry theirs).')
        for f in fields:
            rows_f = combo_rows[f]
            if not rows_f:
                continue
            # A short run can clip a model's strip; align on the shortest so the
            # rows stay column-aligned rather than silently offset.
            n_col = min([len(combo_real[f])] + [len(r) for _, _, r in rows_f])
            ck = {}
            ctx_row, ctx_lab = combo_ctx.get(f, (None, ''))
            if ctx_row is not None:
                ck = dict(context_row=ctx_row[:n_col], context_label=ctx_lab)
            png = _strip_png_bytes(
                [combo_real[f][:n_col]] + [r[:n_col] for _, _, r in rows_f],
                f, (combo_steps or list(range(n_col)))[:n_col],
                row_labels=['Ground truth'] + [_rollout_label(lab, nr)
                                               for lab, nr, _ in rows_f],
                value_mask=combo_mask, **ck)
            (comb_dir / f'{f}.png').write_bytes(png)
            print(f'  combined image → {comb_dir / f"{f}.png"}')
        # A refined axis needs MAE_refined from EVERY model, else one panel would
        # be silently empty for the models that ran without a refiner.
        comb_refined = bool(combo_refined) and all(combo_refined)
        if sep_ref and inc_mae and any(combo_refined) and not comb_refined:
            print('  [note] include_refiner_mae: only '
                  f'{sum(combo_refined)}/{len(combo_refined)} models have a '
                  'refiner, so the combined MAE axis shows base curves only.')
        write_combined_tex(comb_dir, run_id, fields,
                           [lab for lab, _ in combo_tags],
                           col_dt=combo_dt,
                           fluid_params=load_fluid_params(run_dir),
                           has_persist=combo_persist, desc=desc,
                           has_energy=comb_energy, energy_dip=dip_info,
                           has_mae=inc_mae, has_refined=comb_refined,
                           separate_refiner=sep_ref,
                           has_base_rollout=bool(combo_base) and all(combo_base))
        if not inc_indiv:
            # The combined figure has copied every CSV it needs and built its
            # images from the in-memory rows, so the per-model folders are now
            # redundant.  Guarded on comb_dir: an untagged model shares that path.
            gone = 0
            for d in model_dirs:
                if d.resolve() != comb_dir.resolve():
                    shutil.rmtree(d, ignore_errors=True)
                    gone += 1
            print(f'  [individual_rollouts off] removed {gone} per-model '
                  f'folder(s); combined figure only')

    if failed:
        print(f'\nDone with {len(failed)} SKIPPED job(s): {failed}\n'
              f'Report data in {out_root}')
        sys.exit(1)
    print(f'\nDone. Report data in {out_root}')


if __name__ == '__main__':
    main()
