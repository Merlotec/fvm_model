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
     Real / Generated rows plus the refiner-residual row when a refiner is given,
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
      "stride":  2 }                                // shared by all fields of this rollout
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
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

import numpy as np

_HERE = Path(__file__).resolve().parent
_HFM_DIR = _HERE.parents[1] / 'hfm'
sys.path.insert(0, str(_HERE.parent / 'fvm_viewer'))

from strip_export import (FIELD_NAMES, build_renderer, find_timestep_files,
                          load_step, load_gen_frame, load_gen_refined,
                          closest_idx, _strip_indices, _strip_png_bytes)


def run_inference(cfg: dict, model: dict, run_dir: Path, n_predict: int,
                  work_dir: Path) -> Path:
    """Invoke hfm/infer.py for one (model, run); returns the dir with its outputs.

    Per-model keys (dynamics_ckpt, refiner_ckpt, refine_steps, infer_args) override
    the same keys at the top level of the config.
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
    if opt('refiner_ckpt'):
        cmd += ['--refine', str(Path(opt('refiner_ckpt')).resolve()),
                '--refine-steps', str(opt('refine_steps', 6))]
    cmd += list(opt('infer_args', []))
    print(f'  $ {" ".join(cmd)}')
    subprocess.run(cmd, check=True, cwd=str(hfm_dir))
    return work_dir


def render_strips(run_dir: Path, frames_dir: Path, fields: list, stride: int,
                  n_frames: int, out_dir: Path):
    """One filmstrip per field.  Returns (col_dt, {field: (real_row, gen_row)}).

    Same renderer as the web viewer's compare download, with ONE difference: the
    columns are numbered 0,1,2,... consecutively rather than by rollout step, since
    the figure caption carries the spacing (the returned col_dt) instead.

    The rows are handed back so the caller can stack every model's generated row
    under one shared Real row for the combined figure, without re-rendering.
    """
    gen_files = find_timestep_files(str(frames_dir))
    real_files = find_timestep_files(str(run_dir))
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

    idxs = _strip_indices(t_off, stride, n_frames, len(gen_files))
    # Plain 0,1,2,... column labels; the spacing lives in the caption instead.
    step_ids = list(range(len(idxs)))
    # Measured from the frames' own timestamps rather than assumed from
    # stride*save_t, so it stays correct for every dataset format.
    col_ts = [load_gen_frame(gen_files[i])[0] for i in idxs]
    col_dt = (col_ts[1] - col_ts[0]) if len(col_ts) > 1 else None
    if len(idxs) < n_frames:
        print(f'  [note] only {len(idxs)}/{n_frames} columns fit '
              f'({len(gen_files)} frames at stride {stride})')

    written, rows_by_field = [], {}
    for field in fields:
        fi = FIELD_NAMES.index(field)
        real_row, gen_row, detail_row = [], [], []
        any_refined = False
        for i in idxs:
            t, gg, _ = load_gen_frame(gen_files[i])
            gen_row.append(gg[fi])
            _, rcp = load_step(real_files[closest_idx(real_files, t)])
            real_row.append(renderer.render_cell_smooth(rcp).numpy()[fi])
            rg = load_gen_refined(gen_files[i])
            detail_row.append(rg[fi] - gg[fi] if rg is not None else None)
            any_refined = any_refined or rg is not None

        kwargs = {}
        if any_refined:
            present = [d for d in detail_row if d is not None]
            kwargs = dict(detail_row=detail_row,
                          detail_maxabs=max((float(np.abs(d).max()) for d in present),
                                            default=1.0) or 1.0)
        png = _strip_png_bytes([real_row, gen_row], field, step_ids,
                               row_labels=["Real", "Generated"], **kwargs)
        out = out_dir / f'{field}.png'
        out.write_bytes(png)
        written.append(out)
        rows_by_field[field] = (real_row, gen_row)
        print(f'  image → {out}')
    return col_dt, rows_by_field


# Bookkeeping keys in params.json / ic.json that are not fluid parameters.
_NON_FLUID_KEYS = {'problem', 'mesh_uid', 'n_colliders', 'context_id', 'ic_id',
                   'traj_id', 'traj_uid', 'segment_id', 't_offset', 'duration'}

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
    ('save_t',           r'\Delta t'),
]


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
        parts.append(rf'${label} = {_tex_num(params[k])}$')
    return ' Fluid parameters: ' + '; '.join(parts) + '.'


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
                       has_persist: bool = False):
    """Combined figure: every model's MAE curve on one axis, above a filmstrip
    whose rows are Real followed by one row per model.

    Reads each model's metrics_<tag>.csv, copied into this folder so it is
    self-contained and can be dropped into a paper on its own.  No refiner
    curves or residual rows here -- this figure compares MODELS.
    """
    folder = out_dir.name
    safe_id = run_id.replace('_', r'\_')
    fluid_txt = _fluid_caption(fluid_params or {})
    dt_txt = (rf' spaced $\Delta t = {_tex_num(col_dt)}$ apart' if col_dt else '')

    colours, plots, legend = [], [], []
    for i, tag in enumerate(tags):
        k = i % len(_MODEL_COLOURS)
        if i >= len(_MODEL_COLOURS):
            print(f'  [note] >{len(_MODEL_COLOURS)} models: colours reused '
                  f'(dash/marker still differ)')
        cname = f'mdlc{i}'
        colours.append(rf'\definecolor{{{cname}}}{{HTML}}{{{_MODEL_COLOURS[k]}}}')
        plots.append(
            rf'\addplot [{cname}, thick, {_MODEL_DASH[k]}, mark={_MODEL_MARK[k]}, '
            rf'mark size=1.2pt]' '\n'
            rf'    table [x=t, y=MAE, col sep=comma] '
            rf'{{\rolloutdir/metrics_{tag.lower()}.csv}};')
        legend.append(rf'\addlegendentry{{{tag.replace("_", chr(92) + "_")}}}')
    if has_persist:
        colours.append(r'\definecolor{mdlpersist}{HTML}{6B7280}')
        plots.append(r'\addplot [mdlpersist, thick, densely dotted, no marks]'
                     '\n'
                     r'    table [x=t, y=MAE_persist, col sep=comma] '
                     rf'{{\rolloutdir/metrics_{tags[0].lower()}.csv}};')
        legend.append(r'\addlegendentry{persistence}')
    body = '\n'.join(f'{p}\n{l}' for p, l in zip(plots, legend))

    graphics = '\n'.join(
        rf'\includegraphics[width=\linewidth]{{\rolloutdir/{f}.png}}\\[2pt]'
        for f in fields)
    models_txt = ', '.join(t.replace('_', chr(92) + '_') for t in tags)
    tex = rf"""% Auto-generated by scripts/report_pipeline.py -- do not edit by hand.
% Combined figure: all models on one rollout.  Preamble needs:
%   \usepackage{{pgfplots}} \usepackage{{graphicx}} \pgfplotsset{{compat=1.18}}
% Set \def\rolloutroot{{<subdir>/}} (TRAILING SLASH) before \input if the folder
% is not next to the main .tex.
\providecommand{{\rolloutroot}}{{}}
\def\rolloutdir{{\rolloutroot {folder}}}
{chr(10).join(colours)}
\IfFileExists{{\rolloutdir/metrics_{tags[0].lower()}.csv}}{{%
\begin{{figure}}[tbp]
\centering
\begin{{tikzpicture}}
\begin{{axis}}[
    width=0.8\linewidth, height=4.8cm,
    xlabel={{rollout step $t$}}, ylabel={{MAE}},
    grid=major, grid style={{black!12}},
    axis line style={{black!60}},
    tick label style={{font=\small}}, label style={{font=\small}},
    legend style={{font=\small, draw=black!30, at={{(0.03,0.97)}}, anchor=north west}},
    legend cell align=left, ymin=0,
]
{body}
\end{{axis}}
\end{{tikzpicture}}\\[6pt]
{graphics}
\caption{{Model comparison on run \texttt{{{safe_id}}}: MAE over rollout step for
{models_txt}{' against the persistence baseline' if has_persist else ''} (top), and the
rollout filmstrip{'s' if len(fields) > 1 else ''} ({', '.join(fields)}) with the ground
truth above each model's prediction (columns are consecutive samples{dt_txt}, colour
scales shared across all rows and columns).{fluid_txt}}}
\label{{fig:rollout-combined-{folder}}}
\end{{figure}}
}}{{%
\begin{{center}}\fbox{{\parbox{{0.92\linewidth}}{{\textbf{{[report\_pipeline] missing
figure data:}} could not find \texttt{{\detokenize{{{folder}}}/metrics\_*.csv}} (searched
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
                     stride: int = 1):
    """Directly \\input-able figure: MAE-vs-t pgfplots graph + rollout image(s).

    Paths are relative to \\rolloutdir (defaults to the folder name), so drop the
    whole <run_id> folder next to the main .tex and \\input{<run_id>/figure.tex}.
    Requires \\usepackage{pgfplots} + \\usepackage{graphicx} in the preamble.
    """
    folder = out_dir.name              # <tag>_<run_id> (or run_id without models)
    safe_id = run_id.replace('_', r'\_')
    model_txt = (f" under model \\texttt{{{tag.replace('_', chr(92) + '_')}}}"
                 if tag else '')
    fluid_txt = _fluid_caption(fluid_params or {})
    # Columns are numbered 0,1,2,...; their spacing is stated here instead.
    if col_dt:
        step_txt = rf'consecutive samples spaced $\Delta t = {_tex_num(col_dt)}$ apart'
    else:
        step_txt = (f'consecutive samples every {stride} rollout step'
                    + ('s' if stride != 1 else ''))
    refined_plot = ''
    persist_plot = ''
    legend = r'\addlegendentry{base model}' + '\n'
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
% Guard: a wrong \rolloutroot (or a misplaced folder) must be VISIBLE in the PDF,
% not a silently-empty figure with errors buried in the .log.
\IfFileExists{{\rolloutdir/metrics.csv}}{{%
% [tbp]: the filmstrip makes this a TALL float; top-only placement ([t]) can
% exceed \topfraction in narrow-text-block classes (e.g. elsarticle), and LaTeX
% then silently defers the figure to the END of the document.  Allowing float
% pages (p) keeps it near its \input.
\begin{{figure}}[tbp]
\centering
\begin{{tikzpicture}}
\begin{{axis}}[
    width=0.8\linewidth, height=4.6cm,
    xlabel={{rollout step $t$}}, ylabel={{MAE}},
    grid=major, grid style={{black!12}},
    axis line style={{black!60}},
    tick label style={{font=\small}}, label style={{font=\small}},
    legend style={{font=\small, draw=black!30, at={{(0.03,0.97)}}, anchor=north west}},
    legend cell align=left, ymin=0,
]
\addplot [color1mae, thick, mark=*, mark size=1.2pt]
    table [x=t, y=MAE, col sep=comma] {{\rolloutdir/metrics.csv}};
{legend}{refined_plot}{persist_plot}\end{{axis}}
\end{{tikzpicture}}\\[6pt]
{graphics}
\caption{{Autoregressive rollout for run \texttt{{{safe_id}}}{model_txt}: MAE over rollout
step for the base model{' and with the flow-matching refiner' if has_refined else ''}{', against the persistence baseline' if has_persist else ''}
(top), and the rollout filmstrip{'s' if len(fields) > 1 else ''} ({', '.join(fields)};
columns are {step_txt}, colour scales shared across columns).{fluid_txt}}}
\label{{fig:rollout-{folder}}}
\end{{figure}}
}}{{%
\begin{{center}}\fbox{{\parbox{{0.92\linewidth}}{{\textbf{{[report\_pipeline] missing
figure data:}} could not find \texttt{{\detokenize{{{folder}}}/metrics.csv}} (searched
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
        # Accumulated across models for the combined figure.
        combo_rows = {f: [] for f in fields}      # field -> [(tag_raw, gen_row)]
        combo_real = {}                           # field -> real_row (shared)
        combo_tags, combo_dt, combo_persist = [], None, False
        for model in models:
            tag_raw = model.get('tag', '')     # original form, caption only
            tag = tag_raw.lower()              # folders / \rolloutdir / labels
            job += 1
            out_dir = out_root / (f'{tag}_{run_id}' if tag else run_id)
            out_dir.mkdir(parents=True, exist_ok=True)
            n_predict = stride * (n_frames - 1)
            print(f'[{job}/{n_total}] {out_dir.name}: fields={fields} '
                  f'stride={stride} -> n_predict={n_predict}')

            frames_dir = out_dir / 'frames'
            if not args.skip_infer:
                work = out_dir / '_infer_work'
                if work.exists():
                    shutil.rmtree(work)
                run_inference(cfg, model, run_dir, n_predict, work)
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
                # Render straight from the work dir; the raw frames are NOT part of
                # the report output (only image/CSV/tex are).  keep_frames: true in
                # the config retains a frames/ copy for later interactive viewing
                # and for --skip-infer re-rendering.
                col_dt, strip_rows = render_strips(run_dir, produced, fields,
                                                   stride, n_frames, out_dir)
                if cfg.get('keep_frames', False):
                    if frames_dir.exists():
                        shutil.rmtree(frames_dir)
                    shutil.move(str(produced), str(frames_dir))
                elif frames_dir.exists():      # stale copy from an older run
                    shutil.rmtree(frames_dir)
                shutil.rmtree(work)
            else:
                if not frames_dir.exists():
                    raise SystemExit(
                        f'--skip-infer needs {frames_dir}, which only exists when the '
                        f'pipeline ran with "keep_frames": true. Re-run without '
                        f'--skip-infer (or set keep_frames).')
                col_dt, strip_rows = render_strips(run_dir, frames_dir, fields,
                                                   stride, n_frames, out_dir)
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
            write_figure_tex(out_dir, run_id, fields,
                             has_refined=_col('MAE_refined') != '',
                             tag=tag_raw,
                             fluid_params=load_fluid_params(run_dir),
                             has_persist=_col('MAE_persist') != '',
                             col_dt=col_dt, stride=stride)

            # Stash this model's rows/metrics for the combined figure.
            for f, (real_row, gen_row) in strip_rows.items():
                combo_real.setdefault(f, real_row)
                combo_rows[f].append((tag_raw or 'model', gen_row))
            combo_tags.append((tag_raw or 'model', out_dir / 'metrics.csv'))
            combo_dt = combo_dt or col_dt
            combo_persist = combo_persist or (_col('MAE_persist') != '')

        # ---- combined figure: Real on top, one row per model below ----
        # Folder is the BARE run id (no tag prefix) since it spans every model.
        if not combo_tags:
            continue
        comb_dir = out_root / run_id
        comb_dir.mkdir(parents=True, exist_ok=True)
        for label, csv in combo_tags:
            shutil.copy(csv, comb_dir / f'metrics_{label.lower()}.csv')
        for f in fields:
            rows_f = combo_rows[f]
            if not rows_f:
                continue
            # A short run can clip a model's strip; align on the shortest so the
            # rows stay column-aligned rather than silently offset.
            n_col = min([len(combo_real[f])] + [len(r) for _, r in rows_f])
            png = _strip_png_bytes(
                [combo_real[f][:n_col]] + [r[:n_col] for _, r in rows_f],
                f, list(range(n_col)),
                row_labels=['Real'] + [lab for lab, _ in rows_f])
            (comb_dir / f'{f}.png').write_bytes(png)
            print(f'  combined image → {comb_dir / f"{f}.png"}')
        write_combined_tex(comb_dir, run_id, fields,
                           [lab for lab, _ in combo_tags],
                           col_dt=combo_dt,
                           fluid_params=load_fluid_params(run_dir),
                           has_persist=combo_persist)

    if failed:
        print(f'\nDone with {len(failed)} SKIPPED job(s): {failed}\n'
              f'Report data in {out_root}')
        sys.exit(1)
    print(f'\nDone. Report data in {out_root}')


if __name__ == '__main__':
    main()
