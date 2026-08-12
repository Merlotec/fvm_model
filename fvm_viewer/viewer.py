from typing import Optional, Union
"""
FVM solution browser-based viewer.

Runs a local web server — open the printed URL in your browser.
No display / GUI required; works over SSH port-forwarding from HPC nodes.

Usage
-----
    # Single run directory:
    python viewer.py path/to/mu_b_1.0000e-3/

    # Dataset directory (contains multiple run sub-dirs):
    python viewer.py path/to/fvm_gen_datasets/

    # Multi-mesh dataset (root/mesh_<uid>/run_<uid>/, as written by run_gen.py when
    # n_meshes > 1) — pick the mesh from the sidebar dropdown, then the run:
    python viewer.py path/to/fvm_gen_v2/

    # Sample a subset of a large sweep instead of loading all of it:
    #   -s/--sample N   at most N runs per mesh
    #   -m/--meshes M   at most M meshes
    python viewer.py path/to/fvm_gen_v2/ -m 3 -s 4

    # Compare real vs generated side-by-side (-c):
    python viewer.py path/to/fvm_gen_datasets/ -c path/to/infer_out/

    # Custom port:
    python viewer.py path/to/data/ --port 8050

HPC port-forwarding
-------------------
    On the HPC node:
        python viewer.py /path/to/data --port 8050

    On your local machine:
        ssh -L 8050:localhost:8050 user@hpc-node

    Then open http://localhost:8050 in your local browser.

Navigation
----------
    Click a run name on the left panel to switch runs.
    For multi-mesh datasets, the Mesh dropdown above the run list filters runs to one mesh.
    Use Prev / Next buttons or the slider to move between timesteps.
    Use the "Show Δ / Show Values" toggle to switch between absolute values and deltas.
    Plotly figures support scroll-to-zoom and drag-to-pan.
"""

import json
import os
import sys
import pickle
import argparse
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import dash
from dash import dcc, html, Input, Output, State, callback_context

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'fvm_gen'))
from renderer import MeshRenderer
from strip_export import (FIELD_NAMES, RESOLUTION, _LUT, _build_lut, _apply_cmap,
                          load_mesh, build_renderer, find_timestep_files, t_of_file,
                          load_step, load_gen_frame, load_gen_refined, closest_idx,
                          _render_field_strip, _strip_indices, _strip_png_bytes)


def _add_solver_to_path() -> None:
    """Put the FVM solver on sys.path so shared_mesh.pkl can be unpickled.

    Multi-mesh datasets store the geometry once per mesh as a pickled FVMMesh2D, so
    reading one needs `time_fvm` importable — unlike the old per-run mesh_props.npz,
    which is plain numpy.  Mirrors run_gen.py: the `fvm_solver` fork beside this repo,
    overridable with FVM_SOLVER_DIR.  Silent if absent — datasets that only use
    mesh_props.npz do not need the solver at all.
    """
    default_solver = str(Path(__file__).resolve().parents[2] / 'fvm_solver')
    solver_dir = os.environ.get("FVM_SOLVER_DIR", default_solver)
    if not os.path.isdir(solver_dir):
        return
    for p in (solver_dir, os.path.join(solver_dir, "time_fvm")):
        if p not in sys.path:
            sys.path.insert(0, p)


_add_solver_to_path()





# ---------------------------------------------------------------------------
# Video export helpers
# ---------------------------------------------------------------------------



def _render_frame_rgb(
    rows: list[tuple[str, np.ndarray]],
    view_mode: str,
    prev_rows: Optional[list[np.ndarray]] = None,
    zranges: Optional[list[tuple[float, float]]] = None,
    title: str = '',
) -> np.ndarray:
    """Render one video frame as RGB [H, W, 3] using PIL — no matplotlib figure."""
    from PIL import Image, ImageDraw

    CELL_W, CELL_H = 256, 256
    BAR_H   = 22
    LABEL_H = 20   # white strip with numbers above the gradient
    SCALE_H = 20   # gradient strip at the very bottom
    TITLE_H = 28 if title else 0
    n_rows  = len(rows)
    has_scale = bool(zranges)
    total_w = 4 * CELL_W
    total_h = TITLE_H + n_rows * (BAR_H + CELL_H) + (LABEL_H + SCALE_H if has_scale else 0)

    canvas = Image.new('RGB', (total_w, total_h), (255, 255, 255))
    draw   = ImageDraw.Draw(canvas)

    if title:
        draw.text((total_w // 2, TITLE_H // 2), title, fill=(20, 20, 20), anchor='mm')

    y = TITLE_H
    for ri, (label, grid) in enumerate(rows):
        prev  = prev_rows[ri] if prev_rows else None
        y_bar = y
        y_img = y + BAR_H
        draw.rectangle([0, y_bar, total_w, y_bar + BAR_H - 1], fill=(235, 235, 235))
        draw.text((6, y_bar + BAR_H // 2), label, fill=(20, 20, 20), anchor='lm')

        for ci in range(4):
            x0 = ci * CELL_W
            if view_mode == 'delta' and prev is not None:
                data   = grid[ci] - prev[ci]
                maxabs = float(np.abs(data).max()) or 1.0
                thumb  = _apply_cmap(data, 'RdBu_r', -maxabs, maxabs, CELL_W, CELL_H)
            else:
                zmin, zmax = zranges[ci] if zranges else (float(grid[ci].min()), float(grid[ci].max()))
                thumb  = _apply_cmap(grid[ci], 'viridis', zmin, zmax, CELL_W, CELL_H)
            canvas.paste(Image.fromarray(thumb), (x0, y_img))
            draw.text((x0 + 4, y_img + 4), FIELD_NAMES[ci], fill=(240, 240, 240))

        y += BAR_H + CELL_H

    if has_scale:
        # White label strip: channel name + min/max numbers
        y_label = y
        draw.rectangle([0, y_label, total_w, y_label + LABEL_H - 1], fill=(255, 255, 255))
        for ci in range(4):
            x0 = ci * CELL_W
            zmin, zmax = zranges[ci]
            mid = x0 + CELL_W // 2
            draw.text((x0 + 4, y_label + LABEL_H // 2), f"{zmin:.2g}", fill=(20, 20, 20), anchor='lm')
            draw.text((mid,    y_label + LABEL_H // 2), FIELD_NAMES[ci], fill=(20, 20, 20), anchor='mm')
            draw.text((x0 + CELL_W - 4, y_label + LABEL_H // 2), f"{zmax:.2g}", fill=(20, 20, 20), anchor='rm')
        # Gradient strip below the labels
        y_scale = y_label + LABEL_H
        cmap = 'RdBu_r' if view_mode == 'delta' else 'viridis'
        for ci in range(4):
            x0 = ci * CELL_W
            zmin, zmax = zranges[ci]
            grad_data = np.tile(np.linspace(zmin, zmax, CELL_W).astype(np.float32), (SCALE_H, 1))
            grad_rgb  = _apply_cmap(grad_data, cmap, zmin, zmax, CELL_W, SCALE_H)
            canvas.paste(Image.fromarray(grad_rgb), (x0, y_scale))

    return np.array(canvas)








def _encode_apng(frames: list[np.ndarray], fps: int = 8) -> bytes:
    """Encode RGB frames as an animated PNG (APNG) — pure PIL, no subprocess/fork."""
    import io
    from PIL import Image
    imgs = [Image.fromarray(f) for f in frames]
    buf  = io.BytesIO()
    imgs[0].save(
        buf, format='PNG', save_all=True,
        append_images=imgs[1:],
        loop=0, duration=int(1000 / fps),
    )
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Data helpers — real data
# ---------------------------------------------------------------------------

def _has_frames(d: str) -> bool:
    return any(f.startswith("t_") and f.endswith(".npz") for f in os.listdir(d))


def _is_run_dir(d: str) -> bool:
    """A run dir either carries its own mesh (old format) or just frames (new format)."""
    return os.path.exists(os.path.join(d, "mesh_props.npz")) or _has_frames(d)


def _subdirs(d: str) -> list[str]:
    return sorted(
        os.path.join(d, name) for name in os.listdir(d)
        if os.path.isdir(os.path.join(d, name))
    )


def _collect_run_dirs(root_dir: str) -> list[str]:
    """Discover run dirs under `root_dir`, handling both dataset layouts:

      flat (n_meshes == 1):   root/shared_mesh.pkl + root/run_XXXX_<uid>/
      multi-mesh:             root/mesh_<uid>/shared_mesh.pkl + .../run_XXXX_<uid>/

    Both are scanned, so a root mixing the two still resolves.
    """
    if _is_run_dir(root_dir):
        return [root_dir]

    runs, mesh_dirs = [], []
    for child in _subdirs(root_dir):
        (runs if _is_run_dir(child) else mesh_dirs).append(child)

    # One level deeper for the multi-mesh layout: root/mesh_<uid>/run_XXXX_<uid>/
    for mesh_dir in mesh_dirs:
        runs.extend(c for c in _subdirs(mesh_dir) if _is_run_dir(c))

    return sorted(runs)


def find_run_dirs(root_dir: str) -> list[str]:
    runs = _collect_run_dirs(root_dir)
    if not runs:
        raise RuntimeError(f"No run directories found under {root_dir}")
    return runs


def find_gen_run_dirs(root_dir: str) -> list[str]:
    """Find generated-data run dirs (no mesh_props.npz required; just t_*.npz files)."""
    return _collect_run_dirs(root_dir)


def mesh_key(run_dir: str) -> str:
    """Directory whose mesh governs `run_dir` — itself for the old per-run mesh_props.npz
    format, otherwise the parent holding shared_mesh.pkl.  Runs sharing a key share a
    renderer, so a multi-mesh dataset builds one renderer per mesh, not per run."""
    if os.path.exists(os.path.join(run_dir, "mesh_props.npz")):
        return run_dir
    return os.path.dirname(run_dir)


def run_label(root_dir: str, run_dir: str) -> str:
    """Path relative to the dataset root, so multi-mesh runs read as mesh_<uid>/run_<uid>."""
    rel = os.path.relpath(run_dir, root_dir)
    return os.path.basename(run_dir) if rel == "." else rel


def _spread(seq: list, n: Optional[int]) -> list:
    """Take `n` items spread evenly across `seq` (always including the first and last).

    Evenly spread rather than the first n, so a sampled subset still spans the parameter
    sweep instead of clustering on whatever sorted first.  Deterministic — no seed to
    record or reproduce.
    """
    if n is None or n >= len(seq):
        return seq
    if n <= 0:
        return []
    if n == 1:
        return [seq[0]]
    idx = sorted({round(i * (len(seq) - 1) / (n - 1)) for i in range(n)})
    return [seq[i] for i in idx]


def subsample_runs(run_dirs: list[str], per_mesh: Optional[int] = None,
                   max_meshes: Optional[int] = None) -> list[str]:
    """Trim discovered runs to a subset, keeping mesh grouping intact.

    Applied at discovery time, before renderers / frame listings / params are touched —
    so a sampled open of a big sweep never pays for the runs it drops.  `max_meshes` is
    the bigger lever: renderers are built per mesh, so dropping meshes is what actually
    cuts startup work.
    """
    if per_mesh is None and max_meshes is None:
        return run_dirs

    groups: dict[str, list[str]] = {}
    for d in run_dirs:
        groups.setdefault(os.path.dirname(d), []).append(d)

    kept_meshes = _spread(sorted(groups), max_meshes)
    out: list[str] = []
    for parent in kept_meshes:
        out.extend(_spread(groups[parent], per_mesh))
    return sorted(out)


def group_by_mesh_dir(root_dir: str, run_dirs: list[str]) -> tuple[list[dict], list[list[dict]]]:
    """Group runs by the directory that contains them, for the sidebar's mesh filter.

    Grouping is by parent dir rather than by `mesh_key`, because in the old flat format
    every run carries its own mesh_props.npz — keying on the mesh would yield one group
    per run.  Parent dir gives one group for a flat dataset and one per mesh_<uid> for a
    multi-mesh one, which is what the filter should offer.

    Returns (mesh_options, mesh_run_options); run option values are GLOBAL indices into
    `run_dirs`, so selection state stays mesh-agnostic.
    """
    groups: dict[str, list[int]] = {}
    for i, d in enumerate(run_dirs):
        groups.setdefault(os.path.dirname(d), []).append(i)

    mesh_options, mesh_run_options = [], []
    for mi, (parent, idxs) in enumerate(groups.items()):
        rel = os.path.relpath(parent, root_dir)
        mesh_options.append({
            "label": os.path.basename(parent) if rel == "." else rel,
            "value": mi,
        })
        mesh_run_options.append(
            [{"label": os.path.basename(run_dirs[i]), "value": i} for i in idxs]
        )
    return mesh_options, mesh_run_options






def build_renderers(run_dirs: list[str], resolution: tuple[int, int]) -> dict[str, MeshRenderer]:
    """Map every run dir to a renderer, building only one per distinct mesh.

    In a multi-mesh dataset all runs under a `mesh_<uid>/` share one geometry, so this
    turns n_meshes x runs_per_mesh renderer builds into n_meshes.
    """
    by_mesh: dict[str, MeshRenderer] = {}
    out: dict[str, MeshRenderer] = {}
    for d in run_dirs:
        key = mesh_key(d)
        if key not in by_mesh:
            print(f"  mesh {len(by_mesh) + 1}: {os.path.basename(key)}")
            by_mesh[key] = build_renderer(d, resolution)
        out[d] = by_mesh[key]
    return out














# ---------------------------------------------------------------------------
# Plot construction
# ---------------------------------------------------------------------------

_GRAPH_CFG = {"scrollZoom": True, "displayModeBar": True, "displaylogo": False}
_HEATMAP_LAYOUT = dict(
    xaxis  = dict(visible=False, scaleanchor="y"),
    yaxis  = dict(visible=False, autorange="reversed"),
    margin = dict(l=0, r=0, t=36, b=0),
    height = 280,
)
_ROW_LABEL_STYLE = {
    "gridColumn": "1 / -1", "fontWeight": "600", "fontSize": "12px",
    "padding": "4px 8px", "background": "#f0f0f0", "borderRadius": "3px",
}
_TOGGLE_STYLE = {
    "fontSize": "12px", "padding": "4px 12px", "cursor": "pointer",
    "marginLeft": "auto", "borderRadius": "4px", "border": "1px solid #aaa",
    "background": "#fff",
}


def make_field_figure(grid: np.ndarray, title: str,
                      zmin: Optional[float] = None, zmax: Optional[float] = None) -> go.Figure:
    fig = go.Figure(go.Heatmap(
        z=grid, colorscale="Viridis", showscale=True,
        zmin=zmin, zmax=zmax,
        colorbar=dict(thickness=10, len=0.85),
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=12), x=0.5, xanchor="center"),
        uirevision=title,
        **_HEATMAP_LAYOUT,
    )
    return fig


def make_empty_figure(H: int, W: int, title: str) -> go.Figure:
    """Placeholder on the same pixel axes as the field heatmaps, for frames with
    no refiner output (seed frames, or a run produced without --refine)."""
    fig = go.Figure()
    fig.update_layout(
        title=dict(text=title, font=dict(size=12), x=0.5, xanchor="center"),
        xaxis=dict(visible=False, range=[0, W], scaleanchor="y", constrain="domain"),
        yaxis=dict(visible=False, range=[H, 0]),
        margin=dict(l=0, r=0, t=36, b=0), height=280,
        plot_bgcolor="#f7f7f7", uirevision="refined",
    )
    return fig


def make_delta_figure(delta: np.ndarray, title: str, maxabs: Optional[float] = None) -> go.Figure:
    if maxabs is None:
        maxabs = float(np.abs(delta).max()) or 1.0
    fig = go.Figure(go.Heatmap(
        z=delta, colorscale="RdBu_r", showscale=True,
        zmid=0, zmin=-maxabs, zmax=maxabs,
        colorbar=dict(thickness=10, len=0.85),
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=12), x=0.5, xanchor="center"),
        uirevision=title,
        **_HEATMAP_LAYOUT,
    )
    return fig


def make_colorscale_figure(zmin: float, zmax: float, label: str,
                           colorscale: str = "Viridis") -> go.Figure:
    """Thin horizontal colorscale bar as a 1-row heatmap, placed between real and gen rows."""
    n = 200
    z = np.linspace(zmin, zmax, n).reshape(1, n)
    fig = go.Figure(go.Heatmap(z=z, colorscale=colorscale, showscale=False, zmin=zmin, zmax=zmax))
    fig.update_layout(
        title=dict(text=f"{label}  [{zmin:.3g} → {zmax:.3g}]", font=dict(size=9), x=0.5, xanchor="center"),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        margin=dict(l=4, r=4, t=18, b=4),
        height=44,
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    return fig


def load_params(run_dir: str) -> dict:
    path = os.path.join(run_dir, "params.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def _params_children(params: dict) -> list:
    if not params:
        return []
    rows = [
        html.Div(
            f"{k}:  {v:.4g}" if isinstance(v, float) else f"{k}:  {v}",
            style={"fontSize": "10px", "padding": "1px 0", "whiteSpace": "nowrap"},
        )
        for k, v in params.items()
    ]
    return [
        html.Hr(style={"margin": "8px 0", "borderColor": "#ddd"}),
        html.Div("Parameters", style={"fontSize": "11px", "fontWeight": "600", "marginBottom": "3px"}),
        *rows,
    ]


def _params_header_str(params: dict) -> str:
    if not params:
        return ""
    parts = [f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in params.items()]
    return "   ".join(parts)


def _sidebar(run_options: list[dict], mesh_options: Optional[list[dict]] = None) -> html.Div:
    """Run picker, with a mesh dropdown above it for multi-mesh datasets.

    The dropdown is always present so its callback resolves, but stays hidden when the
    dataset has a single mesh — there is nothing to filter by then.
    """
    multi = bool(mesh_options) and len(mesh_options) > 1
    return html.Div([
        html.Div([
            html.H4("Mesh", style={"margin": "0 0 6px 0", "fontSize": "13px", "fontWeight": "600"}),
            dcc.Dropdown(id="mesh-selector", options=mesh_options or [], value=0,
                         clearable=False, style={"fontSize": "11px"}),
        ], style={"display": "block" if multi else "none", "marginBottom": "12px"}),
        html.H4("Runs", style={"margin": "0 0 10px 0", "fontSize": "13px", "fontWeight": "600"}),
        dcc.RadioItems(
            id="run-selector", options=run_options, value=0,
            labelStyle={"display": "block", "fontSize": "11px",
                        "padding": "3px 0", "cursor": "pointer", "wordBreak": "break-all"},
        ),
        html.Div(id="params-display"),
    ], style={"width": "190px", "flexShrink": "0", "padding": "12px 10px",
              "borderRight": "1px solid #ddd", "overflowY": "auto", "fontFamily": "monospace"})


def _nav_bar(show_refined: bool = False) -> html.Div:
    return html.Div([
        html.Button("◀  Prev", id="btn-prev", n_clicks=0,
                    style={"fontSize": "13px", "padding": "5px 16px", "cursor": "pointer"}),
        html.Button("Next  ▶", id="btn-next", n_clicks=0,
                    style={"fontSize": "13px", "padding": "5px 16px", "marginLeft": "10px", "cursor": "pointer"}),
        html.Div(
            dcc.Slider(id="step-slider", min=0, max=0, step=1, value=0, marks=None,
                       tooltip={"placement": "bottom", "always_visible": True}),
            style={"flex": "1", "margin": "0 20px"},
        ),
        dcc.Loading(
            html.Div([
                html.Button("⬇ Download Video", id="btn-download-video", n_clicks=0,
                            style={"fontSize": "12px", "padding": "5px 14px", "cursor": "pointer",
                                   "borderRadius": "4px", "border": "1px solid #aaa", "background": "#fff"}),
                html.Span(id="video-status", style={"fontSize": "11px", "color": "#555", "marginLeft": "6px"}),
            ], style={"display": "flex", "alignItems": "center"}),
            type="dot",
        ),
        dcc.Download(id="video-download"),
        # ---- one-field filmstrip export: N frames as columns in a single PNG ----
        html.Div(style={"width": "1px", "background": "#ddd", "alignSelf": "stretch",
                        "margin": "0 12px"}),
        dcc.Dropdown(id="strip-field", value="rho", clearable=False,
                     options=[{"label": f, "value": f} for f in FIELD_NAMES],
                     style={"width": "72px", "fontSize": "12px"}),
        html.Div(dcc.Input(id="strip-start", type="number", value=0, min=0, step=1,
                           style={"width": "56px", "fontSize": "12px"}),
                 title=("first frame (rollout t: 0 = last seed, 1 = first prediction)"
                        if show_refined else "first frame (step index)"),
                 style={"marginLeft": "6px"}),
        html.Div(dcc.Input(id="strip-stride", type="number", value=1, min=1, step=1,
                           style={"width": "48px", "fontSize": "12px"}),
                 title="frame delta (take every k-th frame)", style={"marginLeft": "4px"}),
        html.Div(dcc.Input(id="strip-count", type="number", value=8, min=1, step=1,
                           style={"width": "48px", "fontSize": "12px"}),
                 title="number of frames", style={"marginLeft": "4px"}),
        # Compare mode only: add a third row with the flow-matching refiner output
        # (grid_refined from infer.py --refine).  Hidden in the normal viewer, where
        # there is nothing to refine.
        html.Div(
            dcc.Checklist(id="strip-refined",
                          options=[{"label": " refined", "value": "refined"}], value=[],
                          style={"fontSize": "12px", "whiteSpace": "nowrap"}),
            title="include the refiner output as a third row",
            style={"marginLeft": "8px", "display": "block" if show_refined else "none"},
        ),
        dcc.Loading(
            html.Button("⬇ Frames", id="btn-download-strip", n_clicks=0,
                        title="export the selected frames of one field as columns in a still image",
                        style={"fontSize": "12px", "padding": "5px 12px", "cursor": "pointer",
                               "marginLeft": "6px", "borderRadius": "4px",
                               "border": "1px solid #aaa", "background": "#fff"}),
            type="dot",
        ),
        dcc.Download(id="strip-download"),
    ], style={"display": "flex", "alignItems": "center", "padding": "8px 12px",
              "borderTop": "1px solid #ddd"})


def _keyboard_js(app: dash.Dash) -> None:
    app.clientside_callback(
        """
        function(_n) {
            document.addEventListener('keydown', function(e) {
                var tag = document.activeElement ? document.activeElement.tagName : '';
                var editable = document.activeElement && document.activeElement.isContentEditable;
                if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT' || editable) return;
                if (e.key === 'ArrowLeft')  document.getElementById('btn-prev').click();
                if (e.key === 'ArrowRight') document.getElementById('btn-next').click();
            });
            return true;
        }
        """,
        Output("_key-init", "data"),
        Input("_key-interval", "n_intervals"),
        prevent_initial_call=True,
    )


# ---------------------------------------------------------------------------
# Normal viewer
# ---------------------------------------------------------------------------

def build_app(root_dir: str, sample: Optional[int] = None,
              max_meshes: Optional[int] = None) -> dash.Dash:
    root_abs = os.path.abspath(root_dir)
    run_dirs = find_run_dirs(root_abs)
    n_found  = len(run_dirs)
    run_dirs = subsample_runs(run_dirs, sample, max_meshes)
    mesh_options, mesh_run_options = group_by_mesh_dir(root_abs, run_dirs)
    labels = [run_label(root_abs, d) for d in run_dirs]

    sampled = f" (sampled from {n_found})" if len(run_dirs) != n_found else ""
    print(f"Found {len(run_dirs)} run(s){sampled} across {len(mesh_options)} mesh dir(s).")
    print("Precomputing renderers...")
    renderers: dict[str, MeshRenderer] = build_renderers(run_dirs, RESOLUTION)
    all_files: dict[str, list[str]]    = {d: find_timestep_files(d) for d in run_dirs}
    all_params: list[dict]             = [load_params(d) for d in run_dirs]
    print("Ready.")

    app  = dash.Dash(__name__, title="FVM Viewer")
    opts = mesh_run_options[0]

    plot_area = dcc.Loading(type="circle", color="#4a90d9", children=html.Div(
        [
            html.Div(id="row-label", children="Fields", style=_ROW_LABEL_STYLE),
            *[dcc.Graph(id=f"plot-{i}", config=_GRAPH_CFG) for i in range(4)],
        ],
        style={"display": "grid", "gridTemplateColumns": "1fr 1fr 1fr 1fr", "gap": "4px", "padding": "4px"},
    ))

    app.layout = html.Div([
        html.Div([
            html.Span("FVM Viewer", style={"fontWeight": "600", "fontSize": "16px", "marginRight": "20px"}),
            html.Span(id="header-info", style={"fontSize": "12px", "color": "#555", "fontFamily": "monospace"}),
            html.Button("Show Δ (Delta)", id="view-toggle", n_clicks=0, style=_TOGGLE_STYLE),
        ], style={"padding": "8px 12px", "borderBottom": "1px solid #ddd", "display": "flex", "alignItems": "baseline"}),
        html.Div([
            _sidebar(opts, mesh_options),
            html.Div([plot_area, _nav_bar()],
                     style={"flex": "1", "overflow": "auto", "display": "flex", "flexDirection": "column"}),
        ], style={"display": "flex", "flex": "1", "overflow": "hidden"}),
        dcc.Store(id="state", data={"run_idx": 0, "step_idx": 0}),
        dcc.Store(id="view-mode", data="absolute"),
        dcc.Store(id="_key-init", data=False),
        dcc.Interval(id="_key-interval", interval=300, max_intervals=1),
    ], style={"display": "flex", "flexDirection": "column", "height": "100vh", "fontFamily": "sans-serif"})

    _keyboard_js(app)

    @app.callback(
        Output("view-mode", "data"),
        Output("view-toggle", "children"),
        Input("view-toggle", "n_clicks"),
        State("view-mode", "data"),
        prevent_initial_call=True,
    )
    def toggle_view(_n, current_mode):
        new_mode = "delta" if current_mode == "absolute" else "absolute"
        label = "Show Values" if new_mode == "delta" else "Show Δ (Delta)"
        return new_mode, label

    @app.callback(
        Output("run-selector", "options"),
        Output("run-selector", "value"),
        Input("mesh-selector", "value"),
        State("state", "data"),
    )
    def filter_runs_by_mesh(mesh_idx, state):
        """Narrow the run list to the selected mesh.  Keeps the current run selected when
        it belongs to that mesh, so re-picking the same mesh is a no-op."""
        options = mesh_run_options[mesh_idx or 0]
        current = (state or {}).get("run_idx", 0)
        value   = current if any(o["value"] == current for o in options) else options[0]["value"]
        return options, value

    @app.callback(
        Output("state", "data"),
        Input("btn-prev", "n_clicks"), Input("btn-next", "n_clicks"),
        Input("run-selector", "value"), Input("step-slider", "value"),
        State("state", "data"), prevent_initial_call=True,
    )
    def update_state(_p, _n, run_sel, slider_val, state):
        triggered = callback_context.triggered_id
        state = dict(state)
        if triggered == "run-selector" and run_sel is not None:
            old_t = t_of_file(all_files[run_dirs[state["run_idx"]]][state["step_idx"]])
            state["run_idx"]  = run_sel
            state["step_idx"] = closest_idx(all_files[run_dirs[run_sel]], old_t)
        elif triggered == "btn-prev":
            run = run_dirs[state["run_idx"]]
            state["step_idx"] = (state["step_idx"] - 1) % len(all_files[run])
        elif triggered == "btn-next":
            run = run_dirs[state["run_idx"]]
            state["step_idx"] = (state["step_idx"] + 1) % len(all_files[run])
        elif triggered == "step-slider" and slider_val is not None:
            state["step_idx"] = int(slider_val)
        return state

    @app.callback(
        Output("plot-0", "figure"), Output("plot-1", "figure"),
        Output("plot-2", "figure"), Output("plot-3", "figure"),
        Output("row-label", "children"),
        Output("header-info", "children"),
        Output("step-slider", "max"), Output("step-slider", "value"),
        Input("state", "data"),
        Input("view-mode", "data"),
    )
    def render(state, view_mode):
        run_dir  = run_dirs[state["run_idx"]]
        step_idx = state["step_idx"]
        files    = all_files[run_dir]
        t, cell_prims = load_step(files[step_idx])
        grid = renderers[run_dir].render_cell_smooth(cell_prims).numpy()

        if view_mode == "delta":
            if step_idx > 0:
                _, prev_prims = load_step(files[step_idx - 1])
                prev_grid = renderers[run_dir].render_cell_smooth(prev_prims).numpy()
                display = grid - prev_grid
            else:
                display = np.zeros_like(grid)
            figs      = [make_delta_figure(display[i], f"Δ{FIELD_NAMES[i]}") for i in range(4)]
            row_label = "Delta (current − previous)"
        else:
            figs      = [make_field_figure(grid[i], FIELD_NAMES[i]) for i in range(4)]
            row_label = "Fields"

        n      = len(files)
        header = f"{labels[state['run_idx']]}   |   step {step_idx + 1}/{n}   |   t = {t:.4g}"
        return (*figs, row_label, header, n - 1, step_idx)

    @app.callback(
        Output("video-download", "data"),
        Output("video-status",   "children"),
        Input("btn-download-video", "n_clicks"),
        State("state",     "data"),
        State("view-mode", "data"),
        prevent_initial_call=True,
    )
    def download_video(_n, state, view_mode):
        run_dir = run_dirs[state["run_idx"]]
        files   = all_files[run_dir]
        # mesh-qualified so downloads from different meshes don't collide on disk
        name    = labels[state["run_idx"]].replace(os.sep, "_")
        print(f'Generating video for {name} ({len(files)} frames)...')

        # Pass 1: compute global zranges (scalar min/max only — no grids kept in RAM)
        zranges = None
        if view_mode == 'absolute':
            mins = [float('inf')] * 4
            maxs = [float('-inf')] * 4
            for path in files:
                _, cp = load_step(path)
                g = renderers[run_dir].render_cell_smooth(cp).numpy()
                for i in range(4):
                    mins[i] = min(mins[i], float(g[i].min()))
                    maxs[i] = max(maxs[i], float(g[i].max()))
            zranges = list(zip(mins, maxs))

        # Pass 2: render frames (each ~1.8 MB as PIL — safe to accumulate)
        frames: list[np.ndarray] = []
        prev_grid = None
        for step_idx, path in enumerate(files):
            t, cp = load_step(path)
            grid  = renderers[run_dir].render_cell_smooth(cp).numpy()
            frames.append(_render_frame_rgb(
                rows=[('', grid)],
                view_mode=view_mode,
                prev_rows=[prev_grid] if (view_mode == 'delta' and prev_grid is not None) else None,
                zranges=zranges,
                title=str(step_idx),   # integer step only — no run name / sim time
            ))
            prev_grid = grid

        video_bytes = _encode_apng(frames)
        print(f'APNG ready ({len(video_bytes) // 1024} KB)')
        return dcc.send_bytes(video_bytes, filename=f'{name}.png'), ''

    @app.callback(
        Output("strip-download", "data"),
        Input("btn-download-strip", "n_clicks"),
        State("state", "data"),
        State("strip-field", "value"), State("strip-start", "value"),
        State("strip-stride", "value"), State("strip-count", "value"),
        prevent_initial_call=True,
    )
    def download_strip(_n, state, field, start, stride, count):
        run_dir = run_dirs[state["run_idx"]]
        files = all_files[run_dir]
        idxs = _strip_indices(start, stride, count, len(files))
        if not idxs:
            return dash.no_update
        fi = FIELD_NAMES.index(field)
        grids = []
        for i in idxs:
            _, cp = load_step(files[i])
            grids.append(renderers[run_dir].render_cell_smooth(cp).numpy()[fi])
        name = labels[state["run_idx"]].replace(os.sep, "_")
        png = _strip_png_bytes([grids], field, idxs)
        fname = f"{name}_{field}_s{idxs[0]}_d{max(1, int(stride or 1))}_n{len(idxs)}.png"
        return dcc.send_bytes(png, filename=fname)

    @app.callback(
        Output("params-display", "children"),
        Input("run-selector", "value"),
    )
    def show_params(run_idx):
        return _params_children(all_params[run_idx or 0])

    return app


# ---------------------------------------------------------------------------
# Compare viewer
# ---------------------------------------------------------------------------

def build_compare_app(real_root: str, gen_root: str, sample: Optional[int] = None,
                      max_meshes: Optional[int] = None) -> dash.Dash:
    """
    Side-by-side viewer: real data on top row, generated predictions on bottom row.
    Only runs present in both directories are listed.
    The slider steps through the generated frames; the closest real frame is shown
    alongside each generated frame.
    A toggle switches both rows between absolute values and deltas.
    """
    real_root_abs = os.path.abspath(real_root)
    gen_root_abs  = os.path.abspath(gen_root)
    real_dirs = find_run_dirs(real_root_abs)
    gen_dirs  = find_gen_run_dirs(gen_root_abs)

    # Pair runs on the mesh-qualified relative path (mesh_<uid>/run_<uid>) so multi-mesh
    # datasets stay unambiguous, falling back to the bare run name when the generated
    # tree is flat but the real one is nested (or vice versa).
    matched_by = ""
    run_names: list[str] = []
    real_run_dirs: list[str] = []
    gen_run_dirs: list[str] = []
    for key_fn, what in ((run_label, "relative path"),
                         (lambda _root, d: os.path.basename(d), "run name")):
        real_map = {key_fn(real_root_abs, d): d for d in real_dirs}
        gen_map  = {key_fn(gen_root_abs, d): d for d in gen_dirs}
        common   = sorted(real_map.keys() & gen_map.keys())
        if common:
            run_names     = common
            real_run_dirs = [real_map[n] for n in common]
            gen_run_dirs  = [gen_map[n] for n in common]
            matched_by    = what
            break

    if not run_names:
        raise RuntimeError(
            f"No runs in common between {real_root} and {gen_root}.\n"
            f"  Real: {sorted(run_label(real_root_abs, d) for d in real_dirs)}\n"
            f"  Gen:  {sorted(run_label(gen_root_abs, d) for d in gen_dirs)}"
        )

    # Sample the matched set, not the two sides independently, so pairing is preserved.
    n_matched = len(run_names)
    kept = set(subsample_runs(real_run_dirs, sample, max_meshes))
    if len(kept) != len(real_run_dirs):
        keep_i       = [i for i, d in enumerate(real_run_dirs) if d in kept]
        run_names     = [run_names[i] for i in keep_i]
        real_run_dirs = [real_run_dirs[i] for i in keep_i]
        gen_run_dirs  = [gen_run_dirs[i] for i in keep_i]

    mesh_options, mesh_run_options = group_by_mesh_dir(real_root_abs, real_run_dirs)
    # Mesh-qualified labels (mesh_<uid>/run_<uid>) for the header — the exact relative
    # path of the REAL run, so it can be copied straight into e.g. a report_pipeline
    # config even when the runs were matched by bare name.
    real_labels = [run_label(real_root_abs, d) for d in real_run_dirs]
    sampled = f" (sampled from {n_matched})" if len(run_names) != n_matched else ""
    print(f"Found {len(run_names)} common run(s){sampled} across {len(mesh_options)} "
          f"mesh dir(s), matched by {matched_by}.")

    # Detect render resolution from the first available gen frame; fall back to RESOLUTION
    gen_files_map:  dict[str, list[str]]    = {d: find_timestep_files(d) for d in gen_run_dirs}
    render_res = RESOLUTION
    for _gd in gen_run_dirs:
        _files = gen_files_map.get(_gd, [])
        if _files:
            try:
                _, _grid, _ = load_gen_frame(_files[0])
                render_res = (_grid.shape[1], _grid.shape[2])
                print(f"Detected gen resolution: {render_res[0]}x{render_res[1]}")
            except Exception:
                pass
            break

    print("Precomputing renderers for real data...")
    renderers:      dict[str, MeshRenderer] = build_renderers(real_run_dirs, render_res)
    real_files_map: dict[str, list[str]]    = {d: find_timestep_files(d) for d in real_run_dirs}
    all_params:     list[dict]              = [load_params(d) for d in real_run_dirs]
    print("Ready.")

    # Precompute number of seed frames per gen run (seeds are always at the front)
    n_seed_map: dict[str, int] = {}
    for _gd in gen_run_dirs:
        _n = 0
        for _f in gen_files_map[_gd]:
            _, _, _is_seed = load_gen_frame(_f)
            if _is_seed:
                _n += 1
            else:
                break
        n_seed_map[_gd] = max(_n, 1)

    app  = dash.Dash(__name__, title="FVM Viewer — Compare")
    opts = mesh_run_options[0]

    plot_area = dcc.Loading(type="circle", color="#4a90d9", children=html.Div(
        [
            html.Div(id="row-label-top", children="Real", style=_ROW_LABEL_STYLE),
            *[dcc.Graph(id=f"plot-top-{i}", config=_GRAPH_CFG) for i in range(4)],
            html.Div(id="row-label-bot", children="Generated", style=_ROW_LABEL_STYLE),
            *[dcc.Graph(id=f"plot-bot-{i}", config=_GRAPH_CFG) for i in range(4)],
            *[dcc.Graph(id=f"plot-scale-{i}", config=_GRAPH_CFG,
                        style={"height": "52px"}) for i in range(4)],
            html.Div(id="row-label-refined",
                     children="Detail model (flow matching) output",
                     style=_ROW_LABEL_STYLE),
            *[dcc.Graph(id=f"plot-refined-{i}", config=_GRAPH_CFG) for i in range(4)],
        ],
        style={"display": "grid", "gridTemplateColumns": "1fr 1fr 1fr 1fr",
               "gap": "4px", "padding": "4px"},
    ))

    app.layout = html.Div([
        html.Div([
            html.Span("FVM Viewer — Compare", style={"fontWeight": "600", "fontSize": "16px", "marginRight": "20px"}),
            html.Span(id="header-info", style={"fontSize": "12px", "color": "#555", "fontFamily": "monospace"}),
            html.Button("Show Δ (Delta)", id="view-toggle", n_clicks=0, style=_TOGGLE_STYLE),
        ], style={"padding": "8px 12px", "borderBottom": "1px solid #ddd", "display": "flex", "alignItems": "baseline"}),
        html.Div([
            _sidebar(opts, mesh_options),
            html.Div([plot_area, _nav_bar(show_refined=True)],
                     style={"flex": "1", "overflow": "auto", "display": "flex", "flexDirection": "column"}),
        ], style={"display": "flex", "flex": "1", "overflow": "hidden"}),
        dcc.Store(id="state", data={"run_idx": 0, "step_idx": 0}),
        dcc.Store(id="view-mode", data="absolute"),
        dcc.Store(id="_key-init", data=False),
        dcc.Interval(id="_key-interval", interval=300, max_intervals=1),
    ], style={"display": "flex", "flexDirection": "column", "height": "100vh", "fontFamily": "sans-serif"})

    _keyboard_js(app)

    @app.callback(
        Output("view-mode", "data"),
        Output("view-toggle", "children"),
        Input("view-toggle", "n_clicks"),
        State("view-mode", "data"),
        prevent_initial_call=True,
    )
    def toggle_view(_n, current_mode):
        new_mode = "delta" if current_mode == "absolute" else "absolute"
        label = "Show Values" if new_mode == "delta" else "Show Δ (Delta)"
        return new_mode, label

    @app.callback(
        Output("run-selector", "options"),
        Output("run-selector", "value"),
        Input("mesh-selector", "value"),
        State("state", "data"),
    )
    def filter_runs_by_mesh(mesh_idx, state):
        """Narrow the run list to the selected mesh.  Keeps the current run selected when
        it belongs to that mesh, so re-picking the same mesh is a no-op."""
        options = mesh_run_options[mesh_idx or 0]
        current = (state or {}).get("run_idx", 0)
        value   = current if any(o["value"] == current for o in options) else options[0]["value"]
        return options, value

    @app.callback(
        Output("state", "data"),
        Input("btn-prev", "n_clicks"), Input("btn-next", "n_clicks"),
        Input("run-selector", "value"), Input("step-slider", "value"),
        State("state", "data"), prevent_initial_call=True,
    )
    def update_state(_p, _n, run_sel, slider_val, state):
        triggered = callback_context.triggered_id
        state = dict(state)
        if triggered == "run-selector" and run_sel is not None:
            gen_t = t_of_file(gen_files_map[gen_run_dirs[state["run_idx"]]][state["step_idx"]])
            state["run_idx"]  = run_sel
            state["step_idx"] = closest_idx(gen_files_map[gen_run_dirs[run_sel]], gen_t)
        elif triggered == "btn-prev":
            gen_files = gen_files_map[gen_run_dirs[state["run_idx"]]]
            state["step_idx"] = (state["step_idx"] - 1) % len(gen_files)
        elif triggered == "btn-next":
            gen_files = gen_files_map[gen_run_dirs[state["run_idx"]]]
            state["step_idx"] = (state["step_idx"] + 1) % len(gen_files)
        elif triggered == "step-slider" and slider_val is not None:
            state["step_idx"] = int(slider_val)
        return state

    @app.callback(
        Output("plot-top-0", "figure"), Output("plot-top-1", "figure"),
        Output("plot-top-2", "figure"), Output("plot-top-3", "figure"),
        Output("plot-scale-0", "figure"), Output("plot-scale-1", "figure"),
        Output("plot-scale-2", "figure"), Output("plot-scale-3", "figure"),
        Output("plot-bot-0", "figure"), Output("plot-bot-1", "figure"),
        Output("plot-bot-2", "figure"), Output("plot-bot-3", "figure"),
        Output("plot-refined-0", "figure"), Output("plot-refined-1", "figure"),
        Output("plot-refined-2", "figure"), Output("plot-refined-3", "figure"),
        Output("row-label-top", "children"),
        Output("row-label-bot", "children"),
        Output("row-label-refined", "children"),
        Output("header-info", "children"),
        Output("step-slider", "max"), Output("step-slider", "value"),
        Input("state", "data"),
        Input("view-mode", "data"),
    )
    def render(state, view_mode):
        run_idx  = state["run_idx"]
        step_idx = state["step_idx"]
        real_dir = real_run_dirs[run_idx]
        gen_dir  = gen_run_dirs[run_idx]
        gen_files  = gen_files_map[gen_dir]
        real_files = real_files_map[real_dir]
        params = all_params[run_idx]

        gen_t, gen_grid, is_seed = load_gen_frame(gen_files[step_idx])
        gen_H, gen_W = gen_grid.shape[1], gen_grid.shape[2]
        refined_grid = load_gen_refined(gen_files[step_idx])
        real_idx  = closest_idx(real_files, gen_t)
        real_t, cell_prims = load_step(real_files[real_idx])
        real_grid = renderers[real_dir].render_cell_smooth(cell_prims).numpy()

        n_seed    = n_seed_map[gen_dir]
        relative_t = step_idx - (n_seed - 1)
        frame_tag = "seed" if is_seed else "pred"
        n         = len(gen_files)

        params_str = _params_header_str(params)
        header = f"{real_labels[run_idx]}   |   t = {relative_t}  ({frame_tag})"
        if params_str:
            header += f"   |   {params_str}"

        if view_mode == "delta":
            if step_idx > 0:
                prev_gen_t, prev_gen_grid, _ = load_gen_frame(gen_files[step_idx - 1])
                prev_real_idx = closest_idx(real_files, prev_gen_t)
                _, prev_cell_prims = load_step(real_files[prev_real_idx])
                prev_real_grid = renderers[real_dir].render_cell_smooth(prev_cell_prims).numpy()
                real_delta = real_grid - prev_real_grid
                gen_delta  = gen_grid  - prev_gen_grid
            else:
                real_delta = np.zeros_like(real_grid)
                gen_delta  = np.zeros_like(gen_grid)
            shared_maxabs = [
                float(max(np.abs(real_delta[i]).max(), np.abs(gen_delta[i]).max())) or 1.0
                for i in range(4)
            ]
            top_figs   = [make_delta_figure(real_delta[i], f"Δ{FIELD_NAMES[i]}  real",       shared_maxabs[i]) for i in range(4)]
            scale_figs = [make_colorscale_figure(-shared_maxabs[i], shared_maxabs[i], FIELD_NAMES[i], "RdBu") for i in range(4)]
            bot_figs   = [make_delta_figure(gen_delta[i],  f"Δ{FIELD_NAMES[i]}  {frame_tag}", shared_maxabs[i]) for i in range(4)]
            top_label = "Real Δ (current − previous)"
            bot_label = "Generated Δ (current − previous)"
        else:
            shared_zmin = [float(min(real_grid[i].min(), gen_grid[i].min())) for i in range(4)]
            shared_zmax = [float(max(real_grid[i].max(), gen_grid[i].max())) for i in range(4)]
            top_figs   = [make_field_figure(real_grid[i], f"{FIELD_NAMES[i]}  real",       shared_zmin[i], shared_zmax[i]) for i in range(4)]
            scale_figs = [make_colorscale_figure(shared_zmin[i], shared_zmax[i], FIELD_NAMES[i]) for i in range(4)]
            bot_figs   = [make_field_figure(gen_grid[i],  f"{FIELD_NAMES[i]}  {frame_tag}", shared_zmin[i], shared_zmax[i]) for i in range(4)]
            top_label = "Real"
            bot_label = "Generated"

        # Detail-model row: the increment the flow-matching refiner adds on top of
        # the base prediction (refined - generated).  This is the refiner's actual
        # output; the refined field itself is nearly identical to the generated one,
        # so plotting it absolutely would just duplicate the row above.
        if refined_grid is not None:
            detail = refined_grid - gen_grid
            det_maxabs = [float(np.abs(detail[i]).max()) or 1.0 for i in range(4)]
            refined_figs = [
                make_delta_figure(detail[i], f"detail {FIELD_NAMES[i]}", det_maxabs[i])
                for i in range(4)
            ]
            rms = float(np.sqrt(np.mean(detail ** 2)))
            refined_label = (f"Detail model (flow matching) output   "
                             f"refined - generated,  RMS = {rms:.4g}")
        else:
            why = "seed frame" if is_seed else "run has no refiner output (use infer.py --refine)"
            refined_figs = [make_empty_figure(gen_H, gen_W, f"detail {FIELD_NAMES[i]}")
                            for i in range(4)]
            refined_label = f"Detail model (flow matching) output   [{why}]"

        return (*top_figs, *scale_figs, *bot_figs, *refined_figs,
                top_label, bot_label, refined_label, header, n - 1, step_idx)

    @app.callback(
        Output("video-download", "data"),
        Output("video-status",   "children"),
        Input("btn-download-video", "n_clicks"),
        State("state",     "data"),
        State("view-mode", "data"),
        prevent_initial_call=True,
    )
    def download_video(_n, state, view_mode):
        run_idx    = state["run_idx"]
        real_dir   = real_run_dirs[run_idx]
        gen_dir    = gen_run_dirs[run_idx]
        gen_files  = gen_files_map[gen_dir]
        real_files = real_files_map[real_dir]
        # run_names may be mesh-qualified (mesh_<uid>/run_<uid>) — flatten for the filename
        name       = run_names[run_idx].replace(os.sep, "_")
        print(f'Generating comparison video for {name} ({len(gen_files)} frames)...')

        # Pass 1: scalar min/max only
        zranges = None
        if view_mode == 'absolute':
            mins = [float('inf')] * 4
            maxs = [float('-inf')] * 4
            for gf in gen_files:
                t, gg, _ = load_gen_frame(gf)
                ri = closest_idx(real_files, t)
                _, rcp = load_step(real_files[ri])
                rg = renderers[real_dir].render_cell_smooth(rcp).numpy()
                for i in range(4):
                    mins[i] = min(mins[i], float(gg[i].min()), float(rg[i].min()))
                    maxs[i] = max(maxs[i], float(gg[i].max()), float(rg[i].max()))
            zranges = list(zip(mins, maxs))

        # Pass 2: render frames (each ~1.8 MB as PIL — safe to accumulate)
        frames: list[np.ndarray] = []
        prev_real = prev_gen = None
        # Rollout-relative indexing, same as the page header and the eval logs:
        # last seed frame = t 0, predictions = t 1, 2, ... (earlier seeds negative).
        t_off = n_seed_map[gen_dir] - 1
        for step_idx, gf in enumerate(gen_files):
            t, gg, _ = load_gen_frame(gf)
            ri = closest_idx(real_files, t)
            _, rcp = load_step(real_files[ri])
            rg = renderers[real_dir].render_cell_smooth(rcp).numpy()
            prev_rows = ([prev_real, prev_gen]
                         if (view_mode == 'delta' and prev_real is not None) else None)
            frames.append(_render_frame_rgb(
                rows=[('Real', rg), ('Generated', gg)],
                view_mode=view_mode,
                prev_rows=prev_rows,
                zranges=zranges,
                title=str(step_idx - t_off),   # rollout t only — no run name / sim time
            ))
            prev_real, prev_gen = rg, gg

        video_bytes = _encode_apng(frames)
        print(f'APNG ready ({len(video_bytes) // 1024} KB)')
        return dcc.send_bytes(video_bytes, filename=f'{name}_comparison.png'), ''

    @app.callback(
        Output("strip-download", "data"),
        Input("btn-download-strip", "n_clicks"),
        State("state", "data"),
        State("strip-field", "value"), State("strip-start", "value"),
        State("strip-stride", "value"), State("strip-count", "value"),
        State("strip-refined", "value"),
        prevent_initial_call=True,
    )
    def download_strip(_n, state, field, start, stride, count, refined_opt):
        run_idx    = state["run_idx"]
        gen_files  = gen_files_map[gen_run_dirs[run_idx]]
        real_files = real_files_map[real_run_dirs[run_idx]]
        # The whole strip works in rollout-relative t (last seed = 0, predictions =
        # 1, 2, ... — same as the page header and the eval logs): the start box is a
        # rollout t, and the printed column headers are rollout t.
        t_off = n_seed_map[gen_run_dirs[run_idx]] - 1
        idxs = _strip_indices(max(0, int(start or 0)) + t_off, stride, count, len(gen_files))
        if not idxs:
            return dash.no_update
        step_ids = [i - t_off for i in idxs]
        fi = FIELD_NAMES.index(field)
        include_refined = bool(refined_opt) and "refined" in refined_opt
        real_row, gen_row, detail_row = [], [], []
        for i in idxs:
            t, gg, _ = load_gen_frame(gen_files[i])
            gen_row.append(gg[fi])
            _, rcp = load_step(real_files[closest_idx(real_files, t)])
            real_row.append(renderers[real_run_dirs[run_idx]].render_cell_smooth(rcp).numpy()[fi])
            if include_refined:
                # Same as the live compare page: show the refiner's RESIDUAL
                # (refined - generated), not the refined field itself.  None for
                # seed frames / runs without --refine -> "n/a" cell.
                rg = load_gen_refined(gen_files[i])
                detail_row.append(rg[fi] - gg[fi] if rg is not None else None)
        name = run_names[run_idx].replace(os.sep, "_")
        kwargs = {}
        if include_refined:
            present = [d for d in detail_row if d is not None]
            kwargs = dict(detail_row=detail_row,
                          detail_maxabs=max((float(np.abs(d).max()) for d in present),
                                            default=1.0) or 1.0)
        png = _strip_png_bytes([real_row, gen_row], field, step_ids,
                               row_labels=["Real", "Generated"], **kwargs)
        fname = f"{name}_{field}_s{step_ids[0]}_d{max(1, int(stride or 1))}_n{len(idxs)}.png"
        return dcc.send_bytes(png, filename=fname)

    @app.callback(
        Output("params-display", "children"),
        Input("run-selector", "value"),
    )
    def show_params(run_idx):
        return _params_children(all_params[run_idx or 0])

    return app


# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="View FVM solution timesteps in a browser.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("directory",
                        help="Real data directory (run dir or dataset dir with sub-dirs)")
    parser.add_argument("-c", "--compare", metavar="GEN_DIR", default=None,
                        help="Generated data directory to compare against. "
                             "Only runs present in both directories will be shown.")
    parser.add_argument("-s", "--sample", type=int, default=None, metavar="N",
                        help="Load at most N runs per mesh, spread evenly across each "
                             "mesh's runs. Sampling happens before any data is read, so "
                             "a big sweep opens without loading every run.")
    parser.add_argument("-m", "--meshes", type=int, default=None, metavar="M",
                        help="Load at most M mesh dirs, spread evenly. Renderers are "
                             "built per mesh, so this is the bigger startup saving.")
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--host", default="127.0.0.1",
                        help="127.0.0.1 for SSH port-forwarding, 0.0.0.0 for direct access")
    args = parser.parse_args()

    # Pre-build colormap LUTs on the main thread before Dash starts
    for _cmap in ('viridis', 'RdBu_r'):
        _LUT[_cmap] = _build_lut(_cmap)

    if args.compare:
        app = build_compare_app(args.directory, args.compare, args.sample, args.meshes)
    else:
        app = build_app(args.directory, args.sample, args.meshes)

    print(f"\n  FVM Viewer running at  http://{args.host}:{args.port}/")
    if args.host == "127.0.0.1":
        print(f"  For HPC port-forwarding: ssh -L {args.port}:localhost:{args.port} user@hpc-node")
    print()
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
