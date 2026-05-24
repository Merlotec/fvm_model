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
    Use Prev / Next buttons or the slider to move between timesteps.
    Use the "Show Δ / Show Values" toggle to switch between absolute values and deltas.
    Plotly figures support scroll-to-zoom and drag-to-pan.
"""

import os
import sys
import tempfile
import argparse
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import dash
from dash import dcc, html, Input, Output, State, callback_context

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'fvm_gen'))
from renderer import MeshRenderer

FIELD_NAMES = ["Vx", "Vy", "rho", "T"]
RESOLUTION  = (512, 512)


# ---------------------------------------------------------------------------
# Video export helpers
# ---------------------------------------------------------------------------

def _apply_cmap(data: np.ndarray, cmap_name: str,
                vmin: float, vmax: float, out_w: int, out_h: int) -> np.ndarray:
    """Apply a colormap to a 2D array; returns uint8 RGB [out_h, out_w, 3]."""
    from matplotlib import colormaps
    from PIL import Image
    norm = np.clip((data.astype(np.float32) - vmin) / max(vmax - vmin, 1e-8), 0, 1)
    rgb  = (colormaps[cmap_name](norm)[:, :, :3] * 255).astype(np.uint8)
    if rgb.shape[:2] != (out_h, out_w):
        rgb = np.array(Image.fromarray(rgb).resize((out_w, out_h), Image.BILINEAR))
    return rgb


def _render_frame_rgb(
    rows: list[tuple[str, np.ndarray]],
    view_mode: str,
    prev_rows: list[np.ndarray] | None = None,
    zranges: list[tuple[float, float]] | None = None,
    title: str = '',
) -> np.ndarray:
    """Render one video frame as RGB [H, W, 3] using PIL — no matplotlib figure."""
    from PIL import Image, ImageDraw

    CELL_W, CELL_H = 256, 256
    BAR_H   = 22
    TITLE_H = 28 if title else 0
    total_w = 4 * CELL_W
    total_h = TITLE_H + len(rows) * (BAR_H + CELL_H)

    canvas = Image.new('RGB', (total_w, total_h), (24, 24, 24))
    draw   = ImageDraw.Draw(canvas)

    if title:
        draw.text((total_w // 2, TITLE_H // 2), title, fill=(210, 210, 210), anchor='mm')

    for ri, (label, grid) in enumerate(rows):
        prev  = prev_rows[ri] if prev_rows else None
        y_bar = TITLE_H + ri * (BAR_H + CELL_H)
        y_img = y_bar + BAR_H
        draw.rectangle([0, y_bar, total_w, y_bar + BAR_H - 1], fill=(45, 45, 45))
        draw.text((6, y_bar + BAR_H // 2), label, fill=(180, 180, 220), anchor='lm')

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
            draw.text((x0 + 4, y_img + 4), FIELD_NAMES[ci], fill=(255, 255, 255))

    return np.array(canvas)


class _CV2Writer:
    """cv2.VideoWriter wrapper with lazy init (dimensions known at first frame)."""

    def __init__(self, path: str, fps: int):
        import cv2
        self._cv2    = cv2
        self._path   = path
        self._fps    = fps
        self._writer = None
        self._wh: tuple[int, int] | None = None

    def append_data(self, frame: np.ndarray) -> None:
        h, w = frame.shape[:2]
        if h % 2: h -= 1
        if w % 2: w -= 1
        if self._writer is None:
            self._wh     = (w, h)
            self._writer = self._cv2.VideoWriter(
                self._path, self._cv2.VideoWriter_fourcc(*'mp4v'), self._fps, (w, h)
            )
        frame = frame[:h, :w]
        self._writer.write(self._cv2.cvtColor(frame, self._cv2.COLOR_RGB2BGR))

    def close(self) -> None:
        if self._writer:
            self._writer.release()


def _open_video_writer(tmp_path: str, fps: int = 8):
    """Return a writer that supports .append_data(frame) and .close().

    Tries cv2 first (C extension, no fork), then imageio-ffmpeg.
    """
    try:
        import cv2  # noqa: F401
        return _CV2Writer(tmp_path, fps)
    except ImportError:
        pass
    try:
        import imageio
        return imageio.get_writer(
            tmp_path, fps=fps, codec='libx264',
            output_params=['-crf', '23', '-pix_fmt', 'yuv420p'],
        )
    except Exception as e:
        raise RuntimeError(
            f'No video encoder available ({e}).\n'
            'Install opencv-python:  pip install opencv-python-headless'
        )


def _append_frame(writer, frame: np.ndarray) -> None:
    writer.append_data(frame)


# ---------------------------------------------------------------------------
# Data helpers — real data
# ---------------------------------------------------------------------------

def find_run_dirs(root_dir: str) -> list[str]:
    if os.path.exists(os.path.join(root_dir, "mesh_props.npz")):
        return [root_dir]
    runs = sorted(
        os.path.join(root_dir, name)
        for name in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, name))
        and os.path.exists(os.path.join(root_dir, name, "mesh_props.npz"))
    )
    if not runs:
        raise RuntimeError(f"No run directories found under {root_dir}")
    return runs


def find_gen_run_dirs(root_dir: str) -> list[str]:
    """Find generated-data run dirs (no mesh_props.npz required; just t_*.npz files)."""
    def _has_frames(d: str) -> bool:
        return any(f.startswith("t_") and f.endswith(".npz") for f in os.listdir(d))

    if _has_frames(root_dir):
        return [root_dir]
    return sorted(
        os.path.join(root_dir, name)
        for name in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, name))
        and _has_frames(os.path.join(root_dir, name))
    )


def load_mesh(run_dir: str) -> dict:
    d = np.load(os.path.join(run_dir, "mesh_props.npz"), allow_pickle=True)
    return {k: d[k] for k in d.files}


def build_renderer(run_dir: str, resolution: tuple[int, int]) -> MeshRenderer:
    H, W = resolution
    cache_path = os.path.join(run_dir, f"renderer_cache_{H}x{W}.pt")
    if os.path.exists(cache_path):
        return MeshRenderer.from_cache(cache_path, device="cpu")
    mesh = load_mesh(run_dir)
    renderer = MeshRenderer(
        vertices   = mesh["vertices"],
        triangles  = mesh["triangles"],
        resolution = resolution,
        device     = "cpu",
    )
    renderer.save_cache(cache_path)
    return renderer


def find_timestep_files(run_dir: str) -> list[str]:
    files = [f for f in os.listdir(run_dir) if f.startswith("t_") and f.endswith(".npz")]
    files.sort(key=lambda f: float(f[2:-4]))
    return [os.path.join(run_dir, f) for f in files]


def t_of_file(path: str) -> float:
    return float(os.path.basename(path)[2:-4])


def load_step(path: str) -> tuple[float, np.ndarray]:
    """Load a raw FVM timestep file; returns (t, cell_primatives) denormalised."""
    d = np.load(path)
    return float(d["t"]), d["cell_primatives"].astype(np.float32) * d["prim_std"] + d["prim_mean"]


def load_gen_frame(path: str) -> tuple[float, np.ndarray, bool]:
    """Load a generated frame; returns (t, grid (4,H,W), is_seed)."""
    d = np.load(path)
    return float(d["t"]), d["grid"].astype(np.float32), bool(d["is_seed"])


def closest_idx(files: list[str], target_t: float) -> int:
    return min(range(len(files)), key=lambda i: abs(t_of_file(files[i]) - target_t))


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
                      zmin: float | None = None, zmax: float | None = None) -> go.Figure:
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


def make_delta_figure(delta: np.ndarray, title: str, maxabs: float | None = None) -> go.Figure:
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


def _sidebar(run_options: list[dict]) -> html.Div:
    return html.Div([
        html.H4("Runs", style={"margin": "0 0 10px 0", "fontSize": "13px", "fontWeight": "600"}),
        dcc.RadioItems(
            id="run-selector", options=run_options, value=0,
            labelStyle={"display": "block", "fontSize": "11px",
                        "padding": "3px 0", "cursor": "pointer", "wordBreak": "break-all"},
        ),
    ], style={"width": "190px", "flexShrink": "0", "padding": "12px 10px",
              "borderRight": "1px solid #ddd", "overflowY": "auto", "fontFamily": "monospace"})


def _nav_bar() -> html.Div:
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

def build_app(root_dir: str) -> dash.Dash:
    run_dirs = find_run_dirs(os.path.abspath(root_dir))

    print("Precomputing renderers...")
    renderers: dict[str, MeshRenderer] = {d: build_renderer(d, RESOLUTION) for d in run_dirs}
    all_files: dict[str, list[str]]    = {d: find_timestep_files(d) for d in run_dirs}
    print("Ready.")

    app  = dash.Dash(__name__, title="FVM Viewer")
    opts = [{"label": os.path.basename(d), "value": i} for i, d in enumerate(run_dirs)]

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
            _sidebar(opts),
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
        header = f"{os.path.basename(run_dir)}   |   step {step_idx + 1}/{n}   |   t = {t:.4g}"
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
        name    = os.path.basename(run_dir)
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

        # Pass 2: render and write one frame at a time
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
            tmp = f.name
        try:
            writer   = _open_video_writer(tmp)
            prev_grid = None
            for path in files:
                t, cp = load_step(path)
                grid  = renderers[run_dir].render_cell_smooth(cp).numpy()
                frame = _render_frame_rgb(
                    rows=[('', grid)],
                    view_mode=view_mode,
                    prev_rows=[prev_grid] if (view_mode == 'delta' and prev_grid is not None) else None,
                    zranges=zranges,
                    title=f'{name}   t = {t:.4g}',
                )
                _append_frame(writer, frame)
                prev_grid = grid
            writer.close()
            with open(tmp, 'rb') as f:
                video_bytes = f.read()
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

        print(f'Video ready ({len(video_bytes) // 1024} KB)')
        return dcc.send_bytes(video_bytes, filename=f'{name}.mp4'), ''

    return app


# ---------------------------------------------------------------------------
# Compare viewer
# ---------------------------------------------------------------------------

def build_compare_app(real_root: str, gen_root: str) -> dash.Dash:
    """
    Side-by-side viewer: real data on top row, generated predictions on bottom row.
    Only runs present in both directories are listed.
    The slider steps through the generated frames; the closest real frame is shown
    alongside each generated frame.
    A toggle switches both rows between absolute values and deltas.
    """
    real_dirs = find_run_dirs(os.path.abspath(real_root))
    gen_dirs  = find_gen_run_dirs(os.path.abspath(gen_root))

    real_by_name = {os.path.basename(d): d for d in real_dirs}
    gen_by_name  = {os.path.basename(d): d for d in gen_dirs}
    common_names = sorted(real_by_name.keys() & gen_by_name.keys())

    if not common_names:
        raise RuntimeError(
            f"No run names in common between {real_root} and {gen_root}.\n"
            f"  Real: {sorted(real_by_name)}\n"
            f"  Gen:  {sorted(gen_by_name)}"
        )

    run_names     = common_names
    real_run_dirs = [real_by_name[n] for n in run_names]
    gen_run_dirs  = [gen_by_name[n]  for n in run_names]

    print(f"Found {len(run_names)} common run(s): {run_names}")

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
    renderers:      dict[str, MeshRenderer] = {d: build_renderer(d, render_res) for d in real_run_dirs}
    real_files_map: dict[str, list[str]]    = {d: find_timestep_files(d) for d in real_run_dirs}
    print("Ready.")

    app  = dash.Dash(__name__, title="FVM Viewer — Compare")
    opts = [{"label": name, "value": i} for i, name in enumerate(run_names)]

    plot_area = dcc.Loading(type="circle", color="#4a90d9", children=html.Div(
        [
            html.Div(id="row-label-top", children="Real", style=_ROW_LABEL_STYLE),
            *[dcc.Graph(id=f"plot-top-{i}", config=_GRAPH_CFG) for i in range(4)],
            html.Div(id="row-label-bot", children="Generated", style=_ROW_LABEL_STYLE),
            *[dcc.Graph(id=f"plot-bot-{i}", config=_GRAPH_CFG) for i in range(4)],
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
            _sidebar(opts),
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
        Output("plot-bot-0", "figure"), Output("plot-bot-1", "figure"),
        Output("plot-bot-2", "figure"), Output("plot-bot-3", "figure"),
        Output("row-label-top", "children"),
        Output("row-label-bot", "children"),
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

        gen_t, gen_grid, is_seed = load_gen_frame(gen_files[step_idx])
        real_idx  = closest_idx(real_files, gen_t)
        real_t, cell_prims = load_step(real_files[real_idx])
        real_grid = renderers[real_dir].render_cell_smooth(cell_prims).numpy()

        frame_tag = "seed" if is_seed else "pred"
        n      = len(gen_files)
        header = (f"{run_names[run_idx]}   |   step {step_idx + 1}/{n}   |   "
                  f"t = {gen_t:.4g} ({frame_tag})   |   real t = {real_t:.4g}")

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
            top_figs  = [make_delta_figure(real_delta[i], f"Δ{FIELD_NAMES[i]}  real",       shared_maxabs[i]) for i in range(4)]
            bot_figs  = [make_delta_figure(gen_delta[i],  f"Δ{FIELD_NAMES[i]}  {frame_tag}", shared_maxabs[i]) for i in range(4)]
            top_label = "Real Δ (current − previous)"
            bot_label = "Generated Δ (current − previous)"
        else:
            shared_zmin = [float(min(real_grid[i].min(), gen_grid[i].min())) for i in range(4)]
            shared_zmax = [float(max(real_grid[i].max(), gen_grid[i].max())) for i in range(4)]
            top_figs  = [make_field_figure(real_grid[i], f"{FIELD_NAMES[i]}  real",       shared_zmin[i], shared_zmax[i]) for i in range(4)]
            bot_figs  = [make_field_figure(gen_grid[i],  f"{FIELD_NAMES[i]}  {frame_tag}", shared_zmin[i], shared_zmax[i]) for i in range(4)]
            top_label = "Real"
            bot_label = "Generated"

        return (*top_figs, *bot_figs, top_label, bot_label, header, n - 1, step_idx)

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
        name       = run_names[run_idx]
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

        # Pass 2: render and write one frame at a time
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
            tmp = f.name
        try:
            writer    = _open_video_writer(tmp)
            prev_real = None
            prev_gen  = None
            for gf in gen_files:
                t, gg, _ = load_gen_frame(gf)
                ri = closest_idx(real_files, t)
                _, rcp = load_step(real_files[ri])
                rg = renderers[real_dir].render_cell_smooth(rcp).numpy()
                prev_rows = ([prev_real, prev_gen]
                             if (view_mode == 'delta' and prev_real is not None) else None)
                frame = _render_frame_rgb(
                    rows=[('Real', rg), ('Generated', gg)],
                    view_mode=view_mode,
                    prev_rows=prev_rows,
                    zranges=zranges,
                    title=f'{name}   t = {t:.4g}',
                )
                _append_frame(writer, frame)
                prev_real, prev_gen = rg, gg
            writer.close()
            with open(tmp, 'rb') as f:
                video_bytes = f.read()
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

        print(f'Video ready ({len(video_bytes) // 1024} KB)')
        return dcc.send_bytes(video_bytes, filename=f'{name}_comparison.mp4'), ''

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
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--host", default="127.0.0.1",
                        help="127.0.0.1 for SSH port-forwarding, 0.0.0.0 for direct access")
    args = parser.parse_args()

    if args.compare:
        app = build_compare_app(args.directory, args.compare)
    else:
        app = build_app(args.directory)

    print(f"\n  FVM Viewer running at  http://{args.host}:{args.port}/")
    if args.host == "127.0.0.1":
        print(f"  For HPC port-forwarding: ssh -L {args.port}:localhost:{args.port} user@hpc-node")
    print()
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
