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
    Plotly figures support scroll-to-zoom and drag-to-pan.
"""

import os
import sys
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


def make_field_figure(grid: np.ndarray, title: str) -> go.Figure:
    fig = go.Figure(go.Heatmap(
        z=grid, colorscale="Viridis", showscale=True,
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
        [dcc.Graph(id=f"plot-{i}", config=_GRAPH_CFG) for i in range(4)],
        style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "4px", "padding": "4px"},
    ))

    app.layout = html.Div([
        html.Div([
            html.Span("FVM Viewer", style={"fontWeight": "600", "fontSize": "16px", "marginRight": "20px"}),
            html.Span(id="header-info", style={"fontSize": "12px", "color": "#555", "fontFamily": "monospace"}),
        ], style={"padding": "8px 12px", "borderBottom": "1px solid #ddd", "display": "flex", "alignItems": "baseline"}),
        html.Div([
            _sidebar(opts),
            html.Div([plot_area, _nav_bar()],
                     style={"flex": "1", "overflow": "auto", "display": "flex", "flexDirection": "column"}),
        ], style={"display": "flex", "flex": "1", "overflow": "hidden"}),
        dcc.Store(id="state", data={"run_idx": 0, "step_idx": 0}),
        dcc.Store(id="_key-init", data=False),
        dcc.Interval(id="_key-interval", interval=300, max_intervals=1),
    ], style={"display": "flex", "flexDirection": "column", "height": "100vh", "fontFamily": "sans-serif"})

    _keyboard_js(app)

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
        Output("header-info", "children"),
        Output("step-slider", "max"), Output("step-slider", "value"),
        Input("state", "data"),
    )
    def render(state):
        run_dir  = run_dirs[state["run_idx"]]
        step_idx = state["step_idx"]
        files    = all_files[run_dir]
        t, cell_prims = load_step(files[step_idx])
        grid = renderers[run_dir].render_cell_smooth(cell_prims).numpy()
        n    = len(files)
        header = f"{os.path.basename(run_dir)}   |   step {step_idx + 1}/{n}   |   t = {t:.4g}"
        figs   = [make_field_figure(grid[i], FIELD_NAMES[i]) for i in range(4)]
        return (*figs, header, n - 1, step_idx)

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

    run_names  = common_names
    real_run_dirs = [real_by_name[n] for n in run_names]
    gen_run_dirs  = [gen_by_name[n]  for n in run_names]

    print(f"Found {len(run_names)} common run(s): {run_names}")
    print("Precomputing renderers for real data...")
    renderers:      dict[str, MeshRenderer] = {d: build_renderer(d, RESOLUTION) for d in real_run_dirs}
    real_files_map: dict[str, list[str]]    = {d: find_timestep_files(d) for d in real_run_dirs}
    gen_files_map:  dict[str, list[str]]    = {d: find_timestep_files(d) for d in gen_run_dirs}
    print("Ready.")

    app  = dash.Dash(__name__, title="FVM Viewer — Compare")
    opts = [{"label": name, "value": i} for i, name in enumerate(run_names)]

    # 8 plots: top row = real (4), bottom row = generated (4)
    row_label = lambda text: html.Div(text, style={
        "gridColumn": "1 / -1", "fontWeight": "600", "fontSize": "12px",
        "padding": "4px 8px", "background": "#f0f0f0", "borderRadius": "3px",
    })
    plot_area = dcc.Loading(type="circle", color="#4a90d9", children=html.Div(
        [
            row_label("Real"),
            *[dcc.Graph(id=f"plot-real-{i}", config=_GRAPH_CFG) for i in range(4)],
            row_label("Generated"),
            *[dcc.Graph(id=f"plot-gen-{i}",  config=_GRAPH_CFG) for i in range(4)],
        ],
        style={"display": "grid", "gridTemplateColumns": "1fr 1fr 1fr 1fr",
               "gap": "4px", "padding": "4px"},
    ))

    app.layout = html.Div([
        html.Div([
            html.Span("FVM Viewer — Compare", style={"fontWeight": "600", "fontSize": "16px", "marginRight": "20px"}),
            html.Span(id="header-info", style={"fontSize": "12px", "color": "#555", "fontFamily": "monospace"}),
        ], style={"padding": "8px 12px", "borderBottom": "1px solid #ddd", "display": "flex", "alignItems": "baseline"}),
        html.Div([
            _sidebar(opts),
            html.Div([plot_area, _nav_bar()],
                     style={"flex": "1", "overflow": "auto", "display": "flex", "flexDirection": "column"}),
        ], style={"display": "flex", "flex": "1", "overflow": "hidden"}),
        dcc.Store(id="state", data={"run_idx": 0, "step_idx": 0}),
        dcc.Store(id="_key-init", data=False),
        dcc.Interval(id="_key-interval", interval=300, max_intervals=1),
    ], style={"display": "flex", "flexDirection": "column", "height": "100vh", "fontFamily": "sans-serif"})

    _keyboard_js(app)

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
        Output("plot-real-0", "figure"), Output("plot-real-1", "figure"),
        Output("plot-real-2", "figure"), Output("plot-real-3", "figure"),
        Output("plot-gen-0",  "figure"), Output("plot-gen-1",  "figure"),
        Output("plot-gen-2",  "figure"), Output("plot-gen-3",  "figure"),
        Output("header-info", "children"),
        Output("step-slider", "max"), Output("step-slider", "value"),
        Input("state", "data"),
    )
    def render(state):
        run_idx  = state["run_idx"]
        step_idx = state["step_idx"]
        real_dir = real_run_dirs[run_idx]
        gen_dir  = gen_run_dirs[run_idx]
        gen_files  = gen_files_map[gen_dir]
        real_files = real_files_map[real_dir]

        # Generated frame
        gen_t, gen_grid, is_seed = load_gen_frame(gen_files[step_idx])

        # Closest real frame by timestamp
        real_idx  = closest_idx(real_files, gen_t)
        real_t, cell_prims = load_step(real_files[real_idx])
        real_grid = renderers[real_dir].render_cell_smooth(cell_prims).numpy()

        frame_tag = "seed" if is_seed else "pred"
        n      = len(gen_files)
        header = (f"{run_names[run_idx]}   |   step {step_idx + 1}/{n}   |   "
                  f"t = {gen_t:.4g} ({frame_tag})   |   real t = {real_t:.4g}")

        real_figs = [make_field_figure(real_grid[i], f"{FIELD_NAMES[i]}  real") for i in range(4)]
        gen_figs  = [make_field_figure(gen_grid[i],  f"{FIELD_NAMES[i]}  {frame_tag}") for i in range(4)]
        return (*real_figs, *gen_figs, header, n - 1, step_idx)

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
