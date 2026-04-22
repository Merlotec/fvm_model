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

# Make fvm_gen importable so we can use MeshRenderer
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'fvm_gen'))
from renderer import MeshRenderer

FIELD_NAMES = ["Vx", "Vy", "rho", "T"]
RESOLUTION  = (512, 512)   # higher than training res for display quality


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def find_run_dirs(root_dir: str) -> list[str]:
    """
    Return sorted list of run directories.
    If root_dir itself is a run (contains mesh_props.npz) return [root_dir].
    Otherwise return all immediate subdirectories that are runs.
    """
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


def load_mesh(run_dir: str) -> dict:
    d = np.load(os.path.join(run_dir, "mesh_props.npz"), allow_pickle=True)
    return {k: d[k] for k in d.files}


def build_renderer(run_dir: str, resolution: tuple[int, int]) -> MeshRenderer:
    """Load or build a MeshRenderer for this run, caching alongside mesh_props.npz."""
    H, W = resolution
    cache_path = os.path.join(run_dir, f"renderer_cache_{H}x{W}.pt")

    if os.path.exists(cache_path):
        return MeshRenderer.from_cache(cache_path, device="cpu")

    mesh = load_mesh(run_dir)
    renderer = MeshRenderer(
        vertices  = mesh["vertices"],
        triangles = mesh["triangles"],
        resolution = resolution,
        device    = "cpu",
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
    d = np.load(path)
    prim_mean = d["prim_mean"]
    prim_std  = d["prim_std"]
    cell_prims = d["cell_primatives"].astype(np.float32) * prim_std + prim_mean
    return float(d["t"]), cell_prims


def closest_idx(files: list[str], target_t: float) -> int:
    return min(range(len(files)), key=lambda i: abs(t_of_file(files[i]) - target_t))


# ---------------------------------------------------------------------------
# Plot construction
# ---------------------------------------------------------------------------

def make_field_figure(grid: np.ndarray, title: str) -> go.Figure:
    """
    Display one (H, W) rendered field channel as a heatmap.

    grid: (H, W) float32 array from MeshRenderer.
    """
    fig = go.Figure(go.Heatmap(
        z          = grid,
        colorscale = "Viridis",
        showscale  = True,
        colorbar   = dict(thickness=12, len=0.85),
    ))
    fig.update_layout(
        title  = dict(text=title, font=dict(size=13), x=0.5, xanchor="center"),
        xaxis  = dict(visible=False, scaleanchor="y"),
        yaxis  = dict(visible=False, autorange="reversed"),
        margin = dict(l=0, r=0, t=36, b=0),
        height = 320,
        uirevision = title,
    )
    return fig


# ---------------------------------------------------------------------------
# Dash app
# ---------------------------------------------------------------------------

def build_app(root_dir: str) -> dash.Dash:
    run_dirs = find_run_dirs(os.path.abspath(root_dir))

    print("Precomputing renderers...")
    renderers:  dict[str, MeshRenderer] = {d: build_renderer(d, RESOLUTION) for d in run_dirs}
    all_files:  dict[str, list[str]]    = {d: find_timestep_files(d)         for d in run_dirs}
    print("Ready.")

    app = dash.Dash(__name__, title="FVM Viewer")

    run_options = [
        {"label": os.path.basename(d), "value": i}
        for i, d in enumerate(run_dirs)
    ]

    sidebar = html.Div([
        html.H4("Runs", style={"margin": "0 0 10px 0", "fontSize": "13px", "fontWeight": "600"}),
        dcc.RadioItems(
            id="run-selector",
            options=run_options,
            value=0,
            labelStyle={
                "display": "block",
                "fontSize": "11px",
                "padding": "3px 0",
                "cursor": "pointer",
                "wordBreak": "break-all",
            },
        ),
    ], style={
        "width": "190px",
        "flexShrink": "0",
        "padding": "12px 10px",
        "borderRight": "1px solid #ddd",
        "overflowY": "auto",
        "fontFamily": "monospace",
    })

    plot_grid = dcc.Loading(
        type="circle",
        color="#4a90d9",
        children=html.Div(
            [dcc.Graph(id=f"plot-{i}", config={
                "scrollZoom": True,
                "displayModeBar": True,
                "displaylogo": False,
            }) for i in range(4)],
            style={
                "display": "grid",
                "gridTemplateColumns": "1fr 1fr",
                "gap": "4px",
                "padding": "4px",
            },
        ),
    )

    nav_bar = html.Div([
        html.Button("◀  Prev", id="btn-prev", n_clicks=0,
                    style={"fontSize": "13px", "padding": "5px 16px", "cursor": "pointer"}),
        html.Button("Next  ▶", id="btn-next", n_clicks=0,
                    style={"fontSize": "13px", "padding": "5px 16px", "marginLeft": "10px", "cursor": "pointer"}),
        html.Div(
            dcc.Slider(
                id="step-slider",
                min=0, max=0, step=1, value=0,
                marks=None,
                tooltip={"placement": "bottom", "always_visible": True},
            ),
            style={"flex": "1", "margin": "0 20px"},
        ),
    ], style={"display": "flex", "alignItems": "center", "padding": "8px 12px", "borderTop": "1px solid #ddd"})

    header = html.Div([
        html.Span("FVM Viewer", style={"fontWeight": "600", "fontSize": "16px", "marginRight": "20px"}),
        html.Span(id="header-info", style={"fontSize": "12px", "color": "#555", "fontFamily": "monospace"}),
    ], style={"padding": "8px 12px", "borderBottom": "1px solid #ddd", "display": "flex", "alignItems": "baseline"})

    app.layout = html.Div([
        header,
        html.Div([
            sidebar,
            html.Div([plot_grid, nav_bar], style={"flex": "1", "overflow": "auto", "display": "flex", "flexDirection": "column"}),
        ], style={"display": "flex", "flex": "1", "overflow": "hidden"}),
        dcc.Store(id="state", data={"run_idx": 0, "step_idx": 0}),
        dcc.Store(id="_key-init", data=False),
        dcc.Interval(id="_key-interval", interval=300, max_intervals=1),
    ], style={"display": "flex", "flexDirection": "column", "height": "100vh", "fontFamily": "sans-serif"})

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

    @app.callback(
        Output("state", "data"),
        Input("btn-prev", "n_clicks"),
        Input("btn-next", "n_clicks"),
        Input("run-selector", "value"),
        Input("step-slider", "value"),
        State("state", "data"),
        prevent_initial_call=True,
    )
    def update_state(_n_prev, _n_next, run_sel, slider_val, state):
        triggered = callback_context.triggered_id
        state = dict(state)

        if triggered == "run-selector" and run_sel is not None:
            old_run = run_dirs[state["run_idx"]]
            old_t   = t_of_file(all_files[old_run][state["step_idx"]]) if all_files[old_run] else 0.0
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
        Output("plot-0", "figure"),
        Output("plot-1", "figure"),
        Output("plot-2", "figure"),
        Output("plot-3", "figure"),
        Output("header-info", "children"),
        Output("step-slider", "max"),
        Output("step-slider", "value"),
        Input("state", "data"),
    )
    def render(state):
        run_idx  = state["run_idx"]
        step_idx = state["step_idx"]
        run_dir  = run_dirs[run_idx]
        files    = all_files[run_dir]
        renderer = renderers[run_dir]

        t, cell_prims = load_step(files[step_idx])
        grid = renderer.render_cell_smooth(cell_prims).numpy()  # (4, H, W)

        n        = len(files)
        run_name = os.path.basename(run_dir)
        header   = f"{run_name}   |   step {step_idx + 1}/{n}   |   t = {t:.4g}"

        figs = [make_field_figure(grid[i], FIELD_NAMES[i]) for i in range(4)]
        return (*figs, header, n - 1, step_idx)

    return app


# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="View FVM solution timesteps in a browser.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "directory",
        help="Run directory (contains mesh_props.npz) or dataset directory (contains run sub-dirs)",
    )
    parser.add_argument("--port", type=int, default=8050, help="Port to listen on (default: 8050)")
    parser.add_argument(
        "--host", default="127.0.0.1",
        help="Bind address. Use 127.0.0.1 (default) for SSH port-forwarding, "
             "or 0.0.0.0 to allow direct network access.",
    )
    args = parser.parse_args()

    app = build_app(args.directory)

    print(f"\n  FVM Viewer running at  http://{args.host}:{args.port}/")
    if args.host == "127.0.0.1":
        print(f"  For HPC port-forwarding: ssh -L {args.port}:localhost:{args.port} user@hpc-node")
    print()

    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
