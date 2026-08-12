"""
Filmstrip/frame rendering + dataset loading helpers shared by the web viewer and
the report pipeline (scripts/report_pipeline.py).

Lives OUTSIDE viewer.py so the pipeline can produce byte-identical images to the
webserver's downloads without importing dash/plotly.
"""

import os
import sys
import pickle
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'fvm_gen'))
from renderer import MeshRenderer


def _add_solver_to_path() -> None:
    """Put the FVM solver on sys.path so shared_mesh.pkl can be unpickled
    (mirrors run_gen.py; silent if absent — mesh_props.npz datasets need no solver)."""
    default_solver = str(Path(__file__).resolve().parents[2] / 'fvm_solver')
    solver_dir = os.environ.get("FVM_SOLVER_DIR", default_solver)
    if not os.path.isdir(solver_dir):
        return
    for p in (solver_dir, os.path.join(solver_dir, "time_fvm")):
        if p not in sys.path:
            sys.path.insert(0, p)


_add_solver_to_path()


FIELD_NAMES = ["Vx", "Vy", "rho", "T"]


RESOLUTION  = (512, 512)


# Colormap LUTs precomputed once at startup — safe to use from any thread.
def _build_lut(name: str) -> np.ndarray:
    from matplotlib import colormaps
    return (colormaps[name](np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)


_LUT: dict[str, np.ndarray] = {}   # populated lazily on first use, before any Dash callbacks


def _apply_cmap(data: np.ndarray, cmap_name: str,
                vmin: float, vmax: float, out_w: int, out_h: int) -> np.ndarray:
    """Apply a precomputed colormap LUT; returns uint8 RGB [out_h, out_w, 3].
    Pure numpy — safe to call from any thread, no matplotlib in the hot path.
    """
    from PIL import Image
    if cmap_name not in _LUT:
        _LUT[cmap_name] = _build_lut(cmap_name)
    lut  = _LUT[cmap_name]
    norm = np.clip((data.astype(np.float32) - vmin) / max(vmax - vmin, 1e-8), 0, 1)
    idx  = (norm * 255).astype(np.uint8)
    rgb  = lut[idx]                                        # [H, W, 3] uint8
    if rgb.shape[:2] != (out_h, out_w):
        rgb = np.array(Image.fromarray(rgb).resize((out_w, out_h), Image.Resampling.BILINEAR))
    return rgb


def _font(size: int):
    """Scalable font for strip labels: matplotlib's bundled DejaVu Sans (present on
    every install), falling back to PIL's default (sized where Pillow >= 10)."""
    from PIL import ImageFont
    try:
        from matplotlib import font_manager
        return ImageFont.truetype(font_manager.findfont('DejaVu Sans'), size)
    except Exception:
        try:
            return ImageFont.load_default(size=size)
        except TypeError:
            return ImageFont.load_default()


def _render_field_strip(rows: list[list[np.ndarray]], field_name: str,
                        zmin: float, zmax: float, step_ids: list[int],
                        row_labels: Optional[list[str]] = None,
                        detail_row: Optional[list] = None,
                        detail_maxabs: float = 1.0) -> np.ndarray:
    """One-field filmstrip: selected frames as consecutive columns in ONE still image.

    rows: [n_rows][n_cols] of 2D field arrays — a single row in the normal viewer,
    Real/Generated rows in compare mode.  All these cells share one viridis range so
    the columns are directly comparable.  `detail_row` optionally appends the
    refiner's RESIDUAL (refined - generated) — the same thing the live compare page
    shows, since the refined field itself is nearly identical to the generated one —
    on its own symmetric RdBu_r scale with its own colourbar.  A None cell (seed
    frame, no refiner output) renders as a grey "n/a" placeholder.  Column headers
    carry the integer step index.
    """
    from PIL import Image, ImageDraw
    CELL, HEAD_H, BAR_H, LABEL_H, SCALE_H = 256, 46, 42, 40, 26
    # Real (scalable) font: PIL's default bitmap font is ~10 px, unreadable once the
    # strip is scaled to a paper's \linewidth.  DejaVu Sans ships with matplotlib
    # (already a dependency); sized default as fallback.
    fnt_head, fnt_bar, fnt_scale = _font(30), _font(27), _font(24)
    n_cols = len(rows[0])
    with_bars = row_labels is not None
    total_w = n_cols * CELL
    total_h = (HEAD_H + len(rows) * ((BAR_H if with_bars else 0) + CELL)
               + LABEL_H + SCALE_H)
    if detail_row is not None:
        total_h += (BAR_H if with_bars else 0) + CELL + LABEL_H + SCALE_H

    canvas = Image.new('RGB', (total_w, total_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    for ci, sid in enumerate(step_ids):
        draw.text((ci * CELL + CELL // 2, HEAD_H // 2), str(sid), fill=(20, 20, 20),
                  anchor='mm', font=fnt_head)

    def _cells(y, row, cmap, lo, hi):
        for ci, g in enumerate(row):
            if g is None:
                draw.rectangle([ci * CELL, y, (ci + 1) * CELL - 1, y + CELL - 1],
                               fill=(240, 240, 240))
                draw.text((ci * CELL + CELL // 2, y + CELL // 2), "n/a",
                          fill=(150, 150, 150), anchor='mm', font=fnt_head)
            else:
                canvas.paste(Image.fromarray(_apply_cmap(g, cmap, lo, hi, CELL, CELL)),
                             (ci * CELL, y))

    def _bar(y, text):
        draw.rectangle([0, y, total_w, y + BAR_H - 1], fill=(235, 235, 235))
        draw.text((8, y + BAR_H // 2), text, fill=(20, 20, 20), anchor='lm', font=fnt_bar)

    def _scale(y, label, cmap, lo, hi):
        draw.rectangle([0, y, total_w, y + LABEL_H - 1], fill=(255, 255, 255))
        draw.text((8, y + LABEL_H // 2), f"{lo:.2g}", fill=(20, 20, 20),
                  anchor='lm', font=fnt_scale)
        draw.text((total_w // 2, y + LABEL_H // 2), label, fill=(20, 20, 20),
                  anchor='mm', font=fnt_scale)
        draw.text((total_w - 8, y + LABEL_H // 2), f"{hi:.2g}", fill=(20, 20, 20),
                  anchor='rm', font=fnt_scale)
        grad = np.tile(np.linspace(lo, hi, total_w).astype(np.float32), (SCALE_H, 1))
        canvas.paste(Image.fromarray(_apply_cmap(grad, cmap, lo, hi, total_w, SCALE_H)),
                     (0, y + LABEL_H))

    y = HEAD_H
    for ri, row in enumerate(rows):
        if with_bars:
            _bar(y, row_labels[ri]); y += BAR_H
        _cells(y, row, 'viridis', zmin, zmax); y += CELL

    if detail_row is not None:
        if with_bars:
            _bar(y, "Refined residual (refined - generated)"); y += BAR_H
        _cells(y, detail_row, 'RdBu_r', -detail_maxabs, detail_maxabs); y += CELL

    _scale(y, field_name, 'viridis', zmin, zmax); y += LABEL_H + SCALE_H
    if detail_row is not None:
        _scale(y, f"{field_name}  refined - generated", 'RdBu_r',
               -detail_maxabs, detail_maxabs)
    return np.array(canvas)


def _strip_indices(start, stride, count, n_files: int) -> list[int]:
    """Frame indices for the filmstrip: `count` frames from `start`, every `stride`-th,
    clipped to what exists.  Inputs arrive from number boxes, so None/garbage → defaults."""
    start = max(0, int(start or 0))
    stride = max(1, int(stride or 1))
    count = max(1, int(count or 8))    # 8 = the UI's default frame count
    return [i for i in range(start, start + stride * count, stride) if i < n_files]


def _strip_png_bytes(rows, field, step_ids, row_labels=None, **strip_kwargs) -> bytes:
    import io
    from PIL import Image
    flat = [g for row in rows for g in row if g is not None]
    zmin = min(float(g.min()) for g in flat)
    zmax = max(float(g.max()) for g in flat)
    img = _render_field_strip(rows, field, zmin, zmax, step_ids, row_labels, **strip_kwargs)
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format='PNG')
    return buf.getvalue()


def load_mesh(run_dir: str) -> dict:
    d = np.load(os.path.join(run_dir, "mesh_props.npz"), allow_pickle=True)
    return {k: d[k] for k in d.files}


def build_renderer(run_dir: str, resolution: tuple[int, int]) -> MeshRenderer:
    H, W = resolution
    mesh_npz   = os.path.join(run_dir, "mesh_props.npz")
    parent_dir = os.path.dirname(run_dir)
    shared_pkl = os.path.join(parent_dir, "shared_mesh.pkl")

    if os.path.exists(mesh_npz):
        # Old format: per-run mesh stored alongside the data
        cache_path = os.path.join(run_dir, f"renderer_cache_{H}x{W}.pt")
        if os.path.exists(cache_path):
            return MeshRenderer.from_cache(cache_path, device="cpu")
        mesh = load_mesh(run_dir)
        renderer = MeshRenderer(
            vertices=mesh["vertices"], triangles=mesh["triangles"],
            resolution=resolution, device="cpu",
        )
    elif os.path.exists(shared_pkl):
        # New format: shared mesh at dataset level, cache stored there too
        cache_path = os.path.join(parent_dir, f"renderer_cache_{H}x{W}.pt")
        with open(shared_pkl, "rb") as f:
            mesh_dict = pickle.load(f)
        fvm_mesh = mesh_dict["mesh"]
        verts = fvm_mesh.vertices.cpu().numpy()
        n_cells = int(fvm_mesh.cells.shape[0])
        x0, x1 = float(verts[:, 0].min()), float(verts[:, 0].max())
        y0, y1 = float(verts[:, 1].min()), float(verts[:, 1].max())
        if os.path.exists(cache_path):
            _r = MeshRenderer.from_cache(cache_path, device="cpu")
            eps = 1e-3
            if (_r._c2v_tri.max().item() + 1 == n_cells
                    and abs(_r.xlim[0] - x0) < eps and abs(_r.xlim[1] - x1) < eps
                    and abs(_r.ylim[0] - y0) < eps and abs(_r.ylim[1] - y1) < eps):
                return _r
            print(f"  Viewer renderer cache stale, rebuilding...")
            os.unlink(cache_path)
        renderer = MeshRenderer(
            vertices=verts,
            triangles=fvm_mesh.cells.cpu().numpy(),
            resolution=resolution, device="cpu",
        )
    else:
        raise FileNotFoundError(
            f"No mesh found for {run_dir}: checked {mesh_npz} and {shared_pkl}"
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


def load_gen_refined(path: str) -> Optional[np.ndarray]:
    """Optional flow-matching refiner output for a generated frame: grid (4,H,W),
    or None if the frame carries no refined data (seed frames, or an inference run
    without --refine)."""
    d = np.load(path)
    if "grid_refined" not in d.files:
        return None
    return d["grid_refined"].astype(np.float32)


def closest_idx(files: list[str], target_t: float) -> int:
    return min(range(len(files)), key=lambda i: abs(t_of_file(files[i]) - target_t))
