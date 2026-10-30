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

# Human-readable names for display.  Two forms because the destinations differ:
# the image is drawn by PIL (plain text, so only glyphs DejaVu actually has), the
# caption is LaTeX (real maths).  FIELD_NAMES stays the key everywhere else --
# config values, filenames, channel order -- so nothing downstream shifts.
FIELD_LABELS = {
    "Vx":  "x velocity (Vx)",
    "Vy":  "y velocity (Vy)",
    "rho": "density (\u03c1)",
    "T":   "temperature (T)",
}
FIELD_TEX = {
    "Vx":  r"$x$ velocity ($v_x$)",
    "Vy":  r"$y$ velocity ($v_y$)",
    "rho": r"density ($\rho$)",
    "T":   r"temperature ($T$)",
}


def field_label(f: str) -> str:
    """Plain-text display name for the image (falls back to the raw key)."""
    return FIELD_LABELS.get(f, f)


def field_tex(f: str) -> str:
    """LaTeX display name for captions (falls back to \texttt{key})."""
    return FIELD_TEX.get(f, rf"\texttt{{{f}}}")


RESOLUTION  = (512, 512)

# Fill for pixels outside the fluid mesh (see _render_field_strip's value_mask).
# Mid grey: no viridis or RdBu_r entry is close to neutral, so solid geometry
# cannot be mistaken for a field value at either end of the scale.
_MASK_RGB = (145, 145, 145)


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
                        detail_maxabs: float = 1.0,
                        context_row: Optional[list] = None,
                        context_label: str = '',
                        context_rows: Optional[list] = None,
                        value_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """One-field filmstrip: selected frames as consecutive columns in ONE still image.

    rows: [n_rows][n_cols] of 2D field arrays — a single row in the normal viewer,
    Real/Generated rows in compare mode.  All these cells share one viridis range so
    the columns are directly comparable.  `detail_row` optionally appends the
    refiner's RESIDUAL (refined - generated) — the same thing the live compare page
    shows, since the refined field itself is nearly identical to the generated one —
    on its own symmetric RdBu_r scale with its own colourbar.  A None cell (seed
    frame, no refiner output) renders as a grey "n/a" placeholder.  Column headers
    carry the integer step index.

    `context_row` prepends the model's INPUT frames above everything else, cut off
    by a heavy dark rule: those frames precede t=0 and run most-recent-first, so
    their columns do NOT line up with the rollout columns below and the divider has
    to say so.  Their colour range is the rollout's, so enabling the block never
    restyles the rows beneath it.

    `context_rows` is the multi-row form: [(label, row), ...] stacked above the same
    heavy rule.  Two of them is the context-comparison figure -- the run's own
    history and the one actually fed to the encoder -- which only reads if both are
    on screen together.  `context_row`/`context_label` remain the single-row
    shortcut and are folded into the same list.

    `value_mask` (H, W bool, True inside the fluid) marks pixels outside the mesh.
    They hold the renderer's 0 fill rather than a field value, so they are painted
    a flat neutral grey instead of being colour-mapped -- see _strip_png_bytes,
    which also excludes them from the range.
    """
    from PIL import Image, ImageDraw
    CELL, HEAD_H, BAR_H, LABEL_H, SCALE_H = 256, 46, 42, 40, 26
    SEP_H = 14                      # heavy rule under the context block
    # Real (scalable) font: PIL's default bitmap font is ~10 px, unreadable once the
    # strip is scaled to a paper's \linewidth.  DejaVu Sans ships with matplotlib
    # (already a dependency); sized default as fallback.
    fnt_head, fnt_bar, fnt_scale = _font(30), _font(27), _font(24)

    def _fit(draw_obj, text, max_w, start=27):
        """Largest font (down to 14px) whose `text` fits `max_w`.

        Row labels are prose and the strip can be as narrow as two columns, so a
        fixed size silently runs off the canvas -- which is how the context label
        lost its second half."""
        for sz in range(start, 13, -3):
            f = _font(sz)
            if draw_obj.textlength(text, font=f) <= max_w:
                return f
        return _font(14)
    n_cols = len(rows[0])
    with_bars = row_labels is not None
    total_w = n_cols * CELL
    total_h = (HEAD_H + len(rows) * ((BAR_H if with_bars else 0) + CELL)
               + LABEL_H + SCALE_H)
    if detail_row is not None:
        total_h += (BAR_H if with_bars else 0) + CELL + LABEL_H + SCALE_H
    ctx_blocks = list(context_rows) if context_rows else []
    if context_row is not None:
        ctx_blocks.insert(0, (context_label, context_row))
    if ctx_blocks:
        total_h += len(ctx_blocks) * (BAR_H + CELL) + SEP_H

    canvas = Image.new('RGB', (total_w, total_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    def _header(y):
        """Column numbers, titled.  Drawn directly above the rollout rows (i.e.
        BELOW any context block) because they index the rollout columns only --
        the context frames above them are not on this numbering."""
        # Label and numbers share one font size so the row reads as a single line.
        draw.text((8, y + HEAD_H // 2), 'Frame:', fill=(90, 90, 90),
                  anchor='lm', font=fnt_scale)
        for ci, sid in enumerate(step_ids):
            draw.text((ci * CELL + CELL // 2, y + HEAD_H // 2), str(sid),
                      fill=(20, 20, 20), anchor='mm', font=fnt_scale)

    def _cells(y, row, cmap, lo, hi, placeholder="n/a"):
        for ci, g in enumerate(row):
            if g is None:
                draw.rectangle([ci * CELL, y, (ci + 1) * CELL - 1, y + CELL - 1],
                               fill=(240, 240, 240))
                draw.text((ci * CELL + CELL // 2, y + CELL // 2), placeholder,
                          fill=(150, 150, 150), anchor='mm', font=fnt_head)
            else:
                rgb = _apply_cmap(g, cmap, lo, hi, CELL, CELL)
                if value_mask is not None and value_mask.shape == g.shape:
                    # Solid geometry: outside the mesh the array holds the
                    # renderer's 0 fill, which after masking the range usually
                    # sits below zmin and would clip to the colormap's end --
                    # indistinguishable from a real extreme.  Flat grey instead.
                    solid = np.array(Image.fromarray(
                        (~value_mask).astype(np.uint8) * 255).resize(
                            (CELL, CELL), Image.Resampling.NEAREST)) > 127
                    rgb[solid] = _MASK_RGB
                canvas.paste(Image.fromarray(rgb), (ci * CELL, y))

    def _bar(y, text, fill=(235, 235, 235)):
        draw.rectangle([0, y, total_w, y + BAR_H - 1], fill=fill)
        draw.text((8, y + BAR_H // 2), text, fill=(20, 20, 20), anchor='lm',
                  font=_fit(draw, text, total_w - 16))

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

    y = 0
    if ctx_blocks:
        # The model's INPUT frames, above the rollout and separated by a heavy
        # rule: they precede t=0 and are ordered most-recent-first, so their
        # columns do NOT correspond to the columns below.  Rendered on the SAME
        # colour range as the rollout (identical field and units) but the range
        # itself is computed from the rollout rows only, so switching this block
        # on never changes how the rows below look.
        for lab, row in ctx_blocks:
            _bar(y, lab, fill=(205, 205, 205))
            y += BAR_H
            _cells(y, row, 'viridis', zmin, zmax, placeholder="")
            y += CELL
        # ONE rule under the whole block: it separates context from rollout, and
        # drawing it per row would read as "these are different sections".
        draw.rectangle([0, y, total_w, y + SEP_H - 1], fill=(45, 45, 45))
        y += SEP_H
    _header(y)
    y += HEAD_H

    for ri, row in enumerate(rows):
        if with_bars:
            _bar(y, row_labels[ri]); y += BAR_H
        _cells(y, row, 'viridis', zmin, zmax); y += CELL

    if detail_row is not None:
        if with_bars:
            _bar(y, "Detail refiner residual"); y += BAR_H
        _cells(y, detail_row, 'RdBu_r', -detail_maxabs, detail_maxabs); y += CELL

    disp = field_label(field_name)
    _scale(y, disp, 'viridis', zmin, zmax); y += LABEL_H + SCALE_H
    if detail_row is not None:
        _scale(y, f"{disp}  refined - generated", 'RdBu_r',
               -detail_maxabs, detail_maxabs)
    return np.array(canvas)


def _strip_indices(start, stride, count, n_files: int) -> list[int]:
    """Frame indices for the filmstrip: `count` frames from `start`, every `stride`-th,
    clipped to what exists.  Inputs arrive from number boxes, so None/garbage → defaults."""
    start = max(0, int(start or 0))
    stride = max(1, int(stride or 1))
    count = max(1, int(count or 8))    # 8 = the UI's default frame count
    return [i for i in range(start, start + stride * count, stride) if i < n_files]


def fluid_mask(renderer, shape) -> Optional[np.ndarray]:
    """(H, W) bool, True inside the fluid mesh, from the renderer's interior index.

    This is the same source hfm's build_pixel_mask uses.  Deriving the mask from
    the VALUES instead (nonzero / finite) would misclassify any fluid cell that is
    legitimately zero, which velocity components hit routinely.  Returns None when
    the renderer exposes no interior index, so callers fall back to their previous
    whole-array behaviour rather than failing.
    """
    idx = getattr(renderer, "_interior_idx", None)
    if idx is None:
        return None
    h, w = shape[-2], shape[-1]
    flat = np.zeros(h * w, dtype=bool)
    flat[np.asarray(idx.cpu() if hasattr(idx, "cpu") else idx)] = True
    return flat.reshape(h, w)


def _strip_png_bytes(rows, field, step_ids, row_labels=None,
                     value_mask: Optional[np.ndarray] = None,
                     **strip_kwargs) -> bytes:
    """value_mask: (H, W) bool, True inside the fluid.

    The renderer fills everything outside the mesh with 0 and the model multiplies
    its output by the same mask, so those pixels are a constant 0 that has nothing
    to do with the field.  Including them in the colour range pins one end at 0,
    which for strictly-positive fields (rho, T, whose fluid values sit nowhere
    near 0) squeezes the entire flow into a sliver of the colormap.  With a mask
    the range comes from fluid pixels only and the masked ones are painted a
    neutral grey instead of being clipped to an end of the scale, where they would
    read as a genuine extreme value.
    """
    import io
    from PIL import Image
    # Range from the ROLLOUT rows only: an added context block must not restyle
    # the comparison beneath it.
    flat = [g for row in rows for g in row if g is not None]   # ROLLOUT rows only
    if value_mask is not None:
        vals = [g[value_mask] for g in flat if g.shape == value_mask.shape]
        vals = [v for v in vals if v.size]
        if not vals:                      # shape mismatch: fall back, don't crash
            value_mask, vals = None, [g.ravel() for g in flat]
    else:
        vals = [g.ravel() for g in flat]
    zmin = min(float(v.min()) for v in vals)
    zmax = max(float(v.max()) for v in vals)
    img = _render_field_strip(rows, field, zmin, zmax, step_ids, row_labels,
                              value_mask=value_mask, **strip_kwargs)
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
