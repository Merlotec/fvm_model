import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from cprint import c_print

# ---- path setup ----
_ROOT       = Path(__file__).resolve().parents[2]
_SOLVER_DIR = _ROOT / 'fvm_solver'
_GEN_DIR    = Path(__file__).resolve().parents[1] / 'fvm_gen'

for _p in (_SOLVER_DIR, _GEN_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from renderer import MeshRenderer

# ---- config ----
with open(Path(__file__).resolve().parent / 'hyperparams.json') as _f:
    _HP = json.load(_f)

DATASET_DIR = _ROOT / 'data' / 'fvm_gen_datasets'
RESOLUTION  = tuple(_HP['resolution'])
PATCH_SIZE  = _HP['patch_size']
EMB_DIM     = _HP['emb_dim']
N_CHANNELS  = _HP['n_channels']
WINDOW_SIZE = _HP['window_size']
NUM_LAYERS  = _HP['num_layers']
FIRST_FRAME = _HP['first_frame']

DELTA_STATS_PATH = Path(__file__).resolve().parent / 'delta_stats.json'
INPUT_STATS_PATH = Path(__file__).resolve().parent / 'input_stats.json'
PIXEL_MASK_PATH  = Path(__file__).resolve().parent / 'pixel_mask.pt'


# ---- renderer ----

def build_renderer(dataset_dir: Path, resolution: tuple[int, int], device: str) -> MeshRenderer:
    """Load or build a MeshRenderer from the shared mesh, caching to disk."""
    H, W = resolution
    cache_path = dataset_dir / f'renderer_cache_{H}x{W}.pt'

    if cache_path.exists():
        c_print(f'Loading renderer cache from {cache_path}', color='green')
        return MeshRenderer.from_cache(str(cache_path), device=device)

    mesh_pkl = dataset_dir / 'shared_mesh.pkl'
    if not mesh_pkl.exists():
        raise FileNotFoundError(f'shared_mesh.pkl not found in {dataset_dir}')

    with open(mesh_pkl, 'rb') as f:
        mesh_dict = pickle.load(f)
    fvm_mesh = mesh_dict['mesh']

    vertices  = fvm_mesh.vertices.cpu().numpy()
    triangles = fvm_mesh.triangles.cpu().numpy()

    c_print('Building renderer (trifinder precomputation)...', color='yellow')
    renderer = MeshRenderer(vertices, triangles, resolution=resolution, device=device)
    renderer.save_cache(str(cache_path))
    c_print(f'Renderer cache saved to {cache_path}', color='green')
    return renderer


# ---- dataset statistics ----

def compute_delta_stats(sim_dirs: list[Path], renderer: MeshRenderer, n_samples: int = 200):
    """Sample consecutive frame pairs to estimate per-channel delta mean and std."""
    all_pairs = []
    for d in sim_dirs:
        files = sorted(
            [f for f in d.iterdir() if f.name.startswith('t_') and f.name.endswith('.npz')],
            key=lambda f: float(f.stem[2:]),
        )
        for i in range(len(files) - 1):
            all_pairs.append((files[i], files[i + 1]))

    n       = min(n_samples, len(all_pairs))
    indices = torch.randperm(len(all_pairs))[:n].tolist()

    def _render(p):
        d    = np.load(p)
        vals = d['cell_primatives'].astype(np.float32) * d['prim_std'] + d['prim_mean']
        return renderer.render_cell_smooth(vals)

    deltas = torch.stack([_render(all_pairs[i][1]) - _render(all_pairs[i][0]) for i in indices])
    mask   = torch.isfinite(deltas)
    safe   = deltas.nan_to_num(0.0)
    count  = mask.sum(dim=(0, 2, 3)).float()
    mean   = (safe * mask).sum(dim=(0, 2, 3)) / count
    var    = ((safe - mean.view(1, -1, 1, 1)) ** 2 * mask).sum(dim=(0, 2, 3)) / count
    std    = var.sqrt().clamp(min=1e-6)
    return mean, std


def compute_input_stats(sim_dirs: list[Path], renderer: MeshRenderer, n_samples: int = 200):
    """Sample individual frames to estimate per-channel input mean and std."""
    all_files = []
    for d in sim_dirs:
        all_files.extend(sorted(
            [f for f in d.iterdir() if f.name.startswith('t_') and f.name.endswith('.npz')]
        ))

    n       = min(n_samples, len(all_files))
    indices = torch.randperm(len(all_files))[:n].tolist()

    sum1  = torch.zeros(N_CHANNELS)
    sum2  = torch.zeros(N_CHANNELS)
    count = torch.zeros(N_CHANNELS)

    for i in indices:
        d        = np.load(all_files[i])
        vals     = d['cell_primatives'].astype(np.float32) * d['prim_std'] + d['prim_mean']
        rendered = renderer.render_cell_smooth(vals)
        for c in range(N_CHANNELS):
            pixels    = rendered[c]
            finite    = pixels[torch.isfinite(pixels)]
            sum1[c]  += finite.sum()
            sum2[c]  += (finite ** 2).sum()
            count[c] += finite.numel()

    mean = sum1 / count
    std  = ((sum2 / count) - mean ** 2).clamp(min=0).sqrt().clamp(min=1e-6)
    return mean, std


def build_pixel_mask(renderer: MeshRenderer, resolution: tuple[int, int]) -> torch.Tensor:
    """Boolean (1, 1, H, W) mask — True for pixels inside the fluid mesh, False elsewhere."""
    H, W = resolution
    mask = torch.zeros(H * W, dtype=torch.bool)
    mask[renderer._interior_idx] = True
    return mask.view(1, 1, H, W)


# ---- diagnostics ----

def print_histogram(values: torch.Tensor, title: str, bins: int = 25, width: int = 50) -> None:
    n = values.numel()
    if n == 0:
        return
    v     = values.float().cpu()
    max_q = 1 << 24
    v_q   = v[torch.randperm(n)[:max_q]] if n > max_q else v
    lo    = torch.quantile(v_q, 0.01).item()
    hi    = torch.quantile(v_q, 0.99).item()
    if lo >= hi:
        hi = lo + 1e-6
    counts = torch.histc(v, bins=bins, min=lo, max=hi)
    edges  = torch.linspace(lo, hi, bins + 1).tolist()
    peak   = counts.max().item()
    print(f'\n{title}  n={n:,}  mean={v.mean():.4f}  std={v.std():.4f}  '
          f'p1={lo:.4f}  p99={hi:.4f}')
    for i in range(bins):
        bar = '█' * int(counts[i].item() / peak * width if peak > 0 else 0)
        print(f'  [{edges[i]:+8.4f}, {edges[i+1]:+8.4f})  {bar}')
    print()
