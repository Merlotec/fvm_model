from pathlib import Path

import numpy as np
import torch
import matplotlib.tri as mtri


class MeshRenderer:
    """
    Rasterizes a fixed triangular mesh to a dense pixel grid.

    Precomputation (triangle lookup + barycentric weights) runs once on CPU
    using matplotlib's fast C trifinder.  All subsequent render() calls are
    pure PyTorch tensor ops that run on the specified device — typically GPU —
    making repeated rendering across timesteps very fast.

    Args:
        vertices:   (N, 2) float vertex coordinates.
        triangles:  (M, 3) int triangle connectivity.
        resolution: (H, W) output image size.  Default matches ViT-224 decoder.
        xlim:       (x_min, x_max); defaults to vertex extents.
        ylim:       (y_min, y_max); defaults to vertex extents.
        device:     torch device for stored tensors and render output.
    """

    def __init__(
        self,
        vertices: np.ndarray,
        triangles: np.ndarray,
        resolution: tuple[int, int] = (224, 224),
        xlim: tuple[float, float] | None = None,
        ylim: tuple[float, float] | None = None,
        device: torch.device | str = "cpu",
    ):
        vertices  = np.asarray(vertices,  dtype=np.float64)
        triangles = np.asarray(triangles, dtype=np.int64)
        H, W = resolution
        self.resolution = resolution
        self._device = torch.device(device)

        x0, x1 = xlim if xlim is not None else (float(vertices[:, 0].min()), float(vertices[:, 0].max()))
        y0, y1 = ylim if ylim is not None else (float(vertices[:, 1].min()), float(vertices[:, 1].max()))
        self.xlim = (x0, x1)
        self.ylim = (y0, y1)

        # Pixel-centre coordinates in domain space
        xs = np.linspace(x0, x1, W)
        ys = np.linspace(y0, y1, H)
        gx, gy = np.meshgrid(xs, ys)
        flat_x = gx.ravel().astype(np.float64)  # (H*W,)
        flat_y = gy.ravel().astype(np.float64)

        # One-time CPU triangle lookup via matplotlib's fast C trifinder.
        # Returns triangle index per pixel, -1 for pixels outside the mesh.
        triang    = mtri.Triangulation(vertices[:, 0], vertices[:, 1], triangles)
        trifinder = triang.get_trifinder()
        tri_idx   = np.asarray(trifinder(flat_x, flat_y), dtype=np.int64)  # (H*W,)

        interior     = tri_idx >= 0
        interior_idx = np.where(interior)[0].astype(np.int64)  # (P,) pixel positions in flat grid
        tri_interior = tri_idx[interior]                        # (P,) triangle index per interior pixel
        tv           = triangles[tri_interior]                  # (P, 3) vertex indices

        # Barycentric weights for interior pixels
        p  = np.stack([flat_x[interior], flat_y[interior]], axis=1)  # (P, 2)
        v0 = vertices[tv[:, 0]]
        v1 = vertices[tv[:, 1]]
        v2 = vertices[tv[:, 2]]

        d1    = v1 - v0
        d2    = v2 - v0
        dp    = p  - v0
        denom = d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]
        b1    = (dp[:, 0] * d2[:, 1] - dp[:, 1] * d2[:, 0]) / denom
        b2    = (d1[:, 0] * dp[:, 1]  - d1[:, 1] * dp[:, 0]) / denom
        b0    = 1.0 - b1 - b2

        # Cell-to-vertex averaging weights for render_cell_smooth().
        # For each (triangle, corner) pair we need the triangle index and vertex
        # index so we can scatter-add cell values into vertex accumulators.
        M        = triangles.shape[0]
        N_verts  = vertices.shape[0]
        # tri_rep[k] = which triangle the k-th (tri,corner) entry belongs to
        tri_rep  = np.repeat(np.arange(M, dtype=np.int64), 3)   # (3M,)
        vtx_flat = triangles.reshape(-1).astype(np.int64)        # (3M,)
        vtx_deg  = np.bincount(vtx_flat, minlength=N_verts).astype(np.float32)  # (N_verts,)

        # Store everything on device — render() never touches CPU again
        dev = self._device
        self._interior_idx = torch.from_numpy(interior_idx).to(dev)                       # (P,)
        self._tri_idx      = torch.from_numpy(tri_interior).to(dev)                       # (P,)
        self._tri_verts    = torch.from_numpy(tv).to(dev)                                 # (P, 3)
        self._bary         = torch.from_numpy(
            np.stack([b0, b1, b2], axis=1).astype(np.float32)
        ).to(dev)                                                                          # (P, 3)
        self._c2v_tri      = torch.from_numpy(tri_rep).to(dev)                            # (3M,)
        self._c2v_vtx      = torch.from_numpy(vtx_flat).to(dev)                           # (3M,)
        self._vtx_deg      = torch.from_numpy(vtx_deg).to(dev)                            # (N_verts,)
        self._n_verts      = int(N_verts)
        self._n_pixels     = H * W
        self._shape        = (H, W)

        # Preallocate output buffers — reused across render calls to avoid
        # repeated allocation.  C dimension is fixed at construction for the
        # common case of 4-channel primitives; caller can pass a different C
        # and the buffer will be reallocated once then cached.
        self._out_C: int = 0
        self._out_buf: torch.Tensor | None = None

    def render(
        self,
        vertex_values: torch.Tensor | np.ndarray,
        fill: float = 0.0,
    ) -> torch.Tensor:
        """
        Bilinear (barycentric) interpolation of per-vertex values onto the grid.

        Args:
            vertex_values: (N,) or (N, C) values at mesh vertices.
            fill:          value for pixels outside the mesh domain.

        Returns:
            (H, W) for scalar input, or (C, H, W) for multi-channel input,
            as a float32 tensor on the renderer's device.
        """
        vals = _to_float32(vertex_values, self._device)
        scalar = vals.ndim == 1
        if scalar:
            vals = vals.unsqueeze(1)
        C = vals.shape[1]

        # Single gather: (P, 3, C), then weighted sum over corners → (P, C)
        corner_vals = vals[self._tri_verts]              # one kernel, not three
        interp = (self._bary.unsqueeze(-1) * corner_vals).sum(dim=1)  # (P, C)

        if C != self._out_C or self._out_buf is None:
            self._out_buf = torch.empty(self._n_pixels, C, dtype=torch.float32, device=self._device)
            self._out_C = C
        self._out_buf.fill_(fill)
        self._out_buf[self._interior_idx] = interp
        out = self._out_buf.view(self._shape[0], self._shape[1], C)

        return out[:, :, 0] if scalar else out.permute(2, 0, 1)  # (H,W) or (C,H,W)

    def render_cell(
        self,
        cell_values: torch.Tensor | np.ndarray,
        fill: float = 0.0,
    ) -> torch.Tensor:
        """
        Flat-shaded render of per-cell (centroid-based) values.

        Each pixel takes the value of the triangle that contains it; no
        interpolation is performed across cell boundaries.

        Args:
            cell_values: (M,) or (M, C) values indexed by triangle index.
            fill:        value for pixels outside the mesh domain.

        Returns:
            (H, W) or (C, H, W) float32 tensor on the renderer's device.
        """
        vals = _to_float32(cell_values, self._device)
        scalar = vals.ndim == 1
        if scalar:
            vals = vals.unsqueeze(1)
        C = vals.shape[1]

        interp = vals[self._tri_idx]   # (P, C)

        out = torch.full((self._n_pixels, C), fill, dtype=torch.float32, device=self._device)
        out[self._interior_idx] = interp
        out = out.view(self._shape[0], self._shape[1], C)

        return out[:, :, 0] if scalar else out.permute(2, 0, 1)

    def render_cell_smooth(
        self,
        cell_values: torch.Tensor | np.ndarray,
        fill: float = 0.0,
    ) -> torch.Tensor:
        """
        Linearly interpolated render of per-cell values.

        Cell values are first averaged to vertices (each vertex gets the mean
        of all triangles that share it), then rendered with barycentric
        interpolation via render().  Produces a smooth result with no visible
        cell boundaries.

        Args:
            cell_values: (M,) or (M, C) values indexed by triangle index.
            fill:        value for pixels outside the mesh domain.

        Returns:
            (H, W) or (C, H, W) float32 tensor on the renderer's device.
        """
        vals = _to_float32(cell_values, self._device)
        scalar = vals.ndim == 1
        if scalar:
            vals = vals.unsqueeze(1)
        C = vals.shape[1]

        # Scatter-add cell values into per-vertex accumulators, then normalise
        vtx_sum = torch.zeros(self._n_verts, C, dtype=torch.float32, device=self._device)
        vtx_sum.scatter_add_(
            0,
            self._c2v_vtx.unsqueeze(1).expand(-1, C),
            vals[self._c2v_tri],
        )
        vtx_vals = vtx_sum / self._vtx_deg.unsqueeze(1)   # (N_verts, C)

        if scalar:
            return self.render(vtx_vals[:, 0], fill=fill)
        return self.render(vtx_vals, fill=fill)

    def save_cache(self, path: str) -> None:
        """
        Write precomputed tensors to disk so __init__ can be skipped on future
        runs with the same mesh and resolution.

        Load back with MeshRenderer.from_cache().
        """
        torch.save({
            "interior_idx": self._interior_idx.cpu(),
            "tri_idx":      self._tri_idx.cpu(),
            "tri_verts":    self._tri_verts.cpu(),
            "bary":         self._bary.cpu(),
            "c2v_tri":      self._c2v_tri.cpu(),
            "c2v_vtx":      self._c2v_vtx.cpu(),
            "vtx_deg":      self._vtx_deg.cpu(),
            "n_verts":      self._n_verts,
            "n_pixels":     self._n_pixels,
            "shape":        self._shape,
            "resolution":   self.resolution,
            "xlim":         self.xlim,
            "ylim":         self.ylim,
        }, path)

    @classmethod
    def from_cache(
        cls,
        path: str,
        device: torch.device | str = "cpu",
    ) -> "MeshRenderer":
        """
        Restore a MeshRenderer from a cache file, bypassing the trifinder
        precomputation entirely.
        """
        dev  = torch.device(device)
        data = torch.load(path, map_location=dev, weights_only=True)
        obj  = cls.__new__(cls)
        obj._interior_idx = data["interior_idx"]
        obj._tri_idx      = data["tri_idx"]
        obj._tri_verts    = data["tri_verts"]
        obj._bary         = data["bary"]
        obj._c2v_tri      = data["c2v_tri"]
        obj._c2v_vtx      = data["c2v_vtx"]
        obj._vtx_deg      = data["vtx_deg"]
        obj._n_verts      = int(data["n_verts"])
        obj._n_pixels     = data["n_pixels"]
        obj._shape        = tuple(data["shape"])
        obj.resolution    = tuple(data["resolution"])
        obj.xlim          = tuple(data["xlim"])
        obj.ylim          = tuple(data["ylim"])
        obj._device       = dev
        obj._out_C        = 0
        obj._out_buf      = None
        return obj

    def to(self, device: torch.device | str) -> "MeshRenderer":
        """Move precomputed tensors to a new device in-place, returning self."""
        device = torch.device(device)
        self._interior_idx = self._interior_idx.to(device)
        self._tri_idx      = self._tri_idx.to(device)
        self._tri_verts    = self._tri_verts.to(device)
        self._bary         = self._bary.to(device)
        self._c2v_tri      = self._c2v_tri.to(device)
        self._c2v_vtx      = self._c2v_vtx.to(device)
        self._vtx_deg      = self._vtx_deg.to(device)
        self._device       = device
        return self

def _to_float32(x: torch.Tensor | np.ndarray, device: torch.device) -> torch.Tensor:
    if isinstance(x, np.ndarray):
        # as_tensor avoids a copy when array is already float32 C-contiguous
        return torch.as_tensor(x, dtype=torch.float32).to(device)
    return x.to(dtype=torch.float32, device=device)


def render_mesh_to_grid(
    vertices: np.ndarray,
    triangles: np.ndarray,
    values: np.ndarray | torch.Tensor,
    resolution: tuple[int, int] = (224, 224),
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    fill: float = 0.0,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """
    One-shot render of per-vertex values onto a fixed pixel grid.

    Use MeshRenderer directly when rendering multiple snapshots on the same
    mesh to amortise the setup cost.

    Returns:
        (H, W) or (C, H, W) float32 tensor.
    """
    return MeshRenderer(vertices, triangles, resolution, xlim, ylim, device).render(values, fill)


def render_from_files(
    mesh_path: str | Path,
    values_path: str | Path,
    cache_path: str | Path,
    resolution: tuple[int, int] = (224, 224),
    write_cache: bool = False,
    device: torch.device | str = "cpu",
    fill: float = 0.0,
) -> torch.Tensor:
    """
    Load a mesh and solution snapshot from disk, render to a pixel grid.

    Checks cache_path for a precomputed renderer before running the trifinder.
    If write_cache=True and the cache does not exist, writes one after building.

    Args:
        mesh_path:   Path to mesh_props.npz (must contain 'vertices' and
                     'triangles' arrays).
        values_path: Path to a timestep file.  Supported formats:
                       - .npz with 'cell_primatives', 'prim_mean', 'prim_std'
                         (raw FVM save — cell-centred, auto-denormalised).
                       - .npz with a 'Us' key (vertex-based values).
        cache_path:  Exact path to the .pt renderer cache file.
        resolution:  (H, W) output grid size.
        write_cache: Write the cache file if it does not already exist.
        device:      Torch device for the renderer and output tensor.
        fill:        Fill value for pixels outside the mesh domain.

    Returns:
        (C, H, W) float32 tensor on the specified device.
    """
    cache_path = Path(cache_path)

    if cache_path.exists():
        renderer = MeshRenderer.from_cache(str(cache_path), device=device)
    else:
        mesh = np.load(mesh_path)
        renderer = MeshRenderer(
            vertices=mesh["vertices"],
            triangles=mesh["triangles"],
            resolution=resolution,
            device=device,
        )
        if write_cache:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            renderer.save_cache(str(cache_path))

    data = np.load(values_path)
    if "cell_primatives" in data:
        values = data["cell_primatives"].astype(np.float32) * data["prim_std"] + data["prim_mean"]
        return renderer.render_cell(values, fill=fill)
    elif "Us" in data:
        return renderer.render(data["Us"], fill=fill)
    else:
        raise ValueError(
            f"{values_path} contains neither 'cell_primatives' nor 'Us'. "
            f"Keys found: {list(data.keys())}"
        )
