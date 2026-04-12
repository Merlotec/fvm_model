"""
FVM solution viewer.

Usage:
    # Point at a single run directory:
    python viewer.py path/to/mu_b_1.0000e-3/

    # Point at a dataset directory (contains multiple run subdirectories):
    python viewer.py path/to/fvm_gen_datasets/

Navigation:
    Left / Right arrow keys  — previous / next timestep within current run
    Up   / Down  arrow keys  — switch to run above / below at the nearest t
    Click a run name in the left panel to jump to it
    Prev / Next buttons      — same as Left / Right
"""

import os
import argparse

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.widgets import Button
from mpl_toolkits.axes_grid1 import make_axes_locatable


FIELD_NAMES = ["Vx", "Vy", "rho", "T"]


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
    """Index of the file whose t is closest to target_t."""
    return min(range(len(files)), key=lambda i: abs(t_of_file(files[i]) - target_t))


# ---------------------------------------------------------------------------
# Viewer
# ---------------------------------------------------------------------------

class Viewer:
    def __init__(self, root_dir: str):
        self.run_dirs = find_run_dirs(os.path.abspath(root_dir))
        self.run_idx  = 0
        self.step_idx = 0
        self._load_run()
        self._build_figure()
        self._render()

    # ---- run / timestep loading --------------------------------------------

    def _load_run(self):
        run_dir = self.run_dirs[self.run_idx]
        mesh = load_mesh(run_dir)
        verts = mesh["vertices"]
        tris  = mesh["triangles"]
        self.triang = mtri.Triangulation(verts[:, 0], verts[:, 1], tris)
        self.files  = find_timestep_files(run_dir)
        if not self.files:
            raise RuntimeError(f"No t_*.npz files in {run_dir}")

    def _switch_run(self, new_idx: int, target_t: float | None = None):
        if new_idx < 0 or new_idx >= len(self.run_dirs):
            return
        if target_t is None:
            target_t = t_of_file(self.files[self.step_idx])
        self.run_idx = new_idx
        self._load_run()
        self.step_idx = closest_idx(self.files, target_t)
        self._rebuild_plots()
        self._refresh_run_list()
        self._render()

    # ---- figure construction -----------------------------------------------

    def _build_figure(self):
        self.fig = plt.figure(figsize=(17, 9))

        # columns: [run list | field plot 1 | field plot 2]
        # rows:    [top plots | bottom plots | buttons]
        gs = self.fig.add_gridspec(
            3, 3,
            width_ratios=[1, 2.2, 2.2],
            height_ratios=[1, 1, 0.1],
            hspace=0.45, wspace=0.4,
            left=0.04, right=0.97, top=0.93, bottom=0.04,
        )

        # -- run list panel (spans both plot rows) --
        self.ax_runs = self.fig.add_subplot(gs[:2, 0])
        self.ax_runs.set_navigate(False)

        # -- 4 field subplots --
        self.axes = [
            self.fig.add_subplot(gs[0, 1]),
            self.fig.add_subplot(gs[0, 2]),
            self.fig.add_subplot(gs[1, 1]),
            self.fig.add_subplot(gs[1, 2]),
        ]
        self.tricolors: list = [None] * 4
        self.colorbars: list = [None] * 4

        self._build_plots()

        # -- Prev / Next buttons --
        ax_prev = self.fig.add_subplot(gs[2, 1])
        ax_next = self.fig.add_subplot(gs[2, 2])
        self.btn_prev = Button(ax_prev, "◀  Prev")
        self.btn_next = Button(ax_next, "Next  ▶")
        self.btn_prev.on_clicked(lambda _: self._step(-1))
        self.btn_next.on_clicked(lambda _: self._step(+1))

        # -- draw run list --
        self._draw_run_list()

        self.fig.canvas.mpl_connect("key_press_event",    self._on_key)
        self.fig.canvas.mpl_connect("button_press_event", self._on_click)

    def _build_plots(self):
        """Create tripcolor artists for the current triangulation."""
        dummy = np.zeros(len(self.triang.triangles))
        for i, ax in enumerate(self.axes):
            ax.cla()
            tc = ax.tripcolor(self.triang, facecolors=dummy, cmap="viridis", shading="flat")
            self.tricolors[i] = tc
            ax.set_aspect("equal", adjustable="box")
            ax.set_title(FIELD_NAMES[i], fontsize=9)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="4%", pad=0.06)
            if self.colorbars[i] is not None:
                self.colorbars[i].ax.remove()
            self.colorbars[i] = self.fig.colorbar(tc, cax=cax)

    def _rebuild_plots(self):
        """Called when switching to a run with a potentially different mesh."""
        self._build_plots()

    # ---- run list ----------------------------------------------------------

    def _draw_run_list(self):
        ax = self.ax_runs
        ax.cla()
        ax.axis("off")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title("Runs", fontsize=9, pad=3)

        n = len(self.run_dirs)
        self._run_list_n = n
        for i, run_dir in enumerate(self.run_dirs):
            name = os.path.basename(run_dir)
            y = 1.0 - (i + 0.5) / n
            selected = (i == self.run_idx)
            ax.text(
                0.05, y, name,
                transform=ax.transAxes,
                fontsize=7.5,
                color="royalblue" if selected else "black",
                fontweight="bold" if selected else "normal",
                va="center", ha="left",
                clip_on=True,
            )

    def _refresh_run_list(self):
        self._draw_run_list()

    # ---- rendering ---------------------------------------------------------

    def _render(self):
        t, cell_prims = load_step(self.files[self.step_idx])
        n = len(self.files)
        run_name = os.path.basename(self.run_dirs[self.run_idx])
        self.fig.suptitle(
            f"{run_name}   |   step {self.step_idx + 1}/{n}   |   t = {t:.4g}",
            fontsize=10,
        )
        for i in range(4):
            values = cell_prims[:, i]
            self.tricolors[i].set_array(values)
            self.tricolors[i].set_clim(values.min(), values.max())
            self.colorbars[i].update_normal(self.tricolors[i])
        self.fig.canvas.draw_idle()

    # ---- navigation --------------------------------------------------------

    def _step(self, delta: int):
        self.step_idx = (self.step_idx + delta) % len(self.files)
        self._render()

    def _on_key(self, event):
        if event.key == "right":
            self._step(+1)
        elif event.key == "left":
            self._step(-1)
        elif event.key == "up":
            self._switch_run(self.run_idx - 1)
        elif event.key == "down":
            self._switch_run(self.run_idx + 1)

    def _on_click(self, event):
        if event.inaxes is not self.ax_runs:
            return
        # Map y position in axes coords → run index
        x_ax, y_ax = self.ax_runs.transAxes.inverted().transform((event.x, event.y))
        idx = int((1.0 - y_ax) * self._run_list_n)
        idx = max(0, min(idx, self._run_list_n - 1))
        if idx != self.run_idx:
            self._switch_run(idx)

    def show(self):
        plt.show()


# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="View FVM solution timesteps.")
    parser.add_argument(
        "directory",
        help="Run directory (contains mesh_props.npz) or dataset directory (contains run subdirs)",
    )
    args = parser.parse_args()
    Viewer(args.directory).show()


if __name__ == "__main__":
    main()
