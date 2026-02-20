#!/usr/bin/env python3
"""
align_p2p.py — Point-to-Point Reference Alignment

Aligns a single LAS point cloud to a CityGML LOD2 model by letting the user
interactively pick corresponding vertices / points in both datasets, then
computing the optimal rigid-body + uniform-scale transform (Umeyama algorithm).

Workflow
--------
1. GUI dialogs: select subfolder → select .las file → select .gml file.
2. GML viewer: click vertices on the 3-D model; they are numbered 1, 2, 3 …
3. Point cloud viewer: click the matching points (same numbering).
4. Umeyama SVD transform is computed and applied to the full cloud.
5. Output is written to  outputs/06_aligned_p2p/<subfolder>/<filename>.las

Usage
-----
    conda activate las-env
    python scripts/LOD2toLOD3/align_p2p.py

Controls (both viewers)
-----------------------
    Left-click a point  → add pick
    Right-click         → remove last pick
    Enter / 'q'         → confirm picks and close viewer
    'd'                 → delete last pick
"""

import os
import sys
import numpy as np
import laspy
from pathlib import Path
from typing import List, Tuple, Optional

# ── imports with graceful errors ────────────────────────────────────────────
try:
    from lxml import etree
except ImportError:
    sys.exit("ERROR: lxml not installed.  Run: pip install lxml")

try:
    import matplotlib
    matplotlib.use("TkAgg")          # interactive backend
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D          # noqa: F401 (registers projection)
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from matplotlib.widgets import Button
except ImportError:
    sys.exit("ERROR: matplotlib not installed.")

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, simpledialog
except ImportError:
    sys.exit("ERROR: tkinter not available.")

# ── constants ────────────────────────────────────────────────────────────────
NAMESPACES = {
    "gml":  "http://www.opengis.net/gml",
    "bldg": "http://www.opengis.net/citygml/building/2.0",
    "core": "http://www.opengis.net/citygml/2.0",
}

BASE_DIR        = Path(__file__).resolve().parents[2]   # project root
PC_BASE         = BASE_DIR / "outputs" / "05_manual_orient_point_cloud"
GML_BASE        = BASE_DIR / "data" / "lod_2"
OUTPUT_BASE     = BASE_DIR / "outputs" / "06_aligned_p2p"

MAX_DISPLAY_PTS = 80_000   # subsample for interactive display


# ============================================================================
#  FILE SELECTION (tkinter)
# ============================================================================

def _tk_root() -> tk.Tk:
    root = tk.Tk()
    root.withdraw()
    root.lift()
    root.attributes("-topmost", True)
    return root


def select_subfolder() -> Optional[Path]:
    """GUI dialog: pick a subfolder from outputs/05_manual_orient_point_cloud."""
    root = _tk_root()
    subfolders = sorted([d for d in PC_BASE.iterdir() if d.is_dir()])
    if not subfolders:
        messagebox.showerror("Error", f"No subfolders found in:\n{PC_BASE}")
        root.destroy()
        return None

    # Simple Listbox dialog
    win = tk.Toplevel(root)
    win.title("Select Subfolder")
    win.geometry("400x300")
    win.grab_set()

    tk.Label(win, text="Select point cloud subfolder:", font=("Arial", 11)).pack(pady=8)
    lb = tk.Listbox(win, selectmode=tk.SINGLE, font=("Courier", 10))
    for sf in subfolders:
        lb.insert(tk.END, sf.name)
    lb.pack(fill=tk.BOTH, expand=True, padx=10)
    lb.selection_set(0)

    chosen = [None]

    def _ok():
        sel = lb.curselection()
        if sel:
            chosen[0] = subfolders[sel[0]]
        win.destroy()

    def _cancel():
        win.destroy()

    btn_frame = tk.Frame(win)
    btn_frame.pack(pady=6)
    tk.Button(btn_frame, text="OK",     width=10, command=_ok).pack(side=tk.LEFT, padx=4)
    tk.Button(btn_frame, text="Cancel", width=10, command=_cancel).pack(side=tk.LEFT, padx=4)

    win.protocol("WM_DELETE_WINDOW", _cancel)
    root.wait_window(win)
    root.destroy()
    return chosen[0]


def select_las_file(subfolder: Path) -> Optional[Path]:
    """GUI dialog: pick one .las file from the selected subfolder."""
    root = _tk_root()
    las_files = sorted(subfolder.glob("*.las"))
    if not las_files:
        messagebox.showerror("Error", f"No .las files in:\n{subfolder}")
        root.destroy()
        return None

    win = tk.Toplevel(root)
    win.title("Select Point Cloud File")
    win.geometry("450x300")
    win.grab_set()

    tk.Label(win, text=f"Subfolder: {subfolder.name}", font=("Arial", 10, "italic")).pack(pady=4)
    tk.Label(win, text="Select a .las file to align:", font=("Arial", 11)).pack(pady=4)
    lb = tk.Listbox(win, selectmode=tk.SINGLE, font=("Courier", 10))
    for f in las_files:
        size_mb = f.stat().st_size / 1_048_576
        lb.insert(tk.END, f"{f.name}  ({size_mb:.1f} MB)")
    lb.pack(fill=tk.BOTH, expand=True, padx=10)
    lb.selection_set(0)

    chosen = [None]

    def _ok():
        sel = lb.curselection()
        if sel:
            chosen[0] = las_files[sel[0]]
        win.destroy()

    def _cancel():
        win.destroy()

    btn_frame = tk.Frame(win)
    btn_frame.pack(pady=6)
    tk.Button(btn_frame, text="OK",     width=10, command=_ok).pack(side=tk.LEFT, padx=4)
    tk.Button(btn_frame, text="Cancel", width=10, command=_cancel).pack(side=tk.LEFT, padx=4)
    win.protocol("WM_DELETE_WINDOW", _cancel)
    root.wait_window(win)
    root.destroy()
    return chosen[0]


def select_gml_file() -> Optional[Path]:
    """GUI dialog: pick a .gml file from data/lod_2."""
    root = _tk_root()
    gml_files = sorted(GML_BASE.glob("*.gml"))
    if not gml_files:
        messagebox.showerror("Error", f"No .gml files in:\n{GML_BASE}")
        root.destroy()
        return None

    if len(gml_files) == 1:
        print(f"  Auto-selected GML: {gml_files[0].name}")
        root.destroy()
        return gml_files[0]

    win = tk.Toplevel(root)
    win.title("Select GML Model")
    win.geometry("500x260")
    win.grab_set()

    tk.Label(win, text="Select a LOD2 GML model:", font=("Arial", 11)).pack(pady=8)
    lb = tk.Listbox(win, selectmode=tk.SINGLE, font=("Courier", 10))
    for f in gml_files:
        size_kb = f.stat().st_size / 1024
        lb.insert(tk.END, f"{f.name}  ({size_kb:.0f} KB)")
    lb.pack(fill=tk.BOTH, expand=True, padx=10)
    lb.selection_set(0)

    chosen = [None]

    def _ok():
        sel = lb.curselection()
        if sel:
            chosen[0] = gml_files[sel[0]]
        win.destroy()

    def _cancel():
        win.destroy()

    btn_frame = tk.Frame(win)
    btn_frame.pack(pady=6)
    tk.Button(btn_frame, text="OK",     width=10, command=_ok).pack(side=tk.LEFT, padx=4)
    tk.Button(btn_frame, text="Cancel", width=10, command=_cancel).pack(side=tk.LEFT, padx=4)
    win.protocol("WM_DELETE_WINDOW", _cancel)
    root.wait_window(win)
    root.destroy()
    return chosen[0]


# ============================================================================
#  GML PARSING
# ============================================================================

def parse_gml_vertices(gml_path: Path) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Parse all WallSurface polygons from the GML.

    Returns
    -------
    unique_verts : (N, 3) array of all unique vertices across all polygons.
    polygons     : list of (M_i, 3) arrays — one per polygon (for wireframe display).
    """
    print(f"Parsing GML: {gml_path.name} …")
    tree = etree.parse(str(gml_path))
    root = tree.getroot()

    wall_elems = root.xpath("//bldg:WallSurface", namespaces=NAMESPACES)
    print(f"  Found {len(wall_elems)} WallSurface elements")

    polygons: List[np.ndarray] = []
    all_verts: List[np.ndarray] = []

    for wall in wall_elems:
        for pos_list in wall.xpath(".//gml:posList", namespaces=NAMESPACES):
            text = (pos_list.text or "").strip()
            if not text:
                continue
            coords = np.array(list(map(float, text.split()))).reshape(-1, 3)
            polygons.append(coords)
            all_verts.append(coords)

    if not all_verts:
        raise RuntimeError("No vertex data found in GML file.")

    stacked     = np.vstack(all_verts)
    unique_verts = np.unique(stacked, axis=0)   # deduplicate

    print(f"  Extracted {len(polygons)} polygons, {len(unique_verts)} unique vertices")
    return unique_verts, polygons


# ============================================================================
#  LAS LOADING
# ============================================================================

def load_las(las_path: Path) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Load a LAS file → (points Nx3, colors Nx3 [0-1] or None)."""
    print(f"Loading point cloud: {las_path.name} …")
    las = laspy.read(str(las_path))
    pts = np.vstack((las.x, las.y, las.z)).T
    colors = None
    if hasattr(las, "red"):
        colors = np.vstack((las.red, las.green, las.blue)).T / 65535.0
    print(f"  {len(pts):,} points loaded")
    return pts, colors


def subsample(pts: np.ndarray, colors: Optional[np.ndarray],
              max_pts: int = MAX_DISPLAY_PTS) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Uniform random subsample for display speed."""
    n = len(pts)
    if n <= max_pts:
        return pts, colors
    idx = np.random.default_rng(42).choice(n, max_pts, replace=False)
    return pts[idx], (colors[idx] if colors is not None else None)


# ============================================================================
#  INTERACTIVE 3-D PICKER
# ============================================================================

_PICK_COLORS = [
    "#e41a1c", "#377eb8", "#4daf4a", "#984ea3",
    "#ff7f00", "#a65628", "#f781bf", "#999999",
    "#ffff33", "#00ced1",
]


class PointPicker3D:
    """
    Interactive matplotlib 3-D viewer for picking labelled points.

    Parameters
    ----------
    display_pts   : (N, 3) array shown as background scatter.
    display_colors: (N, 3) RGB or None (uniform gray shown).
    title         : window title.
    max_picks     : maximum number of picks allowed (0 = unlimited).
    label_prefix  : e.g. "GML" or "Cloud" — used in pick annotations.
    reference_picks: list of 3-D points already chosen in the *other* viewer
                     (shown as faded stars for context).
    """

    def __init__(self,
                 display_pts: np.ndarray,
                 display_colors: Optional[np.ndarray],
                 title: str = "Pick Points",
                 max_picks: int = 0,
                 label_prefix: str = "P",
                 reference_picks: Optional[List[np.ndarray]] = None,
                 polygons: Optional[List[np.ndarray]] = None):

        self.pts         = display_pts
        self.colors      = display_colors
        self.title       = title
        self.max_picks   = max_picks
        self.label_prefix = label_prefix
        self.ref_picks   = reference_picks or []
        self.polygons    = polygons or []

        self.picked_pts: List[np.ndarray] = []     # world-space 3-D points
        self.picked_idx: List[int]         = []     # indices into self.pts
        self._confirmed  = False

        self._build_figure()

    # ------------------------------------------------------------------ build
    def _build_figure(self):
        self.fig = plt.figure(figsize=(13, 8))
        self.fig.patch.set_facecolor("#1e1e2e")
        self.fig.suptitle(
            self.title,
            color="white", fontsize=13, fontweight="bold"
        )

        # Main 3-D axes
        self.ax = self.fig.add_axes([0.05, 0.12, 0.75, 0.82], projection="3d")
        self.ax.set_facecolor("#2a2a3e")
        for pane in (self.ax.xaxis.pane, self.ax.yaxis.pane, self.ax.zaxis.pane):
            pane.fill = False
            pane.set_edgecolor("#555")

        # Instruction panel on the right
        info_ax = self.fig.add_axes([0.81, 0.12, 0.18, 0.82])
        info_ax.set_facecolor("#2a2a3e")
        info_ax.axis("off")
        instructions = (
            "CONTROLS\n"
            "─────────────\n"
            "Left-click → Add pick\n"
            "Right-click → Undo last\n"
            "D key       → Undo last\n"
            "Enter / Q   → Confirm\n\n"
            "PICKS (max: {})\n"
            "─────────────\n"
        ).format(self.max_picks if self.max_picks else "∞")
        self._info_text = info_ax.text(
            0.05, 0.95, instructions,
            transform=info_ax.transAxes,
            va="top", ha="left",
            color="white", fontsize=8,
            fontfamily="monospace",
        )
        self._info_ax = info_ax

        # "Confirm" button
        btn_ax = self.fig.add_axes([0.82, 0.04, 0.14, 0.05])
        self._btn = Button(btn_ax, "Confirm ✓",
                           color="#4caf50", hovercolor="#66bb6a")
        self._btn.label.set_color("white")
        self._btn.on_clicked(self._on_confirm)

        # Draw background point cloud / model
        self._draw_background()

        # Overlay reference picks from the other viewer (faded stars)
        if self.ref_picks:
            rp = np.array(self.ref_picks)
            self.ax.scatter(rp[:, 0], rp[:, 1], rp[:, 2],
                            marker="*", s=120, c="#ffffff", alpha=0.35,
                            depthshade=False, label="_ref")

        # Scatter for the pickable points (enable picker)
        if self.colors is not None:
            c = self.colors
        else:
            c = "#8888bb"

        self._scatter = self.ax.scatter(
            self.pts[:, 0], self.pts[:, 1], self.pts[:, 2],
            c=c, s=1, alpha=0.55, depthshade=True,
            picker=True, pickradius=6,
        )

        # ── Auto-zoom to data extents ──────────────────────────────
        mn, mx = self.pts.min(0), self.pts.max(0)
        pad = (mx - mn).max() * 0.05   # 5 % padding
        self.ax.set_xlim(mn[0] - pad, mx[0] + pad)
        self.ax.set_ylim(mn[1] - pad, mx[1] + pad)
        self.ax.set_zlim(mn[2] - pad, mx[2] + pad)
        # Equal aspect ratio trick for matplotlib 3D
        mid = (mn + mx) / 2
        rng = (mx - mn).max() / 2 + pad
        self.ax.set_xlim(mid[0] - rng, mid[0] + rng)
        self.ax.set_ylim(mid[1] - rng, mid[1] + rng)
        self.ax.set_zlim(mid[2] - rng, mid[2] + rng)

        # Connect events
        self.fig.canvas.mpl_connect("pick_event",        self._on_pick)
        self.fig.canvas.mpl_connect("button_press_event", self._on_rclick)
        self.fig.canvas.mpl_connect("key_press_event",    self._on_key)
        self.fig.canvas.mpl_connect("close_event",        self._on_close)

        # Holds annotation handles for pick labels  (list of (scatter, text))
        self._pick_artists: List = []

        self._update_title()

    # --------------------------------------------------------------- background
    def _draw_background(self):
        """Draw filled polygon faces + outline (GML viewer) or skip (cloud viewer)."""
        if not self.polygons:
            return

        # Build face list: each polygon is a list of (x,y,z) tuples
        # CityGML rings close on themselves — drop the repeated last vertex.
        faces = []
        for poly in self.polygons:
            verts = poly[:-1] if len(poly) > 1 and np.allclose(poly[0], poly[-1]) else poly
            if len(verts) < 3:
                continue
            faces.append([tuple(v) for v in verts])

        if faces:
            pc = Poly3DCollection(
                faces,
                facecolor="#3a6ea5",   # steel-blue fill
                edgecolor="#7ab0e0",   # lighter blue edge
                linewidth=0.4,
                alpha=0.30,
            )
            self.ax.add_collection3d(pc)

        # Bold outlines on top for clarity
        for poly in self.polygons:
            xs = list(poly[:, 0]) + [poly[0, 0]]
            ys = list(poly[:, 1]) + [poly[0, 1]]
            zs = list(poly[:, 2]) + [poly[0, 2]]
            self.ax.plot(xs, ys, zs, color="#7ab0e0", lw=0.7, alpha=0.7)

    # ------------------------------------------------------------------- events
    def _on_pick(self, event):
        """Left-click pick — add the nearest shown point."""
        if event.mouseevent.button != 1:
            return
        if self.max_picks and len(self.picked_pts) >= self.max_picks:
            print(f"  Maximum {self.max_picks} picks reached.")
            return

        ind = event.ind[0]   # index into self.pts
        pt  = self.pts[ind]

        self.picked_pts.append(pt.copy())
        self.picked_idx.append(ind)

        n   = len(self.picked_pts)
        col = _PICK_COLORS[(n - 1) % len(_PICK_COLORS)]

        sc = self.ax.scatter(*pt, s=120, c=col, marker="o",
                              edgecolors="white", linewidths=0.8,
                              zorder=5, depthshade=False)
        tx = self.ax.text(pt[0], pt[1], pt[2],
                          f"  {self.label_prefix}{n}",
                          color=col, fontsize=9, fontweight="bold",
                          zorder=6)
        self._pick_artists.append((sc, tx))

        print(f"  Pick {n}: {self.label_prefix}{n} = ({pt[0]:.3f}, {pt[1]:.3f}, {pt[2]:.3f})")
        self._update_title()
        self.fig.canvas.draw_idle()

    def _on_rclick(self, event):
        """Right-click → undo last pick."""
        if event.button == 3:
            self._undo()

    def _on_key(self, event):
        if event.key in ("d", "D"):
            self._undo()
        elif event.key in ("enter", "q", "Q"):
            self._on_confirm(None)

    def _on_close(self, _event):
        self._confirmed = True   # treat close as confirm

    def _on_confirm(self, _event):
        min_picks = 3
        if len(self.picked_pts) < min_picks:
            # Show warning via matplotlib text (no separate tk dialog)
            msg = (f"Please pick at least {min_picks} points "
                   f"(currently {len(self.picked_pts)}).")
            print(f"  WARNING: {msg}")
            # Flash text on figure
            t = self.fig.text(0.5, 0.5, msg,
                              ha="center", va="center",
                              fontsize=14, color="red",
                              bbox=dict(boxstyle="round", fc="#1e1e2e", ec="red"))
            self.fig.canvas.draw_idle()
            self.fig.canvas.start_event_loop(2)
            t.remove()
            self.fig.canvas.draw_idle()
            return
        self._confirmed = True
        plt.close(self.fig)

    def _undo(self):
        if not self.picked_pts:
            return
        self.picked_pts.pop()
        self.picked_idx.pop()
        sc, tx = self._pick_artists.pop()
        sc.remove()
        tx.remove()
        print(f"  Removed last pick. {len(self.picked_pts)} picks remaining.")
        self._update_title()
        self.fig.canvas.draw_idle()

    def _update_title(self):
        n     = len(self.picked_pts)
        limit = f"/{self.max_picks}" if self.max_picks else ""
        msg   = f"{self.title}   [{n}{limit} picks]"
        self.fig.suptitle(msg, color="white", fontsize=13, fontweight="bold")

        # Update right-panel pick list
        lines = (
            "CONTROLS\n"
            "─────────────\n"
            "Left-click → Add pick\n"
            "Right-click → Undo last\n"
            "D key       → Undo last\n"
            "Enter / Q   → Confirm\n\n"
            f"PICKS ({n}{limit})\n"
            "─────────────\n"
        )
        for i, pt in enumerate(self.picked_pts):
            col_hex = _PICK_COLORS[i % len(_PICK_COLORS)]
            lines += f"{self.label_prefix}{i+1}: ({pt[0]:.2f},{pt[1]:.2f},{pt[2]:.2f})\n"
        self._info_text.set_text(lines)

    # ------------------------------------------------------------------- run
    def run(self) -> List[np.ndarray]:
        """Show the viewer and block until confirmed. Returns list of 3-D picks."""
        plt.show(block=True)
        return list(self.picked_pts)


# ============================================================================
#  UMEYAMA TRANSFORM  (rotation + translation + uniform scale)
# ============================================================================

def umeyama(src: np.ndarray, dst: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Compute the similarity transform  dst ≈ scale * R @ src + t

    Parameters
    ----------
    src, dst : (N, 3) corresponding point arrays

    Returns
    -------
    scale : float
    R     : (3, 3) rotation matrix
    t     : (3,) translation vector
    """
    assert src.shape == dst.shape and src.shape[1] == 3, \
        "src and dst must both be (N, 3) with N >= 3"

    n = src.shape[0]
    mu_src = src.mean(0)
    mu_dst = dst.mean(0)

    src_c = src - mu_src
    dst_c = dst - mu_dst

    var_src = np.sum(src_c ** 2) / n

    H = (dst_c.T @ src_c) / n

    U, S, Vt = np.linalg.svd(H)

    # Correct for reflection
    d = np.linalg.det(U @ Vt)
    D = np.diag([1.0, 1.0, np.sign(d)])

    R     = U @ D @ Vt
    scale = np.sum(S * np.diag(D)) / var_src
    t     = mu_dst - scale * (R @ mu_src)

    return float(scale), R, t


def apply_transform(pts: np.ndarray, scale: float,
                    R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Apply similarity transform: out = scale * R @ pts.T + t"""
    return (scale * (R @ pts.T)).T + t


# ============================================================================
#  SAVE ALIGNED LAS
# ============================================================================

def save_aligned_las(aligned_pts: np.ndarray,
                     colors: Optional[np.ndarray],
                     original_las_path: Path,
                     output_path: Path):
    """Write aligned points to a new LAS file preserving header format."""
    original = laspy.read(str(original_las_path))
    header   = laspy.LasHeader(
        point_format=original.header.point_format,
        version=original.header.version,
    )
    header.offsets = aligned_pts.min(0)
    header.scales  = np.array([0.001, 0.001, 0.001])

    new_las   = laspy.LasData(header)
    new_las.x = aligned_pts[:, 0]
    new_las.y = aligned_pts[:, 1]
    new_las.z = aligned_pts[:, 2]

    if colors is not None:
        new_las.red   = (colors[:, 0] * 65535).astype(np.uint16)
        new_las.green = (colors[:, 1] * 65535).astype(np.uint16)
        new_las.blue  = (colors[:, 2] * 65535).astype(np.uint16)
    elif hasattr(original, "red") and len(original.points) == len(aligned_pts):
        new_las.red   = original.red
        new_las.green = original.green
        new_las.blue  = original.blue

    output_path.parent.mkdir(parents=True, exist_ok=True)
    new_las.write(str(output_path))
    print(f"\n  ✓ Saved: {output_path}  ({len(aligned_pts):,} points)")


# ============================================================================
#  MAIN
# ============================================================================

def main():
    print("=" * 60)
    print("  P2P Alignment — Reference-Point Registration")
    print("=" * 60)

    # ── 1. File selection ────────────────────────────────────────────
    print("\nStep 1: Select files")

    subfolder = select_subfolder()
    if subfolder is None:
        print("Cancelled — no subfolder selected.")
        return

    las_path = select_las_file(subfolder)
    if las_path is None:
        print("Cancelled — no LAS file selected.")
        return

    gml_path = select_gml_file()
    if gml_path is None:
        print("Cancelled — no GML file selected.")
        return

    print(f"\n  Subfolder : {subfolder.name}")
    print(f"  LAS file  : {las_path.name}")
    print(f"  GML file  : {gml_path.name}")

    # ── 2. Load data ─────────────────────────────────────────────────
    print("\nStep 2: Loading data")

    gml_verts, gml_polys = parse_gml_vertices(gml_path)
    las_pts, las_colors  = load_las(las_path)

    # ── 3. GML vertex picker ─────────────────────────────────────────
    print("\nStep 3: Pick reference vertices on the GML model")
    print("  Close the viewer window or press Enter/Q to confirm picks.")

    gml_display, _ = subsample(gml_verts, None, MAX_DISPLAY_PTS)

    gml_picker = PointPicker3D(
        display_pts    = gml_display,
        display_colors = None,
        title          = f"GML Model — {gml_path.name}  |  Click vertices to pick",
        max_picks      = 0,
        label_prefix   = "G",
        polygons       = gml_polys,
    )
    gml_picks = gml_picker.run()

    if len(gml_picks) < 3:
        print(f"\nERROR: Need at least 3 GML picks (got {len(gml_picks)}). Exiting.")
        return

    n_pairs = len(gml_picks)
    print(f"\n  {n_pairs} GML vertices selected:")
    for i, p in enumerate(gml_picks):
        print(f"    G{i+1}: ({p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f})")

    # ── 4. Point cloud picker ─────────────────────────────────────────
    print(f"\nStep 4: Pick {n_pairs} matching point(s) in the point cloud")
    print("  Pick the same number of points as GML picks.")

    pc_display, pc_disp_colors = subsample(las_pts, las_colors, MAX_DISPLAY_PTS)

    pc_picker = PointPicker3D(
        display_pts    = pc_display,
        display_colors = pc_disp_colors,
        title          = (f"Point Cloud — {las_path.name}  "
                          f"|  Pick {n_pairs} point(s) matching G1…G{n_pairs}"),
        max_picks      = n_pairs,
        label_prefix   = "C",
        reference_picks= gml_picks,   # shown as faded stars for context
    )
    pc_picks = pc_picker.run()

    if len(pc_picks) != n_pairs:
        print(f"\nERROR: Need exactly {n_pairs} cloud picks to match GML picks "
              f"(got {len(pc_picks)}). Exiting.")
        return

    print(f"\n  {len(pc_picks)} cloud points selected:")
    for i, p in enumerate(pc_picks):
        print(f"    C{i+1}: ({p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f})")

    # ── 5. Compute transform ──────────────────────────────────────────
    print("\nStep 5: Computing Umeyama similarity transform")

    src = np.array(pc_picks)    # cloud space  (source)
    dst = np.array(gml_picks)   # GML space    (target)

    scale, R, t = umeyama(src, dst)

    print(f"  Scale      : {scale:.6f}")
    print(f"  Translation: {t}")
    print(f"  Rotation   :\n{R}")

    # Verify residuals
    dst_pred = apply_transform(src, scale, R, t)
    residuals = np.linalg.norm(dst_pred - dst, axis=1)
    print(f"  Residuals  : mean={residuals.mean():.4f}  max={residuals.max():.4f}  "
          f"(in GML units)")

    # ── 6. Apply & save ───────────────────────────────────────────────
    print("\nStep 6: Applying transform to full point cloud and saving")

    aligned_pts = apply_transform(las_pts, scale, R, t)
    out_path    = OUTPUT_BASE / subfolder.name / las_path.name
    save_aligned_las(aligned_pts, las_colors, las_path, out_path)

    print("\n" + "=" * 60)
    print("  Alignment complete!")
    print(f"  Output : {out_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
