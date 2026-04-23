#!/usr/bin/env python3
"""
align_p2p_json.py — Point-to-Point Reference Alignment

Aligns a single LAS point cloud to a CityJSON LOD2 model by letting the user
interactively pick corresponding vertices / points in both datasets, then
computing the optimal rigid-body + uniform-scale transform (Umeyama algorithm).

Workflow
--------
1. GUI dialogs: select subfolder → select .las file → select .json file.
2. JSON viewer: click anywhere on an edge (or a vertex) on the 3-D model;
   picks are numbered 1, 2, 3 …
3. Point cloud viewer: click the matching points (same numbering).
4. Umeyama SVD transform is computed and applied to the full cloud.
5. Output is written to outputs/06_aligned_p2p/<subfolder>/<filename>.las

Usage
-----
    conda activate las-env
    python scripts/LOD2toLOD3/04_align_p2p_json.py

Controls (both viewers)
-----------------------
    Left-click a point  → add pick
    Right-click         → remove last pick
    Enter / 'q'         → confirm picks and close viewer
    'd'                 → delete last pick
"""

import os
import sys
import json
import numpy as np
import laspy
from pathlib import Path
from typing import List, Tuple, Optional

try:
    import matplotlib
    matplotlib.use("TkAgg")          # interactive backend
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D, proj3d   # noqa: F401 (registers projection)
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

BASE_DIR        = Path(__file__).resolve().parents[2]   # project root
PC_BASE         = BASE_DIR / "outputs" / "04_manual_cleaned_point_clouds"
JSON_BASE       = BASE_DIR / "outputs" / "00_json_wall_merged"
OUTPUT_BASE     = BASE_DIR / "outputs" / "06_aligned_p2p"

MAX_DISPLAY_PTS = 100_000   # subsample for interactive display


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
    """GUI dialog: pick a subfolder from outputs/04_manual_cleaned_point_clouds."""
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


def select_json_file() -> Optional[Path]:
    """GUI dialog: pick a .json or .cityjson file from JSON_BASE."""
    root = _tk_root()
    json_files = sorted(JSON_BASE.rglob("*.json")) + sorted(JSON_BASE.rglob("*.cityjson"))
    if not json_files:
        messagebox.showerror("Error", f"No .json/.cityjson files in:\n{JSON_BASE}")
        root.destroy()
        return None

    if len(json_files) == 1:
        print(f"  Auto-selected JSON: {json_files[0].name}")
        root.destroy()
        return json_files[0]

    win = tk.Toplevel(root)
    win.title("Select JSON Model")
    win.geometry("500x260")
    win.grab_set()

    tk.Label(win, text="Select a LOD2 JSON model:", font=("Arial", 11)).pack(pady=8)
    lb = tk.Listbox(win, selectmode=tk.SINGLE, font=("Courier", 10))
    for f in json_files:
        size_kb = f.stat().st_size / 1024
        lb.insert(tk.END, f"{f.name}  ({size_kb:.0f} KB)")
    lb.pack(fill=tk.BOTH, expand=True, padx=10)
    lb.selection_set(0)

    chosen = [None]

    def _ok():
        sel = lb.curselection()
        if sel:
            chosen[0] = json_files[sel[0]]
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
#  JSON PARSING
# ============================================================================

def decode_vertices(cm):
    """Decode integer vertices to real-world coordinates."""
    raw       = np.array(cm.get("vertices", []), dtype=np.float64)
    t         = cm.get("transform", {})
    scale     = np.array(t.get("scale",     [1, 1, 1]), dtype=np.float64)
    translate = np.array(t.get("translate", [0, 0, 0]), dtype=np.float64)
    return raw * scale + translate


def parse_json_vertices(json_path: Path) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Parse WallSurface polygons from the CityJSON.

    Returns
    -------
    unique_verts : (N, 3) array of all unique vertices across all polygons.
    polygons     : list of (M_i, 3) arrays — one per polygon (for wireframe display).
    """
    print(f"Parsing JSON: {json_path.name} …")
    with open(json_path, "r", encoding="utf-8") as f:
        cm = json.load(f)

    world_verts = decode_vertices(cm)
    polygons: List[np.ndarray] = []
    
    for obj_id, obj in cm.get("CityObjects", {}).items():
        if obj.get("type") not in ["Building", "BuildingPart"]:
            continue
            
        for geom in obj.get("geometry", []):
            geom_type = geom.get("type")
            if geom_type not in ["Solid", "MultiSurface", "CompositeSurface"]:
                continue
                
            boundaries = geom.get("boundaries", [])
            semantics = geom.get("semantics", {})
            surfaces = semantics.get("surfaces", [])
            values = semantics.get("values", [])
            
            if not boundaries or not surfaces or not values:
                continue
                
            wall_semantic_indices = set()
            for i, srf in enumerate(surfaces):
                if srf.get("type") == "WallSurface":
                    wall_semantic_indices.add(i)
                    
            if not wall_semantic_indices:
                continue
                
            is_solid = (geom_type == "Solid")
            shells = boundaries if is_solid else [boundaries]
            shell_values = values if is_solid else [values]
            
            for shell, s_vals in zip(shells, shell_values):
                for polygon, p_val in zip(shell, s_vals):
                    if p_val in wall_semantic_indices:
                        try:
                            ext_ring = polygon[0]
                            coords = world_verts[np.array(ext_ring)]
                            polygons.append(coords)
                        except (IndexError, TypeError, KeyError):
                            pass

    if not polygons:
        raise RuntimeError("No WallSurface geometry found in JSON file.")
        
    all_verts = [p for p in polygons]
    stacked = np.vstack(all_verts)
    unique_verts = np.unique(stacked, axis=0) # deduplicate
    
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
    label_prefix  : e.g. "JSON" or "Cloud" — used in pick annotations.
    reference_picks: list of 3-D points already chosen in the *other* viewer
                     (shown as faded stars for context).
    edge_picking  : if True (JSON viewer), left-clicks snap to the nearest
                    point along any polygon edge — not just scatter vertices.
    edge_pick_tol : screen-space pixel tolerance for edge picking.
    """

    def __init__(self,
                 display_pts: np.ndarray,
                 display_colors: Optional[np.ndarray],
                 title: str = "Pick Points",
                 max_picks: int = 0,
                 label_prefix: str = "P",
                 reference_picks: Optional[List[np.ndarray]] = None,
                 polygons: Optional[List[np.ndarray]] = None,
                 edge_picking: bool = False,
                 edge_pick_tol: float = 12.0):

        self.pts          = display_pts
        self.colors       = display_colors
        self.title        = title
        self.max_picks    = max_picks
        self.label_prefix = label_prefix
        self.ref_picks    = reference_picks or []
        self.polygons     = polygons or []
        self.edge_picking = edge_picking
        self.edge_pick_tol = edge_pick_tol

        self.picked_pts: List[np.ndarray] = []     # world-space 3-D points
        self.picked_idx: List[int]         = []     # indices into self.pts (scatter mode)
        self._confirmed  = False

        # Pre-build edge segment list from polygons for fast picking
        self._edge_segments: List[Tuple[np.ndarray, np.ndarray]] = []
        for poly in self.polygons:
            verts = poly[:-1] if len(poly) > 1 and np.allclose(poly[0], poly[-1]) else poly
            for i in range(len(verts)):
                self._edge_segments.append((verts[i], verts[(i + 1) % len(verts)]))

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
        self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        self.fig.canvas.mpl_connect("key_press_event",    self._on_key)
        self.fig.canvas.mpl_connect("close_event",        self._on_close)

        # Holds annotation handles for pick labels  (list of (scatter, text))
        self._pick_artists: List = []
        self._edge_pick_pending = False   # guard to avoid double-registering

        self._update_title()

    # --------------------------------------------------------------- background
    def _draw_background(self):
        """Draw filled polygon faces + outline (JSON viewer) or skip (cloud viewer)."""
        if not self.polygons:
            return

        # Build face list: each polygon is a list of (x,y,z) tuples
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
        """Scatter pick_event — used in cloud (non-edge) mode only."""
        if self.edge_picking:
            return   # edge mode handles its own left-click via _on_click
        if event.mouseevent.button != 1:
            return
        if self.max_picks and len(self.picked_pts) >= self.max_picks:
            print(f"  Maximum {self.max_picks} picks reached.")
            return

        ind = event.ind[0]   # index into self.pts
        pt  = self.pts[ind]
        self._register_pick(pt)

    def _on_click(self, event):
        """Unified mouse-button handler.

        Right-click  → undo last pick (both modes).
        Left-click   → edge-pick in JSON mode; ignored in cloud mode
                       (cloud mode uses pick_event from the scatter).
        """
        if event.button == 3:
            self._undo()
            return

        if event.button != 1 or not self.edge_picking:
            return
        if event.inaxes is not self.ax:
            return
        if self.max_picks and len(self.picked_pts) >= self.max_picks:
            print(f"  Maximum {self.max_picks} picks reached.")
            return

        # Project all edge segments to screen space and find closest
        pt3d = self._closest_edge_point(event.x, event.y)
        if pt3d is None:
            return   # click was too far from any edge
        self._register_pick(pt3d)

    # --------------------------------------------------------- edge projection
    def _closest_edge_point(self, xd: float, yd: float) -> Optional[np.ndarray]:
        """
        Given a display-space click (xd, yd), find the 3-D point on the
        nearest polygon edge that is closest to the click in screen space.

        Returns the 3-D point (world coords) or None if no edge is within
        self.edge_pick_tol pixels.
        """
        if not self._edge_segments:
            return None

        # Get the current projection matrix
        M = self.ax.get_proj()

        best_dist  = float("inf")
        best_pt3d  = None

        for p0, p1 in self._edge_segments:
            # Project endpoints to display (pixel) coordinates
            x0s, y0s, _ = proj3d.proj_transform(p0[0], p0[1], p0[2], M)
            x1s, y1s, _ = proj3d.proj_transform(p1[0], p1[1], p1[2], M)
            # Convert from axes-normalised [-1,1] to display pixels
            x0d, y0d = self.ax.transData.transform((x0s, y0s))
            x1d, y1d = self.ax.transData.transform((x1s, y1s))

            # Parametric closest-point-on-segment
            dx, dy = x1d - x0d, y1d - y0d
            seg_len2 = dx * dx + dy * dy
            if seg_len2 == 0:
                t = 0.0
            else:
                t = ((xd - x0d) * dx + (yd - y0d) * dy) / seg_len2
                t = max(0.0, min(1.0, t))

            cx = x0d + t * dx
            cy = y0d + t * dy
            dist = ((xd - cx) ** 2 + (yd - cy) ** 2) ** 0.5

            if dist < best_dist:
                best_dist = dist
                best_pt3d = p0 + t * (p1 - p0)   # interpolated 3-D point

        if best_dist <= self.edge_pick_tol:
            return best_pt3d
        return None

    # ---------------------------------------------------------- shared register
    def _register_pick(self, pt: np.ndarray):
        """Append a 3-D pick, draw its marker and label."""
        self.picked_pts.append(pt.copy())

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
        if self.picked_idx:
            self.picked_idx.pop()   # only populated in scatter mode
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

    json_path = select_json_file()
    if json_path is None:
        print("Cancelled — no JSON file selected.")
        return

    print(f"\n  Subfolder : {subfolder.name}")
    print(f"  LAS file  : {las_path.name}")
    print(f"  JSON file : {json_path.name}")

    # ── 2. Load data ─────────────────────────────────────────────────
    print("\nStep 2: Loading data")

    json_verts, json_polys = parse_json_vertices(json_path)
    las_pts, las_colors  = load_las(las_path)

    # ── 3. JSON vertex picker ─────────────────────────────────────────
    print("\nStep 3: Pick reference vertices on the JSON model")
    print("  Close the viewer window or press Enter/Q to confirm picks.")

    json_display, _ = subsample(json_verts, None, MAX_DISPLAY_PTS)

    json_picker = PointPicker3D(
        display_pts    = json_display,
        display_colors = None,
        title          = f"JSON Model — {json_path.name}  |  Click anywhere on an edge to pick",
        max_picks      = 0,
        label_prefix   = "J",
        polygons       = json_polys,
        edge_picking   = True,
    )
    json_picks = json_picker.run()

    if len(json_picks) < 3:
        print(f"\nERROR: Need at least 3 JSON picks (got {len(json_picks)}). Exiting.")
        return

    n_pairs = len(json_picks)
    print(f"\n  {n_pairs} JSON vertices selected:")
    for i, p in enumerate(json_picks):
        print(f"    J{i+1}: ({p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f})")

    # ── 4. Point cloud picker ─────────────────────────────────────────
    print(f"\nStep 4: Pick {n_pairs} matching point(s) in the point cloud")
    print("  Pick the same number of points as JSON picks.")

    pc_display, pc_disp_colors = subsample(las_pts, las_colors, MAX_DISPLAY_PTS)

    pc_picker = PointPicker3D(
        display_pts    = pc_display,
        display_colors = pc_disp_colors,
        title          = (f"Point Cloud — {las_path.name}  "
                          f"|  Pick {n_pairs} point(s) matching J1…J{n_pairs}"),
        max_picks      = n_pairs,
        label_prefix   = "C",
        reference_picks= json_picks,   # shown as faded stars for context
    )
    pc_picks = pc_picker.run()

    if len(pc_picks) != n_pairs:
        print(f"\nERROR: Need exactly {n_pairs} cloud picks to match JSON picks "
              f"(got {len(pc_picks)}). Exiting.")
        return

    print(f"\n  {len(pc_picks)} cloud points selected:")
    for i, p in enumerate(pc_picks):
        print(f"    C{i+1}: ({p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f})")

    # ── 5. Compute transform ──────────────────────────────────────────
    print("\nStep 5: Computing Umeyama similarity transform")

    src = np.array(pc_picks)    # cloud space  (source)
    dst = np.array(json_picks)   # JSON space    (target)

    scale, R, t = umeyama(src, dst)

    print(f"  Scale      : {scale:.6f}")
    print(f"  Translation: {t}")
    print(f"  Rotation   :\n{R}")

    # Verify residuals
    dst_pred = apply_transform(src, scale, R, t)
    residuals = np.linalg.norm(dst_pred - dst, axis=1)
    print(f"  Residuals  : mean={residuals.mean():.4f}  max={residuals.max():.4f}  "
          f"(in JSON units)")

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
