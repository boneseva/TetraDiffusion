"""
TetraDiffusion — Gradio Diagnostic Dashboard
=============================================
No-code inference interface for structural biologists and paper reviewers.

Architecture
------------
The app gracefully degrades in three tiers:
  1. **Full model mode** — a valid ``--config_path`` is provided; the real
     ``Trainer`` + ``GaussianDiffusion`` pipeline runs end-to-end.
  2. **Demo mode** — no config supplied; a synthetic SDF blob is generated
     using signed-distance math so reviewers can click "Generate" immediately
     without uploading any files.
  3. **Exemplar overlay** — whenever a ``.pt`` or ``.obj`` exemplar is
     uploaded the ``ExemplarLossProfile`` is instantiated and injected into
     the global ``organelle_loss_registry`` under the key ``"exemplar-driven"``.

Usage (from repo root)
-----------------------
    # Demo mode (no model required):
    python visualization/app.py

    # Full model mode:
    python visualization/app.py --config_path results/<run_name>

    # With a specific CUDA device:
    python visualization/app.py --config_path results/<run_name> --cuda_device 1
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import tempfile
import uuid
import warnings
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Repo root path injection — allows running from any CWD
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ---------------------------------------------------------------------------
# Environment & warnings hygiene (mirrors inference.py)
# ---------------------------------------------------------------------------
warnings.filterwarnings("ignore", message="None of the inputs have requires_grad=True.*",
                        category=UserWarning)
warnings.simplefilter(action="ignore", category=FutureWarning)
os.environ.setdefault("WANDB_MODE", "offline")
os.environ.setdefault("WANDB_DISABLED", "true")
os.environ.setdefault("WANDB_SILENT", "true")

torch.set_float32_matmul_precision("high")
try:
    torch._dynamo.disable()  # avoid nvcc / inductor permission issues
except Exception:
    pass

# ---------------------------------------------------------------------------
# Lazy imports — only pulled once the UI is actually used
# ---------------------------------------------------------------------------
import gradio as gr  # noqa: E402  (after path setup)
import plotly.graph_objects as go  # noqa: E402


# ===========================================================================
# Scratch-space helpers
# ===========================================================================
_SCRATCH_DIR = Path(tempfile.gettempdir()) / "tetradiffusion_app"
_SCRATCH_DIR.mkdir(parents=True, exist_ok=True)


def _unique_obj_path() -> Path:
    """Return a unique, collision-safe .obj path inside the scratch dir."""
    return _SCRATCH_DIR / f"{uuid.uuid4().hex}.obj"


def _cleanup_stale_scratch(max_age_seconds: int = 3600) -> None:
    """Delete .obj files older than *max_age_seconds* from the scratch dir."""
    import time
    now = time.time()
    for p in _SCRATCH_DIR.glob("*.obj"):
        try:
            if (now - p.stat().st_mtime) > max_age_seconds:
                p.unlink(missing_ok=True)
        except OSError:
            pass


# ===========================================================================
# CLI argument parsing
# ===========================================================================
_parser = argparse.ArgumentParser(
    description="TetraDiffusion Gradio Diagnostic Dashboard",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
_parser.add_argument("--config_path", type=str, default=None,
                     help="Path to a trained run directory (config.yaml + ds.pth + *.pt). "
                          "Omit to launch in demo mode with synthetic data.")
_parser.add_argument("--cuda_device", type=int, default=0,
                     help="CUDA device index.  Ignored when CUDA is unavailable.")
_parser.add_argument("--server_port", type=int, default=7860)
_parser.add_argument("--share", action="store_true",
                     help="Create a public Gradio share link.")
_args, _unknown = _parser.parse_known_args()


# ===========================================================================
# Global model state  (loaded once at startup; None = demo mode)
# ===========================================================================
_trainer: Optional[object] = None   # lib.Trainer.Trainer
_cfg: Optional[object] = None       # OmegaConf DictConfig
_device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_DEMO_MODE: bool = (_args.config_path is None)

if not _DEMO_MODE:
    # ------------------------------------------------------------------
    # Full-model startup sequence — mirrors inference.py
    # ------------------------------------------------------------------
    try:
        from omegaconf import OmegaConf
        from lib.Trainer import Trainer

        if torch.cuda.is_available():
            os.environ["CUDA_VISIBLE_DEVICES"] = str(_args.cuda_device)
            _device = torch.device("cuda")
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            _device = torch.device("cpu")

        _cfg = OmegaConf.load(os.path.join(_args.config_path, "config.yaml"))
        _trainer = Trainer(
            train_batch_size=_cfg.training.batch_size,
            save_and_sample_every=_cfg.training.test_every,
            results_folder=_cfg.results_folder,
            config_folder=_args.config_path,
            num_samples=1,
            train_lr=_cfg.training.lr,
            train_num_steps=_cfg.training.num_steps,
            gradient_accumulate_every=_cfg.training.ga,
            ema_decay=_cfg.training.ema_decay,
            cfg=_cfg,
            inference=True,
        )
        print("[app] Full model loaded. Running in model mode.")
    except Exception as _e:
        print(f"[app] WARNING: Could not load model ({_e}). Falling back to demo mode.")
        _DEMO_MODE = True
        _trainer = None
        _cfg = None


# ===========================================================================
# ExemplarLossProfile injection helpers
# ===========================================================================
def _try_inject_exemplar(file_path: Optional[str]) -> str:
    """
    If *file_path* is a valid .pt exemplar, instantiate ``ExemplarLossProfile``
    and register it under ``"exemplar-driven"`` in the global registry.

    Returns a short status string suitable for displaying in a Gradio textbox.
    """
    if not file_path or not os.path.isfile(file_path):
        return "No exemplar file — using standard biological priors."

    ext = Path(file_path).suffix.lower()
    if ext not in (".pt",):
        return f"Exemplar type '{ext}' loaded for reference (curvature stats unavailable without .pt)."

    if _DEMO_MODE or _trainer is None:
        return "Demo mode: exemplar uploaded but model not loaded — curvature prior is simulated."

    try:
        from lib.ops.BioConstraints import ExemplarLossProfile, organelle_loss_registry
        profile = ExemplarLossProfile(
            exemplar_path=file_path,
            mask=_trainer.ds.mask,
            neighbors=_trainer.ds.neighbors[-1].long(),
        )
        organelle_loss_registry.register_loss("exemplar-driven", profile)
        return (
            f"✅ Exemplar injected → ExemplarLossProfile\n"
            f"   mean curvature = {profile.target_mean_curvature:.4f}\n"
            f"   std  curvature = {profile.target_std_curvature:.4f}\n"
            f"   max depth      = {profile.target_max_depth:.4f}"
        )
    except Exception as exc:
        return f"⚠ Exemplar load failed: {exc}"


# ===========================================================================
# Synthetic demo geometry helpers  (DEMO MODE only)
# ===========================================================================
_RNG = np.random.default_rng(42)

_ORGANELLE_PARAMS: dict = {
    # (radius, elongation_z, tubule_noise_amp)
    "mitochondria": (0.35, 1.8, 0.12),
    "lysosome":     (0.30, 1.0, 0.04),
    "golgi":        (0.28, 0.5, 0.20),
    "er":           (0.20, 2.5, 0.25),
    "exemplar-driven": (0.32, 1.2, 0.10),
}


def _demo_sdf_sphere(verts: np.ndarray,
                     radius: float = 0.30,
                     elongation_z: float = 1.0,
                     noise_amp: float = 0.08,
                     rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Compute a noisy ellipsoid SDF on *verts*  (N, 3)  → SDF (N,)."""
    if rng is None:
        rng = np.random.default_rng()
    x, y, z = verts[:, 0], verts[:, 1], verts[:, 2]
    # Surface perturbation using low-frequency spherical-ish noise
    noise = noise_amp * rng.standard_normal(len(x))
    dist = np.sqrt(x**2 + y**2 + (z / elongation_z)**2) + noise
    return dist - radius


def _build_demo_mesh(
    organelle: str,
    cfg_scale: float,
    bio_weight: float,
    rng_seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a synthetic triangulated mesh using marching cubes on a regular grid.

    Returns (vertices, faces, sdf_values_on_verts).
    """
    try:
        from skimage.measure import marching_cubes  # scikit-image
    except ImportError:
        raise RuntimeError(
            "scikit-image is required for demo mode.  "
            "Install it with: pip install scikit-image"
        )

    if rng_seed is None:
        rng_seed = int(np.random.randint(0, 2**31))
    rng = np.random.default_rng(rng_seed)

    radius, elongation_z, noise_amp = _ORGANELLE_PARAMS.get(
        organelle, (0.30, 1.0, 0.08)
    )
    # Bias by CFG: stronger CFG → tighter, rounder shape (less noise)
    scaled_noise = noise_amp * max(0.2, 1.5 / cfg_scale)
    # Bias by bio_weight: higher → smoother surface
    scaled_noise *= max(0.1, 1.0 - bio_weight * 0.3)

    # Volumetric grid
    N = 64
    lin = np.linspace(-0.8, 0.8, N)
    xs, ys, zs = np.meshgrid(lin, lin, lin, indexing="ij")
    verts_flat = np.stack([xs.ravel(), ys.ravel(), zs.ravel()], axis=1)
    sdf_flat = _demo_sdf_sphere(verts_flat, radius=radius,
                                 elongation_z=elongation_z,
                                 noise_amp=scaled_noise, rng=rng)
    sdf_vol = sdf_flat.reshape(N, N, N)

    verts, faces, normals, _ = marching_cubes(sdf_vol, level=0.0,
                                               spacing=(1.6 / N,) * 3)
    # Centre vertices
    verts -= verts.mean(axis=0)

    # Also compute per-vertex SDF for analytics
    from scipy.interpolate import RegularGridInterpolator
    interp = RegularGridInterpolator(
        (lin, lin, lin), sdf_vol, method="linear", bounds_error=False, fill_value=0.5
    )
    sdf_on_verts = interp(verts)

    return verts.astype(np.float32), faces.astype(np.int32), sdf_on_verts.astype(np.float32)


# ===========================================================================
# Mesh analytics helpers
# ===========================================================================

def _compute_surface_area(verts: np.ndarray, faces: np.ndarray) -> float:
    """Compute total surface area of a triangulated mesh."""
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    area = 0.5 * np.linalg.norm(cross, axis=1).sum()
    return float(area)


def _compute_volume(verts: np.ndarray, faces: np.ndarray) -> float:
    """Compute signed volume via divergence theorem (assumes closed mesh)."""
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    vol = np.sum(v0 * np.cross(v1, v2)) / 6.0
    return float(abs(vol))


def _graph_laplacian_np(signal: np.ndarray,
                        faces: np.ndarray) -> np.ndarray:
    """
    Compute the uniform graph Laplacian Δ_i = x_i − mean_{j∈N(i)} x_j
    for each vertex, using the face-edge adjacency as the graph.

    Parameters
    ----------
    signal : (N,)  per-vertex scalar.
    faces  : (F, 3)  triangle indices.

    Returns
    -------
    (N,)  Laplacian values.
    """
    N = len(signal)
    from collections import defaultdict
    adj: dict = defaultdict(set)
    for tri in faces:
        i, j, k = int(tri[0]), int(tri[1]), int(tri[2])
        adj[i].update([j, k])
        adj[j].update([i, k])
        adj[k].update([i, j])

    lap = np.zeros(N, dtype=np.float32)
    for i in range(N):
        nbrs = list(adj[i])
        if nbrs:
            lap[i] = signal[i] - np.mean(signal[nbrs])
    return lap


def _curvature_histogram(
    verts: np.ndarray,
    faces: np.ndarray,
    sdf: np.ndarray,
    n_bins: int = 30,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the surface-weighted curvature histogram H ≈ −Δ_graph(sdf).

    Returns (bin_centres, counts).
    """
    surface_weight = np.exp(-np.abs(sdf) / 0.15)
    lap = _graph_laplacian_np(sdf, faces)
    curv = -lap  # H ≈ −Δ(φ)

    # Use surface weighting: duplicate vertices proportionally to weight
    weighted_curv = curv * surface_weight
    counts, edges = np.histogram(weighted_curv, bins=n_bins)
    centres = 0.5 * (edges[:-1] + edges[1:])
    return centres.astype(np.float32), counts.astype(np.float32)


def _build_comparison_figure(
    gen_centres: np.ndarray, gen_counts: np.ndarray,
    ref_centres: Optional[np.ndarray] = None,
    ref_counts: Optional[np.ndarray] = None,
) -> go.Figure:
    """
    Build a Plotly dual-bar chart for the curvature histogram comparison.
    """
    fig = go.Figure()
    bar_width = float(gen_centres[1] - gen_centres[0]) * 0.4 if len(gen_centres) > 1 else 0.05

    # Generated
    fig.add_trace(go.Bar(
        x=gen_centres - bar_width / 2,
        y=gen_counts,
        width=bar_width,
        name="Generated",
        marker=dict(color="rgba(99, 180, 255, 0.85)",
                    line=dict(color="rgba(60, 130, 220, 1.0)", width=1)),
    ))

    # Reference / exemplar
    if ref_centres is not None and ref_counts is not None:
        fig.add_trace(go.Bar(
            x=ref_centres + bar_width / 2,
            y=ref_counts,
            width=bar_width,
            name="Reference Exemplar",
            marker=dict(color="rgba(255, 160, 80, 0.85)",
                        line=dict(color="rgba(220, 110, 40, 1.0)", width=1)),
        ))

    fig.update_layout(
        title=dict(
            text="Surface Curvature Histogram  H ≈ −Δ<sub>graph</sub>(φ)",
            font=dict(size=15, color="#e0e8f5"),
        ),
        xaxis=dict(
            title="Weighted Mean Curvature",
            color="#9ab0cc",
            gridcolor="#1e2d3d",
            zeroline=True, zerolinecolor="#2a4060",
        ),
        yaxis=dict(
            title="Frequency (surface-weighted)",
            color="#9ab0cc",
            gridcolor="#1e2d3d",
        ),
        barmode="overlay",
        bargap=0.0,
        legend=dict(
            font=dict(color="#cdd8e8"),
            bgcolor="rgba(10,18,30,0.6)",
            bordercolor="#2a4060",
            borderwidth=1,
        ),
        paper_bgcolor="#0a1218",
        plot_bgcolor="#0d1c28",
        font=dict(family="Inter, Arial, sans-serif", color="#cdd8e8"),
        margin=dict(l=60, r=30, t=60, b=60),
    )
    return fig


def _build_metrics_markdown(
    surface_area: float,
    volume: float,
    n_verts: int,
    n_faces: int,
    organelle: str,
    device_str: str,
    mode: str,
) -> str:
    sv_ratio = surface_area / volume if volume > 1e-9 else float("inf")
    return f"""
### 📐 Morphological Scorecard

| Metric | Value |
|---|---|
| **Organelle Profile** | `{organelle}` |
| **Surface Area (A)** | `{surface_area:.5f}` |
| **Internal Volume (V)** | `{volume:.5f}` |
| **Surface-to-Volume (A/V)** | `{sv_ratio:.4f}` |
| **Mesh Vertices** | `{n_verts:,}` |
| **Mesh Faces** | `{n_faces:,}` |
| **Compute Device** | `{device_str}` |
| **Pipeline Mode** | `{mode}` |

---
*S/V ratio interpretation: higher values indicate more folded / tubular geometry
(e.g. ER, mitochondrial cristae); lower values correspond to rounder, vesicle-like
organelles (e.g. lysosomes).*
"""


# ===========================================================================
# Projection image builder  (2D top-down SDF projection)
# ===========================================================================

def _build_projection_image(verts: np.ndarray, sdf: np.ndarray,
                             grid_size: int = 64) -> np.ndarray:
    """
    Build a 2D top-down binary projection of the organelle interior.

    Interior = SDF < 0 (inside the zero level-set).

    Returns an (H, W, 3) uint8 RGB array.
    """
    x, y = verts[:, 0], verts[:, 1]
    interior = (sdf < 0).astype(np.float32)

    xi = ((x - x.min()) / (x.max() - x.min() + 1e-6) * (grid_size - 1)).astype(int)
    yi = ((y - y.min()) / (y.max() - y.min() + 1e-6) * (grid_size - 1)).astype(int)
    xi = np.clip(xi, 0, grid_size - 1)
    yi = np.clip(yi, 0, grid_size - 1)

    img = np.zeros((grid_size, grid_size), dtype=np.float32)
    np.add.at(img, (yi, xi), interior)
    max_val = img.max()
    if max_val > 0:
        img /= max_val

    # Colour-map: dark-blue → cyan for EM aesthetic
    rgb = np.zeros((grid_size, grid_size, 3), dtype=np.uint8)
    rgb[..., 0] = (img * 40).astype(np.uint8)        # R
    rgb[..., 1] = (img * 180).astype(np.uint8)       # G
    rgb[..., 2] = (img * 255).astype(np.uint8)       # B

    # 4× nearest-neighbour upscale for display
    scale = 4
    rgb_large = rgb.repeat(scale, axis=0).repeat(scale, axis=1)
    return rgb_large


# ===========================================================================
# Real-model inference path
# ===========================================================================

@torch.no_grad()
def _run_model_inference(
    organelle: str,
    cfg_scale: float,
    bio_weight: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Execute one denoising trajectory using the loaded Trainer / GaussianDiffusion.

    Returns (vertices, faces, sdf_on_verts).
    """
    assert _trainer is not None and _cfg is not None

    # Temporarily patch CFG scale and bio weight
    try:
        from omegaconf import OmegaConf
        if OmegaConf.select(_cfg, "image_cond"):
            OmegaConf.update(_cfg, "image_cond.cfg_guidance_scale", cfg_scale)
    except Exception:
        pass

    acc = _trainer.accelerator
    if hasattr(_trainer, "ema") and _trainer.ema is not None:
        sampling_model = _trainer.ema.ema_model
    elif hasattr(acc, "unwrap_model"):
        sampling_model = acc.unwrap_model(_trainer.model)
    else:
        sampling_model = getattr(_trainer.model, "module", _trainer.model)

    sampling_model.eval()

    dev_type = "cuda" if _device.type == "cuda" else "cpu"
    if dev_type == "cuda":
        with torch.autocast(device_type=dev_type):
            raw = list(sampling_model.sample(batch_size=1, deterministic=True))
    else:
        raw = list(sampling_model.sample(batch_size=1, deterministic=True))

    sample = torch.stack(raw, dim=0)  # (steps, 1, N, C) or (1, N, C)
    if sample.dim() == 4:
        sample = sample[-1]  # take last step

    # Decode mesh
    ds = _trainer.ds
    verts_t, colors_t, faces_t = ds.get_mesh(sample)
    verts_np = verts_t.cpu().numpy().astype(np.float32)
    faces_np = faces_t.cpu().numpy().astype(np.int32)

    # SDF channel (denormalised)
    sdf_raw = sample.squeeze().cpu()[:, 0]
    sdf_np = (sdf_raw * (ds.sdfs_max - ds.sdfs_min) + ds.sdfs_min).numpy().astype(np.float32)
    # Map full vertex sdf to surface vertices via nearest-point lookup (cheap)
    sdf_on_verts = np.interp(
        np.linspace(0, len(sdf_np) - 1, len(verts_np)),
        np.arange(len(sdf_np)), sdf_np,
    ).astype(np.float32)

    return verts_np, faces_np, sdf_on_verts


# ===========================================================================
# Core inference function — wired to the Gradio button
# ===========================================================================

def fn_run_inference(
    exemplar_file,       # gr.File → filepath string or None
    organelle: str,
    cfg_scale: float,
    bio_weight: float,
) -> Tuple[str, str, go.Figure, str]:
    """
    Unified inference callback.

    Returns
    -------
    exemplar_status : str   Short message about the exemplar injection.
    mesh_path       : str   Absolute path to the generated .obj file.
    plotly_fig      : go.Figure
    metrics_md      : str   Markdown scorecard.
    """
    # ── Step 0: housekeeping ──────────────────────────────────────────────
    _cleanup_stale_scratch()

    # ── Step 1: parameter ingestion & exemplar injection ─────────────────
    file_path: Optional[str] = (
        exemplar_file.name if hasattr(exemplar_file, "name") else exemplar_file
        if isinstance(exemplar_file, str) else None
    )
    exemplar_status = _try_inject_exemplar(file_path)

    # ── Step 2: hardware routing ──────────────────────────────────────────
    device_str = str(_device)

    # ── Step 3: mesh generation ───────────────────────────────────────────
    mode_str: str
    if not _DEMO_MODE and _trainer is not None:
        # --- Real model path ---
        try:
            verts, faces, sdf = _run_model_inference(organelle, cfg_scale, bio_weight)
            mode_str = f"Model inference · {device_str}"
        except Exception as exc:
            print(f"[app] Model inference failed, falling back to demo: {exc}")
            verts, faces, sdf = _build_demo_mesh(organelle, cfg_scale, bio_weight)
            mode_str = f"Demo fallback (inference error) · {device_str}"
    else:
        # --- Demo / synthetic path ---
        verts, faces, sdf = _build_demo_mesh(organelle, cfg_scale, bio_weight)
        mode_str = f"Synthetic demo · {device_str}"

    # ── Step 4: surface extraction & analytics ────────────────────────────
    # Curvature histogram of generated shape
    gen_centres, gen_counts = _curvature_histogram(verts, faces, sdf)

    # Optional reference histogram from exemplar (.pt)
    ref_centres, ref_counts = None, None
    if file_path and os.path.isfile(file_path) and file_path.endswith(".pt"):
        try:
            raw = torch.load(file_path, map_location="cpu", weights_only=False)
            ref_sdf_full = raw[0].float().numpy()
            # Use only a subset for speed if very large
            n_ref = min(len(ref_sdf_full), 10000)
            ref_sdf_sub = ref_sdf_full[:n_ref]
            # Build a tiny dummy mesh from random triangles on vertex indices for
            # the graph Laplacian: for the histogram we only need approximate signal.
            ref_idx = np.arange(n_ref)
            dummy_faces = np.stack([
                ref_idx[:-2:3], ref_idx[1:-1:3], ref_idx[2::3]
            ], axis=1)[:len(ref_idx) // 3]
            dummy_verts = np.zeros((n_ref, 3), dtype=np.float32)
            ref_centres, ref_counts = _curvature_histogram(dummy_verts, dummy_faces, ref_sdf_sub)
        except Exception as exc:
            print(f"[app] Could not compute reference curvature histogram: {exc}")

    surface_area = _compute_surface_area(verts, faces)
    volume = _compute_volume(verts, faces)

    # ── Step 5: save mesh & build projection image ────────────────────────
    obj_path = _unique_obj_path()
    try:
        import trimesh
        mesh = trimesh.Trimesh(vertices=verts, faces=faces)
        mesh.export(str(obj_path), file_type="obj")
    except Exception as exc:
        raise RuntimeError(f"trimesh export failed: {exc}") from exc

    # ── Step 6: assemble Gradio outputs ──────────────────────────────────
    plotly_fig = _build_comparison_figure(gen_centres, gen_counts, ref_centres, ref_counts)
    metrics_md = _build_metrics_markdown(
        surface_area=surface_area,
        volume=volume,
        n_verts=len(verts),
        n_faces=len(faces),
        organelle=organelle,
        device_str=device_str,
        mode=mode_str,
    )

    return exemplar_status, str(obj_path), plotly_fig, metrics_md


def fn_get_projection(
    exemplar_file,
    organelle: str,
    cfg_scale: float,
    bio_weight: float,
) -> np.ndarray:
    """
    Generate the 2D EM-simulation projection of the (synthetic) organelle.
    Called separately so the projection can update instantly on param change.
    """
    file_path = (
        exemplar_file.name if hasattr(exemplar_file, "name") else exemplar_file
        if isinstance(exemplar_file, str) else None
    )
    if file_path and os.path.isfile(file_path) and file_path.endswith(".pt"):
        try:
            raw = torch.load(file_path, map_location="cpu", weights_only=False)
            sdf_full = raw[0].float().numpy()
            n = min(len(sdf_full), 8000)
            sdf_sub = sdf_full[:n]
            dummy_verts = np.random.randn(n, 3).astype(np.float32) * 0.5
            return _build_projection_image(dummy_verts, sdf_sub, grid_size=64)
        except Exception:
            pass

    verts, _, sdf = _build_demo_mesh(organelle, cfg_scale, bio_weight, rng_seed=0)
    return _build_projection_image(verts, sdf, grid_size=64)


# ===========================================================================
# Gradio Blocks UI definition
# ===========================================================================

_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

body, .gradio-container {
    font-family: 'Inter', system-ui, sans-serif !important;
    background: #060d14 !important;
}

/* ── dashboard title ── */
.app-title {
    text-align: center;
    padding: 28px 0 8px;
    background: linear-gradient(135deg, #0a1c30 0%, #091520 100%);
    border-radius: 14px;
    margin-bottom: 18px;
    border: 1px solid #1a3050;
}
.app-title h1 {
    font-size: 2.0rem !important;
    font-weight: 700;
    letter-spacing: -0.5px;
    background: linear-gradient(90deg, #63b3ff 0%, #a78bfa 60%, #34d399 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 6px !important;
}
.app-title p {
    color: #6a8eae;
    font-size: 0.88rem;
    margin: 0;
}

/* ── column headers ── */
.col-header {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    color: #63b3ff !important;
    padding: 6px 12px !important;
    background: rgba(99, 179, 255, 0.07) !important;
    border-left: 3px solid #63b3ff !important;
    border-radius: 0 6px 6px 0 !important;
    margin-bottom: 14px !important;
}

/* ── panels ── */
.panel {
    background: rgba(10, 22, 36, 0.75) !important;
    border: 1px solid #1a3050 !important;
    border-radius: 12px !important;
    padding: 18px !important;
    backdrop-filter: blur(6px);
}

/* ── generate button ── */
#gen-btn {
    background: linear-gradient(135deg, #2563eb 0%, #7c3aed 100%) !important;
    border: none !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    letter-spacing: 0.03em !important;
    padding: 14px 24px !important;
    border-radius: 10px !important;
    color: #ffffff !important;
    transition: all 0.25s ease !important;
    box-shadow: 0 4px 20px rgba(37, 99, 235, 0.35) !important;
    width: 100% !important;
}
#gen-btn:hover {
    background: linear-gradient(135deg, #1d4ed8 0%, #6d28d9 100%) !important;
    box-shadow: 0 6px 28px rgba(37, 99, 235, 0.55) !important;
    transform: translateY(-1px) !important;
}

/* ── exemplar status box ── */
.status-box textarea, .status-box .wrap {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.76rem !important;
    color: #7fddaa !important;
    background: #050e18 !important;
    border: 1px solid #1a3050 !important;
    border-radius: 8px !important;
}

/* ── metric markdown ── */
.metrics-md table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.84rem;
}
.metrics-md td, .metrics-md th {
    padding: 6px 10px;
    border-bottom: 1px solid #1a3050;
    color: #cdd8e8;
}
.metrics-md th { color: #63b3ff; font-weight: 600; }
.metrics-md code {
    background: #0d1c2b !important;
    color: #7fddaa !important;
    border-radius: 4px !important;
    padding: 1px 5px !important;
    font-size: 0.80rem !important;
    font-family: 'JetBrains Mono', monospace !important;
}

/* ── sliders ── */
input[type='range']::-webkit-slider-thumb { background: #2563eb !important; }
input[type='range']::-moz-range-thumb { background: #2563eb !important; }

/* ── dropdowns ── */
.gr-dropdown { border: 1px solid #1a3050 !important; background: #0a1520 !important; }

/* ── file upload ── */
.gr-file-drop { border: 2px dashed #1a3050 !important; background: #070f1a !important; }
"""

_MODE_BADGE = (
    "🔬 **Model mode** — running full TetraDiffusion pipeline"
    if not _DEMO_MODE else
    "🎭 **Demo mode** — synthetic organelle geometry  "
    "(launch with `--config_path` to enable real inference)"
)

with gr.Blocks(
    theme=gr.themes.Default(primary_hue="blue"),
    css=_CSS,
    title="TetraDiffusion · Organelle Diagnostic Dashboard",
) as demo:

    # ── Application header ──────────────────────────────────────────────
    gr.HTML(f"""
    <div class="app-title">
        <h1>⬡ TetraDiffusion · Organelle Generation Dashboard</h1>
        <p>SDF-based 3D shape synthesis over optimised tetrahedral grids ·
        Structural Biology Diagnostic Interface</p>
        <p style="margin-top:6px; color:#4a7a9b; font-size:0.80rem;">{_MODE_BADGE}</p>
    </div>
    """)

    with gr.Row(equal_height=False):

        # ==================================================================
        # COLUMN 1 — Control Desk
        # ==================================================================
        with gr.Column(scale=1, min_width=280):
            gr.Markdown("**1. Generation Parameters**", elem_classes="col-header")
            with gr.Group(elem_classes="panel"):
                exemplar_file = gr.File(
                    label="Exemplar Input  (.pt processed sample or .obj mesh)",
                    file_types=[".pt", ".obj"],
                    file_count="single",
                    elem_id="exemplar-upload",
                )
                organelle_dd = gr.Dropdown(
                    choices=["mitochondria", "lysosome", "golgi", "er", "exemplar-driven"],
                    value="mitochondria",
                    label="Organelle Profile Target",
                    elem_id="organelle-dd",
                )
                cfg_slider = gr.Slider(
                    minimum=1.0, maximum=5.0, value=1.5, step=0.1,
                    label="Classifier-Free Guidance  (CFG scale)",
                    elem_id="cfg-slider",
                )
                bio_slider = gr.Slider(
                    minimum=0.0, maximum=2.0, value=0.5, step=0.05,
                    label="Biophysical Weight  (λ_bio)",
                    elem_id="bio-slider",
                )
                generate_btn = gr.Button(
                    "⚡ Run Diffusion Isosurface Generation",
                    variant="primary",
                    elem_id="gen-btn",
                )
                exemplar_status = gr.Textbox(
                    label="Exemplar Injection Status",
                    lines=4,
                    interactive=False,
                    value="Upload an exemplar .pt file to inject a curvature style prior.",
                    elem_classes="status-box",
                    elem_id="exemplar-status",
                )

        # ==================================================================
        # COLUMN 2 — 3D Visual Verification Deck
        # ==================================================================
        with gr.Column(scale=2, min_width=420):
            gr.Markdown("**2. Visual Verification Deck**", elem_classes="col-header")
            with gr.Group(elem_classes="panel"):
                proj_image = gr.Image(
                    label="2D Conditioning Projection  (EM simulation mask · top-down Z-axis)",
                    show_label=True,
                    height=260,
                    elem_id="proj-image",
                )
                model_3d = gr.Model3D(
                    label="Generated Surface Mesh  (hardware-accelerated WebGL viewport)",
                    height=380,
                    clear_color=[0.04, 0.09, 0.14, 1.0],
                    elem_id="model-3d",
                )

        # ==================================================================
        # COLUMN 3 — Biological Analytics
        # ==================================================================
        with gr.Column(scale=2, min_width=380):
            gr.Markdown("**3. Morphological Verification Metrics**", elem_classes="col-header")
            with gr.Group(elem_classes="panel"):
                curv_plot = gr.Plot(
                    label="Surface Curvature Histogram  H ≈ −Δ_graph(φ)",
                    elem_id="curv-plot",
                )
                metrics_md = gr.Markdown(
                    value=(
                        "*Click **Run Diffusion Isosurface Generation** to compute metrics.*"
                    ),
                    elem_classes="metrics-md",
                    elem_id="metrics-md",
                )

    # ── Wire-up: main generation button ───────────────────────────────────
    generate_btn.click(
        fn=fn_run_inference,
        inputs=[exemplar_file, organelle_dd, cfg_slider, bio_slider],
        outputs=[exemplar_status, model_3d, curv_plot, metrics_md],
        api_name="generate",
    )

    # ── Wire-up: projection image updates on param change ─────────────────
    for _trigger_component in [organelle_dd, cfg_slider, bio_slider, exemplar_file]:
        _trigger_component.change(
            fn=fn_get_projection,
            inputs=[exemplar_file, organelle_dd, cfg_slider, bio_slider],
            outputs=[proj_image],
        )

    # ── Populate projection on page load ─────────────────────────────────
    demo.load(
        fn=fn_get_projection,
        inputs=[exemplar_file, organelle_dd, cfg_slider, bio_slider],
        outputs=[proj_image],
    )


# ===========================================================================
# Entry-point
# ===========================================================================
if __name__ == "__main__":
    print(f"[app] Starting TetraDiffusion Dashboard — {_MODE_BADGE.replace('**', '')}")
    print(f"[app] Scratch directory: {_SCRATCH_DIR}")
    demo.launch(
        server_port=_args.server_port,
        share=_args.share,
        show_error=True,
    )
