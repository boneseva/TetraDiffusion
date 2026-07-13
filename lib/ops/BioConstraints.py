"""
BioConstraints.py — Biologically-inspired regularisation losses for tetrahedral
diffusion-based organelle shape generation.

Motivation (from dissertation proposal section 2.2 / 3.1)
----------------------------------------------------------
Organelle membranes are lipid bilayers whose equilibrium shape is governed by the
Helfrich bending energy:

    E_bend = (κ/2) ∫ H² dA  +  κ_G ∫ K dA

where H is the mean curvature and K is the Gaussian curvature.  Low bending
energy ↔ smooth, gently curved surfaces — exactly what biological membranes
(mitochondria, ER, lysosomes, Golgi) exhibit.

Two differentiable proxy losses are provided:

1. Laplacian displacement loss  (``laplacian``)
   Penalises high-frequency roughness in the vertex-displacement field.
   In the graph signal-processing sense the uniform graph Laplacian of d is:
       Δd_i = d_i − mean_{j∈N(i)} d_j
   Minimising ||Δd||² enforces a smooth deformation field, which produces
   smooth surface geometry. This is the same regulariser used by DMTet during
   the preprocessing fitting phase.

2. Mean-curvature / bending-energy loss  (``curvature``)
   For a signed distance field φ with |∇φ| ≈ 1 (a valid SDF), the graph
   Laplacian approximates the mean curvature:
       H ≈ −Δ_graph(φ)
   We weight the loss by exp(−|φ|/σ) to concentrate on surface vertices
   (where |φ| ≈ 0).  Minimising ||Δ_graph(φ)||² · exp(−|φ|/σ) is therefore
   a differentiable proxy for Helfrich bending energy.

Both losses are applied to the model's *predicted clean sample* x_start,
recovered during training using the diffusion model's own v-prediction or
ε-prediction.  This is analogous to manifold-constrained diffusion (Chung
et al., 2022) where constraints are enforced via the projected gradient of
the clean-sample estimate at each denoising step.

SNR weighting
-------------
At high noise levels the predicted x_start is essentially random noise and
the geometry constraints would provide noisy, uninformative gradients.  Both
losses are therefore multiplied by a soft SNR weight:

    w(t) = SNR(t) / (SNR(t) + 1)  ∈ [0, 1)

which smoothly interpolates from 0 at t=1 (pure noise) to ~1 at t≈0 (clean),
ensuring the biological prior only influences the model when its predictions
are already roughly meaningful.
"""

from __future__ import annotations

import torch
from torch import Tensor
from typing import Callable, Dict, List, Optional


# ---------------------------------------------------------------------------
# Internal helper: differentiable graph-Laplacian of a per-vertex signal
# ---------------------------------------------------------------------------

def _graph_laplacian(signal: Tensor, neighbors: Tensor) -> Tensor:
    """
    Compute the uniform graph Laplacian  Δ_i = x_i − mean_{j∈N(i)} x_j
    for every vertex on the tetrahedral graph.

    Parameters
    ----------
    signal : Tensor [B, N, C]
        Per-vertex signal (features or SDF).
    neighbors : Tensor [N, K]
        Neighbour-index table.  −1 marks absent / padding entries.
        Assumed to already be restricted to the active (unmasked) vertex set.

    Returns
    -------
    Tensor [B, N, C]  — same shape as ``signal``.
    """
    B, N, C = signal.shape
    device = signal.device

    valid = (neighbors >= 0)              # [N, K]  bool
    n_valid = valid.float().sum(dim=1, keepdim=True).clamp(min=1.0)  # [N, 1]

    # Redirect −1 indices to a throw-away padding slot at position N
    neigh_safe = neighbors.clone()
    neigh_safe[~valid] = N

    # Pad signal with one zero vertex so index N is always harmless
    padded = torch.cat(
        [signal, signal.new_zeros(B, 1, C)], dim=1
    )                                     # [B, N+1, C]

    # Gather neighbour features → [B, N, K, C]
    neigh_feats = padded[:, neigh_safe, :]

    # Zero-out invalid entries and compute uniform mean
    neigh_feats = neigh_feats * valid.unsqueeze(0).unsqueeze(-1)  # broadcast
    mean_neigh  = neigh_feats.sum(dim=2) / n_valid.unsqueeze(0)   # [B, N, C]

    return signal - mean_neigh


# ---------------------------------------------------------------------------
# Public loss functions
# ---------------------------------------------------------------------------

def laplacian_displacement_loss(
    x_start: Tensor,
    neighbors: Tensor,
) -> Tensor:
    """
    Uniform Laplacian smoothness loss on the displacement channels of the
    predicted clean sample.

    Biological rationale
    --------------------
    Organelle membranes are smooth continuous surfaces.  High-frequency
    oscillations in the deformation field produce kinked, jagged surfaces.
    Penalising ||Δd||² enforces membrane smoothness.

    Parameters
    ----------
    x_start : Tensor [B, N, C]
        Predicted clean sample in the diffusion latent space.
        Channel layout: [sdf(1) | disp_x, disp_y, disp_z(3) | color(0 or 3)].
    neighbors : Tensor [N, K]
        Neighbour table (−1 = absent).

    Returns
    -------
    Scalar Tensor — mean squared Laplacian of the displacement field.
    """
    disp = x_start[:, :, 1:4]          # channels 1,2,3 = dx, dy, dz
    lap  = _graph_laplacian(disp, neighbors)  # [B, N, 3]
    return lap.pow(2).mean()


def mean_curvature_loss(
    x_start: Tensor,
    neighbors: Tensor,
    surface_softness: float = 0.15,
) -> Tensor:
    """
    Mean-curvature / bending-energy proxy loss on the SDF channel of the
    predicted clean sample.

    Biological rationale
    --------------------
    For a valid SDF φ with |∇φ| ≈ 1, the graph Laplacian approximates
    mean curvature:  H ≈ −Δ_graph(φ).  Penalising ||Δ_graph(φ)||² near
    the zero-level set (surface vertices) is a differentiable proxy for the
    Helfrich bending energy E = (κ/2) ∫ H² dA.

    Parameters
    ----------
    x_start : Tensor [B, N, C]
        Predicted clean sample in the diffusion latent space.
        Channel 0 = normalised SDF.
    neighbors : Tensor [N, K]
        Neighbour table (−1 = absent).
    surface_softness : float
        Controls the width of the surface weighting kernel
        exp(−|φ|/surface_softness).  Smaller = tighter to the surface.

    Returns
    -------
    Scalar Tensor — surface-weighted mean squared graph-Laplacian of SDF.
    """
    sdf  = x_start[:, :, 0:1]          # [B, N, 1]  — keep dim for _graph_laplacian
    lap  = _graph_laplacian(sdf, neighbors)   # [B, N, 1]

    # Weight by proximity to surface
    surface_weight = torch.exp(-sdf.abs() / surface_softness)  # [B, N, 1]

    return (lap.pow(2) * surface_weight).mean()


def eikonal_loss(
    x_start: Tensor,
    neighbors: Tensor,
    edge_lengths: Tensor,
) -> Tensor:
    """
    Eikonal regularisation loss: penalises deviations of the per-vertex
    SDF gradient magnitude from 1.

    Biological / mathematical rationale
    ------------------------------------
    The curvature approximation H ≈ −Δ_graph(φ) is valid only when φ
    satisfies the Eikonal equation |∇φ| = 1 (i.e., φ is a true signed
    distance field).  During diffusion training the model's intermediate
    clean-sample estimate x̂_0 is not guaranteed to satisfy this property.
    Penalising deviations from |∇φ| ≈ 1 protects the curvature estimate
    against SDF-gradient collapse or explosion and improves the overall
    geometric quality of generated meshes.

    Implementation
    --------------
    For each vertex i and neighbour j we form a directional-derivative
    estimate:

        ∂φ/∂r_{ij} ≈ (φ_j − φ_i) / ‖v_j − v_i‖

    The squared gradient magnitude at i is approximated by the mean of
    squared directional derivatives over N(i).  The loss penalises
    (|∇φ_i|² − 1)².

    Parameters
    ----------
    x_start : Tensor [B, N, C]
        Predicted clean sample; channel 0 is the SDF.
    neighbors : Tensor [N, K]
        Neighbour-index table (−1 = absent / padding).
    edge_lengths : Tensor [N, K]
        Precomputed Euclidean length of each graph edge.
        Invalid entries (corresponding to neighbors == −1) should be
        set to 1.0 to avoid division-by-zero.

    Returns
    -------
    Scalar Tensor.
    """
    B, N, C = x_start.shape
    sdf = x_start[:, :, 0]         # [B, N]

    valid = (neighbors >= 0)        # [N, K]
    n_valid = valid.float().sum(dim=1, keepdim=True).clamp(min=1.0)  # [N, 1]

    neigh_safe = neighbors.clone()
    neigh_safe[~valid] = N          # redirect to throw-away padding slot

    # Pad SDF with one zero vertex so index N is harmless
    padded_sdf = torch.cat([sdf, sdf.new_zeros(B, 1)], dim=1)  # [B, N+1]

    # Gather neighbour SDF values → [B, N, K]
    neigh_sdf = padded_sdf[:, neigh_safe]

    # Directional derivatives: (φ_j − φ_i) / ‖v_j − v_i‖  → [B, N, K]
    inv_len = (1.0 / edge_lengths.unsqueeze(0).clamp(min=1e-6))  # [1, N, K]
    dd = (neigh_sdf - sdf.unsqueeze(-1)) * inv_len               # [B, N, K]

    # Zero out invalid entries
    dd = dd * valid.unsqueeze(0).float()

    # Mean squared directional derivative = approximation of |∇φ|²
    grad_sq = dd.pow(2).sum(dim=2) / n_valid.unsqueeze(0)        # [B, N]

    # Eikonal penalty: (|∇φ|² − 1)²
    return (grad_sq - 1.0).pow(2).mean()


def sdf_background_loss(
    x_start: Tensor,
    bg_threshold: float = 0.3,
) -> Tensor:
    """
    Surface-weighted SDF background hinge loss.

    Problem it solves
    -----------------
    The diffusion model must predict, for *every* vertex in the tetrahedral
    grid, whether it is inside (SDF < 0) or outside (SDF > 0) the organelle.
    The main MSE diffusion loss treats all vertices equally, so a handful of
    spuriously negative predictions far from the true organelle surface can
    accumulate undetected — yet each such patch becomes a ghost mesh component
    when marching tets applies the hard ``torch.sign`` threshold at extraction
    time.

    This loss adds a smooth hinge penalty:

        L_bg = mean( bg_weight(φ_gt) · ReLU(−φ_pred) )

    where:
        bg_weight(φ_gt) = ReLU( φ_gt − bg_threshold )   (zero inside surface band)
        bg_threshold    = normalised SDF cutoff above which a vertex is
                          considered "definitely outside" the organelle

    Vertices near the true surface (|φ_gt| < bg_threshold) are exempted so
    the loss does not fight the geometry near the zero-level set.  Background
    vertices (φ_gt > bg_threshold) are penalised quadratically when the
    predicted SDF goes negative.  The weight grows with distance from the
    surface, making distant spurious predictions pay more.

    Parameters
    ----------
    x_start : Tensor [B, N, C]
        Predicted clean sample in diffusion latent space.
        Channel 0 is the normalised SDF; range ≈ [−1, 1] after
        ``normalize_to_neg_one_to_one``.
    bg_threshold : float
        Normalised SDF value above which a vertex is treated as background.
        Typical range 0.2–0.5.  Corresponds to a fraction of the full
        sdfs_max − sdfs_min range in physical units.

    Returns
    -------
    Scalar Tensor — mean background hinge penalty.
    """
    sdf_pred = x_start[:, :, 0]                      # [B, N]  predicted SDF (normalised)
    # Ground-truth sign comes from x_start itself — the "clean" x_start
    # passed here is the ground-truth sample, not the model's prediction.
    # Both are needed; see usage in DDPM.p_losses.
    bg_weight = torch.relu(sdf_pred.detach() - bg_threshold)   # [B, N]  ≥ 0 outside band
    hinge     = torch.relu(-sdf_pred)                          # [B, N]  ≥ 0 when pred < 0
    return (bg_weight * hinge).mean()


def biological_constraints_loss(
    x_start: Tensor,
    neighbors: Tensor,
    loss_type: str = "both",
    surface_softness: float = 0.15,
    edge_lengths: Optional[Tensor] = None,
    eikonal_weight: float = 0.0,
) -> Tensor:
    """
    Combined biological constraint loss.

    Parameters
    ----------
    x_start : Tensor [B, N, C]
    neighbors : Tensor [N, K]
    loss_type : str
        ``"laplacian"``  — displacement smoothness only.
        ``"curvature"``  — mean curvature / bending energy only.
        ``"both"``       — sum of both (default).
    surface_softness : float
        Passed to ``mean_curvature_loss``.
    edge_lengths : Tensor [N, K] or None
        Required when ``eikonal_weight > 0``.  Precomputed Euclidean
        edge lengths with 1.0 in invalid (−1 neighbour) slots.
    eikonal_weight : float
        Weight for the optional Eikonal regularisation term.  When zero
        (default) the term is not evaluated.

    Returns
    -------
    Scalar Tensor.
    """
    total = x_start.new_zeros(1).squeeze()

    if loss_type in ("laplacian", "both"):
        total = total + laplacian_displacement_loss(x_start, neighbors)

    if loss_type in ("curvature", "both"):
        total = total + mean_curvature_loss(
            x_start, neighbors, surface_softness=surface_softness
        )

    if eikonal_weight > 0.0 and edge_lengths is not None:
        total = total + eikonal_weight * eikonal_loss(
            x_start, neighbors, edge_lengths
        )

    return total


# ---------------------------------------------------------------------------
# ExemplarLossProfile — style prior derived from a single reference sample
# ---------------------------------------------------------------------------

class ExemplarLossProfile:
    """
    Derives a curvature-based style prior from a single pre-processed exemplar
    file (``sample_*.pt`` / ``sample.pth`` format containing ``[sdf, disp, color]``).

    At construction time the target curvature statistics (mean and std of the
    surface-weighted graph-Laplacian of the SDF) and the maximum interior depth
    are computed under ``torch.no_grad()`` and cached.  During training these
    are compared against the model's predicted clean sample to impose a soft
    morphological shape prior.

    Parameters
    ----------
    exemplar_path : str
        Path to a ``.pt`` file whose first element is a per-vertex SDF tensor.
    mask : torch.Tensor [N_full]
        Boolean pruning mask (0 = kept, non-zero = pruned).
    neighbors : torch.Tensor [N_kept, K]
        Finest-resolution graph neighbour table for the pruned vertex set
        (−1 in absent/padding slots).
    """

    def __init__(
        self,
        exemplar_path: str,
        mask: torch.Tensor,
        neighbors: torch.Tensor,
    ) -> None:
        self._neighbors_cpu = neighbors.long().cpu()   # [N_kept, K]

        # ── Load and isolate kept vertices ──────────────────────────────────
        raw = torch.load(exemplar_path, weights_only=False)
        # raw is [sdf, disp, color] — take the SDF channel
        sdf_full: torch.Tensor = raw[0].float().cpu()  # [N_full]
        self.target_sdf = sdf_full[mask == 0]          # [N_kept]

        # ── Compute target curvature statistics (no gradient tape needed) ──
        with torch.no_grad():
            # Surface-weighted SDF for curvature proxy
            sdf_b = self.target_sdf.unsqueeze(0).unsqueeze(-1)  # [1, N, 1]
            surface_weight = torch.exp(-sdf_b.abs() / 0.15)     # [1, N, 1]

            lap = _graph_laplacian(sdf_b, self._neighbors_cpu)   # [1, N, 1]
            weighted_lap = (lap * surface_weight).squeeze()       # [N]

            self.target_mean_curvature: float = float(weighted_lap.mean().item())
            self.target_std_curvature: float  = float(weighted_lap.std().item())

            # Maximum interior depth (absolute value of the most negative SDF)
            interior = self.target_sdf[self.target_sdf < 0]
            self.target_max_depth: float = (
                float(interior.min().abs().item()) if interior.numel() > 0 else 1.0
            )

        print(
            f"[ExemplarLossProfile] Loaded exemplar '{exemplar_path}': "
            f"mean_curvature={self.target_mean_curvature:.4f}, "
            f"std_curvature={self.target_std_curvature:.4f}, "
            f"max_depth={self.target_max_depth:.4f}"
        )

    # ------------------------------------------------------------------
    def __call__(
        self,
        x_start: Tensor,
        neighbors: Tensor,
        kwargs: dict,
    ) -> Tensor:
        """
        Compute exemplar-style regularisation losses.

        1. **Curvature matching** — penalise the MSE between the
           surface-weighted curvature mean of ``x_start`` and the target.
        2. **Depth cap** — penalise interior SDF values whose magnitude
           exceeds the exemplar's deepest point, discouraging over-inflation.

        Parameters
        ----------
        x_start : Tensor [B, N, C]
            Predicted clean sample (normalised diffusion space).
        neighbors : Tensor [N, K]
            Neighbour table passed from the DDPM training loop.
        kwargs : dict
            Extra keyword arguments (unused; accepted for interface compatibility).

        Returns
        -------
        Scalar Tensor.
        """
        dev = x_start.device
        neigh = neighbors.to(dev)               # [N, K]

        sdf = x_start[:, :, 0:1]               # [B, N, 1]
        surface_weight = torch.exp(-sdf.abs() / 0.15)

        lap  = _graph_laplacian(sdf, neigh)     # [B, N, 1]
        pred_curv = (lap * surface_weight)      # [B, N, 1]

        # Curvature mean MSE against target
        pred_mean = pred_curv.mean()
        target_mean = torch.tensor(
            self.target_mean_curvature, dtype=x_start.dtype, device=dev
        )
        curv_loss = torch.nn.functional.mse_loss(pred_mean, target_mean)

        # Depth cap: penalise interior vertices that are deeper than exemplar
        sdf_vals = x_start[:, :, 0]            # [B, N]
        interior_excess = (-sdf_vals - self.target_max_depth).clamp(min=0.0)
        depth_loss = interior_excess.pow(2).mean()

        return curv_loss + depth_loss


# ---------------------------------------------------------------------------
# OrganelleLossRegistry — plugin registry for custom bio losses
# ---------------------------------------------------------------------------

class OrganelleLossRegistry:
    """
    A lightweight registry that maps organelle names to lists of custom
    loss modules (callables).  The baseline Laplacian + curvature losses are
    always evaluated regardless of the registered custom losses.

    Usage
    -----
    ::

        organelle_loss_registry.register_loss("mito", my_mito_loss_fn)
        bio = organelle_loss_registry.compute_loss(
            "mito", x_start, neighbors, surface_softness=0.1
        )
    """

    def __init__(self) -> None:
        self._registry: Dict[str, List[Callable]] = {}

    # ------------------------------------------------------------------
    def register_loss(self, organelle_name: str, loss_fn: Callable) -> None:
        """
        Register a custom loss callable for a given organelle name.

        Multiple callables may be registered for the same name; they are
        accumulated in order.

        Parameters
        ----------
        organelle_name : str
            Arbitrary tag used as a key (e.g. ``"exemplar"``, ``"mito"``).
        loss_fn : Callable
            Must accept ``(x_start, neighbors, kwargs) -> Scalar Tensor``.
        """
        self._registry.setdefault(organelle_name, []).append(loss_fn)

    # ------------------------------------------------------------------
    def compute_loss(
        self,
        organelle_name: str,
        x_start: Tensor,
        neighbors: Tensor,
        **kwargs,
    ) -> Tensor:
        """
        Compute the combined biological loss for one training step.

        Always includes the baseline Laplacian-displacement and
        mean-curvature losses.  If ``organelle_name`` has registered
        custom callables, their outputs are accumulated on top.

        Parameters
        ----------
        organelle_name : str
            Key to look up in the registry.
        x_start : Tensor [B, N, C]
            Predicted clean sample.
        neighbors : Tensor [N, K]
            Graph neighbour table.
        **kwargs
            Forwarded to both the baseline helper and custom callables.
            Recognised keys: ``surface_softness``, ``edge_lengths``,
            ``eikonal_weight``.

        Returns
        -------
        Scalar Tensor.
        """
        surface_softness  = kwargs.get("surface_softness", 0.15)
        edge_lengths       = kwargs.get("edge_lengths", None)
        eikonal_weight     = kwargs.get("eikonal_weight", 0.0)
        sdf_bg_loss_weight = kwargs.get("sdf_bg_loss_weight", 0.0)
        sdf_bg_threshold   = kwargs.get("sdf_bg_threshold", 0.3)

        # ── Baseline losses (always active) ─────────────────────────────
        total = laplacian_displacement_loss(x_start, neighbors)
        total = total + mean_curvature_loss(
            x_start, neighbors, surface_softness=surface_softness
        )

        if eikonal_weight > 0.0 and edge_lengths is not None:
            total = total + eikonal_weight * eikonal_loss(
                x_start, neighbors, edge_lengths
            )

        # ── Surface-weighted SDF background loss ─────────────────────────
        if sdf_bg_loss_weight > 0.0:
            total = total + sdf_bg_loss_weight * sdf_background_loss(
                x_start, bg_threshold=sdf_bg_threshold
            )

        # ── Custom registered losses ─────────────────────────────────────
        for loss_fn in self._registry.get(organelle_name, []):
            total = total + loss_fn(x_start, neighbors, kwargs)

        return total


# ---------------------------------------------------------------------------
# Global singleton registry — imported and used by DDPM.py and main.py
# ---------------------------------------------------------------------------

organelle_loss_registry = OrganelleLossRegistry()
