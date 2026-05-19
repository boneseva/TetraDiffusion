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


def biological_constraints_loss(
    x_start: Tensor,
    neighbors: Tensor,
    loss_type: str = "both",
    surface_softness: float = 0.15,
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

    return total

