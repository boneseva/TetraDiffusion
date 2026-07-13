from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
import trimesh
import torch


# ---------------------------------------------------------------------------
# WandB point-cloud + mesh helpers
# ---------------------------------------------------------------------------

def _verts_to_point_cloud(verts: torch.Tensor, colors: torch.Tensor | None = None) -> np.ndarray:
    """
    Build a numpy array suitable for wandb.Object3D point-cloud logging.
    Shape (N, 3) for XYZ-only, or (N, 6) for XYZ+RGB (R/G/B in 0-255).
    """
    pts = verts.cpu().float().numpy()
    if colors is not None:
        rgb = (torch.clamp(colors, 0, 1) * 255).to(torch.uint8).cpu().numpy()
        pts = np.concatenate([pts, rgb], axis=1)   # (N, 6)
    return pts


def log_training_samples_to_wandb(dataset, n_samples: int = 4, step: int = 0) -> None:
    """
    Log point clouds of real training samples to WandB at startup so you can
    verify the data loading / normalisation looks sensible.

    Logged keys:  training_data/sample_0, training_data/sample_1, …
    """
    try:
        import wandb
    except ImportError:
        return

    panels = {}
    for i in range(min(n_samples, len(dataset.paths_train))):
        try:
            raw = torch.load(dataset.paths_train[i], map_location="cpu", weights_only=False)
            sdf_r, deform_r, color_r = raw[0], raw[1], raw[2]

            mask = dataset.mask
            sdf    = sdf_r[mask == 0]
            deform = deform_r[mask == 0, :3]   # drop weight-mask column
            color  = color_r[mask == 0, :]

            sdf_n, deform_n, color_n = dataset._normalize(sdf, deform, color)
            data = torch.cat([sdf_n.unsqueeze(-1), deform_n, color_n], -1).unsqueeze(0)
            verts, mesh_color, _ = dataset.get_mesh(data)

            has_color = dataset.config.dataset.color
            pts = _verts_to_point_cloud(verts, mesh_color if has_color else None)
            panels[f"training_data/sample_{i}"] = wandb.Object3D(pts)
        except Exception as e:
            print(f"[wandb] training sample {i} point cloud failed: {e}")

    if panels:
        wandb.log(panels, step=step)
        print(f"[wandb] logged {len(panels)} training sample point cloud(s).")


def meshes_to_wandb_point_clouds(
    all_meshes: torch.Tensor,
    dataset,
    config,
    prefix: str = "generated",
) -> dict:
    """
    Convert a batch of raw model output tensors to WandB point-cloud entries.

    ``sample()`` runs one full denoising pass per entry in
    ``config.diffusion.sampling_steps`` and concatenates the results, so
    ``all_meshes`` has shape ``[num_samples * len(sampling_steps), ...]``.
    We only log the slice that corresponds to the *last* (highest step-count)
    sampling schedule so that each input organelle produces exactly one W&B
    entry rather than one per step-count variant.

    Returns a dict  {key: wandb.Object3D}  ready to pass to wandb.log().
    """
    try:
        import wandb
    except ImportError:
        return {}

    step_counts = list(config.diffusion.sampling_steps)
    n_steps = len(step_counts)          # number of sampling-step configs
    n_total = len(all_meshes)

    if n_steps == 0 or n_total == 0:
        return {}

    # Each sampling-step config produces an equal-sized block of meshes.
    # Only keep the last block (best / highest step count).
    samples_per_step = n_total // n_steps
    last_block_start = (n_steps - 1) * samples_per_step
    best_meshes = all_meshes[last_block_start:]
    best_step   = step_counts[-1]

    panels = {}
    for j, mesh in enumerate(best_meshes):
        try:
            if config.dataset.color:
                verts, mesh_color, _ = dataset.get_mesh(mesh.unsqueeze(0))
                pts = _verts_to_point_cloud(verts, mesh_color)
            else:
                verts, mesh_color, _ = dataset.get_mesh(mesh.unsqueeze(0))
                pts = _verts_to_point_cloud(verts)
            panels[f"{prefix}/steps{best_step}_{j}"] = wandb.Object3D(pts)
        except Exception as e:
            print(f"[wandb] point cloud for mesh {j} (step {best_step}): {e}")

    return panels


# ---------------------------------------------------------------------------
# Mesh save helpers
# ---------------------------------------------------------------------------

def plot_and_save_meshes(all_meshes, dataset, config, name, k, file_prefix=None):
    """
    Save generated meshes as OBJ files.

    Returns:
        list[Path]  Paths of every OBJ file written to disk.
    """
    step_counts = list(config.diffusion.sampling_steps)
    total_per_step = Counter(step_counts)
    seen_per_step = Counter()

    saved_paths = []
    for i, mesh in enumerate(all_meshes):
        mesh = mesh.unsqueeze(0)
        step_count = step_counts[i]
        variant_index = seen_per_step[step_count]
        seen_per_step[step_count] += 1

        step_label = f"stepsize_{step_count}"
        if total_per_step[step_count] > 1:
            step_label = f"{step_label}_variant_{variant_index}"

        if config.dataset.color:
            mesh_verts, mesh_color, mesh_faces = dataset.get_mesh(mesh)
            mesh_color = torch.clamp(mesh_color, 0, 1) * 255
        else:
            mesh_verts, mesh_color, mesh_faces = dataset.get_mesh_wo_color(mesh)

        path = save_mesh(
            mesh_verts, mesh_color, mesh_faces,
            name, k, step_label, config.dataset.color,
            file_prefix=file_prefix,
        )
        saved_paths.append(path)

    return saved_paths


def save_mesh(mesh_verts, mesh_color, mesh_faces, name, k, i, has_color, file_prefix=None):
    """
    Save the mesh as an OBJ file.  Returns the Path of the written file.
    """
    mesh = trimesh.Trimesh(vertices=mesh_verts.cpu().numpy(), faces=mesh_faces.cpu().numpy())

    if has_color:
        mesh.visual.vertex_colors = mesh_color.to(torch.uint8).cpu().detach().numpy()

    filename = f"{k}_{i}.obj" if not file_prefix else f"{file_prefix}_{k}_{i}.obj"
    out_path = Path(name) / filename
    mesh.export(out_path)
    return out_path
