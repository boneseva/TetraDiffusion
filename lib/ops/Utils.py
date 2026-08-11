from pathlib import Path
from collections import Counter
from typing import Any, List, Optional
import numpy as np
import torch
import trimesh


def _verts_to_point_cloud(verts: torch.Tensor, colors: Optional[torch.Tensor] = None) -> np.ndarray:
    pts = verts.cpu().float().numpy()
    if colors is not None:
        rgb = (torch.clamp(colors, 0, 1) * 255).to(torch.uint8).cpu().numpy()
        pts = np.concatenate([pts, rgb], axis=1)
    return pts



def log_training_samples_to_wandb(dataset: Any, n_samples: int = 4, step: int = 0) -> None:
    """
    Log point clouds of ground-truth training samples to WandB at startup
    to verify data normalization and grid alignment.
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
            sdf = sdf_r[mask == 0]
            deform = deform_r[mask == 0, :3]
            color = color_r[mask == 0, :]

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
    dataset: Any,
    config: Any,
    prefix: str = "generated",
) -> dict:
    """
    Convert a batch of raw model output tensors to WandB point-cloud entries.
    Extracts only the highest sampling-step outputs to avoid redundant logs.
    """
    try:
        import wandb
    except ImportError:
        return {}

    step_counts = list(config.diffusion.sampling_steps)
    n_steps = len(step_counts)
    n_total = len(all_meshes)

    if n_steps == 0 or n_total == 0:
        return {}

    samples_per_step = n_total // n_steps
    last_block_start = (n_steps - 1) * samples_per_step
    best_meshes = all_meshes[last_block_start:]
    best_step = step_counts[-1]

    panels = {}
    for j, mesh in enumerate(best_meshes):
        try:
            verts, mesh_color, _ = dataset.get_mesh(mesh.unsqueeze(0))
            pts = _verts_to_point_cloud(
                verts, mesh_color if config.dataset.color else None
            )
            panels[f"{prefix}/steps{best_step}_{j}"] = wandb.Object3D(pts)
        except Exception as e:
            print(f"[wandb] point cloud for mesh {j} (step {best_step}): {e}")

    return panels


def plot_and_save_meshes(
    all_meshes: torch.Tensor,
    dataset: Any,
    config: Any,
    name: str,
    k: int,
    file_prefix: Optional[str] = None,
) -> List[Path]:
    """
    Save a batch of generated meshes as OBJ files on disk.

    Returns:
        List of Path objects pointing to written .obj files.
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

        mesh_verts, mesh_color, mesh_faces = dataset.get_mesh(mesh)
        if config.dataset.color and mesh_color is not None:
            mesh_color = torch.clamp(mesh_color, 0, 1) * 255

        path = save_mesh(
            mesh_verts,
            mesh_color,
            mesh_faces,
            name,
            k,
            step_label,
            config.dataset.color,
            file_prefix=file_prefix,
        )
        saved_paths.append(path)

    return saved_paths


def save_mesh(
    mesh_verts: torch.Tensor,
    mesh_color: Optional[torch.Tensor],
    mesh_faces: torch.Tensor,
    name: str,
    k: int,
    i: str,
    has_color: bool,
    file_prefix: Optional[str] = None,
) -> Path:
    """Save a single 3D mesh as an OBJ file using Trimesh."""
    mesh = trimesh.Trimesh(
        vertices=mesh_verts.cpu().numpy(), faces=mesh_faces.cpu().numpy()
    )

    if has_color and mesh_color is not None:
        mesh.visual.vertex_colors = (
            mesh_color.to(torch.uint8).cpu().detach().numpy()
        )

    filename = f"{k}_{i}.obj" if not file_prefix else f"{file_prefix}_{k}_{i}.obj"
    out_path = Path(name) / filename
    mesh.export(out_path)
    return out_path

