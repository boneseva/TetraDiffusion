from pathlib import Path
from collections import Counter, defaultdict
import tempfile
import os
import trimesh
import torch


# ---------------------------------------------------------------------------
# WandB 3-D mesh helpers
# ---------------------------------------------------------------------------

def _mesh_to_wandb_object3d(verts, mesh_color, faces, has_color):
    """
    Convert a mesh (verts/faces/colors as torch tensors) to a wandb.Object3D
    by writing a temporary OBJ file.  Returns None if wandb is unavailable.
    """
    try:
        import wandb
    except ImportError:
        return None

    mesh = trimesh.Trimesh(
        vertices=verts.cpu().numpy(),
        faces=faces.cpu().numpy(),
    )
    if has_color and mesh_color is not None:
        mesh.visual.vertex_colors = (
            torch.clamp(mesh_color, 0, 1) * 255
        ).to(torch.uint8).cpu().numpy()

    tmp = tempfile.NamedTemporaryFile(suffix=".obj", delete=False)
    tmp.close()
    try:
        mesh.export(tmp.name)
        obj3d = wandb.Object3D(open(tmp.name))
    finally:
        os.unlink(tmp.name)
    return obj3d


def log_augmentation_preview_to_wandb(dataset, n_samples: int = 2, step: int = 0) -> None:
    """
    Log a side-by-side comparison of the original training sample and its three
    axis-aligned reflections to WandB so you can visually verify that the
    augmentation is geometrically correct.

    Logged keys (WandB panel names):
        augmentation/sample_<i>/original
        augmentation/sample_<i>/flip_X
        augmentation/sample_<i>/flip_Y
        augmentation/sample_<i>/flip_Z

    Only runs if dataset.flip_perms is not None (i.e. augment=True in config).
    """
    try:
        import wandb
    except ImportError:
        print("[aug preview] wandb not available — skipping preview.")
        return

    if getattr(dataset, 'flip_perms', None) is None:
        return

    mask = dataset.mask  # 0 = keep

    def _raw_to_wandb_obj(sdf_raw, deform_raw, color_raw):
        """Apply mask → normalize → get_mesh → wandb.Object3D."""
        sdf       = sdf_raw[mask == 0]
        deform    = deform_raw[mask == 0, :]
        color     = color_raw[mask == 0, :]
        deform_xyz = deform[:, :3]          # drop weight-mask column
        sdf_n, deform_n, color_n = dataset._normalize(sdf, deform_xyz, color)
        data = torch.cat([sdf_n.unsqueeze(-1), deform_n, color_n], -1).unsqueeze(0)
        verts, mesh_color, faces = dataset.get_mesh(data)
        return _mesh_to_wandb_object3d(
            verts, mesh_color, faces,
            has_color=dataset.config.dataset.color,
        )

    axis_labels = ["flip_X", "flip_Y", "flip_Z"]
    panels = {}

    for sample_idx in range(min(n_samples, len(dataset.paths_train))):
        raw = torch.load(
            dataset.paths_train[sample_idx], map_location="cpu", weights_only=False
        )
        sdf_r, deform_r, color_r = raw[0], raw[1], raw[2]

        # Original
        try:
            obj = _raw_to_wandb_obj(sdf_r, deform_r, color_r)
            if obj is not None:
                panels[f"augmentation/sample_{sample_idx}/original"] = obj
        except Exception as e:
            print(f"[aug preview] original sample {sample_idx}: {e}")

        # Three axis flips
        for axis, perm in enumerate(getattr(dataset, 'flip_perms', [])):
            sdf_a    = sdf_r[perm]
            deform_a = deform_r[perm].clone()
            deform_a[:, axis] = -deform_a[:, axis]   # negate spatial component only
            color_a  = color_r[perm]
            try:
                obj = _raw_to_wandb_obj(sdf_a, deform_a, color_a)
                if obj is not None:
                    panels[f"augmentation/sample_{sample_idx}/{axis_labels[axis]}"] = obj
            except Exception as e:
                print(f"[aug preview] {axis_labels[axis]} sample {sample_idx}: {e}")

    if panels:
        wandb.log(panels, step=step)
        print(f"[aug preview] logged {len(panels)} mesh panel(s) to WandB.")
    else:
        print("[aug preview] no panels could be generated.")


# ---------------------------------------------------------------------------
# Existing helpers (save_mesh / plot_and_save_meshes) — now return file paths
# ---------------------------------------------------------------------------

def plot_and_save_meshes(all_meshes, dataset, config, name, k, file_prefix=None):
    """
    Plots and saves meshes from generated samples.

    Returns:
        list[Path]  Paths of every OBJ file written to disk.
    """
    step_counts = list(config.diffusion.sampling_steps)
    total_per_step = Counter(step_counts)
    seen_per_step = defaultdict(int)

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


def save_mesh(mesh_verts, mesh_color, mesh_faces, name, k, i, has_color, file_prefix=None):
    """
    Saves the mesh as an OBJ file.  Returns the Path of the written file.
    """
    mesh = trimesh.Trimesh(vertices=mesh_verts.cpu().numpy(), faces=mesh_faces.cpu().numpy())

    if has_color:
        mesh.visual.vertex_colors = mesh_color.to(torch.uint8).cpu().detach().numpy()

    filename = f"{k}_{i}.obj" if not file_prefix else f"{file_prefix}_{k}_{i}.obj"
    out_path = Path(name) / filename
    mesh.export(out_path)
    return out_path
