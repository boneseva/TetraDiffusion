import trimesh
import torch


def plot_and_save_meshes(all_meshes, dataset, config, name, k):
    """
    Plots and saves meshes from generated samples.

    Args:
        all_meshes (tensor): Tensor containing the generated meshes.
        dataset (object): Dataset object containing mesh-related functions.
        config (object): Configuration object with dataset settings.
        name (str): Base name for saving mesh files.
        k (int): Index for naming the files.
    """
    for i, mesh in enumerate(all_meshes):
        mesh = mesh.unsqueeze(0)

        if config.dataset.color:
            mesh_verts, mesh_color, mesh_faces = dataset.get_mesh(mesh)
            mesh_color = torch.clamp(mesh_color, 0, 1) * 255
        else:
            mesh_verts, mesh_color, mesh_faces = dataset.get_mesh_wo_color(mesh)

        save_mesh(mesh_verts, mesh_color, mesh_faces, name, k, f"stepsize_{config.diffusion.sampling_steps[i]}", config.dataset.color)


def save_mesh(mesh_verts, mesh_color, mesh_faces, name, k, i, has_color):
    """
    Saves the mesh as an OBJ file.

    Args:
        mesh_verts (tensor): Vertices of the mesh.
        mesh_color (tensor): Colors of the mesh vertices.
        mesh_faces (tensor): Faces of the mesh.
        name (str): Base name for saving mesh files.
        k (int): Index for naming the files.
        i (int): Index for naming the files.
        has_color (bool): Flag indicating if the mesh has color.
    """
    mesh = trimesh.Trimesh(vertices=mesh_verts.cpu().numpy(), faces=mesh_faces.cpu().numpy())

    if has_color:
        mesh.visual.vertex_colors = mesh_color.to(torch.uint8).cpu().detach().numpy()

    mesh.export(f"{name}/{k}_{i}.obj")
