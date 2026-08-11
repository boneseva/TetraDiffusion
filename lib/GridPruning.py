from typing import List, Tuple
import torch


def index_select(inputs: torch.Tensor, indices: torch.Tensor, dim: int) -> torch.Tensor:
    outputs = inputs.index_select(dim, indices.view(-1))
    if indices.dim() > 1:
        if dim < 0:
            dim += inputs.dim()
        output_shape = inputs.shape[:dim] + indices.shape + inputs.shape[dim + 1 :]
        outputs = outputs.view(*output_shape)
    return outputs


def tetra_subdivide(
    cur_verts_features: torch.Tensor,
    next_parents: torch.Tensor,
    device: str | torch.device,
    any_mode: bool = True,
) -> torch.Tensor:
    next_parents = next_parents.to(device)
    cur_verts_features = cur_verts_features.to(device)
    mask = next_parents == -1

    valid_vals = next_parents[~mask]
    next_parents[mask] = (
        int(valid_vals.max().item()) + 1
        if valid_vals.numel() > 0
        else next_parents.shape[0]
    )
    cur_verts_features = torch.nn.functional.pad(
        cur_verts_features, (0, 0, 0, 1, 0, 0), value=0
    )
    neighbors = index_select(cur_verts_features, next_parents, dim=1)

    if any_mode:
        return neighbors.mean(2)

    else:
        num_nans = torch.sum(torch.isnan(neighbors), 2)
        neighbors[0, num_nans.squeeze() < 5, :, :] = 0
        return neighbors.mean(2)


def mark_neighbors_as_deleted(neighbors: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mark deleted neighbor entries as -1 and filter out deleted vertices."""
    delete_me = mask.squeeze(1) == -1
    keep_me = mask.squeeze(1) != -1
    neighbors[delete_me, 0] = -1

    for i in range(len(neighbors)):
        c_neighbors = neighbors[i]
        if c_neighbors[0] == -1:
            continue

        for j in range(1, len(c_neighbors)):
            if c_neighbors[j] != -1:
                if neighbors[c_neighbors[j], 0] == -1:
                    neighbors[i, j] = -1

    return neighbors[keep_me, :]


def mapping_index_function(keep_me: torch.Tensor, vert_indices: torch.Tensor) -> torch.Tensor:
    """
    Create a mapping array from old vertex indices to new contiguous index positions
    after pruning unused vertices.
    """
    mapping_function = torch.full_like(keep_me, 1e12, dtype=torch.long)
    mapping_function[vert_indices] = torch.arange(0, len(vert_indices))
    return mapping_function


def crop_all_data(
    vert_list: List[torch.Tensor],
    tetra_list: List[torch.Tensor],
    neighbor_list: List[torch.Tensor],
    upsample_list: List[torch.Tensor],
    downsample_list: List[torch.Tensor],
    mask_list: List[torch.Tensor],
) -> Tuple[
    List[torch.Tensor],
    List[torch.Tensor],
    List[torch.Tensor],
    List[torch.Tensor],
    List[torch.Tensor],
]:
    """
    Crop and re-index all grid levels according to mask_list.

    Step 1: Crop vertex positions.
    Step 2: Update tetrahedra cell indices.
    Step 3: Update adjacency neighbor graphs.
    Step 4: Update multi-resolution upsample/downsample parent links.
    """
    print("[GridPruning] Cropping empty tetrahedral grid cells...")

    # Step 1: Remove unused vertices
    cropped_vertices = []
    index_mapping_list = []
    for c, verts in enumerate(vert_list):
        idx = mask_list[c].squeeze(1) != -1
        vert_indices = torch.arange(0, len(verts))
        index_mapping_list.append(vert_indices[idx])
        cropped_vertices.append(verts[idx, :])

    # Step 2: Update tetrahedra list
    cropped_tetrahedra = []
    for c, tets in enumerate(tetra_list):
        keep_me = mask_list[c].squeeze(1) != -1
        vert_indices = index_mapping_list[c]
        mapping_function = mapping_index_function(keep_me, vert_indices)
        mask = torch.isin(tets, vert_indices).all(dim=1)
        leftover_tets = tets[mask].long()
        leftover_tets = mapping_function[leftover_tets]
        cropped_tetrahedra.append(leftover_tets)

    # Step 3: Update neighbors graph
    cropped_neighbors = []
    for c, neigh in enumerate(neighbor_list):
        keep_me = mask_list[c].squeeze(1) != -1
        vert_indices = index_mapping_list[c]
        mapping_function = mapping_index_function(keep_me, vert_indices)
        cN = mark_neighbors_as_deleted(neigh, mask_list[c]).long()
        no_neighbor = cN == -1
        cN = mapping_function[cN]
        cN[no_neighbor] = -1
        cropped_neighbors.append(cN)

    # Step 4: Update upsample links
    cropped_upsample = []
    for c_idx in range(1, len(mask_list)):
        i = c_idx - 1
        keep_me = mask_list[c_idx].squeeze(1) != -1
        only_upsample_to_that = upsample_list[i][keep_me]
        iml = index_mapping_list[c_idx - 1]
        for j in range(len(only_upsample_to_that)):
            c_upsample = only_upsample_to_that[j]
            mask = ~torch.isin(c_upsample, iml)
            c_upsample[mask] = -1
            only_upsample_to_that[j] = c_upsample

        keep_me_before = mask_list[c_idx - 1].squeeze(1) != -1
        vert_indices_before = index_mapping_list[c_idx - 1]
        mapping_function = mapping_index_function(keep_me_before, vert_indices_before)
        no_neighbor = only_upsample_to_that == -1
        only_upsample_to_that = mapping_function[only_upsample_to_that]
        only_upsample_to_that[no_neighbor] = -1
        cropped_upsample.append(only_upsample_to_that)

    # Step 5: Update downsample links
    cropped_downsample = []
    for c, dsample in enumerate(downsample_list):
        keep_me = mask_list[c].squeeze(1) != -1
        only_downsample_to_that = dsample[keep_me]
        iml = index_mapping_list[c + 1]
        for j in range(len(only_downsample_to_that)):
            c_downsample = only_downsample_to_that[j]
            mask = ~torch.isin(c_downsample, iml)
            c_downsample[mask] = -1
            only_downsample_to_that[j] = c_downsample

        vert_indices_after = index_mapping_list[c + 1]
        keep_me_after = mask_list[c + 1].squeeze(1) != -1
        mapping_function = mapping_index_function(keep_me_after, vert_indices_after)
        no_neighbor = only_downsample_to_that == -1
        only_downsample_to_that = mapping_function[only_downsample_to_that]
        only_downsample_to_that[no_neighbor] = -1
        cropped_downsample.append(only_downsample_to_that)

    print("[GridPruning] Successfully pruned tetrahedral grid dataset.")
    return (
        cropped_vertices,
        cropped_tetrahedra,
        cropped_neighbors,
        cropped_upsample,
        cropped_downsample,
    )


def mask_cube(
    mask: torch.Tensor,
    vertices: List[torch.Tensor],
    tetra_cubes: List[torch.Tensor],
    neighbors: List[torch.Tensor],
    upsample: List[torch.Tensor],
    downsample: List[torch.Tensor],
    cuda_device: str | torch.device,
    any_mode: bool = True,
) -> Tuple[
    Tuple[
        List[torch.Tensor],
        List[torch.Tensor],
        List[torch.Tensor],
        List[torch.Tensor],
        List[torch.Tensor],
    ],
    torch.Tensor,
]:
    """
    Main entry point for tetrahedral grid pruning.

    Takes initial vertex occupancy masks and computes multi-resolution level masks.
    Calls crop_all_data to crop vertices, tetrahedra, neighbors, upsample, and downsample structures.
    """
    # Convert mask notation: 1 -> 0 (data/keep), 0 -> -1 (remove)
    mask = torch.where(mask == 1, 0, -1).unsqueeze(-1)
    mask = torch.where(mask == -1, 0, torch.nan)[None, ...]

    # Propagate mask across neighbors
    mask = tetra_subdivide(mask, neighbors[-1].clone(), cuda_device, any_mode=any_mode)
    mask = tetra_subdivide(mask, neighbors[-1].clone(), cuda_device, any_mode=any_mode)

    mask_list = []
    mask_list.append(torch.where(torch.nan_to_num(mask, -1) == -1, 0, -1)[0].cpu())

    for i in reversed(range(len(downsample))):
        mask = tetra_subdivide(mask, downsample[i].clone(), cuda_device, any_mode=any_mode)
        mask_list.append(torch.where(torch.nan_to_num(mask, -1) == -1, 0, -1)[0].cpu())

    mask_list = mask_list[::-1]
    pruned_structures = crop_all_data(
        vertices, tetra_cubes, neighbors, upsample, downsample, mask_list
    )
    return pruned_structures, mask_list[-1]

