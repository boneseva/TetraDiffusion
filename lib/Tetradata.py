import torch
from pathlib import Path
from torch.utils.data.dataset import Dataset
import numpy as np
from glob import glob
from lib.GridPruning import mask_cube
from tqdm import tqdm
import pandas as pd
import re as _re




def marching_cube_get_idx(sdf_n, tet_fx4):
    num_triangles_table = torch.tensor([0, 1, 1, 2, 1, 2, 2, 1, 1, 2, 2, 1, 2, 1, 1, 0], dtype=torch.long,  device=sdf_n.device)
    with torch.no_grad():
        occ_n = sdf_n > 0
        occ_fx4 = occ_n[tet_fx4.reshape(-1)].reshape(-1, 4)
        occ_sum = torch.sum(occ_fx4, -1)
        valid_tets = (occ_sum > 0) & (occ_sum < 4)

        v_id = torch.pow(2, torch.arange(4, dtype=torch.long, device=sdf_n.device))
        tetindex = (occ_fx4[valid_tets] * v_id.unsqueeze(0)).sum(-1)
        num_triangles = num_triangles_table[tetindex]

        tet_idx = torch.arange(tet_fx4.shape[0], device=sdf_n.device)[valid_tets]
        tet_idx = torch.cat(
            (tet_idx[num_triangles == 1], tet_idx[num_triangles == 2].unsqueeze(-1).expand(-1, 2).reshape(-1)), dim=0)
        return tet_idx


class MeshLoader(Dataset):

    def __init__(self, config, device, cuda_device, accelerator):
        super(MeshLoader, self).__init__()
        self.device = device
        self.accelerator = accelerator

        self.cuda_device = cuda_device
        self.config = config
        self.grid_res = self.config.dataset.grid_res

        self.triangle_table = torch.tensor([
            [-1, -1, -1, -1, -1, -1],
            [1, 0, 2, -1, -1, -1],
            [4, 0, 3, -1, -1, -1],
            [1, 4, 2, 1, 3, 4],
            [3, 1, 5, -1, -1, -1],
            [2, 3, 0, 2, 5, 3],
            [1, 4, 0, 1, 5, 4],
            [4, 2, 5, -1, -1, -1],
            [4, 5, 2, -1, -1, -1],
            [4, 1, 0, 4, 5, 1],
            [3, 2, 0, 3, 5, 2],
            [1, 3, 5, -1, -1, -1],
            [4, 1, 2, 4, 3, 1],
            [3, 0, 4, -1, -1, -1],
            [2, 0, 1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1]
        ], dtype=torch.long, device=self.device)

        self.num_triangles_table = torch.tensor([0, 1, 1, 2, 1, 2, 2, 1, 1, 2, 2, 1, 2, 1, 1, 0], dtype=torch.long,device=self.device)

        cube_range = self.config.dataset.cube_range

        self.vertices = [torch.tensor(np.load(f"tetrahedra/{self.grid_res}/{i}_tets.npz")['vertices'], dtype=torch.float32) for i in cube_range]
        self.tetra_cubes = [torch.tensor(np.load(f"tetrahedra/{self.grid_res}/{i}_tets.npz")['indices'], dtype=torch.int32) for i in cube_range]
        self.neighbors = [torch.load(f"tetrahedra/{self.grid_res}/neighbors_{i}_sorted.pth", map_location="cpu", weights_only=False).int() for i in cube_range]
        self.upsample = [torch.load(f"tetrahedra/{self.grid_res}/upsample_{i}_sorted.pth", map_location="cpu", weights_only=False)[0] for i in cube_range[:-1]]
        self.downsample = [torch.load(f"tetrahedra/{self.grid_res}/downsample_{i}_sorted.pth", map_location="cpu", weights_only=False)[0] for i in cube_range[1:]]

        for i in range(len(self.vertices)):
            self.vertices[i] = self.vertices[i] - torch.mean(self.vertices[i], 0)

        self.tet_verts = self.vertices[-1].to(self.device)
        self.tet_faces = self.tetra_cubes[-1].to(self.device)

        edges = torch.tensor([0, 1, 0, 2, 0, 3, 1, 2, 1, 3, 2, 3], dtype=torch.long, device="cpu")
        all_edges = self.tet_faces[:, edges].reshape(-1, 2)
        all_edges_sorted = torch.sort(all_edges, dim=1)[0]
        self.all_edges = torch.unique(all_edges_sorted, dim=0)
        self.base_tet_edges = torch.tensor([0, 1, 0, 2, 0, 3, 1, 2, 1, 3, 2, 3], dtype=torch.long, device='cpu')
        self.num_points = len(self.tet_verts)

        self.mask_verts = torch.zeros((len(self.tet_verts))).int()

        self.paths_train, self.paths_test = self._init_gt_iterative()
        self.accelerator.wait_for_everyone()

        if self.config.dataset.grid_pruning:
            (self.vertices, self.tetra_cubes, self.neighbors, self.upsample, self.downsample), self.mask = mask_cube(
                self.mask_verts, self.vertices, self.tetra_cubes, self.neighbors, self.upsample, self.downsample,
                cuda_device)
            self.mask = self.mask.squeeze(1).int()
            print("using", np.round(len(self.vertices[-1]) / len(self.tet_verts) * 100, 2), "% of the data")
            self.tet_verts = self.vertices[-1]
            self.tet_faces = self.tetra_cubes[-1]
        else:
            self.mask = torch.zeros_like(self.tet_verts[:, 0]).cpu()
        self.accelerator.wait_for_everyone()
        self.get_statistics()
        self.accelerator.wait_for_everyone()
        self.tet_verts = self.tet_verts.cpu()
        torch.cuda.empty_cache()

        # ------------------------------------------------------------------
        # Optional: precompute projection grid indices for image conditioning.
        # Indices are fixed (same tetrahedral grid for all samples); SDF values
        # vary per sample — that's what makes each projection unique.
        # ------------------------------------------------------------------
        self.use_image_cond = bool(
            getattr(getattr(self.config, 'image_cond', None), 'enabled', False)
        )
        if self.use_image_cond:
            self._proj_grid_size = int(
                getattr(self.config.image_cond, 'proj_size', 64)
            )
            self._precompute_projection_indices()
            print(f"[MeshLoader] Image projection enabled  "
                  f"(grid {self._proj_grid_size}×{self._proj_grid_size})")


    def _precompute_projection_indices(self):
        """
        Precompute X and Y pixel indices for projecting tetrahedral vertex
        positions onto a 2D grid.  Called once at init; result is reused in
        every __getitem__ call.  The vertex positions are fixed across
        all samples; only the SDF values (interior/exterior) vary.
        """
        gs = self._proj_grid_size
        verts = self.tet_verts[self.mask == 0].float()  # (N_kept, 3)
        xs, ys = verts[:, 0], verts[:, 1]
        self._proj_xi = ((xs - xs.min()) / (xs.max() - xs.min() + 1e-6) * (gs - 1)).long().clamp(0, gs - 1)
        self._proj_yi = ((ys - ys.min()) / (ys.max() - ys.min() + 1e-6) * (gs - 1)).long().clamp(0, gs - 1)

    def _generate_projection(self, sdf_masked: torch.Tensor) -> torch.Tensor:
        """
        Generate a top-down (Z-axis) binary projection of the organelle.

        Each pixel accumulates the number of *interior* vertices (SDF < 0)
        that project onto it; the result is normalised to [0, 1] and returned
        as a (1, proj_size, proj_size) float32 tensor.

        This image is used as the 2D conditioning signal during training and
        can be replaced by a real EM microscopy image at inference time.

        Args:
            sdf_masked: SDF values for the masked (kept) vertices, shape (N,).
        """
        if not hasattr(self, '_proj_xi'):
            self._precompute_projection_indices()

        gs = self._proj_grid_size
        interior = (sdf_masked < 0).float()   # (N,)  1 = inside organelle

        img = torch.zeros(gs, gs, dtype=torch.float32)
        img.index_put_((self._proj_yi, self._proj_xi), interior, accumulate=True)

        max_val = img.max()
        if max_val > 0:
            img = img / max_val

        return img.unsqueeze(0)   # (1, H, W)

    def mask_sdfs_or_disps_it(self, sdf, displacement, color):

        with torch.no_grad():
            masked_disps = torch.full_like(displacement, self.config.dataset.deform_masking_value)
            masked_color = torch.full_like(color, -0.01)

            masked_sdf = torch.where(sdf >= 0, self.config.dataset.sdf_masking_value,
                                     -1 * self.config.dataset.sdf_masking_value).float()
            weight_mask = torch.zeros_like(sdf)

            idx = marching_cube_get_idx(sdf, self.tet_faces)

            selected_tets = self.tet_faces[idx].reshape(-1, 2)
            selected_tets = self.sort_edges(selected_tets)
            unique_edges, idx_map = torch.unique(selected_tets, dim=0, return_inverse=True)

            masked_sdf[unique_edges.reshape(-1)] = sdf[unique_edges.reshape(-1)]
            weight_mask[unique_edges.reshape(-1)] = 1

            masked_disps[unique_edges.reshape(-1)] = displacement[unique_edges.reshape(-1)]
            masked_color[unique_edges.reshape(-1)] = color[unique_edges.reshape(-1)]

            self.mask_verts[unique_edges.reshape(-1)] += 1
            masked_disps = torch.cat([masked_disps, weight_mask.unsqueeze(-1)], -1)

        return masked_sdf.cpu().detach(), masked_disps.cpu().detach(), masked_color.cpu().detach()

    def get_stats(self, array):
        array = torch.flatten(array, 0, 1)

        max = torch.max(array, 0)[0].detach().cpu()
        min = torch.min(array, 0)[0].detach().cpu()

        mean = torch.mean(array, 0).detach().cpu()
        std = torch.std(array, 0).detach().cpu()
        return max, min, mean, std

    def sort_edges(self, edges_ex2):
        with torch.no_grad():
            order = (edges_ex2[:, 0] > edges_ex2[:, 1]).long()
            order = order.unsqueeze(dim=1)
            a = torch.gather(input=edges_ex2, index=order, dim=1)
            b = torch.gather(input=edges_ex2, index=1 - order, dim=1)
        return torch.stack([a, b], -1)

    def get_statistics(self):
        self.color_max = torch.tensor([-1000., -1000., -1000.])
        self.sdfs_max = torch.tensor([-1000.])
        self.deform_max = torch.tensor([-1000., -1000., -1000.])

        self.color_min = torch.tensor([1000., 1000., 1000.])
        self.deform_min = torch.tensor([1000., 1000., 1000.])
        self.sdfs_min = torch.tensor([1000.])

        self.sdfs_sum = torch.tensor([0.])
        self.sdfs_sum2 = torch.tensor([0.])
        self.deform_sum = torch.tensor([0., 0, 0])
        self.deform_sum2 = torch.tensor([0., 0, 0])

        self.color_sum = torch.tensor([0., 0., 0.])
        self.color_sum2 = torch.tensor([0., 0., 0.])

        n = 0
        for idx in range(len(self.paths_train)):
            sample = torch.load(self.paths_train[idx], map_location='cpu', weights_only=False)
            sdfs, deform, color = sample[0], sample[1], sample[2]

            sdfs = sdfs[self.mask == 0]
            deform = deform[self.mask == 0, :]
            color = color[self.mask == 0, :]

            color_max, color_min, _, _ = self.get_stats(color[None, :, :])
            deform_max, deform_min, _, _ = self.get_stats(deform[None, :, :3])
            sdfs_max, sdfs_min, _, _ = self.get_stats(sdfs.unsqueeze(-1)[None, :, :])

            self.sdfs_sum += torch.sum(sdfs, 0)
            self.sdfs_sum2 += torch.sum(torch.square(sdfs), 0)
            self.deform_sum += torch.sum(deform[..., :3], 0)
            self.deform_sum2 += torch.sum(torch.square(deform[..., :3]), 0)
            n += len(sdfs)
            self.color_sum += torch.sum(color, 0)
            self.color_sum2 += torch.sum(torch.square(color), 0)

            self.color_max = torch.max(torch.stack([color_max, self.color_max], 0), 0)[0]
            self.sdfs_max = torch.max(torch.stack([sdfs_max, self.sdfs_max], 0), 0)[0]
            self.deform_max = torch.max(torch.stack([deform_max, self.deform_max], 0), 0)[0]

            self.color_min = torch.min(torch.stack([color_min, self.color_min], 0), 0)[0]
            self.deform_min = torch.min(torch.stack([deform_min, self.deform_min], 0), 0)[0]
            self.sdfs_min = torch.min(torch.stack([sdfs_min, self.sdfs_min], 0), 0)[0]

        self.sdfs_mean = self.sdfs_sum / n
        self.color_mean = self.color_sum / n
        self.deform_mean = self.deform_sum / n
        self.deform_std = torch.sqrt((self.deform_sum2 / n) - torch.square(self.deform_mean))
        self.sdfs_std = torch.sqrt((self.sdfs_sum2 / n) - torch.square(self.sdfs_mean))
        self.color_std = torch.sqrt((self.color_sum2 / n) - torch.square(self.color_mean))

        print("data statistics")
        print("color")
        print(self.color_max, self.color_min, self.color_mean, self.color_std)
        print("deform")
        print(self.deform_max, self.deform_min, self.deform_mean, self.deform_std)
        print("sdfs")
        print(self.sdfs_max, self.sdfs_min, self.sdfs_mean, self.sdfs_std)


    def _init_gt_iterative(self):
        paths_train = list()
        paths_test = list()
        splits_csv = getattr(self.config, 'splits_csv', None) or "lib/all.csv"
        try:
            self.splits = pd.read_csv(splits_csv)
        except FileNotFoundError:
            try:
                self.splits = pd.read_csv("lib/all.csv")
            except FileNotFoundError:
                self.splits = pd.read_csv("all.csv")

        # Use an organelle-specific subdirectory so that preprocessing runs for
        # different organelles never overwrite each other's cached sample files.
        # Without this, a second organelle's training would silently overwrite
        # sample_*.pt files referenced by the first organelle's ds_cache, causing
        # the model to train on the wrong organelle's preprocessed data.
        category_key = "_".join(sorted(self.config.dataset.shapenet_ids))
        root = Path(f'{self.config.data_path}/preprocessed_data/samples/{category_key}/')
        root.mkdir(parents=True, exist_ok=True)
        (root / 'train').mkdir(parents=True, exist_ok=True)
        (root / 'val').mkdir(parents=True, exist_ok=True)

        print('Starting to create dataset.')

        file_list = []
        self.shapenet_ids = {}
        counter = 0
        for shapenetid in self.config.dataset.shapenet_ids:
            self.shapenet_ids[shapenetid] = counter
            # Try the standard nested layout first: {category}/{model_id}/mesh_data/sample.pth
            found = glob(f"{self.config.data_path}/{shapenetid}/*/*/sample.pth")
            if not found:
                # Fall back to flat layout: {category}/{model_id}/sample.pth
                found = glob(f"{self.config.data_path}/{shapenetid}/*/sample.pth")
                if found:
                    print(f"[MeshLoader] Using flat layout (no mesh_data/ subdir) for '{shapenetid}'")
                else:
                    print(f"[MeshLoader] WARNING: no sample.pth found for '{shapenetid}' "
                          f"under '{self.config.data_path}/{shapenetid}'. "
                          f"Check that preprocessing has completed and --data_path is correct.")
            file_list.extend(found)
            counter += 1

        self.tet_faces = self.tet_faces.to(self.cuda_device)
        self.mask_verts = self.mask_verts.to(self.cuda_device)

        for i in tqdm(range(len(file_list[:self.config.dataset.num_samples]))):

            name = file_list[i]
            # model_id is the first subdirectory under the category folder,
            # regardless of whether the layout is {model_id}/mesh_data/sample.pth
            # or the flat {model_id}/sample.pth.
            # Split on both / and \ for cross-platform safety.
            name_parts = _re.split(r'[\\/]', name)
            # Find the category in the path parts and take the next element as model_id.
            model_id_idx = None
            for shapenetid in self.config.dataset.shapenet_ids:
                if shapenetid in name_parts:
                    idx = name_parts.index(shapenetid)
                    if idx + 1 < len(name_parts):
                        model_id_idx = idx + 1
                    break
            model_id = name_parts[model_id_idx] if model_id_idx is not None else name_parts[-3]

            # The CSV-based train/test split is only used for ShapeNet categories
            # that appear in lib/all.csv.  For organelle data the IDs won't be in
            # the CSV, so we skip the assertion and the split filter in that case.
            in_csv = model_id in self.splits['modelId'].values

            if self.config.dataset.train_split and in_csv:
                model_split = self.splits.loc[self.splits['modelId'] == model_id]['split'].values[0]
                if model_split != "train":
                    continue

            sdfs, deform, color = torch.load(name, map_location=self.cuda_device, weights_only=False)
            sdfs = torch.tensor(sdfs).to(self.cuda_device)
            deform = torch.tensor(deform).to(self.cuda_device)
            color = torch.tensor(color).to(self.cuda_device)

            if self.config.dataset.mask_data:
                sdfs, deform, color = self.mask_sdfs_or_disps_it(sdfs, deform, color)

            elif self.config.dataset.grid_pruning:  ## if only grid_pruning, do not mask the data but create the cube mask
                self.mask_sdfs_or_disps_it(sdfs, deform, color)

            if self.accelerator.is_main_process:
                torch.save([sdfs.detach().cpu(), deform.detach().cpu(), color.detach().cpu(), name],
                           root / f'sample_{i}.pt')
            paths_train.append(str(root / f'sample_{i}.pt'))

        if self.config.dataset.grid_pruning:
            if self.config.dataset.mask_lossy:
                self.mask_verts = torch.where(self.mask_verts > self.config.dataset.threshold, 1, 0)
            else:
                self.mask_verts = torch.where(self.mask_verts > 0, 1, 0)

        self.tet_faces = self.tet_faces.cpu()
        self.mask_verts = self.mask_verts.cpu()
        return paths_train, paths_test


    def _denormalize(self, sdf, displacement, color):

        sdf = sdf * (self.sdfs_max.to(sdf.device) - self.sdfs_min.to(sdf.device)) + self.sdfs_min.to(sdf.device)
        displacement = displacement * (self.deform_max.to(displacement.device) - self.deform_min.to(displacement.device)) + self.deform_min.to(displacement.device)
        color = color * (self.color_max.to(displacement.device) - self.color_min.to(displacement.device)) + self.color_min.to(displacement.device)

        color = torch.clamp(color, 0, 1)
        return sdf, displacement, color

    def _normalize(self, sdf, displacement, color):

        sdf = ((sdf - self.sdfs_min) / (self.sdfs_max - self.sdfs_min))
        displacement = ((displacement - self.deform_min) / (self.deform_max - self.deform_min))
        color = ((color - self.color_min) / (self.color_max - self.color_min))

        return sdf, displacement, color

    def marching_cube(self, pos_nx3, sdf_n, tet_fx4):
        with torch.no_grad():
            occ_n = sdf_n > 0
            occ_fx4 = occ_n[tet_fx4.reshape(-1)].reshape(-1, 4)
            occ_sum = torch.sum(occ_fx4, -1)
            valid_tets = (occ_sum > 0) & (occ_sum < 4)

            # find all vertices
            all_edges = tet_fx4[valid_tets][:, self.base_tet_edges].reshape(-1, 2)
            all_edges = self.sort_edges(all_edges)
            unique_edges, idx_map = torch.unique(all_edges, dim=0, return_inverse=True)

            unique_edges = unique_edges.long()
            mask_edges = occ_n[unique_edges.reshape(-1)].reshape(-1, 2).sum(-1) == 1
            mapping = torch.ones((unique_edges.shape[0]), dtype=torch.long, device=self.device) * -1
            mapping[mask_edges] = torch.arange(mask_edges.sum(), dtype=torch.long, device=self.device)
            idx_map = mapping[idx_map]  # map edges to verts

            interp_v = unique_edges[mask_edges]
        edges_to_interp = pos_nx3[interp_v.reshape(-1)].reshape(-1, 2, 6)
        edges_to_interp_sdf = sdf_n[interp_v.reshape(-1)].reshape(-1, 2, 1)
        edges_to_interp_sdf[:, -1] *= -1

        denominator = edges_to_interp_sdf.sum(1, keepdim=True)

        edges_to_interp_sdf = torch.flip(edges_to_interp_sdf, [1]) / denominator
        verts = (edges_to_interp * edges_to_interp_sdf).sum(1)

        idx_map = idx_map.reshape(-1, 6)

        v_id = torch.pow(2, torch.arange(4, dtype=torch.long, device=self.device))
        tetindex = (occ_fx4[valid_tets] * v_id.unsqueeze(0)).sum(-1)
        num_triangles = self.num_triangles_table[tetindex]

        # Generate triangle indices
        faces = torch.cat((
            torch.gather(input=idx_map[num_triangles == 1], dim=1,
                         index=self.triangle_table[tetindex[num_triangles == 1]][:, :3]).reshape(-1, 3),
            torch.gather(input=idx_map[num_triangles == 2], dim=1,
                         index=self.triangle_table[tetindex[num_triangles == 2]][:, :6]).reshape(-1, 3),
        ), dim=0)

        return verts, faces

    def get_mesh(self, sample):
        sample = sample.squeeze().cpu()
        sdf, deform, color = self._denormalize(sample[:, 0], sample[:, 1:4], sample[:, 4:])

        sdf = torch.sign(sdf)
        v_deformed = self.tet_verts + 2 / (self.grid_res * 2) * deform * 2.0
        v_feats = torch.cat([v_deformed, color], dim=-1)

        verts_color, faces = self.marching_cube(v_feats, sdf, self.tet_faces.to(self.device))

        return verts_color[:, :3], verts_color[:, 3:], faces

    def __len__(self):

        if self.config.dataset.training:
            return len(self.paths_train)
        else:
            return len(self.paths_test)


    def __getitem__(self, idx):
        if self.config.dataset.training:
            sample = torch.load(self.paths_train[idx], map_location='cpu', weights_only=False)
        else:
            sample = torch.load(self.paths_test[idx], map_location='cpu', weights_only=False)

        sdf, displacements, colors = sample[0], sample[1], sample[2]


        sdf = sdf[self.mask == 0]
        displacements = displacements[self.mask == 0, :]
        colors = colors[self.mask == 0, :]
        displacements, mask = displacements[:, :3], displacements[:, -1]
        sdf, displacements, colors = self._normalize(sdf, displacements, colors)
        data = torch.cat([sdf.unsqueeze(-1), displacements, colors], -1)

        if not self.config.dataset.color:
            data = data[:, :4]

        # When image conditioning is enabled, generate a 2D projection of the
        # organelle and return it alongside the 3D data.  The Trainer unpacks
        # the tuple; when disabled only the data tensor is returned so the
        # existing unconditional training path is completely unchanged.
        if getattr(self, 'use_image_cond', False):
            image = self._generate_projection(sdf)  # (1, H, W)
            return data, image

        return data
