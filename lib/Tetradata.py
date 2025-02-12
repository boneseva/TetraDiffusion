import torch
from pathlib import Path
from torch.utils.data.dataset import Dataset
import numpy as np
from glob import glob
from lib.GridPruning import mask_cube
from tqdm import tqdm
import pandas as pd


@torch.jit.script
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
        self.neighbors = [torch.load(f"tetrahedra/{self.grid_res}/neighbors_{i}_sorted.pth", map_location="cpu").int() for i in cube_range]
        self.upsample = [torch.load(f"tetrahedra/{self.grid_res}/upsample_{i}_sorted.pth", map_location="cpu")[0] for i in cube_range[:-1]]
        self.downsample = [torch.load(f"tetrahedra/{self.grid_res}/downsample_{i}_sorted.pth", map_location="cpu")[0] for i in cube_range[1:]]

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
            sample = torch.load(self.paths_train[idx], map_location='cpu')
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
        try:
            self.splits = pd.read_csv("lib/all.csv")
        except:
            self.splits = pd.read_csv("all.csv")

        root = Path(f'{self.config.data_path}/preprocessed_data/samples/')
        root.mkdir(parents=True, exist_ok=True)
        (root / 'train').mkdir(parents=True, exist_ok=True)
        (root / 'val').mkdir(parents=True, exist_ok=True)

        print('Starting to create dataset.')

        file_list = []
        self.shapenet_ids = {}
        counter = 0
        for shapenetid in self.config.dataset.shapenet_ids:
            self.shapenet_ids[shapenetid] = counter
            file_list.extend(glob(f"{self.config.data_path}/{shapenetid}/*/*/sample.pth"))
            counter += 1

        self.tet_faces = self.tet_faces.to(self.cuda_device)
        self.mask_verts = self.mask_verts.to(self.cuda_device)

        for i in tqdm(range(len(file_list[:self.config.dataset.num_samples]))):

            name = file_list[i]
            model_id = name.split('/')[-3]

            assert model_id in self.splits['modelId'].values

            if self.config.dataset.train_split:
                model_split = self.splits.loc[self.splits['modelId'] == model_id]['split'].values[0]
                if model_split != "train":
                    continue

            sdfs, deform, color = torch.load(name, map_location=self.cuda_device)
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
            sample = torch.load(self.paths_train[idx], map_location='cpu')
        else:
            sample = torch.load(self.paths_test[idx], map_location='cpu')

        sdf, displacements, colors = sample[0], sample[1], sample[2]

        sdf = sdf[self.mask == 0]
        displacements = displacements[self.mask == 0, :]
        colors = colors[self.mask == 0, :]
        displacements, mask = displacements[:, :3], displacements[:, -1]
        sdf, displacements, colors = self._normalize(sdf, displacements, colors)
        data = torch.cat([sdf.unsqueeze(-1), displacements, colors], -1)

        if self.config.dataset.color:
            return data[:, :]
        else:
            return data[:, :4]
