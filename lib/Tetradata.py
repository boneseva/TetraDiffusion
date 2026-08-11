from __future__ import annotations

from glob import glob
import os
from pathlib import Path
import random
import re as _re
from typing import Any, List, Tuple, Union

import hashlib
import json
import math
import numpy as np
import pandas as pd
import torch
from torch.utils.data.dataset import Dataset
from tqdm import tqdm

from lib.GridPruning import mask_cube


def select_prefix(paths: List[str], fraction: float, seed: int) -> List[str]:
    """
    Deterministically shuffle a list of paths using a seed, and return the prefix
    corresponding to math.ceil(len(paths) * fraction).
    Ensures nested subsets: 25% ⊂ 50% ⊂ 75% ⊂ 100%.
    """
    if not paths:
        raise ValueError("paths list cannot be empty.")
    if not (0.0 < fraction <= 1.0):
        raise ValueError(f"dataset_fraction must be in (0, 1.0], got {fraction}")

    rng = random.Random(seed)
    shuffled = list(sorted(paths))
    rng.shuffle(shuffled)
    num_samples = max(1, math.ceil(len(shuffled) * fraction))
    return shuffled[:num_samples]



def marching_cube_get_idx(sdf_n: torch.Tensor, tet_fx4: torch.Tensor) -> torch.Tensor:
    num_triangles_table = torch.tensor(
        [0, 1, 1, 2, 1, 2, 2, 1, 1, 2, 2, 1, 2, 1, 1, 0],
        dtype=torch.long,
        device=sdf_n.device,
    )

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
            (
                tet_idx[num_triangles == 1],
                tet_idx[num_triangles == 2].unsqueeze(-1).expand(-1, 2).reshape(-1),
            ),
            dim=0,
        )
        return tet_idx


class MeshLoader(Dataset):
    """
    PyTorch Dataset loading tetrahedral grid samples, performing spatial pruning,
    normalizing feature channels, and performing Deformable Tetrahedral Marching (DMTet).
    """

    def __init__(
        self,
        config: Any,
        device: torch.device,
        cuda_device: Union[int, str, torch.device],
        accelerator: Any,
    ):
        super().__init__()
        self.device = device
        self.accelerator = accelerator
        self.cuda_device = cuda_device
        self.config = config
        self.grid_res = self.config.dataset.grid_res

        # Fail-fast validation for dataset_fraction
        self.dataset_fraction = float(getattr(self.config.dataset, "dataset_fraction", 1.0))
        self.seed = int(getattr(self.config, "seed", 42))
        if not (0.0 < self.dataset_fraction <= 1.0):
            raise ValueError(f"dataset_fraction must be in (0, 1.0], got {self.dataset_fraction}")

        self.triangle_table = torch.tensor(
            [
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
                [-1, -1, -1, -1, -1, -1],
            ],
            dtype=torch.long,
            device=self.device,
        )

        self.num_triangles_table = torch.tensor(
            [0, 1, 1, 2, 1, 2, 2, 1, 1, 2, 2, 1, 2, 1, 1, 0],
            dtype=torch.long,
            device=self.device,
        )

        cube_range = self.config.dataset.cube_range

        self.vertices = [
            torch.tensor(
                np.load(f"tetrahedra/{self.grid_res}/{i}_tets.npz")["vertices"],
                dtype=torch.float32,
            )
            for i in cube_range
        ]
        self.tetra_cubes = [
            torch.tensor(
                np.load(f"tetrahedra/{self.grid_res}/{i}_tets.npz")["indices"],
                dtype=torch.int32,
            )
            for i in cube_range
        ]
        self.neighbors = [
            torch.load(
                f"tetrahedra/{self.grid_res}/neighbors_{i}_sorted.pth",
                map_location="cpu",
                weights_only=False,
            ).int()
            for i in cube_range
        ]
        self.upsample = [
            torch.load(
                f"tetrahedra/{self.grid_res}/upsample_{i}_sorted.pth",
                map_location="cpu",
                weights_only=False,
            )[0]
            for i in cube_range[:-1]
        ]
        self.downsample = [
            torch.load(
                f"tetrahedra/{self.grid_res}/downsample_{i}_sorted.pth",
                map_location="cpu",
                weights_only=False,
            )[0]
            for i in cube_range[1:]
        ]

        for i in range(len(self.vertices)):
            self.vertices[i] = self.vertices[i] - torch.mean(self.vertices[i], 0)

        self.tet_verts = self.vertices[-1].to(self.device)
        self.tet_faces = self.tetra_cubes[-1].to(self.device)

        edges = torch.tensor(
            [0, 1, 0, 2, 0, 3, 1, 2, 1, 3, 2, 3], dtype=torch.long, device="cpu"
        )
        all_edges = self.tet_faces[:, edges].reshape(-1, 2)
        all_edges_sorted = torch.sort(all_edges, dim=1)[0]
        self.all_edges = torch.unique(all_edges_sorted, dim=0)
        self.base_tet_edges = torch.tensor(
            [0, 1, 0, 2, 0, 3, 1, 2, 1, 3, 2, 3], dtype=torch.long, device="cpu"
        )
        self.num_points = len(self.tet_verts)

        self.mask_verts = torch.zeros((len(self.tet_verts))).int()

        self.paths_train, self.paths_test = self._init_gt_iterative()
        self.accelerator.wait_for_everyone()

        # Split Assertions
        if len(self.paths_train) == 0:
            raise RuntimeError("[MeshLoader] Error: paths_train is empty. Check data_path and category configuration.")
        is_train_split = bool(getattr(self.config.dataset, "train_split", True))
        if is_train_split and len(self.paths_test) == 0:
            raise RuntimeError(
                "[MeshLoader] Error: train_split=True was requested, but paths_test is empty! "
                "Check splits CSV file and ensure test samples exist for the category."
            )

        if self.config.dataset.grid_pruning:
            (
                (self.vertices, self.tetra_cubes, self.neighbors, self.upsample, self.downsample),
                self.mask,
            ) = mask_cube(
                self.mask_verts,
                self.vertices,
                self.tetra_cubes,
                self.neighbors,
                self.upsample,
                self.downsample,
                cuda_device,
            )
            self.mask = self.mask.squeeze(1).int()
            print(
                f"[MeshLoader] Grid pruning active: retaining "
                f"{np.round(len(self.vertices[-1]) / len(self.tet_verts) * 100, 2)}% of grid vertices."
            )
            self.tet_verts = self.vertices[-1]
            self.tet_faces = self.tetra_cubes[-1]
        else:
            self.mask = torch.zeros_like(self.tet_verts[:, 0]).cpu()

        self.accelerator.wait_for_everyone()
        self.get_statistics()
        self.accelerator.wait_for_everyone()

        # Upfront fraction validation & deterministic nested prefix sub-sampling on train set ONLY
        self.dataset_fraction = float(getattr(self.config.dataset, "dataset_fraction", 1.0))
        self.seed = int(getattr(self.config, "seed", 42))

        if not (0.0 < self.dataset_fraction <= 1.0):
            raise ValueError(f"dataset_fraction must be in (0, 1.0], got {self.dataset_fraction}")

        self.paths_train_full = list(self.paths_train)
        if self.dataset_fraction < 1.0:
            self.paths_train = select_prefix(self.paths_train_full, self.dataset_fraction, self.seed)
            print(
                f"[MeshLoader] Sub-sampled train set to {len(self.paths_train)}/{len(self.paths_train_full)} "
                f"samples (fraction={self.dataset_fraction}, seed={self.seed})."
            )

        self.tet_verts = self.tet_verts.cpu()
        torch.cuda.empty_cache()

        # Optional 2D image projection precomputation
        self.use_image_cond = bool(
            getattr(getattr(self.config, "image_cond", None), "enabled", False)
        )
        if self.use_image_cond:
            self._proj_grid_size = int(getattr(self.config.image_cond, "proj_size", 64))
            self._precompute_projection_indices()
            print(
                f"[MeshLoader] Image projection enabled "
                f"(grid {self._proj_grid_size}x{self._proj_grid_size})"
            )

    def _precompute_projection_indices(self) -> None:
        """Precompute X and Y pixel indices for top-down 2D image projection."""
        gs = self._proj_grid_size
        verts = self.tet_verts[self.mask == 0].float()
        xs, ys = verts[:, 0], verts[:, 1]
        self._proj_xi = (
            ((xs - xs.min()) / (xs.max() - xs.min() + 1e-6) * (gs - 1))
            .long()
            .clamp(0, gs - 1)
        )
        self._proj_yi = (
            ((ys - ys.min()) / (ys.max() - ys.min() + 1e-6) * (gs - 1))
            .long()
            .clamp(0, gs - 1)
        )

    def _generate_projection(self, sdf_masked: torch.Tensor) -> torch.Tensor:
        """Generate a top-down binary projection image of the interior SDF region."""
        if not hasattr(self, "_proj_xi"):
            self._precompute_projection_indices()

        gs = self._proj_grid_size
        interior = (sdf_masked < 0).float()

        img = torch.zeros(gs, gs, dtype=torch.float32)
        img.index_put_((self._proj_yi, self._proj_xi), interior, accumulate=True)

        max_val = img.max()
        if max_val > 0:
            img = img / max_val

        return img.unsqueeze(0)

    def mask_sdfs_or_disps_it(
        self, sdf: torch.Tensor, displacement: torch.Tensor, color: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply truncation masks to SDF, displacement, and color attributes near boundaries."""
        with torch.no_grad():
            masked_disps = torch.full_like(
                displacement, self.config.dataset.deform_masking_value
            )
            masked_color = torch.full_like(color, -0.01)

            masked_sdf = torch.where(
                sdf >= 0,
                self.config.dataset.sdf_masking_value,
                -1 * self.config.dataset.sdf_masking_value,
            ).float()
            weight_mask = torch.zeros_like(sdf)

            idx = marching_cube_get_idx(sdf, self.tet_faces)

            selected_tets = self.tet_faces[idx].reshape(-1, 2)
            selected_tets = self.sort_edges(selected_tets)
            unique_edges, _ = torch.unique(selected_tets, dim=0, return_inverse=True)

            masked_sdf[unique_edges.reshape(-1)] = sdf[unique_edges.reshape(-1)]
            weight_mask[unique_edges.reshape(-1)] = 1

            masked_disps[unique_edges.reshape(-1)] = displacement[
                unique_edges.reshape(-1)
            ]
            masked_color[unique_edges.reshape(-1)] = color[unique_edges.reshape(-1)]

            self.mask_verts[unique_edges.reshape(-1)] += 1
            masked_disps = torch.cat([masked_disps, weight_mask.unsqueeze(-1)], -1)

        return (
            masked_sdf.cpu().detach(),
            masked_disps.cpu().detach(),
            masked_color.cpu().detach(),
        )

    def get_stats(self, array: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute max, min, mean, and std statistics for a feature tensor."""
        array = torch.flatten(array, 0, 1)

        max_val = torch.max(array, 0)[0].detach().cpu()
        min_val = torch.min(array, 0)[0].detach().cpu()
        mean_val = torch.mean(array, 0).detach().cpu()
        std_val = torch.std(array, 0).detach().cpu()
        return max_val, min_val, mean_val, std_val

    def sort_edges(self, edges_ex2: torch.Tensor) -> torch.Tensor:
        """Sort vertex edge pairs so (v1, v2) satisfies v1 < v2."""
        with torch.no_grad():
            order = (edges_ex2[:, 0] > edges_ex2[:, 1]).long().unsqueeze(dim=1)
            a = torch.gather(input=edges_ex2, index=order, dim=1)
            b = torch.gather(input=edges_ex2, index=1 - order, dim=1)
        return torch.stack([a, b], -1)

    def get_statistics(self) -> None:
        """Compute global dataset mean, std, min, and max statistics across training samples."""
        self.color_max = torch.tensor([-1000.0, -1000.0, -1000.0])
        self.sdfs_max = torch.tensor([-1000.0])
        self.deform_max = torch.tensor([-1000.0, -1000.0, -1000.0])

        self.color_min = torch.tensor([1000.0, 1000.0, 1000.0])
        self.deform_min = torch.tensor([1000.0, 1000.0, 1000.0])
        self.sdfs_min = torch.tensor([1000.0])

        self.sdfs_sum = torch.tensor([0.0])
        self.sdfs_sum2 = torch.tensor([0.0])
        self.deform_sum = torch.tensor([0.0, 0.0, 0.0])
        self.deform_sum2 = torch.tensor([0.0, 0.0, 0.0])

        self.color_sum = torch.tensor([0.0, 0.0, 0.0])
        self.color_sum2 = torch.tensor([0.0, 0.0, 0.0])

        n = 0
        for idx in range(len(self.paths_train)):
            sample = torch.load(self.paths_train[idx], map_location="cpu", weights_only=False)
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

        print("[MeshLoader] Dataset statistics loaded:")
        print(f"  SDF:    mean={self.sdfs_mean.item():.4f}, std={self.sdfs_std.item():.4f}")
        print(f"  Deform: mean={self.deform_mean.tolist()}, std={self.deform_std.tolist()}")

    def _init_gt_iterative(self) -> Tuple[List[str], List[str]]:
        """Discover raw dataset files, preprocess, and write cached sample_i.pt tensors."""
        paths_train = list()
        paths_test = list()
        splits_csv = getattr(self.config, "splits_csv", None) or "lib/all.csv"
        try:
            self.splits = pd.read_csv(splits_csv)
        except FileNotFoundError:
            try:
                self.splits = pd.read_csv("lib/all.csv")
            except FileNotFoundError:
                self.splits = pd.read_csv("all.csv")

        category_key = "_".join(sorted(self.config.dataset.shapenet_ids))
        root = Path(f"{self.config.data_path}/preprocessed_data/samples/{category_key}/")
        root.mkdir(parents=True, exist_ok=True)
        train_root = root / "train"
        val_root   = root / "val"
        train_root.mkdir(parents=True, exist_ok=True)
        val_root.mkdir(parents=True, exist_ok=True)

        print(f"[MeshLoader] Initializing dataset for categories: {self.config.dataset.shapenet_ids}")
        print(f"[MeshLoader] Cache dirs: train={train_root}  val={val_root}")

        file_list = []
        self.shapenet_ids = {}
        counter = 0
        for shapenetid in self.config.dataset.shapenet_ids:
            self.shapenet_ids[shapenetid] = counter
            found = sorted(glob(f"{self.config.data_path}/{shapenetid}/*/*/sample.pth"))
            if not found:
                found = sorted(glob(f"{self.config.data_path}/{shapenetid}/*/sample.pth"))
                if found:
                    print(f"[MeshLoader] Found flat layout (no mesh_data/ subdir) for '{shapenetid}'")
                else:
                    print(
                        f"[MeshLoader] WARNING: No sample.pth found under '{self.config.data_path}/{shapenetid}'."
                    )
            file_list.extend(found)
            counter += 1

        self.tet_faces = self.tet_faces.to(self.cuda_device)
        self.mask_verts = self.mask_verts.to(self.cuda_device)

        train_i = 0
        val_i   = 0

        for i in tqdm(range(len(file_list[: self.config.dataset.num_samples])), desc="Caching dataset samples"):
            name = file_list[i]
            name_parts = _re.split(r"[\\/]", name)
            model_id_idx = None
            for shapenetid in self.config.dataset.shapenet_ids:
                if shapenetid in name_parts:
                    idx = name_parts.index(shapenetid)
                    if idx + 1 < len(name_parts):
                        model_id_idx = idx + 1
                    break
            raw_model_id = name_parts[model_id_idx] if model_id_idx is not None else name_parts[-3]
            prefixed_model_id = f"{shapenetid}_{raw_model_id}" if not raw_model_id.startswith(f"{shapenetid}_") else raw_model_id

            if prefixed_model_id in self.splits["modelId"].values:
                model_id = prefixed_model_id
            else:
                model_id = raw_model_id
            in_csv = model_id in self.splits["modelId"].values

            # Determine split; if in CSV use CSV split column, else fallback to deterministic 80/20 split
            is_val = False
            if self.config.dataset.train_split:
                if in_csv:
                    model_split = self.splits.loc[self.splits["modelId"] == model_id]["split"].values[0]
                    if model_split in ("val", "test"):
                        is_val = True
                    elif model_split != "train":
                        # Unknown split value — skip
                        continue
                else:
                    # Deterministic 80/20 split fallback for samples not in CSV
                    hash_val = int(hashlib.md5(model_id.encode("utf-8")).hexdigest(), 16)
                    if (hash_val % 100) < 20:
                        is_val = True

            sdfs, deform, color = torch.load(name, map_location=self.cuda_device, weights_only=False)
            sdfs = torch.tensor(sdfs).to(self.cuda_device)
            deform = torch.tensor(deform).to(self.cuda_device)
            color = torch.tensor(color).to(self.cuda_device)

            if self.config.dataset.mask_data:
                sdfs, deform, color = self.mask_sdfs_or_disps_it(sdfs, deform, color)
            elif self.config.dataset.grid_pruning:
                self.mask_sdfs_or_disps_it(sdfs, deform, color)

            cache_root = val_root   if is_val else train_root
            cache_idx  = val_i      if is_val else train_i

            if self.accelerator.is_main_process:
                torch.save(
                    [sdfs.detach().cpu(), deform.detach().cpu(), color.detach().cpu(), name],
                    cache_root / f"sample_{cache_idx}.pt",
                )

            if is_val:
                paths_test.append(str(val_root / f"sample_{val_i}.pt"))
                val_i += 1
            else:
                paths_train.append(str(train_root / f"sample_{train_i}.pt"))
                train_i += 1

        print(f"[MeshLoader] Cached {train_i} train + {val_i} val samples.")

        if self.config.dataset.grid_pruning:
            if self.config.dataset.mask_lossy:
                self.mask_verts = torch.where(
                    self.mask_verts > self.config.dataset.threshold, 1, 0
                )
            else:
                self.mask_verts = torch.where(self.mask_verts > 0, 1, 0)

        self.tet_faces = self.tet_faces.cpu()
        self.mask_verts = self.mask_verts.cpu()
        return paths_train, paths_test

    def _denormalize(
        self, sdf: torch.Tensor, displacement: torch.Tensor, color: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Denormalize features back to absolute physical scale."""
        sdf = sdf * (self.sdfs_max.to(sdf.device) - self.sdfs_min.to(sdf.device)) + self.sdfs_min.to(sdf.device)
        displacement = (
            displacement * (self.deform_max.to(displacement.device) - self.deform_min.to(displacement.device))
            + self.deform_min.to(displacement.device)
        )
        if color is not None and color.numel() > 0:
            color = (
                color * (self.color_max.to(color.device) - self.color_min.to(color.device))
                + self.color_min.to(color.device)
            )
            color = torch.clamp(color, 0, 1)

        return sdf, displacement, color

    def _normalize(
        self, sdf: torch.Tensor, displacement: torch.Tensor, color: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Normalize feature channels into range [0, 1]."""
        sdf = (sdf - self.sdfs_min) / (self.sdfs_max - self.sdfs_min).clamp(min=1e-6)
        displacement = (displacement - self.deform_min) / (
            self.deform_max - self.deform_min
        ).clamp(min=1e-6)
        color = (color - self.color_min) / (self.color_max - self.color_min).clamp(min=1e-6)
        return sdf, displacement, color

    def marching_cube(
        self, pos_nx3: torch.Tensor, sdf_n: torch.Tensor, tet_fx4: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run Deformable Tetrahedral Marching (DMTet) to extract 3D surface mesh."""
        with torch.no_grad():
            occ_n = sdf_n > 0
            occ_fx4 = occ_n[tet_fx4.reshape(-1)].reshape(-1, 4)
            occ_sum = torch.sum(occ_fx4, -1)
            valid_tets = (occ_sum > 0) & (occ_sum < 4)

            all_edges = tet_fx4[valid_tets][:, self.base_tet_edges].reshape(-1, 2)
            all_edges = self.sort_edges(all_edges)
            unique_edges, idx_map = torch.unique(all_edges, dim=0, return_inverse=True)

            unique_edges = unique_edges.long()
            mask_edges = occ_n[unique_edges.reshape(-1)].reshape(-1, 2).sum(-1) == 1
            mapping = (
                torch.ones((unique_edges.shape[0]), dtype=torch.long, device=self.device)
                * -1
            )
            mapping[mask_edges] = torch.arange(
                mask_edges.sum(), dtype=torch.long, device=self.device
            )
            idx_map = mapping[idx_map]

            interp_v = unique_edges[mask_edges]
        num_feats = pos_nx3.shape[-1]
        edges_to_interp = pos_nx3[interp_v.reshape(-1)].reshape(-1, 2, num_feats)
        edges_to_interp_sdf = sdf_n[interp_v.reshape(-1)].reshape(-1, 2, 1)
        edges_to_interp_sdf[:, -1] *= -1

        denominator = edges_to_interp_sdf.sum(1, keepdim=True)
        edges_to_interp_sdf = torch.flip(edges_to_interp_sdf, [1]) / denominator
        verts = (edges_to_interp * edges_to_interp_sdf).sum(1)

        idx_map = idx_map.reshape(-1, 6)

        v_id = torch.pow(2, torch.arange(4, dtype=torch.long, device=self.device))
        tetindex = (occ_fx4[valid_tets] * v_id.unsqueeze(0)).sum(-1)
        num_triangles = self.num_triangles_table[tetindex]

        faces = torch.cat(
            (
                torch.gather(
                    input=idx_map[num_triangles == 1],
                    dim=1,
                    index=self.triangle_table[tetindex[num_triangles == 1]][:, :3],
                ).reshape(-1, 3),
                torch.gather(
                    input=idx_map[num_triangles == 2],
                    dim=1,
                    index=self.triangle_table[tetindex[num_triangles == 2]][:, :6],
                ).reshape(-1, 3),
            ),
            dim=0,
        )

        return verts, faces

    def get_mesh(self, sample: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        """
        Extract 3D mesh vertices, colors, and faces from model prediction tensor.

        Handles both 7-channel (color ON) and 4-channel (color OFF) feature tensors.
        """
        sample = sample.squeeze().cpu()
        if sample.shape[-1] >= 7:
            sdf, deform, color = self._denormalize(
                sample[:, 0], sample[:, 1:4], sample[:, 4:]
            )
        else:
            sdf, deform, _ = self._denormalize(sample[:, 0], sample[:, 1:4], None)
            color = None

        sdf = torch.sign(sdf)
        v_deformed = self.tet_verts + 2 / (self.grid_res * 2) * deform * 2.0

        if color is not None:
            v_feats = torch.cat([v_deformed, color], dim=-1)
            verts_color, faces = self.marching_cube(
                v_feats, sdf, self.tet_faces.to(self.device)
            )
            return verts_color[:, :3], verts_color[:, 3:], faces
        else:
            v_feats = v_deformed
            verts, faces = self.marching_cube(
                v_feats, sdf, self.tet_faces.to(self.device)
            )
            return verts[:, :3], None, faces

    def get_mesh_wo_color(self, sample: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        """Alias for get_mesh for backward compatibility when dataset.color=False."""
        return self.get_mesh(sample)


    # ------------------------------------------------------------------
    # Real-image conditioning loader
    # ------------------------------------------------------------------

    def _load_real_image(
        self, sample_pth_path: str, sdf: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Load a randomly chosen XY conditioning image from alongside sample.pth.

        The image files (image_xy_p*.npy) are written by
        link_images_to_samples.py into the same mesh_data/ directory as
        sample.pth.  They are already anisotropy-corrected, letterboxed,
        and per-image normalised to [0,1] by the extraction pipeline.

        Returns
        -------
        (1, proj_size, proj_size) float32 tensor.

        Raises
        ------
        FileNotFoundError
            If no image_xy_p*.npy files are found and
            image_cond.allow_synthetic_fallback is False (default).
        """
        sample_dir = Path(sample_pth_path).parent
        gs = getattr(self, "_proj_grid_size", 64)

        candidates = sorted(sample_dir.glob("image_xy_p*.npy"))

        if not candidates:
            allow_fallback = bool(
                getattr(
                    getattr(self.config, "image_cond", None),
                    "allow_synthetic_fallback",
                    False,
                )
            )
            if allow_fallback:
                if not getattr(self, "_missing_image_warned", False):
                    print(
                        f"[MeshLoader] WARNING: No image_xy_p*.npy in "
                        f"{sample_dir}. Falling back to synthetic projection "
                        f"(image_cond.allow_synthetic_fallback=True)."
                    )
                    self._missing_image_warned = True
                if sdf is None:
                    raise ValueError(
                        "sdf tensor is required when allow_synthetic_fallback=True"
                    )
                return self._generate_projection(sdf)
            raise FileNotFoundError(
                f"No image_xy_p*.npy found in {sample_dir}. "
                f"Run scripts/link_images_to_samples.py first, or set "
                f"image_cond.allow_synthetic_fallback: True to debug."
            )

        chosen = random.choice(candidates)
        arr = np.load(str(chosen))

        # ── Validate ───────────────────────────────────────────────────
        if arr.ndim != 2:
            raise ValueError(
                f"Expected 2-D array in {chosen}, got shape {arr.shape}"
            )
        if not np.isfinite(arr).all():
            raise ValueError(f"Non-finite values in {chosen}")
        arr = arr.astype(np.float32)

        # ── Re-apply normalization (guards against files saved before
        #    normalization was added to the extraction script) ──────────
        arr = normalize_image(arr)  # constant images → 0.0 correctly

        # ── Resize to model's expected resolution via PyTorch ──────────
        t = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
        if arr.shape != (gs, gs):
            t = torch.nn.functional.interpolate(
                t, size=(gs, gs), mode="bilinear", align_corners=False
            )
        return t.squeeze(0).float()  # (1, H, W)

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.paths_train) if self.config.dataset.training else len(self.paths_test)

    def __getitem__(self, idx: int) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if self.config.dataset.training:
            sample = torch.load(self.paths_train[idx], map_location="cpu", weights_only=False)
        else:
            sample = torch.load(self.paths_test[idx], map_location="cpu", weights_only=False)

        sdf, displacements, colors = sample[0], sample[1], sample[2]

        sdf = sdf[self.mask == 0]
        displacements = displacements[self.mask == 0, :]
        colors = colors[self.mask == 0, :]
        displacements = displacements[:, :3]
        sdf, displacements, colors = self._normalize(sdf, displacements, colors)
        data = torch.cat([sdf.unsqueeze(-1), displacements, colors], -1)

        if not self.config.dataset.color:
            data = data[:, :4]

        if getattr(self, "use_image_cond", False):
            # sample[3] is the original sample.pth path stored during caching.
            # Real conditioning images sit alongside it in the same directory.
            image = self._load_real_image(sample[3], sdf)
            return data, image

        return data

