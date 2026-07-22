import numpy as np
import scipy.spatial
import scipy.stats
import trimesh

def normalize_point_cloud(pc):
    """
    Center point cloud at origin and scale to unit bounding sphere.
    This ensures Chamfer Distance and F-Score are scale-invariant and standardized.
    """
    if len(pc) == 0:
        return pc
    centroid = np.mean(pc, axis=0)
    pc_centered = pc - centroid
    max_radius = np.max(np.linalg.norm(pc_centered, axis=1))
    if max_radius > 1e-7:
        return pc_centered / max_radius
    return pc_centered

def sample_point_cloud(mesh, num_points=2048, normalize=True):
    """
    Sample points uniformly from the surface of the mesh or point cloud.
    If empty or face-less, samples from vertices or unit sphere.
    """
    if isinstance(mesh, trimesh.Scene):
        try:
            mesh = mesh.dump(concatenate=True)
        except Exception:
            mesh = None

    vertices = getattr(mesh, 'vertices', None)
    faces = getattr(mesh, 'faces', None)

    # If completely empty mesh / 0 vertices / failed load
    if vertices is None or len(vertices) == 0:
        pts = np.random.randn(num_points, 3)
        pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        return pts

    # If point cloud or no faces present
    if isinstance(mesh, trimesh.PointCloud) or faces is None or len(faces) == 0:
        idx = np.random.choice(len(vertices), num_points, replace=True)
        pts = vertices[idx]
    else:
        try:
            pts, _ = trimesh.sample.sample_surface(mesh, num_points)
        except Exception:
            idx = np.random.choice(len(vertices), num_points, replace=True)
            pts = vertices[idx]

    if normalize:
        pts = normalize_point_cloud(pts)

    return pts

def compute_chamfer_and_fscore(gen_points, gt_points, fscore_threshold=0.05):
    """
    Compute Bidirectional Chamfer Distance and F-Score between two normalized point clouds.
    """
    gen_tree = scipy.spatial.KDTree(gen_points)
    gt_tree = scipy.spatial.KDTree(gt_points)
    
    dist_gen_to_gt, _ = gt_tree.query(gen_points, k=1)
    dist_gt_to_gen, _ = gt_tree.query(gt_points, k=1)
    
    cd = float(np.mean(dist_gen_to_gt**2) + np.mean(dist_gt_to_gen**2))
    
    precision = np.mean(dist_gen_to_gt < fscore_threshold)
    recall = np.mean(dist_gt_to_gen < fscore_threshold)
    if precision + recall > 0:
        fscore = float(2.0 * (precision * recall) / (precision + recall))
    else:
        fscore = 0.0
        
    return cd, fscore

def compute_coverage(gen_pcs, gt_pcs, fscore_threshold=0.05):
    """
    Coverage (COV): % of GT point clouds matched by at least one generated shape.
    Higher is better (max 100.0%).
    """
    if len(gen_pcs) == 0 or len(gt_pcs) == 0:
        return 0.0

    matched_gt = set()
    for g_pc in gen_pcs:
        best_idx = -1
        best_cd = float('inf')
        for idx, t_pc in enumerate(gt_pcs):
            cd, _ = compute_chamfer_and_fscore(g_pc, t_pc, fscore_threshold)
            if cd < best_cd:
                best_cd = cd
                best_idx = idx
        if best_idx >= 0:
            matched_gt.add(best_idx)

    return float(len(matched_gt) / len(gt_pcs)) * 100.0

def compute_1nn_accuracy_decomposed(gen_pcs, gt_pcs, fscore_threshold=0.05, num_trials=10):
    """
    Compute 1-NN Classifier Accuracy with BALANCED 1:1 subsampling.
    Randomly subsamples min(len(gen), len(gt)) shapes across `num_trials` to remove size-imbalance bias.
    
    Ideal score for 1NN_Total, 1NN_Fake, 1NN_Real is 50.0%.
    """
    N = len(gen_pcs)
    M = len(gt_pcs)
    if N == 0 or M == 0:
        return {"1NN_Total (%)": 0.0, "1NN_Fake (%)": 0.0, "1NN_Real (%)": 0.0}

    K_sub = min(N, M)
    
    total_accs = []
    fake_accs = []
    real_accs = []

    rng = np.random.RandomState(42)

    for trial in range(num_trials):
        gen_indices = rng.choice(N, K_sub, replace=False) if N > K_sub else np.arange(N)
        gt_indices = rng.choice(M, K_sub, replace=False) if M > K_sub else np.arange(M)

        sub_gen = [gen_pcs[i] for i in gen_indices]
        sub_gt = [gt_pcs[i] for i in gt_indices]

        all_pcs = sub_gen + sub_gt
        labels = [0] * K_sub + [1] * K_sub
        Total_K = 2 * K_sub

        D = np.full((Total_K, Total_K), float('inf'))
        for i in range(Total_K):
            for j in range(i + 1, Total_K):
                cd, _ = compute_chamfer_and_fscore(all_pcs[i], all_pcs[j], fscore_threshold)
                D[i, j] = cd
                D[j, i] = cd

        fake_correct = 0
        for i in range(K_sub):
            nn_idx = np.argmin(D[i])
            if labels[nn_idx] == 0:
                fake_correct += 1

        real_correct = 0
        for i in range(K_sub, Total_K):
            nn_idx = np.argmin(D[i])
            if labels[nn_idx] == 1:
                real_correct += 1

        fake_accs.append((fake_correct / K_sub) * 100.0)
        real_accs.append((real_correct / K_sub) * 100.0)
        total_accs.append(((fake_correct + real_correct) / Total_K) * 100.0)

    return {
        "1NN_Total (%)": float(np.mean(total_accs)),
        "1NN_Fake (%)": float(np.mean(fake_accs)),
        "1NN_Real (%)": float(np.mean(real_accs))
    }

def compute_morphological_features(mesh):
    """
    Extract physical 3D morphological parameters from a mesh:
    - volume
    - surface area
    - aspect ratio (bounding box maximum-to-minimum dimension ratio)
    """
    if isinstance(mesh, trimesh.Scene):
        try:
            mesh = mesh.dump(concatenate=True)
        except Exception:
            mesh = None

    if mesh is None or isinstance(mesh, trimesh.PointCloud) or not hasattr(mesh, 'vertices') or len(mesh.vertices) == 0:
        return {"volume": 0.0, "area": 0.0, "aspect_ratio": 1.0}

    vol = abs(getattr(mesh, 'volume', 0.0))
    area = getattr(mesh, 'area', 0.0)

    try:
        extents = mesh.extents
        min_ext = min(extents)
        aspect_ratio = float(max(extents) / max(min_ext, 1e-7))
    except Exception:
        aspect_ratio = 1.0

    return {
        "volume": float(vol),
        "area": float(area),
        "aspect_ratio": float(aspect_ratio)
    }

def compute_wasserstein_distances(gen_features, gt_features):
    """
    Compute 1D Wasserstein Distance (W1) between generated and GT distributions
    for volume, surface area, and aspect ratio.
    Lower values indicate better distribution matching.
    """
    if not gen_features or not gt_features:
        return {"W1_Volume": 0.0, "W1_Area": 0.0, "W1_Aspect": 0.0}

    gen_vols = [f["volume"] for f in gen_features]
    gt_vols = [f["volume"] for f in gt_features]

    gen_areas = [f["area"] for f in gen_features]
    gt_areas = [f["area"] for f in gt_features]

    gen_aspects = [f["aspect_ratio"] for f in gen_features]
    gt_aspects = [f["aspect_ratio"] for f in gt_features]

    w1_vol = float(scipy.stats.wasserstein_distance(gen_vols, gt_vols))
    w1_area = float(scipy.stats.wasserstein_distance(gen_areas, gt_areas))
    w1_aspect = float(scipy.stats.wasserstein_distance(gen_aspects, gt_aspects))

    return {
        "W1_Volume": w1_vol,
        "W1_Area": w1_area,
        "W1_Aspect": w1_aspect
    }

def compute_sphericity(mesh):
    """
    Sphericity = (pi^(1/3) * (6 * Volume)^(2/3)) / Area
    Ranges from 0 to 1, with 1.0 indicating a perfect sphere.
    """
    if isinstance(mesh, trimesh.Scene):
        try:
            mesh = mesh.dump(concatenate=True)
        except Exception:
            mesh = None

    if mesh is None or isinstance(mesh, trimesh.PointCloud) or not hasattr(mesh, 'faces') or len(getattr(mesh, 'faces', [])) == 0:
        return 0.0

    try:
        vol = abs(getattr(mesh, 'volume', 0.0))
        area = getattr(mesh, 'area', 0.0)
        if area < 1e-7:
            return 0.0
        sphericity = (np.pi**(1.0/3.0) * (6.0 * vol)**(2.0/3.0)) / area
        return float(np.clip(sphericity, 0.0, 1.0))
    except Exception:
        return 0.0

def compute_mesh_quality(mesh):
    """
    Returns statistics about mesh quality:
    - watertight: True if closed
    - connected_components: Number of disconnected components
    - degenerate_faces_fraction: fraction of faces with area < 1e-7
    """
    if isinstance(mesh, trimesh.Scene):
        try:
            mesh = mesh.dump(concatenate=True)
        except Exception:
            mesh = None

    if mesh is None or isinstance(mesh, trimesh.PointCloud) or not hasattr(mesh, 'faces') or len(getattr(mesh, 'faces', [])) == 0:
        return {
            "watertight": False,
            "connected_components": 1,
            "degenerate_faces_fraction": 1.0
        }

    watertight = bool(getattr(mesh, 'is_watertight', False))
    try:
        cc_list = mesh.split(only_watertight=False)
        cc_count = len(cc_list) if len(cc_list) > 0 else 1
    except Exception:
        cc_count = 1

    total_faces = len(mesh.faces)
    if total_faces > 0:
        try:
            if hasattr(mesh, 'area_faces'):
                face_areas = mesh.area_faces
            elif hasattr(mesh, 'face_areas'):
                face_areas = mesh.face_areas
            else:
                v0 = mesh.vertices[mesh.faces[:, 0]]
                v1 = mesh.vertices[mesh.faces[:, 1]]
                v2 = mesh.vertices[mesh.faces[:, 2]]
                face_areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
            degenerate_count = np.sum(face_areas < 1e-7)
            degen_fraction = float(degenerate_count) / total_faces
        except Exception:
            degen_fraction = 0.0
    else:
        degen_fraction = 1.0

    return {
        "watertight": watertight,
        "connected_components": cc_count,
        "degenerate_faces_fraction": degen_fraction
    }
