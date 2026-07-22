import numpy as np
import scipy.spatial
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
    If normalize=True, scales point cloud to unit bounding sphere.
    """
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)

    vertices = getattr(mesh, 'vertices', None)
    faces = getattr(mesh, 'faces', None)

    if vertices is None or len(vertices) == 0:
        pts = np.random.randn(num_points, 3)
        pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        return pts

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
    dist_gt_to_gen, _ = gen_tree.query(gt_points, k=1)
    
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

def compute_1nn_accuracy(gen_pcs, gt_pcs, fscore_threshold=0.05):
    """
    1-NN Classifier Accuracy between generated set and GT set.
    Ideal score is 50.0% (50.0 = indistinguishable from real data).
    """
    N = len(gen_pcs)
    M = len(gt_pcs)
    if N == 0 or M == 0:
        return 0.0

    all_pcs = list(gen_pcs) + list(gt_pcs)
    labels = [0] * N + [1] * M
    K = N + M

    D = np.full((K, K), float('inf'))
    for i in range(K):
        for j in range(i + 1, K):
            cd, _ = compute_chamfer_and_fscore(all_pcs[i], all_pcs[j], fscore_threshold)
            D[i, j] = cd
            D[j, i] = cd

    correct = 0
    for i in range(K):
        nn_idx = np.argmin(D[i])
        if labels[nn_idx] == labels[i]:
            correct += 1

    return float(correct / K) * 100.0

def compute_sphericity(mesh):
    """
    Sphericity = (pi^(1/3) * (6 * Volume)^(2/3)) / Area
    Ranges from 0 to 1, with 1.0 indicating a perfect sphere.
    """
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)

    if isinstance(mesh, trimesh.PointCloud) or not hasattr(mesh, 'faces') or len(getattr(mesh, 'faces', [])) == 0:
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
        mesh = mesh.dump(concatenate=True)

    if isinstance(mesh, trimesh.PointCloud) or not hasattr(mesh, 'faces') or len(getattr(mesh, 'faces', [])) == 0:
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
