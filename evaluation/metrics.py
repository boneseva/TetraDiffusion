import numpy as np
import scipy.spatial
import trimesh

def sample_point_cloud(mesh, num_points=2048):
    """
    Sample points uniformly from the surface of the mesh.
    If the mesh is empty or has no faces, returns random points on a sphere.
    """
    if len(mesh.faces) == 0 or len(mesh.vertices) == 0:
        # Fallback for empty/degenerate meshes
        pts = np.random.randn(num_points, 3)
        pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        return pts
    try:
        # sample_surface is standard in trimesh
        points, _ = trimesh.sample.sample_surface(mesh, num_points)
        return points
    except Exception:
        # Backup sampler
        return mesh.vertices[np.random.choice(len(mesh.vertices), num_points, replace=True)]

def compute_chamfer_and_fscore(gen_points, gt_points, fscore_threshold=0.02):
    """
    Compute Bidirectional Chamfer Distance and F-Score between two point clouds.
    CD = (1/N)*sum_i(min_j ||g_i - t_j||^2) + (1/M)*sum_j(min_i ||t_j - g_i||^2)
    """
    gen_tree = scipy.spatial.KDTree(gen_points)
    gt_tree = scipy.spatial.KDTree(gt_points)
    
    # Distance from generated points to closest GT points
    dist_gen_to_gt, _ = gt_tree.query(gen_points, k=1)
    # Distance from GT points to closest generated points
    dist_gt_to_gen, _ = gen_tree.query(gt_points, k=1)
    
    # Chamfer Distance (mean squared Euclidean distance)
    cd = np.mean(dist_gen_to_gt**2) + np.mean(dist_gt_to_gen**2)
    
    # F-score: percentage of points within the threshold distance
    precision = np.mean(dist_gen_to_gt < fscore_threshold)
    recall = np.mean(dist_gt_to_gen < fscore_threshold)
    if precision + recall > 0:
        fscore = 2 * (precision * recall) / (precision + recall)
    else:
        fscore = 0.0
        
    return cd, fscore

def compute_sphericity(mesh):
    """
    Sphericity = (pi^(1/3) * (6 * Volume)^(2/3)) / Area
    Ranges from 0 to 1, with 1.0 indicating a perfect sphere.
    """
    try:
        vol = abs(mesh.volume)
        area = mesh.area
        if area < 1e-7:
            return 0.0
        sphericity = (np.pi**(1.0/3.0) * (6.0 * vol)**(2.0/3.0)) / area
        return np.clip(sphericity, 0.0, 1.0)
    except Exception:
        return 0.0

def compute_mesh_quality(mesh):
    """
    Returns statistics about mesh quality:
    - watertight: True if closed
    - connected_components: Number of disconnected components
    - degenerate_faces_fraction: fraction of faces with area < 1e-7
    """
    watertight = bool(mesh.is_watertight)
    try:
        # split returns list of separate mesh components
        cc_list = mesh.split(only_watertight=False)
        cc_count = len(cc_list) if len(cc_list) > 0 else 1
    except Exception:
        cc_count = 1
        
    total_faces = len(mesh.faces)
    if total_faces > 0:
        degenerate_count = np.sum(mesh.face_areas < 1e-7)
        degen_fraction = float(degenerate_count) / total_faces
    else:
        degen_fraction = 1.0
        
    return {
        "watertight": watertight,
        "connected_components": cc_count,
        "degenerate_faces_fraction": degen_fraction
    }
