import numpy as np
import scipy.spatial
import trimesh

def sample_point_cloud(mesh, num_points=2048):
    """
    Sample points uniformly from the surface of the mesh or point cloud.
    If empty or face-less, samples from vertices or unit sphere.
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
        return vertices[idx]

    try:
        points, _ = trimesh.sample.sample_surface(mesh, num_points)
        return points
    except Exception:
        idx = np.random.choice(len(vertices), num_points, replace=True)
        return vertices[idx]

def compute_chamfer_and_fscore(gen_points, gt_points, fscore_threshold=0.02):
    """
    Compute Bidirectional Chamfer Distance and F-Score between two point clouds.
    """
    gen_tree = scipy.spatial.KDTree(gen_points)
    gt_tree = scipy.spatial.KDTree(gt_points)
    
    dist_gen_to_gt, _ = gt_tree.query(gen_points, k=1)
    dist_gt_to_gen, _ = gen_tree.query(gt_points, k=1)
    
    cd = float(np.mean(dist_gen_to_gt**2) + np.mean(dist_gt_to_gen**2))
    
    precision = np.mean(dist_gen_to_gt < fscore_threshold)
    recall = np.mean(dist_gt_to_gen < fscore_threshold)
    if precision + recall > 0:
        fscore = float(2 * (precision * recall) / (precision + recall))
    else:
        fscore = 0.0
        
    return cd, fscore

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
