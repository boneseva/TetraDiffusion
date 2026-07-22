#!/usr/bin/env python3
"""
plot_shape_space_html.py — Interactive Web HTML 3D Shape Space & Orbit Viewer.

Generates a standalone, self-contained interactive HTML web page with:
  1. 2D MDS Metric Space Map (Plotly.js): Click any shape dot to inspect it.
  2. Interactive 3D WebGL Orbit Viewer (Three.js): Rotate 360°, pan, zoom, and lighting controls
     to inspect real GT vs generated 3D organelle meshes from any camera angle.
  3. 1-NN Match Highlighting: Instantly highlights a shape's nearest neighbor on click.
"""

import os
import glob
import argparse
import time
import json
import numpy as np
import scipy.spatial
import trimesh

def normalize_point_cloud(pc):
    """Center point cloud at origin and scale to unit bounding sphere."""
    if len(pc) == 0:
        return pc
    centroid = np.mean(pc, axis=0)
    pc_centered = pc - centroid
    max_radius = np.max(np.linalg.norm(pc_centered, axis=1))
    if max_radius > 1e-7:
        return pc_centered / max_radius
    return pc_centered

def sample_point_cloud(mesh, num_points=1500):
    """Sample normalized point cloud from mesh or point cloud."""
    if isinstance(mesh, trimesh.Scene):
        try:
            mesh = mesh.dump(concatenate=True)
        except Exception:
            mesh = None

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

    return normalize_point_cloud(pts)

def compute_chamfer_distance(pc1, pc2):
    """Compute bidirectional Chamfer Distance between two point clouds."""
    tree1 = scipy.spatial.KDTree(pc1)
    tree2 = scipy.spatial.KDTree(pc2)
    d1, _ = tree2.query(pc1, k=1)
    d2, _ = tree1.query(pc2, k=1)
    return float(np.mean(d1**2) + np.mean(d2**2))

def classical_mds(D, n_components=2):
    """Classical Multidimensional Scaling (PCoA) from distance matrix D."""
    K = D.shape[0]
    H = np.eye(K) - np.ones((K, K)) / K
    B = -0.5 * H.dot(D**2).dot(H)
    evals, evecs = np.linalg.eigh(B)
    idx = np.argsort(evals)[::-1][:n_components]
    evals = np.maximum(evals[idx], 0.0)
    evecs = evecs[:, idx]
    return evecs * np.sqrt(evals)

def build_interactive_html(shapes_data, mds_coords, nn_indices, run_name):
    """Build standalone self-contained HTML page with Plotly + Three.js."""

    shapes_json = json.dumps(shapes_data)
    coords_json = json.dumps(mds_coords.tolist())
    nn_json = json.dumps(nn_indices)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TetraDiffusion — 3D Shape Space Explorer ({run_name})</title>

    <!-- Load Plotly & Three.js CDN -->
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>

    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background: #111827;
            color: #f3f4f6;
            height: 100vh;
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }}
        header {{
            background: #1f2937;
            padding: 12px 24px;
            border-bottom: 1px solid #374151;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        header h1 {{ font-size: 18px; font-weight: 700; color: #60a5fa; }}
        header span {{ font-size: 13px; color: #9ca3af; }}

        .container {{
            flex: 1;
            display: flex;
            height: calc(100vh - 55px);
        }}
        #plot-container {{
            flex: 1.1;
            background: #111827;
            border-right: 1px solid #374151;
            position: relative;
        }}
        #viewer-container {{
            flex: 0.9;
            background: #030712;
            display: flex;
            flex-direction: column;
            position: relative;
        }}
        #webgl-canvas {{
            width: 100%;
            height: 100%;
            display: block;
        }}
        .info-panel {{
            position: absolute;
            top: 16px;
            left: 16px;
            background: rgba(17, 24, 39, 0.85);
            backdrop-filter: blur(8px);
            padding: 12px 16px;
            border-radius: 8px;
            border: 1px solid #374151;
            font-size: 13px;
            z-index: 10;
            pointer-events: none;
        }}
        .info-panel h3 {{ font-size: 14px; margin-bottom: 6px; color: #38bdf8; }}
        .badge {{
            display: inline-block;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 11px;
            font-weight: 600;
            margin-right: 6px;
        }}
        .badge-gt {{ background: #1d4ed8; color: #dbeafe; }}
        .badge-gen {{ background: #b91c1c; color: #fee2e2; }}

        .controls-hint {{
            position: absolute;
            bottom: 16px;
            left: 16px;
            background: rgba(17, 24, 39, 0.8);
            padding: 8px 12px;
            border-radius: 6px;
            font-size: 11px;
            color: #9ca3af;
            z-index: 10;
        }}
    </style>
</head>
<body>
    <header>
        <h1>3D Organelle Shape Space Explorer</h1>
        <span>Run: <strong>{run_name}</strong> | Click any dot on the map to orbit the 3D shape</span>
    </header>

    <div class="container">
        <!-- 2D Metric Space Plot -->
        <div id="plot-container"></div>

        <!-- 3D Orbit Viewer -->
        <div id="viewer-container">
            <div class="info-panel" id="info-panel">
                <h3 id="shape-name">Select a shape on the left</h3>
                <p id="shape-meta">Click any blue (GT) or red (Generated) point to view 3D geometry.</p>
            </div>
            <div class="controls-hint">💡 <strong>3D Controls:</strong> Left-click + Drag to rotate | Right-click to pan | Scroll to zoom</div>
            <div id="canvas-holder" style="width: 100%; height: 100%;"></div>
        </div>
    </div>

    <script>
        const shapes = {shapes_json};
        const coords = {coords_json};
        const nnIndices = {nn_json};

        // Prepare Plotly data
        const gtIdxs = [];
        const genIdxs = [];
        shapes.forEach((s, idx) => {{
            if (s.is_gt) gtIdxs.push(idx);
            else genIdxs.push(idx);
        }});

        const gtTrace = {{
            x: gtIdxs.map(i => coords[i][0]),
            y: gtIdxs.map(i => coords[i][1]),
            mode: 'markers',
            type: 'scatter',
            name: 'Ground Truth (Real)',
            text: gtIdxs.map(i => shapes[i].name),
            marker: {{ color: '#3b82f6', size: 10, symbol: 'circle', line: {{ color: '#1d4ed8', width: 1.5 }} }},
            customdata: gtIdxs
        }};

        const genTrace = {{
            x: genIdxs.map(i => coords[i][0]),
            y: genIdxs.map(i => coords[i][1]),
            mode: 'markers',
            type: 'scatter',
            name: 'Generated (Model)',
            text: genIdxs.map(i => shapes[i].name),
            marker: {{ color: '#ef4444', size: 9, symbol: 'triangle-up', line: {{ color: '#b91c1c', width: 1.5 }} }},
            customdata: genIdxs
        }};

        // Draw 1-NN connecting lines
        const linesTrace = {{
            x: [],
            y: [],
            mode: 'lines',
            type: 'scatter',
            name: '1-NN Connections',
            line: {{ color: '#4b5563', width: 1, dash: 'dot' }},
            hoverinfo: 'none',
            showlegend: true
        }};

        shapes.forEach((s, i) => {{
            const j = nnIndices[i];
            linesTrace.x.push(coords[i][0], coords[j][0], null);
            linesTrace.y.push(coords[i][1], coords[j][1], null);
        }});

        const layout = {{
            title: {{ text: 'MDS 2D Shape Metric Space (Chamfer Distance)', font: {{ color: '#e5e7eb', size: 14 }} }},
            paper_bgcolor: '#111827',
            plot_bgcolor: '#1f2937',
            xaxis: {{ gridcolor: '#374151', zerolinecolor: '#4b5563', tickfont: {{ color: '#9ca3af' }} }},
            yaxis: {{ gridcolor: '#374151', zerolinecolor: '#4b5563', tickfont: {{ color: '#9ca3af' }} }},
            legend: {{ font: {{ color: '#e5e7eb' }}, bgcolor: 'rgba(31,41,55,0.8)' }},
            margin: {{ l: 50, r: 20, t: 50, b: 50 }}
        }};

        Plotly.newPlot('plot-container', [linesTrace, gtTrace, genTrace], layout, {{ responsive: true }});

        // Setup Three.js 3D Viewer
        const holder = document.getElementById('canvas-holder');
        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0x030712);

        const camera = new THREE.PerspectiveCamera(45, holder.clientWidth / holder.clientHeight, 0.1, 100);
        camera.position.set(0, 0, 2.5);

        const renderer = new THREE.WebGLRenderer({{ antialias: true }});
        renderer.setSize(holder.clientWidth, holder.clientHeight);
        renderer.setPixelRatio(window.devicePixelRatio);
        holder.appendChild(renderer.domElement);

        const controls = new THREE.OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.dampingFactor = 0.05;

        // Lighting
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
        scene.add(ambientLight);
        const dirLight1 = new THREE.DirectionalLight(0xffffff, 0.8);
        dirLight1.position.set(1, 2, 3);
        scene.add(dirLight1);
        const dirLight2 = new THREE.DirectionalLight(0x38bdf8, 0.4);
        dirLight2.position.set(-2, -1, -2);
        scene.add(dirLight2);

        let currentPointCloudMesh = null;

        function loadShapeIn3D(shapeIdx) {{
            const s = shapes[shapeIdx];
            const nnIdx = nnIndices[shapeIdx];
            const nnShape = shapes[nnIdx];

            // Update info panel
            const panelName = document.getElementById('shape-name');
            const panelMeta = document.getElementById('shape-meta');
            
            const badgeClass = s.is_gt ? 'badge-gt' : 'badge-gen';
            const badgeText = s.is_gt ? 'GROUND TRUTH' : 'GENERATED';
            panelName.innerHTML = `<span class="badge ${{badgeClass}}">${{badgeText}}</span> ${{s.name}}`;
            panelMeta.innerHTML = `Nearest Neighbor 1-NN match: <strong>${{nnShape.name}}</strong> (${{nnShape.is_gt ? 'Ground Truth' : 'Generated'}})`;

            // Remove existing mesh
            if (currentPointCloudMesh) scene.remove(currentPointCloudMesh);

            // Create PointCloud Geometry
            const geometry = new THREE.BufferGeometry();
            const positions = new Float32Array(s.pc.flat());
            geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));

            const colorHex = s.is_gt ? 0x3b82f6 : 0xef4444;
            const material = new THREE.PointsMaterial({{
                color: colorHex,
                size: 0.035,
                sizeAttenuation: true
            }});

            currentPointCloudMesh = new THREE.Points(geometry, material);
            scene.add(currentPointCloudMesh);
        }}

        // Handle Plotly click events
        document.getElementById('plot-container').on('plotly_click', function(data) {{
            if (data.points.length > 0) {{
                const point = data.points[0];
                if (point.customdata !== undefined) {{
                    loadShapeIn3D(point.customdata);
                }}
            }}
        }});

        // Load first shape by default
        loadShapeIn3D(0);

        // Animation Loop
        function animate() {{
            requestAnimationFrame(animate);
            controls.update();
            renderer.render(scene, camera);
        }}
        animate();

        // Handle Window Resize
        window.addEventListener('resize', () => {{
            camera.aspect = holder.clientWidth / holder.clientHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(holder.clientWidth, holder.clientHeight);
        }});
    </script>
</body>
</html>
"""
    return html_content

def main():
    parser = argparse.ArgumentParser(description="Generate Interactive Web HTML 3D Shape Space Explorer.")
    parser.add_argument("--run_dir", type=str, required=True, help="Path to run inference output folder.")
    parser.add_argument("--gt_dir", type=str, default="../data_test/organelles/lyso", help="Path to GT directory.")
    parser.add_argument("--points", type=int, default=1500, help="Number of points per cloud.")
    parser.add_argument("--max_gen", type=int, default=50, help="Max generated shapes to include in web explorer.")
    parser.add_argument("--output", type=str, default=None, help="Output HTML filename.")
    args = parser.parse_args()

    gt_files = sorted(glob.glob(os.path.join(args.gt_dir, "*.obj")))
    gen_files = sorted(glob.glob(os.path.join(args.run_dir, "*.obj")))[:args.max_gen]

    if not gt_files or not gen_files:
        print(f"Error: Need both GT files ({len(gt_files)}) and Gen files ({len(gen_files)}).")
        return

    print(f"Loading {len(gt_files)} GT meshes and {len(gen_files)} Generated meshes for HTML explorer...")
    shapes_data = []

    for f in gt_files:
        pc = sample_point_cloud(trimesh.load(f), args.points)
        shapes_data.append({
            "name": os.path.basename(f),
            "is_gt": True,
            "pc": pc.tolist()
        })

    for f in gen_files:
        pc = sample_point_cloud(trimesh.load(f), args.points)
        shapes_data.append({
            "name": os.path.basename(f),
            "is_gt": False,
            "pc": pc.tolist()
        })

    num_gt = len(gt_files)
    num_gen = len(gen_files)
    Total = num_gt + num_gen

    print(f"Computing {Total}x{Total} pairwise Chamfer Distance matrix...")
    t0 = time.time()
    all_pcs = [np.array(s["pc"]) for s in shapes_data]
    D = np.zeros((Total, Total))
    for i in range(Total):
        for j in range(i + 1, Total):
            cd = compute_chamfer_distance(all_pcs[i], all_pcs[j])
            D[i, j] = cd
            D[j, i] = cd
    print(f"Distance matrix computed in {time.time() - t0:.1f}s.")

    nn_indices = []
    for i in range(Total):
        D_temp = D[i].copy()
        D_temp[i] = float('inf')
        nn_indices.append(int(np.argmin(D_temp)))

    print("Projecting distance matrix to 2D via Classical MDS...")
    coords = classical_mds(D, n_components=2)

    run_name = os.path.basename(os.path.normpath(args.run_dir))
    html_page = build_interactive_html(shapes_data, coords, nn_indices, run_name)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    out_name = args.output or f"shape_space_interactive_{run_name}.html"
    out_path = os.path.join(results_dir, out_name)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html_page)

    print(f"\nInteractive 3D Shape Space Web Explorer saved to:")
    print(f"  → {out_path}")
    print(f"\nTo open, double-click '{out_path}' or open it in any web browser!")

if __name__ == '__main__':
    main()
