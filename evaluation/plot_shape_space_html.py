#!/usr/bin/env python3
"""
plot_shape_space_html.py — Interactive Web HTML 3D Shape Space & Orbit Mesh Explorer.

Generates a standalone, self-contained interactive HTML web page with:
  1. Modern 2D MDS Metric Space Map (Plotly.js): Click any shape dot to inspect it.
  2. Interactive 3D WebGL Orbit Mesh Viewer (Three.js): Soft satin-shaded 3D surface meshes
     (GT vs. Generated organelles) with realistic hemisphere lighting (no harsh glare).
  3. Minimizable 1-NN Secondary 3D Viewer: A floating inset 3D viewer showing the selected
     shape's 1-NN nearest neighbor, with a toggle button to collapse/minimize it, synchronized camera rotation,
     and color coding (GT vs. Gen).
"""

import os
import glob
import argparse
import time
import json
import numpy as np
import scipy.spatial
import trimesh

def process_mesh_and_pc(file_path, num_points=1500):
    """
    Load 3D mesh, normalize vertices to unit bounding sphere, extract mesh geometry
    and sample a point cloud for metric space calculations.
    """
    try:
        mesh = trimesh.load(file_path)
    except Exception as e:
        print(f"Warning: Failed to load {file_path}: {e}")
        mesh = None

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
        return pts, {"vertices": [], "faces": []}

    # Center at origin and scale to unit sphere
    centroid = np.mean(vertices, axis=0)
    verts_centered = vertices - centroid
    max_radius = np.max(np.linalg.norm(verts_centered, axis=1))
    scale = (1.0 / max_radius) if max_radius > 1e-7 else 1.0
    verts_norm = verts_centered * scale

    if faces is None or len(faces) == 0:
        idx = np.random.choice(len(verts_norm), num_points, replace=True)
        pc = verts_norm[idx]
        mesh_dict = {"vertices": np.round(verts_norm, 4).tolist(), "faces": []}
    else:
        # Simplify mesh if face count is very high (> 4000) for lightweight HTML payload
        if len(faces) > 4000:
            try:
                simplified = mesh.simplify_quadric_decimation(3000)
                s_verts = (simplified.vertices - centroid) * scale
                s_faces = simplified.faces.tolist()
                mesh_dict = {"vertices": np.round(s_verts, 4).tolist(), "faces": s_faces}
            except Exception:
                mesh_dict = {"vertices": np.round(verts_norm, 4).tolist(), "faces": faces.tolist()}
        else:
            mesh_dict = {"vertices": np.round(verts_norm, 4).tolist(), "faces": faces.tolist()}

        try:
            temp_mesh = trimesh.Trimesh(vertices=verts_norm, faces=faces, process=False)
            pc, _ = trimesh.sample.sample_surface(temp_mesh, num_points)
        except Exception:
            idx = np.random.choice(len(verts_norm), num_points, replace=True)
            pc = verts_norm[idx]

    return pc, mesh_dict

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

def build_interactive_html(shapes_data, mds_coords, nn_indices, D_matrix, run_name):
    """Build standalone self-contained HTML page with Plotly + Three.js 3D Mesh Viewers."""

    shapes_json = json.dumps(shapes_data)
    coords_json = json.dumps(mds_coords.tolist())
    nn_json = json.dumps(nn_indices)
    distances_json = json.dumps(np.round(D_matrix, 6).tolist())

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TetraDiffusion — 3D Shape Space Explorer ({run_name})</title>

    <!-- Google Fonts Inter -->
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">

    <!-- Load Plotly & Three.js CDN -->
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>

    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background: #090d16;
            color: #f8fafc;
            height: 100vh;
            display: flex;
            flex-direction: column;
            overflow: hidden;
            -webkit-font-smoothing: antialiased;
        }}

        header {{
            background: rgba(15, 23, 42, 0.8);
            backdrop-filter: blur(12px);
            padding: 12px 24px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.08);
            display: flex;
            justify-content: space-between;
            align-items: center;
            z-index: 50;
        }}
        header .brand {{
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        header .brand-dot {{
            width: 10px;
            height: 10px;
            border-radius: 50%;
            background: #38bdf8;
            box-shadow: 0 0 10px #38bdf8;
        }}
        header h1 {{ font-size: 16px; font-weight: 600; color: #f8fafc; letter-spacing: -0.01em; }}
        header .run-tag {{ font-size: 13px; color: #94a3b8; font-weight: 400; }}
        header .run-tag strong {{ color: #38bdf8; font-weight: 500; }}

        .container {{
            flex: 1;
            display: flex;
            height: calc(100vh - 53px);
        }}
        #plot-container {{
            flex: 1.15;
            background: #090d16;
            border-right: 1px solid rgba(255, 255, 255, 0.08);
            position: relative;
        }}
        #viewer-container {{
            flex: 0.85;
            background: #060911;
            display: flex;
            flex-direction: column;
            position: relative;
            overflow: hidden;
        }}
        #canvas-holder {{
            width: 100%;
            height: 100%;
            display: block;
        }}

        /* Modern Glassmorphism Info Panel */
        .info-panel {{
            position: absolute;
            top: 16px;
            left: 16px;
            background: rgba(15, 23, 42, 0.75);
            backdrop-filter: blur(16px) saturate(180%);
            padding: 14px 18px;
            border-radius: 12px;
            border: 1px solid rgba(255, 255, 255, 0.08);
            font-size: 13px;
            z-index: 10;
            pointer-events: none;
            max-width: 380px;
            box-shadow: 0 20px 30px -10px rgba(0, 0, 0, 0.7);
        }}
        .info-panel h3 {{
            font-size: 14px;
            font-weight: 600;
            margin-bottom: 6px;
            color: #f8fafc;
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        .info-panel p {{
            color: #94a3b8;
            line-height: 1.45;
            font-size: 12px;
        }}

        /* Badges */
        .badge {{
            display: inline-flex;
            align-items: center;
            padding: 3px 8px;
            border-radius: 6px;
            font-size: 11px;
            font-weight: 600;
            letter-spacing: 0.02em;
            text-transform: uppercase;
        }}
        .badge-gt {{
            background: rgba(56, 189, 248, 0.15);
            color: #38bdf8;
            border: 1px solid rgba(56, 189, 248, 0.3);
            box-shadow: 0 0 10px rgba(56, 189, 248, 0.15);
        }}
        .badge-gen {{
            background: rgba(251, 113, 133, 0.15);
            color: #fb7185;
            border: 1px solid rgba(251, 113, 133, 0.3);
            box-shadow: 0 0 10px rgba(251, 113, 133, 0.15);
        }}

        .controls-hint {{
            position: absolute;
            bottom: 16px;
            left: 16px;
            background: rgba(15, 23, 42, 0.75);
            backdrop-filter: blur(12px);
            padding: 8px 14px;
            border-radius: 8px;
            border: 1px solid rgba(255, 255, 255, 0.06);
            font-size: 11px;
            color: #64748b;
            z-index: 10;
            pointer-events: none;
        }}

        /* Floating 1-NN Secondary 3D Viewer Inset (Minimizable) */
        #nn-card {{
            position: absolute;
            bottom: 16px;
            right: 16px;
            width: 280px;
            height: 260px;
            background: rgba(15, 23, 42, 0.85);
            backdrop-filter: blur(16px) saturate(180%);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 12px;
            display: flex;
            flex-direction: column;
            overflow: hidden;
            box-shadow: 0 20px 30px -10px rgba(0, 0, 0, 0.7);
            z-index: 20;
            transition: all 0.35s cubic-bezier(0.16, 1, 0.3, 1);
        }}

        #nn-card.minimized {{
            height: 42px !important;
            width: 230px !important;
        }}

        .nn-card-header {{
            padding: 10px 12px;
            background: rgba(30, 41, 59, 0.7);
            border-bottom: 1px solid rgba(255, 255, 255, 0.08);
            font-size: 12px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            user-select: none;
        }}
        .nn-card-header span.title {{ font-weight: 500; color: #cbd5e1; font-size: 12px; }}

        .toggle-btn {{
            background: rgba(255, 255, 255, 0.06);
            border: 1px solid rgba(255, 255, 255, 0.1);
            color: #94a3b8;
            cursor: pointer;
            width: 22px;
            height: 22px;
            border-radius: 5px;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: all 0.2s ease;
        }}
        .toggle-btn:hover {{
            background: rgba(255, 255, 255, 0.15);
            color: #f8fafc;
        }}

        #nn-canvas-holder {{
            flex: 1;
            position: relative;
            width: 100%;
            height: 100%;
            background: #060911;
            transition: opacity 0.25s ease;
        }}
        #nn-card.minimized #nn-canvas-holder,
        #nn-card.minimized .nn-footer-label {{
            opacity: 0;
            pointer-events: none;
        }}

        .nn-footer-label {{
            position: absolute;
            bottom: 6px;
            left: 8px;
            right: 8px;
            font-size: 11px;
            color: #cbd5e1;
            background: rgba(15, 23, 42, 0.75);
            backdrop-filter: blur(8px);
            padding: 4px 8px;
            border-radius: 6px;
            pointer-events: none;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            border: 1px solid rgba(255, 255, 255, 0.05);
            transition: opacity 0.25s ease;
        }}
    </style>
</head>
<body>
    <header>
        <div class="brand">
            <div class="brand-dot"></div>
            <h1>3D Organelle Shape Space Explorer</h1>
        </div>
        <span class="run-tag">Run: <strong>{run_name}</strong> | Click dots to orbit 3D meshes</span>
    </header>

    <div class="container">
        <!-- 2D Metric Space Plot -->
        <div id="plot-container"></div>

        <!-- 3D Orbit Viewer -->
        <div id="viewer-container">
            <div class="info-panel" id="info-panel">
                <h3 id="shape-name">Select a shape on the left</h3>
                <p id="shape-meta">Click any blue (GT) or red (Generated) point to orbit 3D mesh geometry.</p>
            </div>
            
            <div class="controls-hint">💡 <strong>3D Controls:</strong> Left-click + Drag to rotate | Right-click to pan | Scroll to zoom</div>
            
            <!-- Main Three.js Canvas Holder -->
            <div id="canvas-holder"></div>

            <!-- Inset 1-NN Secondary 3D Viewer (Minimizable) -->
            <div id="nn-card">
                <div class="nn-card-header">
                    <div style="display: flex; align-items: center; gap: 6px;">
                        <span class="title">1-NN Match</span>
                        <span id="nn-badge" class="badge"></span>
                    </div>
                    <button class="toggle-btn" id="nn-toggle-btn" onclick="toggleNNCard()" title="Minimize / Expand 1-NN Viewer">
                        <svg id="toggle-icon" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
                            <line x1="5" y1="12" x2="19" y2="12"></line>
                        </svg>
                    </button>
                </div>
                <div id="nn-canvas-holder"></div>
                <div class="nn-footer-label" id="nn-footer-name">No shape selected</div>
            </div>
        </div>
    </div>

    <script>
        const shapes = {shapes_json};
        const coords = {coords_json};
        const nnIndices = {nn_json};
        const distances = {distances_json};

        // Toggle Minimize / Expand 1-NN Viewer Card
        function toggleNNCard() {{
            const card = document.getElementById('nn-card');
            const icon = document.getElementById('toggle-icon');
            card.classList.toggle('minimized');

            if (card.classList.contains('minimized')) {{
                // Expand icon (plus / maximize)
                icon.innerHTML = '<polyline points="15 3 21 3 21 9"></polyline><polyline points="9 21 3 21 3 15"></polyline><line x1="21" y1="3" x2="14" y2="10"></line><line x1="3" y1="21" x2="10" y2="14"></line>';
            }} else {{
                // Minimize icon (minus)
                icon.innerHTML = '<line x1="5" y1="12" x2="19" y2="12"></line>';
            }}
        }}

        // Separate indices into GT and Generated
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
            marker: {{ color: '#38bdf8', size: 10, symbol: 'circle', line: {{ color: '#0284c7', width: 1.5 }} }},
            customdata: gtIdxs
        }};

        const genTrace = {{
            x: genIdxs.map(i => coords[i][0]),
            y: genIdxs.map(i => coords[i][1]),
            mode: 'markers',
            type: 'scatter',
            name: 'Generated (Model)',
            text: genIdxs.map(i => shapes[i].name),
            marker: {{ color: '#fb7185', size: 9, symbol: 'triangle-up', line: {{ color: '#e11d48', width: 1.5 }} }},
            customdata: genIdxs
        }};

        // Draw 1-NN connecting lines
        const linesTrace = {{
            x: [],
            y: [],
            mode: 'lines',
            type: 'scatter',
            name: '1-NN Connections',
            line: {{ color: '#334155', width: 1, dash: 'dot' }},
            hoverinfo: 'none',
            showlegend: true
        }};

        shapes.forEach((s, i) => {{
            const j = nnIndices[i];
            linesTrace.x.push(coords[i][0], coords[j][0], null);
            linesTrace.y.push(coords[i][1], coords[j][1], null);
        }});

        const layout = {{
            title: {{ text: 'MDS 2D Shape Metric Space (Chamfer Distance)', font: {{ family: 'Inter', color: '#f8fafc', size: 14 }} }},
            paper_bgcolor: '#090d16',
            plot_bgcolor: '#0f172a',
            xaxis: {{ gridcolor: '#1e293b', zerolinecolor: '#334155', tickfont: {{ family: 'Inter', color: '#64748b' }} }},
            yaxis: {{ gridcolor: '#1e293b', zerolinecolor: '#334155', tickfont: {{ family: 'Inter', color: '#64748b' }} }},
            legend: {{ font: {{ family: 'Inter', color: '#cbd5e1' }}, bgcolor: 'rgba(15, 23, 42, 0.8)' }},
            margin: {{ l: 50, r: 20, t: 50, b: 50 }}
        }};

        Plotly.newPlot('plot-container', [linesTrace, gtTrace, genTrace], layout, {{ responsive: true }});

        // Setup Main Three.js 3D Viewer
        const holder = document.getElementById('canvas-holder');
        const mainScene = new THREE.Scene();
        mainScene.background = new THREE.Color(0x060911);

        const mainCamera = new THREE.PerspectiveCamera(45, holder.clientWidth / holder.clientHeight, 0.1, 100);
        mainCamera.position.set(0, 0, 2.5);

        const mainRenderer = new THREE.WebGLRenderer({{ antialias: true }});
        mainRenderer.setSize(holder.clientWidth, holder.clientHeight);
        mainRenderer.setPixelRatio(window.devicePixelRatio);
        holder.appendChild(mainRenderer.domElement);

        const mainControls = new THREE.OrbitControls(mainCamera, mainRenderer.domElement);
        mainControls.enableDamping = true;
        mainControls.dampingFactor = 0.05;

        // Setup 1-NN Secondary Inset 3D Viewer
        const nnHolder = document.getElementById('nn-canvas-holder');
        const nnScene = new THREE.Scene();
        nnScene.background = new THREE.Color(0x060911);

        const nnCamera = new THREE.PerspectiveCamera(45, nnHolder.clientWidth / nnHolder.clientHeight, 0.1, 100);
        nnCamera.position.set(0, 0, 2.5);

        const nnRenderer = new THREE.WebGLRenderer({{ antialias: true }});
        nnRenderer.setSize(nnHolder.clientWidth, nnHolder.clientHeight);
        nnRenderer.setPixelRatio(window.devicePixelRatio);
        nnHolder.appendChild(nnRenderer.domElement);

        // Soft, Natural 3D Lighting Setup (Eliminates harsh plastic shine!)
        function setupLighting(scene) {{
            // Sky & ground ambient gradient light
            const hemiLight = new THREE.HemisphereLight(0xe0f2fe, 0x0f172a, 0.75);
            scene.add(hemiLight);

            // Key directional light (soft main light)
            const keyLight = new THREE.DirectionalLight(0xffffff, 0.65);
            keyLight.position.set(2, 4, 3);
            scene.add(keyLight);

            // Fill light (cyan tint for organic contrast)
            const fillLight = new THREE.DirectionalLight(0x38bdf8, 0.25);
            fillLight.position.set(-3, 1, -2);
            scene.add(fillLight);

            // Soft rim light
            const rimLight = new THREE.DirectionalLight(0xfb7185, 0.2);
            rimLight.position.set(0, -3, 2);
            scene.add(rimLight);
        }}
        setupLighting(mainScene);
        setupLighting(nnScene);

        function createMeshObject(shapeData) {{
            const geometry = new THREE.BufferGeometry();
            const hasMesh = shapeData.mesh && shapeData.mesh.vertices && shapeData.mesh.vertices.length > 0;

            let object3D;
            if (hasMesh && shapeData.mesh.faces && shapeData.mesh.faces.length > 0) {{
                // Construct 3D Surface Mesh
                const vertices = new Float32Array(shapeData.mesh.vertices.flat());
                geometry.setAttribute('position', new THREE.BufferAttribute(vertices, 3));

                const faces = new Uint32Array(shapeData.mesh.faces.flat());
                geometry.setIndex(new THREE.BufferAttribute(faces, 1));
                geometry.computeVertexNormals();

                // Blue for GT, Red for Generated (using MeshStandardMaterial with satin matte finish)
                const colorHex = shapeData.is_gt ? 0x38bdf8 : 0xfb7185;
                const material = new THREE.MeshStandardMaterial({{
                    color: colorHex,
                    roughness: 0.48,      // Satin-matte finish (no plastic glare!)
                    metalness: 0.12,       // Gentle surface response
                    side: THREE.DoubleSide
                }});

                object3D = new THREE.Mesh(geometry, material);
            }} else {{
                // Fallback to Point Cloud if no faces exist
                const positions = new Float32Array(shapeData.pc.flat());
                geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));

                const colorHex = shapeData.is_gt ? 0x38bdf8 : 0xfb7185;
                const material = new THREE.PointsMaterial({{
                    color: colorHex,
                    size: 0.035,
                    sizeAttenuation: true
                }});

                object3D = new THREE.Points(geometry, material);
            }}

            return object3D;
        }}

        let currentMainMesh = null;
        let currentNNMesh = null;

        function loadShapeIn3D(shapeIdx) {{
            const s = shapes[shapeIdx];
            const nnIdx = nnIndices[shapeIdx];
            const nnShape = shapes[nnIdx];
            const distVal = distances[shapeIdx][nnIdx];

            // Update main info panel
            const panelName = document.getElementById('shape-name');
            const panelMeta = document.getElementById('shape-meta');

            const badgeClass = s.is_gt ? 'badge-gt' : 'badge-gen';
            const badgeText = s.is_gt ? 'GROUND TRUTH' : 'GENERATED';
            panelName.innerHTML = `<span class="badge ${{badgeClass}}">${{badgeText}}</span> ${{s.name}}`;

            const nnBadgeClass = nnShape.is_gt ? 'badge-gt' : 'badge-gen';
            const nnBadgeText = nnShape.is_gt ? 'Ground Truth' : 'Generated';
            panelMeta.innerHTML = `1-NN Chamfer Distance: <strong>${{distVal.toFixed(5)}}</strong><br>Closest Match: <span class="badge ${{nnBadgeClass}}">${{nnBadgeText}}</span> <strong>${{nnShape.name}}</strong>`;

            // Update 1-NN Secondary Inset Card Header & Footer
            const nnBadge = document.getElementById('nn-badge');
            nnBadge.className = `badge ${{nnBadgeClass}}`;
            nnBadge.innerText = nnShape.is_gt ? 'GT' : 'GEN';
            document.getElementById('nn-footer-name').innerText = nnShape.name;

            // Load main 3D mesh
            if (currentMainMesh) mainScene.remove(currentMainMesh);
            currentMainMesh = createMeshObject(s);
            mainScene.add(currentMainMesh);

            // Load 1-NN 3D mesh
            if (currentNNMesh) nnScene.remove(currentNNMesh);
            currentNNMesh = createMeshObject(nnShape);
            nnScene.add(currentNNMesh);
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

        // Animation Loop with Synchronized Cameras
        function animate() {{
            requestAnimationFrame(animate);
            mainControls.update();

            // Synchronize 1-NN camera rotation, orientation and distance with main camera
            nnCamera.position.copy(mainCamera.position);
            nnCamera.quaternion.copy(mainCamera.quaternion);
            nnCamera.zoom = mainCamera.zoom;
            nnCamera.updateProjectionMatrix();

            mainRenderer.render(mainScene, mainCamera);
            nnRenderer.render(nnScene, nnCamera);
        }}
        animate();

        // Handle Window & Canvas Resizing
        window.addEventListener('resize', () => {{
            mainCamera.aspect = holder.clientWidth / holder.clientHeight;
            mainCamera.updateProjectionMatrix();
            mainRenderer.setSize(holder.clientWidth, holder.clientHeight);

            nnCamera.aspect = nnHolder.clientWidth / nnHolder.clientHeight;
            nnCamera.updateProjectionMatrix();
            nnRenderer.setSize(nnHolder.clientWidth, nnHolder.clientHeight);
        }});
    </script>
</body>
</html>
"""
    return html_content

def main():
    parser = argparse.ArgumentParser(description="Generate Interactive Web HTML 3D Shape Space & Orbit Mesh Explorer.")
    parser.add_argument("--run_dir", type=str, required=True, help="Path to run inference output folder.")
    parser.add_argument("--gt_dir", type=str, default="../data_test/organelles/lyso", help="Path to GT directory.")
    parser.add_argument("--points", type=int, default=1500, help="Number of points per cloud for metric computation.")
    parser.add_argument("--max_gen", type=int, default=50, help="Max generated shapes to include in web explorer.")
    parser.add_argument("--output", type=str, default=None, help="Output HTML filename.")
    args = parser.parse_args()

    gt_files = sorted(glob.glob(os.path.join(args.gt_dir, "*.obj")))
    gen_files = sorted(glob.glob(os.path.join(args.run_dir, "*.obj")))[:args.max_gen]

    if not gt_files or not gen_files:
        print(f"Error: Need both GT files ({len(gt_files)}) and Gen files ({len(gen_files)}).")
        return

    print(f"Processing {len(gt_files)} GT meshes and {len(gen_files)} Generated meshes for HTML explorer...")
    shapes_data = []
    all_pcs = []

    for f in gt_files:
        pc, mesh_dict = process_mesh_and_pc(f, num_points=args.points)
        all_pcs.append(pc)
        shapes_data.append({
            "name": os.path.basename(f),
            "is_gt": True,
            "pc": pc.tolist(),
            "mesh": mesh_dict
        })

    for f in gen_files:
        pc, mesh_dict = process_mesh_and_pc(f, num_points=args.points)
        all_pcs.append(pc)
        shapes_data.append({
            "name": os.path.basename(f),
            "is_gt": False,
            "pc": pc.tolist(),
            "mesh": mesh_dict
        })

    num_gt = len(gt_files)
    num_gen = len(gen_files)
    Total = num_gt + num_gen

    print(f"Computing {Total}x{Total} pairwise Chamfer Distance matrix...")
    t0 = time.time()
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
    html_page = build_interactive_html(shapes_data, coords, nn_indices, D, run_name)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    out_name = args.output or f"shape_space_interactive_{run_name}.html"
    out_path = os.path.join(results_dir, out_name)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html_page)

    print(f"\nInteractive 3D Mesh Shape Space Web Explorer saved to:")
    print(f"  -> {out_path}")
    print(f"\nTo open, double-click '{out_path}' or open it in any web browser!")

if __name__ == '__main__':
    main()
