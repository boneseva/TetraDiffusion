#!/usr/bin/env python3
"""
plot_shape_space_html.py — Interactive Web HTML 3D Shape Space & Orbit Mesh Explorer.

Generates a standalone, self-contained interactive HTML web page with:
  1. 2D MDS Metric Space Map (Plotly.js): Click any shape dot to inspect it.
  2. Interactive 3D WebGL Orbit Mesh Viewer (Three.js): Soft matte shaded 3D surface meshes
     (GT vs. Generated organelles) with realistic diffuse lighting and double-sided rendering.
  3. Synchronized & Minimizable 1-NN Secondary 3D Viewer: A floating inset 3D viewer showing the selected
     shape's 1-NN nearest neighbor with synchronized camera rotation, accurate color coding (GT vs. Gen),
     and a collapsible interface.
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

    <!-- Google Fonts -->
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">

    <!-- Load Plotly & Three.js CDN -->
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>

    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background: #090d16;
            color: #f1f5f9;
            height: 100vh;
            display: flex;
            flex-direction: column;
            overflow: hidden;
            -webkit-font-smoothing: antialiased;
        }}
        header {{
            background: #0f172a;
            padding: 14px 28px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.08);
            display: flex;
            justify-content: space-between;
            align-items: center;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
            z-index: 50;
        }}
        header h1 {{
            font-size: 18px;
            font-weight: 700;
            background: linear-gradient(135deg, #38bdf8 0%, #818cf8 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        header .run-badge {{
            font-size: 12px;
            color: #94a3b8;
            background: rgba(30, 41, 59, 0.8);
            padding: 5px 12px;
            border-radius: 20px;
            border: 1px solid rgba(255, 255, 255, 0.08);
        }}
        header .run-badge strong {{ color: #38bdf8; }}

        .container {{
            flex: 1;
            display: flex;
            height: calc(100vh - 57px);
        }}
        #plot-container {{
            flex: 1.15;
            background: #090d16;
            border-right: 1px solid rgba(255, 255, 255, 0.08);
            position: relative;
        }}
        #viewer-container {{
            flex: 0.85;
            background: #030712;
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

        /* Modern Glassmorphic Info Panel */
        .info-panel {{
            position: absolute;
            top: 20px;
            left: 20px;
            background: rgba(15, 23, 42, 0.82);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            padding: 16px 20px;
            border-radius: 14px;
            border: 1px solid rgba(255, 255, 255, 0.12);
            font-size: 13px;
            z-index: 10;
            pointer-events: none;
            max-width: 380px;
            box-shadow: 0 12px 30px rgba(0, 0, 0, 0.4);
        }}
        .info-panel h3 {{ font-size: 15px; font-weight: 600; margin-bottom: 8px; color: #f8fafc; display: flex; align-items: center; flex-wrap: wrap; gap: 6px; }}
        .info-panel p {{ color: #94a3b8; line-height: 1.5; font-size: 12.5px; }}

        .badge {{
            display: inline-flex;
            align-items: center;
            padding: 3px 9px;
            border-radius: 6px;
            font-size: 10px;
            font-weight: 700;
            letter-spacing: 0.5px;
            text-transform: uppercase;
        }}
        .badge-gt {{ background: rgba(59, 130, 246, 0.15); color: #60a5fa; border: 1px solid rgba(59, 130, 246, 0.35); }}
        .badge-gen {{ background: rgba(239, 68, 68, 0.15); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.35); }}

        .controls-hint {{
            position: absolute;
            bottom: 20px;
            left: 20px;
            background: rgba(15, 23, 42, 0.75);
            backdrop-filter: blur(12px);
            padding: 8px 14px;
            border-radius: 8px;
            border: 1px solid rgba(255, 255, 255, 0.08);
            font-size: 11.5px;
            color: #94a3b8;
            z-index: 10;
            pointer-events: none;
        }}
        .controls-hint strong {{ color: #cbd5e1; }}

        /* Collapsible & Floating 1-NN Secondary 3D Viewer Inset */
        #nn-card {{
            position: absolute;
            bottom: 20px;
            right: 20px;
            width: 280px;
            height: 250px;
            background: rgba(15, 23, 42, 0.85);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid rgba(255, 255, 255, 0.12);
            border-radius: 14px;
            display: flex;
            flex-direction: column;
            overflow: hidden;
            box-shadow: 0 20px 35px -5px rgba(0, 0, 0, 0.6);
            z-index: 20;
            transition: height 0.3s cubic-bezier(0.4, 0, 0.2, 1), width 0.3s cubic-bezier(0.4, 0, 0.2, 1), background-color 0.2s ease;
        }}

        #nn-card.collapsed {{
            height: 38px;
            width: 240px;
            background: rgba(15, 23, 42, 0.92);
        }}

        #nn-card.collapsed .nn-card-body {{
            display: none;
        }}

        .nn-card-header {{
            padding: 8px 12px;
            background: rgba(30, 41, 59, 0.85);
            border-bottom: 1px solid rgba(255, 255, 255, 0.08);
            font-size: 12px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            user-select: none;
        }}
        .nn-header-left {{
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        .nn-card-header span.title {{ font-weight: 600; color: #cbd5e1; font-size: 11.5px; }}

        .minimize-btn {{
            background: rgba(255, 255, 255, 0.08);
            border: 1px solid rgba(255, 255, 255, 0.15);
            color: #94a3b8;
            width: 20px;
            height: 20px;
            border-radius: 5px;
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            font-size: 13px;
            font-weight: 700;
            line-height: 1;
            transition: all 0.15s ease;
        }}

        .minimize-btn:hover {{
            background: rgba(255, 255, 255, 0.2);
            color: #f8fafc;
            border-color: rgba(255, 255, 255, 0.3);
        }}

        .nn-card-body {{
            flex: 1;
            position: relative;
            display: flex;
            flex-direction: column;
            width: 100%;
            height: calc(100% - 38px);
        }}

        #nn-canvas-holder {{
            flex: 1;
            position: relative;
            width: 100%;
            height: 100%;
            background: #030712;
        }}
        .nn-footer-label {{
            position: absolute;
            bottom: 8px;
            left: 8px;
            right: 8px;
            font-size: 11px;
            color: #e2e8f0;
            background: rgba(15, 23, 42, 0.75);
            backdrop-filter: blur(8px);
            padding: 4px 10px;
            border-radius: 6px;
            border: 1px solid rgba(255, 255, 255, 0.08);
            pointer-events: none;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }}
    </style>
</head>
<body>
    <header>
        <h1>💎 3D Organelle Shape Space Explorer</h1>
        <div class="run-badge">Run: <strong>{run_name}</strong></div>
    </header>

    <div class="container">
        <!-- 2D Metric Space Plot -->
        <div id="plot-container"></div>

        <!-- 3D Orbit Viewer -->
        <div id="viewer-container">
            <div class="info-panel" id="info-panel">
                <h3 id="shape-name">Select a shape on the left map</h3>
                <p id="shape-meta">Click any blue (GT) or red (Generated) point to view 3D mesh geometry.</p>
            </div>
            
            <div class="controls-hint">💡 <strong>3D Orbit:</strong> Left-click + Drag to rotate | Right-click to pan | Scroll to zoom</div>
            
            <!-- Main Three.js Canvas Holder -->
            <div id="canvas-holder"></div>

            <!-- Inset 1-NN Secondary 3D Viewer (Collapsible) -->
            <div id="nn-card">
                <div class="nn-card-header">
                    <div class="nn-header-left">
                        <button id="nn-toggle-btn" class="minimize-btn" title="Minimize/Expand 1-NN Viewer">−</button>
                        <span class="title">1-NN Match</span>
                    </div>
                    <span id="nn-badge" class="badge"></span>
                </div>
                <div class="nn-card-body" id="nn-card-body">
                    <div id="nn-canvas-holder"></div>
                    <div class="nn-footer-label" id="nn-footer-name">No shape selected</div>
                </div>
            </div>
        </div>
    </div>

    <script>
        const shapes = {shapes_json};
        const coords = {coords_json};
        const nnIndices = {nn_json};
        const distances = {distances_json};

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
            title: {{ text: 'MDS 2D Shape Metric Space (Chamfer Distance)', font: {{ color: '#f1f5f9', size: 14, family: 'Inter' }} }},
            paper_bgcolor: '#090d16',
            plot_bgcolor: '#0f172a',
            xaxis: {{ gridcolor: 'rgba(51, 65, 85, 0.4)', zerolinecolor: 'rgba(71, 85, 105, 0.5)', tickfont: {{ color: '#94a3b8', family: 'Inter' }} }},
            yaxis: {{ gridcolor: 'rgba(51, 65, 85, 0.4)', zerolinecolor: 'rgba(71, 85, 105, 0.5)', tickfont: {{ color: '#94a3b8', family: 'Inter' }} }},
            legend: {{ font: {{ color: '#e2e8f0', family: 'Inter' }}, bgcolor: 'rgba(15, 23, 42, 0.8)' }},
            margin: {{ l: 50, r: 20, t: 50, b: 50 }}
        }};

        Plotly.newPlot('plot-container', [linesTrace, gtTrace, genTrace], layout, {{ responsive: true }});

        // Setup Main Three.js 3D Viewer
        const holder = document.getElementById('canvas-holder');
        const mainScene = new THREE.Scene();
        mainScene.background = new THREE.Color(0x030712);

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
        nnScene.background = new THREE.Color(0x030712);

        const nnCamera = new THREE.PerspectiveCamera(45, nnHolder.clientWidth / nnHolder.clientHeight, 0.1, 100);
        nnCamera.position.set(0, 0, 2.5);

        const nnRenderer = new THREE.WebGLRenderer({{ antialias: true }});
        nnRenderer.setSize(nnHolder.clientWidth, nnHolder.clientHeight);
        nnRenderer.setPixelRatio(window.devicePixelRatio);
        nnHolder.appendChild(nnRenderer.domElement);

        // Soft, Non-Shiny Matte Lighting Setup
        function setupLighting(scene) {{
            // Soft Hemispheric Skylight (White sky, deep slate ground)
            const hemiLight = new THREE.HemisphereLight(0xf8fafc, 0x0f172a, 0.75);
            scene.add(hemiLight);

            // Key Light (Soft main light from top-right)
            const keyLight = new THREE.DirectionalLight(0xffffff, 0.75);
            keyLight.position.set(2, 3, 2);
            scene.add(keyLight);

            // Fill Light (Cool blue tint from left)
            const fillLight = new THREE.DirectionalLight(0x60a5fa, 0.35);
            fillLight.position.set(-2, 1, -2);
            scene.add(fillLight);

            // Ambient Fill (prevents dark shadow pitch)
            const ambientLight = new THREE.AmbientLight(0xffffff, 0.25);
            scene.add(ambientLight);
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

                // Vibrant Blue for GT, Vibrant Coral Red for Generated
                const colorHex = shapeData.is_gt ? 0x3b82f6 : 0xef4444;

                // Soft Matte Standard Material (High roughness -> no shiny glare!)
                const material = new THREE.MeshStandardMaterial({{
                    color: colorHex,
                    roughness: 0.55,
                    metalness: 0.10,
                    side: THREE.DoubleSide
                }});

                object3D = new THREE.Mesh(geometry, material);
            }} else {{
                // Fallback to Point Cloud if no faces exist
                const positions = new Float32Array(shapeData.pc.flat());
                geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));

                const colorHex = shapeData.is_gt ? 0x3b82f6 : 0xef4444;
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

        // Minimize / Expand 1-NN Secondary Card Toggle
        const toggleBtn = document.getElementById('nn-toggle-btn');
        const nnCard = document.getElementById('nn-card');

        toggleBtn.addEventListener('click', (e) => {{
            e.stopPropagation();
            nnCard.classList.toggle('collapsed');
            if (nnCard.classList.contains('collapsed')) {{
                toggleBtn.innerText = '+';
                toggleBtn.title = 'Expand 1-NN Viewer';
            }} else {{
                toggleBtn.innerText = '−';
                toggleBtn.title = 'Minimize 1-NN Viewer';
                setTimeout(() => {{
                    nnCamera.aspect = nnHolder.clientWidth / nnHolder.clientHeight;
                    nnCamera.updateProjectionMatrix();
                    nnRenderer.setSize(nnHolder.clientWidth, nnHolder.clientHeight);
                }}, 60);
            }}
        }});

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
            if (!nnCard.classList.contains('collapsed')) {{
                nnRenderer.render(nnScene, nnCamera);
            }}
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
