# Copyright (c) 2020-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved. 
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction, 
# disclosure or distribution of this material and related documentation 
# without an express license agreement from NVIDIA CORPORATION or 
# its affiliates is strictly prohibited.

import os
import torch

from . import texture
from . import mesh
from . import material

######################################################################################
# Utility functions
######################################################################################

def _find_mat(materials, name):
    for mat in materials:
        if mat['name'] == name:
            return mat
    return materials[0] # Materials 0 is the default

######################################################################################
# Create mesh object from objfile
######################################################################################

def load_obj(filename, clear_ks=True, mtl_override=None, scale=0.95):
    obj_path = os.path.dirname(filename)

    # Read entire file
    with open(filename, 'r') as f:
        lines = f.readlines()

    # Load materials
    all_materials = [
        {
            'name' : '_default_mat',
            'bsdf' : 'pbr',
            'kd'   : texture.Texture2D(torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32, device='cuda')),
            'ks'   : texture.Texture2D(torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device='cuda'))
        }
    ]
    if mtl_override is None:
        for line in lines:
            parts = line.split()
            if len(parts) == 0:
                continue
            if parts[0] == 'mtllib':
                if len(parts) < 2:
                    print(f"Warning: malformed mtllib line in {filename}: {line.strip()}")
                    continue
                mtl_file = os.path.join(obj_path, parts[1])
                if os.path.exists(mtl_file):
                    try:
                        all_materials += material.load_mtl(mtl_file, clear_ks)
                    except Exception as e:
                        print(f"Warning: failed to load mtl {mtl_file}: {e}")
                else:
                    print(f"Warning: mtllib referenced but not found: {mtl_file}; using default material")
    else:
        if os.path.exists(mtl_override):
            try:
                all_materials += material.load_mtl(mtl_override)
            except Exception as e:
                print(f"Warning: failed to load mtl override {mtl_override}: {e}")
        else:
            print(f"Warning: mtl_override provided but file does not exist: {mtl_override}; using default material")
    # load vertices
    vertices, texcoords, normals  = [], [], []
    for line in lines:
        if len(line.split()) == 0:
            continue
        
        prefix = line.split()[0].lower()
        if prefix == 'v':
            vertices.append([float(v) for v in line.split()[1:]])
        elif prefix == 'vt':
            val = [abs(float(v)) % 1 if float(v) != 1.0 else 1.0 for v in line.split()[1:]]
            texcoords.append([val[0], 1.0 - val[1]])
        elif prefix == 'vn':
            normals.append([float(v) for v in line.split()[1:]])
    # load faces
    # Start with the default material so faces before any 'usemtl' use it
    activeMatIdx = 0
    used_materials = [all_materials[0]]
    faces, tfaces, nfaces, mfaces = [], [], [], []
    for line in lines:
        if len(line.split()) == 0:
            continue

        prefix = line.split()[0].lower()
        if prefix == 'usemtl': # Track used materials
            parts = line.split()
            if len(parts) < 2:
                print(f"Warning: malformed usemtl line in {filename}: {line.strip()}")
                continue
            mat = _find_mat(all_materials, parts[1])
            if not mat in used_materials:
                used_materials.append(mat)
            activeMatIdx = used_materials.index(mat)
        elif prefix == 'f': # Parse face
            vs = line.split()[1:]
            nv = len(vs)

            def parse_vertex(vstr):
                parts = vstr.split('/')
                # v, v/vt, v//vn, v/vt/vn
                v_idx = int(parts[0]) - 1 if parts[0] != '' else -1
                t_idx = int(parts[1]) - 1 if len(parts) > 1 and parts[1] != '' else -1
                n_idx = int(parts[2]) - 1 if len(parts) > 2 and parts[2] != '' else -1
                return v_idx, t_idx, n_idx

            v0, t0, n0 = parse_vertex(vs[0])
            for i in range(nv - 2): # Triangulate polygons
                v1, t1, n1 = parse_vertex(vs[i + 1])
                v2, t2, n2 = parse_vertex(vs[i + 2])

                # Validate indices against available arrays (avoid out-of-range errors)
                valid = True
                if v0 < 0 or v1 < 0 or v2 < 0 or v0 >= len(vertices) or v1 >= len(vertices) or v2 >= len(vertices):
                    valid = False
                if t0 != -1 or t1 != -1 or t2 != -1:
                    if len(texcoords) == 0:
                        valid = False
                    else:
                        if (t0 != -1 and (t0 < 0 or t0 >= len(texcoords))) or (t1 != -1 and (t1 < 0 or t1 >= len(texcoords))) or (t2 != -1 and (t2 < 0 or t2 >= len(texcoords))):
                            valid = False
                if n0 != -1 or n1 != -1 or n2 != -1:
                    if len(normals) == 0:
                        valid = False
                    else:
                        if (n0 != -1 and (n0 < 0 or n0 >= len(normals))) or (n1 != -1 and (n1 < 0 or n1 >= len(normals))) or (n2 != -1 and (n2 < 0 or n2 >= len(normals))):
                            valid = False

                if not valid:
                    print(f"Warning: skipping malformed face in {filename}: {vs[0]} {vs[i+1]} {vs[i+2]}")
                    continue

                mfaces.append(activeMatIdx)
                faces.append([v0, v1, v2])
                tfaces.append([t0, t1, t2])
                nfaces.append([n0, n1, n2])
    assert len(tfaces) == len(faces) and len(nfaces) == len (faces)

    # Create an "uber" material by combining all textures into a larger texture
    if len(used_materials) > 1:
        uber_material, texcoords, tfaces = material.merge_materials(used_materials, texcoords, tfaces, mfaces)
    else:
        uber_material = used_materials[0]

    vertices = torch.tensor(vertices, dtype=torch.float32, device='cuda')
    texcoords = torch.tensor(texcoords, dtype=torch.float32, device='cuda') if len(texcoords) > 0 else None
    normals = torch.tensor(normals, dtype=torch.float32, device='cuda') if len(normals) > 0 else None
    
    faces = torch.tensor(faces, dtype=torch.int64, device='cuda')
    tfaces = torch.tensor(tfaces, dtype=torch.int64, device='cuda') if texcoords is not None else None
    nfaces = torch.tensor(nfaces, dtype=torch.int64, device='cuda') if normals is not None else None
    current_mesh = mesh.Mesh(vertices, faces, normals, nfaces, texcoords, tfaces, material=uber_material)
    return mesh._center(current_mesh, scale=scale)

######################################################################################
# Save mesh object to objfile
######################################################################################

def write_obj(folder, mesh, save_material=True):
    obj_file = os.path.join(folder, 'mesh.obj')
    print("Writing mesh: ", obj_file)
    with open(obj_file, "w") as f:
        f.write("mtllib mesh.mtl\n")
        f.write("g default\n")

        v_pos = mesh.v_pos.detach().cpu().numpy() if mesh.v_pos is not None else None
        v_nrm = mesh.v_nrm.detach().cpu().numpy() if mesh.v_nrm is not None else None
        v_tex = mesh.v_tex.detach().cpu().numpy() if mesh.v_tex is not None else None

        t_pos_idx = mesh.t_pos_idx.detach().cpu().numpy() if mesh.t_pos_idx is not None else None
        t_nrm_idx = mesh.t_nrm_idx.detach().cpu().numpy() if mesh.t_nrm_idx is not None else None
        t_tex_idx = mesh.t_tex_idx.detach().cpu().numpy() if mesh.t_tex_idx is not None else None

        print("    writing %d vertices" % len(v_pos))
        for v in v_pos:
            f.write('v {} {} {} \n'.format(v[0], v[1], v[2]))
       
        if v_tex is not None:
            print("    writing %d texcoords" % len(v_tex))
            assert(len(t_pos_idx) == len(t_tex_idx))
            for v in v_tex:
                f.write('vt {} {} \n'.format(v[0], 1.0 - v[1]))

        if v_nrm is not None:
            print("    writing %d normals" % len(v_nrm))
            assert(len(t_pos_idx) == len(t_nrm_idx))
            for v in v_nrm:
                f.write('vn {} {} {}\n'.format(v[0], v[1], v[2]))

        # faces
        f.write("s 1 \n")
        f.write("g pMesh1\n")
        f.write("usemtl defaultMat\n")

        # Write faces
        print("    writing %d faces" % len(t_pos_idx))
        for i in range(len(t_pos_idx)):
            f.write("f ")
            for j in range(3):
                f.write(' %s/%s/%s' % (str(t_pos_idx[i][j]+1), '' if v_tex is None else str(t_tex_idx[i][j]+1), '' if v_nrm is None else str(t_nrm_idx[i][j]+1)))
            f.write("\n")

    if save_material:
        mtl_file = os.path.join(folder, 'mesh.mtl')
        print("Writing material: ", mtl_file)
        material.save_mtl(mtl_file, mesh.material)

    print("Done exporting mesh")
