import os
import argparse
import glob
import subprocess
import shutil
import trimesh
import numpy as np


def find_input_image(input_dir):
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_files = []
    for ext in extensions:
        image_files.extend(glob.glob(os.path.join(input_dir, ext)))
        image_files.extend(glob.glob(os.path.join(input_dir, ext.upper())))
    
    if not image_files:
        raise FileNotFoundError(f"No image files found in {input_dir}.")
    
    image_files.sort()
    selected_image = image_files[0]
    print(f"Found image: {selected_image}")
    return selected_image


def postprocess_face_mesh(input_path, output_path, subdivide_passes=1, scale_factor=1000.0):
    """
    Clean and smooth the DECA face shell, but keep eye/mouth openings.
    """
    print("=" * 60)
    print("POST-PROCESSING FACE MESH")
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print("=" * 60)
    
    mesh = trimesh.load(input_path, process=False, force='mesh')
    print(f"Loaded:   {len(mesh.vertices)} verts, {len(mesh.faces)} faces")
    
    mesh.merge_vertices(merge_tex=True, merge_norm=True)
    print(f"Merged:   {len(mesh.vertices)} verts, {len(mesh.faces)} faces")
    
    try:
        mesh.update_faces(mesh.nondegenerate_faces())
        mesh.update_faces(mesh.unique_faces())
        mesh.remove_unreferenced_vertices()
        print(f"Cleaned:  {len(mesh.vertices)} verts, {len(mesh.faces)} faces")
    except Exception as e:
        print(f"Cleanup skipped: {e}")
    
    for i in range(subdivide_passes):
        try:
            mesh = mesh.subdivide()
            print(f"Subdiv{i+1}: {len(mesh.vertices)} verts, {len(mesh.faces)} faces")
        except Exception as e:
            print(f"Subdivision {i+1} failed: {e}, stopping")
            break
    
    try:
        trimesh.repair.fix_normals(mesh)
        print("Normals fixed")
    except Exception as e:
        print(f"Normal fix failed: {e}")
    
    print(f"Scaling by {scale_factor}x")
    mesh.apply_scale(scale_factor)
    print(f"New extents: {mesh.extents} (mm)")
    
    try:
        _ = mesh.vertex_normals
        print("Vertex normals recomputed")
    except Exception as e:
        print(f"Normal recompute failed: {e}")
    
    print(f"Final area: {mesh.area} mm^2")
    print(f"Watertight: {mesh.is_watertight}")
    
    mesh.export(output_path)
    print(f"SUCCESS: {output_path}")
    print("=" * 60)


def generate_gnm_organs(output_dir, scale_factor=1000.0):
    """
    Generate eyeballs, teeth, and tongue using GNM Head via body-models.
    """
    print("=" * 60)
    print("GENERATING INTERNAL ORGANS WITH GNM HEAD")
    print("=" * 60)
    
    try:
        from body_models.gnm.numpy import GNM
    except ImportError:
        print("body-models not installed. Trying pip install...")
        subprocess.run(["pip", "install", "body-models[torch]"], check=True)
        from body_models.gnm.numpy import GNM
    
    print("Loading GNM Head model (weights download on first run)...")
    gnm = GNM()
    
    params = gnm.get_rest_pose(batch_dims=(1,))
    print(f"GNM parameters: {list(params.keys())}")
    
    vertices = gnm.forward_vertices(**params)
    faces = gnm.triangles
    
    print(f"Full GNM mesh: {len(vertices)} verts, {len(faces)} faces")
    
    # Extract vertex groups for internal parts
    if hasattr(gnm, 'vertex_group_names'):
        print(f"Available vertex groups: {gnm.vertex_group_names}")
    
    # Build a combined scene
    scene = trimesh.Scene()
    
    # Helper to extract a submesh by vertex group name
    def extract_part(group_names, part_name):
        try:
            if hasattr(gnm, 'vertex_group_indices'):
                indices = gnm.vertex_group_indices(*group_names)
                if len(indices) == 0:
                    print(f"  {part_name}: no vertices found")
                    return None
                
                # Get faces that use only these vertices
                mask = np.zeros(len(vertices), dtype=bool)
                mask[indices] = True
                face_mask = mask[faces].all(axis=1)
                
                if not face_mask.any():
                    print(f"  {part_name}: no faces found")
                    return None
                
                sub_faces = faces[face_mask]
                # Remap vertex indices
                unique_verts = np.unique(sub_faces)
                remap = {old: new for new, old in enumerate(unique_verts)}
                new_faces = np.vectorize(remap.get)(sub_faces)
                sub_verts = vertices[unique_verts]
                
                part_mesh = trimesh.Trimesh(vertices=sub_verts, faces=new_faces, process=False)
                part_mesh.apply_scale(scale_factor)
                print(f"  {part_name}: {len(sub_verts)} verts, {len(new_faces)} faces")
                return part_mesh
            else:
                print(f"  {part_name}: vertex_group_indices not available")
                return None
        except Exception as e:
            print(f"  {part_name}: failed - {e}")
            return None
    
    # Try to extract each internal part
    eyeballs = extract_part(['eye_interiors', 'eye_exteriors'], 'eyeballs')
    teeth = extract_part(['upper_teeth_and_gums', 'lower_teeth_and_gums'], 'teeth')
    tongue = extract_part(['tongue'], 'tongue')
    
    # Save individual parts
    if eyeballs is not None:
        eyeballs.export(os.path.join(output_dir, "eyeballs.obj"))
        scene.add_geometry(eyeballs, node_name="eyeballs")
    if teeth is not None:
        teeth.export(os.path.join(output_dir, "teeth.obj"))
        scene.add_geometry(teeth, node_name="teeth")
    if tongue is not None:
        tongue.export(os.path.join(output_dir, "tongue.obj"))
        scene.add_geometry(tongue, node_name="tongue")
    
    # Save combined organs scene
    if scene.geometry:
        scene.export(os.path.join(output_dir, "internal_organs.obj"))
        print(f"Saved combined organs: {len(scene.geometry)} parts")
    else:
        print("No organs extracted - saving full GNM head as fallback")
        full_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        full_mesh.apply_scale(scale_factor)
        full_mesh.export(os.path.join(output_dir, "gnm_full_head.obj"))
    
    print("=" * 60)
    return scene


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--deca_path", type=str, required=True)
    parser.add_argument("--subdivide", type=int, default=1)
    parser.add_argument("--scale", type=float, default=1000.0)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    input_image_path = find_input_image(args.input_dir)

    # --- STEP 1: RUN DECA FOR FACE SHELL ---
    print("Running DECA Inference...")
    
    deca_cmd = [
        "python", "demos/demo_reconstruct.py",
        "-i", os.path.abspath(input_image_path),
        "--saveObj", "True",
        "--saveDepth", "True",
        "--useTex", "True",
        "--rasterizer_type", "pytorch3d"
    ]
    
    result = subprocess.run(
        deca_cmd, cwd=args.deca_path, capture_output=True, text=True
    )
    print("--- DECA STDOUT ---")
    print(result.stdout)
    print("--- DECA STDERR ---")
    print(result.stderr)
    if result.returncode != 0:
        raise RuntimeError("DECA failed")
    
    base_name = os.path.splitext(os.path.basename(input_image_path))[0]
    obj_files = glob.glob(
        os.path.join(args.deca_path, "**", f"{base_name}*", "*.obj"),
        recursive=True
    )
    if not obj_files:
        raise FileNotFoundError("No OBJ file found in DECA output")
    raw_obj = obj_files[0]
    
    # Save raw DECA output
    raw_output = os.path.join(args.output_dir, "female_face_mesh_raw.obj")
    shutil.copy(raw_obj, raw_output)
    print(f"Saved raw DECA output: {raw_output}")
    
    # Post-process face shell (keep openings for organs)
    processed_face = os.path.join(args.output_dir, "female_face_shell.obj")
    postprocess_face_mesh(raw_obj, processed_face, subdivide_passes=args.subdivide, scale_factor=args.scale)

    # --- STEP 2: GENERATE INTERNAL ORGANS WITH GNM ---
    generate_gnm_organs(args.output_dir, scale_factor=args.scale)

    print("Face + organs pipeline completed!")


if __name__ == "__main__":
    main()
