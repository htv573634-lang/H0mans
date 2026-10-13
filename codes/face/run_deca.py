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


def postprocess_mesh(input_path, output_path, subdivide_passes=1):
    """
    Safe mesh processing:
      1. Merge duplicate vertices (with tolerance) so trimesh treats it as one surface
      2. Remove degenerate/duplicate faces
      3. Subdivide to add geometric detail
      4. Recompute vertex normals for smooth shading
    NO Laplacian smoothing (it was collapsing the mesh).
    """
    print("=" * 60)
    print("POST-PROCESSING MESH")
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print("=" * 60)
    
    mesh = trimesh.load(input_path, process=False, force='mesh')
    print(f"Loaded:   {len(mesh.vertices)} verts, {len(mesh.faces)} faces")
    print(f"Bounds:   min={mesh.bounds[0]}, max={mesh.bounds[1]}")
    print(f"Extents:  {mesh.extents}")
    
    # Merge duplicate vertices - DECA splits vertices at UV seams.
    # Use a tiny tolerance so only exact duplicates merge.
    mesh.merge_vertices(merge_tex=True, merge_norm=True)
    print(f"Merged:   {len(mesh.vertices)} verts, {len(mesh.faces)} faces")
    
    # Remove degenerate faces (zero-area triangles that break viewers)
    try:
        mesh.update_faces(mesh.nondegenerate_faces())
        mesh.update_faces(mesh.unique_faces())
        mesh.remove_unreferenced_vertices()
        print(f"Cleaned:  {len(mesh.vertices)} verts, {len(mesh.faces)} faces")
    except Exception as e:
        print(f"Cleanup skipped: {e}")
    
    # Subdivide using trimesh's loop subdivision (preserves shape better than default)
    for i in range(subdivide_passes):
        try:
            mesh = mesh.subdivide()
            print(f"Subdiv{i+1}: {len(mesh.vertices)} verts, {len(mesh.faces)} faces")
        except Exception as e:
            print(f"Subdivision {i+1} failed: {e}, stopping")
            break
    
    # Recompute vertex normals -> smooth shading in every viewer
    try:
        _ = mesh.vertex_normals
        print("Vertex normals recomputed")
    except Exception as e:
        print(f"Normal recompute failed: {e}")
    
    # Validate before export
    print(f"Final bounds: min={mesh.bounds[0]}, max={mesh.bounds[1]}")
    print(f"Final extents: {mesh.extents}")
    print(f"Final area: {mesh.area}")
    
    if mesh.area < 1e-6:
        raise RuntimeError("Mesh area is zero - mesh is degenerate, refusing to export")
    
    mesh.export(output_path)
    print(f"SUCCESS: {output_path}")
    print(f"Final:    {len(mesh.vertices)} verts, {len(mesh.faces)} faces")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--deca_path", type=str, required=True)
    parser.add_argument("--subdivide", type=int, default=1,
                        help="Number of subdivision passes (0=raw, 1=smooth, 2=very dense)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    input_image_path = find_input_image(args.input_dir)

    print("Running DECA Inference...")
    
    deca_cmd = [
        "python", "demos/demo_reconstruct.py",
        "-i", os.path.abspath(input_image_path),
        "--saveObj", "True",
        "--saveDepth", "True",
        "--useTex", "True",
        "--rasterizer_type", "pytorch3d"
    ]
    
    print(f"Executing from {args.deca_path}: {' '.join(deca_cmd)}")
    
    try:
        result = subprocess.run(
            deca_cmd,
            cwd=args.deca_path,
            capture_output=True,
            text=True
        )
        
        print("--- DECA STDOUT ---")
        print(result.stdout)
        print("--- DECA STDERR ---")
        print(result.stderr)
        
        if result.returncode != 0:
            raise RuntimeError(f"DECA failed with exit code {result.returncode}")

        # Locate DECA's raw OBJ
        base_name = os.path.splitext(os.path.basename(input_image_path))[0]
        obj_files = glob.glob(
            os.path.join(args.deca_path, "**", f"{base_name}*", "*.obj"),
            recursive=True
        )
        
        if not obj_files:
            raise FileNotFoundError("No OBJ file found in DECA output")
        
        raw_obj = obj_files[0]
        print(f"Found DECA raw output: {raw_obj}")
        
        # Save the raw mesh (always good to keep as backup)
        raw_output = os.path.join(args.output_dir, "female_face_mesh_raw.obj")
        shutil.copy(raw_obj, raw_output)
        print(f"Saved raw mesh: {raw_output}")
        
        # Post-process with safe settings
        final_output = os.path.join(args.output_dir, "female_face_mesh.obj")
        postprocess_mesh(raw_obj, final_output, subdivide_passes=args.subdivide)

    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        # Fallback: copy raw mesh as final output if processing fails
        try:
            base_name = os.path.splitext(os.path.basename(input_image_path))[0]
            obj_files = glob.glob(
                os.path.join(args.deca_path, "**", f"{base_name}*", "*.obj"),
                recursive=True
            )
            if obj_files:
                shutil.copy(obj_files[0], os.path.join(args.output_dir, "female_face_mesh.obj"))
                print("Fallback: copied raw DECA mesh as output")
        except Exception:
            pass

    print("Face construction completed!")


if __name__ == "__main__":
    main()
