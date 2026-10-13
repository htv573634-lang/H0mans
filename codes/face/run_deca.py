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
    Load DECA raw OBJ, subdivide + smooth, recompute vertex normals, export.
    Fixes the faceted / low-poly look.
    """
    print("=" * 60)
    print("POST-PROCESSING MESH (smoothing)")
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print("=" * 60)
    
    mesh = trimesh.load(input_path, process=False, force='mesh')
    print(f"Loaded:   {len(mesh.vertices)} verts, {len(mesh.faces)} faces")
    
    # Merge duplicate vertices (DECA splits verts for UV seams)
    mesh.merge_vertices()
    print(f"Merged:   {len(mesh.vertices)} verts, {len(mesh.faces)} faces")
    
    # Remove degenerate faces
    try:
        mesh.remove_degenerate_faces()
        mesh.remove_duplicate_faces()
        print(f"Cleaned:  {len(mesh.vertices)} verts, {len(mesh.faces)} faces")
    except Exception as e:
        print(f"Cleanup skipped: {e}")
    
    # Subdivide for smoother surface
    for i in range(subdivide_passes):
        mesh = mesh.subdivide()
        print(f"Subdiv{i+1}: {len(mesh.vertices)} verts, {len(mesh.faces)} faces")
    
    # Gentle Laplacian smoothing (preserves overall shape)
    try:
        trimesh.smoothing.filter_laplacian(mesh, lamb=0.3, iterations=2)
        print("Laplacian smoothing applied")
    except Exception as e:
        print(f"Laplacian smoothing skipped: {e}")
    
    # Recompute vertex normals -> this makes viewers show SMOOTH shading
    mesh.vertex_normals = None
    _ = mesh.vertex_normals
    print("Vertex normals recomputed (smooth shading)")
    
    # Export
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
                        help="Number of subdivision passes (1=fast, 2=smooth, 3=very dense)")
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
        
        # Save the raw mesh for reference
        raw_output = os.path.join(args.output_dir, "female_face_mesh_raw.obj")
        shutil.copy(raw_obj, raw_output)
        print(f"Saved raw mesh: {raw_output}")
        
        # Post-process: subdivide + smooth
        final_output = os.path.join(args.output_dir, "female_face_mesh.obj")
        postprocess_mesh(raw_obj, final_output, subdivide_passes=args.subdivide)

    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        output_path = os.path.join(args.output_dir, "female_face_mesh.obj")
        with open(output_path, "w") as f:
            f.write(f"# Error during DECA processing: {e}\n")

    print("Face construction completed!")


if __name__ == "__main__":
    main()
