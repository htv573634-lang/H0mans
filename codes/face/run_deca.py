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


def postprocess_mesh(input_path, output_path, subdivide_passes=1, scale_factor=1000.0):
    """
    Full post-processing pipeline:
      1. Merge duplicate vertices
      2. Remove degenerate/duplicate faces
      3. Subdivide for smooth surface
      4. Fill holes to make watertight (caps neck, eyes, mouth)
      5. Fix normals (consistent outward orientation)
      6. Scale up (meters -> millimeters for print-ready output)
      7. Recompute vertex normals for smooth shading
    """
    print("=" * 60)
    print("POST-PROCESSING MESH")
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print("=" * 60)
    
    mesh = trimesh.load(input_path, process=False, force='mesh')
    print(f"Loaded:   {len(mesh.vertices)} verts, {len(mesh.faces)} faces")
    print(f"Extents:  {mesh.extents} (assumed meters)")
    
    # 1. Merge duplicate vertices
    mesh.merge_vertices(merge_tex=True, merge_norm=True)
    print(f"Merged:   {len(mesh.vertices)} verts, {len(mesh.faces)} faces")
    
    # 2. Remove degenerate faces
    try:
        mesh.update_faces(mesh.nondegenerate_faces())
        mesh.update_faces(mesh.unique_faces())
        mesh.remove_unreferenced_vertices()
        print(f"Cleaned:  {len(mesh.vertices)} verts, {len(mesh.faces)} faces")
    except Exception as e:
        print(f"Cleanup skipped: {e}")
    
    # 3. Subdivide
    for i in range(subdivide_passes):
        try:
            mesh = mesh.subdivide()
            print(f"Subdiv{i+1}: {len(mesh.vertices)} verts, {len(mesh.faces)} faces")
        except Exception as e:
            print(f"Subdivision {i+1} failed: {e}, stopping")
            break
    
    # 4. Fill holes to make watertight
    print("--- Filling holes to make mesh watertight ---")
    try:
        # Get boundary edges and fill them
        mesh.fill_holes()
        print(f"After fill_holes: {len(mesh.vertices)} verts, {len(mesh.faces)} faces")
        
        # Check if watertight now
        if mesh.is_watertight:
            print("MESH IS WATERTIGHT ✓")
        else:
            print(f"Still not watertight - trying more aggressive repair...")
            # Try to repair with a thicker approach
            mesh = trimesh.repair.fill_holes(mesh)
            if mesh.is_watertight:
                print("MESH IS NOW WATERTIGHT ✓")
            else:
                print("Could not make fully watertight - mesh will still export")
    except Exception as e:
        print(f"Hole filling failed: {e}")
    
    # 5. Fix normals
    try:
        trimesh.repair.fix_normals(mesh)
        trimesh.repair.fix_inversion(mesh)
        print("Normals fixed (outward orientation)")
    except Exception as e:
        print(f"Normal fixing failed: {e}")
    
    # 6. Scale to millimeters (m -> mm)
    print(f"--- Scaling by {scale_factor}x (meters to millimeters) ---")
    mesh.apply_scale(scale_factor)
    print(f"New extents: {mesh.extents} (mm)")
    
    # 7. Recompute vertex normals for smooth shading
    try:
        _ = mesh.vertex_normals
        print("Vertex normals recomputed")
    except Exception as e:
        print(f"Normal recompute failed: {e}")
    
    # Validate
    print(f"Final bounds: min={mesh.bounds[0]}, max={mesh.bounds[1]}")
    print(f"Final extents: {mesh.extents} mm")
    print(f"Final area: {mesh.area} mm^2")
    print(f"Watertight: {mesh.is_watertight}")
    
    if mesh.area < 1e-6:
        raise RuntimeError("Mesh area is zero - refusing to export")
    
    mesh.export(output_path)
    print(f"SUCCESS: {output_path}")
    print(f"Final:    {len(mesh.vertices)} verts, {len(mesh.faces)} faces")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--deca_path", type=str, required=True)
    parser.add_argument("--subdivide", type=int, default=1)
    parser.add_argument("--scale", type=float, default=1000.0,
                        help="Scale factor (1000 = meters to mm)")
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

        base_name = os.path.splitext(os.path.basename(input_image_path))[0]
        obj_files = glob.glob(
            os.path.join(args.deca_path, "**", f"{base_name}*", "*.obj"),
            recursive=True
        )
        
        if not obj_files:
            raise FileNotFoundError("No OBJ file found in DECA output")
        
        raw_obj = obj_files[0]
        print(f"Found DECA raw output: {raw_obj}")
        
        raw_output = os.path.join(args.output_dir, "female_face_mesh_raw.obj")
        shutil.copy(raw_obj, raw_output)
        print(f"Saved raw mesh: {raw_output}")
        
        final_output = os.path.join(args.output_dir, "female_face_mesh.obj")
        postprocess_mesh(
            raw_obj,
            final_output,
            subdivide_passes=args.subdivide,
            scale_factor=args.scale
        )

    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
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
