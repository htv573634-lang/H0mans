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
    Clean and smooth the DECA face shell, keeping eye/mouth openings intact
    so internal organs (from GNM) can be placed inside later.
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
            print(f"Subdivision {i+1} failed: {e}")
            break

    try:
        trimesh.repair.fix_normals(mesh)
        print("Normals fixed")
    except Exception as e:
        print(f"Normal fix failed: {e}")

    mesh.apply_scale(scale_factor)
    print(f"Scaled by {scale_factor}x, extents: {mesh.extents} (mm)")

    try:
        _ = mesh.vertex_normals
        print("Vertex normals recomputed")
    except Exception as e:
        print(f"Normal recompute failed: {e}")

    mesh.export(output_path)
    print(f"SUCCESS: {output_path}")
    print(f"Final: {len(mesh.vertices)} verts, {len(mesh.faces)} faces")
    print("=" * 60)


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

    # --- RUN DECA ---
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

    raw_output = os.path.join(args.output_dir, "female_face_mesh_raw.obj")
    shutil.copy(raw_obj, raw_output)
    print(f"Saved raw DECA output: {raw_output}")

    processed_face = os.path.join(args.output_dir, "female_face_shell.obj")
    postprocess_face_mesh(raw_obj, processed_face,
                          subdivide_passes=args.subdivide,
                          scale_factor=args.scale)

    print("DECA face reconstruction completed!")


if __name__ == "__main__":
    main()
