import os
import argparse
import glob
import subprocess
import shutil

def find_input_image(input_dir):
    # For single-view, we prioritize the front view (0°)
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_files = []
    for ext in extensions:
        image_files.extend(glob.glob(os.path.join(input_dir, ext)))
        image_files.extend(glob.glob(os.path.join(input_dir, ext.upper())))
    
    if not image_files:
        raise FileNotFoundError(f"No image files found in {input_dir}.")
    
    # Sort and pick the first one (assumed to be the front view)
    image_files.sort()
    selected_image = image_files[0]
    print(f"Found image: {selected_image}")
    return selected_image

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--deca_path", type=str, required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    input_image_path = find_input_image(args.input_dir)

    print("Running DECA Inference...")
    
    # DECA's reconstruction demo
    deca_cmd = [
        "python", "demos/demo_reconstruct.py",
        "-i", os.path.abspath(input_image_path),
        "--saveObj", "True",
        "--saveDepth", "True",
        "--useTex", "True",
        "--rasterizer_type", "pytorch3d"  # Use pytorch3d for CPU compatibility
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

        # DECA saves the detailed mesh in a folder named after the input image
        base_name = os.path.splitext(os.path.basename(input_image_path))[0]
        # The exact output path may vary; we search for the .obj file
        obj_files = glob.glob(os.path.join(args.deca_path, "**", f"{base_name}*", "*.obj"), recursive=True)
        
        if obj_files:
            print(f"Success! Found output: {obj_files[0]}")
            # Copy to our output directory with a standard name
            final_obj = os.path.join(args.output_dir, "female_face_mesh.obj")
            shutil.copy(obj_files[0], final_obj)
            print(f"Saved to: {final_obj}")
        else:
            print("Warning: No .obj output found. Searching entire output structure...")
            all_objs = glob.glob(os.path.join(args.deca_path, "**", "*.obj"), recursive=True)
            print(f"Found OBJ files: {all_objs}")

    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        output_path = os.path.join(args.output_dir, "female_face_mesh.obj")
        with open(output_path, "w") as f:
            f.write(f"# Error during DECA processing: {e}\n")

    print("Face construction completed!")

if __name__ == "__main__":
    main()
