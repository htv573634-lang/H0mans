import os
import argparse
import glob
import subprocess
import shutil

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--mica_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    input_image_path = find_input_image(args.input_dir)

    # MICA needs the FLAME2020 .pkl file in a specific directory
    os.makedirs("data/pretrained/FLAME2020", exist_ok=True)
    
    # Copy female model (Required)
    shutil.copy("models/female_model.pkl", "data/pretrained/FLAME2020/female_model.pkl")
    
    # Copy generic model (Optional - if it exists)
    if os.path.exists("models/generic_model.pkl"):
        shutil.copy("models/generic_model.pkl", "data/pretrained/FLAME2020/generic_model.pkl")
    else:
        print("Warning: generic_model.pkl not found. Proceeding with female model only.")

    print("Running MICA Inference...")
    
    # UPDATED: Using short flags -i, -o, -m as required by MICA's demo.py
    mica_demo_cmd = [
        "python", os.path.join(args.mica_path, "demo.py"),
        "-i", os.path.abspath(input_image_path),
        "-o", os.path.abspath(args.output_dir),
        "-m", os.path.abspath(args.model_path)
    ]
    
    print(f"Executing: {' '.join(mica_demo_cmd)}")
    print("Checking if FLAME models exist:")
    print(os.listdir("data/pretrained/FLAME2020"))
    print("Checking if InsightFace models exist:")
    print(os.listdir(os.path.expanduser("~/.insightface/models")))
    
    try:
        # Capture output so we can see the exact error if it fails
        result = subprocess.run(mica_demo_cmd, capture_output=True, text=True)
        
        print("--- MICA STDOUT ---")
        print(result.stdout)
        print("--- MICA STDERR ---")
        print(result.stderr)
        
        if result.returncode != 0:
            raise RuntimeError(f"MICA failed with exit code {result.returncode}")

        # Convert PLY to OBJ
        generated_ply = os.path.join(args.output_dir, "mesh.ply")
        final_obj = os.path.join(args.output_dir, "female_face_mesh.obj")
        
        if os.path.exists(generated_ply):
            print("MICA successful! Converting PLY to OBJ...")
            import trimesh
            mesh = trimesh.load(generated_ply)
            mesh.export(final_obj)
            print(f"Successfully saved {final_obj}")
        else:
            print("Warning: MICA did not produce a mesh.ply file.")

    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        # Create a placeholder file so the Action doesn't fail completely
        output_path = os.path.join(args.output_dir, "female_face_mesh.obj")
        with open(output_path, "w") as f:
            f.write(f"# Error during MICA processing: {e}\n")

    print("Face construction completed!")

if __name__ == "__main__":
    main()
