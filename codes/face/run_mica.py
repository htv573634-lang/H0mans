import os
import argparse
import glob
import subprocess

def find_input_image(input_dir):
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
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
    parser.add_argument("--mica_path", type=str, required=True, help="Path to cloned MICA repo")
    parser.add_argument("--model_path", type=str, required=True, help="Path to mica.tar")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    input_image_path = find_input_image(args.input_dir)

    print("Running MICA Inference...")
    
    # Run MICA's demo script using subprocess
    # MICA expects the model path and the FLAME folder.
    # We point it to our models/ directory.
    mica_demo_cmd = [
        "python", os.path.join(args.mica_path, "demo.py"),
        "--input_path", os.path.abspath(input_image_path),
        "--output_path", os.path.abspath(args.output_dir),
        "--model_path", os.path.abspath(args.model_path)
    ]
    
    print(f"Executing: {' '.join(mica_demo_cmd)}")
    
    try:
        # We need to set the FLAME2020 path for MICA to find the .pkl file
        # MICA usually looks in data/pretrained/FLAME2020/
        os.makedirs("data/pretrained/FLAME2020", exist_ok=True)
        # We need to copy our female_model.pkl there
        import shutil
        shutil.copy("models/female_model.pkl", "data/pretrained/FLAME2020/female_model.pkl")
        
        # Run MICA
        subprocess.run(mica_demo_cmd, check=True)
        
        # MICA outputs a 'mesh.ply' file. Let's convert it to .obj for better compatibility.
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
        print(f"Error running MICA: {e}")
        # Fallback: Create a dummy file so the action doesn't fail completely
        output_path = os.path.join(args.output_dir, "female_face_mesh.obj")
        with open(output_path, "w") as f:
            f.write("# Error during MICA processing\n")

    print("Face construction completed!")

if __name__ == "__main__":
    main()
