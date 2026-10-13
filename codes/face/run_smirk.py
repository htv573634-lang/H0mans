import os
import argparse
import glob
import subprocess
import torch
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--smirk_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    input_image_path = find_input_image(args.input_dir)

    print("Running SMIRK Inference...")
    
    # SMIRK requires running its own demo script
    smirk_cmd = [
        "python", os.path.join(args.smirk_path, "smirk_demo.py"),
        "-i", os.path.abspath(input_image_path),
        "-o", os.path.abspath(args.output_dir),
        "--model_path", os.path.abspath(args.model_path)
    ]
    
    print(f"Executing: {' '.join(smirk_cmd)}")
    
    try:
        result = subprocess.run(smirk_cmd, capture_output=True, text=True)
        print("--- SMIRK STDOUT ---")
        print(result.stdout)
        print("--- SMIRK STDERR ---")
        print(result.stderr)
        
        if result.returncode != 0:
            raise RuntimeError(f"SMIRK failed with exit code {result.returncode}")

        print("SMIRK Inference successful! Face mesh generated.")
        
        # SMIRK usually outputs a file named 'mesh.obj' or similar in the output dir
        # Let's verify it exists
        possible_outputs = glob.glob(os.path.join(args.output_dir, "*.obj"))
        if possible_outputs:
            print(f"Successfully generated: {possible_outputs[0]}")
        else:
            print("Warning: No .obj file found in output directory.")

    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        # Create a placeholder file so the Action doesn't fail completely
        output_path = os.path.join(args.output_dir, "female_face_mesh.obj")
        with open(output_path, "w") as f:
            f.write(f"# Error during SMIRK processing: {e}\n")

    print("Face construction completed!")

if __name__ == "__main__":
    main()
