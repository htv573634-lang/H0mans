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
    parser.add_argument("--ddfa_path", type=str, required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    input_image_path = find_input_image(args.input_dir)

    print("Running 3DDFA_v2 Inference...")
    
    # 3DDFA_v2's demo.py: -f for input file, -o for output dir, --onnx for CPU acceleration
    demo_cmd = [
        "python3", "demo.py",
        "-f", os.path.abspath(input_image_path),
        "-o", os.path.abspath(args.output_dir),
        "--onnx"
    ]
    
    print(f"Executing from {args.ddfa_path}: {' '.join(demo_cmd)}")
    
    try:
        result = subprocess.run(
            demo_cmd,
            cwd=args.ddfa_path,  # Run from the 3DDFA_V2 repo root
            capture_output=True,
            text=True
        )
        
        print("--- 3DDFA_v2 STDOUT ---")
        print(result.stdout)
        print("--- 3DDFA_v2 STDERR ---")
        print(result.stderr)
        
        if result.returncode != 0:
            raise RuntimeError(f"3DDFA_v2 failed with exit code {result.returncode}")

        # The demo saves the obj as <input_name>.obj in the output directory
        base_name = os.path.splitext(os.path.basename(input_image_path))[0]
        generated_obj = os.path.join(args.output_dir, f"{base_name}.obj")
        final_obj_path = os.path.join(args.output_dir, "female_face_mesh.obj")
        
        if os.path.exists(generated_obj):
            print(f"Success! Found output: {generated_obj}")
            shutil.move(generated_obj, final_obj_path)
            print(f"Renamed to: {final_obj_path}")
        else:
            print(f"Warning: Expected output not found at {generated_obj}")
            obj_files = glob.glob(os.path.join(args.output_dir, "*.obj"))
            print(f"OBJ files in output dir: {obj_files}")

    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        output_path = os.path.join(args.output_dir, "female_face_mesh.obj")
        with open(output_path, "w") as f:
            f.write(f"# Error during 3DDFA_v2 processing: {e}\n")

    print("Face construction completed!")

if __name__ == "__main__":
    main()
