import os
import argparse
import glob

def find_input_image(input_dir):
    """
    Scans the input directory for any image file with common extensions.
    Returns the path to the first valid image found.
    """
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    image_files = []
    
    for ext in extensions:
        # Look for files matching the extension, case-insensitive
        image_files.extend(glob.glob(os.path.join(input_dir, ext)))
        image_files.extend(glob.glob(os.path.join(input_dir, ext.upper())))
    
    if not image_files:
        raise FileNotFoundError(f"No image files found in {input_dir}. Please upload a face image.")
    
    # Sort files so we consistently pick the first one (e.g., if you upload 0.jpg and 90.jpg)
    image_files.sort()
    selected_image = image_files[0]
    
    print(f"Found image: {selected_image}")
    return selected_image

def main():
    parser = argparse.ArgumentParser(description="Process face images to build 3D model")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing input face images")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save output meshes")
    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Find the input image
    try:
        input_image_path = find_input_image(args.input_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    # 2. Here is where MICA code will go
    print(f"Placeholder: Loading MICA model...")
    print(f"Placeholder: Processing {input_image_path}...")
    
    # 3. Save the output
    output_path = os.path.join(args.output_dir, "female_face_mesh.obj")
    print(f"Placeholder: Saving mesh to {output_path}...")
    
    # Create a dummy file so the GitHub Action has something to upload
    with open(output_path, "w") as f:
        f.write("# Dummy face mesh output\n")
    
    print("Face construction completed successfully!")

if __name__ == "__main__":
    main()
