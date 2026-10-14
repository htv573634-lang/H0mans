import os
import sys
import argparse
import trimesh
import numpy as np


def generate_gnm_organs(output_dir, scale_factor=1000.0):
    """Generate eyeballs, teeth, and tongue using GNM Head via body-models."""
    print("=" * 60)
    print("GENERATING INTERNAL ORGANS WITH GNM HEAD")
    print("=" * 60)

    from body_models.gnm.numpy import GNM

    print("Loading GNM Head model (weights download on first run)...")
    model = GNM()
    print(f"GNM loaded. NUM_SHAPE_COEFFS={model.NUM_SHAPE_COEFFS}, "
          f"NUM_EXPR_COEFFS={model.NUM_EXPR_COEFFS}, "
          f"NUM_JOINTS={model.NUM_JOINTS}")

    params = model.get_rest_pose(batch_dims=(1,))
    print(f"Rest pose params: {list(params.keys())}")

    vertices = model.forward_vertices(**params)
    if vertices.ndim == 3:
        vertices = vertices[0]
    vertices = np.asarray(vertices)

    faces = np.asarray(model.triangles) if hasattr(model, 'triangles') else None
    if faces is None:
        for attr in ['faces', 'triangles', 'F']:
            if hasattr(model, attr):
                faces = np.asarray(getattr(model, attr))
                break

    if faces is None:
        print("Could not find face topology in GNM model")
        return

    print(f"GNM full mesh: {len(vertices)} verts, {len(faces)} faces")

    # Inspect vertex groups
    print("--- Inspecting vertex groups ---")
    if hasattr(model, 'vertex_group_names'):
        names = model.vertex_group_names
        print(f"Vertex group names: {names}")
    else:
        print("No vertex_group_names attribute found")
        names = None

    scene = trimesh.Scene()
    extracted_parts = []

    candidates = {
        'eyeballs': ['eyeballs', 'eyes', 'eye_balls', 'eye_interiors',
                     'eye_exteriors', 'ocular', 'left_eye', 'right_eye'],
        'teeth': ['teeth', 'upper_teeth', 'lower_teeth', 'upper_teeth_and_gums',
                  'lower_teeth_and_gums', 'dental', 'intraoral', 'gums'],
        'tongue': ['tongue', 'oral_tongue', 'lingual'],
    }

    for part_name, candidate_names in candidates.items():
        found = False
        for cand in candidate_names:
            try:
                if hasattr(model, 'vertex_group_indices') and names:
                    if cand in names or any(cand in str(n) for n in names):
                        indices = model.vertex_group_indices(cand)
                        if indices is not None and len(indices) > 0:
                            indices = np.asarray(indices)
                            mask = np.zeros(len(vertices), dtype=bool)
                            mask[indices] = True
                            face_mask = mask[faces].all(axis=1)

                            if face_mask.any():
                                sub_faces = faces[face_mask]
                                unique_verts = np.unique(sub_faces)
                                remap = {old: new for new, old in enumerate(unique_verts)}
                                new_faces = np.vectorize(remap.get)(sub_faces)
                                sub_verts = vertices[unique_verts]

                                part_mesh = trimesh.Trimesh(
                                    vertices=sub_verts, faces=new_faces, process=False
                                )
                                part_mesh.apply_scale(scale_factor)

                                part_path = os.path.join(output_dir, f"{part_name}.obj")
                                part_mesh.export(part_path)
                                print(f"  {part_name}: {len(sub_verts)} verts, "
                                      f"{len(new_faces)} faces -> {part_path}")
                                scene.add_geometry(part_mesh, node_name=part_name)
                                extracted_parts.append(part_name)
                                found = True
                                break
            except Exception as e:
                print(f"  {cand}: failed - {e}")
                continue

        if not found:
            print(f"  {part_name}: no matching vertex group found")

    if scene.geometry:
        organs_path = os.path.join(output_dir, "internal_organs.obj")
        scene.export(organs_path)
        print(f"Saved combined organs ({len(extracted_parts)} parts): {organs_path}")
    else:
        print("No organs extracted - saving full GNM head as fallback")
        full_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        full_mesh.apply_scale(scale_factor)
        full_path = os.path.join(output_dir, "gnm_full_head.obj")
        full_mesh.export(full_path)
        print(f"Saved full GNM head: {full_path}")

    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--scale", type=float, default=1000.0)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    generate_gnm_organs(args.output_dir, scale_factor=args.scale)
