import os
import struct
import json
import argparse
import numpy as np
import trimesh


def write_glb_file(gltf_dict, binary_data, output_path):
    """Write a GLB file directly (bypasses pygltflib bugs)."""
    json_str = json.dumps(gltf_dict, separators=(',', ':'))
    json_bytes = json_str.encode('utf-8')
    json_pad = (4 - len(json_bytes) % 4) % 4
    json_bytes += b' ' * json_pad

    bin_pad = (4 - len(binary_data) % 4) % 4
    bin_bytes = binary_data + b'\x00' * bin_pad

    total_length = 12 + 8 + len(json_bytes) + 8 + len(bin_bytes)

    with open(output_path, 'wb') as f:
        f.write(struct.pack('<I', 0x46546C67))    # magic 'glTF'
        f.write(struct.pack('<I', 2))              # version
        f.write(struct.pack('<I', total_length))   # total length
        f.write(struct.pack('<I', len(json_bytes)))
        f.write(struct.pack('<I', 0x4E4F534A))     # 'JSON'
        f.write(json_bytes)
        f.write(struct.pack('<I', len(bin_bytes)))
        f.write(struct.pack('<I', 0x004E4942))     # 'BIN\0'
        f.write(bin_bytes)


def build_glb_with_morphs(base_verts, faces, morph_dict, timeline,
                          output_path, uv=None):
    """Build a GLB with morph targets + a keyframed weights animation."""
    morph_names = list(morph_dict.keys())
    n_morphs = len(morph_names)

    bin_parts = []
    current_offset = [0]
    buffer_views = []
    accessors = []

    def append_aligned(data_bytes):
        pad = (-current_offset[0]) % 4
        if pad:
            bin_parts.append(b'\x00' * pad)
            current_offset[0] += pad
        offset = current_offset[0]
        bin_parts.append(data_bytes)
        current_offset[0] += len(data_bytes)
        return offset

    def add_accessor(np_array, target=None, need_bounds=False):
        offset = append_aligned(np_array.tobytes())
        bv_idx = len(buffer_views)
        bv = {
            "buffer": 0,
            "byteOffset": offset,
            "byteLength": int(np_array.nbytes),
        }
        if target:
            bv["target"] = target
        buffer_views.append(bv)

        dtype_map = {
            np.dtype(np.float32): 5126,
            np.dtype(np.uint32): 5125,
            np.dtype(np.uint16): 5123,
            np.dtype(np.uint8): 5121,
        }
        comp_type = dtype_map[np_array.dtype]

        if np_array.ndim == 1:
            type_str = "SCALAR"
            count = len(np_array)
        else:
            dim = np_array.shape[1]
            type_str = {2: "VEC2", 3: "VEC3", 4: "VEC4"}[dim]
            count = np_array.shape[0]

        acc = {
            "bufferView": bv_idx,
            "componentType": comp_type,
            "count": int(count),
            "type": type_str,
        }
        if need_bounds and np_array.dtype == np.float32:
            if type_str == "VEC3":
                acc["min"] = np_array.min(axis=0).tolist()
                acc["max"] = np_array.max(axis=0).tolist()
            elif type_str == "SCALAR":
                acc["min"] = [float(np_array.min())]
                acc["max"] = [float(np_array.max())]

        acc_idx = len(accessors)
        accessors.append(acc)
        return acc_idx

    # Base mesh
    pos_acc = add_accessor(base_verts.astype(np.float32), target=34962,
                           need_bounds=True)
    idx_acc = add_accessor(faces.flatten().astype(np.uint32), target=34963)

    # Morph target deltas
    delta_accs = []
    for name in morph_names:
        delta = morph_dict[name].astype(np.float32)
        acc = add_accessor(delta, target=34962)
        delta_accs.append(acc)

    # Animation times
    times = np.array([t for t, _ in timeline], dtype=np.float32)
    time_acc = add_accessor(times, need_bounds=True)

    # Animation weights (flat)
    weights_flat = []
    for _, weight_dict in timeline:
        for name in morph_names:
            weights_flat.append(float(weight_dict.get(name, 0.0)))
    weight_acc = add_accessor(np.array(weights_flat, dtype=np.float32))

    # Attributes
    attributes = {"POSITION": pos_acc}
    if uv is not None:
        uv_acc = add_accessor(uv.astype(np.float32), target=34962)
        attributes["TEXCOORD_0"] = uv_acc

    morph_targets = [{"POSITION": delta_accs[i]} for i in range(n_morphs)]

    gltf_dict = {
        "asset": {"version": "2.0", "generator": "H0mans-GNM"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "name": "GNM_Head"}],
        "meshes": [{
            "primitives": [{
                "attributes": attributes,
                "indices": idx_acc,
                "targets": morph_targets,
            }],
            "weights": [0.0] * n_morphs,
            "extras": {"targetNames": morph_names},
        }],
        "animations": [{
            "name": "Expression_Timeline",
            "channels": [{
                "sampler": 0,
                "target": {"node": 0, "path": "weights"},
            }],
            "samplers": [{
                "input": time_acc,
                "output": weight_acc,
                "interpolation": "LINEAR",
            }],
        }],
        "buffers": [{"byteLength": 0}],
        "bufferViews": buffer_views,
        "accessors": accessors,
    }

    bin_data = b''.join(bin_parts)
    bin_data += b'\x00' * ((-len(bin_data)) % 4)
    gltf_dict["buffers"][0]["byteLength"] = len(bin_data)

    write_glb_file(gltf_dict, bin_data, output_path)

    print(f"Saved GLB: {output_path} "
          f"({len(bin_data) / 1024 / 1024:.2f} MB, "
          f"{n_morphs} morph targets, "
          f"{len(timeline)} keyframes)")


def export_static_obj(vertices, faces, output_path, scale_factor=1000.0):
    mesh = trimesh.Trimesh(vertices=vertices * scale_factor,
                           faces=faces, process=False)
    mesh.export(output_path)
    return mesh


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--scale", type=float, default=1000.0)
    parser.add_argument("--n_morphs", type=int, default=6)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("GNM HEAD → ANIMATED GLB PIPELINE")
    print("=" * 60)

    from body_models.gnm.numpy import GNM

    print("Loading GNM Head model...")
    model = GNM()
    print(f"NUM_SHAPE_COEFFS={model.NUM_SHAPE_COEFFS}, "
          f"NUM_EXPR_COEFFS={model.NUM_EXPR_COEFFS}, "
          f"NUM_JOINTS={model.NUM_JOINTS}")

    # Faces
    faces = None
    for attr in ['triangles', 'faces', 'F']:
        if hasattr(model, attr):
            faces = np.asarray(getattr(model, attr))
            print(f"Faces via '{attr}': {faces.shape}")
            break
    if faces is None:
        raise RuntimeError("Cannot find face topology")
    faces = faces.astype(np.uint32)

    # UV coordinates - try more attribute names
    uv = None
    for attr in ['uv_coordinates', 'uvs', 'UV', 'texcoords', 'uv',
                 'texcoord', 'texture_coordinates', 'uvs_coordinates']:
        if hasattr(model, attr):
            val = getattr(model, attr)
            try:
                uv = np.asarray(val)
                if uv.ndim == 2 and uv.shape[1] == 2:
                    print(f"UVs via '{attr}': shape={uv.shape}")
                    break
                else:
                    uv = None
            except Exception:
                uv = None

    if uv is None:
        print("No UV coordinates found — exporting without textures")
        print("Available model attributes (non-underscore):")
        for attr in sorted(dir(model)):
            if not attr.startswith('_'):
                try:
                    val = getattr(model, attr)
                    if hasattr(val, 'shape'):
                        print(f"  {attr}: shape={val.shape}, dtype={val.dtype}")
                    elif isinstance(val, (int, float, str, list, dict, tuple)):
                        s = str(val)
                        if len(s) < 120:
                            print(f"  {attr}: {type(val).__name__} = {s}")
                except Exception:
                    pass

    # Base mesh
    neutral_params = model.get_rest_pose(batch_dims=(1,))
    base_verts = model.forward_vertices(**neutral_params)
    if base_verts.ndim == 3:
        base_verts = base_verts[0]
    base_verts = np.asarray(base_verts, dtype=np.float32)
    print(f"Base mesh: {base_verts.shape}")

    if 'expression' not in neutral_params:
        raise RuntimeError("'expression' not found in GNM params")
    base_expr = np.asarray(neutral_params['expression'])
    n_expr_coefs = base_expr.shape[-1]
    print(f"Expression coefficients: {n_expr_coefs}")

    # Generate morph targets
    print(f"\n--- Generating {args.n_morphs} morph targets ---")
    morph_dict = {}
    for i in range(args.n_morphs):
        expr_vec = np.zeros_like(base_expr)
        expr_vec[..., i] = 3.0
        params_i = dict(neutral_params)
        params_i['expression'] = expr_vec
        verts_i = model.forward_vertices(**params_i)
        if verts_i.ndim == 3:
            verts_i = verts_i[0]
        verts_i = np.asarray(verts_i, dtype=np.float32)
        delta = verts_i - base_verts
        morph_dict[f"expr_{i}"] = delta
        mag = float(np.linalg.norm(delta, axis=1).mean())
        print(f"  expr_{i}: mean delta = {mag:.5f} m")

    # Timeline
    print(f"\n--- Building timeline ---")
    timeline = [(0.0, {}), (0.5, {})]
    for i in range(args.n_morphs):
        t0 = 0.5 + i * 2.5
        timeline.append((t0 + 0.3, {f"expr_{i}": 1.0}))
        timeline.append((t0 + 2.0, {f"expr_{i}": 1.0}))
        timeline.append((t0 + 2.5, {}))
    duration = timeline[-1][0]
    print(f"Total: {duration:.1f}s, {len(timeline)} keyframes")

    # Static OBJs
    print(f"\n--- Exporting static OBJs ---")
    neutral_path = os.path.join(args.output_dir, "gnm_neutral.obj")
    export_static_obj(base_verts, faces, neutral_path, args.scale)
    print(f"  {neutral_path}")
    for i in range(args.n_morphs):
        verts_i = base_verts + morph_dict[f"expr_{i}"]
        expr_path = os.path.join(args.output_dir, f"gnm_expr_{i}.obj")
        export_static_obj(verts_i, faces, expr_path, args.scale)
        print(f"  {expr_path}")

    # Animated GLB
    print(f"\n--- Building animated GLB ---")
    glb_path = os.path.join(args.output_dir, "gnm_head_animated.glb")
    build_glb_with_morphs(
        base_verts=base_verts * args.scale,
        faces=faces,
        morph_dict={k: v * args.scale for k, v in morph_dict.items()},
        timeline=timeline,
        output_path=glb_path,
        uv=uv,
    )

    print("=" * 60)
    print("GNM export complete!")


if __name__ == "__main__":
    main()
