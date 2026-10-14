import os
import argparse
import numpy as np
import trimesh
from pygltflib import (
    GLTF2, Scene, Node, Mesh, Primitive, Attributes, Buffer, BufferView,
    Accessor, Animation, AnimationChannel, AnimationChannelTarget,
    AnimationSampler, Asset
)


def build_glb_with_morphs(base_verts, faces, morph_dict, timeline, output_path,
                          uv=None):
    """
    Build a GLB with morph targets and a keyframed animation driving morph weights.

    base_verts: (N, 3) float32 - base mesh in mm
    faces: (F, 3) uint32 - triangle indices
    morph_dict: {name: (N, 3) float32} - position deltas per morph target
    timeline: [(time_seconds, {name: weight}), ...]
    uv: (N, 2) float32 optional - texture coordinates
    """
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
        buffer_views.append(BufferView(
            buffer=0,
            byteOffset=offset,
            byteLength=int(np_array.nbytes),
            target=target,
        ))

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

        accessor = Accessor(
            bufferView=bv_idx,
            componentType=comp_type,
            count=int(count),
            type=type_str,
        )

        if need_bounds and np_array.dtype == np.float32:
            if type_str == "VEC3":
                accessor.min = np_array.min(axis=0).tolist()
                accessor.max = np_array.max(axis=0).tolist()
            elif type_str == "SCALAR":
                accessor.min = [float(np_array.min())]
                accessor.max = [float(np_array.max())]

        acc_idx = len(accessors)
        accessors.append(accessor)
        return acc_idx

    # --- Add base mesh data ---
    pos_acc = add_accessor(base_verts.astype(np.float32),
                           target=34962, need_bounds=True)
    idx_acc = add_accessor(faces.flatten().astype(np.uint32), target=34963)

    # --- Morph target deltas ---
    delta_accs = []
    for name in morph_names:
        delta = morph_dict[name].astype(np.float32)
        acc = add_accessor(delta, target=34962)
        delta_accs.append(acc)

    # --- Animation times ---
    times = np.array([t for t, _ in timeline], dtype=np.float32)
    time_acc = add_accessor(times, need_bounds=True)

    # --- Animation weights (flat array: keyframes × morph targets) ---
    weights_flat = []
    for _, weight_dict in timeline:
        for name in morph_names:
            weights_flat.append(float(weight_dict.get(name, 0.0)))
    weight_acc = add_accessor(np.array(weights_flat, dtype=np.float32))

    # --- UVs ---
    attr_dict = {"POSITION": pos_acc}
    if uv is not None:
        uv_acc = add_accessor(uv.astype(np.float32), target=34962)
        attr_dict["TEXCOORD_0"] = uv_acc

    # --- Build primitive with morph targets ---
    morph_targets = [{"POSITION": delta_accs[i]} for i in range(n_morphs)]
    prim = Primitive(
        attributes=Attributes(**attr_dict),
        indices=idx_acc,
        targets=morph_targets,
    )

    mesh = Mesh(
        primitives=[prim],
        weights=[0.0] * n_morphs,
        extras={"targetNames": morph_names},
    )

    # --- Animation channel (single channel drives all morph weights) ---
    sampler = AnimationSampler(
        input=time_acc,
        output=weight_acc,
        interpolation="LINEAR",
    )
    channel = AnimationChannel(
        sampler=0,
        target=AnimationChannelTarget(node=0, path="weights"),
    )
    animation = Animation(
        channels=[channel],
        samplers=[sampler],
        name="Expression_Timeline",
    )

    node = Node(mesh=0, name="GNM_Head")
    scene = Scene(nodes=[0])
    asset = Asset(version="2.0", generator="H0mans-GNM")

    bin_data = b''.join(bin_parts)
    bin_data += b'\x00' * ((-len(bin_data)) % 4)
    buffer = Buffer(byteLength=len(bin_data))

    gltf = GLTF2(
        asset=asset,
        scene=0,
        scenes=[scene],
        nodes=[node],
        meshes=[mesh],
        animations=[animation],
        buffers=[buffer],
        bufferViews=buffer_views,
        accessors=accessors,
    )
    gltf.binary_blob = bin_data
    gltf.save(output_path)

    print(f"Saved GLB: {output_path} "
          f"({len(bin_data) / 1024 / 1024:.2f} MB, "
          f"{n_morphs} morph targets, "
          f"{len(timeline)} keyframes)")


def export_static_obj(vertices, faces, uv, output_path, scale_factor=1000.0):
    """Export a static OBJ for reference."""
    mesh = trimesh.Trimesh(vertices=vertices * scale_factor,
                           faces=faces, process=False)
    mesh.export(output_path)
    return mesh


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--scale", type=float, default=1000.0)
    parser.add_argument("--n_morphs", type=int, default=6,
                        help="Number of morph targets to generate")
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

    # --- Faces ---
    faces = None
    for attr in ['triangles', 'faces', 'F']:
        if hasattr(model, attr):
            faces = np.asarray(getattr(model, attr))
            print(f"Faces found via '{attr}': {faces.shape}")
            break
    if faces is None:
        raise RuntimeError("Cannot find face topology on GNM model")
    faces = faces.astype(np.uint32)

    # --- UVs ---
    uv = None
    for attr in ['uv_coordinates', 'uvs', 'UV', 'texcoords']:
        if hasattr(model, attr):
            uv = np.asarray(getattr(model, attr))
            print(f"UVs found via '{attr}': shape={uv.shape}")
            break
    if uv is None:
        print("No UV coordinates found — will export without textures.")

    # --- Base (neutral) mesh ---
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
    print(f"Expression coefficients available: {n_expr_coefs}")

    # --- Generate morph targets from first N expression coefficients ---
    print(f"\n--- Generating {args.n_morphs} morph targets ---")
    morph_dict = {}
    for i in range(args.n_morphs):
        expr_vec = np.zeros_like(base_expr)
        expr_vec[..., i] = 3.0  # amplify the coefficient

        params_i = dict(neutral_params)
        params_i['expression'] = expr_vec

        verts_i = model.forward_vertices(**params_i)
        if verts_i.ndim == 3:
            verts_i = verts_i[0]
        verts_i = np.asarray(verts_i, dtype=np.float32)

        delta = verts_i - base_verts
        name = f"expr_{i}"
        morph_dict[name] = delta

        mag = float(np.linalg.norm(delta, axis=1).mean())
        print(f"  {name}: mean delta magnitude = {mag:.4f} m")

    # --- Build timeline: each expression shows for ~2 seconds ---
    print(f"\n--- Building timeline ---")
    timeline = []
    timeline.append((0.0, {}))       # start neutral
    timeline.append((0.5, {}))       # hold neutral briefly

    for i in range(args.n_morphs):
        t0 = 0.5 + i * 2.5
        timeline.append((t0 + 0.3, {f"expr_{i}": 1.0}))   # ramp up
        timeline.append((t0 + 2.0, {f"expr_{i}": 1.0}))   # hold
        timeline.append((t0 + 2.5, {}))                    # back to neutral

    duration = timeline[-1][0]
    print(f"Timeline: {len(timeline)} keyframes, total {duration:.1f}s")
    for t, w in timeline:
        if w:
            print(f"  t={t:5.2f}s  {list(w.keys())[0]} = 1.0")

    # --- Export static OBJs (for inspection) ---
    print(f"\n--- Exporting static OBJs ---")
    neutral_path = os.path.join(args.output_dir, "gnm_neutral.obj")
    export_static_obj(base_verts, faces, uv, neutral_path, args.scale)
    print(f"  {neutral_path}")

    for i in range(args.n_morphs):
        verts_i = base_verts + morph_dict[f"expr_{i}"]
        expr_path = os.path.join(args.output_dir, f"gnm_expr_{i}.obj")
        export_static_obj(verts_i, faces, uv, expr_path, args.scale)
        print(f"  {expr_path}")

    # --- Build animated GLB ---
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
