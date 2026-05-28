import os
import glob
import traceback

import trimesh
import numpy as np
import laspy

from tqdm import tqdm


def process_file(file_path, output_dir, n_samples=1000000):
    """
    Convert a single GLB file into LAS point cloud.
    """

    try:
        filename = os.path.basename(file_path)
        name, _ = os.path.splitext(filename)

        output_path = os.path.join(output_dir, f"{name}.las")

        print(f"Processing: {filename}")

        mesh = trimesh.load(file_path)

        final_points = []
        final_colors = []

        # --------------------------------------------------
        # SCENE HANDLING
        # --------------------------------------------------

        if isinstance(mesh, trimesh.Scene):

            point_clouds = [
                g
                for g in mesh.geometry.values()
                if isinstance(g, trimesh.points.PointCloud)
            ]

            meshes = [
                g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)
            ]

            # ----------------------------------------------
            # POINT CLOUDS
            # ----------------------------------------------

            for pc in point_clouds:

                final_points.append(pc.vertices)

                if hasattr(pc, "colors") and len(pc.colors) > 0:
                    final_colors.append(pc.colors)

                else:
                    final_colors.append(
                        np.ones((len(pc.vertices), 4), dtype=np.uint8) * 255
                    )

            # ----------------------------------------------
            # MESHES
            # ----------------------------------------------

            if meshes:

                combined_mesh = trimesh.util.concatenate(meshes)

                if not combined_mesh.is_empty:

                    points, face_indices = trimesh.sample.sample_surface(
                        combined_mesh, n_samples
                    )

                    final_points.append(points)

                    colors = sample_mesh_colors(combined_mesh, points, face_indices)

                    final_colors.append(colors)

        # --------------------------------------------------
        # SINGLE MESH
        # --------------------------------------------------

        elif isinstance(mesh, trimesh.Trimesh):

            points, face_indices = trimesh.sample.sample_surface(mesh, n_samples)

            final_points.append(points)

            colors = sample_mesh_colors(mesh, points, face_indices)

            final_colors.append(colors)

        else:

            print(f"Skipping {filename}: Unsupported geometry type")
            return False

        # --------------------------------------------------
        # VALIDATION
        # --------------------------------------------------

        if not final_points:

            print(f"Skipping {filename}: No valid geometry")
            return False

        all_points = np.vstack(final_points)
        all_colors = np.vstack(final_colors)

        if len(all_points) == 0:

            print(f"Skipping {filename}: 0 points")
            return False

        # --------------------------------------------------
        # LAS EXPORT
        # --------------------------------------------------

        header = laspy.LasHeader(point_format=3, version="1.2")

        header.scales = np.array([0.001, 0.001, 0.001])
        header.offsets = np.min(all_points, axis=0)

        las = laspy.LasData(header)

        las.x = all_points[:, 0]
        las.y = all_points[:, 1]
        las.z = all_points[:, 2]

        if all_colors.shape[1] == 4:
            all_colors = all_colors[:, :3]

        las.red = all_colors[:, 0].astype(np.uint16) * 256
        las.green = all_colors[:, 1].astype(np.uint16) * 256
        las.blue = all_colors[:, 2].astype(np.uint16) * 256

        os.makedirs(output_dir, exist_ok=True)

        las.write(output_path)

        print(f"Saved: {output_path}")

        return True

    except Exception as e:

        traceback.print_exc()
        print(f"Error processing {file_path}: {e}")

        return False


def sample_mesh_colors(mesh, points, face_indices):
    """
    Sample interpolated RGB colors from mesh vertices.
    """

    if hasattr(mesh.visual, "to_color"):

        try:
            mesh.visual = mesh.visual.to_color()
        except:
            pass

    if not hasattr(mesh.visual, "vertex_colors"):

        mesh.visual.vertex_colors = (
            np.ones((len(mesh.vertices), 4), dtype=np.uint8) * 255
        )

    triangles = mesh.vertices[mesh.faces[face_indices]]

    v0 = triangles[:, 0, :]
    v1 = triangles[:, 1, :]
    v2 = triangles[:, 2, :]

    p = points

    v0v1 = v1 - v0
    v0v2 = v2 - v0
    v0p = p - v0

    d00 = np.einsum("ij,ij->i", v0v1, v0v1)
    d01 = np.einsum("ij,ij->i", v0v1, v0v2)
    d11 = np.einsum("ij,ij->i", v0v2, v0v2)
    d20 = np.einsum("ij,ij->i", v0p, v0v1)
    d21 = np.einsum("ij,ij->i", v0p, v0v2)

    denom = d00 * d11 - d01 * d01
    denom[denom == 0] = 1.0

    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w

    faces = mesh.faces[face_indices]

    if hasattr(mesh.visual, "vertex_colors"):

        face_colors = mesh.visual.vertex_colors[faces]

    else:

        face_colors = np.ones((len(faces), 3, 4), dtype=np.uint8) * 255

    c0 = face_colors[:, 0, :]
    c1 = face_colors[:, 1, :]
    c2 = face_colors[:, 2, :]

    point_colors = (u[:, None] * c0 + v[:, None] * c1 + w[:, None] * c2).astype(
        np.uint8
    )

    return point_colors


def run(input_dir, output_dir, samples=1000000, recursive=True):
    """
    Pipeline-safe GLB → LAS conversion step.
    """

    if input_dir is None:
        raise ValueError("input_dir is required")

    if output_dir is None:
        raise ValueError("output_dir is required")

    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Directory not found: {input_dir}")

    os.makedirs(output_dir, exist_ok=True)

    print("STEP 05: GLB → LAS Conversion")

    if recursive:

        glb_files = glob.glob(os.path.join(input_dir, "**", "*.glb"), recursive=True)

    else:

        glb_files = glob.glob(os.path.join(input_dir, "*.glb"))

    if not glb_files:
        raise RuntimeError("No GLB files found")

    print(f"Found {len(glb_files)} GLB files")

    processed_count = 0

    for file_path in tqdm(glb_files):

        rel_path = os.path.relpath(os.path.dirname(file_path), input_dir)

        if rel_path == ".":
            current_output_dir = output_dir

        else:
            current_output_dir = os.path.join(output_dir, rel_path)

        success = process_file(
            file_path=file_path, output_dir=current_output_dir, n_samples=samples
        )

        if success:
            processed_count += 1

    print(f"GLB conversion complete: {processed_count} files processed")

    return {"output_dir": output_dir, "processed_files": processed_count}


# --------------------------------------------------
# CLI ENTRY
# --------------------------------------------------

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)

    parser.add_argument("--samples", type=int, default=1000000)

    args = parser.parse_args()

    run(input_dir=args.input_dir, output_dir=args.output_dir, samples=args.samples)
