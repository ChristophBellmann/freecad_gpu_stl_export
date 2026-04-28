import argparse
import csv
import os
import struct
import sys
from collections import defaultdict

import numpy as np

try:
    import torch
except ModuleNotFoundError:
    torch = None


def read_profile_csv(path):
    loops = defaultdict(list)

    with open(path, newline="") as f:
        reader = csv.reader(f)
        first = next(reader)

        def is_header(row):
            try:
                [float(x) for x in row]
                return False
            except ValueError:
                return True

        rows = reader if is_header(first) else [first, *reader]

        for row in rows:
            row = [x.strip() for x in row if x.strip() != ""]
            if len(row) == 2:
                loop_id = 0
                r = float(row[0])
                z = float(row[1])
            elif len(row) >= 3:
                loop_id = int(float(row[0]))
                r = float(row[1])
                z = float(row[2])
            else:
                continue

            loops[loop_id].append((r, z))

    return {k: np.array(v, dtype=np.float32) for k, v in loops.items()}


def remove_duplicate_closing_point(points, eps=1e-6):
    if len(points) >= 2:
        if np.linalg.norm(points[0] - points[-1]) < eps:
            return points[:-1]
    return points


def make_loop_triangles(points, segments, device, closed=True):
    points = remove_duplicate_closing_point(points)

    if len(points) < 2:
        return None

    p = torch.tensor(points, dtype=torch.float32, device=device)

    if closed:
        p0 = p
        p1 = torch.roll(p, shifts=-1, dims=0)
    else:
        p0 = p[:-1]
        p1 = p[1:]

    theta = torch.linspace(
        0.0,
        2.0 * torch.pi,
        segments + 1,
        device=device,
        dtype=torch.float32,
    )[:-1]

    c = torch.cos(theta)
    s = torch.sin(theta)

    def rotate(profile_points):
        r = profile_points[:, 0]
        z = profile_points[:, 1]

        x = r[:, None] * c[None, :]
        y = r[:, None] * s[None, :]
        zz = z[:, None].expand_as(x)

        return torch.stack((x, y, zz), dim=-1)

    v00 = rotate(p0)
    v10 = rotate(p1)

    v01 = torch.roll(v00, shifts=-1, dims=1)
    v11 = torch.roll(v10, shifts=-1, dims=1)

    tri_a = torch.stack((v00, v10, v11), dim=2).reshape(-1, 3, 3)
    tri_b = torch.stack((v00, v11, v01), dim=2).reshape(-1, 3, 3)

    return torch.cat((tri_a, tri_b), dim=0)


def make_loop_triangles_numpy(points, segments, closed=True):
    points = remove_duplicate_closing_point(points).astype(np.float32, copy=False)

    if len(points) < 2:
        return None

    if closed:
        p0 = points
        p1 = np.roll(points, shift=-1, axis=0)
    else:
        p0 = points[:-1]
        p1 = points[1:]

    theta = np.linspace(0.0, 2.0 * np.pi, segments, endpoint=False, dtype=np.float32)
    c = np.cos(theta)
    s = np.sin(theta)

    def rotate(profile_points):
        r = profile_points[:, 0]
        z = profile_points[:, 1]
        x = r[:, None] * c[None, :]
        y = r[:, None] * s[None, :]
        zz = np.broadcast_to(z[:, None], x.shape)
        return np.stack((x, y, zz), axis=-1)

    v00 = rotate(p0)
    v10 = rotate(p1)
    v01 = np.roll(v00, shift=-1, axis=1)
    v11 = np.roll(v10, shift=-1, axis=1)

    tri_a = np.stack((v00, v10, v11), axis=2).reshape(-1, 3, 3)
    tri_b = np.stack((v00, v11, v01), axis=2).reshape(-1, 3, 3)

    return np.concatenate((tri_a, tri_b), axis=0)


def write_binary_stl(path, triangles):
    if isinstance(triangles, np.ndarray):
        v1 = triangles[:, 1] - triangles[:, 0]
        v2 = triangles[:, 2] - triangles[:, 0]

        normals = np.cross(v1, v2, axis=1)
        lengths = np.linalg.norm(normals, axis=1)

        mask = lengths > 1e-12
        tri_cpu = triangles[mask].astype("<f4", copy=False)
        normal_cpu = (normals[mask] / lengths[mask, None]).astype("<f4", copy=False)
        return write_binary_stl_arrays(path, tri_cpu, normal_cpu)

    v1 = triangles[:, 1] - triangles[:, 0]
    v2 = triangles[:, 2] - triangles[:, 0]

    normals = torch.cross(v1, v2, dim=1)
    lengths = torch.linalg.norm(normals, dim=1)

    mask = lengths > 1e-12
    triangles = triangles[mask]
    normals = normals[mask] / lengths[mask, None]

    tri_cpu = triangles.detach().cpu().numpy().astype("<f4", copy=False)
    normal_cpu = normals.detach().cpu().numpy().astype("<f4", copy=False)

    return write_binary_stl_arrays(path, tri_cpu, normal_cpu)


def write_binary_stl_arrays(path, tri_cpu, normal_cpu):
    output_dir = os.path.dirname(path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    count = tri_cpu.shape[0]
    dtype = np.dtype([
        ("normal", "<f4", (3,)),
        ("vertices", "<f4", (3, 3)),
        ("attr", "<u2"),
    ])

    data = np.zeros(count, dtype=dtype)
    data["normal"] = normal_cpu
    data["vertices"] = tri_cpu

    with open(path, "wb") as f:
        header = b"GPU revolved STL from profile CSV"
        f.write(header.ljust(80, b" "))
        f.write(struct.pack("<I", count))
        data.tofile(f)

    return count


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("profile_csv")
    ap.add_argument("output_stl")
    ap.add_argument("--segments", type=int, default=2048)
    ap.add_argument("--open-profile", action="store_true")
    ap.add_argument("--flip", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    if torch is None:
        device = None
    elif args.cpu:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("device:", device if device is not None else "numpy-cpu")

    if device is not None and device.type != "cuda":
        print("WARNING: GPU not active. Check ROCm/PyTorch installation.", file=sys.stderr)
    elif device is None:
        print("WARNING: PyTorch not installed. Using NumPy CPU fallback.", file=sys.stderr)

    loops = read_profile_csv(args.profile_csv)

    all_triangles = []

    for loop_id, points in sorted(loops.items()):
        if device is None:
            tris = make_loop_triangles_numpy(
                points=points,
                segments=args.segments,
                closed=not args.open_profile,
            )
        else:
            tris = make_loop_triangles(
                points=points,
                segments=args.segments,
                device=device,
                closed=not args.open_profile,
            )

        if tris is None:
            continue

        all_triangles.append(tris)
        print(f"loop {loop_id}: {len(points)} profile points -> {tris.shape[0]} triangles")

    if not all_triangles:
        print("No triangles generated.")
        sys.exit(4)

    if device is None:
        triangles = np.concatenate(all_triangles, axis=0)
    else:
        triangles = torch.cat(all_triangles, dim=0)

    if args.flip:
        triangles = triangles[:, [0, 2, 1], :]

    count = write_binary_stl(args.output_stl, triangles)

    print(f"written: {args.output_stl}")
    print(f"triangles: {count}")


if __name__ == "__main__":
    main()
