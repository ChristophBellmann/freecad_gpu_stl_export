import argparse
import csv
import os
import zipfile

import numpy as np


def read_shape_brep(fcstd_path, shape_name):
    member = f"{shape_name}.Shape.brp"
    with zipfile.ZipFile(fcstd_path) as zf:
        with zf.open(member) as f:
            return f.read().decode("utf-8")


def parse_curves(brep_text):
    lines = [line.strip() for line in brep_text.splitlines() if line.strip()]
    curves = []

    i = 0
    while i < len(lines):
        parts = lines[i].split()
        if len(parts) == 2 and parts[0] == "Curves":
            count = int(parts[1])
            i += 1
            for _ in range(count):
                parts = lines[i].split()
                curve_type = int(parts[0])

                if curve_type == 1:
                    values = [float(x) for x in parts[1:]]
                    curves.append({
                        "type": "line",
                        "origin": np.array(values[:3], dtype=float),
                        "direction": np.array(values[3:6], dtype=float),
                    })
                    i += 1
                    continue

                if curve_type == 7:
                    degree = int(parts[3])
                    pole_count = int(parts[4])
                    knot_count = int(parts[5])
                    values = [float(x) for x in parts[6:]]

                    poles = []
                    weights = []
                    for j in range(pole_count):
                        x, y, z, w = values[j * 4:(j + 1) * 4]
                        poles.append((x, y, z))
                        weights.append(w)

                    knot_parts = lines[i + 1].split()
                    knots = []
                    for j in range(knot_count):
                        knot = float(knot_parts[j * 2])
                        multiplicity = int(knot_parts[j * 2 + 1])
                        knots.extend([knot] * multiplicity)

                    curves.append({
                        "type": "bspline",
                        "degree": degree,
                        "poles": np.array(poles, dtype=float),
                        "weights": np.array(weights, dtype=float),
                        "knots": np.array(knots, dtype=float),
                    })
                    i += 2
                    continue

                raise ValueError(f"Unsupported BREP curve type: {curve_type}")
            return curves

        i += 1

    raise ValueError("No Curves section found in BREP.")


def basis_function(i, degree, knots, u):
    if degree == 0:
        if knots[i] <= u < knots[i + 1]:
            return 1.0
        if u == knots[-1] and knots[i] <= u <= knots[i + 1]:
            return 1.0
        return 0.0

    left = 0.0
    left_den = knots[i + degree] - knots[i]
    if left_den:
        left = (u - knots[i]) / left_den * basis_function(i, degree - 1, knots, u)

    right = 0.0
    right_den = knots[i + degree + 1] - knots[i + 1]
    if right_den:
        right = (
            (knots[i + degree + 1] - u)
            / right_den
            * basis_function(i + 1, degree - 1, knots, u)
        )

    return left + right


def sample_bspline(curve, count):
    degree = curve["degree"]
    poles = curve["poles"]
    weights = curve["weights"]
    knots = curve["knots"]

    u0 = knots[degree]
    u1 = knots[-degree - 1]
    samples = []

    for u in np.linspace(u0, u1, count):
        numerator = np.zeros(3)
        denominator = 0.0

        for i, (point, weight) in enumerate(zip(poles, weights)):
            b = basis_function(i, degree, knots, float(u))
            bw = b * weight
            numerator += bw * point
            denominator += bw

        samples.append(numerator / denominator)

    return np.array(samples, dtype=float)


def close_enough(a, b, eps=1e-6):
    return np.linalg.norm(a[:2] - b[:2]) <= eps


def build_profile_loops(curves, samples_per_spline):
    splines = [curve for curve in curves if curve["type"] == "bspline"]
    lines = [curve for curve in curves if curve["type"] == "line"]

    if len(splines) < 2 or len(lines) < 2:
        raise ValueError("Expected at least two B-spline profile curves and two closing lines.")

    outer = sample_bspline(splines[0], samples_per_spline)
    inner_curve = sample_bspline(splines[1], max(8, samples_per_spline // 2))

    inner_line = None
    for line in lines:
        start = line["origin"]
        end = line["origin"] + line["direction"] * np.linalg.norm(
            inner_curve[0] - inner_curve[-1]
        )
        if close_enough(start, inner_curve[-1]) and close_enough(end, inner_curve[0]):
            inner_line = np.array([start, end])
            break

    if inner_line is None:
        inner_line = np.array([inner_curve[-1], inner_curve[0]])

    return [outer, np.vstack((inner_line, inner_curve[1:]))]


def write_profile_csv(path, loops):
    output_dir = os.path.dirname(path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["loop_id", "r_mm", "z_mm"])

        for loop_id, points in enumerate(loops):
            for p in points:
                writer.writerow([loop_id, abs(float(p[0])), float(p[1])])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("fcstd")
    parser.add_argument("output_csv")
    parser.add_argument(
        "--shape",
        default="RingFrisbeeProfile_FeatureBand",
        help="Shape member name inside the FCStd file, without .Shape.brp",
    )
    parser.add_argument("--samples-per-spline", type=int, default=256)
    args = parser.parse_args()

    brep_text = read_shape_brep(args.fcstd, args.shape)
    curves = parse_curves(brep_text)
    loops = build_profile_loops(curves, args.samples_per_spline)
    write_profile_csv(args.output_csv, loops)

    total = sum(len(loop) for loop in loops)
    print(f"exported {total} points from {len(loops)} loop(s): {args.output_csv}")
    print(f"source: {os.path.basename(args.fcstd)}:{args.shape}.Shape.brp")


if __name__ == "__main__":
    main()
