import csv
import os
import sys
import FreeCAD as App


def script_args(argv):
    args = argv[1:]
    if args and os.path.basename(args[0]) == os.path.basename(__file__):
        args = args[1:]
    return args


args = script_args(sys.argv)

if len(args) < 2:
    print("Usage: FreeCADCmd export_profile_csv.py input.FCStd output.csv [sketch_name] [deflection_mm]")
    sys.exit(1)

fcstd = args[0]
out_csv = args[1]
sketch_name = args[2] if len(args) >= 3 else "RingFrisbeeProfile_FeatureBand"
deflection = float(args[3]) if len(args) >= 4 else 0.005

doc = App.openDocument(fcstd)
sketch = doc.getObject(sketch_name)

if sketch is None:
    print(f"Sketch not found: {sketch_name}")
    print("Available objects:")
    for obj in doc.Objects:
        print(" ", obj.Name, obj.TypeId)
    sys.exit(2)

shape = sketch.Shape
wires = list(shape.Wires)

if not wires:
    print("No closed wires found in sketch shape.")
    sys.exit(3)

rows = []

for loop_id, wire in enumerate(wires):
    pts = wire.discretize(Deflection=deflection)

    clean = []
    for p in pts:
        if not clean:
            clean.append(p)
            continue
        q = clean[-1]
        if (p.sub(q)).Length > 1e-7:
            clean.append(p)

    for p in clean:
        # FreeCAD sketch: X = Radius zur V-Achse, Y = Höhe
        r = abs(float(p.x))
        z = float(p.y)
        rows.append((loop_id, r, z))

output_dir = os.path.dirname(out_csv)
if output_dir:
    os.makedirs(output_dir, exist_ok=True)

with open(out_csv, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["loop_id", "r_mm", "z_mm"])
    w.writerows(rows)

print(f"exported {len(rows)} points from {len(wires)} loop(s): {out_csv}")
print(f"deflection: {deflection:g} mm")
