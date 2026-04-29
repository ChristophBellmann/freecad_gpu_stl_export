# FreeCAD STL Export

Export a FreeCAD `.FCStd` profile from `in/` to a binary STL in `out/` using a single command.

```bash
python3 export_stl.py
```

## Default Behavior

```text
Input:    the only *.FCStd file in in/
Output:   out/<input-file-name>.stl
Preset:   fine
Backend:  GPU (CUDA/NVIDIA or ROCm) from .venv/
```

The script automatically re-executes itself inside the local virtual environment at `.venv/`.
On ROCm systems, additional ROCm paths are injected automatically when available.
On NVIDIA/CUDA systems, no ROCm-specific setup is required.

## CPU Mode

Use CPU instead of GPU:

```bash
python3 export_stl.py --cpu
```

## Multiple FreeCAD Files

If `in/` contains more than one `.FCStd` file, pass the input explicitly:

```bash
python3 export_stl.py in/my_model.FCStd
```

## Custom Output Path

```bash
python3 export_stl.py --output out/my_export.stl
```

## Quality Presets

```bash
python3 export_stl.py --preset draft
python3 export_stl.py --preset standard
python3 export_stl.py --preset fine
```

| Preset | Profile Samples | Rotation Segments |
| --- | ---: | ---: |
| `draft` | 96 | 512 |
| `standard` | 384 | 4096 |
| `fine` | 768 | 8192 |

Override preset values directly:

```bash
python3 export_stl.py --samples 768 --segments 8192
```

## Additional Flags

```text
--flip              Reverse triangle orientation
--shape NAME        Shape member inside the FCStd archive
--open-profile      Do not auto-close profile loops
```

If face orientation is inverted (inside-out), use:

```bash
python3 export_stl.py --flip
```

Note: On ROCm, PyTorch may still report `device: cuda`. This is expected and maps to the HIP backend.
