# FreeCAD STL Export

Export a FreeCAD `.FCStd` profile from `in/` to a binary STL in `out/` using a single command.

```bash
python3 export_stl.py
```

## Setup (`.venv`)

Create the local virtual environment and install dependencies:

```bash
./setup_venv.sh cpu
```

GPU variants:

```bash
./setup_venv.sh cuda
./setup_venv.sh rocm
./setup_venv.sh rocm-custom /opt/rocm-custom
```

Optional: override the PyTorch wheel index URL:

```bash
TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124 ./setup_venv.sh cuda
TORCH_INDEX_URL=https://download.pytorch.org/whl/rocm6.2.4 ./setup_venv.sh rocm
```

Custom torch (works for CPU/CUDA/ROCm modes):

```bash
TORCH_WHEEL=/path/to/torch-*.whl ./setup_venv.sh rocm-custom /path/to/rocm
TORCH_SPEC='torch==2.7.1+rocm6.2.4' ./setup_venv.sh rocm
```

Optional NumPy override (default is `numpy<2` for broad torch-wheel compatibility):

```bash
NUMPY_SPEC='numpy==1.26.4' ./setup_venv.sh rocm
```

For custom ROCm installations:

- `./setup_venv.sh rocm-custom /path/to/your/rocm` installs ROCm PyTorch and stores the path in `.rocm_path.local`.
- At runtime, `export_stl.py` resolves ROCm in this order:
  1. `ROCM_PATH`
  2. `FREECAD_ROCM_PATH`
  3. `.rocm_path.local`
  4. `/opt/rocm`

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

## NVIDIA / CUDA

For NVIDIA GPUs, install a CUDA-enabled PyTorch build inside `.venv`.
The script uses `torch.cuda.is_available()` and `torch.device("cuda")`, so CUDA-backed PyTorch runs on NVIDIA GPUs without extra flags.

Quick check:

```bash
python3 -c "import torch; print('cuda_available=', torch.cuda.is_available(), 'cuda=', torch.version.cuda)"
python3 export_stl.py --preset draft --shape <SHAPE_NAME>
```

If `cuda_available=True`, GPU export should work.

If a machine has both CUDA and ROCm stacks installed, prefer running in a clean CUDA environment for NVIDIA to avoid accidental library path conflicts.

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
