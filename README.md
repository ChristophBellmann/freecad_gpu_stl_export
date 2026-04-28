# FreeCAD GPU STL Export

Fensterfreier STL-Export fuer FreeCAD-Revolutionsprofile:

1. Profilkurve aus der `.FCStd` lesen
2. Profilpunkte dichter samplen
3. Profil per GPU/CPU rotieren
4. binaeres STL schreiben

Der Export nutzt die Sketch-/BREP-Geometrie direkt und nicht FreeCADs fertigen Mesher.

## Verzeichnislayout

```text
in/      lokale FreeCAD-Dateien (*.FCStd), nicht in Git
out/     generierte CSV/STL-Dateien, nicht in Git
*.py     Exportskripte
```

`in/` und `out/` sind bewusst als lokale Arbeitsordner gedacht. Grosse Dateien bleiben aus dem Git-Repo raus.

## Schnellstart

Profilpunkte aus der gespeicherten FreeCAD-BREP-Kurve erzeugen:

```bash
python3 export_profile_brep_csv.py \
  in/BisFree_revolution_test123.FCStd \
  out/profile_smooth.csv \
  --samples-per-spline 384
```

STL mit CPU-Fallback schreiben:

```bash
python3 gpu_revolve_to_stl.py \
  out/profile_smooth.csv \
  out/revolution_gpu_smooth.stl \
  --segments 4096 \
  --cpu
```

STL mit deiner Custom-ROCm-PyTorch-Umgebung schreiben:

```bash
/media/christoph/some_space/Compute/ML-Lab/examples/rocm711_torch_example/.venv/bin/python-rocm \
  gpu_revolve_to_stl.py \
  out/profile_smooth.csv \
  out/revolution_gpu_rocm_smooth.stl \
  --segments 4096
```

Bei ROCm meldet PyTorch trotzdem `device: cuda`; das ist der HIP-Backend-Pfad.

## Qualitaet

Zwei Werte bestimmen die sichtbare Rundung:

| Stellschraube | Wirkung |
| --- | --- |
| `--samples-per-spline` | glattere Profilkurve, mehr Punkte entlang der FreeCAD-Kurve |
| `--segments` | glattere Rotation um die Achse, mehr Umfangssegmente |

Das alte `out/profile.csv` hatte nur 29 Profilzeilen. Das ist fuer glatte Profilrundungen zu grob. Mit `--samples-per-spline 384` entstehen hier 577 Profilzeilen.

Pragmatische Werte:

```bash
# schnelle Vorschau
python3 gpu_revolve_to_stl.py out/profile_smooth.csv out/preview.stl --segments 512 --cpu

# guter Export
python3 gpu_revolve_to_stl.py out/profile_smooth.csv out/final.stl --segments 4096 --cpu

# sehr feine Umfangsrundung, grosse Datei
python3 gpu_revolve_to_stl.py out/profile_smooth.csv out/final_fine.stl --segments 8192 --cpu
```

## Flags

### `export_profile_brep_csv.py`

```text
fcstd                    Eingabe: FreeCAD-Datei
output_csv               Ausgabe: Profilpunkte als CSV
--shape NAME             Shape im FCStd-Archiv, ohne .Shape.brp
--samples-per-spline N   Anzahl Samples pro B-Spline-Profilkurve
```

Default-Shape:

```text
RingFrisbeeProfile_FeatureBand
```

### `gpu_revolve_to_stl.py`

```text
profile_csv              Eingabe: Profilpunkte
output_stl               Ausgabe: binaeres STL
--segments N             Umfangssegmente der Rotation, Default: 2048
--cpu                    Torch-CPU erzwingen; ohne Torch: NumPy-Fallback
--flip                   Dreiecksorientierung umdrehen, falls innen/aussen vertauscht ist
--open-profile           Profil nicht automatisch schliessen
```

## Hinweise

- Wenn die Oberflaeche innen statt aussen orientiert ist, denselben Export mit `--flip` wiederholen.
- `--segments` glattert nur die Rotation. Fuer glatte Profilradien muss das CSV-Profil selbst dicht genug sein.
- Ohne installierten PyTorch nutzt `gpu_revolve_to_stl.py` automatisch NumPy auf der CPU.
- Ausgabeverzeichnisse werden bei Bedarf automatisch angelegt.
