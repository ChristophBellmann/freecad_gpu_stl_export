FreeCAD-Sketch einmal als Profilpunkte exportieren
→ Profilpunkte auf GPU/CPU rotieren
→ Dreiecke erzeugen
→ binäres STL schreiben


Das erzeugt ein STL aus der Sketch-Geometrie, nicht aus FreeCADs fertigem BREP-Mesher. Für deine Ring-/Revolutionsform ist das genau der schnelle Weg.

Fensterfreier Export direkt aus der im FCStd gespeicherten BREP-Kurve:

python3 export_profile_brep_csv.py in/BisFree_revolution_test123.FCStd out/profile_smooth.csv --samples-per-spline 384

python3 gpu_revolve_to_stl.py out/profile_smooth.csv out/revolution_gpu_smooth.stl --segments 4096 --cpu

Das alte `out/profile.csv` hatte nur 29 Profilpunkte. Für sichtbare Rundungen ist das zu grob; `out/profile_smooth.csv` hat bei obigem Befehl 577 Profilpunkte.

Mit deiner Custom-ROCm-7.11-PyTorch-Umgebung:

/media/christoph/some_space/Compute/ML-Lab/examples/rocm711_torch_example/.venv/bin/python-rocm gpu_revolve_to_stl.py out/profile_smooth.csv out/revolution_gpu_rocm_smooth.stl --segments 4096

Ohne PyTorch nutzt der Exporter automatisch NumPy auf der CPU. Mit `python-rocm` sollte `device: cuda` erscheinen; bei ROCm ist das PyTorchs HIP-Backend.

Für glattere Rundung:

python3 gpu_revolve_to_stl.py out/profile.csv out/revolution_gpu.stl --segments 8192

Wichtig: `--segments` glättet nur die Rotation um die Achse. Für glattere Profilkurven muss auch das CSV-Profil dichter sein, z.B. mit `export_profile_brep_csv.py`.

Für schnellere Vorschau:

python3 gpu_revolve_to_stl.py out/profile.csv out/revolution_gpu_preview.stl --segments 512

Wenn die Oberfläche innen statt außen orientiert ist:

python3 gpu_revolve_to_stl.py out/profile.csv out/revolution_gpu.stl --segments 4096 --flip
