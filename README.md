# FreeCAD STL Export

Ein Schritt: FreeCAD-Datei aus `in/` als feines STL nach `out/` exportieren.

```bash
python3 export_stl.py
```

Default:

```text
Eingabe:  die einzige *.FCStd in in/
Ausgabe:  out/<freecad-dateiname>.stl
Preset:   fine
Backend:  GPU/ROCm aus .venv/
```

Die lokale venv liegt hier:

```text
.venv/
```

Das Skript startet sich automatisch in diese venv neu und setzt die noetigen ROCm-Pfade. Fuer die Bedienung ist kein Aktivieren und kein extra Wrapper noetig.

## CPU

CPU statt GPU:

```bash
python3 export_stl.py --cpu
```

## Mehrere FreeCAD-Dateien

Wenn mehr als eine `.FCStd` in `in/` liegt:

```bash
python3 export_stl.py in/mein_modell.FCStd
```

## Ausgabe

Anderer STL-Pfad:

```bash
python3 export_stl.py --output out/mein_export.stl
```

## Qualitaet

```bash
python3 export_stl.py --preset draft     # schnell
python3 export_stl.py --preset standard  # mittel
python3 export_stl.py --preset fine      # default
```

| Preset | Profil-Samples | Rotationssegmente |
| --- | ---: | ---: |
| `draft` | 96 | 512 |
| `standard` | 384 | 4096 |
| `fine` | 768 | 8192 |

Direkt ueberschreiben:

```bash
python3 export_stl.py --samples 768 --segments 8192
```

## Weitere Flags

```text
--flip              Dreiecksorientierung umdrehen
--shape NAME        Shape im FCStd-Archiv
--open-profile      Profil nicht automatisch schliessen
```

Wenn die Oberflaeche innen statt aussen orientiert ist:

```bash
python3 export_stl.py --flip
```

Hinweis: PyTorch meldet bei ROCm intern `device: cuda`; das ist normal und meint den HIP-Backend-Pfad.
