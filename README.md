# Football Speed Analytics (Python)

A student project that analyses football match video and produces per-player statistics:
tracking, distance covered, speed, heatmaps, player photos and a simple HTML report.

Built with Python, OpenCV, Ultralytics YOLO26 and the BoT-SORT tracker.

> **Honesty note.** Track IDs are tracking numbers, not real player identities.
> Distances are only in meters when the field is calibrated, and only valid
> while the camera does not move. If the scale is estimated, every output says so.

## What it does

| Step | Script | Output |
|---|---|---|
| 1. Detect and track people, ignore spectators outside the field | `src/main.py` | `output/players_detected.mp4`, `output/tracks.csv` |
| 2. Check tracking quality (lost frames, short IDs) | `src/check_tracks.py` | console table |
| 3. Calibrate the field: pixels → meters (homography) | `src/calibrate.py` | `output/calibration.json`, `output/calibration_check.png` |
| 4. Distance, average and max speed, heatmaps, trajectories | `src/analyze.py` | `output/stats.csv`, `output/players/player_N/*.png` |
| 5. Player photos, team by shirt colour, HTML report | `src/report.py` | `output/report.html` |

## How the measurements stay honest

- Distance is never summed across tracking gaps longer than 10 frames.
- Positions are smoothed (median over 0.4 s) so detector jitter does not add fake meters.
- Steps faster than 11 m/s are treated as tracking errors and not counted.
- Max speed is measured over 1-second windows.
- Tracks shorter than 5 seconds get `N/A` for speed.
- Detections outside the calibrated field are ignored.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

For a GPU, install PyTorch with CUDA from https://pytorch.org first.
Model weights (`yolo26s.pt`) are downloaded by Ultralytics automatically.

## Usage

Put a video at `videos/match.mp4`, then run from the project folder:

```bash
python src/calibrate.py    # click field points, type their meters
python src/main.py
python src/analyze.py
python src/report.py       # then open output/report.html
```

Settings (model, image size, field size, colours) are at the top of each script.

## Limitations

- Works best with a fixed or hovering camera (e.g. a drone). A moving broadcast camera needs per-frame calibration, which is not implemented yet.
- Players outside the camera view cannot be tracked.
- Team detection by shirt colour uses hue ranges that must be adjusted for each match.
- Referees on the field are labelled by colour only.
