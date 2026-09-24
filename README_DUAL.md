# Dual YOLO Static Branding Tracker — CHASE + State Farm Right Basket

This is an extension of the existing **CHASE court-logo YOLO pipeline**. The CHASE model and its logic are kept separate and are not replaced with ORB or a new dual-ORB tracker.

## What this version does

In one pass over the same broadcast video it:

1. Runs the existing `models/chase_court_logo.pt` detector.
2. Runs a new YOLO detector trained specifically for the **State Farm logo on the selected physical right-side basket**.
3. Uses two independent `IntervalTracker` instances, so the time intervals cannot contaminate one another.
4. Uses the existing fast scene-cut detector; a camera cut closes both active intervals.
5. Tolerates short detector misses using `--max-missed`.
6. Writes two independent CSV reports:
   - `output/chase_intervals.csv`
   - `output/statefarm_intervals.csv`
7. Writes one simultaneous detection video:
   - `output/dual_tracking.mp4`
8. Writes one run summary:
   - `output/dual_summary.json`

## Important: how the State Farm target is defined

State Farm appears on both baskets and can also appear on LED/ribbon signage. Therefore the new training set must label **only the intended basket logo**. Do not label the opposite basket or LED-board State Farm graphics. Those become hard negatives/background for the detector.

Do not use a fixed screen-left/screen-right assumption when the broadcast changes camera angle. The default selector is `highest-conf`/model-driven; temporal continuity is used when multiple candidates occur. If your exact assessment definition means "right side of the current image", use `--statefarm-policy image-right`.

## Project additions

```text
dataset_statefarm/
├── raw/                         # 3+ extracted candidate images
├── labels/raw/                 # YOLO labels produced by annotation tool
├── images/train/
├── images/val/
├── labels/train/
├── labels/val/
└── data.yaml

models/
├── chase_court_logo.pt          # existing model — leave unchanged
└── statefarm_right_basket.pt    # new model after training

src/
├── extract_statefarm_frames.py
├── annotate_statefarm.py
├── prepare_statefarm_dataset.py
├── train_statefarm.py
└── analyze_dual.py

output/
├── chase_intervals.csv
├── statefarm_intervals.csv
├── dual_tracking.mp4
└── dual_summary.json
```

## Step-by-step PowerShell workflow

### 1. Enter the project

```powershell
cd C:\Users\<YOUR_USER>\Downloads\Frame_Tracker\Frame_Tracker
```

If you extracted the supplied ZIP elsewhere, use that folder instead.

### 2. Create/activate Python 3.11 64-bit environment

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For an NVIDIA GPU, install the appropriate CUDA-enabled PyTorch build if your existing environment does not already have one.

Verify:

```powershell
python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

### 3. Put the source video in the expected location

```text
data/source_video.mp4
```

The supplied ZIP intentionally does not contain the large source video; it contains the existing dataset/model/output artifacts. Do not create a fake video to satisfy the command.

### 4. Extract exactly three State Farm candidate frames

Start by extracting three frames at different points in the broadcast. This is only candidate generation; visually verify that they contain the **target physical basket**.

Example:

```powershell
python src/extract_statefarm_frames.py `
  --video data/source_video.mp4 `
  --output-dir dataset_statefarm/raw `
  --count 100 `
  --interval 10 `
  --start 900
```

For better training, use three different camera views. If the first three candidates are not useful, change `--start` and repeat. The goal is **three different views of the target basket**, not three consecutive near-identical frames.

### 5. Annotate only the target basket logo

```powershell
python src/annotate_statefarm.py `
  --images dataset_statefarm/raw `
  --labels dataset_statefarm/labels/raw
```

Controls:

- Drag with left mouse: draw the State Farm logo box.
- `s`: save and next.
- `n` or Space: negative frame; saves an empty label.
- `c`: clear box.
- `q`/Esc: quit.

**Critical annotation rule:** If the opposite basket or an LED State Farm logo is visible, leave it unlabeled. Only the intended physical right-side basket gets class `0`.

### 6. Split into YOLO train/validation folders


```powershell
python src/prepare_statefarm_dataset.py `
  --images dataset_statefarm/raw `
  --labels dataset_statefarm/labels/raw `
  --output dataset_statefarm `
  --val-count 38
```


### 7. Train the new State Farm YOLO model

```powershell
python src/train_statefarm.py `
  --data dataset_statefarm/data.yaml `
  --base-model yolo11n.pt `
  --epochs 60 `
  --batch 8 `
  --imgsz 960 `
  --device auto `
  --output models/statefarm_right_basket.pt
```

Do **not** retrain or replace `models/chase_court_logo.pt` unless you separately intend to improve the CHASE detector.

### 8. Run simultaneous tracking

```powershell
python src/analyze_dual.py `
  --video data/source_video.mp4 `
  --chase-model models/chase_court_logo.pt `
  --statefarm-model models/statefarm_right_basket.pt `
  --fps 10 `
  --chase-confidence 0.45 `
  --statefarm-confidence 0.35 `
  --max-missed 2 `
  --device auto `
  --output-dir output `
  --save-debug-video `
  --save-detection-frames
```

If by "right side" you specifically mean **right side of the current video frame**, add:

```powershell
--statefarm-policy image-right
```

Otherwise leave the default model-driven selection so camera changes do not hard-code screen coordinates.

### 9. Check the outputs

```powershell
Get-ChildItem output
```

You should get:

```text
output/chase_intervals.csv
output/statefarm_intervals.csv
output/dual_tracking.mp4
output/dual_summary.json
```

The two CSVs have the same interval schema as the original CHASE implementation:

```text
appearance_id,start_time,end_time,duration_seconds,x1,y1,x2,y2,width,height,average_confidence,detection_count
```

## Why two models instead of one combined model?

The existing CHASE detector is already validated in the supplied project and the requirement is to add State Farm without disturbing it. Two independent one-class YOLO models preserve that behavior:

```text
                    SAME VIDEO FRAME
                          │
              ┌───────────┴───────────┐
              ▼                       ▼
       CHASE YOLO model        State Farm YOLO model
              │                       │
       CHASE detections       Right-basket detections
              │                       │
       CHASE IntervalTracker   StateFarm IntervalTracker
              │                       │
       chase_intervals.csv    statefarm_intervals.csv
              └───────────┬───────────┘
                          ▼
                 dual_tracking.mp4
```

This is simultaneous tracking in a single pass, not an ORB-based dual tracker.

## Assessment alignment

The assessment asks for static logo detection, interval start/end/duration, bounding boxes, treatment of camera cuts/occlusion/zoom, a summary, and average processing time under one second per frame. The original project already implements these requirements for CHASE; this extension applies the same temporal interval mechanism independently to State Farm.

The original implementation's median bounding box is retained for each interval. The two-CSV requirement is an intentional extension of the single-asset assessment output, so each asset has an independently auditable timeline.


