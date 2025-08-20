#!/usr/bin/env python3
import os
from pathlib import Path
from glob import glob
import yaml
from ultralytics import YOLO

# ==============================
# Paths
# ==============================
PROJECT = Path("/home/student/A_project")

# existing datasets
SYNTH_BASE = Path("/home/student/SYNTH_DATA/train_val_split")


# your already-created pseudo labels
PSEUDO_ROOT      = PROJECT / "PSEUDO_LABELS"
PSEUDO_TRAIN_IMG = PSEUDO_ROOT / "images"
PSEUDO_TRAIN_LBL = PSEUDO_ROOT / "labels"

# runs and checkpoints
RUNS_DIR   = PROJECT / "yolo11_runs"
RUNS_DIR.mkdir(parents=True, exist_ok=True)

# choose starting weights
START_WEIGHTS = PROJECT / "yolo11_runs/yolo11x_pose_retrain_with_pseudo2/weights/best.pt"

# merged YAML path
MERGED_YAML = PROJECT / "yolo11_pose_merged_with_pseudo.yaml"

# ==============================
# Checks
# ==============================
def assert_dir(p: Path, desc: str):
    assert p.is_dir(), f"Missing {desc}: {p}"

# originals
assert_dir(SYNTH_BASE / "train" / "images", "SYNTH train images")
assert_dir(SYNTH_BASE / "train" / "labels", "SYNTH train labels")
assert_dir(SYNTH_BASE / "val"   / "images", "SYNTH val images")
assert_dir(SYNTH_BASE / "val"   / "labels", "SYNTH val labels")



# pseudo set
assert_dir(PSEUDO_TRAIN_IMG, "PSEUDO train images")
assert_dir(PSEUDO_TRAIN_LBL, "PSEUDO train labels")

assert START_WEIGHTS.exists(), f"Starting weights not found: {START_WEIGHTS}"

# quick counts
def count_pairs(img_dir: Path, lbl_dir: Path, tag: str):
    imgs = sorted(glob(str(img_dir / "*.*")))
    lbls = sorted(glob(str(lbl_dir / "*.txt")))
    print(f"{tag}: {len(imgs)} images, {len(lbls)} labels")

count_pairs(SYNTH_BASE / "train" / "images", SYNTH_BASE / "train" / "labels", "SYNTH train")

count_pairs(PSEUDO_TRAIN_IMG,                   PSEUDO_TRAIN_LBL,                   "PSEUDO train")
count_pairs(SYNTH_BASE / "val"   / "images",   SYNTH_BASE / "val"   / "labels",   "SYNTH val")


# ==============================
# Write merged YAML
# ==============================
yaml_cfg = {
    "train": [
        str(SYNTH_BASE / "train" / "images"),

        str(PSEUDO_TRAIN_IMG)
    ],
    "val": [
        str(SYNTH_BASE / "val" / "images"),
     
    ],
    "nc": 2,
    "names": ["tweezer", "needle holder"],
    "kpt_shape": [5, 3],
    "keypoint_line": [],
    "roboflow": None
}
with open(MERGED_YAML, "w") as f:
    yaml.dump(yaml_cfg, f, sort_keys=False)
print(f"\nMerged YAML written to: {MERGED_YAML}")

# ==============================
# Train
# ==============================
def main():
    model = YOLO(str(START_WEIGHTS))
    run_name = "yolo11x_pose_retrain_with_pseudo"

    print("\nStarting training on merged dataset: SYNTH + OCCLUDED + PSEUDO")
    results = model.train(
        data=str(MERGED_YAML),
        device=0,
        imgsz=640,
        epochs=20,          # change if you want longer
        batch=8,
        workers=4,

        # Loss weights
        box=7.5,
        pose=1.0,
        kobj=1.0,
        cls=0.20,
        dfl=1.0,

        # Optimizer and LR schedule
        optimizer="SGD",
        lr0=0.01,
        lrf=0.01,
        cos_lr=True,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,

        # Data augmentation
        hsv_h=0.015, hsv_s=0.60, hsv_v=0.55,
        degrees=5.0, translate=0.08, scale=0.50, shear=2.0,
        fliplr=0.5, flipud=0.0,
        mosaic=0.6, mixup=0.1, copy_paste=0.25,
        close_mosaic=10,

        # Logging and output
        project=str(RUNS_DIR),
        name=run_name,
        pretrained=True,
        save=True,
        plots=True,
    )

    best_w = RUNS_DIR / run_name / "weights" / "best.pt"
    print("\nTraining finished.")
    print(f"Run folder: {RUNS_DIR / run_name}")
    if best_w.exists():
        print(f"Best weights: {best_w}")
    else:
        print("Best weights not found under run folder")

if __name__ == "__main__":
    main()
