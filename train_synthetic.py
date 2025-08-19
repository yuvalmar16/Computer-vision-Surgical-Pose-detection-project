#!/usr/bin/env python3
import os
from pathlib import Path
from glob import glob
import yaml
from ultralytics import YOLO

# ==============================
# Paths
# ==============================
BASE = Path("/home/student/SYNTH_DATA/train_val_split")  # dataset (already split)


train_images = BASE / "train" / "images"
train_labels = BASE / "train" / "labels"
val_images   = BASE / "val"   / "images"
val_labels   = BASE / "val"   / "labels"






PROJECT_ROOT = Path("/home/student/A_project")
PROJECT_ROOT.mkdir(parents=True, exist_ok=True)

EXPORT_DIR   = PROJECT_ROOT / "yolo11_runs"     # where Ultralytics will write runs
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

CKPT_DIR     = PROJECT_ROOT / "checkpoints"     # cache for pretrained weights
CKPT_DIR.mkdir(parents=True, exist_ok=True)

DATA_YAML    = PROJECT_ROOT / "yolo11_pose.yaml"  # dataset yaml

# ==============================
# Sanity checks
# ==============================
def assert_dir(p: Path, desc: str):
    assert p.is_dir(), f" Missing {desc}: {p}"

# plain
assert_dir(train_images, "train images dir")
assert_dir(train_labels, "train labels dir")
assert_dir(val_images,   "val images dir")
assert_dir(val_labels,   "val labels dir")



# Optional: quick counts
def count_pairs(img_dir: Path, lbl_dir: Path):
    imgs = sorted(glob(str(img_dir / "*.*")))
    lbls = sorted(glob(str(lbl_dir / "*.txt")))
    print(f" {img_dir.parent.parent.name}/{img_dir.parent.name}: {len(imgs)} images, {len(lbls)} labels")

# plain
count_pairs(train_images, train_labels)
count_pairs(val_images,   val_labels)


# Optional: quick counts
def count_pairs(img_dir: Path, lbl_dir: Path):
    imgs = sorted(glob(str(img_dir / "*.*")))
    lbls = sorted(glob(str(lbl_dir / "*.txt")))
    print(f" {img_dir.parent.name}: {len(imgs)} images, {len(lbls)} labels")

count_pairs(train_images, train_labels)
count_pairs(val_images,   val_labels)

# ==============================
# Dataset YAML (2 classes, 5 keypoints)
# ==============================
yaml_data = {

    "train": [str(train_images)],
    "val":   [str(val_images)],
    "nc": 2,
    "names": ["tweezer", "needle holder"],  # class 0, class 1
    "kpt_shape": [5, 3],                     # 5 kps, each (x,y,v)
    "keypoint_line": [],                     # no skeleton lines for now
    "roboflow": None
}
with open(DATA_YAML, "w") as f:
    yaml.dump(yaml_data, f, sort_keys=False)
print(f"📄 Dataset YAML written to: {DATA_YAML}")


# ==============================
# Load base model
# ==============================
model_name = "yolo11x-pose.pt"
model_path = CKPT_DIR / model_name
if not model_path.exists():
    print(f" Downloading {model_name} to {model_path} ...")
    from ultralytics.utils.downloads import attempt_download_asset
    url = f"https://github.com/ultralytics/assets/releases/download/v0.0.0/{model_name}"
    model_path = Path(attempt_download_asset(url, str(model_path)))
    print(" Download complete.")

model = YOLO(str(model_path))
print(f" Loaded pose model: {model_path}")

# ==============================
# Train with your augmentations + loss weights
# ==============================
print(" Starting YOLOv11 Pose fine-tuning...")
results = model.train(
    data=str(DATA_YAML),      # dataset config
    device=0,                 # GPU 0
    imgsz=640,                # input size
    epochs=90,                # a bit longer than 60 as requested (more epochs)
    batch=8,                  # raise from 4 if GPU allows, else set back to 4
    workers=4,

    # ---- Loss weights ----
    box=7.5,
    pose=1.0,
    kobj=1.0,
    cls=0.20,
    dfl=1.0,

    # ---- Optimizer & LR schedule ----
    optimizer="SGD",
    lr0=0.01,
    lrf=0.01,         # final LR fraction for cosine
    cos_lr=True,
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=3,
    warmup_momentum=0.8,
    warmup_bias_lr=0.1,

    # ---- Data augmentation ----
    # Photometric
    hsv_h=0.015, hsv_s=0.60, hsv_v=0.55,
    # Geometric
    degrees=5.0, translate=0.08, scale=0.50, shear=2.0,
    fliplr=0.5, flipud=0.0,
    # Mix strategies
    mosaic=0.6, mixup=0.1, copy_paste=0.25,
    close_mosaic=10,   # disable mosaic in last 10 epochs

    # ---- Logging & output ----
    project=str(EXPORT_DIR),
    name="yolo11x_pose_tools_aug",
    pretrained=True,
    save=True,
    plots=True,
)

print(" Training finished.")
print(f" Runs saved under: {EXPORT_DIR / 'yolo11x_pose_tools_aug'}")

# ==============================
# Optional: Validate best.pt on val
# ==============================
best_w = EXPORT_DIR / "yolo11x_pose_tools_aug" / "weights" / "best.pt"
if best_w.exists():
    print(" Validating best checkpoint on val split...")
    model_best = YOLO(str(best_w))
    metrics = model_best.val(data=str(DATA_YAML), imgsz=640, split="val", device=0)
    print(metrics)
else:
    print("ℹ best.pt not found yet.")
