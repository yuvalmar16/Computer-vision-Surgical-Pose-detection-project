#!/usr/bin/env python3
import os
import glob
import shutil
from pathlib import Path
import cv2
from ultralytics import YOLO

# ========= paths =========

MODEL_PATH = "/home/student/A_project/yolo11_runs/yolo11x_pose_retrain_with_pseudo4/weights/best.pt"
FRAMES_DIR = "/home/student/A_project/small_video_frames_30fps"   # input images
PSEUDO_DIR = "/home/student/A_project/PSEUDO_LABELS"              # output root
IM_OUT     = Path(PSEUDO_DIR) / "images"
LBL_OUT    = Path(PSEUDO_DIR) / "labels"

# ========= inference settings =========
IOU_THR  = 0.60
CONF_THR = 0.80

# class id -> name (for logging only)
CLASS_NAMES = {
    0: "tweezer",
    1: "NH",
}

def natural_key(path: str):
    import re
    fname = os.path.basename(path)
    return [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', fname)]

def main():
    # prepare folders
    IM_OUT.mkdir(parents=True, exist_ok=True)
    LBL_OUT.mkdir(parents=True, exist_ok=True)

    # collect frames
    patterns = ["*.png", "*.jpg", "*.jpeg"]
    frame_paths = []
    for p in patterns:
        frame_paths.extend(glob.glob(os.path.join(FRAMES_DIR, p)))
    if not frame_paths:
        raise FileNotFoundError(f"No images found in {FRAMES_DIR}")
    frame_paths.sort(key=natural_key)

    # load model
    assert os.path.exists(MODEL_PATH), f"Model not found: {MODEL_PATH}"
    model = YOLO(MODEL_PATH)
    print(f"Loaded model: {MODEL_PATH}")
    print(f"Found {len(frame_paths)} images")

    saved_images = 0
    saved_labels = 0

    for i, fp in enumerate(frame_paths, 1):
        img = cv2.imread(fp)
        if img is None:
            print(f"[WARN] unreadable image, skipping: {fp}")
            continue

        # predict
        results_list = model.predict(img, verbose=False, iou=IOU_THR, conf=CONF_THR)
        if not results_list:
            continue
        res = results_list[0]

        # if there are no boxes, skip creating label
        if res.boxes is None or len(res.boxes) == 0:
            if i % 50 == 0:
                print(f"[{i}] no detections above conf {CONF_THR:.2f}")
            continue

        # write YOLO detection labels: "<cls> <xc> <yc> <w> <h>"
        stem = Path(fp).stem
        lbl_path = LBL_OUT / f"{stem}.txt"

        # gather detections above threshold
        lines = []
        for (xywhn, cls_t, conf_t) in zip(res.boxes.xywhn, res.boxes.cls, res.boxes.conf):
            conf = float(conf_t.item())
            if conf < CONF_THR:
                continue
            xc, yc, w, h = [float(v) for v in xywhn.tolist()]
            cls_id = int(cls_t.item())   # 0 = tweezer, 1 = NH
            lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")

        if not lines:
            if i % 50 == 0:
                print(f"[{i}] filtered out all detections < {CONF_THR:.2f}")
            continue

        # save label
        with open(lbl_path, "w") as f:
            f.write("\n".join(lines) + "\n")
        saved_labels += 1

        # copy image into pseudo set
        dst_img = IM_OUT / Path(fp).name
        if not dst_img.exists():
            shutil.copy(fp, dst_img)
            saved_images += 1

        if i % 50 == 0 or i == len(frame_paths):
            print(f"[{i}] wrote {len(lines)} dets -> {lbl_path.name}")

    print("\nDone.")
    print(f"Images saved: {saved_images}  -> {IM_OUT}")
    print(f"Label files:  {saved_labels}  -> {LBL_OUT}")
    print("Label format: '<class> <xc> <yc> <w> <h>' normalized to [0,1]")

if __name__ == "__main__":
    main()
