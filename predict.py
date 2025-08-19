#!/usr/bin/env python3
import os
import cv2
import argparse
from pathlib import Path
from ultralytics import YOLO

# ==============================
# Defaults 
# ==============================
DEFAULT_BASE_DIR = Path("/home/student/A_project")
DEFAULT_IMAGE    = Path("/home/student/A_project/small_video_frames_30fps/frame_00000.png")

# Detection thresholds
IOU_THR  = 0.40
CONF_THR = 0.60

# Class names
CLASS_NAMES = {
    0: "tweezers",
    1: "Needle Holder",
}

# 5-keypoint chain (adjust if needed)
CONNECTIONS = [(0, 2), (1, 2), (2, 3), (3, 4)]

# Keypoint map (label + BGR color); stable, no shuffling
KP_MAP = {
    0: ("Left holder",  (0, 0, 255)),    # Red
    1: ("Right holder", (255, 0, 255)),  # Magenta
    2: ("Hinge center", (0, 255, 255)),  # Yellow
    3: ("Head",         (0, 255, 0)),    # Green
    4: ("Extra",        (255, 255, 0)),  # Cyan (if exists)
}

def annotate_frame(frame, results):
    """Draw detections + keypoints (same style as your video pipeline)."""
    # Boxes
    if results.boxes is not None:
        for box, cls_id, conf in zip(results.boxes.xyxy, results.boxes.cls, results.boxes.conf):
            x1, y1, x2, y2 = map(int, box)
            cls_id = int(cls_id.item())
            label = CLASS_NAMES.get(cls_id, f"id{cls_id}")
            conf_val = float(conf.item())
            cv2.rectangle(frame, (x1, y1), (x2, y2), (208, 224, 64), 2)
            cv2.putText(frame, f"{label} {conf_val:.2f}",
                        (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0), 2)

    # Keypoints + skeleton
    if results.keypoints is not None:
        for kp_set in results.keypoints.xy:
            # points
            for idx, (x, y) in enumerate(kp_set):
                if x > 0 and y > 0:
                    _, color = KP_MAP.get(idx, (f"K{idx}", (200, 200, 200)))
                    cv2.circle(frame, (int(x), int(y)), 5, color, -1)
            # skeleton
            for (s, e) in CONNECTIONS:
                if s < len(kp_set) and e < len(kp_set):
                    x1, y1 = kp_set[s]
                    x2, y2 = kp_set[e]
                    if x1 > 0 and y1 > 0 and x2 > 0 and y2 > 0:
                        cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)
    return frame

def parse_args():
    ap = argparse.ArgumentParser(description="Run YOLO pose on a single image and save annotated copy.")
    ap.add_argument("--base_dir", type=Path, default=DEFAULT_BASE_DIR, help="Base project directory.")
    ap.add_argument("--image",    type=Path, default=DEFAULT_IMAGE,    help="Path to input image.")
    ap.add_argument("--out",      type=Path, default=None,             help="Output image path (optional).")
    ap.add_argument("--iou",      type=float, default=IOU_THR,         help="IOU threshold.")
    ap.add_argument("--conf",     type=float, default=CONF_THR,        help="Confidence threshold.")
    return ap.parse_args()

def main():
    args = parse_args()

    BASE_DIR   = args.base_dir
    IMG_PATH   = args.image
    IOU        = args.iou
    CONF       = args.conf

    # Model path (same structure as your video script)
    MODEL_PATH = BASE_DIR / "yolo11_runs" / "yolo11x_pose_retrain_with_pseudo4" / "weights" / "best.pt"
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    if not IMG_PATH.exists():
        raise FileNotFoundError(f"Image not found: {IMG_PATH}")

    OUT_PATH = args.out
    if OUT_PATH is None:
        # Save to base_dir/output/single_pred/<image_stem>_pred.png
        out_dir = BASE_DIR / "output" / "single_pred"
        out_dir.mkdir(parents=True, exist_ok=True)
        OUT_PATH = out_dir / f"{IMG_PATH.stem}_pred.png"

    # Load image
    frame = cv2.imread(str(IMG_PATH))
    if frame is None:
        raise RuntimeError(f"Failed to read image: {IMG_PATH}")

    # Load model and predict
    model = YOLO(str(MODEL_PATH))
    res_list = model.predict(frame, verbose=False, iou=IOU, conf=CONF)

    # Annotate
    for res in res_list:
        frame = annotate_frame(frame, res)

    # Save annotated image
    cv2.imwrite(str(OUT_PATH), frame)
    print(f"[INFO] Saved annotated image to: {OUT_PATH}")

if __name__ == "__main__":
    main()
