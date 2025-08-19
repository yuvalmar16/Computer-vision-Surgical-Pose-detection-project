#!/usr/bin/env python3
import os
import re
import glob
import cv2
import argparse
from pathlib import Path
from ultralytics import YOLO

# ==============================
# Defaults : overide with your paths
# ==============================
DEFAULT_BASE_DIR = Path("/home/student/A_project")
DEFAULT_VIDEO    = Path("/datashare/project/vids_test/4_2_24_A_1_small.mp4")

# Detection thresholds
IOU_THR  = 0.40
CONF_THR = 0.60

# Keypoint connections (for 5-keypoint skeleton)
CONNECTIONS = [(0, 2), (1, 2), (2, 3), (3, 4)]

# Class names
CLASS_NAMES = {
    0: "tweezers",
    1: "Needle Holder",
}

# Keypoint map (label + BGR color)
KP_MAP = {
    0: ("Left holder",  (0, 0, 255)),    # Red
    1: ("Right holder", (255, 0, 255)),  # Magenta
    2: ("Hinge center", (0, 255, 255)),  # Yellow
    3: ("Head",         (0, 255, 0)),    # Green
    4: ("Extra",        (255, 255, 0)),  # Cyan (if exists)
}


# ==============================
# Helpers
# ==============================
def natural_key(path: str):
    """Sort frame_00001.png < frame_00002.png < ..."""
    fname = os.path.basename(path)
    return [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', fname)]


def extract_frames(video_path: Path, frames_dir: Path, target_fps: int = 30) -> int:
    """Extract frames from video at target_fps."""
    frames_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if not src_fps or src_fps <= 0:
        # Fallback if FPS is unavailable
        src_fps = target_fps

    frame_interval = max(1, int(round(src_fps / target_fps)))
    print(f"[INFO] Video FPS: {src_fps:.2f}, saving every {frame_interval} frames.")

    frame_idx, saved_idx = 0, 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_interval == 0:
            out_file = frames_dir / f"frame_{saved_idx:05d}.png"
            cv2.imwrite(str(out_file), frame)
            saved_idx += 1
        frame_idx += 1

    cap.release()
    print(f"[INFO] Extracted {saved_idx} frames to {frames_dir}")
    return saved_idx


def annotate_frame(frame, results):
    """Draw detections + keypoints."""
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


def run_inference(model_path: Path, frames_dir: Path, out_dir: Path, out_video: Path, fps: int = 30, make_video: bool = True):
    """Run YOLO inference on frames and save annotated frames + (optional) video."""
    out_dir.mkdir(parents=True, exist_ok=True)

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = YOLO(str(model_path))

    frame_paths = []
    for p in ("*.png", "*.jpg", "*.jpeg"):
        frame_paths.extend(glob.glob(str(frames_dir / p)))
    frame_paths.sort(key=natural_key)

    if not frame_paths:
        raise FileNotFoundError(f"No frames found in {frames_dir}")

    vw = None
    if make_video:
        sample = cv2.imread(frame_paths[0])
        if sample is None:
            raise RuntimeError(f"Cannot read first frame: {frame_paths[0]}")
        H, W = sample.shape[:2]
        out_video.parent.mkdir(parents=True, exist_ok=True)
        vw = cv2.VideoWriter(str(out_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    for i, fp in enumerate(frame_paths, 1):
        frame = cv2.imread(fp)
        if frame is None:
            print(f"[WARN] Skipping unreadable frame: {fp}")
            continue

        res_list = model.predict(frame, verbose=False, iou=IOU_THR, conf=CONF_THR)
        for res in res_list:
            frame = annotate_frame(frame, res)

        out_path = out_dir / os.path.basename(fp)
        cv2.imwrite(str(out_path), frame)
        if vw is not None:
            vw.write(frame)

        if i % 50 == 0 or i == len(frame_paths):
            print(f"[INFO] Processed {i}/{len(frame_paths)} frames")

    if vw is not None:
        vw.release()
        print(f"[INFO] Saved video: {out_video}")


# ==============================
# Main
# ==============================
def parse_args():
    ap = argparse.ArgumentParser(description="Extract frames and run YOLO pose inference.")
    ap.add_argument("--base_dir",  type=Path, default=DEFAULT_BASE_DIR, help="Base project directory.")
    ap.add_argument("--video",     type=Path, default=DEFAULT_VIDEO,    help="Input video path.")
    ap.add_argument("--target_fps", type=int, default=30,               help="FPS to sample from video.")
    ap.add_argument("--make_video", action="store_true",                 help="Also export annotated MP4.")
    return ap.parse_args()


def main():
    args = parse_args()

    BASE_DIR   = args.base_dir
    VIDEO_PATH = args.video
    TARGET_FPS = args.target_fps
    MAKE_VIDEO = args.make_video

    # Derived paths
    FRAMES_DIR = BASE_DIR / "small_video_frames_30fps"
    OUT_DIR    = BASE_DIR / "output" / "frames_pred_pseudo"
    OUT_VIDEO  = BASE_DIR / "output" / "pred_video_pseudo.mp4"
    MODEL_PATH = BASE_DIR / "yolo11_runs" / "yolo11x_pose_retrain_with_pseudo4" / "weights" / "best.pt"

    # Step 1: extract frames
    extract_frames(VIDEO_PATH, FRAMES_DIR, TARGET_FPS)

    # Step 2: run inference
    run_inference(MODEL_PATH, FRAMES_DIR, OUT_DIR, OUT_VIDEO, TARGET_FPS, MAKE_VIDEO)


if __name__ == "__main__":
    main()
