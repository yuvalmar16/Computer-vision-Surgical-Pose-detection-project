
# 2D Pose Estimation for Surgical Instruments Using Synthetic Data & Domain Adaptation

This project implements a **3-phase pipeline** to estimate **2D keypoints** on surgical instruments.  
The workflow combines **synthetic data generation** with **domain adaptation via pseudo-labeling** on real videos.

---

## 📌 Pipeline Overview
1. **Synthetic Data Generation** with BlenderProc (`synthetic_data_generator.py`)
2. **Train on Synthetic Data** (`train_synthetic.py`)
3. **Generate Pseudo-Labels** on real videos (`create_pesudo_labels.py`)
4. **Retrain with Synthetic + Pseudo-Labeled Data** (`domain_adoption.py`)
5. **Inference** on images & videos (`predict.py`, `video.py`)

---

## ⚙️ Installation

### 1. Environment Setup
```bash
# clone repo
git clone https://github.com/yuvalmar16/Computer-vision-Surgical-Pose-detection-project.git
cd surgical-2D-pose

# create environment
conda create -n synth python=3.10
conda activate synth
pip install blenderproc

# install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```
## Phase 1: Synthetic data generation
We render images of needle holders and tweezers, place random clutter & lights, optionally add white occluders, and export:
RGB JPEGs, HDF5 (BlenderProc outputs),
a visualized image with the 5 keypoints overlaid.

🔹 Keypoint Generation
The ground truth (GT) keypoints are generated automatically from the synthetic segmentation masks, not manually labeled.

For each tool:

- Extract its layout from the segmentation map.

- Find the two farthest points → endpoints.

- Find one perpendicular extreme point → lateral side.

- Run KMeans (k=2) on the contour → two midpoints.

That gives 5 keypoints per instrument, saved in YOLO-pose format with bounding box + keypoints.

Commands:
```bash
# Non-occluded synthetic set (choose how many images to create with -n)
blenderproc run  synthetic_data_generator.py -n 1000
```

Outputs:
```bash
/YOUR_DIR/SYNTH_DATA/train_val_split/{train,val}/{images,labels}
```
The generator already handles keypoint extraction and train/val split building.


## Phase 2: Train on Synthetic Data
Train on the sythetic data set using yolo11x-pose
```bash
python3 train_with_synthetic.py
```

## Phase 3: Pseudo-Labeling & Refinement
1.Run pseudo-label generation:
```bash
python3 create_pesudo_labels.py
```
Outputs:
```bash
/YOUR_DIR/PESUDO_LABELS/{images,labels}
```
2.Retrain with merged dataset (synthetic + pseudo):

```bash
python3 domain_adoption.py

```


## Image inference

Update paths inside predict.py:
```bash
model_path = "weights/best.pt"
image_path = "sample.png"
output_image_path = "output.jpg"
```

Run:
```bash
python3 predict.py
```



## Video inference

Update paths inside video.py:
```bash

model_path = "weights/best.pt"
video_path = "input.mp4"
output_video_path = "output.mp4"
```

Run:
```bash
python3 video.py

```
After this we evaluate on the video the link of results provided with model weights bellow:
[Surigal_2D_Pose_Google Drive Folder](https://drive.google.com/drive/folders/1B4zjFWaf5tngw3oOqnTNtEHWG6FRDidf?usp=sharing)
