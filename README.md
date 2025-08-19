
# 2D Pose Estimation for Surgical Instruments Using Synthetic Data & Domain Adaptation

This project implements a **3-phase pipeline** to estimate **2D keypoints** on surgical instruments.  
The workflow combines **synthetic data generation** with **domain adaptation via pseudo-labeling** on real videos.

---

## 📌 Pipeline Overview
1. **Synthetic Data Generation** with BlenderProc (`synthetic_data_generator.py`)
2. **Train on Synthetic Data** (YOLO Pose)
3. **Generate Pseudo-Labels** on real videos (`create_pesudo_labels.py`)
4. **Retrain with Synthetic + Pseudo-Labeled Data** (`train_with_pesudo.py`)
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
Pose keypoints (5 per object) in YOLO format,
a visualized image with the 5 keypoints overlaid.

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
python train_with_synthetic.py
```
after this we evaluate on the video the link of results provided bellow:
```bash
  asssssss
```
## Phase 3: Pseudo-Labeling & Refinement
Run pseudo-label generation:
```bash
python3 create_pesudo_labels.py
```
Outputs:
```bash
/YOUR_DIR/PESUDO_LABELS/{images,labels}
```









