# Group Activity Recognition using Graph Neural Networks (MP-GCN Adaptation)

This project adapts a state-of-the-art Graph Convolutional Network (MP-GCN, ECCV 2024) for group activity recognition using real-world surveillance data.

Unlike the original implementation, which operates on clean benchmark datasets, this work focuses on applying the model to noisy, real-world data collected from multiple cameras in an outdoor environment.

---

## My Contributions

- Built the data pipeline from raw videos to structured skeleton data
- Extracted human keypoints using pose estimation (YOLO-based)
- Curated and labeled a custom dataset (248 videos)
- Reformulated the classification problem to handle class imbalance (6 → 2 classes)
- Adapted preprocessing and data generation pipeline

---

## Pipeline

Raw Video → Pose Estimation → Skeleton Extraction → Graph Construction → MP-GCN → Classification

---

## Dataset

- Multi-camera surveillance videos (outdoor playground)
- 248 manually selected samples
- Original 6 classes reduced to 2 (Transit vs Social)
- Skeleton-based representation

---

## Model

- MP-GCN (Graph Convolutional Network)
- Based on ECCV 2024 paper

---

## Results

- Top-1 Accuracy: 76.0%
- MPCA: 70.83%
- Mean Loss: 2.02

---

## Key Insight

Applying state-of-the-art models to real-world data requires significant effort in data preparation, cleaning, and problem reformulation.

Model performance depends heavily on dataset quality, not just architecture.

---

## How to Run

```bash
pip install -r requirements.txt
python main.py -c config/playground/mpgcn.yaml
