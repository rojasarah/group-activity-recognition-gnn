# Group Activity Recognition with MP-GCN and YOLO Pose Estimation

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-orange)
![Computer Vision](https://img.shields.io/badge/Computer%20Vision-Pose%20Estimation-purple)
![Status](https://img.shields.io/badge/Status-Portfolio%20Project-lightgrey)
![License](https://img.shields.io/badge/License-MIT-green)

This repository documents a portfolio adaptation of **MP-GCN** for **skeleton-based group activity recognition** using real-world playground surveillance videos.

The original MP-GCN model, introduced in the ECCV 2024 paper *"Skeleton-based Group Activity Recognition via Spatial-Temporal Panoramic Graph"*, was designed for benchmark group-activity datasets. In this project, the goal was to adapt the workflow to noisier real-world footage collected from **7 outdoor playground surveillance cameras**.

The project combines **YOLO-based human pose estimation**, skeleton keypoint extraction, dataset preprocessing, and an adapted MP-GCN workflow to classify group-level activities.

---

## 🎥 Visual Demo

This demo shows the **pose-estimation stage** of the pipeline. YOLO was used to detect people and extract 2D human skeleton keypoints from playground surveillance footage. These skeletons were later used as input features for the MP-GCN model.

[![YOLO Pose Estimation Demo](https://img.youtube.com/vi/wpbdV4CB5o4/0.jpg)](https://youtu.be/wpbdV4CB5o4)

---

## ⚙️ My Contributions

- Prepared a real-world playground video dataset for skeleton-based group activity recognition.
- Contributed to the manual selection and labeling of **248 surveillance video clips**.
- Implemented and tested a YOLO-based pose estimation pipeline to extract 2D human skeletons.
- Ran experiments with different pose-detection configurations to improve skeleton extraction quality.
- Tested detection strategies for distant people in wide surveillance-camera views.
- Adapted the dataset structure to make it compatible with the MP-GCN workflow.
- Reformulated the original 6 activity labels into a 2-class setting: **Transit** vs **Social**.
- Supported model training and evaluation, reaching **76.0% Top-1 Accuracy** and **70.83% MPCA**.

---

## 🔄 Pipeline

```text
Raw Surveillance Video
        ↓
YOLO Pose Estimation
        ↓
2D Skeleton Keypoint Extraction
        ↓
Dataset Formatting and Label Mapping
        ↓
Graph-Based Skeleton Representation
        ↓
MP-GCN Model
        ↓
Group Activity Classification
```

The pipeline converts real-world surveillance videos into skeleton-based graph representations suitable for group activity recognition.

---

## 📦 Dataset

The original team project used outdoor playground surveillance footage collected from **7 cameras**.

### Dataset characteristics

* **248 manually selected video clips**
* Real-world outdoor surveillance conditions
* Wide camera views with small or distant people
* Skeleton-based representation with up to **6 people per frame**
* Original 6 activity categories:

  * `Transit`
  * `Social_People`
  * `Play_Object_Normal`
  * `Play_Object_Risk`
  * `Adult_Assisting`
  * `Negative_Contact`

Due to class imbalance, the final classification task was reformulated into two broader activity groups:

```text
Transit vs Social
```

Because the real dataset is private, this repository includes a **small synthetic sample dataset** under:

```text
data/raw/playground_dataset/
```

The sample data allows the repository structure and demo pipeline to be inspected and executed without exposing the original surveillance dataset.

Additional dataset notes are available in [`docs/dataset_description.md`](./docs/dataset_description.md).

---

## 🤖 Model

The project is based on **MP-GCN**, a graph convolutional network architecture for skeleton-based group activity recognition.

### Model details

* **Architecture:** Multi-Person Panoramic Graph Convolutional Network
* **Task:** Skeleton-based group activity recognition
* **Input:** 2D human skeleton keypoints
* **Output:** Group activity class
* **Adapted classes:** `Transit` and `Social`

> This repository does **not** claim to re-implement MP-GCN from scratch.
> It documents the adaptation of an existing research pipeline to a real-world playground dataset, along with the preprocessing and demo structure used for portfolio review.

---

## 📈 Results

On the adapted real playground dataset, the team project obtained:

| Metric                  |     Result |
| ----------------------- | ---------: |
| Top-1 Accuracy          |  **76.0%** |
| Mean Per-Class Accuracy | **70.83%** |
| Mean Loss               |   **2.02** |

The best evaluation checkpoint reached **76.0% accuracy**. The accuracy curve also showed fluctuations across epochs, which is expected when working with a small, imbalanced, and noisy real-world video dataset.

![Evaluation Accuracy Across Epochs](assets/accuracy_curve.png)

---

## 🧪 Demo Scope

This repository includes a runnable demo using synthetic sample data.

The demo is intended to show:

* project structure
* dataset formatting
* configuration files
* simplified training and evaluation flow
* how skeleton-based data is organized for graph-based activity recognition

The demo does **not** reproduce the full private training pipeline or the original dataset results.

---

## 🗂️ Repository Structure

```text
group-activity-recognition-gnn/
│
├── README.md
├── main.py
├── requirements.txt
├── .gitignore
├── LICENSE
│
├── assets/
│   ├── pipeline_diagram.png
│   ├── sample_skeleton.png
│   └── accuracy_curve.png
│
├── config/
│   ├── mpgcn.yaml
│   └── playground_gendata.yaml
│
├── data/
│   └── raw/
│       └── playground_dataset/
│           ├── yolo_outputs/
│           │   ├── video_001.npy
│           │   ├── video_002.npy
│           │   └── video_003.npy
│           ├── annotations.pkl
│           ├── annotations_raw.txt
│           └── objects.yaml
│
├── docs/
│   └── dataset_description.md
│
├── results/
│   └── demo_metrics.json
│
├── scripts/
│   └── generate_dummy_dataset.py
│
└── src/
    └── yolo_videos.py
```

---

## 🚀 How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Optional: regenerate the sample dataset

```bash
python scripts/generate_dummy_dataset.py
```

### 3. Generate processed demo data

```bash
python main.py -c config/playground_gendata.yaml --generate_data
```

### 4. Run demo training

```bash
python main.py -c config/mpgcn.yaml
```

### 5. Run demo evaluation

```bash
python main.py -c config/mpgcn.yaml --evaluate
```

The commands above run a **portfolio demo** using synthetic sample data.

Full training on the original project requires the private dataset and the complete MP-GCN implementation.

---

## ⚠️ Limitations

* The original surveillance dataset is private and is not included in this repository.
* The runnable demo uses synthetic sample data to reproduce the expected project structure.
* The full MP-GCN training pipeline requires the original implementation and private dataset.
* Real-world surveillance footage introduced challenges such as distant people, noisy pose estimation, class imbalance, and limited labeled data.
* Evaluation accuracy fluctuated across epochs, likely due to the small and imbalanced dataset.

---

## 💡 Key Takeaways

This project demonstrates experience with:

* computer vision preprocessing for real-world video data
* human pose estimation using YOLO
* skeleton-based activity recognition
* graph neural network workflows
* dataset curation and label reformulation
* model evaluation under noisy and imbalanced data conditions
* adapting research code into a structured portfolio project

---

## 🔗 References

* Zhengcen Li, Xinle Chang, Yueran Li, Jingyong Su. *Skeleton-based Group Activity Recognition via Spatial-Temporal Panoramic Graph*. ECCV 2024.
* Original MP-GCN implementation: [https://github.com/mgiant/MP-GCN](https://github.com/mgiant/MP-GCN)
