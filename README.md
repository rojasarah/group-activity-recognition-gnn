# Group Activity Recognition using Graph Neural Networks (MP-GCN Adaptation)

This repository presents my portfolio adaptation of **MP-GCN** for **group activity recognition** using real-world surveillance data.

The original MP-GCN model, introduced in the ECCV 2024 paper *“Skeleton-based Group Activity Recognition via Spatial-Temporal Panoramic Graph”*, was developed for clean benchmark datasets. In this project, the goal was to adapt the pipeline to noisy, real-world playground footage collected from **7 surveillance cameras**.

This version is designed for portfolio review: it documents the real project, includes a small runnable demo, and preserves a clean structure without exposing the private dataset.

---

## My Contributions

- Built the pipeline from raw videos to structured skeleton data suitable for graph-based activity recognition
- Extracted human keypoints using a YOLO-based pose estimation pipeline
- Contributed to dataset curation and labeling for **248 selected clips**
- Reformulated the original **6-class** problem into a **2-class** setting (**Transit** vs **Social**) to address class imbalance
- Adapted preprocessing and data generation steps for compatibility with the MP-GCN workflow

---

## Pipeline

Raw Video → Pose Estimation (YOLO) → Skeleton Extraction → Graph Construction → MP-GCN → Classification

Assets illustrating the pipeline are included in [`assets/`](./assets).

---

## Dataset

The original project used surveillance footage from **7 cameras** in an outdoor playground environment.

Key characteristics:
- **248 manually selected video clips**
- **6 original classes**: `Transit`, `Social_People`, `Play_Object_Normal`, `Play_Object_Risk`, `Adult_Assisting`, `Negative_Contact`
- **2 final classes** after reformulation: `Transit` and `Social`
- Skeleton-based representation with up to **6 people per frame**

Because the real dataset is private, this repository includes a **small synthetic sample dataset** under `data/raw/playground_dataset/` so the structure and demo pipeline can be inspected and executed.

Additional dataset notes are available in [`docs/dataset_description.md`](./docs/dataset_description.md).

---

## Model

- **Architecture**: MP-GCN (Graph Convolutional Network)
- **Reference**: ECCV 2024 paper and official MP-GCN implementation
- **This repo**: focuses on the adaptation, data pipeline, and a runnable portfolio demo

> **Important:** This repository does **not** re-implement the full original MP-GCN training code.  
> It provides a clean, honest portfolio version with a minimal demo entry point (`main.py`) and sample data.

---

## Results from the Team Project

On the adapted playground dataset, the team project obtained:

- **Top-1 Accuracy:** 76.0%
- **Mean Per-Class Accuracy (MPCA):** 70.83%
- **Mean Loss:** 2.02

These metrics correspond to the real adapted project and are included here as project results, not as outputs of the synthetic demo dataset.

---

## Repository Structure

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
│   └── sample_skeleton.png
│
├── config/
│   ├── mpgcn.yaml
│   └── playground_gendata.yaml
│
├── data/raw/playground_dataset/
│   ├── yolo_outputs/
│   │   ├── video_001.npy
│   │   ├── video_002.npy
│   │   └── video_003.npy
│   ├── annotations.pkl
│   ├── annotations_raw.txt
│   └── objects.yaml
│
├── docs/
│   └── dataset_description.md
│
├── scripts/
│   └── generate_dummy_dataset.py
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

## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. (Optional) Regenerate the sample dataset

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

The commands above run a **portfolio demo** using the synthetic sample dataset.  
Full training on the original project requires the private dataset and the complete MP-GCN implementation.

---

## Why This Repository Exists

This repository is meant to show:
- my work adapting a research pipeline to real-world data
- how I handled dataset preparation and class imbalance
- my ability to structure an ML/CV project clearly for technical review

It is a portfolio artifact, not a release of the original private dataset.

---

## References

- Zhengcen Li, Xinle Chang, Yueran Li, Jingyong Su. *Skeleton-based Group Activity Recognition via Spatial-Temporal Panoramic Graph*. ECCV 2024.
- Original MP-GCN implementation: <https://github.com/mgiant/MP-GCN>
