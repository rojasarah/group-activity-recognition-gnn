# Group Activity Recognition using Graph Neural Networks (MP‑GCN Adaptation)

This repository contains my personal adaptation of the MP‑GCN model for **group activity recognition**.  
The original model, proposed in the 2024 ECCV paper *“Skeleton‑based Group Activity Recognition via Spatial‑Temporal Panoramic Graph”*, was designed for clean, benchmark datasets.  
Here the goal is to apply the same architecture to messy, **real‑world surveillance footage** collected from seven cameras in an outdoor playground.  
The project is part of my portfolio and demonstrates my ability to build an end‑to‑end machine‑learning pipeline, adapt research code, and handle noisy data.

---

## ⚙️ My Contributions

- **Data pipeline:** Constructed an automated pipeline that converts raw video files into structured skeleton data suitable for MP‑GCN.  
  This includes running a YOLO‑based pose‑estimation model to extract human keypoints and saving them as NumPy arrays.
- **Dataset curation:** Participated in selecting and annotating 248 video clips from seven cameras positioned around an outdoor playground.  
  We manually labelled the original six activity classes and later collapsed them into two (``Transit`` vs ``Social``) to address class imbalance.
- **Problem reformulation:** Adapted the classification problem from six classes to two classes because the ``Transit`` class dominated the dataset.  
  This improved the evaluation metrics while retaining the essence of group‑activity recognition.
- **Pre‑processing and augmentation:** Modified the original data‑generation scripts to work with the new dataset structure.  
  Added routines for graph construction, temporal augmentation, and balancing techniques.

Although the MP‑GCN architecture itself was not my invention, I integrated and tuned the model to work on this challenging dataset.

---

## 🧪 Pipeline

The following figure illustrates the end‑to‑end workflow implemented in this project.  
It starts with raw video files and ends with an MP‑GCN model producing class predictions:

```
Raw Video  →  Pose Estimation (YOLO)  →  Skeleton Extraction  →  Graph Construction  →  MP‑GCN  →  Classification
```

An image version of this pipeline can be found in the [`assets`](./assets) directory.

---

## 📦 Dataset

The data used in this project comes from **seven fixed surveillance cameras** covering different sections of an outdoor playground.  
Key details:

- **248 clips** manually selected and annotated.  
- **Multi‑camera**: each clip may contain overlapping fields of view; annotation is based on the dominant view.  
- **6 original classes**: ``Transit``, ``Social_People``, ``Play_Object_Normal``, ``Play_Object_Risk``, ``Adult_Assisting`` and ``Negative_Contact``.  
- **2 final classes**: to mitigate severe class imbalance, we merged the five non‑transit classes into a single ``Social`` class.  
- **Skeleton representation**: up to 6 people are detected per frame; each skeleton is represented by 2D keypoints.

A more detailed description of the dataset and its challenges is provided in [`docs/dataset_description.md`](./docs/dataset_description.md).  

---

## 🧠 Model

This project uses the **MP‑GCN** architecture – a Graph Convolutional Network designed to model interactions between multiple people over time.  
The implementation here is based on the official repository accompanying the ECCV 2024 paper.  
I have made the necessary adjustments to ingest our skeleton data and to accommodate the reduced set of classes.

---

## 📈 Results

When trained and evaluated on the curated playground dataset, the adapted MP‑GCN model achieves the following metrics:

- **Top‑1 Accuracy**: 76.0 %
- **Mean Per‑Class Accuracy (MPCA)**: 70.83 %
- **Mean Loss**: 2.02

Note that these numbers are specific to the reduced two‑class problem and the limited 248‑video dataset.

---

## ✨ Key Insights

- **Data quality matters**: applying state‑of‑the‑art architectures to real‑world footage requires significant effort in data cleaning and labelling.  
- **Class imbalance can dominate performance**: merging rare classes into broader categories was essential for meaningful evaluation.  
- **Graph‑based models shine on interaction data**: MP‑GCN effectively captures interactions between individuals when provided with reliable skeletons.

---

## 🛠️ Repository Structure

```
group‑activity‑recognition‑gnn/
│
├── README.md                # Project overview and usage instructions (this file)
├── requirements.txt         # Python dependencies
├── .gitignore               # Files and folders ignored by Git
├── LICENSE                  # Project licence (MIT)
│
├── config/                  # YAML configuration files for data generation and training
│   ├── mpgcn.yaml           # Training configuration (placeholder – customise for your setup)
│   └── playground_gendata.yaml # Data‑generation configuration (placeholder)
│
├── src/                     # Source code for the data pipeline
│   ├── yolo_videos.py       # Script to run pose estimation and save skeletons
│   └── annotationsFormatting.ipynb # Notebook for formatting annotations (optional)
│
├── assets/                  # Figures and visual aids
│   ├── pipeline_diagram.png # Visual representation of the pipeline
│   └── sample_skeleton.png  # Example of extracted skeletons
│
└── docs/
    └── dataset_description.md  # Additional notes on the dataset
```

> **Note**: the `config` files provided here are placeholders.  
> You should copy the original YAML files from the official MP‑GCN repository or your team’s repository and adjust them for your data paths.

---

## 🚀 Usage

1. **Install dependencies**

   From the project root, install the required Python packages:

   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare the dataset**

   Follow these high‑level steps to generate the skeleton data:

   ```bash
   # 1. Run pose estimation to extract skeletons
   python src/yolo_videos.py --input_dir path/to/raw_videos --output_dir path/to/yolo_outputs

   # 2. Format annotations and generate training/validation splits
   #    (see annotationsFormatting.ipynb for an example)

   # 3. Generate graph data using the config file
   python main.py -c config/playground_gendata.yaml --generate_data
   ```

   After running the last command, the processed data will be placed under a `data/playground` directory with `.npy` and `.pkl` files ready for training.

3. **Train the model**

   Once data generation is complete, train the MP‑GCN model using:

   ```bash
   python main.py -c config/mpgcn.yaml
   ```

4. **Evaluate**

   To evaluate a trained model checkpoint, run:

   ```bash
   python main.py -c config/mpgcn.yaml --evaluate
   ```

Please consult the original [MP‑GCN repository](https://github.com/mgiant/MP-GCN) for more details on configuration options and advanced usage.

---

## 📄 License

This project is licensed under the MIT License.  See the [LICENSE](./LICENSE) file for details.

---

## 🔗 References

- Zhengcen Li, Xinle Chang, Yueran Li, Jingyong Su. *Skeleton‑based Group Activity Recognition via Spatial‑Temporal Panoramic Graph*. ECCV 2024.  
- Official MP‑GCN implementation: <https://github.com/mgiant/MP-GCN>
