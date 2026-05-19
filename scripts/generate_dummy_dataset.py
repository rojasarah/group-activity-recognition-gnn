"""
Generate a small dummy dataset for the playground MP‑GCN adaptation.

This script creates a minimal dataset under ``data/raw/playground_dataset``
so that the repository can be used out‑of‑the‑box for demonstration
purposes or as a portfolio artefact. The generated data does not
contain any real person information – it consists of randomly
generated skeleton keypoints and a few example annotation files to
illustrate how the full pipeline should look.

Running this script will create the following structure:

```
data/raw/playground_dataset/
├── yolo_outputs/
│   ├── video_001.npy
│   ├── video_002.npy
│   └── video_003.npy
├── annotations_raw.txt
├── objects.yaml
└── annotations.pkl
```

* ``annotations_raw.txt`` lists the original six class labels assigned to
  each example video. The labels are purely illustrative.
* ``objects.yaml`` contains dummy object locations for each camera. It
  demonstrates how real object annotations might be structured.
* ``yolo_outputs/*.npy`` files contain randomly generated 2D skeletons
  representing up to six people over ten frames. Each file holds a
  Python list of length 10, where each element is a list of detected
  persons for that frame. Each person is represented by a (N, 2)
  NumPy array of ``(x, y)`` coordinates. The number of keypoints per
  person (``N``) is fixed at 17, matching common human pose models.
* ``annotations.pkl`` is a serialized dictionary with keys
  ``'video_id'``, ``'video_name'``, ``'label'`` and ``'split'``.
  It maps each example video to its label and assigns a simple train
  or validation split (e.g. the first two videos are training data
  and the last one is validation data).

To run the script, execute:

```
python scripts/generate_dummy_dataset.py
```

Note that this dummy dataset is not meant for real model training.
It is provided to ensure that the repository contains a complete
dataset structure for demonstration, portfolio, and CI purposes.
"""

import os
import random
import pickle
from pathlib import Path
from typing import List, Dict

import numpy as np
import yaml


def generate_skeleton(num_frames: int = 10, num_people: int = 3) -> List[List[np.ndarray]]:
    """Generate a list of frames with random 2D keypoint arrays.

    Each frame contains up to ``num_people`` individuals. Each individual
    is represented by a NumPy array of shape (17, 2) with coordinates
    sampled from a uniform distribution in the range [0, 1].
    """
    frames: List[List[np.ndarray]] = []
    for _ in range(num_frames):
        frame = []
        for _ in range(num_people):
            keypoints = np.random.rand(17, 2).astype(np.float32)
            frame.append(keypoints)
        frames.append(frame)
    return frames


def create_yolo_outputs(output_dir: Path) -> List[str]:
    """Create dummy skeleton files and return a list of video names.

    The function generates three files named ``video_001.npy``,
    ``video_002.npy`` and ``video_003.npy`` with random skeletons.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    video_names = ["video_001", "video_002", "video_003"]
    for name in video_names:
        skeleton = generate_skeleton(num_frames=10, num_people=3)
        np.save(output_dir / f"{name}.npy", skeleton)
    return video_names


def create_annotations_raw(annotation_file: Path, video_names: List[str]) -> None:
    """Write a simple annotations_raw.txt file assigning classes to videos.

    The raw file maps each video name to one of the original six
    classes. We use two classes multiple times to demonstrate how
    multiple examples may share a label. The labels chosen here are
    arbitrary.
    """
    original_classes = [
        "Transit",
        "Social_People",
        "Play_Object_Normal",
        "Play_Object_Risk",
        "Adult_Assisting",
        "Negative_Contact",
    ]
    # Assign each video one of the above classes. Cycle through the
    # class list if there are more videos than classes.
    with annotation_file.open("w") as f:
        for idx, name in enumerate(video_names):
            label = original_classes[idx % len(original_classes)]
            f.write(f"{name}.mp4 {label}\n")


def create_objects_yaml(objects_file: Path) -> None:
    """Create a dummy objects.yaml file with example object annotations.

    The YAML structure consists of a list of cameras. Each camera has
    a list of objects; each object is given an ID and a pair of
    coordinates representing its approximate location.
    """
    objects_data = [
        {
            "camera": 1,
            "objects": [
                {"id": "swing", "location": [0.1, 0.5]},
                {"id": "slide", "location": [0.4, 0.2]},
                {"id": "climbing_hill", "location": [0.7, 0.8]},
            ],
        },
        {
            "camera": 2,
            "objects": [
                {"id": "bench", "location": [0.2, 0.4]},
                {"id": "trash_can", "location": [0.8, 0.3]},
            ],
        },
    ]
    with objects_file.open("w") as f:
        yaml.dump(objects_data, f)


def create_annotations_pkl(pkl_file: Path, video_names: List[str]) -> None:
    """Create a simple annotations.pkl file for the dummy dataset.

    The pickle file contains a dictionary with keys ``'video_id'``,
    ``'video_name'``, ``'label'`` and ``'split'``. Video IDs are
    integer indices, labels are simplified to two classes: 0 for
    ``Transit`` and 1 for ``Social`` (representing any other
    behaviour). The split assigns the first two videos to ``train``
    and the last one to ``val``.
    """
    data: Dict[str, List] = {
        "video_id": [],
        "video_name": [],
        "label": [],
        "split": [],
    }
    for idx, name in enumerate(video_names):
        data["video_id"].append(idx)
        data["video_name"].append(name)
        # Label: 0 for Transit, 1 for Social (covers all other behaviours)
        data["label"].append(0 if idx % 2 == 0 else 1)
        split = "train" if idx < 2 else "val"
        data["split"].append(split)
    with pkl_file.open("wb") as f:
        pickle.dump(data, f)


def main() -> None:
    # Determine root folder relative to this script
    repo_root = Path(__file__).resolve().parent.parent
    dataset_root = repo_root / "data" / "raw" / "playground_dataset"
    yolo_dir = dataset_root / "yolo_outputs"
    # Create skeletons and obtain video names
    video_names = create_yolo_outputs(yolo_dir)
    # Create annotations_raw.txt
    create_annotations_raw(dataset_root / "annotations_raw.txt", video_names)
    # Create objects.yaml
    create_objects_yaml(dataset_root / "objects.yaml")
    # Create annotations.pkl with train/val splits
    create_annotations_pkl(dataset_root / "annotations.pkl", video_names)
    print(f"Dummy dataset created at {dataset_root} with videos: {video_names}")


if __name__ == "__main__":
    main()