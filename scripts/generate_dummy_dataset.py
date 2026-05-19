from __future__ import annotations
import pickle
from pathlib import Path
import numpy as np
import yaml

def generate_skeleton(num_frames: int = 10, num_people: int = 3):
    frames = []
    for _ in range(num_frames):
        frame = []
        for _ in range(num_people):
            keypoints = np.random.rand(17, 2).astype(np.float32)
            frame.append(keypoints)
        frames.append(frame)
    return frames

def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    dataset_root = repo_root / "data" / "raw" / "playground_dataset"
    yolo_dir = dataset_root / "yolo_outputs"
    yolo_dir.mkdir(parents=True, exist_ok=True)

    video_names = ["video_001", "video_002", "video_003"]
    for name in video_names:
        np.save(yolo_dir / f"{name}.npy", generate_skeleton())

    (dataset_root / "annotations_raw.txt").write_text(
        "video_001.mp4 Transit\nvideo_002.mp4 Social_People\nvideo_003.mp4 Play_Object_Normal\n",
        encoding="utf-8"
    )

    objects_data = [
        {"camera": 1, "objects": [
            {"id": "swing", "location": [0.1, 0.5]},
            {"id": "slide", "location": [0.4, 0.2]}
        ]},
        {"camera": 2, "objects": [
            {"id": "bench", "location": [0.2, 0.4]},
            {"id": "trash_can", "location": [0.8, 0.3]}
        ]}
    ]
    with (dataset_root / "objects.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(objects_data, f)

    annotations = {
        "video_id": [0, 1, 2],
        "video_name": ["video_001", "video_002", "video_003"],
        "label": [0, 1, 0],
        "split": ["train", "train", "val"],
    }
    with (dataset_root / "annotations.pkl").open("wb") as f:
        pickle.dump(annotations, f)

    print(f"Dummy dataset created at {dataset_root}")

if __name__ == "__main__":
    main()
