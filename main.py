"""Minimal runnable entry point for the MP-GCN portfolio project.

This file does NOT re-implement the full ECCV 2024 MP-GCN training code.
Instead, it provides a lightweight demonstration pipeline so the repository
is coherent and runnable without private data.

Supported modes:
- --generate_data : reads the dummy/sample dataset and writes processed arrays
- train           : simulates a training pass and writes a demo metrics file
- --evaluate      : prints demo evaluation metrics

For full training, users should combine this repository with the original
MP-GCN implementation and the private dataset described in the README.
"""

from __future__ import annotations
import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import yaml


DEMO_METRICS = {
    "top1_accuracy": 0.76,
    "mpca": 0.7083,
    "mean_loss": 2.02,
    "mode": "demo"
}


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def generate_data(cfg: dict) -> None:
    dataset_root = Path(cfg["dataset_root_folder"])
    out_folder = Path(cfg["out_folder"])
    yolo_dir = dataset_root / "yolo_outputs"
    ann_path = dataset_root / "annotations.pkl"

    if not yolo_dir.exists():
        raise FileNotFoundError(f"Missing yolo_outputs directory: {yolo_dir}")
    if not ann_path.exists():
        raise FileNotFoundError(f"Missing annotations.pkl: {ann_path}")

    out_folder.mkdir(parents=True, exist_ok=True)

    with ann_path.open("rb") as f:
        annotations = pickle.load(f)

    names = annotations["video_name"]
    labels = annotations["label"]
    splits = annotations["split"]

    train_samples, eval_samples = [], []
    train_labels, eval_labels = [], []

    for name, label, split in zip(names, labels, splits):
        npy_path = yolo_dir / f"{name}.npy"
        sample = np.load(npy_path, allow_pickle=True)

        # Simple fixed-size feature representation for demo purposes:
        # mean keypoint coordinates across all people and frames.
        coords = []
        for frame in sample:
            for person in frame:
                arr = np.asarray(person, dtype=float)
                if arr.size:
                    coords.append(arr.mean(axis=0))
        if coords:
            feature = np.concatenate([np.mean(coords, axis=0), np.std(coords, axis=0)])
        else:
            feature = np.zeros(4, dtype=float)

        if split == "train":
            train_samples.append(feature)
            train_labels.append(label)
        else:
            eval_samples.append(feature)
            eval_labels.append(label)

    train_data = np.asarray(train_samples, dtype=float)
    eval_data = np.asarray(eval_samples, dtype=float)

    np.save(out_folder / "train_data.npy", train_data)
    np.save(out_folder / "eval_data.npy", eval_data)

    with (out_folder / "train_label.pkl").open("wb") as f:
        pickle.dump(train_labels, f)
    with (out_folder / "eval_label.pkl").open("wb") as f:
        pickle.dump(eval_labels, f)

    print(f"[OK] Generated demo data in: {out_folder}")
    print(f"      train_data shape: {train_data.shape}")
    print(f"      eval_data shape:  {eval_data.shape}")


def train_demo(cfg: dict) -> None:
    out_folder = Path(cfg["out_folder"])
    train_data_path = out_folder / "train_data.npy"
    if not train_data_path.exists():
        raise FileNotFoundError(
            "Processed data not found. Run --generate_data first."
        )

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    with (results_dir / "demo_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(DEMO_METRICS, f, indent=2)

    print("[OK] Demo training completed.")
    print(json.dumps(DEMO_METRICS, indent=2))


def evaluate_demo() -> None:
    results_path = Path("results") / "demo_metrics.json"
    if results_path.exists():
        with results_path.open("r", encoding="utf-8") as f:
            metrics = json.load(f)
    else:
        metrics = DEMO_METRICS

    print("[OK] Demo evaluation metrics:")
    print(json.dumps(metrics, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Demo runner for MP-GCN portfolio repo")
    parser.add_argument("-c", "--config", required=True, help="Path to YAML config")
    parser.add_argument("--generate_data", action="store_true", help="Generate demo processed data")
    parser.add_argument("--evaluate", action="store_true", help="Run demo evaluation")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    cfg = load_config(Path(args.config))

    if args.generate_data:
        generate_data(cfg)
    elif args.evaluate:
        evaluate_demo()
    else:
        train_demo(cfg)


if __name__ == "__main__":
    main()
