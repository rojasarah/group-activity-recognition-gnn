"""
Run YOLO‑based pose estimation on raw videos and save skeletons to disk.

This script is a simplified wrapper around the Ultralytics YOLO implementation.
It detects up to ``--max_people`` individuals per frame and stores their 2D keypoints
as ``.npy`` files for subsequent graph construction.

Usage:
    python src/yolo_videos.py --input_dir path/to/raw/videos --output_dir path/to/yolo_outputs

You should customise this script according to your hardware and desired pose model.
"""

import argparse
import os
from pathlib import Path

import numpy as np
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    """Parse command‑line arguments."""
    parser = argparse.ArgumentParser(description="Run YOLO pose estimation on videos")
    parser.add_argument("--input_dir", required=True, type=str,
                        help="Directory containing raw .mp4 videos")
    parser.add_argument("--output_dir", required=True, type=str,
                        help="Directory to save .npy files with skeleton keypoints")
    parser.add_argument("--max_people", type=int, default=6,
                        help="Maximum number of people to detect per frame")
    parser.add_argument("--model", type=str, default="yolov8n-pose.pt",
                        help="Path to a YOLO pose model checkpoint")
    return parser.parse_args()


def run_pose_estimation(input_dir: str, output_dir: str, model_path: str, max_people: int) -> None:
    """Run pose estimation on all .mp4 files in ``input_dir`` and save outputs."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load YOLO pose model
    model = YOLO(model_path)

    for video_file in input_path.glob("*.mp4"):
        print(f"Processing {video_file}…")
        results = model(video_file)
        # Each ``results`` item corresponds to a frame; extract keypoints per person
        skeletons = []
        for frame in results:
            frame_keypoints = []
            for person_idx, person in enumerate(frame.keypoints):
                if person_idx >= max_people:
                    break
                # person.xy is Nx2 array of x,y coordinates
                frame_keypoints.append(person.xy.numpy())
            skeletons.append(frame_keypoints)
        # Save per‑video skeletons as NumPy array
        np.save(output_path / (video_file.stem + ".npy"), skeletons)
        print(f"Saved skeletons to {output_path / (video_file.stem + '.npy')}\n")


def main() -> None:
    args = parse_args()
    run_pose_estimation(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        model_path=args.model,
        max_people=args.max_people,
    )


if __name__ == "__main__":
    main()
