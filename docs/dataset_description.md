# Playground Dataset Description

This document provides additional context for the surveillance dataset used in the
MP‑GCN playground adaptation.  The dataset was collected as part of a class
project at the **Centro para el Futuro de las Ciudades** (Tecnológico de Monterrey)
and captures natural interactions in an outdoor playground.

## Cameras and environment

- **Number of cameras:** 7 fixed surveillance cameras.
- **Location:** An outdoor playground within the university campus.  Each camera covers a different area; some areas overlap.
- **Data format:** Raw `.mp4` files, one per clip.  Each clip is typically a few seconds long and contains one or more children or adults engaged in activities.

## Annotation process

1. **Manual selection:**  From the continuous surveillance footage, 248 clips were selected to capture a variety of activities.  Only clips containing clear human activity were retained.
2. **Initial labelling:**  Each clip was annotated with one of six classes:
   - `Transit`: individuals moving from one point to another (e.g. walking across the playground)
   - `Social_People`: social interaction between people
   - `Play_Object_Normal`: normal play with objects (e.g. playing with a ball)
   - `Play_Object_Risk`: playing with objects in potentially unsafe ways
   - `Adult_Assisting`: an adult assisting a child
   - `Negative_Contact`: pushing, shoving or other negative interactions
3. **Class merging:**  The dataset is heavily imbalanced; most clips belong to the `Transit` class.  To mitigate this, the five non‑transit classes were merged into a single `Social` class.  The final dataset therefore contains **two classes**: `Transit` and `Social`.
4. **Train/validation split:**  The original MP‑GCN scripts require a pickle file (`annotations.pkl`) containing video IDs, labels and split assignments.  A notebook (`annotationsFormatting.ipynb`) is provided in the `src/` folder to help convert the raw annotations into this format.

## Skeleton extraction

Human poses are extracted from the raw videos using the **YOLO v8 pose model** provided by the [Ultralytics](https://github.com/ultralytics) library.  For each frame, up to six people are detected, and the XY coordinates of their keypoints are saved as NumPy arrays.  These skeletons are then used to build spatial‑temporal graphs for the MP‑GCN.

## Challenges and considerations

- **Occlusions and noise:** Surveillance cameras are mounted high and may occlude body parts.  Pose estimation can fail or produce missing keypoints, which need to be handled during preprocessing.
- **Lighting and weather:** The playground is outdoors; changes in lighting and weather can degrade detection quality.
- **Class imbalance:** The overwhelming majority of clips depict transit behaviour.  To obtain meaningful evaluation, the minor classes were combined into `Social`.

Despite these challenges, the dataset provides a realistic testbed for graph‑based activity recognition models and demonstrates the feasibility of applying MP‑GCN beyond curated benchmarks.