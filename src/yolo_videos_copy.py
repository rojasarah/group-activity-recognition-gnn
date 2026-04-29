#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO Pose + ByteTrack (loop manual con OpenCV y persistencia de tracks)

Exporta por video:
    1) .npy -> poses normalizadas: poses.shape == (T, K_max, 17, 2)
    2) .mp4/.avi -> video pintado con esqueletos (sobre FONDO BLANCO) para QA visual

Requisitos:
    pip install ultralytics opencv-python numpy tqdm
"""

from __future__ import annotations
import os, re, glob
from typing import Dict, Iterable, Tuple, List

import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

# --------------------------------------
# 0) Constantes y utilidades
# --------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

# Índices COCO-17 en Ultralytics
IDX = {
    "L_SHOULDER": 5,
    "R_SHOULDER": 6,
    "L_HIP": 11,
    "R_HIP": 12,
}

def ensure_dir(path: str) -> None:
    """Crea carpeta si no existe (idempotente)."""
    os.makedirs(path, exist_ok=True)

def safe_stem(path: str) -> str:
    """Obtiene nombre base sin extensión ni espacios."""
    base = os.path.basename(path)
    base = re.sub(r"\s+", "_", base)
    return os.path.splitext(base)[0]

# -----------------------------------------------------------
# [AGREGADO] Abrir un VideoWriter robusto (fallback códecs)
# -----------------------------------------------------------
def open_video_writer(out_dir: str, stem: str, fps: float, size: Tuple[int, int]) -> Tuple[cv2.VideoWriter, str]:
    w, h = size
    candidates = [
        ("mp4v", ".mp4"),
        ("avc1", ".mp4"),
        ("XVID", ".avi"),
        ("MJPG", ".avi")
    ]

    for fourcc_name, ext in candidates:
        fourcc = cv2.VideoWriter_fourcc(*fourcc_name)
        out_path = os.path.join(out_dir, f"{stem}_pose{ext}")
        writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
        if writer.isOpened():
            print(f"[INFO] VideoWriter: codec={fourcc_name} → {out_path}")
            return writer, out_path

    raise RuntimeError("No se pudo abrir ningún VideoWriter (revisa códecs/ffmpeg de OpenCV).")


# ---------------------------------------
# 1) Normalización por persona
# ---------------------------------------
def normalize_pose_pelvis_torso(kps: np.ndarray) -> np.ndarray:
    """
    Normaliza keypoints por persona:
        - Traslada pelvis (centro caderas) a (0,0)
        - Escala por longitud de torso (centro hombros ↔ centro caderas)
    """
    if kps.size == 0:
        return kps

    out = kps.copy().astype(np.float32)
    L_HIP, R_HIP = IDX["L_HIP"], IDX["R_HIP"]
    L_SH,  R_SH  = IDX["L_SHOULDER"], IDX["R_SHOULDER"]

    for i in range(out.shape[0]):
        hip = (out[i, L_HIP] + out[i, R_HIP]) / 2.0
        sh  = (out[i, L_SH]  + out[i, R_SH])  / 2.0
        scale = float(np.linalg.norm(sh - hip))

        if scale < 1e-6:  # evita división por cero
            scale = 1e-6

        out[i] = (out[i]) / scale

    return out


# ---------------------------------------
# 2) Asignación estable ID → slot 0..K-1
# ---------------------------------------
def assign_track_ids_to_slots(
    track_ids: Iterable[int],
    K_max: int,
    prev_map: Dict[int, int],
) -> Tuple[np.ndarray, Dict[int, int]]:
    """Mapea IDs del tracker a slots fijos [0..K_max)."""

    used = set(prev_map.values())
    slots: List[int] = []
    next_free = 0
    new_map: Dict[int, int] = {}

    for tid in track_ids:
        tid = int(tid)
        if tid in prev_map and prev_map[tid] != -1:
            s = prev_map[tid]
        else:
            while next_free in used and next_free < K_max:
                next_free += 1
            s = next_free if next_free < K_max else -1
            if s != -1:
                used.add(s)
                next_free += 1

        if s != -1:
            new_map[tid] = s
            slots.append(s)

    return np.asarray(slots, dtype=np.int32), new_map


# ---------------------------------------
# 3) Proceso por video
# ---------------------------------------
def process_one_video(
    model: YOLO,
    video_path: str,
    npy_dir: str,
    viz_dir: str,
    K_max: int = 8,
    step: int = 2,
    imgsz: int = 736,
    conf: float = 0.45,
    iou: float = 0.70,
    tracker_cfg: str = "bytetrack.yaml",
) -> Tuple[str, str]:
    """
    Ejecuta tracking+pose por video:
        - Guarda .npy con poses normalizadas: (T, K_max, 17, 2)
        - Guarda video con esqueletos sobre FONDO BLANCO
    """

    stem = safe_stem(video_path)
    out_npz = os.path.join(npy_dir, f"{stem}.npz")
    out_mp4 = os.path.join(viz_dir, f"{stem}_pose.mp4")

    # (a) Abrir video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir: {video_path}")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    out_fps = fps / max(1, step)

    # (b) Abrir writer
    writer, out_mp4 = open_video_writer(viz_dir, stem, out_fps, (w, h))

    seq: List[np.ndarray] = []
    id2slot: Dict[int, int] = {}

    frame_i = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if step > 1 and (frame_i % step != 0):
            frame_i += 1
            continue

        # tracking+pose
        results_list = model.track(
            source=frame,
            persist=True,
            tracker=tracker_cfg,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            max_det=K_max * 2,
            verbose=False
        )
        r = results_list[0]

        # Fondo blanco
        white = np.full((h, w, 3), 255, dtype=np.uint8)
        try:
            annotated = r.plot(img=white, boxes=False, labels=False)
        except TypeError:
            annotated = r.plot(im=white, boxes=False, labels=False)

        writer.write(annotated)

        # ---- Si no hay detecciones
        if (r.keypoints is None) or (r.keypoints.xyn is None):
            seq.append(np.full((K_max, 17, 2), np.nan, dtype=np.float32))
            frame_i += 1
            continue

        # IDs del tracker
        if r.boxes is not None and r.boxes.id is not None:
            ids = r.boxes.id.cpu().numpy()
        else:
            ids = np.arange(r.keypoints.xyn.shape[0])

        # keypoints normalizados
        kps_img = r.keypoints.xyn.cpu().numpy()[:, :, :2].astype(np.float32)
        kps_norm = normalize_pose_pelvis_torso(kps_img)

        # Asignar slots
        slots, id2slot = assign_track_ids_to_slots(ids, K_max, id2slot)

        # Guardar frame K_max
        frame_k = np.full((K_max, 17, 2), np.nan, dtype=np.float32)
        for i, s in enumerate(slots):
            if 0 <= s < K_max:
                frame_k[s] = kps_norm[i]

        seq.append(frame_k)
        frame_i += 1

    cap.release()
    writer.release()

    # Guardar .npy + .npz
    poses = np.stack(seq, axis=0) if seq else np.zeros((0, K_max, 17, 2), dtype=np.float32)

    out_npy = out_npz.replace(".npz", ".npy")
    np.save(out_npy, poses)
    np.savez_compressed(out_npz, poses=poses)

    return out_npy, out_mp4


# ---------------------------------------
# 4) Orquestador por carpeta
# ---------------------------------------
def run_yolo_folder(
    videos_dir: str,
    out_root: str,
    K_max: int = 6,
    step: int = 2,
    model_name: str = "yolo11l-pose.pt",
    tracker_cfg: str = "bytetrack.yaml",
    imgsz: int = 736,
    conf: float = 0.45,
    iou: float = 0.70,
    pattern: str = "*.mp4",
) -> None:

    npy_dir = os.path.join(out_root, "npy")
    viz_dir = os.path.join(out_root, "viz")
    ensure_dir(npy_dir)
    ensure_dir(viz_dir)

    model = YOLO(model_name)
    videos = sorted(glob.glob(os.path.join(videos_dir, pattern)))

    for v in tqdm(videos, desc="Procesando videos"):
        try:
            process_one_video(
                model=model,
                video_path=v,
                npy_dir=npy_dir,
                viz_dir=viz_dir,
                K_max=K_max,
                step=step,
                imgsz=imgsz,
                conf=conf,
                iou=iou,
                tracker_cfg=tracker_cfg,
            )
        except Exception as e:
            print(f"[ERROR] {os.path.basename(v)} -> {e}")


# ---------------------------------------
# 5) Main
# ---------------------------------------
if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    videos_dir = os.path.join(BASE_DIR, "downloads")
    out_root   = os.path.join(BASE_DIR, "yolo_data_out")

    run_yolo_folder(
        videos_dir=videos_dir,
        out_root=out_root,
        K_max=6,
        step=1,
        model_name="yolo11l-pose.pt",
        tracker_cfg="bytetrack.yaml",
        imgsz=1280,
        conf=0.15,
        iou=0.70,
    )
