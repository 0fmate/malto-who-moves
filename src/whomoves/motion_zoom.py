"""Motion-guided zoom: look closer only where pixels move.

1. Background subtraction (OpenCV MOG2): every pixel keeps a small mixture of Gaussians describing
   its usual colors; a pixel far from all of them is foreground. Shadows (darker versions of the
   background) are flagged by MOG2 and dropped. The model updates over time, so parked cars become
   background, and leaves that shake for minutes are partly absorbed too.
2. The foreground mask is cleaned (morphological opening removes isolated noisy pixels, closing
   fills holes) and split into connected blobs; tiny blobs and huge ones (light changes) are dropped.
3. Around each blob a square crop several times its size is cut and enlarged by the detector, so a
   15 px person becomes ~80 px, the size COCO detectors are good at. Boxes touching the crop border
   are cut objects and are dropped; the rest are shifted back to frame coordinates.

The detector still decides WHAT the object is (person, vehicle, animal): motion alone would also
fire on leaves, rain and reflections. Motion only decides WHERE to look harder.
"""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .detect import COLUMNS, EDGE_PX, Detector


class BlobFinder:
    """Foreground blobs of a video, frame by frame (frames must be fed in order)."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.mog = cv2.createBackgroundSubtractorMOG2(history=cfg["history"], varThreshold=cfg["var_threshold"],
                                                      detectShadows=True)
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    def blobs(self, bgr: np.ndarray, learn: bool = True) -> np.ndarray:
        """(n, 4) blob boxes x1, y1, x2, y2, largest first."""
        mask = self.mog.apply(bgr, learningRate=-1 if learn else 0)
        mask = np.where(mask == 255, 255, 0).astype(np.uint8)  # 127 = shadow
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel, iterations=2)
        n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        stats = stats[1:]
        area = stats[:, cv2.CC_STAT_AREA]
        max_area = self.cfg["max_blob_area_frac"] * mask.shape[0] * mask.shape[1]
        keep = (area >= self.cfg["min_blob_area"]) & (area <= max_area)
        s = stats[keep][np.argsort(-area[keep])]
        x, y, w, h = s[:, 0], s[:, 1], s[:, 2], s[:, 3]
        return np.stack([x, y, x + w, y + h], axis=1) if len(s) else np.zeros((0, 4), int)


def crops_for(blobs: np.ndarray, width: int, height: int, cfg: dict) -> list[tuple[int, int, int, int]]:
    """Square crops around the blobs, at most max_crops, skipping blobs already inside a chosen crop."""
    crops = []
    for x1, y1, x2, y2 in blobs:
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        if any(a <= cx <= c and b <= cy <= d for a, b, c, d in crops):
            continue
        side = int(np.clip(cfg["crop_scale"] * max(x2 - x1, y2 - y1), cfg["min_crop"], cfg["max_crop"]))
        side = min(side, width, height)
        x0 = int(np.clip(cx - side / 2, 0, width - side))
        y0 = int(np.clip(cy - side / 2, 0, height - side))
        crops.append((x0, y0, x0 + side, y0 + side))
        if len(crops) == cfg["max_crops"]:
            break
    return crops


def detect_video_zoom(video_path: Path, video_id: str, detector: Detector, cfg: dict,
                      max_frames: int | None = None) -> pd.DataFrame:
    """Detections on motion crops every cfg["stride"] frames (MOG2 itself sees every frame)."""
    finder = BlobFinder(cfg)
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    chunks, idx, n_crops = [], 0, 0
    while True:
        ok, bgr = cap.read()
        if not ok or (max_frames is not None and idx >= max_frames):
            break
        blobs = finder.blobs(bgr)
        if idx % cfg["stride"] == 0 and idx >= cfg["warmup_frames"] and len(blobs):
            h, w = bgr.shape[:2]
            crops = crops_for(blobs, w, h, cfg)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            images = [np.ascontiguousarray(rgb[y0:y1, x0:x1]) for x0, y0, x1, y1 in crops]
            n_crops += len(images)
            for (x0, y0, x1, y1), (b, s, lab) in zip(crops, detector._predict_chunked(images)):
                side = x1 - x0
                cut = ((x0 > 0) & (b[:, 0] <= EDGE_PX)) | ((y0 > 0) & (b[:, 1] <= EDGE_PX)) \
                    | ((x1 < w) & (b[:, 2] >= side - EDGE_PX)) | ((y1 < h) & (b[:, 3] >= side - EDGE_PX))
                cls = np.array([detector.label_to_class.get(int(i)) for i in lab], dtype=object)
                keep = ~cut & (cls != None)  # noqa: E711
                if not keep.any():
                    continue
                bb = b[keep] + np.array([x0, y0, x0, y0], dtype=b.dtype)
                chunks.append(pd.DataFrame({
                    "video_id": video_id, "frame": idx, "t": round(idx / fps, 3),
                    "x": bb[:, 0], "y": bb[:, 1], "w": bb[:, 2] - bb[:, 0], "h": bb[:, 3] - bb[:, 1],
                    "score": s[keep], "class": cls[keep],
                    "label": [detector.labels[int(i)] for i in lab[keep]],
                }))
        idx += 1
    cap.release()
    df = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=COLUMNS)
    df = df[COLUMNS].astype({"frame": "int32", "t": "float32", "x": "float32", "y": "float32",
                             "w": "float32", "h": "float32", "score": "float32"})
    df.attrs["n_frames"] = idx
    df.attrs["n_crops"] = n_crops
    return df
