"""Pretrained object detector (no vision-language models).

Two backends, chosen with config["detector"]["name"]:
- torchvision Faster R-CNN (code BSD-3, COCO weights): a two-stage detector. A region proposal
  network suggests boxes on a feature pyramid, then a second head classifies and refines them.
  The feature pyramid and the high input resolution (~1333x750) help with small, far-away people.
- RT-DETR via Hugging Face transformers (Apache-2.0, COCO + Objects365 weights): a real-time
  transformer detector that predicts a fixed set of boxes directly, with no proposals and no NMS.
  Faster, but its default 640x640 input shrinks people that are already 20-40 px tall.

COCO labels are mapped to person / vehicle / animal with config["detector"]["class_map"];
anything else (birds included) is dropped.

Output: one row per detection with video_id, frame, t, x, y, w, h (pixels, top-left, like
train_tracks.csv), score, class (competition class) and label (original COCO label, e.g. "truck").
"""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

from .io import iter_frames

TORCHVISION_MODELS = {
    "fasterrcnn_resnet50_fpn_v2": "FasterRCNN_ResNet50_FPN_V2_Weights",
    "fasterrcnn_resnet50_fpn": "FasterRCNN_ResNet50_FPN_Weights",
    "fasterrcnn_mobilenet_v3_large_fpn": "FasterRCNN_MobileNet_V3_Large_FPN_Weights",
}
DEFAULT_RTDETR = "PekingU/rtdetr_r50vd_coco_o365"
COLUMNS = ["video_id", "frame", "t", "x", "y", "w", "h", "score", "class", "label"]


class Detector:
    """Wraps a pretrained detector: list of RGB frames in, filtered and mapped boxes out."""

    def __init__(self, cfg_det: dict, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.name = cfg_det["name"]
        self.threshold = float(cfg_det["score_threshold"])
        if self.name in TORCHVISION_MODELS:
            self._init_torchvision(cfg_det)
        elif self.name.startswith("rtdetr"):
            self._init_rtdetr(cfg_det)
        else:
            raise ValueError(f"unknown detector {self.name!r}")
        # COCO label id -> competition class (None = dropped).
        class_map = cfg_det["class_map"]
        self.label_to_class = {i: class_map.get(name) for i, name in self.labels.items()}

    def _init_torchvision(self, c: dict) -> None:
        from torchvision.models import detection

        weights = getattr(detection, TORCHVISION_MODELS[self.name]).DEFAULT
        self.labels = dict(enumerate(weights.meta["categories"]))
        kwargs = {"box_score_thresh": self.threshold, "box_detections_per_img": c.get("max_detections", 300)}
        if c.get("min_size"):  # resize so that the short side is min_size (default 800)
            kwargs.update(min_size=c["min_size"], max_size=c.get("max_size", 2000))
        self.model = getattr(detection, self.name)(weights=weights, **kwargs).eval().to(self.device)
        self.half = bool(c.get("half", True)) and self.device == "cuda"
        self._predict = self._predict_torchvision

    def _predict_torchvision(self, frames: list[np.ndarray]) -> list[tuple]:
        imgs = [torch.from_numpy(f).to(self.device).permute(2, 0, 1).float().div_(255) for f in frames]
        with torch.inference_mode(), torch.autocast(device_type=self.device.split(":")[0], enabled=self.half):
            out = self.model(imgs)
        return [(o["boxes"].float().cpu().numpy(), o["scores"].float().cpu().numpy(), o["labels"].cpu().numpy())
                for o in out]

    def _init_rtdetr(self, c: dict) -> None:
        from transformers import RTDetrForObjectDetection, RTDetrImageProcessor

        model_id = c.get("hf_model", DEFAULT_RTDETR)
        self.processor = RTDetrImageProcessor.from_pretrained(model_id)
        if c.get("input_size"):  # square input side, default 640
            self.processor.size = {"height": c["input_size"], "width": c["input_size"]}
        self.model = RTDetrForObjectDetection.from_pretrained(model_id).eval().to(self.device)
        self.labels = {int(i): name for i, name in self.model.config.id2label.items()}
        self._predict = self._predict_rtdetr

    def _predict_rtdetr(self, frames: list[np.ndarray]) -> list[tuple]:
        inputs = self.processor(images=frames, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            out = self.model(**inputs)
        sizes = torch.tensor([f.shape[:2] for f in frames], device=self.device)
        results = self.processor.post_process_object_detection(out, target_sizes=sizes, threshold=self.threshold)
        return [(r["boxes"].float().cpu().numpy(), r["scores"].float().cpu().numpy(), r["labels"].cpu().numpy())
                for r in results]

    def detect(self, frames: list[np.ndarray]) -> list[pd.DataFrame]:
        """Detect on a batch of RGB frames; one DataFrame (x, y, w, h, score, class, label) per frame."""
        results = []
        for boxes, scores, label_ids in self._predict(frames):
            classes = [self.label_to_class.get(int(i)) for i in label_ids]
            keep = np.array([c is not None for c in classes], dtype=bool)
            b = boxes[keep] if len(boxes) else np.zeros((0, 4), np.float32)
            results.append(pd.DataFrame({
                "x": b[:, 0], "y": b[:, 1], "w": b[:, 2] - b[:, 0], "h": b[:, 3] - b[:, 1],
                "score": scores[keep],
                "class": [c for c, k in zip(classes, keep) if k],
                "label": [self.labels[int(i)] for i, k in zip(label_ids, keep) if k],
            }))
        return results


def detect_video(video_path: Path, video_id: str, detector: Detector, stride: int = 1,
                 batch_size: int = 4, max_frames: int | None = None) -> pd.DataFrame:
    """Run the detector on every `stride`-th frame of one video; one row per detection."""
    chunks, batch, meta = [], [], []

    def flush():
        for (frame_idx, t), dets in zip(meta, detector.detect(batch)):
            chunks.append(dets.assign(video_id=video_id, frame=frame_idx, t=round(t, 3)))
        batch.clear()
        meta.clear()

    n_frames = 0
    for frame_idx, t, bgr in iter_frames(video_path, stride):
        if max_frames is not None and n_frames >= max_frames:
            break
        batch.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        meta.append((frame_idx, t))
        n_frames += 1
        if len(batch) == batch_size:
            flush()
    if batch:
        flush()

    df = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=COLUMNS)
    df = df[COLUMNS].astype({"frame": "int32", "t": "float32", "x": "float32", "y": "float32",
                             "w": "float32", "h": "float32", "score": "float32"})
    df.attrs["n_frames"] = n_frames
    return df
