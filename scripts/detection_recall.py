"""How many annotated objects does the detector find? Compares cached detections with train_tracks.csv.

A ground-truth box counts as found in a frame if a detection of the same class overlaps it with
IoU >= 0.5. Two views:
- per box: share of annotated boxes found, by role and box height (small objects are the risk);
- per target track: share of moving targets found in at least 30% of their moving frames.
An object the detector never finds cannot be alarmed by anything downstream.

Usage: python scripts/detection_recall.py <detections_dir>   (only train videos present there are used)
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from whomoves.config import load_config  # noqa: E402
from whomoves.io import load_csv  # noqa: E402

IOU_MIN = 0.5
HEIGHT_BINS = [0, 30, 50, 100, 10_000]
HEIGHT_LABELS = ["<30px", "30-50px", "50-100px", ">100px"]


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between (n, 4) and (m, 4) boxes given as x, y, w, h."""
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 0] + a[:, None, 2], b[None, :, 0] + b[None, :, 2])
    y2 = np.minimum(a[:, None, 1] + a[:, None, 3], b[None, :, 1] + b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    union = a[:, None, 2] * a[:, None, 3] + b[None, :, 2] * b[None, :, 3] - inter
    return inter / np.maximum(union, 1e-9)


def match_video(gt: pd.DataFrame, det: pd.DataFrame) -> np.ndarray:
    """For each GT row, whether a same-class detection in the same frame has IoU >= IOU_MIN."""
    found = np.zeros(len(gt), dtype=bool)
    det_by_frame = {f: g for f, g in det.groupby("frame")}
    for f, g in gt.groupby("frame"):
        d = det_by_frame.get(f)
        if d is None:
            continue
        iou = iou_matrix(g[["x", "y", "w", "h"]].to_numpy(float), d[["x", "y", "w", "h"]].to_numpy(float))
        same_class = g["class"].to_numpy()[:, None] == d["class"].to_numpy()[None, :]
        found[gt.index.get_indexer(g.index)] = ((iou >= IOU_MIN) & same_class).any(axis=1)
    return found


def main() -> None:
    det_dir = Path(sys.argv[1])
    stride = json.loads((det_dir / "run_info.json").read_text()).get("frame_stride", 1)
    data_dir = load_config()["paths"]["data_dir"]
    tracks = load_csv(data_dir, "train_tracks")
    tracks = tracks[tracks["role"].isin(["target", "bonus", "static", "driver"]) & (tracks["frame"] % stride == 0)]

    parts = []
    for path in sorted(det_dir.glob("train_*.parquet")):
        vid = path.stem
        gt = tracks[tracks["video_id"] == vid].reset_index(drop=True)
        det = pd.read_parquet(path)
        gt = gt[gt["frame"] <= det["frame"].max()] if len(det) else gt.iloc[:0]
        parts.append(gt.assign(found=match_video(gt, det)))
    if not parts:
        sys.exit(f"no train_*.parquet in {det_dir}")
    r = pd.concat(parts, ignore_index=True)
    r["height"] = pd.cut(r["h"], HEIGHT_BINS, labels=HEIGHT_LABELS)

    print(f"{det_dir.name}: {len(parts)} train videos, {len(r)} annotated boxes (stride {stride})\n")
    print("Share of boxes found, by role and height (n boxes in brackets):")
    table = r.groupby(["role", "height"], observed=True)["found"].agg(["mean", "size"])
    print(table.apply(lambda x: f"{x['mean']:.2f} ({int(x['size'])})", axis=1).unstack("height").fillna("-").to_string())

    moving = r[r["role"].isin(["target", "bonus"]) & (r["moving"] == 1)]
    per_track = moving.groupby(["video_id", "track_id", "role", "class"])["found"].mean()
    ok = (per_track >= 0.3).groupby(level=["role", "class"]).agg(["mean", "size"])
    print("\nMoving targets found in >= 30% of their moving frames:")
    print(ok.rename(columns={"mean": "share", "size": "n_tracks"}).round(2).to_string())


if __name__ == "__main__":
    main()
