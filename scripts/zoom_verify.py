"""Second look at each candidate alarm: crop around its box, enlarge the crop and run the detector again.

Why: most false alarms left are tiny boxes (median 1.9% of the frame height, ~14 px) with a mid detector score.
Detectors are much better on 50-100 px objects than below 30 px (experiments.md, detector comparison), so a
crop enlarged ~6x shows whether the object is really a person / vehicle. A real small person should be found
again with a high score; a shadow or a piece of background should not.

For every candidate (rule alarm, person 0.75 / others 2.5, confirm 5) of the tiled test-like clips (t90) and of
the test videos, two frames of its track: the alarm row and the middle row of the track. Crop: square centered
on the box, side max(CONTEXT x box side, MIN_SIDE) px, resized to CROP px. Output per crop: zoom_score = best
score of a same-class detection overlapping the box (IoU >= 0.3, in crop coordinates), zoom_any = same for any
class kept by class_map. One parquet per video in <cache_dir>/zoom_verify/ (resumable).

Usage: python scripts/zoom_verify.py [--limit-videos N]
"""

import argparse
import pickle
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import sweep_lookahead as sw  # noqa: E402
from whomoves.alarms import alarm_rows, rule_moving  # noqa: E402
from whomoves.config import load_config  # noqa: E402
from whomoves.detect import Detector  # noqa: E402
from whomoves.io import iter_frames, load_csv, video_path  # noqa: E402

KEY = ["video_id", "track_id"]
THETA_PERSON = 0.75
CONTEXT, MIN_SIDE, CROP = 4.0, 64, 384
BATCH = 4


def clip_offsets(clip_videos: pd.DataFrame) -> dict:
    """Clip id -> first frame in the original video (clips are consecutive from frame 0, see eval_clips.py)."""
    out = {}
    cv = clip_videos.assign(base=clip_videos["video_id"].str.split("@").str[0],
                            k=clip_videos["video_id"].str.split("@").str[1].astype(int)).sort_values(["base", "k"])
    for _, g in cv.groupby("base"):
        f0 = 0
        for cid, dur, fps in zip(g["video_id"], g["duration_s"], g["fps"]):
            out[cid] = f0
            f0 += int(round(dur * fps))
    return out


def requests(feats: pd.DataFrame, videos: pd.DataFrame, driver: float, offsets: dict | None) -> pd.DataFrame:
    """Two crops per candidate: its alarm row and the middle row of its track."""
    rows = alarm_rows(feats, rule_moving(feats, sw.THETA, THETA_PERSON), sw.CONFIRM, driver)
    tracks = feats.merge(rows[KEY], on=KEY).sort_values(KEY + ["t"], kind="stable")
    pos = tracks.groupby(KEY, sort=False).cumcount()
    size = tracks.groupby(KEY, sort=False)["t"].transform("size")
    mid = tracks[pos == size // 2]
    req = pd.concat([rows.assign(which="alarm"), mid.assign(which="mid")], ignore_index=True)
    req = req[KEY + ["which", "frame", "x", "y", "w", "h", "class"]].copy()
    req["base"] = req["video_id"].str.split("@").str[0]
    if offsets is not None:
        req["frame"] = req["frame"] + req["video_id"].map(offsets)
    return req


def crop_box(bgr: np.ndarray, x: float, y: float, w: float, h: float) -> tuple[np.ndarray, np.ndarray]:
    """Enlarged square crop around the box (RGB, CROP x CROP) and the box in crop coordinates (x0, y0, x1, y1)."""
    H, W = bgr.shape[:2]
    side = max(CONTEXT * max(w, h), MIN_SIDE)
    cx, cy = x + w / 2, y + h / 2
    x0 = int(np.clip(cx - side / 2, 0, max(W - side, 0)))
    y0 = int(np.clip(cy - side / 2, 0, max(H - side, 0)))
    x1, y1 = int(min(x0 + side, W)), int(min(y0 + side, H))
    crop = cv2.resize(bgr[y0:y1, x0:x1], (CROP, CROP), interpolation=cv2.INTER_CUBIC)
    sx, sy = CROP / (x1 - x0), CROP / (y1 - y0)
    box = np.array([(x - x0) * sx, (y - y0) * sy, (x + w - x0) * sx, (y + h - y0) * sy])
    return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), box


def iou(boxes: np.ndarray, box: np.ndarray) -> np.ndarray:
    iw = np.clip(np.minimum(boxes[:, 2], box[2]) - np.maximum(boxes[:, 0], box[0]), 0, None)
    ih = np.clip(np.minimum(boxes[:, 3], box[3]) - np.maximum(boxes[:, 1], box[1]), 0, None)
    inter = iw * ih
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) + (box[2] - box[0]) * (box[3] - box[1])
    return inter / np.maximum(area - inter, 1e-6)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit-videos", type=int)
    args = parser.parse_args()
    cfg = load_config()
    data_dir, cache, driver = cfg["paths"]["data_dir"], cfg["paths"]["cache_dir"], cfg["motion"]["driver_inside"]
    out_dir = cache / "zoom_verify"
    out_dir.mkdir(exist_ok=True)

    c = pickle.loads((cache / "clips90_t2x2.pkl").read_bytes())
    req = [requests(c["feats"], c["videos"], driver, clip_offsets(c["videos"]))]
    tv = load_csv(data_dir, "test_videos")
    ft = pd.read_parquet(cache / "features" / f"{sw.TILED}.parquet")
    req.append(requests(ft[ft["video_id"].isin(tv["video_id"])], tv, driver, None))
    req = pd.concat(req, ignore_index=True)
    bases = sorted(req["base"].unique())[: args.limit_videos]
    print(f"{len(req)} crops on {req['base'].nunique()} videos", flush=True)

    det_cfg = dict(cfg["detector"], score_threshold=0.05, min_size=CROP, max_size=CROP, tiles=None)
    torch.set_num_threads(8)
    detector = Detector(det_cfg)
    t0, done = time.time(), 0
    for base in bases:
        out = out_dir / f"{base}.parquet"
        if out.exists():
            continue
        r = req[req["base"] == base].reset_index(drop=True)
        need = r.groupby("frame").groups
        last = r["frame"].max()
        crops, boxes, idx = [], [], []
        for f, _, bgr in iter_frames(video_path(data_dir, base)):
            if f in need:
                for i in need[f]:
                    img, box = crop_box(bgr, *r.loc[i, ["x", "y", "w", "h"]].astype(float))
                    crops.append(img)
                    boxes.append(box)
                    idx.append(i)
            if f >= last:
                break
        zs, za = np.zeros(len(r)), np.zeros(len(r))
        for s in range(0, len(crops), BATCH):
            for j, box, d in zip(idx[s:s + BATCH], boxes[s:s + BATCH], detector.detect(crops[s:s + BATCH])):
                if len(d):
                    b = d[["x", "y", "w", "h"]].to_numpy(float)
                    b[:, 2:] += b[:, :2]
                    ov = iou(b, box) >= 0.3
                    za[j] = d["score"].to_numpy()[ov].max(initial=0.0)
                    zs[j] = d["score"].to_numpy()[ov & (d["class"] == r.loc[j, "class"]).to_numpy()].max(initial=0.0)
        r.assign(zoom_score=zs, zoom_any=za).to_parquet(out, index=False)
        done += len(r)
        el = time.time() - t0
        print(f"{base}: {len(r)} crops; total {done} in {el / 60:.1f} min ({done / max(el, 1):.2f} crops/s)", flush=True)
    print("done")


if __name__ == "__main__":
    main()
