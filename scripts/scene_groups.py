"""Group the training videos by scene, so that validation folds never share a scene.

The data has no scene column, and many training clips come from the same parking lots. For each
video we build a "background": the per-pixel median of frames sampled across the whole clip (moving
objects disappear), downscaled and normalized for brightness. Two videos are the same scene if their
backgrounds correlate strongly, also allowing a horizontal mirror. Average-linkage clustering on
1 - correlation then gives the groups. Fully automatic: no video is labeled by hand.

Output: <cache_dir>/scene_groups.csv (video_id, scene) and outputs/scene_groups.jpg (thumbnails per scene,
to check the grouping by eye).

Usage: python scripts/scene_groups.py [--threshold 0.5] [--frames 15]
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from whomoves.config import load_config  # noqa: E402
from whomoves.io import load_csv, video_path  # noqa: E402

SIZE = (96, 54)  # downscaled background, width x height


def background(path: Path, n_frames: int) -> tuple[np.ndarray, np.ndarray]:
    """Median of n_frames frames spread over the video: (normalized gray thumbnail, color thumbnail)."""
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for idx in np.linspace(0, max(total - 1, 0), n_frames).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if ok:
            frames.append(cv2.resize(frame, SIZE, interpolation=cv2.INTER_AREA))
    cap.release()
    color = np.median(np.stack(frames), axis=0).astype(np.uint8)
    gray = cv2.GaussianBlur(cv2.cvtColor(color, cv2.COLOR_BGR2GRAY).astype(np.float32), (3, 3), 0)
    return (gray - gray.mean()) / (gray.std() + 1e-6), color


def correlation_matrix(bgs: list[np.ndarray]) -> np.ndarray:
    """Pearson correlation between backgrounds, the best of direct and mirrored."""
    flat = np.stack([b.ravel() for b in bgs])
    flat_mirror = np.stack([b[:, ::-1].ravel() for b in bgs])
    n = flat.shape[1]
    return np.maximum(flat @ flat.T, flat @ flat_mirror.T) / n


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0.5, help="max 1 - correlation inside a scene")
    parser.add_argument("--frames", type=int, default=15)
    args = parser.parse_args()

    cfg = load_config()
    data_dir = cfg["paths"]["data_dir"]
    videos = load_csv(data_dir, "train_videos")
    # Backgrounds are slow to compute (seeking in mp4 files): cache them, so the threshold can be retuned instantly.
    bg_cache = cfg["paths"]["cache_dir"] / "scene_backgrounds.npz"
    cached = dict(np.load(bg_cache)) if bg_cache.exists() else {}
    for v in videos["video_id"]:
        if f"{v}_gray" not in cached:
            cached[f"{v}_gray"], cached[f"{v}_color"] = background(video_path(data_dir, v), args.frames)
    np.savez_compressed(bg_cache, **cached)
    bgs = [cached[f"{v}_gray"] for v in videos["video_id"]]
    thumbs = [cached[f"{v}_color"] for v in videos["video_id"]]

    corr = np.clip(correlation_matrix(list(bgs)), -1, 1)
    dist = 1 - corr
    np.fill_diagonal(dist, 0)
    tree = linkage(squareform(dist, checks=False), "average")
    labels = fcluster(tree, args.threshold, "distance")
    print(f"distances of the last 8 merges (a big jump separates scenes): {tree[-8:, 2].round(2)}")
    out = pd.DataFrame({"video_id": videos["video_id"], "scene": labels,
                        "source": videos["source_dataset"], "minutes": videos["duration_s"] / 60})

    nearest = np.where(np.eye(len(corr), dtype=bool), -1, corr).max(axis=1)
    print(f"nearest-neighbour correlation: quantiles 10/50/90% = {np.quantile(nearest, [.1, .5, .9]).round(2)}")
    sizes = out.groupby("scene").agg(videos=("video_id", "size"), minutes=("minutes", "sum"),
                                     sources=("source", lambda s: ",".join(sorted(set(s)))))
    print(f"{len(sizes)} scenes at threshold {args.threshold}; videos per scene: "
          f"{sorted(sizes['videos'].tolist(), reverse=True)}")
    print(sizes.sort_values("minutes", ascending=False).head(8).round(1).to_string())

    out_csv = cfg["paths"]["cache_dir"] / "scene_groups.csv"
    out[["video_id", "scene"]].to_csv(out_csv, index=False)

    # Contact sheet: one row per scene (up to 8 thumbnails), to check the grouping by eye.
    rows = []
    for scene, g in out.groupby("scene"):
        tiles = [thumbs[i] for i in g.index[:8]]
        tiles += [np.zeros_like(thumbs[0])] * (8 - len(tiles))
        row = np.hstack(tiles)
        cv2.putText(row, str(scene), (2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        rows.append(row)
    sheet_path = cfg["paths"]["output_dir"] / "scene_groups.jpg"
    sheet_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(sheet_path), np.vstack(rows))
    print(f"-> {out_csv}\n-> {sheet_path}")


if __name__ == "__main__":
    main()
