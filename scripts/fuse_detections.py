"""Fuse two detection runs: all boxes of a base run + only the small boxes of a finer-tiled run.

Finer tiles help small objects but cut big ones into jittery pieces (parked cars then seem to move).
Taking from the fine run only boxes shorter than --max-h, and merging per frame with class-aware NMS,
keeps the extra small objects without the jitter on big ones.

Output: <cache_dir>/detections/<base>+<fine>_h<max-h>/<video_id>.parquet (videos present in both runs).

Usage: python scripts/fuse_detections.py --base fasterrcnn_resnet50_fpn_v2_s1_t2x2 \\
                                         --fine fasterrcnn_resnet50_fpn_v2_s1_t3x3 [--max-h 40]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torchvision.ops import batched_nms, box_iou

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from whomoves.config import load_config  # noqa: E402
from whomoves.detect import CLASS_INDEX, COLUMNS  # noqa: E402


def novel(base: pd.DataFrame, fine: pd.DataFrame, max_overlap: float) -> pd.DataFrame:
    """Fine boxes that overlap no base box of the same frame (any class) by more than max_overlap IoU.

    Two sources disagreeing slightly on the box of one object make its track jitter (parked cars then
    seem to move); keeping only objects the base run missed avoids mixing sources on one object.
    """
    if fine.empty or base.empty:
        return fine
    keep = np.ones(len(fine), bool)
    base_by_frame = {f: g[["x", "y", "w", "h"]].to_numpy(np.float32) for f, g in base.groupby("frame")}
    for frame, idx in fine.groupby("frame").indices.items():
        b = base_by_frame.get(frame)
        if b is None:
            continue
        a = fine.iloc[idx][["x", "y", "w", "h"]].to_numpy(np.float32)
        a, b = a.copy(), b.copy()
        a[:, 2:] += a[:, :2]
        b[:, 2:] += b[:, :2]
        iou = box_iou(torch.from_numpy(a), torch.from_numpy(b)).numpy()
        keep[idx] = iou.max(axis=1) <= max_overlap
    return fine[keep]


def fuse(base: pd.DataFrame, fine: pd.DataFrame, max_h: float, iou: float,
         max_overlap: float | None = None) -> pd.DataFrame:
    fine = fine[fine["h"] < max_h]
    if max_overlap is not None:
        fine = novel(base, fine, max_overlap)
    df = pd.concat([base, fine], ignore_index=True)
    if df.empty:
        return df[COLUMNS]
    # One NMS over the whole video: frame and class together form the group, so boxes only compete
    # with boxes of the same class in the same frame.
    group = df["frame"].to_numpy(np.int64) * 10 + df["class"].map(CLASS_INDEX).to_numpy(np.int64)
    boxes = df[["x", "y", "w", "h"]].to_numpy(np.float32)
    boxes[:, 2:] += boxes[:, :2]
    keep = batched_nms(torch.from_numpy(boxes), torch.from_numpy(df["score"].to_numpy(np.float32)),
                       torch.from_numpy(group), iou).numpy()
    return df.iloc[np.sort(keep)][COLUMNS].reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--fine", required=True)
    parser.add_argument("--max-h", type=float, default=40)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--only-new", type=float, default=None, metavar="IOU",
                        help="keep only fine boxes overlapping no base box by more than this IoU")
    args = parser.parse_args()
    det = load_config()["paths"]["cache_dir"] / "detections"
    suffix = "" if args.only_new is None else f"_n{args.only_new:g}"
    out = det / f"{args.base}+{args.fine}_h{args.max_h:g}{suffix}"
    out.mkdir(parents=True, exist_ok=True)
    info = json.loads((det / args.base / "run_info.json").read_text())
    (out / "run_info.json").write_text(json.dumps({**info, "fused_with": args.fine, "max_h": args.max_h,
                                                   "only_new": args.only_new}, indent=2))
    names = sorted({p.name for p in (det / args.base).glob("*.parquet")} & {p.name for p in (det / args.fine).glob("*.parquet")})
    n_base = n_out = 0
    for name in names:
        base, fine = pd.read_parquet(det / args.base / name), pd.read_parquet(det / args.fine / name)
        fused = fuse(base, fine, args.max_h, args.iou, args.only_new)
        fused.to_parquet(out / name, index=False)
        n_base, n_out = n_base + len(base), n_out + len(fused)
    print(f"{len(names)} videos: {n_base:,} base boxes -> {n_out:,} fused (+{(n_out / n_base - 1) * 100:.1f}%) -> {out.name}")


if __name__ == "__main__":
    main()
