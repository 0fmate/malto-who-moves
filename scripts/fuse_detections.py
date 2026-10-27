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
from torchvision.ops import batched_nms

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from whomoves.config import load_config  # noqa: E402
from whomoves.detect import CLASS_INDEX, COLUMNS  # noqa: E402


def fuse(base: pd.DataFrame, fine: pd.DataFrame, max_h: float, iou: float) -> pd.DataFrame:
    df = pd.concat([base, fine[fine["h"] < max_h]], ignore_index=True)
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
    args = parser.parse_args()
    det = load_config()["paths"]["cache_dir"] / "detections"
    out = det / f"{args.base}+{args.fine}_h{args.max_h:g}"
    out.mkdir(parents=True, exist_ok=True)
    info = json.loads((det / args.base / "run_info.json").read_text())
    (out / "run_info.json").write_text(json.dumps({**info, "fused_with": args.fine, "max_h": args.max_h}, indent=2))
    names = sorted({p.name for p in (det / args.base).glob("*.parquet")} & {p.name for p in (det / args.fine).glob("*.parquet")})
    n_base = n_out = 0
    for name in names:
        base, fine = pd.read_parquet(det / args.base / name), pd.read_parquet(det / args.fine / name)
        fused = fuse(base, fine, args.max_h, args.iou)
        fused.to_parquet(out / name, index=False)
        n_base, n_out = n_base + len(base), n_out + len(fused)
    print(f"{len(names)} videos: {n_base:,} base boxes -> {n_out:,} fused (+{(n_out / n_base - 1) * 100:.1f}%) -> {out.name}")


if __name__ == "__main__":
    main()
