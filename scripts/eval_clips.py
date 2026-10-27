"""Validate on test-like clips: cut every training video into 35-120 s pieces and score each piece alone.

Test videos are 1-3 minute cuts (35-120 s here) that may start while something already moves. On a cut:
- our tracks restart at the clip start, so "distance from the first position" restarts too;
- ground truth is re-derived from train_tracks.csv inside the clip: a target is a target of the clip
  if it is marked moving in at least one clip frame (t_move_start clipped to the clip start); a target
  that only moves after the clip ends is still in the clip, so it becomes "static" (alarming it costs 2).

Unknown: how the organizers treat a target that moved BEFORE the clip and stands still inside it
(`moving` stays 1 = "has started moving"). Variant "strict" keeps it as a target; variant "lenient" makes
targets that move less than MIN_CLIP_MOVE box heights inside the clip neutral (ambiguous).

Usage: python scripts/eval_clips.py [--det-run fasterrcnn_resnet50_fpn_v2_s1] [--videos ...] [--seed 0]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from whomoves.alarms import make_alarms, rule_moving  # noqa: E402
from whomoves.config import load_config  # noqa: E402
from whomoves.features import track_features  # noqa: E402
from whomoves.io import load_csv  # noqa: E402
from whomoves.metric import GroundTruth, score  # noqa: E402

MIN_LEN_S, MAX_LEN_S = 35.0, 120.0
MIN_CLIP_MOVE = 0.5  # box heights


def clip_bounds(n_frames: int, fps: float, rng: np.random.Generator) -> list[tuple[int, int]]:
    """Consecutive [start, end) frame ranges of random length; a tail shorter than MIN_LEN_S is dropped."""
    bounds, start = [], 0
    while (n_frames - start) / fps >= MIN_LEN_S:
        end = min(n_frames, start + int(rng.uniform(MIN_LEN_S, MAX_LEN_S) * fps))
        bounds.append((start, end))
        start = end
    return bounds or [(0, n_frames)]


def cut(df: pd.DataFrame, f0: int, f1: int, fps: float, clip_id: str) -> pd.DataFrame:
    part = df[(df["frame"] >= f0) & (df["frame"] < f1)].copy()
    part["frame"] -= f0
    part["t"] = part["frame"] / fps
    part["video_id"] = clip_id
    return part


def clip_targets(gt_rows: pd.DataFrame, targets: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Targets of one clip (+ how far each moves inside it) and the clip rows with updated roles."""
    tids = set(targets["track_id"])
    movers = gt_rows[gt_rows["track_id"].isin(tids)]
    rows = []
    for tid, g in movers.groupby("track_id"):
        m = g[g["moving"] == 1]
        if m.empty:
            continue
        c = np.c_[g["x"] + g["w"] / 2, g["y"] + g["h"] / 2]
        move = np.linalg.norm(c - c[0], axis=1).max() / max(g["h"].median(), 1.0)
        tgt = targets[targets["track_id"] == tid].iloc[0]
        rows.append({"video_id": g["video_id"].iloc[0], "track_id": tid, "t_move_start": m["t"].min(),
                     "t_end": g["t"].max(), "class": tgt["class"], "bonus": tgt["bonus"], "clip_move": move})
    clip_t = pd.DataFrame(rows, columns=["video_id", "track_id", "t_move_start", "t_end", "class", "bonus", "clip_move"])
    # Targets that only start moving after the clip: still objects inside the clip.
    still = gt_rows["track_id"].isin(tids - set(clip_t["track_id"]))
    gt_rows = gt_rows.assign(role=gt_rows["role"].where(~still, "static"))
    return clip_t, gt_rows


def build_clips(det_run: str, only_videos: list[str] | None = None, seed: int = 0) -> dict:
    """Clip features, clip videos, clip targets and the two ground-truth variants."""
    cfg = load_config()
    data_dir, cache = cfg["paths"]["data_dir"], cfg["paths"]["cache_dir"]
    rng = np.random.default_rng(seed)

    videos = load_csv(data_dir, "train_videos").set_index("video_id")
    tracks = load_csv(data_dir, "train_tracks")
    targets = load_csv(data_dir, "train_alarms")
    track_dir = cache / "tracks" / det_run
    vids = sorted(p.stem for p in track_dir.glob("train_*.parquet"))
    if only_videos:
        vids = [v for v in vids if v in set(only_videos)]

    clip_videos, gt_parts, tgt_parts, feat_parts = [], [], [], []
    tracks_by_video = dict(tuple(tracks.groupby("video_id")))
    for vid in vids:
        v = videos.loc[vid]
        fps = float(v["fps"])
        pred = pd.read_parquet(track_dir / f"{vid}.parquet")
        n_frames = int(round(v["duration_s"] * fps))
        for k, (f0, f1) in enumerate(clip_bounds(n_frames, fps, rng)):
            cid = f"{vid}@{k}"
            clip_videos.append({"video_id": cid, "width": v["width"], "height": v["height"], "fps": fps,
                                "duration_s": (f1 - f0) / fps})
            ct, rows = clip_targets(cut(tracks_by_video[vid], f0, f1, fps, cid), targets[targets["video_id"] == vid])
            gt_parts.append(rows)
            tgt_parts.append(ct)
            p = cut(pred, f0, f1, fps, cid)
            if len(p):
                feat_parts.append(track_features(p, int(v["width"]), int(v["height"])))

    cv = pd.DataFrame(clip_videos)
    gt_rows, clip_t = pd.concat(gt_parts, ignore_index=True), pd.concat(tgt_parts, ignore_index=True)
    feats = pd.concat(feat_parts, ignore_index=True)
    print(f"{len(vids)} videos -> {len(cv)} clips (median {cv['duration_s'].median():.0f} s), "
          f"{int((~clip_t['bonus'].astype(bool)).sum())} clip targets")

    # Lenient variant: targets that barely move inside the clip become neutral.
    small = clip_t[clip_t["clip_move"] < MIN_CLIP_MOVE]
    lenient_rows = gt_rows.copy()
    key = lenient_rows["video_id"] + "|" + lenient_rows["track_id"]
    lenient_rows.loc[key.isin(small["video_id"] + "|" + small["track_id"]), "role"] = "ambiguous"
    variants = {"strict": GroundTruth(gt_rows, clip_t, cv),
                "lenient": GroundTruth(lenient_rows, clip_t[clip_t["clip_move"] >= MIN_CLIP_MOVE], cv)}
    return {"feats": feats, "videos": cv, "targets": clip_t, "variants": variants,
            "driver": cfg["motion"]["driver_inside"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--det-run", default="fasterrcnn_resnet50_fpn_v2_s1")
    parser.add_argument("--videos", nargs="*")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    c = build_clips(args.det_run, args.videos, args.seed)
    feats, cv, clip_t, variants, driver = c["feats"], c["videos"], c["targets"], c["variants"], c["driver"]

    print("F1 (P / target recall) on clips; rows: theta, confirm")
    for theta in [1.0, 1.5, 2.0, 2.5, 3.0]:
        for confirm in [1, 5]:
            a = make_alarms(feats, rule_moving(feats, theta), cv, confirm, driver)
            line = [f"{name} {r['f1']:.3f} ({r['precision']:.3f} / {r['target_recall']:.3f})"
                    for name, r in ((n, score(a, g)) for n, g in variants.items())]
            print(f"  theta {theta:3.1f} confirm {confirm:2d}  " + "   ".join(line))

    a = make_alarms(feats, rule_moving(feats, 2.0), cv, 5, driver)
    r = score(a, variants["strict"])
    print("\ntheta 2.0 confirm 5, strict outcomes:", r["outcomes"])
    t = r["targets"].merge(clip_t[["video_id", "track_id", "clip_move", "t_move_start"]], on=["video_id", "track_id"])
    t = t[~t["bonus"].astype(bool)]
    t["moves_in_clip"] = pd.cut(t["clip_move"], [-1, 0.5, 1, 2, 4, np.inf], labels=["<0.5", "0.5-1", "1-2", "2-4", ">=4"])
    t["already_moving"] = t["t_move_start"] < 0.5
    print("target recall by how far the target moves inside the clip (box heights):")
    print(t.groupby("moves_in_clip", observed=True)["hit"].agg(["size", "mean"]).round(2).T.to_string())
    print("target recall (n), rows: already moving at clip start, columns: moves inside the clip:")
    g = t.groupby(["already_moving", "moves_in_clip"], observed=True)["hit"]
    print((g.mean().round(2).astype(str) + " (" + g.size().astype(str) + ")").unstack().to_string())


if __name__ == "__main__":
    main()
