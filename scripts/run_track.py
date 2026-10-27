"""Track the cached detections of every video; on train videos also attach the ground-truth labels.

Output: <cache_dir>/tracks/<detections run>/<video_id>.parquet (overwritten at every run: fast).
Then prints a quality report on the train videos:
- predicted tracks by the GT role they match ("none" = no annotated object);
- how many GT targets get a predicted track, and in how many pieces;
- ceiling scores with the competition metric: what pillar 3 could reach at best on these tracks
  (one alarm per track that matches a target, when the GT says it starts moving), and what we get
  with no motion filter at all (one alarm per track).

Usage: python scripts/run_track.py [--det-run fasterrcnn_resnet50_fpn_v2_s1] [--videos ...] [--no-report | --report-only]
"""

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from whomoves.config import load_config  # noqa: E402
from whomoves.io import load_csv  # noqa: E402
from whomoves.match import match_tracks  # noqa: E402
from whomoves.metric import GroundTruth, score  # noqa: E402
from whomoves.track import stitch_tracks, track_video  # noqa: E402


def alarms_at(rows: pd.DataFrame, videos: pd.DataFrame) -> pd.DataFrame:
    """One alarm per row, at the box center, normalized."""
    r = rows.merge(videos[["video_id", "width", "height"]], on="video_id")
    return pd.DataFrame({"video_id": r["video_id"], "t": r["t"], "x": (r["x"] + r["w"] / 2) / r["width"],
                         "y": (r["y"] + r["h"] / 2) / r["height"], "class": r["class"]})


def report(tracks: pd.DataFrame, data_dir: Path) -> None:
    gt_tracks = load_csv(data_dir, "train_tracks")
    gt_alarms = load_csv(data_dir, "train_alarms")
    videos = load_csv(data_dir, "train_videos")
    vids = sorted(tracks["video_id"].unique())
    gt = GroundTruth(gt_tracks, gt_alarms, videos)

    per_track = tracks.groupby(["video_id", "track_id"]).agg(
        role=("gt_role", "first"), cls=("class", "first"), n=("frame", "size"))
    print("Predicted tracks by matched GT role (median length in frames):")
    summary = per_track.groupby("role").agg(tracks=("n", "size"), median_len=("n", "median"))
    print(summary.sort_values("tracks", ascending=False).to_string(), "\n")

    targets = gt_alarms[gt_alarms["video_id"].isin(vids) & (gt_alarms["bonus"] == 0)]
    pieces = (tracks.dropna(subset=["gt_track_id"]).groupby(["video_id", "gt_track_id"])["track_id"].nunique()
              .rename("pieces"))
    t = targets.merge(pieces, left_on=["video_id", "track_id"], right_index=True, how="left")
    t["pieces"] = t["pieces"].fillna(0)
    found = t[t["pieces"] > 0]
    print(f"GT targets with >= 1 predicted track: {len(found)}/{len(t)} "
          f"(person {found['class'].eq('person').sum()}/{t['class'].eq('person').sum()}, "
          f"vehicle {found['class'].eq('vehicle').sum()}/{t['class'].eq('vehicle').sum()}); "
          f"pieces per found target: median {found['pieces'].median():.0f}, mean {found['pieces'].mean():.1f}\n")

    # Ceiling: perfect motion decision on our tracks.
    movers = tracks[tracks["gt_role"].isin(["target", "bonus"]) & (tracks["gt_moving"] == 1)]
    oracle = alarms_at(movers.sort_values("frame").groupby(["video_id", "track_id"]).head(1), videos)
    # No motion filter: one alarm per track, at its first detection.
    every = alarms_at(tracks.sort_values("frame").groupby(["video_id", "track_id"]).head(1), videos)
    for name, a in [("ceiling (perfect pillar 3)", oracle), ("no motion filter", every)]:
        r = score(a, gt, video_ids=vids)
        print(f"{name:28s} F1 {r['f1']:.3f}  P {r['precision']:.3f}  R {r['recall']:.3f}  "
              f"target recall {r['target_recall']:.3f}  "
              f"alarms {len(a)}  {dict(sorted(r['outcomes'].items(), key=lambda kv: -kv[1]))}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--det-run", default="fasterrcnn_resnet50_fpn_v2_s1")
    parser.add_argument("--videos", nargs="*")
    parser.add_argument("--no-report", action="store_true")
    parser.add_argument("--report-only", action="store_true", help="reuse the cached tracks, do not re-track")
    parser.add_argument("--stitch", action="store_true", help="enable tracker.stitch; outputs go to <run>_st")
    parser.add_argument("--min-det-score", type=float,
                        help="override tracker.min_det_score; outputs go to <det-run>_min<score%%> (use that name downstream)")
    args = parser.parse_args()

    cfg = load_config()
    data_dir, cache = cfg["paths"]["data_dir"], cfg["paths"]["cache_dir"]
    det_dir = cache / "detections" / args.det_run
    run_name = args.det_run
    if args.min_det_score is not None:
        cfg["tracker"]["min_det_score"] = args.min_det_score
        run_name += f"_min{round(args.min_det_score * 100):02d}"
    if args.stitch:
        cfg["tracker"]["stitch"] = True
        run_name += "_st"
    out_dir = cache / "tracks" / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    gt_tracks = load_csv(data_dir, "train_tracks")
    gt_by_video = dict(tuple(gt_tracks.groupby("video_id")))
    paths = sorted(det_dir.glob("*.parquet"))
    if args.videos:
        paths = [p for p in paths if p.stem in set(args.videos)]

    if args.report_only:
        train = [pd.read_parquet(out_dir / f"{p.stem}.parquet") for p in paths if p.stem in gt_by_video]
        report(pd.concat(train, ignore_index=True), data_dir)
        return

    t0, done = time.time(), []
    for path in paths:
        vid = path.stem
        tr = track_video(pd.read_parquet(path), cfg["tracker"])
        if cfg["tracker"].get("stitch"):
            tr = stitch_tracks(tr, cfg["tracker"])
        if vid in gt_by_video:
            tr = match_tracks(tr, gt_by_video[vid], **cfg["matching"])
        tr.to_parquet(out_dir / f"{vid}.parquet", index=False)
        done.append(tr)
    print(f"tracked {len(paths)} videos in {time.time() - t0:.0f}s -> {out_dir}\n")

    train = [t for t in done if "gt_role" in t.columns]
    if train and not args.no_report:
        report(pd.concat(train, ignore_index=True), data_dir)


if __name__ == "__main__":
    main()
