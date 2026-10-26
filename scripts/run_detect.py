"""Run the detector on the competition videos and cache one parquet file per video.

Resumable: videos whose file already exists are skipped, so after an interruption just run it again.
Output: <cache_dir>/detections/<detector>_s<stride>[_m<min_size>][_t<nx>x<ny>][_c<score %>]/<video_id>.parquet (+ run_info.json)

Usage:
  python scripts/run_detect.py                          # all 138 videos (train + test)
  python scripts/run_detect.py --split test --limit 2   # first 2 test videos
  python scripts/run_detect.py --videos train_0001e6b8 --max-frames 50   # speed test, nothing saved
  python scripts/run_detect.py --shard 0 --num-shards 2 # half of the videos (one process per GPU)
"""

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from whomoves.config import load_config  # noqa: E402
from whomoves.detect import Detector, detect_video  # noqa: E402
from whomoves.io import load_csv, video_path  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "test", "all"], default="all")
    parser.add_argument("--videos", nargs="*", help="explicit video_ids (overrides --split)")
    parser.add_argument("--add-videos", nargs="*", default=[], help="video_ids added to the --split list")
    parser.add_argument("--limit", type=int, help="only the first N videos")
    parser.add_argument("--max-frames", type=int, help="speed test: stop after N frames per video, save nothing")
    parser.add_argument("--detector", help="override config detector.name")
    parser.add_argument("--stride", type=int, help="override config detector.frame_stride")
    parser.add_argument("--min-size", type=int, help="override config detector.min_size (torchvision)")
    parser.add_argument("--tiles", help="tile grid, e.g. 2x2 (overrides config detector.tiles)")
    parser.add_argument("--score-threshold", type=float, help="override config detector.score_threshold")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    args = parser.parse_args()

    cfg = load_config()
    det_cfg = dict(cfg["detector"])
    if args.detector:
        det_cfg["name"] = args.detector
    if args.min_size:
        det_cfg["min_size"] = args.min_size
    if args.tiles:
        det_cfg["tiles"] = [int(n) for n in args.tiles.lower().split("x")]
    default_threshold = det_cfg["score_threshold"]
    if args.score_threshold is not None:
        det_cfg["score_threshold"] = args.score_threshold
    stride = args.stride or det_cfg.get("frame_stride", 1)
    data_dir = cfg["paths"]["data_dir"]

    if args.videos:
        video_ids = args.videos
    else:
        splits = ["train", "test"] if args.split == "all" else [args.split]
        video_ids = args.add_videos + [v for s in splits for v in load_csv(data_dir, f"{s}_videos")["video_id"]]
    video_ids = video_ids[args.shard::args.num_shards][: args.limit]

    run_name = (f"{det_cfg['name']}_s{stride}" + (f"_m{det_cfg['min_size']}" if det_cfg.get("min_size") else "")
                + (f"_t{det_cfg['tiles'][0]}x{det_cfg['tiles'][1]}" if det_cfg.get("tiles") else "")
                + (f"_c{round(det_cfg['score_threshold'] * 100):02d}" if det_cfg["score_threshold"] != default_threshold else ""))
    out_dir = cfg["paths"]["cache_dir"] / "detections" / run_name
    save = args.max_frames is None
    if save:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "run_info.json").write_text(json.dumps({**det_cfg, "frame_stride": stride}, indent=2))

    t0 = time.time()
    detector = Detector(det_cfg)
    print(f"{det_cfg['name']} on {detector.device}, stride {stride}, {len(video_ids)} videos "
          f"(shard {args.shard}/{args.num_shards}), model loaded in {time.time() - t0:.0f}s")

    done = skipped = total_frames = 0
    t_start = time.time()
    for vid in video_ids:
        out = out_dir / f"{vid}.parquet"
        if save and out.exists():
            skipped += 1
            continue
        t1 = time.time()
        df = detect_video(video_path(data_dir, vid), vid, detector, stride=stride,
                          batch_size=det_cfg.get("batch_size", 4), max_frames=args.max_frames)
        dt = time.time() - t1
        n_frames = df.attrs["n_frames"]
        if save:
            tmp = out.with_suffix(".tmp")
            df.to_parquet(tmp, index=False)
            tmp.replace(out)  # atomic: a half-written file never looks finished
        done += 1
        total_frames += n_frames
        counts = df["class"].value_counts().to_dict()
        print(f"{vid}: {n_frames} frames, {len(df)} dets ({len(df) / max(n_frames, 1):.1f}/frame) "
              f"{counts}, {n_frames / dt:.1f} fps")

    elapsed = time.time() - t_start
    print(f"done {done}, skipped {skipped} already cached, {total_frames} frames in {elapsed / 60:.1f} min"
          + (f" ({total_frames / elapsed:.1f} fps)" if elapsed > 0 and total_frames else ""))
    if save:
        print(f"output: {out_dir}")


if __name__ == "__main__":
    main()
