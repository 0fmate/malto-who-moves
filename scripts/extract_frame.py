"""Save one frame of a video as a JPEG (optionally with ground-truth boxes drawn).

Usage: python scripts/extract_frame.py <video_id> <t_seconds> [--gt]
Output: outputs/frames/<video_id>_<t>.jpg
"""

import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from whomoves.config import load_config  # noqa: E402
from whomoves.io import load_csv, read_frame_at, video_path  # noqa: E402

ROLE_COLORS = {  # BGR
    "target": (0, 200, 0), "bonus": (0, 200, 200), "static": (0, 0, 255),
    "disturbance": (255, 0, 255), "driver": (255, 128, 0),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("video_id")
    parser.add_argument("t", type=float)
    parser.add_argument("--gt", action="store_true", help="draw train_tracks boxes (train only)")
    args = parser.parse_args()

    cfg = load_config()
    data_dir = cfg["paths"]["data_dir"]
    frame = read_frame_at(video_path(data_dir, args.video_id), args.t)

    if args.gt and args.video_id.startswith("train_"):
        tracks = load_csv(data_dir, "train_tracks")
        rows = tracks[(tracks.video_id == args.video_id) & ((tracks.t - args.t).abs() < 0.051)]
        for r in rows.itertuples():
            color = ROLE_COLORS.get(r.role, (200, 200, 200))
            p1, p2 = (int(r.x), int(r.y)), (int(r.x + r.w), int(r.y + r.h))
            cv2.rectangle(frame, p1, p2, color, 2)
            label = f"{r.track_id} {r.role} {'M' if r.moving else '-'}"
            cv2.putText(frame, label, (p1[0], max(p1[1] - 4, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
        print(f"{len(rows)} GT boxes drawn")

    out = cfg["paths"]["output_dir"] / "frames" / f"{args.video_id}_{args.t:.1f}.jpg"
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), frame)
    print(out)


if __name__ == "__main__":
    main()
