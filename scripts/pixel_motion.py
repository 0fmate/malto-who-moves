"""Pixel evidence of motion for each candidate alarm: does the image inside the box differ from the background?

Why: most false alarms left sit on real objects (zoom_verify.py: the detector finds them again up close), so the
error is the motion decision. A track that jumps between two parked cars, or a box that drifts on a person
standing still, moves on paper, but the pixels inside the box match the background, because the object has been
there all along. A real mover is not part of the background where it is now.

Background: temporal median of gray frames sampled every SAMPLE_S seconds within +-WINDOW_S of the row, using only
frames of the same clip (as on test, where a video is all we have). Per sampled track row: fg_box = share of box
pixels whose blurred gray level differs from the background by > DIFF_THR, fg_ring = the same on a ring around the box
(one box size wide; high when the camera shakes or the light changes).

Rows: candidates of the rule (person 0.75 / others 2.5, confirm 5) on the tiled test-like clips (t90) and on the test
videos; up to N_ROWS track rows evenly spread over [alarm - 2 s, alarm + 4 s]. One parquet per video in
<cache_dir>/pixel_motion/ (resumable).

Usage: python scripts/pixel_motion.py [--limit-videos N]
"""

import argparse
import pickle
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import learn_alarm_filter as L  # noqa: E402
import sweep_lookahead as sw  # noqa: E402
from whomoves.config import load_config  # noqa: E402
from whomoves.io import load_csv, video_path  # noqa: E402
from zoom_verify import clip_offsets  # noqa: E402

KEY = ["video_id", "track_id"]
THETA_PERSON = 0.75
SAMPLE_S, WINDOW_S, N_ROWS = 5.0, 25.0, 6
BEFORE_S, AFTER_S = 2.0, 4.0
DIFF_THR, BLUR = 25, 5


def requests(feats: pd.DataFrame, videos: pd.DataFrame, driver: float, offsets: dict | None) -> pd.DataFrame:
    """Up to N_ROWS track rows around the alarm of every candidate, with the clip range in original frames."""
    cand = L.candidate_rows(feats, THETA_PERSON, driver)[KEY + ["t"]].rename(columns={"t": "t_alarm"})
    tr = feats.merge(cand, on=KEY)
    tr = tr[(tr["t"] >= tr["t_alarm"] - BEFORE_S) & (tr["t"] <= tr["t_alarm"] + AFTER_S)].sort_values(KEY + ["t"])
    pos = tr.groupby(KEY).cumcount()
    n = tr.groupby(KEY)["t"].transform("size")
    step = np.maximum(n // N_ROWS, 1)
    req = tr[(pos % step == 0) & (pos // step < N_ROWS)][KEY + ["frame", "x", "y", "w", "h"]].copy()
    v = videos.set_index("video_id")
    req["base"] = req["video_id"].str.split("@").str[0]
    off = req["video_id"].map(offsets) if offsets is not None else 0
    req["frame"] = req["frame"] + off
    req["c0"] = off
    req["c1"] = off + (req["video_id"].map(v["duration_s"] * v["fps"])).round().astype(int)
    req["fps"] = req["video_id"].map(v["fps"])
    return req


def gray(bgr: np.ndarray) -> np.ndarray:
    return cv2.GaussianBlur(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (BLUR, BLUR), 0)


def fg_share(diff: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> float:
    return float(diff[y0:y1, x0:x1].mean()) if x1 > x0 and y1 > y0 else 0.0


def measure(diff: np.ndarray, x: float, y: float, w: float, h: float) -> tuple[float, float]:
    """Foreground share inside the box and on a ring one box size wide around it."""
    H, W = diff.shape
    bx0, by0 = int(np.clip(x, 0, W - 1)), int(np.clip(y, 0, H - 1))
    bx1, by1 = int(np.clip(np.ceil(x + w), bx0 + 1, W)), int(np.clip(np.ceil(y + h), by0 + 1, H))
    rx0, ry0 = int(np.clip(x - w, 0, W)), int(np.clip(y - h, 0, H))
    rx1, ry1 = int(np.clip(x + 2 * w, 0, W)), int(np.clip(y + 2 * h, 0, H))
    inner = diff[by0:by1, bx0:bx1].sum()
    ring_n = (rx1 - rx0) * (ry1 - ry0) - (bx1 - bx0) * (by1 - by0)
    ring = (diff[ry0:ry1, rx0:rx1].sum() - inner) / max(ring_n, 1)
    return fg_share(diff, bx0, by0, bx1, by1), float(ring)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit-videos", type=int)
    args = parser.parse_args()
    cfg = load_config()
    data_dir, cache, driver = cfg["paths"]["data_dir"], cfg["paths"]["cache_dir"], cfg["motion"]["driver_inside"]
    out_dir = cache / "pixel_motion"
    out_dir.mkdir(exist_ok=True)

    c = pickle.loads((cache / "clips90_t2x2.pkl").read_bytes())
    req = [requests(c["feats"], c["videos"], driver, clip_offsets(c["videos"]))]
    tv = load_csv(data_dir, "test_videos")
    ft = pd.read_parquet(cache / "features" / f"{sw.TILED}.parquet")
    req.append(requests(ft[ft["video_id"].isin(tv["video_id"])], tv, driver, None))
    req = pd.concat(req, ignore_index=True)
    bases = sorted(req["base"].unique())[: args.limit_videos]
    print(f"{len(req)} rows of {req.groupby(KEY).ngroups} candidates on {req['base'].nunique()} videos", flush=True)

    t0 = time.time()
    for base in bases:
        out = out_dir / f"{base}.parquet"
        if out.exists():
            continue
        r = req[req["base"] == base].reset_index(drop=True)
        fps = float(r["fps"].iloc[0])
        every = max(int(round(SAMPLE_S * fps)), 1)
        need = set(r["frame"])
        last = int(max(r["frame"].max(), r["c1"].max()))
        samples, frames = {}, {}
        cap = cv2.VideoCapture(str(video_path(data_dir, base)))
        f = 0
        while f <= last and cap.grab():  # decode only the frames we need
            if f % every == 0 or f in need:
                ok, bgr = cap.retrieve()
                if ok:
                    g = gray(bgr)
                    if f % every == 0:
                        samples[f] = g
                    if f in need:
                        frames[f] = g
            f += 1
        cap.release()

        fg_box, fg_ring = np.zeros(len(r)), np.zeros(len(r))
        bg_cache = {}
        win = int(round(WINDOW_S * fps))
        for i, row in r.iterrows():
            if row["frame"] not in frames:
                continue
            center = int(round(row["frame"] / every)) * every
            key = (row["c0"], row["c1"], center)
            if key not in bg_cache:  # median of the samples of this clip within the window
                s = [g for k, g in samples.items() if row["c0"] <= k < row["c1"] and abs(k - center) <= win]
                bg_cache[key] = np.median(np.stack(s), axis=0).astype(np.int16) if len(s) >= 3 else None
            bg = bg_cache[key]
            if bg is None:
                fg_box[i] = fg_ring[i] = np.nan
                continue
            diff = (np.abs(frames[row["frame"]].astype(np.int16) - bg) > DIFF_THR).astype(np.float32)
            fg_box[i], fg_ring[i] = measure(diff, row["x"], row["y"], row["w"], row["h"])
        r.assign(fg_box=fg_box, fg_ring=fg_ring)[KEY + ["frame", "fg_box", "fg_ring"]].to_parquet(out, index=False)
        print(f"{base}: {len(r)} rows ({time.time() - t0:.0f} s)", flush=True)
    print("done")


if __name__ == "__main__":
    main()
