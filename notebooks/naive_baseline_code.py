# Code cells of the organizers' naive baseline notebook (outputs stripped).
# Original: ../who-moves-naive-baseline.ipynb. Reference only, not part of our pipeline.

# %% [markdown]
# # Who Moves? Naive baseline
# 
# The task: in each test video, raise one alarm (time, position, class) for every person, vehicle or animal that actually moves, and none for parked cars, people standing still, leaves, shadows or rain.
# 
# This baseline reads two frames per second, runs a COCO-pretrained detector on them, links each box to the most overlapping box of the previous frame, and raises an alarm when a linked box has moved more than half its height from where it was first seen. It writes `submission.csv` in the competition format.
# 
# Public leaderboard score: **0.603**. Each step below says what it does and which errors it causes, so you know where to start.

# %% [markdown]
# ## Settings and paths
# On Kaggle the competition data and the weights dataset are mounted under `/kaggle/input`. To run elsewhere, set `DATA_DIR`, `WEIGHTS_DIRS` and `OUT_PATH` as environment variables; `LIMIT=N` processes only the first N test videos.

# %% [code cell 1]
import glob, os, time
import pandas as pd

DATA_DIR = os.environ.get("DATA_DIR", "/kaggle/input")          # where test_videos.csv and test/*.mp4 live
WEIGHTS_DIRS = os.environ.get("WEIGHTS_DIRS", "/kaggle/input").split(os.pathsep)
OUT_PATH = os.environ.get("OUT_PATH", "/kaggle/working/submission.csv")
LIMIT = int(os.environ.get("LIMIT", "0"))                      # >0: only the first N videos (quick test)

# %% [markdown]
# ## How the baseline works, step by step
# 
# **1. Reading and sampling.** One frame every `EVERY_S = 0.5` s is decoded; the frames in between are skipped. The detector sees 2 frames per second of a 10 fps video. A target needs to be detected and linked in at least two samples before it can alarm, so someone who crosses a corner of the frame in about a second has at most two chances, and often none.
# 
# **2. Detector.** `fasterrcnn_mobilenet_v3_large_fpn` from torchvision, pretrained on COCO, boxes with score >= 0.5. COCO classes are mapped to `person`, `vehicle` (car, motorcycle, bus, train, truck) and `animal` (cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe); bicycles and birds are dropped. The detector resizes each frame so its short side is 800 px and its smallest anchor is 32 px, so people far away, 20 to 30 px tall in many of these scenes, are often never detected, and no later step can recover them. It also has no notion of "inside": a driver detected through the windscreen is a `person` box like any other.
# 
# **3. Association.** A box is linked to a box of the same class in the previous sample if their IoU is at least 0.3, best pairs first. Nothing predicts where an object will be. Two boxes of equal size have IoU 0.3 when one is shifted sideways by about 54% of its width, so anything that moves sideways by more than about half its own width in 0.5 s is never linked: each sample starts a new track. A person is about 2.5 times taller than wide, which puts this limit at roughly a fifth of their height per sample. A track with one missed detection, or one sample in which the class changes (a person seen as a car for one frame), is dropped and restarts from zero.
# 
# **4. Alarm rule.** A track raises one alarm, at the current box centre, the first time its centre is more than `K = 0.5` box heights from the centre of its first box. The two rules work against each other: a person walking sideways has to cover half a height in steps of at most a fifth of a height, so they need about four consecutive linked samples (1.5 s), and someone walking briskly across the view is linked rarely or not at all. A person who is already walking when first detected alarms only after travelling half a body height, which is late. A parked car whose box centre drifts by more than half the box height, for example when someone walks in front of it and the box shrinks, raises a false alarm; and since a lost track restarts as a new one, the same parked car can alarm again later, and each of those alarms counts as a weighted false alarm.
# 
# **5. Submission.** One row per test video, `video_id,alarms`, with `alarms` written as `t x y class;...` (seconds from the start, box centre normalised to 0-1). Videos with no alarm get an empty field; they must still be present.
# 
# The cell below is generated from `baseline/naive_core.py`, the module the local script `baseline/naive.py` also imports. The parameters are at the top: `EVERY_S`, `K`, `SCORE_MIN`, `IOU_MIN`, `MODEL_NAME`.

# %% [code cell 2]
"""Naive baseline for "Who Moves?".

  1. read one frame every EVERY_S seconds (0.5 s by default);
  2. run a COCO-pretrained torchvision Faster R-CNN on it;
  3. keep person / vehicle / animal boxes with score >= SCORE_MIN;
  4. link each box to a box of the same class in the previous sampled frame if their IoU >= IOU_MIN;
  5. raise one alarm per track when the box centre is more than K box heights away from
     where the track started.

There is no motion model, no background model and no tracker memory. The same parameters
are used for every video. torchvision is BSD-3; nothing else here needs a licence.
"""
from __future__ import annotations

import glob
import os
import time
from dataclasses import dataclass, field

import cv2
import numpy as np
import torch
import torchvision
from torchvision.models import detection as tvd

# ----------------------------------------------------------------------------------------------
# Parameters. The rules allow one setting for all videos, never one per video.
# ----------------------------------------------------------------------------------------------
EVERY_S = 0.5     # seconds between two analysed frames
K = 0.5           # alarm when centre displacement > K * box height
SCORE_MIN = 0.5   # detector confidence threshold
IOU_MIN = 0.3     # IoU needed to link a box to the previous sampled frame
MODEL_NAME = "fasterrcnn_mobilenet_v3_large_fpn"

# torchvision COCO category ids (91-index convention) -> challenge class.
# bicycle (2) is not mapped: a cyclist is scored as a person. bird (16) is not a target.
COCO_TO_CLASS = {
    1: "person",
    3: "vehicle", 4: "vehicle", 6: "vehicle", 7: "vehicle", 8: "vehicle",   # car, motorcycle, bus, train, truck
    17: "animal", 18: "animal", 19: "animal", 20: "animal", 21: "animal",   # cat, dog, horse, sheep, cow
    22: "animal", 23: "animal", 24: "animal", 25: "animal",                 # elephant, bear, zebra, giraffe
}

MODELS = {
    "fasterrcnn_resnet50_fpn_v2": (tvd.fasterrcnn_resnet50_fpn_v2,
                                   tvd.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT),
    "fasterrcnn_mobilenet_v3_large_fpn": (tvd.fasterrcnn_mobilenet_v3_large_fpn,
                                          tvd.FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT),
}


# ----------------------------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------------------------
def find_weights(model_name: str, search_dirs=("/kaggle/input",)) -> str | None:
    """Return the first `<model_name>*.pth` found below `search_dirs`, or None."""
    for d in search_dirs:
        hits = sorted(glob.glob(os.path.join(d, "**", f"{model_name}*.pth"), recursive=True))
        if hits:
            return hits[0]
    return None


def load_model(model_name: str = MODEL_NAME, device: str = "cpu", weights_path: str | None = None):
    builder, default_weights = MODELS[model_name]
    if weights_path:
        # Offline (Kaggle without internet): empty architecture + attached checkpoint.
        model = builder(weights=None, weights_backbone=None)
        model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    else:
        # Online: torchvision downloads the same COCO weights.
        model = builder(weights=default_weights)
    model.eval().to(device)
    return model


@torch.inference_mode()
def detect(model, frames_bgr: list[np.ndarray], device: str) -> list[list[tuple]]:
    """Return, per frame, a list of (x1, y1, x2, y2, cls) in pixels.

    torchvision resizes each frame so its short side is 800 px, and the smallest anchor of this
    MobileNet model is 32 px: objects much smaller than that are rarely found.
    """
    tensors = [torch.from_numpy(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float().div(255).to(device)
               for f in frames_bgr]
    outs = model(tensors)
    result = []
    for o in outs:
        boxes = o["boxes"].float().cpu().numpy()
        labels = o["labels"].cpu().numpy()
        scores = o["scores"].float().cpu().numpy()
        dets = []
        for b, l, s in zip(boxes, labels, scores):
            cls = COCO_TO_CLASS.get(int(l))
            if cls is not None and s >= SCORE_MIN:
                dets.append((float(b[0]), float(b[1]), float(b[2]), float(b[3]), cls))
        result.append(dets)
    return result


# ----------------------------------------------------------------------------------------------
# Frame sampling
# ----------------------------------------------------------------------------------------------
def sample_frames(path: str, every_s: float = EVERY_S):
    """Yield (t_seconds, frame_bgr, width, height) every `every_s` seconds.

    Frames in between are skipped with grab() (no decoding to RGB), so the detector never sees them.
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"cannot open {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    if fps <= 0:
        fps = 10.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    idx = 0
    k = 0
    next_idx = 0
    while True:
        if idx == next_idx:
            ok, frame = cap.read()
            if not ok:
                break
            yield idx / fps, frame, w, h
            k += 1
            next_idx = int(round(k * every_s * fps))
        else:
            if not cap.grab():
                break
        idx += 1
    cap.release()


# ----------------------------------------------------------------------------------------------
# Naive association + alarm rule
# ----------------------------------------------------------------------------------------------
def iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


@dataclass
class Track:
    cls: str
    box: tuple
    x0: float          # centre of the first box of the track; displacement is measured from here
    y0: float
    alarmed: bool = False


@dataclass
class NaiveAlarmer:
    k: float = K
    iou_min: float = IOU_MIN
    tracks: list = field(default_factory=list)
    alarms: list = field(default_factory=list)   # (t, x_norm, y_norm, cls)

    def update(self, t: float, dets: list[tuple], w: int, h: int):
        # Greedy matching: every (track, detection) pair of the same class with IoU >= iou_min,
        # best IoU first. A box that moved more than about half its width in EVERY_S seconds has
        # IoU < 0.3 with its previous box, so it is not linked and starts a new track.
        pairs = []
        for ti, tr in enumerate(self.tracks):
            for di, d in enumerate(dets):
                if d[4] == tr.cls:
                    v = iou(tr.box, d)
                    if v >= self.iou_min:
                        pairs.append((v, ti, di))
        pairs.sort(key=lambda p: -p[0])
        used_t, used_d = set(), set()
        new_tracks = []
        for _, ti, di in pairs:
            if ti in used_t or di in used_d:
                continue
            used_t.add(ti)
            used_d.add(di)
            tr = self.tracks[ti]
            d = dets[di]
            tr.box = d[:4]
            cx, cy = (d[0] + d[2]) / 2, (d[1] + d[3]) / 2
            bh = d[3] - d[1]
            if not tr.alarmed and np.hypot(cx - tr.x0, cy - tr.y0) > self.k * bh:
                tr.alarmed = True
                self.alarms.append((t, cx / w, cy / h, tr.cls))
            new_tracks.append(tr)
        # Unmatched detections start new tracks; unmatched tracks are dropped at once, so one missed
        # detection restarts the displacement count from zero.
        for di, d in enumerate(dets):
            if di not in used_d:
                new_tracks.append(Track(cls=d[4], box=d[:4], x0=(d[0] + d[2]) / 2, y0=(d[1] + d[3]) / 2))
        self.tracks = new_tracks


def process_video(model, path: str, device: str, every_s: float = EVERY_S, k: float = K,
                  batch: int = 4) -> tuple[list, float, int]:
    """Return (alarms, time_of_last_sampled_frame_s, n_sampled_frames) for one video."""
    alarmer = NaiveAlarmer(k=k)
    buf = []
    n = 0
    last_t = 0.0

    def flush():
        dets = detect(model, [f for _, f, _, _ in buf], device)
        for (t, _, w, h), d in zip(buf, dets):
            alarmer.update(t, d, w, h)
        buf.clear()

    for t, frame, w, h in sample_frames(path, every_s):
        buf.append((t, frame, w, h))
        n += 1
        last_t = t
        if len(buf) >= batch:
            flush()
    if buf:
        flush()
    return alarmer.alarms, last_t, n


def format_alarms(alarms) -> str:
    # Kaggle rejects empty cells, so a video with no alarm is written as a single ";".
    return ";".join(f"{t:.2f} {x:.4f} {y:.4f} {c}" for t, x, y, c in alarms) or ";"


def run(model, videos: list[tuple[str, str]], device: str, every_s: float = EVERY_S, k: float = K,
        batch: int = 4, log=print) -> list[dict]:
    """videos = [(video_id, path)]. Return one row {video_id, alarms} per video, empty alarms included."""
    rows = []
    for i, (vid, path) in enumerate(videos):
        t0 = time.time()
        alarms, dur, n = process_video(model, path, device, every_s, k, batch)
        dt = time.time() - t0
        rows.append({"video_id": vid, "alarms": format_alarms(alarms)})
        log(f"[{i + 1}/{len(videos)}] {vid}: {len(alarms)} alarms, {n} frames, "
            f"{dur:.0f} s of video in {dt:.0f} s ({60 * dt / max(dur, 1e-6):.1f} s per video minute)")
    return rows


# %% [markdown]
# ## Device and model weights
# The notebook runs without internet: attach the torchvision checkpoint as a Kaggle dataset (*Add Input*, a dataset containing `fasterrcnn_mobilenet_v3_large_fpn-fb6a3cc7.pth`). If no attached file is found and internet is on, torchvision downloads the same weights.

# %% [code cell 3]
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
weights_path = find_weights(MODEL_NAME, WEIGHTS_DIRS)
print("device:", device, "| weights:", weights_path or "download (needs internet)")
model = load_model(MODEL_NAME, device, weights_path)

# %% [markdown]
# ## Test videos
# `test_videos.csv` lists the `video_id`s; each video is `test/<video_id>.mp4`.

# %% [code cell 4]
csv_path = (sorted(glob.glob(os.path.join(DATA_DIR, "**", "test_videos.csv"), recursive=True))
            or sorted(glob.glob(os.path.join(DATA_DIR, "**", "videos.csv"), recursive=True)))[0]
videos_df = pd.read_csv(csv_path, dtype=str)
if "split" in videos_df.columns:
    videos_df = videos_df[videos_df["split"] == "test"]
base = os.path.dirname(csv_path)
videos = []
for _, r in videos_df.iterrows():
    if "path" in videos_df.columns and isinstance(r.get("path"), str):
        p = r["path"]
    else:  # test/<id>.mp4 next to videos.csv, or anywhere below it
        p = os.path.join(base, "test", f"{r['video_id']}.mp4")
        if not os.path.exists(p):
            p = sorted(glob.glob(os.path.join(base, "**", f"{r['video_id']}.mp4"), recursive=True))[0]
    videos.append((r["video_id"], p))
if LIMIT:
    videos = videos[:LIMIT]
print(len(videos), "videos from", csv_path)

# %% [markdown]
# ## Run and write `submission.csv`

# %% [code cell 5]
t0 = time.time()
rows = run(model, videos, device, EVERY_S, K, batch=4)
sub = pd.DataFrame(rows, columns=["video_id", "alarms"])
sub.to_csv(OUT_PATH, index=False)
print(f"wrote {OUT_PATH}: {len(sub)} videos, {sum(len(a.split(';')) for a in sub.alarms.fillna('') if a)} alarms in {time.time() - t0:.0f} s")
sub.head()

# %% [markdown]
# ## Looking at the errors on a training video
# The training set comes with the annotation: `train_tracks.csv` has one row per object and frame (`video_id, track_id, role, class, frame, t, x, y, w, h, moving`, boxes in pixels with `x, y` the top-left corner), and `train_alarms.csv` has one row per expected alarm (`track_id, t_move_start, t_end, class`). The cell below runs the baseline on one training video (the shortest one, or `TRAIN_VIDEO` if set) and draws a few frames:
# 
# - green: a target while it is moving (an alarm inside its box, enlarged by 20%, is a hit);
# - orange: a parked or still object, or a target that has not started to move yet (an alarm here is a false alarm of weight 2);
# - red cross: a baseline alarm raised within 0.5 s of that frame.
# 
# The frames shown are the first baseline alarms and the moments just after the first targets start moving, so you see both false alarms and misses. The cell is skipped if the training files are not attached.

# %% [code cell 6]
import cv2
import matplotlib.pyplot as plt

TRAIN_VIDEO = os.environ.get("TRAIN_VIDEO", "")      # empty: the shortest training video with a target
N_FRAMES = 6

tracks_csv = sorted(glob.glob(os.path.join(DATA_DIR, "**", "train_tracks.csv"), recursive=True))
if not tracks_csv:
    print("train_tracks.csv not found under", DATA_DIR, "- skipping the visual check")
else:
    tbase = os.path.dirname(tracks_csv[0])
    tracks = pd.read_csv(tracks_csv[0], dtype={"video_id": str})
    expected = pd.read_csv(os.path.join(tbase, "train_alarms.csv"), dtype={"video_id": str})
    vid = TRAIN_VIDEO
    if not vid:
        info = pd.read_csv(os.path.join(tbase, "train_videos.csv"), dtype={"video_id": str})
        info = info[info["video_id"].isin(expected["video_id"])]
        vid = info.sort_values("duration_s")["video_id"].iloc[0]
    hits = sorted(glob.glob(os.path.join(tbase, "**", f"{vid}.mp4"), recursive=True))
    if not hits:
        print("video", vid, "not found under", tbase, "- skipping the visual check")
    else:
        vpath = hits[0]
        exp = expected[expected["video_id"] == vid].sort_values("t_move_start")
        trk = tracks[tracks["video_id"] == vid]
        alarms, _, _ = process_video(model, vpath, device)
        print(f"{vid}: {len(exp)} expected alarms, {len(alarms)} baseline alarms")
        print(exp[["track_id", "t_move_start", "t_end", "class"]].to_string(index=False))
        print("baseline:", format_alarms(alarms) or "(none)")

        times = [a[0] for a in alarms[:N_FRAMES // 2]]
        times += [t + 1.0 for t in exp["t_move_start"] if not any(abs(t + 1.0 - u) < 2 for u in times)]
        times = sorted(times[:N_FRAMES])
        cap = cv2.VideoCapture(vpath)
        fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
        cols = 2
        rows_n = max(1, (len(times) + cols - 1) // cols)
        fig, axes = plt.subplots(rows_n, cols, figsize=(14, 4 * rows_n), squeeze=False)
        for ax in axes.flat:
            ax.axis("off")
        for ax, t in zip(axes.flat, times):
            f = int(round(t * fps))
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ok, img = cap.read()
            if not ok:
                continue
            H, W = img.shape[:2]
            for _, b in trk[trk["frame"] == f].iterrows():
                if b["role"] in ("target", "bonus") and b["moving"] == 1:
                    color = (0, 200, 0)
                elif b["role"] in ("target", "bonus", "static"):
                    color = (0, 140, 255)
                else:
                    continue   # disturbance, driver, other, ambiguous: not drawn, to keep the picture readable
                p1, p2 = (int(b["x"]), int(b["y"])), (int(b["x"] + b["w"]), int(b["y"] + b["h"]))
                cv2.rectangle(img, p1, p2, color, 2)
                cv2.putText(img, f"{b['track_id']} {b['class']}", (p1[0], max(12, p1[1] - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            for ta, xa, ya, ca in alarms:
                if abs(ta - t) <= 0.5:
                    c = (int(xa * W), int(ya * H))
                    cv2.drawMarker(img, c, (0, 0, 255), cv2.MARKER_CROSS, 24, 3)
                    cv2.putText(img, ca, (c[0] + 8, c[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            ax.set_title(f"{vid}  t = {t:.1f} s")
        cap.release()
        plt.tight_layout()
        plt.show()

# %% [markdown]
# ## Where to go from here
# Each direction below comes from one of the failures above. Check it on the training videos before building on it: `train_tracks.csv` tells you, frame by frame, what should and should not alarm.
# 
# - **Which targets are never alarmed, and why?** For the targets the baseline misses, was the object detected at all, or detected and never linked? The two need different fixes.
# - **Linking.** Sideways motion and single missed detections break the tracks. What changes if the association predicts where a box will be, keeps a lost track for a few samples, or uses more frames per second? What does it cost in run time?
# - **What "moving" looks like in the data.** The `moving` column is the definition used for scoring. Plot the box centre over time for parked cars and for walking people: what separates detector jitter from someone going somewhere? Net displacement over a window, a consistent direction, the speed relative to the box size?
# - **Small objects.** Look at the heights of the target boxes in the training data. How many are below what the detector finds? Pixel-level motion (frame differences, a background model) can say where to look, but it also fires on leaves, shadows and rain.
# - **Delay.** Compare `t_move_start` with the time of the baseline alarm on the targets it does catch. How much of the delay comes from the sampling, and how much from the half-height rule?
# - **Drivers and class.** A person box inside a moving vehicle box is not a target. And one class per detection is noisy: what does a vote along the whole track change?
# 
# The rules that matter for your method: the same method and settings for every video, with no hand-drawn zones, masks or per-video thresholds; no model that takes or produces text (no CLIP, no open-vocabulary detectors); and a public notebook that regenerates your submission from the competition data within the Kaggle limits.
