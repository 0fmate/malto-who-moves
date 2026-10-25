"""Local re-implementation of the competition scorer (see CLAUDE.md, "How the score works").

Rules to implement:
- A target is a track with role "target" (and "bonus", which only adds points).
- An alarm hits a target if t is in [t_move_start - 1, t_end + 3] and (x, y) falls inside the
  target's box at that time, enlarged by 20% on each side.
- Alarms are processed in time order. First hit on a target: 1.0 (right class) or 0.6 (wrong).
  Later alarms on an already-hit target are ignored.
- Other alarms are false alarms: weight 2 on static / disturbance / driver / not-yet-moving
  targets, weight 1 elsewhere; neutral on "other" / "ambiguous" / borderline cases.
- F1 over all videos together: P = points / (hits + weighted FA), R = points / n_targets.

Sanity check: scoring train_alarms.csv against train_tracks.csv must give 1.0.
"""

import pandas as pd


def score(alarms: pd.DataFrame, tracks: pd.DataFrame, targets: pd.DataFrame) -> dict:
    """Score alarms (video_id, t, x, y, class; x, y normalized).

    Returns a dict with f1, precision, recall, points, hits, false-alarm counts by role,
    so that errors can be broken down for the error analysis.
    """
    raise NotImplementedError
