# src/time_features.py - Clock/time-management features
"""Mines the embedded Lichess clock annotations ([%clk H:MM:SS]) present in
essentially every game (unlike %eval, which needs engine analysis) - so
these features are available for the full ~994k games, not just the 11%
engine-analyzed subset.

How a player manages their clock is a skill signal distinct from move
quality itself: time trouble, snap decisions, and pacing all correlate
with skill independently of whether a given move was objectively good.
"""
import re

import numpy as np

CLOCK_OR_EVAL_TOKEN_RE = re.compile(r"\d+\.+|\{[^}]*\}|\S+")
CLK_TAG_RE = re.compile(r"\[%clk\s+([^\]]+)\]")
ANNOTATION_RE = re.compile(r"[!?]+$")
RESULT_TOKENS = {"1-0", "0-1", "1/2-1/2", "*"}

LOW_TIME_SECONDS = 10.0
FAST_MOVE_SECONDS = 1.0
OPENING_PLIES = 20


def parse_moves_with_clock(move_sequence: str) -> list[tuple[str, str | None]]:
    """[(san, clk_string_or_None), ...] in ply order."""
    pairs = []
    pending_san = None
    for tok in CLOCK_OR_EVAL_TOKEN_RE.finditer(move_sequence):
        t = tok.group(0)
        if t[0].isdigit() and t.rstrip(".").isdigit():
            continue  # move number
        if t.startswith("{"):
            if pending_san is not None:
                m = CLK_TAG_RE.search(t)
                pairs.append((pending_san, m.group(1) if m else None))
                pending_san = None
            continue
        if t in RESULT_TOKENS:
            continue
        if pending_san is not None:
            pairs.append((pending_san, None))
        pending_san = ANNOTATION_RE.sub("", t)
    if pending_san is not None:
        pairs.append((pending_san, None))
    return pairs


def _clk_to_seconds(clk_str: str | None) -> float | None:
    if not clk_str:
        return None
    parts = clk_str.strip().split(":")
    try:
        parts = [float(p) for p in parts]
    except ValueError:
        return None
    while len(parts) < 3:
        parts.insert(0, 0.0)
    h, m, s = parts
    return h * 3600 + m * 60 + s


def parse_time_control(time_control: str) -> tuple[float, float] | None:
    if not isinstance(time_control, str) or "+" not in time_control:
        return None
    base, _, inc = time_control.partition("+")
    try:
        return float(base), float(inc)
    except ValueError:
        return None


def _empty_time_features() -> dict:
    return {
        "clock_coverage": 0.0,
        "white_time_per_move": 0.0, "black_time_per_move": 0.0,
        "white_time_std": 0.0, "black_time_std": 0.0,
        "white_low_time_moves": 0, "black_low_time_moves": 0,
        "white_fast_moves": 0, "black_fast_moves": 0,
        "white_first_low_time_ply": -1, "black_first_low_time_ply": -1,
        "white_opening_time_per_move": 0.0, "black_opening_time_per_move": 0.0,
        "white_pace_used_by_ply20": 0.0, "black_pace_used_by_ply20": 0.0,
        "time_features_ok": False,
    }


def extract_time_features(move_sequence: str, time_control: str) -> dict:
    feat = _empty_time_features()
    tc = parse_time_control(time_control)
    if tc is None:
        return feat
    base_seconds, increment = tc
    if base_seconds <= 0:
        return feat

    pairs = parse_moves_with_clock(move_sequence)
    if not pairs:
        return feat

    clocks = [_clk_to_seconds(c) for _, c in pairs]
    n = len(clocks)
    feat["clock_coverage"] = sum(c is not None for c in clocks) / n
    if feat["clock_coverage"] < 0.9:
        return feat

    prev = {True: base_seconds, False: base_seconds}  # keyed by is_white
    time_spent = {True: [], False: []}
    low_time_ply = {True: -1, False: -1}
    opening_time = {True: [], False: []}
    pace_used = {True: None, False: None}

    for i, c in enumerate(clocks):
        if c is None:
            continue
        is_white = (i % 2 == 0)
        spent = max(0.0, prev[is_white] + increment - c)
        time_spent[is_white].append(spent)
        if i < OPENING_PLIES:
            opening_time[is_white].append(spent)
        if c <= LOW_TIME_SECONDS and low_time_ply[is_white] == -1:
            low_time_ply[is_white] = i + 1
        if i == OPENING_PLIES - 1 or (i == n - 1 and i < OPENING_PLIES):
            pace_used[is_white] = max(0.0, (base_seconds - c) / base_seconds)
        prev[is_white] = c

    for is_white, side in ((True, "white"), (False, "black")):
        spent = np.array(time_spent[is_white])
        if spent.size:
            feat[f"{side}_time_per_move"] = float(spent.mean())
            feat[f"{side}_time_std"] = float(spent.std())
            feat[f"{side}_low_time_moves"] = int((np.array([c for j, c in enumerate(clocks) if c is not None and j % 2 == (0 if is_white else 1)]) <= LOW_TIME_SECONDS).sum())
            feat[f"{side}_fast_moves"] = int((spent <= FAST_MOVE_SECONDS).sum())
        feat[f"{side}_first_low_time_ply"] = low_time_ply[is_white]
        opening = np.array(opening_time[is_white])
        if opening.size:
            feat[f"{side}_opening_time_per_move"] = float(opening.mean())
        if pace_used[is_white] is not None:
            feat[f"{side}_pace_used_by_ply20"] = pace_used[is_white]

    feat["time_features_ok"] = True
    return feat
