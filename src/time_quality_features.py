# src/time_quality_features.py - Time x move-quality interaction at key moments
"""Joint extraction from the same comment block ([%eval ...] [%clk ...])
that engine_features.py and time_features.py each parse separately and then
discard the per-ply pairing. This module keeps that pairing to answer the
actual question: does this player allocate their thinking time toward the
moments that matter, and does that time pay off in move quality?

Only covers the ~114k engine-analyzed games - the "critical moment" signal
needs real eval, which time-only games don't have.

"Critical" is approximated (we don't have multi-PV data - see the earlier
discussion) as: high local eval volatility (the position has been swinging)
OR a near-balanced eval (small mistakes matter more when the game is close).
Neither is as good as true multi-PV move-spread, but both are legitimate,
cheap proxies for "this moment demanded care."
"""
import re

import numpy as np

from src.engine_features import MATE_SCORE_CP, _eval_to_cp
from src.time_features import _clk_to_seconds, parse_time_control

TOKEN_RE = re.compile(r"\d+\.+|\{[^}]*\}|\S+")
EVAL_TAG_RE = re.compile(r"\[%eval\s+([^\]]+)\]")
CLK_TAG_RE = re.compile(r"\[%clk\s+([^\]]+)\]")
ANNOTATION_RE = re.compile(r"[!?]+$")
RESULT_TOKENS = {"1-0", "0-1", "1/2-1/2", "*"}

VOLATILITY_WINDOW = 5
CRITICAL_VOLATILITY_CP = 80.0   # rolling std of eval over the window
CRITICAL_CLOSE_EVAL_CP = 150.0  # or the position is still close to balanced
FAST_MOVE_SECONDS = 2.0
KEY_MOMENT_LOSS_CP = 100.0      # "got it wrong" bar at a critical juncture
MIN_COVERAGE = 0.5


def parse_moves_with_eval_and_clock(move_sequence: str) -> list[tuple[str, str | None, str | None]]:
    pairs = []
    pending_san = None
    for tok in TOKEN_RE.finditer(move_sequence):
        t = tok.group(0)
        if t[0].isdigit() and t.rstrip(".").isdigit():
            continue
        if t.startswith("{"):
            if pending_san is not None:
                em = EVAL_TAG_RE.search(t)
                cm = CLK_TAG_RE.search(t)
                pairs.append((pending_san, em.group(1) if em else None, cm.group(1) if cm else None))
                pending_san = None
            continue
        if t in RESULT_TOKENS:
            continue
        if pending_san is not None:
            pairs.append((pending_san, None, None))
        pending_san = ANNOTATION_RE.sub("", t)
    if pending_san is not None:
        pairs.append((pending_san, None, None))
    return pairs


def _empty_features() -> dict:
    return {
        "coverage": 0.0,
        "white_critical_time_ratio": 1.0, "black_critical_time_ratio": 1.0,
        "white_critical_loss": 0.0, "black_critical_loss": 0.0,
        "white_quiet_loss": 0.0, "black_quiet_loss": 0.0,
        "white_fast_critical_errors": 0, "black_fast_critical_errors": 0,
        "white_slow_critical_success": 0, "black_slow_critical_success": 0,
        "white_time_loss_corr": 0.0, "black_time_loss_corr": 0.0,
        "white_critical_moments": 0, "black_critical_moments": 0,
        "time_quality_features_ok": False,
    }


def extract_time_quality_features(move_sequence: str, time_control: str) -> dict:
    feat = _empty_features()
    tc = parse_time_control(time_control)
    if tc is None or tc[0] <= 0:
        return feat
    base_seconds, increment = tc

    pairs = parse_moves_with_eval_and_clock(move_sequence)
    if not pairs:
        return feat

    evals_raw = [_eval_to_cp(e) for _, e, _ in pairs]
    clocks_raw = [_clk_to_seconds(c) for _, _, c in pairs]
    n = len(pairs)
    eval_cov = sum(e is not None for e in evals_raw) / n
    clk_cov = sum(c is not None for c in clocks_raw) / n
    feat["coverage"] = min(eval_cov, clk_cov)
    if feat["coverage"] < MIN_COVERAGE:
        return feat

    # Forward-fill both sequences (rare gaps) then clip eval.
    eval_arr, last = [], 0.0
    for e in evals_raw:
        if e is not None:
            last = e
        eval_arr.append(last)
    eval_arr = np.clip(np.array(eval_arr, dtype=float), -MATE_SCORE_CP, MATE_SCORE_CP)

    prev_eval = np.concatenate(([0.0], eval_arr[:-1]))
    delta = eval_arr - prev_eval
    is_white = (np.arange(n) % 2 == 0)
    loss = np.where(is_white, np.maximum(0.0, -delta), np.maximum(0.0, delta))

    # Rolling local volatility - a cheap proxy for "the position is unstable
    # right now", i.e. a moment that demands care.
    pad = np.pad(eval_arr, (VOLATILITY_WINDOW - 1, 0), mode="edge")
    volatility = np.array([pad[i:i + VOLATILITY_WINDOW].std() for i in range(n)])
    is_critical = (volatility >= CRITICAL_VOLATILITY_CP) | (np.abs(prev_eval) <= CRITICAL_CLOSE_EVAL_CP)

    # Time spent per ply, clock-only (mirrors time_features.py).
    clock_prev = {True: base_seconds, False: base_seconds}
    time_spent = np.zeros(n)
    for i, c in enumerate(clocks_raw):
        side = bool(is_white[i])
        if c is None:
            time_spent[i] = np.nan
            continue
        time_spent[i] = max(0.0, clock_prev[side] + increment - c)
        clock_prev[side] = c

    for mask, side in ((is_white, "white"), (~is_white, "black")):
        side_time = time_spent[mask]
        side_loss = loss[mask]
        side_crit = is_critical[mask]
        valid = ~np.isnan(side_time)
        if valid.sum() < 4:
            continue

        t, l, crit = side_time[valid], side_loss[valid], side_crit[valid]
        crit_time = t[crit]
        quiet_time = t[~crit]
        if crit_time.size and quiet_time.size and quiet_time.mean() > 0:
            feat[f"{side}_critical_time_ratio"] = float(crit_time.mean() / quiet_time.mean())
        if crit_time.size:
            feat[f"{side}_critical_loss"] = float(l[crit].mean())
            feat[f"{side}_critical_moments"] = int(crit.sum())
        if quiet_time.size:
            feat[f"{side}_quiet_loss"] = float(l[~crit].mean())

        feat[f"{side}_fast_critical_errors"] = int(((t <= FAST_MOVE_SECONDS) & crit & (l >= KEY_MOMENT_LOSS_CP)).sum())
        feat[f"{side}_slow_critical_success"] = int(((t > FAST_MOVE_SECONDS) & crit & (l < KEY_MOMENT_LOSS_CP / 2)).sum())

        if t.std() > 0 and l.std() > 0:
            feat[f"{side}_time_loss_corr"] = float(np.corrcoef(t, l)[0, 1])

    feat["time_quality_features_ok"] = True
    return feat
