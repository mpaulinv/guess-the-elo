# src/engine_features.py - Lichess/Stockfish %eval-based features
"""Mines the embedded Lichess engine analysis ([%eval ...] comments) present
in ~11.4% of games' move_sequence, rather than running our own engine.

Goes beyond a single ACPL scalar to features drawn from the shape of the
game's evaluation history: phase-split accuracy (opening/middlegame/
endgame), blunder magnitude and timing, position volatility, and whether a
sustained *engine* advantage (not just material) actually got converted -
a much stronger version of the material-based conversion feature, since
eval captures positional advantage too.

%eval values in Lichess PGN exports are from White's perspective: positive
means White is better. Mate scores are given as '#N' (mate in N for the
side ahead) and are converted to a large capped centipawn value.
"""
import re

import numpy as np

from src.signal_utils import longest_sustained_run

CLOCK_OR_EVAL_TOKEN_RE = re.compile(r"\d+\.+|\{[^}]*\}|\S+")
EVAL_TAG_RE = re.compile(r"\[%eval\s+([^\]]+)\]")
ANNOTATION_RE = re.compile(r"[!?]+$")
RESULT_TOKENS = {"1-0", "0-1", "1/2-1/2", "*"}

MATE_SCORE_CP = 2000.0  # capped centipawn-equivalent magnitude for '#N' mate scores
BLUNDER_CP = 300.0
MISTAKE_CP = 100.0
INACCURACY_CP = 50.0
MIN_EVAL_COVERAGE = 0.5  # skip games where fewer than half the plies have an eval tag

OPENING_PLIES = 20  # first 10 full moves/side
ENDGAME_PLIES = 20  # last 10 full moves/side (only counted if it doesn't overlap the opening window)

# "Clear advantage" threshold for the eval-based sustained-advantage feature
# (2 pawns-equivalent) and how many consecutive plies it must hold for.
EVAL_ADVANTAGE_CP = 200.0
MIN_SUSTAIN_PLIES = 6


def parse_moves_with_eval(move_sequence: str) -> list[tuple[str, str | None]]:
    """[(san, eval_string_or_None), ...] in ply order."""
    pairs = []
    pending_san = None
    for tok in CLOCK_OR_EVAL_TOKEN_RE.finditer(move_sequence):
        t = tok.group(0)
        if t[0].isdigit() and t.rstrip(".").isdigit():
            continue  # move number, e.g. '1.' / '12...'
        if t.startswith("{"):
            if pending_san is not None:
                m = EVAL_TAG_RE.search(t)
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


def _eval_to_cp(eval_str: str | None) -> float | None:
    if not eval_str:
        return None
    eval_str = eval_str.strip()
    if eval_str.startswith("#"):
        try:
            n = int(eval_str[1:])
        except ValueError:
            return None
        return MATE_SCORE_CP if n > 0 else -MATE_SCORE_CP
    try:
        return float(eval_str) * 100.0
    except ValueError:
        return None


def _eval_to_mate(eval_str: str | None) -> int | None:
    """Signed mate distance from White's perspective (matches %eval sign
    convention): +N = White mates in N, -N = Black mates in N. None if this
    ply's eval isn't a mate score."""
    if not eval_str:
        return None
    eval_str = eval_str.strip()
    if not eval_str.startswith("#"):
        return None
    try:
        return int(eval_str[1:])
    except ValueError:
        return None


def _empty_engine_features() -> dict:
    return {
        "eval_coverage": 0.0,
        "white_acpl": 0.0, "black_acpl": 0.0, "avg_acpl": 0.0,
        "white_blunders": 0, "black_blunders": 0,
        "white_mistakes": 0, "black_mistakes": 0,
        "white_inaccuracies": 0, "black_inaccuracies": 0,
        "max_loss_white": 0.0, "max_loss_black": 0.0,
        "first_blunder_ply": -1,
        "eval_std": 0.0,
        "opening_acpl": 0.0, "middlegame_acpl": 0.0, "endgame_acpl": 0.0,
        "has_middlegame": 0, "has_endgame": 0,
        "balance_ratio": 0.0,
        "sustained_eval_advantage_plies": 0,
        "eval_advantage_converted": -1,
        "big_eval_advantage_not_converted": 0,
        "white_missed_mate_in_1": 0, "black_missed_mate_in_1": 0,
        "white_missed_mate_in_2": 0, "black_missed_mate_in_2": 0,
        "engine_features_ok": False,
    }


def extract_engine_features(move_sequence: str, result: str = "") -> dict:
    pairs = parse_moves_with_eval(move_sequence)
    feat = _empty_engine_features()
    if not pairs:
        return feat

    evals_raw = [_eval_to_cp(e) for _, e in pairs]
    n = len(evals_raw)
    feat["eval_coverage"] = sum(e is not None for e in evals_raw) / n
    if feat["eval_coverage"] < MIN_EVAL_COVERAGE:
        return feat

    # Missed forced mates: a clean, discrete signal distinct from continuous
    # centipawn loss - beginners routinely fail to find mate-in-1/2 even
    # when it's on the board. SAN marks checkmate with a trailing '#', which
    # tells us directly whether the mate was actually delivered.
    mate_list = [_eval_to_mate(e) for _, e in pairs]
    for i in range(1, n):
        mate_before = mate_list[i - 1]
        if mate_before is None:
            continue
        mover_is_white = (i % 2 == 0)
        mover_mate_distance = mate_before if mover_is_white else -mate_before
        if mover_mate_distance <= 0:
            continue  # the forced mate belongs to the opponent, not the mover
        san = pairs[i][0]
        delivered_now = san.endswith("#")
        side = "white" if mover_is_white else "black"
        if mover_mate_distance == 1 and not delivered_now:
            feat[f"{side}_missed_mate_in_1"] += 1
        elif mover_mate_distance == 2 and not delivered_now:
            delivered_next = (i + 2 < n) and pairs[i + 2][0].endswith("#")
            if not delivered_next:
                feat[f"{side}_missed_mate_in_2"] += 1

    # Forward-fill the rare missing eval so every ply has a usable value -
    # coverage is ~99% in practice, so this barely perturbs anything.
    filled = []
    last = 0.0
    for e in evals_raw:
        if e is not None:
            last = e
        filled.append(last)
    arr = np.array(filled, dtype=float)

    # Clip eval to +/- the mate-score cap first - Stockfish sometimes reports
    # near-forced-mate positions as large decimal pawn values (e.g. 60-90)
    # rather than '#N', which would otherwise blow up loss/max_loss.
    arr = np.clip(arr, -MATE_SCORE_CP, MATE_SCORE_CP)
    prev = np.concatenate(([0.0], arr[:-1]))
    delta = arr - prev
    is_white = (np.arange(n) % 2 == 0)  # ply index 0 (=ply 1) is White's move
    loss = np.where(is_white, np.maximum(0.0, -delta), np.maximum(0.0, delta))

    for mask, key in ((is_white, "white"), (~is_white, "black")):
        side_loss = loss[mask]
        if side_loss.size:
            feat[f"{key}_acpl"] = float(side_loss.mean())
            feat[f"{key}_blunders"] = int((side_loss >= BLUNDER_CP).sum())
            feat[f"{key}_mistakes"] = int(((side_loss >= MISTAKE_CP) & (side_loss < BLUNDER_CP)).sum())
            feat[f"{key}_inaccuracies"] = int(((side_loss >= INACCURACY_CP) & (side_loss < MISTAKE_CP)).sum())
            feat[f"max_loss_{key}"] = float(side_loss.max())
    feat["avg_acpl"] = (feat["white_acpl"] + feat["black_acpl"]) / 2

    blunder_idx = np.flatnonzero(loss >= BLUNDER_CP)
    feat["first_blunder_ply"] = int(blunder_idx[0] + 1) if blunder_idx.size else -1
    feat["eval_std"] = float(arr.std())
    feat["balance_ratio"] = float((np.abs(arr) < INACCURACY_CP * 2).mean())

    # Phase split: opening = first N plies, endgame = last N plies (only if
    # they don't overlap the opening window), everything else = middlegame.
    opening_end = min(OPENING_PLIES, n)
    endgame_start = max(opening_end, n - ENDGAME_PLIES)
    feat["opening_acpl"] = float(loss[:opening_end].mean())
    middlegame = loss[opening_end:endgame_start]
    endgame = loss[endgame_start:]
    if middlegame.size:
        feat["middlegame_acpl"] = float(middlegame.mean())
        feat["has_middlegame"] = 1
    if endgame.size:
        feat["endgame_acpl"] = float(endgame.mean())
        feat["has_endgame"] = 1

    # Eval-based sustained advantage + conversion (see module docstring) -
    # the richer, positional analogue of the earlier material-only version.
    run_len, run_sign = longest_sustained_run(arr, EVAL_ADVANTAGE_CP)
    feat["sustained_eval_advantage_plies"] = run_len
    if run_len >= MIN_SUSTAIN_PLIES and run_sign != 0 and result in ("1-0", "0-1", "1/2-1/2"):
        leader_is_white = run_sign > 0
        leader_won = (leader_is_white and result == "1-0") or (not leader_is_white and result == "0-1")
        feat["eval_advantage_converted"] = int(leader_won)
        if not leader_won:
            feat["big_eval_advantage_not_converted"] = 1

    feat["engine_features_ok"] = True
    return feat
