# src/features.py - Baseline feature extraction for the Elo predictor
"""Turn the enhanced-extraction CSV into a feature matrix + target for the
RandomForest baseline.

Only columns that are legitimately available from a raw PGN (i.e. not derived
from the players' Elo) are used as features. `elo_diff`, `white_rating_diff`
and `black_rating_diff` are intentionally excluded — they leak the target.
"""
from pathlib import Path

import pandas as pd

# Columns pulled from the CSV. `move_sequence` is excluded on load - it's by
# far the largest column and none of the current features need the raw PGN
# movetext, only the counts already computed during extraction.
USECOLS = [
    "game_id", "result", "white_elo", "black_elo", "avg_elo", "eco", "termination",
    "time_class", "actual_move_count", "calculated_ply_count",
    "captures_count", "checks_count", "castling_count", "promotion_count",
    "has_engine_analysis", "quality_score", "is_quality_game",
    "white_title", "black_title",
]

ELO_BRACKETS = [
    (0, 800, "Sub-beginner (<800)"),
    (800, 1200, "Beginner (800-1200)"),
    (1200, 1600, "Intermediate (1200-1600)"),
    (1600, 2000, "Advanced (1600-2000)"),
    (2000, 2400, "Expert (2000-2400)"),
    (2400, 4000, "Master+ (2400+)"),
]


def load_quality_games(csv_path: str | Path, usecols=USECOLS) -> pd.DataFrame:
    """Load the extraction CSV, keeping only rows flagged as quality games
    with a known time class (drops the ~1.6k 'unknown' time-control rows)."""
    df = pd.read_csv(csv_path, usecols=usecols)
    df = df[df["is_quality_game"] & (df["time_class"] != "unknown")].copy()
    return df.reset_index(drop=True)


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Build the baseline feature matrix X and target y (avg_elo) from a
    loaded games DataFrame (see load_quality_games)."""
    feat = pd.DataFrame(index=df.index)

    # Game-length / activity counts.
    feat["ply_count"] = df["calculated_ply_count"]
    feat["captures_count"] = df["captures_count"]
    feat["checks_count"] = df["checks_count"]
    feat["castling_count"] = df["castling_count"]
    feat["promotion_count"] = df["promotion_count"]

    # Rates, normalized by game length - captures/checks per move tend to
    # separate skill level better than raw counts (short tactical blitz
    # games vs. long grinds both get compared fairly).
    ply_safe = feat["ply_count"].clip(lower=1)
    feat["captures_per_ply"] = feat["captures_count"] / ply_safe
    feat["checks_per_ply"] = feat["checks_count"] / ply_safe
    feat["promotions_per_ply"] = feat["promotion_count"] / ply_safe

    # Player metadata available from PGN headers, not derived from Elo.
    feat["has_engine_analysis"] = df["has_engine_analysis"].astype(int)
    feat["white_has_title"] = df["white_title"].notna().astype(int)
    feat["black_has_title"] = df["black_title"].notna().astype(int)
    feat["either_has_title"] = (feat["white_has_title"] | feat["black_has_title"]).astype(int)

    # Opening family (first letter of ECO code, A-E) - coarser than the 486
    # distinct ECO codes, which is too high-cardinality for a first baseline.
    eco_family = df["eco"].str[0]
    feat = feat.join(pd.get_dummies(eco_family, prefix="eco_family"))

    feat = feat.join(pd.get_dummies(df["time_class"], prefix="time_class"))
    feat = feat.join(pd.get_dummies(df["termination"], prefix="termination"))
    feat = feat.join(pd.get_dummies(df["result"], prefix="result"))

    y = df["avg_elo"]
    return feat, y


def elo_bracket(elo: pd.Series) -> pd.Series:
    """Label each Elo value with its bracket name (see ELO_BRACKETS)."""
    labels = pd.Series(index=elo.index, dtype=object)
    for lo, hi, name in ELO_BRACKETS:
        labels[(elo >= lo) & (elo < hi)] = name
    return labels
