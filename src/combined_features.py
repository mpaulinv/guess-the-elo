# src/combined_features.py - Joins tabular + move-replay features
"""Combines the basic per-game counts (src.features) with the richer
move-replay features (src.move_features / scripts/extract_move_features.py):
development speed, material swings, hanging pieces, book depth, and
sustained material-advantage conversion.
"""
from pathlib import Path

import pandas as pd

from src.features import build_features

MOVE_FEATURE_COLUMNS = [
    "book_depth_ply",
    "material_std", "material_max_swing", "material_final_abs",
    "material_num_big_swings", "max_material_advantage",
    "sustained_advantage_plies", "advantage_converted", "big_advantage_not_converted",
    "white_developed_by_ply20", "black_developed_by_ply20",
    "white_castled_ply", "black_castled_ply",
    "white_pawn_moves_by_ply20", "black_pawn_moves_by_ply20",
    "hanging_piece_events",
]


def load_move_features(parts_dir: str | Path) -> pd.DataFrame:
    parts_dir = Path(parts_dir)
    parts = sorted(parts_dir.glob("part_*.parquet"))
    if not parts:
        raise FileNotFoundError(f"no move-feature parts found in {parts_dir}")
    df = pd.concat((pd.read_parquet(p) for p in parts), ignore_index=True)
    return df[df["parse_ok"]].drop_duplicates(subset="game_id")


def build_combined_features(base_df: pd.DataFrame, move_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """base_df: output of src.features.load_quality_games (has game_id).
    move_df: output of load_move_features above."""
    merged = base_df.merge(move_df[["game_id"] + MOVE_FEATURE_COLUMNS], on="game_id", how="inner")

    X_basic, y = build_features(merged)
    X_move = merged[MOVE_FEATURE_COLUMNS].reset_index(drop=True)
    X = pd.concat([X_basic.reset_index(drop=True), X_move], axis=1)
    return X, y
