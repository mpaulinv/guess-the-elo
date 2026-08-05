# scripts/extract_time_quality_features.py - Key-moment time x quality features
"""Extracts the joint time/eval interaction features (see
src/time_quality_features.py) for every engine-analyzed quality game.

Usage: python scripts/extract_time_quality_features.py
"""
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.time_quality_features import extract_time_quality_features  # noqa: E402

DATA_CSV = Path(__file__).resolve().parent.parent / "data" / "enhanced_extraction" / "enhanced_experiment_20250620_203308.csv"
OUT_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "time_quality_features.parquet"
USECOLS = ["game_id", "move_sequence", "time_control", "avg_elo", "is_quality_game", "time_class", "has_engine_analysis"]


def main():
    print("Loading engine-analyzed quality games...")
    df = pd.read_csv(DATA_CSV, usecols=USECOLS)
    df = df[df["is_quality_game"] & (df["time_class"] != "unknown") & df["has_engine_analysis"]]
    df = df.reset_index(drop=True)
    print(f"  {len(df):,} games")

    t0 = time.time()
    feats = [extract_time_quality_features(seq, tc) for seq, tc in zip(df["move_sequence"], df["time_control"])]
    feat_df = pd.DataFrame(feats)
    print(f"  Extracted in {time.time() - t0:.1f}s")
    print(feat_df["time_quality_features_ok"].value_counts())

    out = pd.concat([df[["game_id"]].reset_index(drop=True), feat_df], axis=1)
    out = out[out["time_quality_features_ok"]].reset_index(drop=True)
    out.to_parquet(OUT_PATH)
    print(f"Saved {len(out):,} rows: {OUT_PATH}")


if __name__ == "__main__":
    main()
