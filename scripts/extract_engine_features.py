# scripts/extract_engine_features.py - Mine embedded Lichess %eval comments
"""Extracts ACPL / blunder-mistake-inaccuracy features (see
src/engine_features.py) for every quality game that has embedded engine
analysis. Fast enough to run single-threaded (~30s for ~114k games).

Usage: python scripts/extract_engine_features.py
"""
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.engine_features import extract_engine_features  # noqa: E402

DATA_CSV = Path(__file__).resolve().parent.parent / "data" / "enhanced_extraction" / "enhanced_experiment_20250620_203308.csv"
OUT_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "engine_features.parquet"
USECOLS = ["game_id", "move_sequence", "avg_elo", "is_quality_game", "time_class", "has_engine_analysis", "result"]


def main():
    print("Loading engine-analyzed quality games...")
    df = pd.read_csv(DATA_CSV, usecols=USECOLS)
    df = df[df["is_quality_game"] & (df["time_class"] != "unknown") & df["has_engine_analysis"]]
    df = df.reset_index(drop=True)
    print(f"  {len(df):,} games")

    t0 = time.time()
    feats = [extract_engine_features(seq, result=r) for seq, r in zip(df["move_sequence"], df["result"])]
    feat_df = pd.DataFrame(feats)
    print(f"  Extracted in {time.time() - t0:.1f}s")
    print(feat_df["engine_features_ok"].value_counts())

    out = pd.concat([df[["game_id"]].reset_index(drop=True), feat_df], axis=1)
    out = out[out["engine_features_ok"]].reset_index(drop=True)
    out.to_parquet(OUT_PATH)
    print(f"Saved {len(out):,} rows: {OUT_PATH}")


if __name__ == "__main__":
    main()
