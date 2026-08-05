# scripts/extract_time_features.py - Clock/time-management features, full dataset
"""Extracts time-management features (see src/time_features.py) for every
quality game - unlike engine %eval, clock annotations are present in
essentially all games, so this covers the full ~996k games.

Usage: python scripts/extract_time_features.py
"""
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.time_features import extract_time_features  # noqa: E402

DATA_CSV = Path(__file__).resolve().parent.parent / "data" / "enhanced_extraction" / "enhanced_experiment_20250620_203308.csv"
OUT_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "time_features.parquet"
USECOLS = ["game_id", "move_sequence", "time_control", "avg_elo", "is_quality_game", "time_class"]
CHUNK_SIZE = 100_000


def main():
    print("Extracting time features for all quality games...")
    t0 = time.time()
    parts = []
    reader = pd.read_csv(DATA_CSV, usecols=USECOLS, chunksize=CHUNK_SIZE)
    n_done = 0
    for chunk in reader:
        q = chunk[chunk["is_quality_game"] & (chunk["time_class"] != "unknown")]
        if q.empty:
            continue
        feats = [extract_time_features(seq, tc) for seq, tc in zip(q["move_sequence"], q["time_control"])]
        feat_df = pd.DataFrame(feats)
        out = pd.concat([q[["game_id"]].reset_index(drop=True), feat_df], axis=1)
        parts.append(out[out["time_features_ok"]])
        n_done += len(q)
        print(f"  ...{n_done:,} games processed ({time.time()-t0:.1f}s elapsed)")

    result = pd.concat(parts, ignore_index=True)
    result.to_parquet(OUT_PATH)
    print(f"\nDone in {time.time()-t0:.1f}s. Saved {len(result):,} rows: {OUT_PATH}")


if __name__ == "__main__":
    main()
