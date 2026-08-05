# scripts/build_player_elo.py - Per-player Elo lookup for the NN
"""tokens.npy/vocab.json/lengths.npy/meta.parquet (built by build_move_vocab.py)
are load-bearing for the already-trained nn_v12 model (ensemble_predict.py,
analyze_length_vs_error.py both read them) - they are NOT touched here.
Instead this writes a small side table of white_elo/black_elo per game_id,
row-joinable by game_id onto meta.parquet, for the per-player bucket
classifier (train_nn_bucket_model.py) without re-tokenizing anything.

Usage: python scripts/build_player_elo.py
"""
from pathlib import Path

import polars as pl

DATA_CSV = Path(__file__).resolve().parent.parent / "data" / "enhanced_extraction" / "enhanced_experiment_20250620_203308.csv"
OUT_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "nn" / "player_elo.parquet"


def main():
    df = (
        pl.scan_csv(DATA_CSV, schema_overrides={"white_elo": pl.Float64, "black_elo": pl.Float64})
        .select(["game_id", "white_elo", "black_elo", "is_quality_game", "time_class"])
        .filter(pl.col("is_quality_game") & (pl.col("time_class") != "unknown"))
        .select(["game_id", "white_elo", "black_elo"])
        .collect(engine="streaming")
    )
    df.write_parquet(OUT_PATH)
    print(f"Saved {df.shape} to {OUT_PATH}")


if __name__ == "__main__":
    main()
