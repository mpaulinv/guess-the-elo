# scripts/analyze_player_convergence.py - Does averaging per-game predictions
# converge to a player's actual Elo?
"""Picks a handful of players with several games in the HELD-OUT split
(val+test - never used for gradient updates, so this is a fair generalization
check, not just re-scoring memorized training examples), runs the NN bucket
classifier's per-game prediction (softmax-weighted expected Elo, not just the
argmax bucket) on each of their games in chronological order, and tracks the
running average vs. their actual recorded Elo (last game in the dataset, as
a stand-in for "current" rating).

Games in this dataset were pulled from essentially random public games, so
most players only appear once or twice - reproduces the exact val/test split
train_nn_bucket_model.py used (same RandomState(42) permutation) and reports
how many held-out players actually clear the --min-games bar before running
anything, rather than assuming a large panel is available.

Uses the in-progress checkpoint if the full training run hasn't finished yet
- results will improve once training completes, but the convergence shape
should already be informative.

Must run under the Python 3.10 env (torch):
  "C:\\Users\\mario\\AppData\\Local\\Programs\\Python\\Python310\\python.exe" scripts/analyze_player_convergence.py [--checkpoint-tag TAG] [--n-players N] [--min-games N]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_nn_bucket_model import (  # noqa: E402
    EloGRUClassifier, BUCKET_MIDPOINTS, NN_DIR, BRACKET_PROBS_PATH, MODELS_DIR, RANDOM_STATE, load_data,
)

DATA_CSV = Path(__file__).resolve().parent.parent / "data" / "enhanced_extraction" / "enhanced_experiment_20250620_203308.csv"


def holdout_game_ids(min_ply: int):
    """Reproduces train_nn_bucket_model.py's exact val/test split (same seed,
    same load_data call with no limit -> same n and same permutation) and
    returns the game_ids that fall in val or test - i.e. never used for a
    gradient update, only (for val) checkpoint selection."""
    _tokens, _lengths, _bp, _wb, _bb, _prob_cols, game_ids = load_data(min_ply, limit=None)
    n = len(game_ids)
    rng = np.random.RandomState(RANDOM_STATE)
    perm = rng.permutation(n)
    n_test = int(n * 0.1)
    n_val = int(n * 0.1)
    holdout_positions = perm[:n_test + n_val]  # test + val, excludes train
    return set(game_ids[holdout_positions].tolist())


def load_player_games(min_ply: int, holdout_ids: set):
    base = (
        pl.scan_csv(DATA_CSV)
        .select(["game_id", "white_player", "black_player", "white_elo", "black_elo",
                  "utc_date", "utc_time", "is_quality_game", "time_class"])
        .filter(pl.col("is_quality_game") & (pl.col("time_class") != "unknown"))
        .filter(pl.col("game_id").is_in(holdout_ids))
        .collect(engine="streaming")
    )

    white_rows = base.select(
        pl.col("game_id"), pl.col("white_player").alias("player"), pl.col("white_elo").alias("elo"),
        pl.lit("white").alias("color"), pl.col("utc_date"), pl.col("utc_time"),
    )
    black_rows = base.select(
        pl.col("game_id"), pl.col("black_player").alias("player"), pl.col("black_elo").alias("elo"),
        pl.lit("black").alias("color"), pl.col("utc_date"), pl.col("utc_time"),
    )
    long_df = pl.concat([white_rows, black_rows]).with_columns(
        pl.concat_str(["utc_date", "utc_time"], separator=" ").alias("played_at")
    )

    meta = pl.read_parquet(NN_DIR / "meta.parquet").with_row_index("index")
    lengths_full = np.load(NN_DIR / "lengths.npy")
    bracket_probs_df = pl.read_parquet(BRACKET_PROBS_PATH)
    prob_cols = [c for c in bracket_probs_df.columns if c.startswith("p_")]

    merged = long_df.join(meta.select(["game_id", "index"]), on="game_id", how="inner") \
                     .join(bracket_probs_df, on="game_id", how="inner")

    idx_all = merged["index"].to_numpy()
    keep = lengths_full[idx_all] >= min_ply
    merged = merged.filter(pl.Series(keep))

    return merged, prob_cols


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint-tag", type=str, default="bucket_full_arch_large_30ep")
    ap.add_argument("--n-players", type=int, default=10)
    ap.add_argument("--min-games", type=int, default=None, help="default: auto-pick based on what's actually available in the holdout set")
    ap.add_argument("--min-ply", type=int, default=10)
    args = ap.parse_args()

    print("Reproducing the exact val+test split used for training (held out from gradient updates)...")
    holdout_ids = holdout_game_ids(args.min_ply)
    print(f"  {len(holdout_ids):,} held-out games")

    print("Loading player/game data restricted to the held-out set...")
    merged, prob_cols = load_player_games(args.min_ply, holdout_ids)

    game_counts = merged.group_by("player").agg(pl.len().alias("n_games")).sort("n_games", descending=True)
    n = game_counts["n_games"].to_numpy()
    print(f"  {len(n):,} distinct players appear in the held-out set")
    for thresh in [2, 3, 5, 8, 10, 15, 20]:
        print(f"    players with >= {thresh} held-out games: {int((n >= thresh).sum())}")

    min_games = args.min_games
    if min_games is None:
        # pick the largest threshold that still yields at least n_players candidates
        for thresh in [20, 15, 10, 8, 5, 3, 2]:
            if int((n >= thresh).sum()) >= args.n_players:
                min_games = thresh
                break
        else:
            min_games = 2
        print(f"  auto-selected --min-games={min_games}")

    eligible = game_counts.filter(pl.col("n_games") >= min_games)["player"].to_list()
    if not eligible:
        print(f"No players meet min_games={min_games} in the held-out set. Try a lower --min-games.")
        return

    rng = np.random.RandomState(RANDOM_STATE)
    chosen = rng.choice(eligible, size=min(args.n_players, len(eligible)), replace=False)

    metrics_path = MODELS_DIR / f"nn_{args.checkpoint_tag}_metrics.json"
    ckpt_path = MODELS_DIR / f"nn_{args.checkpoint_tag}_best.pt"
    cfg = json.loads(metrics_path.read_text()) if metrics_path.exists() else None
    embed_dim = cfg["embed_dim"] if cfg else 48
    hidden_dim = cfg["hidden_dim"] if cfg else 96
    mlp_hidden = cfg.get("mlp_hidden", 64) if cfg else 64
    gru_layers = cfg.get("gru_layers", 1) if cfg else 1
    bidirectional = cfg.get("bidirectional", False) if cfg else False
    vocab_size = len(json.loads((NN_DIR / "vocab.json").read_text()))

    print(f"\nLoading checkpoint {ckpt_path} (embed={embed_dim} hidden={hidden_dim})...")
    model = EloGRUClassifier(
        vocab_size=vocab_size, embed_dim=embed_dim, hidden_dim=hidden_dim, bracket_dim=len(prob_cols),
        mlp_hidden=mlp_hidden, gru_layers=gru_layers, bidirectional=bidirectional,
    )
    model.load_state_dict(torch.load(ckpt_path, weights_only=True))
    model.eval()

    tokens_full = np.load(NN_DIR / "tokens.npy")
    lengths_full = np.load(NN_DIR / "lengths.npy")
    midpoints = torch.from_numpy(BUCKET_MIDPOINTS).float()

    results = {}
    for i, player in enumerate(chosen):
        pdf = merged.filter(pl.col("player") == player).sort("played_at")
        idx = pdf["index"].to_numpy()
        colors = pdf["color"].to_list()
        actual_elo = pdf["elo"].to_numpy()

        tok_t = torch.from_numpy(tokens_full[idx]).long()
        len_t = torch.from_numpy(lengths_full[idx]).long().clamp(min=1)
        bp_t = torch.from_numpy(pdf.select(prob_cols).to_numpy().astype(np.float32)).float()

        with torch.no_grad():
            white_logits, black_logits = model(tok_t, len_t, bp_t)
            white_probs = torch.softmax(white_logits, dim=1)
            black_probs = torch.softmax(black_logits, dim=1)
            white_exp = (white_probs * midpoints).sum(dim=1)
            black_exp = (black_probs * midpoints).sum(dim=1)

        is_white = np.array([c == "white" for c in colors])
        predicted = np.where(is_white, white_exp.numpy(), black_exp.numpy())

        running_avg = np.cumsum(predicted) / np.arange(1, len(predicted) + 1)
        final_actual = actual_elo[-1]

        key = f"player_{i+1}"  # avoid non-ASCII usernames hitting the Windows console
        results[key] = {
            "n_games": len(idx),
            "final_actual_elo": float(final_actual),
            "actual_elo_range": [float(actual_elo.min()), float(actual_elo.max())],
            "running_avg_prediction": running_avg.round(1).tolist(),
            "per_game_prediction": predicted.round(1).tolist(),
            "per_game_actual_elo": actual_elo.tolist(),
        }
        n_to_100 = next((j + 1 for j, v in enumerate(running_avg) if abs(v - final_actual) <= 100), None)
        n_to_50 = next((j + 1 for j, v in enumerate(running_avg) if abs(v - final_actual) <= 50), None)
        print(f"{key:10s} n_games={len(idx):4d} actual={final_actual:6.0f} "
              f"final_running_avg={running_avg[-1]:6.1f} "
              f"first_within_100={n_to_100} first_within_50={n_to_50}")

    out_path = MODELS_DIR / "player_convergence_analysis.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
