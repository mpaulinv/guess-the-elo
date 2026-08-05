# scripts/check_short_game_convergence.py - does excluding short games (<15
# ply) from a player's held-out game history improve how well the running
# average of per-game predictions converges to their true Elo?
"""The cutoff-sweep in models/length_vs_error_analysis.json already showed
short games barely move aggregate per-game MAE (they're too rare a slice of
the data). But that's a different question from convergence: if the model
specifically hedges toward the population mean on short (low-information)
games, a handful of them in a player's history could pull their MULTI-GAME
running average toward the mean more than their share, hurting the
shrinkage-bias slope even though aggregate MAE barely moves.

Runs the exact same convergence check as finalize_bracket_moe_eval.py, on
the SAME set of players (paired comparison, not a fresh random sample),
once using their full held-out game history and once excluding any game
under --min-ply plies, and compares the resulting slopes directly.

Must run under the Python 3.10 env:
  "C:\\Users\\mario\\AppData\\Local\\Programs\\Python\\Python310\\python.exe" scripts/check_short_game_convergence.py
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_bracket_moe_gpu import BUCKET_MIDPOINTS, RANDOM_STATE  # noqa: E402
from finalize_bracket_moe_eval import (  # noqa: E402
    DATA_DIR, MODELS_DIR, load_model, load_player_games, holdout_game_ids, convergence_check,
)
from train_bracket_moe_gpu import load_parquet_dataset  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, default=str(MODELS_DIR / "bracket_moe_gpu_epoch065.pt"))
    ap.add_argument("--embed-dim", type=int, default=24)
    ap.add_argument("--hidden-dim", type=int, default=48)
    ap.add_argument("--n-players", type=int, default=20)
    ap.add_argument("--min-ply-filter", type=int, default=15)
    ap.add_argument("--data-dir", type=str, default=str(DATA_DIR))
    args = ap.parse_args()
    data_dir = Path(args.data_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    midpoints_t = torch.from_numpy(BUCKET_MIDPOINTS).float().to(device)

    manifest = json.loads((data_dir / "manifest.json").read_text())
    print(f"Loading full dataset from {data_dir}...")
    t0 = time.time()
    tokens, lengths, time_spent, white_bucket, black_bucket, white_masks, black_masks = load_parquet_dataset(
        data_dir, manifest["max_len"], manifest["num_experts"],
    )
    game_ids = pq.read_table(data_dir / "dataset.parquet", columns=["game_id"])["game_id"].to_pylist()
    print(f"  {len(tokens):,} games loaded in {time.time()-t0:.1f}s")

    n = len(white_bucket)
    train_idx, val_idx, test_idx = holdout_game_ids(game_ids, n)
    holdout_ids = set(np.array(game_ids)[np.concatenate([val_idx, test_idx])].tolist())
    print(f"  {len(holdout_ids):,} held-out game_ids")

    print("Scanning source CSVs for player identity (polars streaming)...")
    t0 = time.time()
    long_df = load_player_games(holdout_ids)
    print(f"  {long_df['player'].nunique():,} distinct players, loaded in {time.time()-t0:.1f}s")

    game_id_to_idx = {g: i for i, g in enumerate(game_ids)}
    vocab_size = manifest["vocab_size"]
    model, ckpt_epoch = load_model(args.checkpoint, vocab_size, args.embed_dim, args.hidden_dim, device)
    print(f"Loaded checkpoint epoch {ckpt_epoch}")

    print(f"\n-- Baseline: all held-out games --")
    slope_all, chosen = convergence_check(
        model, device, midpoints_t, "all_games", game_id_to_idx, tokens, lengths, time_spent,
        long_df, args.n_players, None,
    )

    print(f"\n-- Excluding games under {args.min_ply_filter} ply (SAME players) --")
    slope_filtered, _ = convergence_check(
        model, device, midpoints_t, f"min_ply_{args.min_ply_filter}", game_id_to_idx, tokens, lengths, time_spent,
        long_df, args.n_players, None, min_ply_filter=args.min_ply_filter, chosen_override=chosen,
    )

    print(f"\n{'='*60}")
    print(f"Shrinkage slope, all games:              {slope_all:.3f}")
    print(f"Shrinkage slope, excluding <{args.min_ply_filter} ply:   {slope_filtered:.3f}")
    print(f"Difference: {slope_filtered - slope_all:+.3f} (positive = filtering helped convergence)")


if __name__ == "__main__":
    main()
