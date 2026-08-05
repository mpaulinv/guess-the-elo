# scripts/finalize_bracket_moe_eval.py - final closing analysis for the
# bracket-experts MoE, comparing the two GPU-trained checkpoint candidates
# (epoch 30 vs epoch 65) on three things:
#   1. Overall test-set accuracy/adj_acc/MAE for each checkpoint
#   2. How error breaks down across true-Elo rating bands (does either
#      checkpoint fall apart at the rare high/low ends?)
#   3. Player convergence: does averaging per-game predictions across a
#      player's held-out games converge to their true Elo, or does it show
#      the same regression-to-the-mean shrinkage the original single-model
#      regression had (analyze_player_convergence.py found slope ~0.598)?
#
# Reuses the exact model classes and streaming parquet loader from
# train_bracket_moe_gpu.py, and the exact held-out-split/player-matching
# methodology from analyze_player_convergence.py, adapted to this dataset
# (which carries game_id/white_elo/black_elo directly in the parquet, so no
# join to meta.parquet is needed for the accuracy parts - only the player
# convergence part needs to go back to the source CSVs for player identity).
#
# Usage:
#   "C:\\Users\\mario\\AppData\\Local\\Programs\\Python\\Python310\\python.exe" scripts/finalize_bracket_moe_eval.py
import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_bracket_moe_gpu import (  # noqa: E402
    BracketExpertsMoE, BracketDataset, load_parquet_dataset, BUCKET_MIDPOINTS, NUM_BUCKETS,
    BUCKET_LO, BUCKET_WIDTH, RANDOM_STATE,
)

DATA_DIR = Path(r"C:\Users\mario\OneDrive\Documents\guess-the-elo\data\processed\nn_bracket_moe")
MODELS_DIR = Path(r"C:\Users\mario\OneDrive\Documents\guess-the-elo\models")
SOURCE_CSVS = [
    Path(r"C:\Users\mario\OneDrive\Documents\guess-the-elo\data\enhanced_extraction\enhanced_experiment_20250620_203308.csv"),
    Path(r"C:\Users\mario\OneDrive\Documents\guess-the-elo\data\enhanced_extraction_2023_04\enhanced_experiment_20260717_000231.csv"),
]


def load_model(checkpoint_path, vocab_size, embed_dim, hidden_dim, device):
    model = BracketExpertsMoE(vocab_size=vocab_size, embed_dim=embed_dim, hidden_dim=hidden_dim).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()
    epoch = ckpt.get("epoch") if isinstance(ckpt, dict) else None
    return model, epoch


def eval_with_buckets(model, loader, device, midpoints):
    """Per-example predictions + true buckets, so we can slice by rating band afterward."""
    all_pred_w, all_pred_b, all_true_w, all_true_b = [], [], [], []
    with torch.no_grad():
        for tok, length, tspent, white_y, black_y, _wm, _bm in loader:
            tok, length, tspent = tok.to(device), length.to(device), tspent.to(device)
            final_white, final_black, _, _ = model(tok, length, tspent)
            all_pred_w.append(final_white.argmax(1).cpu())
            all_pred_b.append(final_black.argmax(1).cpu())
            all_true_w.append(white_y)
            all_true_b.append(black_y)
    pred_w = torch.cat(all_pred_w).numpy()
    pred_b = torch.cat(all_pred_b).numpy()
    true_w = torch.cat(all_true_w).numpy()
    true_b = torch.cat(all_true_b).numpy()
    return pred_w, pred_b, true_w, true_b


def summarize(pred, true, midpoints_np):
    correct = (pred == true)
    adj = (np.abs(pred - true) <= 1)
    mae = np.abs(midpoints_np[pred] - midpoints_np[true])
    return {"n": len(true), "acc": float(correct.mean()), "adj_acc": float(adj.mean()), "mae": float(mae.mean())}


def by_rating_band(pred, true, midpoints_np):
    rows = []
    for b in range(NUM_BUCKETS):
        mask = true == b
        n = int(mask.sum())
        lo, hi = BUCKET_LO + b * BUCKET_WIDTH, BUCKET_LO + (b + 1) * BUCKET_WIDTH
        if n == 0:
            rows.append({"bucket": f"{lo}-{hi}", "n": 0, "acc": None, "adj_acc": None, "mae": None})
            continue
        s = summarize(pred[mask], true[mask], midpoints_np)
        rows.append({"bucket": f"{lo}-{hi}", "n": n, "acc": round(s["acc"], 3), "adj_acc": round(s["adj_acc"], 3), "mae": round(s["mae"], 1)})
    return rows


def print_band_table(label, rows):
    print(f"\n  {label} by true-Elo band:")
    print(f"    {'band':>12s} {'n':>8s} {'acc':>6s} {'adj_acc':>8s} {'mae':>7s}")
    for r in rows:
        if r["n"] == 0:
            print(f"    {r['bucket']:>12s} {0:>8d} {'--':>6s} {'--':>8s} {'--':>7s}")
        else:
            print(f"    {r['bucket']:>12s} {r['n']:>8d} {r['acc']:>6.3f} {r['adj_acc']:>8.3f} {r['mae']:>7.1f}")


def holdout_game_ids(game_ids_all, n):
    """Exact same split logic as train_bracket_moe_gpu.py's main()."""
    rng = np.random.RandomState(RANDOM_STATE)
    perm = rng.permutation(n)
    n_test, n_val = int(n * 0.1), int(n * 0.1)
    test_idx, val_idx, train_idx = perm[:n_test], perm[n_test:n_test + n_val], perm[n_test + n_val:]
    return train_idx, val_idx, test_idx


def load_player_games(holdout_ids: set):
    # polars lazy-scan + streaming collect, matching analyze_player_convergence.py's
    # proven-fast pattern - the two source CSVs are ~9GB combined, and a pandas
    # chunked-read + isin() scan over that (tried first) was still running after
    # nearly an hour. game_ids in holdout_ids already passed the quality/min_ply
    # filter during preprocess_bracket_moe_data.py, so filtering on game_id
    # membership alone is sufficient - no need to re-check quality/ply here.
    holdout_list = list(holdout_ids)
    frames = []
    for path in SOURCE_CSVS:
        base = (
            pl.scan_csv(path)
            .select(["game_id", "white_player", "black_player", "white_elo", "black_elo", "utc_date", "utc_time"])
            .filter(pl.col("game_id").is_in(holdout_list))
            .collect(engine="streaming")
        )
        frames.append(base)
    df = pl.concat(frames).with_columns(
        pl.concat_str(["utc_date", "utc_time"], separator=" ").alias("played_at")
    )

    white = df.select(
        pl.col("game_id"), pl.col("white_player").alias("player"), pl.col("white_elo").alias("elo"),
        pl.lit("white").alias("color"), pl.col("played_at"),
    )
    black = df.select(
        pl.col("game_id"), pl.col("black_player").alias("player"), pl.col("black_elo").alias("elo"),
        pl.lit("black").alias("color"), pl.col("played_at"),
    )
    return pl.concat([white, black]).to_pandas()


def convergence_check(model, device, midpoints, tag, game_id_to_idx, tokens, lengths, time_spent, long_df, n_players, min_games, min_ply_filter=None, chosen_override=None):
    game_counts = long_df.groupby("player").size().sort_values(ascending=False)
    for thresh in [20, 15, 10, 8, 5, 3, 2]:
        if (game_counts >= thresh).sum() >= n_players:
            min_games = thresh
            break
    else:
        min_games = 2
    print(f"  [{tag}] auto-selected min_games={min_games} ({(game_counts >= min_games).sum()} eligible players)")

    eligible = game_counts[game_counts >= min_games].index.tolist()
    if chosen_override is not None:
        chosen = chosen_override
    else:
        rng = np.random.RandomState(RANDOM_STATE)
        chosen = rng.choice(eligible, size=min(n_players, len(eligible)), replace=False)

    final_preds, final_actuals = [], []
    for i, player in enumerate(chosen):
        pdf = long_df[long_df["player"] == player].sort_values("played_at")
        idx = np.array([game_id_to_idx[g] for g in pdf["game_id"]])
        colors = np.array(pdf["color"].tolist())
        actual_elo = pdf["elo"].to_numpy()
        if min_ply_filter is not None:
            keep = lengths[idx] >= min_ply_filter
            n_dropped = (~keep).sum()
            idx, colors, actual_elo = idx[keep], colors[keep], actual_elo[keep]
            if len(idx) < 2:
                print(f"    player_{i+1:02d} skipped - fewer than 2 games left after dropping {n_dropped} short ones")
                continue
        colors = colors.tolist()

        tok_t = torch.from_numpy(tokens[idx]).long().to(device)
        len_t = torch.from_numpy(lengths[idx]).long().clamp(min=1).to(device)
        ts_t = torch.from_numpy(time_spent[idx]).float().to(device)

        with torch.no_grad():
            final_white, final_black, _, _ = model(tok_t, len_t, ts_t)
            white_probs = torch.softmax(final_white, dim=1)
            black_probs = torch.softmax(final_black, dim=1)
            white_exp = (white_probs * midpoints).sum(dim=1).cpu().numpy()
            black_exp = (black_probs * midpoints).sum(dim=1).cpu().numpy()

        is_white = np.array([c == "white" for c in colors])
        predicted = np.where(is_white, white_exp, black_exp)
        running_avg = np.cumsum(predicted) / np.arange(1, len(predicted) + 1)
        final_actual = actual_elo[-1]
        final_preds.append(running_avg[-1])
        final_actuals.append(final_actual)
        n_to_100 = next((j + 1 for j, v in enumerate(running_avg) if abs(v - final_actual) <= 100), None)
        print(f"    player_{i+1:02d} n_games={len(idx):3d} actual={final_actual:6.0f} final_running_avg={running_avg[-1]:7.1f} first_within_100={n_to_100}")

    final_preds, final_actuals = np.array(final_preds), np.array(final_actuals)
    if len(final_preds) >= 2:
        slope, intercept = np.polyfit(final_actuals, final_preds, 1)
        mae = float(np.abs(final_preds - final_actuals).mean())
        print(f"  [{tag}] shrinkage-bias slope (predicted vs actual, 1.0=no shrinkage): {slope:.3f}  final-prediction MAE: {mae:.1f}")
        return slope, chosen
    return None, chosen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embed-dim", type=int, default=24)
    ap.add_argument("--hidden-dim", type=int, default=48)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--n-players", type=int, default=15)
    ap.add_argument("--min-ply", type=int, default=10)
    ap.add_argument("--data-dir", type=str, default=str(DATA_DIR))
    ap.add_argument("--checkpoints", type=str, default=None, help="comma-separated tags to evaluate, e.g. epoch065 or epoch050,epoch065 - default: all that exist in models/")
    args = ap.parse_args()
    data_dir = Path(args.data_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    midpoints_np = BUCKET_MIDPOINTS
    midpoints_t = torch.from_numpy(BUCKET_MIDPOINTS).float().to(device)

    manifest = json.loads((data_dir / "manifest.json").read_text())
    vocab_size = manifest["vocab_size"]

    print(f"Loading full dataset from {data_dir}...")
    t0 = time.time()
    tokens, lengths, time_spent, white_bucket, black_bucket, white_masks, black_masks = load_parquet_dataset(
        data_dir, manifest["max_len"], manifest["num_experts"],
    )
    game_ids = pq.read_table(data_dir / "dataset.parquet", columns=["game_id"])["game_id"].to_pylist()
    print(f"  {len(tokens):,} games loaded in {time.time()-t0:.1f}s")

    n = len(white_bucket)
    train_idx, val_idx, test_idx = holdout_game_ids(game_ids, n)
    print(f"train={len(train_idx):,} val={len(val_idx):,} test={len(test_idx):,}")

    full_dataset = BracketDataset(tokens, lengths, time_spent, white_bucket, black_bucket, white_masks, black_masks)
    test_loader = DataLoader(Subset(full_dataset, test_idx), batch_size=args.batch_size, num_workers=args.num_workers)

    candidate_checkpoints = {
        "epoch030": MODELS_DIR / "bracket_moe_gpu_epoch030.pt",
        "epoch050": MODELS_DIR / "bracket_moe_gpu_epoch050.pt",
        "epoch065": MODELS_DIR / "bracket_moe_gpu_epoch065.pt",
    }
    if args.checkpoints:
        wanted = set(args.checkpoints.split(","))
        checkpoints = {k: v for k, v in candidate_checkpoints.items() if k in wanted}
    else:
        checkpoints = {k: v for k, v in candidate_checkpoints.items() if v.exists()}
    for k, v in candidate_checkpoints.items():
        if k not in checkpoints:
            print(f"  (skipping {k}: {v} not found)")
    print(f"Evaluating: {list(checkpoints.keys())}")

    # ---- Part 1 & 2: test accuracy + rating-band breakdown ----
    summaries = {}
    for tag, path in checkpoints.items():
        print(f"\n{'='*70}\n{tag} ({path.name})\n{'='*70}")
        model, ckpt_epoch = load_model(path, vocab_size, args.embed_dim, args.hidden_dim, device)
        print(f"  checkpoint epoch field: {ckpt_epoch}")
        t0 = time.time()
        pred_w, pred_b, true_w, true_b = eval_with_buckets(model, test_loader, device, midpoints_t)
        print(f"  test inference done in {time.time()-t0:.1f}s")

        overall_w = summarize(pred_w, true_w, midpoints_np)
        overall_b = summarize(pred_b, true_b, midpoints_np)
        print(f"  TEST overall: white acc={overall_w['acc']:.3f} adj_acc={overall_w['adj_acc']:.3f} mae={overall_w['mae']:.1f} | "
              f"black acc={overall_b['acc']:.3f} adj_acc={overall_b['adj_acc']:.3f} mae={overall_b['mae']:.1f}")

        rows_w = by_rating_band(pred_w, true_w, midpoints_np)
        rows_b = by_rating_band(pred_b, true_b, midpoints_np)
        print_band_table("WHITE", rows_w)
        print_band_table("BLACK", rows_b)

        summaries[tag] = {"overall_white": overall_w, "overall_black": overall_b, "by_band_white": rows_w, "by_band_black": rows_b, "checkpoint_epoch": ckpt_epoch}
        del model
        torch.cuda.empty_cache() if device.type == "cuda" else None

    # ---- Part 3: player convergence / shrinkage check ----
    print(f"\n{'='*70}\nPlayer convergence check\n{'='*70}")
    holdout_ids = set(np.array(game_ids)[np.concatenate([val_idx, test_idx])].tolist())
    print(f"  {len(holdout_ids):,} held-out game_ids")
    long_df = load_player_games(holdout_ids)
    print(f"  {long_df['player'].nunique():,} distinct players in held-out set")
    game_id_to_idx = {g: i for i, g in enumerate(game_ids)}

    slopes = {}
    for tag, path in checkpoints.items():
        print(f"\n  -- {tag} --")
        model, _ = load_model(path, vocab_size, args.embed_dim, args.hidden_dim, device)
        slope, _chosen = convergence_check(model, device, midpoints_t, tag, game_id_to_idx, tokens, lengths, time_spent, long_df, args.n_players, None)
        slopes[tag] = slope
        del model

    print(f"\nReference: original single-model regression had shrinkage slope ~0.598 (1.0 = perfect convergence, 0 = pure regression-to-mean).")
    for tag, s in slopes.items():
        print(f"  {tag}: slope={s}")

    out = {"test_set_summary": summaries, "convergence_slopes": slopes}
    out_path = MODELS_DIR / "bracket_moe_final_eval.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
