# scripts/train_nn_time_aware_experiment.py - does per-ply clock time help?
"""Same harness as train_nn_board_aware_experiment.py (token-only baseline vs
token+extra-feature), but the extra feature is per-ply time-spent rather than
material balance. Reuses src/time_features.py's clock parsing (already mines
the embedded [%clk H:MM:SS] annotations for ~994k games, no board replay
needed - much cheaper than the material-balance experiment).

Motivation: an external benchmark (RatingNet, arXiv:2409.11506) reports clock
time takes their CNN-LSTM's MAE from 239 to 182 - the single biggest lever in
that paper. Our current tokenizer discards clock annotations entirely
(CLOCK_RE strips them in src/move_features.py before tokenization).

Usage: python scripts/train_nn_time_aware_experiment.py [--limit N] [--epochs N]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_nn_bucket_model import load_data, NUM_BUCKETS, NN_DIR, MODELS_DIR, RANDOM_STATE  # noqa: E402
from train_nn_board_aware_experiment import train_variant, MAX_LEN  # noqa: E402
from src.time_features import parse_moves_with_clock, parse_time_control, _clk_to_seconds  # noqa: E402

DATA_CSV = Path(__file__).resolve().parent.parent / "data" / "enhanced_extraction" / "enhanced_experiment_20250620_203308.csv"
TIME_SCALE = 30.0  # normalize seconds-spent to roughly O(1)


def compute_time_spent(move_sequence: str, time_control: str, max_len: int = MAX_LEN):
    out = np.zeros(max_len, dtype=np.float32)
    tc = parse_time_control(time_control)
    if tc is None:
        return out, 0.0
    base_seconds, increment = tc
    if base_seconds <= 0:
        return out, 0.0

    pairs = parse_moves_with_clock(move_sequence)[:max_len]
    clocks = [_clk_to_seconds(c) for _, c in pairs]
    n = len(clocks)
    if n == 0:
        return out, 0.0
    coverage = sum(c is not None for c in clocks) / n

    prev = {True: base_seconds, False: base_seconds}
    for i, c in enumerate(clocks):
        if c is None:
            continue
        is_white = (i % 2 == 0)
        spent = max(0.0, prev[is_white] + increment - c)
        out[i] = spent / TIME_SCALE
        prev[is_white] = c
    return out, coverage


def get_move_and_time_control(game_ids: np.ndarray):
    id_order = pl.DataFrame({"game_id": game_ids, "order": np.arange(len(game_ids))})
    rows = (
        pl.scan_csv(DATA_CSV)
        .select(["game_id", "move_sequence", "time_control"])
        .filter(pl.col("game_id").is_in(game_ids.tolist()))
        .collect(engine="streaming")
    )
    merged = id_order.join(rows, on="game_id", how="left").sort("order")
    return merged["move_sequence"].to_list(), merged["time_control"].to_list()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=20000)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--embed-dim", type=int, default=48)
    ap.add_argument("--hidden-dim", type=int, default=96)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lr-min", type=float, default=1e-5)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--min-ply", type=int, default=10)
    args = ap.parse_args()

    torch.manual_seed(RANDOM_STATE)
    torch.set_num_threads(args.threads)
    print(f"torch using {torch.get_num_threads()} thread(s)")

    print(f"Loading {args.limit} games...")
    tokens, lengths, bracket_probs, white_bucket, black_bucket, prob_cols, game_ids = load_data(args.min_ply, args.limit)
    vocab_size = len(json.loads((NN_DIR / "vocab.json").read_text()))

    print("Computing per-ply time-spent from embedded clock annotations (no board replay - should be quick)...")
    t0 = time.time()
    move_sequences, time_controls = get_move_and_time_control(game_ids)
    results = [compute_time_spent(seq, tc) for seq, tc in zip(move_sequences, time_controls)]
    time_spent = np.stack([r[0] for r in results])
    coverage = np.array([r[1] for r in results])
    print(f"  done in {time.time()-t0:.1f}s. mean clock coverage: {coverage.mean():.3f}, "
          f"games with coverage>=0.9: {(coverage >= 0.9).mean()*100:.1f}%")

    n = len(white_bucket)
    rng = np.random.RandomState(RANDOM_STATE)
    perm = rng.permutation(n)
    n_test, n_val = int(n * 0.1), int(n * 0.1)
    test_idx, val_idx, train_idx = perm[:n_test], perm[n_test:n_test + n_val], perm[n_test + n_val:]
    print(f"train={len(train_idx):,} val={len(val_idx):,} test={len(test_idx):,}")

    combined = np.concatenate([white_bucket[train_idx], black_bucket[train_idx]])
    counts = np.clip(np.bincount(combined, minlength=NUM_BUCKETS).astype(np.float64), 1, None)
    weights = (counts.sum() / (NUM_BUCKETS * counts)) ** 0.5
    class_weights = torch.tensor(weights, dtype=torch.float32)

    results_out = {}
    for name, use_extra in [("token-only baseline", False), ("token + clock-time", True)]:
        results_out[name] = train_variant(
            name, use_extra, tokens, lengths, bracket_probs, time_spent, white_bucket, black_bucket,
            prob_cols, vocab_size, train_idx, val_idx, test_idx, args.epochs, class_weights, args,
        )

    print("\n=== SUMMARY ===")
    for name, m in results_out.items():
        print(f"{name:28s} loss={m['loss']:.3f} acc(w/b)={m['acc_w']:.3f}/{m['acc_b']:.3f} adj_acc(w/b)={m['adj_w']:.3f}/{m['adj_b']:.3f}")

    out_path = MODELS_DIR / "time_aware_experiment_results.json"
    out_path.write_text(json.dumps(results_out, indent=2))
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
