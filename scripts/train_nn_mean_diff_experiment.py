# scripts/train_nn_mean_diff_experiment.py - CPU sanity check for a target
# decomposition idea: instead of predicting white_elo and black_elo directly,
# predict avg_elo = (white+black)/2 and diff_elo = white-black, then
# reconstruct white/black from those two. The hope is that "overall game
# skill level" (avg) and "who's ahead in this specific game" (diff) are each
# individually easier to learn than the entangled white/black pair.
#
# Reuses the exact same backbone (EloGRUClassifier) and data
# (data/processed/nn/tokens.npy etc.) as train_nn_bucket_model.py, at the
# same scale/hyperparameters as the existing models/nn_arch_large_classweighted_sqrt_metrics.json
# benchmark, so results are directly comparable to that known number
# (acc 26.2%/26.2%, adj_acc 67.4%/67.5%, MAE 242/243 for white/black).
#
# avg_elo is bucketed the same way as white/black elo (200-wide, 400-3200).
# diff_elo is bucketed via equal-frequency (quantile) bins fit on the train
# split, so its 14 buckets are balanced by construction - isolates whether
# the underlying signal is easier/harder to learn, without a class-imbalance
# confound like the one white/black elo has.
#
# Must run under the Python 3.10 env:
#   "C:\\Users\\mario\\AppData\\Local\\Programs\\Python\\Python310\\python.exe" scripts/train_nn_mean_diff_experiment.py
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.train_nn_bucket_model import EloGRUClassifier, elo_to_bucket, BUCKET_LO, BUCKET_WIDTH, NUM_BUCKETS, BUCKET_MIDPOINTS  # noqa: E402

NN_DIR = Path(__file__).resolve().parent.parent / "data" / "processed" / "nn"
BRACKET_PROBS_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "bracket_probs_full.parquet"
PLAYER_ELO_PATH = NN_DIR / "player_elo.parquet"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
RANDOM_STATE = 42
NUM_DIFF_BUCKETS = 14


def load_raw(min_ply: int, limit=None):
    """Same join as train_nn_bucket_model.load_data, but returns raw white_elo/
    black_elo (not pre-bucketed) so avg/diff can be derived from them."""
    tokens_full = np.load(NN_DIR / "tokens.npy")
    lengths_full = np.load(NN_DIR / "lengths.npy")

    meta = pl.read_parquet(NN_DIR / "meta.parquet").with_row_index("index")
    bracket_probs_df = pl.read_parquet(BRACKET_PROBS_PATH)
    player_elo = pl.read_parquet(PLAYER_ELO_PATH)
    prob_cols = [c for c in bracket_probs_df.columns if c.startswith("p_")]

    merged = meta.join(bracket_probs_df, on="game_id", how="inner").join(player_elo, on="game_id", how="inner")

    idx = merged["index"].to_numpy()
    keep = lengths_full[idx] >= min_ply
    merged = merged.filter(pl.Series(keep))
    idx = idx[keep]

    if limit is not None:
        rng = np.random.RandomState(RANDOM_STATE)
        sel = rng.choice(len(idx), size=min(limit, len(idx)), replace=False)
        idx = idx[sel]
        merged = merged[sel]

    white_elo = merged["white_elo"].to_numpy().astype(np.float64)
    black_elo = merged["black_elo"].to_numpy().astype(np.float64)
    bracket_probs = merged.select(prob_cols).to_numpy().astype(np.float32)

    return tokens_full[idx], lengths_full[idx], bracket_probs, white_elo, black_elo, prob_cols


def diff_to_bucket(diff: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.clip(np.searchsorted(edges, diff, side="right") - 1, 0, NUM_DIFF_BUCKETS - 1).astype(np.int64)


class MeanDiffDataset(Dataset):
    def __init__(self, tokens, lengths, bracket_probs, avg_bucket, diff_bucket):
        self.tokens = torch.from_numpy(tokens).long()
        self.lengths = torch.from_numpy(lengths).long().clamp(min=1)
        self.bracket_probs = torch.from_numpy(bracket_probs).float()
        self.avg_bucket = torch.from_numpy(avg_bucket).long()
        self.diff_bucket = torch.from_numpy(diff_bucket).long()

    def __len__(self):
        return len(self.avg_bucket)

    def __getitem__(self, idx):
        return (self.tokens[idx], self.lengths[idx], self.bracket_probs[idx],
                self.avg_bucket[idx], self.diff_bucket[idx])


def run_epoch(model, loader, optimizer, criterion_avg, criterion_diff, train: bool, diff_midpoints, grad_clip=1.0):
    model.train(train)
    total_loss, n = 0.0, 0
    correct_avg, correct_diff, adj_avg, adj_diff = 0, 0, 0, 0
    mae_avg, mae_diff, mae_white_recon, mae_black_recon = 0.0, 0.0, 0.0, 0.0
    for tok, length, bprobs, avg_y, diff_y in loader:
        if train:
            optimizer.zero_grad()
        avg_logits, diff_logits = model(tok, length, bprobs)
        loss = criterion_avg(avg_logits, avg_y) + criterion_diff(diff_logits, diff_y)
        if train:
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        bs = len(avg_y)
        total_loss += loss.item() * bs
        n += bs
        with torch.no_grad():
            pred_avg = avg_logits.argmax(dim=1)
            pred_diff = diff_logits.argmax(dim=1)
            correct_avg += (pred_avg == avg_y).sum().item()
            correct_diff += (pred_diff == diff_y).sum().item()
            adj_avg += ((pred_avg - avg_y).abs() <= 1).sum().item()
            adj_diff += ((pred_diff - diff_y).abs() <= 1).sum().item()

            avg_mid = torch.from_numpy(BUCKET_MIDPOINTS).float()
            pred_avg_elo = avg_mid[pred_avg]
            true_avg_elo = avg_mid[avg_y]
            pred_diff_elo = diff_midpoints[pred_diff]
            true_diff_elo = diff_midpoints[diff_y]

            mae_avg += (pred_avg_elo - true_avg_elo).abs().sum().item()
            mae_diff += (pred_diff_elo - true_diff_elo).abs().sum().item()

            pred_white = pred_avg_elo + pred_diff_elo / 2
            pred_black = pred_avg_elo - pred_diff_elo / 2
            true_white = true_avg_elo + true_diff_elo / 2
            true_black = true_avg_elo - true_diff_elo / 2
            mae_white_recon += (pred_white - true_white).abs().sum().item()
            mae_black_recon += (pred_black - true_black).abs().sum().item()

    return {
        "loss": total_loss / n,
        "acc_avg": correct_avg / n, "acc_diff": correct_diff / n,
        "adj_acc_avg": adj_avg / n, "adj_acc_diff": adj_diff / n,
        "mae_avg": mae_avg / n, "mae_diff": mae_diff / n,
        "mae_white_recon": mae_white_recon / n, "mae_black_recon": mae_black_recon / n,
    }


def fmt(m):
    return (f"loss={m['loss']:.3f} acc(avg/diff)={m['acc_avg']:.3f}/{m['acc_diff']:.3f} "
            f"adj(avg/diff)={m['adj_acc_avg']:.3f}/{m['adj_acc_diff']:.3f} "
            f"mae(avg/diff)={m['mae_avg']:.1f}/{m['mae_diff']:.1f} "
            f"recon_mae(w/b)={m['mae_white_recon']:.1f}/{m['mae_black_recon']:.1f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=100_000, help="matches nn_arch_large_classweighted_sqrt's scale for direct comparability")
    ap.add_argument("--min-ply", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--embed-dim", type=int, default=48)
    ap.add_argument("--hidden-dim", type=int, default=96)
    ap.add_argument("--mlp-hidden", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--weight-power", type=float, default=0.5)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lr-min", type=float, default=1e-5)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--model-tag", type=str, default="mean_diff_experiment")
    args = ap.parse_args()

    torch.manual_seed(RANDOM_STATE)
    torch.set_num_threads(args.threads)
    print(f"torch using {torch.get_num_threads()} threads")

    print(f"Loading data (min_ply={args.min_ply}, limit={args.limit:,})...")
    t0 = time.time()
    tokens, lengths, bracket_probs, white_elo, black_elo, prob_cols = load_raw(args.min_ply, args.limit)
    avg_elo = (white_elo + black_elo) / 2
    diff_elo = white_elo - black_elo
    print(f"  {len(white_elo):,} games loaded in {time.time()-t0:.1f}s (seq_len={tokens.shape[1]})")
    print(f"  diff_elo stats: mean={diff_elo.mean():.1f} std={diff_elo.std():.1f} "
          f"p5={np.percentile(diff_elo, 5):.0f} p50={np.percentile(diff_elo, 50):.0f} p95={np.percentile(diff_elo, 95):.0f}")

    vocab = json.loads((NN_DIR / "vocab.json").read_text())
    vocab_size = len(vocab)

    n = len(white_elo)
    rng = np.random.RandomState(RANDOM_STATE)
    perm = rng.permutation(n)
    n_test = int(n * 0.1)
    n_val = int(n * 0.1)
    test_idx, val_idx, train_idx = perm[:n_test], perm[n_test:n_test + n_val], perm[n_test + n_val:]

    avg_bucket_all = elo_to_bucket(avg_elo)
    # equal-frequency bins fit on the TRAIN split only, so diff's 14 buckets
    # are balanced by construction - isolates learnability from imbalance.
    quantiles = np.linspace(0, 1, NUM_DIFF_BUCKETS + 1)
    edges = np.quantile(diff_elo[train_idx], quantiles)
    edges[0], edges[-1] = -np.inf, np.inf
    diff_bucket_all = diff_to_bucket(diff_elo, edges)
    diff_midpoints = torch.tensor([(max(edges[i], -1500) + min(edges[i + 1], 1500)) / 2 for i in range(NUM_DIFF_BUCKETS)], dtype=torch.float32)
    print(f"  diff bucket edges (train-fit quantiles): {np.round(edges[1:-1], 0).tolist()}")

    def subset(idx):
        return MeanDiffDataset(tokens[idx], lengths[idx], bracket_probs[idx], avg_bucket_all[idx], diff_bucket_all[idx])

    train_ds, val_ds, test_ds = subset(train_idx), subset(val_idx), subset(test_idx)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=512)
    test_loader = DataLoader(test_ds, batch_size=512)
    print(f"  train={len(train_ds):,} val={len(val_ds):,} test={len(test_ds):,}")

    model = EloGRUClassifier(
        vocab_size=vocab_size, embed_dim=args.embed_dim, hidden_dim=args.hidden_dim, bracket_dim=len(prob_cols),
        num_buckets=NUM_BUCKETS, mlp_hidden=args.mlp_hidden, dropout=args.dropout,
    )
    # NUM_DIFF_BUCKETS == NUM_BUCKETS (both 14) so the existing white_head/black_head
    # shapes are reused directly as avg_head/diff_head - no architecture change needed.
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr_min)

    def class_weights_for(bucket_train, num_buckets):
        counts = np.clip(np.bincount(bucket_train, minlength=num_buckets).astype(np.float64), 1, None)
        weights = (counts.sum() / (num_buckets * counts)) ** args.weight_power
        return torch.tensor(weights, dtype=torch.float32)

    avg_weights = class_weights_for(avg_bucket_all[train_idx], NUM_BUCKETS)
    diff_weights = class_weights_for(diff_bucket_all[train_idx], NUM_DIFF_BUCKETS)
    print(f"  avg class weights: {np.round(avg_weights.numpy(), 2).tolist()}")
    print(f"  diff class weights: {np.round(diff_weights.numpy(), 2).tolist()} (should be ~1.0 everywhere - quantile bins are balanced by construction)")
    criterion_avg = nn.CrossEntropyLoss(weight=avg_weights)
    criterion_diff = nn.CrossEntropyLoss(weight=diff_weights)

    ckpt_path = MODELS_DIR / f"nn_{args.model_tag}_best.pt"
    metrics_path = MODELS_DIR / f"nn_{args.model_tag}_metrics.json"

    print(f"\nTraining for {args.epochs} epochs...", flush=True)
    best_val_loss = float("inf")
    best_epoch = 0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_m = run_epoch(model, train_loader, optimizer, criterion_avg, criterion_diff, train=True, diff_midpoints=diff_midpoints, grad_clip=args.grad_clip)
        with torch.no_grad():
            val_m = run_epoch(model, val_loader, optimizer, criterion_avg, criterion_diff, train=False, diff_midpoints=diff_midpoints)
        scheduler.step()
        improved = val_m["loss"] < best_val_loss
        print(f"  epoch {epoch}: train[{fmt(train_m)}] val[{fmt(val_m)}] ({time.time()-t0:.1f}s){' *' if improved else ''}", flush=True)
        if improved:
            best_val_loss = val_m["loss"]
            best_epoch = epoch
            torch.save(model.state_dict(), ckpt_path)

    model.load_state_dict(torch.load(ckpt_path, weights_only=True))
    with torch.no_grad():
        test_m = run_epoch(model, test_loader, optimizer, criterion_avg, criterion_diff, train=False, diff_midpoints=diff_midpoints)
    print(f"\nFinal test metrics (best val checkpoint, epoch {best_epoch}): {fmt(test_m)}", flush=True)
    print(f"Compare recon_mae(w/b) above against the existing single-model benchmark: MAE 242/243 (nn_arch_large_classweighted_sqrt) or 221/220 (full-scale nn_bucket_full_arch_large_sqrtw_30ep)")

    metrics_path.write_text(json.dumps({
        "model": f"MeanDiffEloGRUClassifier_{args.model_tag}", "n_train": len(train_ds), "n_val": len(val_ds), "n_test": len(test_ds),
        "vocab_size": vocab_size, "epochs": args.epochs, "best_epoch": best_epoch, "min_ply": args.min_ply,
        "num_buckets": NUM_BUCKETS, "num_diff_buckets": NUM_DIFF_BUCKETS,
        "best_val_loss": round(best_val_loss, 4), "test_metrics": test_m,
        "diff_bucket_edges": edges[1:-1].tolist(),
        "embed_dim": args.embed_dim, "hidden_dim": args.hidden_dim, "mlp_hidden": args.mlp_hidden, "dropout": args.dropout,
        "weight_power": args.weight_power,
    }, indent=2))
    print(f"Saved: {metrics_path}", flush=True)


if __name__ == "__main__":
    main()
