# scripts/eval_bracket_moe_checkpoint.py - compute exact MAE for a trained
# bracket-MoE checkpoint (train_bracket_moe_gpu.py doesn't report MAE itself,
# only accuracy/adj_accuracy). Reproduces the exact same train/val/test split
# (RandomState(42), same percentages) the training run used, so the test set
# evaluated here is genuinely held-out - never seen during training or for
# checkpoint selection (that used val loss).
#
# Prints val-set accuracy/adj_acc too, as a sanity check: if those match the
# numbers already printed during training, the split/model/checkpoint are
# correctly reproduced and the test-set MAE below can be trusted.
#
# Usage:
#   "C:\\Users\\mario\\AppData\\Local\\Programs\\Python\\Python310\\python.exe" scripts/eval_bracket_moe_checkpoint.py --checkpoint "C:\Users\mario\Downloads\bracket_moe_gpu_best.pt"
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_bracket_moe_gpu import (  # noqa: E402
    BracketExpertsMoE, BracketDataset, load_parquet_dataset, BUCKET_MIDPOINTS, RANDOM_STATE,
)

DATA_DIR = Path(r"C:\Users\mario\OneDrive\Documents\guess-the-elo\data\processed\nn_bracket_moe")


def eval_loader(model, loader, device, midpoints):
    correct_w, correct_b, adj_w, adj_b, n = 0, 0, 0, 0, 0
    mae_w, mae_b = 0.0, 0.0
    with torch.no_grad():
        for tok, length, tspent, white_y, black_y, white_masks, black_masks in loader:
            tok, length, tspent = tok.to(device), length.to(device), tspent.to(device)
            white_y, black_y = white_y.to(device), black_y.to(device)
            final_white, final_black, _, _ = model(tok, length, tspent)
            pred_w, pred_b = final_white.argmax(1), final_black.argmax(1)

            bs = len(white_y)
            n += bs
            correct_w += (pred_w == white_y).sum().item()
            correct_b += (pred_b == black_y).sum().item()
            adj_w += ((pred_w - white_y).abs() <= 1).sum().item()
            adj_b += ((pred_b - black_y).abs() <= 1).sum().item()
            mae_w += (midpoints[pred_w] - midpoints[white_y]).abs().sum().item()
            mae_b += (midpoints[pred_b] - midpoints[black_y]).abs().sum().item()
    return {
        "n": n, "acc_w": correct_w / n, "acc_b": correct_b / n,
        "adj_w": adj_w / n, "adj_b": adj_b / n,
        "mae_w": mae_w / n, "mae_b": mae_b / n,
    }


def fmt(m):
    return (f"n={m['n']:,} acc(w/b)={m['acc_w']:.3f}/{m['acc_b']:.3f} "
            f"adj_acc(w/b)={m['adj_w']:.3f}/{m['adj_b']:.3f} mae(w/b)={m['mae_w']:.1f}/{m['mae_b']:.1f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--data-dir", type=str, default=str(DATA_DIR))
    ap.add_argument("--embed-dim", type=int, default=24)
    ap.add_argument("--hidden-dim", type=int, default=48)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--val-sample", type=int, default=5000, help="subsample of val for a quick sanity check against training-time numbers")
    ap.add_argument("--test-sample", type=int, default=50_000, help="subsample of test - plenty for a stable MAE estimate, much faster than the full ~395k")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    data_dir = Path(args.data_dir)
    manifest = json.loads((data_dir / "manifest.json").read_text())
    vocab_size = manifest["vocab_size"]

    print(f"Loading dataset from {data_dir}...")
    t0 = time.time()
    tokens, lengths, time_spent, white_bucket, black_bucket, white_masks, black_masks = load_parquet_dataset(
        data_dir, manifest["max_len"], manifest["num_experts"],
    )
    print(f"  {len(tokens):,} games loaded in {time.time()-t0:.1f}s")

    # Exact same split as train_bracket_moe_gpu.py's main()
    n = len(white_bucket)
    rng = np.random.RandomState(RANDOM_STATE)
    perm = rng.permutation(n)
    n_test, n_val = int(n * 0.1), int(n * 0.1)
    test_idx, val_idx, train_idx = perm[:n_test], perm[n_test:n_test + n_val], perm[n_test + n_val:]
    print(f"train={len(train_idx):,} val={len(val_idx):,} test={len(test_idx):,}")

    subrng = np.random.RandomState(RANDOM_STATE)
    if args.val_sample and args.val_sample < len(val_idx):
        val_idx = val_idx[subrng.choice(len(val_idx), size=args.val_sample, replace=False)]
    if args.test_sample and args.test_sample < len(test_idx):
        test_idx = test_idx[subrng.choice(len(test_idx), size=args.test_sample, replace=False)]
    print(f"evaluating on val_sample={len(val_idx):,} test_sample={len(test_idx):,}")

    full_dataset = BracketDataset(tokens, lengths, time_spent, white_bucket, black_bucket, white_masks, black_masks)
    val_loader = DataLoader(Subset(full_dataset, val_idx), batch_size=args.batch_size, num_workers=args.num_workers)
    test_loader = DataLoader(Subset(full_dataset, test_idx), batch_size=args.batch_size, num_workers=args.num_workers)

    model = BracketExpertsMoE(vocab_size=vocab_size, embed_dim=args.embed_dim, hidden_dim=args.hidden_dim).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state_dict = ckpt["model"] if "model" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Loaded checkpoint: {args.checkpoint}" + (f" (epoch {ckpt['epoch']})" if isinstance(ckpt, dict) and "epoch" in ckpt else ""))

    midpoints = torch.from_numpy(BUCKET_MIDPOINTS).float().to(device)

    print("\nEvaluating on VAL split (sanity check vs. training-time printed numbers)...")
    val_m = eval_loader(model, val_loader, device, midpoints)
    print(f"  val: {fmt(val_m)}")

    print("\nEvaluating on TEST split (never seen during training or checkpoint selection)...")
    test_m = eval_loader(model, test_loader, device, midpoints)
    print(f"  test: {fmt(test_m)}")


if __name__ == "__main__":
    main()
