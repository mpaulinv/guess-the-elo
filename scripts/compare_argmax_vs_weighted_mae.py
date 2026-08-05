# scripts/compare_argmax_vs_weighted_mae.py - does the probability-weighted
# Elo point estimate (what app/inference.py actually returns) beat just
# using the argmax bucket's midpoint (what every MAE reported so far used)?
import json
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_bracket_moe_gpu import (  # noqa: E402
    BracketExpertsMoE, BracketDataset, load_parquet_dataset, BUCKET_MIDPOINTS, RANDOM_STATE,
)

DATA_DIR = Path(r"C:\Users\mario\OneDrive\Documents\guess-the-elo\data\processed\nn_bracket_moe")
CHECKPOINT = Path(r"C:\Users\mario\OneDrive\Documents\guess-the-elo\models\bracket_moe_gpu_epoch065.pt")
TEST_SAMPLE = 100_000


def main():
    device = torch.device("cpu")
    manifest = json.loads((DATA_DIR / "manifest.json").read_text())
    vocab_size = manifest["vocab_size"]

    print("Loading dataset...")
    t0 = time.time()
    tokens, lengths, time_spent, white_bucket, black_bucket, white_masks, black_masks = load_parquet_dataset(
        DATA_DIR, manifest["max_len"], manifest["num_experts"],
    )
    # raw continuous elo, not loaded by load_parquet_dataset - read separately (cheap, 2 float32 cols)
    raw = pq.read_table(DATA_DIR / "dataset.parquet", columns=["white_elo", "black_elo"])
    white_elo_raw = raw["white_elo"].to_numpy()
    black_elo_raw = raw["black_elo"].to_numpy()
    print(f"  loaded in {time.time()-t0:.1f}s")

    n = len(white_bucket)
    rng = np.random.RandomState(RANDOM_STATE)
    perm = rng.permutation(n)
    n_test, n_val = int(n * 0.1), int(n * 0.1)
    test_idx = perm[:n_test]
    sub_rng = np.random.RandomState(RANDOM_STATE)
    test_idx = test_idx[sub_rng.choice(len(test_idx), size=min(TEST_SAMPLE, len(test_idx)), replace=False)]
    print(f"  evaluating on {len(test_idx):,} test games")

    full_dataset = BracketDataset(tokens, lengths, time_spent, white_bucket, black_bucket, white_masks, black_masks)
    loader = DataLoader(Subset(full_dataset, test_idx), batch_size=1024, num_workers=2)

    model = BracketExpertsMoE(vocab_size=vocab_size, embed_dim=24, hidden_dim=48).to(device)
    ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    model.eval()
    midpoints = torch.from_numpy(BUCKET_MIDPOINTS).float()

    argmax_w, argmax_b, weighted_w, weighted_b, true_bucket_mid_w, true_bucket_mid_b = [], [], [], [], [], []
    t0 = time.time()
    with torch.no_grad():
        for tok, length, tspent, white_y, black_y, _wm, _bm in loader:
            final_white, final_black, _, _ = model(tok, length, tspent)
            wp = torch.softmax(final_white, dim=1)
            bp = torch.softmax(final_black, dim=1)

            argmax_w.append(midpoints[wp.argmax(1)])
            argmax_b.append(midpoints[bp.argmax(1)])
            weighted_w.append((wp * midpoints).sum(dim=1))
            weighted_b.append((bp * midpoints).sum(dim=1))
            true_bucket_mid_w.append(midpoints[white_y])
            true_bucket_mid_b.append(midpoints[black_y])
    print(f"  inference done in {time.time()-t0:.1f}s")

    argmax_w = torch.cat(argmax_w).numpy()
    argmax_b = torch.cat(argmax_b).numpy()
    weighted_w = torch.cat(weighted_w).numpy()
    weighted_b = torch.cat(weighted_b).numpy()
    true_bucket_mid_w = torch.cat(true_bucket_mid_w).numpy()
    true_bucket_mid_b = torch.cat(true_bucket_mid_b).numpy()
    true_raw_w = white_elo_raw[test_idx]
    true_raw_b = black_elo_raw[test_idx]

    def mae(pred, true):
        return float(np.abs(pred - true).mean())

    print(f"\n{'='*70}")
    print("MAE vs. TRUE BUCKET MIDPOINT (matches the 191.8/190.7 already reported):")
    print(f"  argmax:   white={mae(argmax_w, true_bucket_mid_w):.1f}  black={mae(argmax_b, true_bucket_mid_b):.1f}")
    print(f"  weighted: white={mae(weighted_w, true_bucket_mid_w):.1f}  black={mae(weighted_b, true_bucket_mid_b):.1f}")

    print(f"\nMAE vs. RAW CONTINUOUS TRUE ELO (the more honest ground truth):")
    print(f"  argmax:   white={mae(argmax_w, true_raw_w):.1f}  black={mae(argmax_b, true_raw_b):.1f}")
    print(f"  weighted: white={mae(weighted_w, true_raw_w):.1f}  black={mae(weighted_b, true_raw_b):.1f}")


if __name__ == "__main__":
    main()
