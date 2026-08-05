# scripts/train_nn_model.py - GRU over raw moves, conditioned on RF/XGB bracket prior
"""Embeds each SAN move token, encodes the sequence with a GRU, concatenates
the classifier's predicted bracket probabilities (classifier_full_v6, see
scripts/train_full_classifier.py) into the final hidden state, then a small
MLP head regresses avg_elo. Runs on ALL quality games (no engine-analysis
subset restriction - the whole point is learning from moves alone).

Must run under the Python 3.10 env (torch DLL issue on 3.13 on this
machine):
  "C:\\Users\\mario\\AppData\\Local\\Programs\\Python\\Python310\\python.exe" scripts/train_nn_model.py [--limit N] [--epochs N]

Usage: python scripts/train_nn_model.py [--limit N] [--epochs N]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

NN_DIR = Path(__file__).resolve().parent.parent / "data" / "processed" / "nn"
BRACKET_PROBS_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "bracket_probs_full.parquet"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
RANDOM_STATE = 42


class ChessGameDataset(Dataset):
    def __init__(self, tokens, lengths, bracket_probs, targets):
        self.tokens = torch.from_numpy(tokens).long()
        self.lengths = torch.from_numpy(lengths).long().clamp(min=1)
        self.bracket_probs = torch.from_numpy(bracket_probs).float()
        self.targets = torch.from_numpy(targets).float()

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.tokens[idx], self.lengths[idx], self.bracket_probs[idx], self.targets[idx]


class EloGRU(nn.Module):
    def __init__(self, vocab_size, embed_dim=48, hidden_dim=96, bracket_dim=6, mlp_hidden=64):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim + bracket_dim, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(mlp_hidden, 1),
        )

    def forward(self, tokens, lengths, bracket_probs):
        emb = self.embed(tokens)
        packed = nn.utils.rnn.pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h_n = self.gru(packed)
        h = h_n[-1]
        combined = torch.cat([h, bracket_probs], dim=1)
        return self.head(combined).squeeze(-1)


def load_data(limit=None):
    tokens = np.load(NN_DIR / "tokens.npy")
    lengths = np.load(NN_DIR / "lengths.npy")
    meta = pd.read_parquet(NN_DIR / "meta.parquet")
    bracket_probs = pd.read_parquet(BRACKET_PROBS_PATH)

    merged = meta.reset_index().merge(bracket_probs, on="game_id", how="inner")
    idx = merged["index"].values
    prob_cols = [c for c in bracket_probs.columns if c.startswith("p_")]

    if limit is not None:
        rng = np.random.RandomState(RANDOM_STATE)
        sel = rng.choice(len(idx), size=min(limit, len(idx)), replace=False)
        idx = idx[sel]
        merged = merged.iloc[sel]

    return (
        tokens[idx], lengths[idx],
        merged[prob_cols].values.astype(np.float32),
        merged["avg_elo"].values.astype(np.float32),
        prob_cols,
    )


def run_epoch(model, loader, optimizer, criterion, train: bool, grad_clip: float = 0.0):
    model.train(train)
    total_loss, n = 0.0, 0
    for tok, length, bprobs, target in loader:
        if train:
            optimizer.zero_grad()
        pred = model(tok, length, bprobs)
        loss = criterion(pred, target)
        if train:
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        total_loss += loss.item() * len(target)
        n += len(target)
    return total_loss / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="subsample size, for quick prototyping")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--embed-dim", type=int, default=48)
    ap.add_argument("--hidden-dim", type=int, default=96)
    ap.add_argument("--max-len", type=int, default=None, help="further truncate sequences at load time")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lr-min", type=float, default=1e-5, help="cosine schedule floor")
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--model-tag", type=str, default="v7", help="output filename tag - use a NEW tag to avoid overwriting a model in use")
    args = ap.parse_args()

    torch.manual_seed(RANDOM_STATE)
    torch.set_num_threads(args.threads)
    print(f"torch using {torch.get_num_threads()} threads")

    print("Loading data...")
    t0 = time.time()
    tokens, lengths, bracket_probs, targets, prob_cols = load_data(args.limit)
    if args.max_len is not None:
        tokens = tokens[:, :args.max_len]
        lengths = np.minimum(lengths, args.max_len)
    print(f"  {len(targets):,} games loaded in {time.time()-t0:.1f}s (seq_len={tokens.shape[1]})")

    vocab = json.loads((NN_DIR / "vocab.json").read_text())
    vocab_size = len(vocab)

    n = len(targets)
    rng = np.random.RandomState(RANDOM_STATE)
    perm = rng.permutation(n)
    n_test = int(n * 0.1)
    n_val = int(n * 0.1)
    test_idx, val_idx, train_idx = perm[:n_test], perm[n_test:n_test + n_val], perm[n_test + n_val:]

    # Normalize the target using train-split stats only - raw Elo (400-3000)
    # is a bad scale for a freshly-initialized network at lr=1e-3; without
    # this the model converges far too slowly to be useful in a few epochs.
    elo_mean = float(targets[train_idx].mean())
    elo_std = float(targets[train_idx].std())
    targets_norm = (targets - elo_mean) / elo_std

    def subset(idx):
        return ChessGameDataset(tokens[idx], lengths[idx], bracket_probs[idx], targets_norm[idx])

    train_ds, val_ds, test_ds = subset(train_idx), subset(val_idx), subset(test_idx)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=512)
    test_loader = DataLoader(test_ds, batch_size=512)
    print(f"  train={len(train_ds):,} val={len(val_ds):,} test={len(test_ds):,}")

    model = EloGRU(vocab_size=vocab_size, embed_dim=args.embed_dim, hidden_dim=args.hidden_dim, bracket_dim=len(prob_cols))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr_min)
    criterion = nn.L1Loss()  # trains directly on MAE

    ckpt_path = MODELS_DIR / f"nn_{args.model_tag}_best.pt"
    metrics_path = MODELS_DIR / f"nn_{args.model_tag}_metrics.json"
    print(f"checkpoint -> {ckpt_path}", flush=True)

    print(f"\nTraining for {args.epochs} epochs (batch_size={args.batch_size}, lr={args.lr}->{args.lr_min}, "
          f"weight_decay={args.weight_decay})...", flush=True)
    print(f"  target normalization: mean={elo_mean:.1f} std={elo_std:.1f}", flush=True)
    best_val = float("inf")
    best_epoch = 0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_mae = run_epoch(model, train_loader, optimizer, criterion, train=True, grad_clip=args.grad_clip) * elo_std
        with torch.no_grad():
            val_mae = run_epoch(model, val_loader, optimizer, criterion, train=False) * elo_std
        cur_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        improved = val_mae < best_val
        print(f"  epoch {epoch}: train_MAE={train_mae:.1f} val_MAE={val_mae:.1f} lr={cur_lr:.2e} "
              f"({time.time()-t0:.1f}s){' *' if improved else ''}", flush=True)
        if improved:
            best_val = val_mae
            best_epoch = epoch
            torch.save(model.state_dict(), ckpt_path)

    model.load_state_dict(torch.load(ckpt_path, weights_only=True))
    with torch.no_grad():
        test_mae = run_epoch(model, test_loader, optimizer, criterion, train=False) * elo_std
    print(f"\nFinal test MAE (best val checkpoint, epoch {best_epoch}): {test_mae:.1f}", flush=True)

    metrics_path.write_text(json.dumps({
        "model": f"EloGRU_{args.model_tag}", "n_train": len(train_ds), "n_val": len(val_ds), "n_test": len(test_ds),
        "vocab_size": vocab_size, "epochs": args.epochs, "best_epoch": best_epoch, "best_val_mae": round(best_val, 1),
        "test_mae": round(test_mae, 1), "elo_mean": elo_mean, "elo_std": elo_std,
        "lr": args.lr, "lr_min": args.lr_min, "weight_decay": args.weight_decay,
        "embed_dim": args.embed_dim, "hidden_dim": args.hidden_dim,
    }, indent=2))
    print(f"Saved: {metrics_path}", flush=True)


if __name__ == "__main__":
    main()
