# scripts/train_nn_moe_experiment.py - does per-band specialization help?
"""Tests the MoE hypothesis cheaply: instead of training a new gating network
from scratch (with the usual collapse risk - see load-balancing-loss
literature), reuse the EXISTING classifier_full_v6 bracket_probs directly as
fixed gate weights. Its 6 columns (Sub-beginner <800, Beginner 800-1200,
Intermediate 1200-1600, Advanced 1600-2000, Expert 2000-2400, Master+ 2400+)
are exactly the wide, data-balanced bands needed - the 14 narrow buckets used
elsewhere would starve a dedicated expert at the top (2800-3000 has only 402
total examples across the whole dataset; merging everything 2400+ into one
band gives that expert ~18k).

Architecture: shared token embedding + GRU trunk (same as EloGRUClassifier),
then 6 small per-band expert heads per color. Each expert outputs a full
14-bucket distribution; the final prediction is the bracket_probs-weighted
MIXTURE of the experts' probability distributions (not their logits - a
proper mixture-of-distributions, so NLLLoss on the log-mixture is the
correct objective, not cross-entropy on summed logits).

Compares against the plain single-head EloGRUClassifier (train_nn_bucket_model.py)
on the same subsample/epochs, plus a per-band accuracy breakdown - the whole
point of MoE is band-specific improvement, not necessarily a big aggregate
MAE change.

Usage: python scripts/train_nn_moe_experiment.py [--limit N] [--epochs N]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_nn_bucket_model import (  # noqa: E402
    load_data, EloGRUClassifier, NUM_BUCKETS, BUCKET_MIDPOINTS, MODELS_DIR, RANDOM_STATE,
)

BAND_NAMES = ["Sub-beginner (<800)", "Beginner (800-1200)", "Intermediate (1200-1600)",
              "Advanced (1600-2000)", "Expert (2000-2400)", "Master+ (2400+)"]


class ChessGameDataset(Dataset):
    def __init__(self, tokens, lengths, bracket_probs, white_bucket, black_bucket):
        self.tokens = torch.from_numpy(tokens).long()
        self.lengths = torch.from_numpy(lengths).long().clamp(min=1)
        self.bracket_probs = torch.from_numpy(bracket_probs).float()
        self.white_bucket = torch.from_numpy(white_bucket).long()
        self.black_bucket = torch.from_numpy(black_bucket).long()

    def __len__(self):
        return len(self.white_bucket)

    def __getitem__(self, idx):
        return (self.tokens[idx], self.lengths[idx], self.bracket_probs[idx],
                self.white_bucket[idx], self.black_bucket[idx])


class EloMoEClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, bracket_dim, num_buckets=NUM_BUCKETS, mlp_hidden=64, dropout=0.2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        combined_dim = hidden_dim + bracket_dim
        self.num_experts = bracket_dim
        self.white_experts = nn.ModuleList([
            nn.Sequential(nn.Linear(combined_dim, mlp_hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(mlp_hidden, num_buckets))
            for _ in range(self.num_experts)
        ])
        self.black_experts = nn.ModuleList([
            nn.Sequential(nn.Linear(combined_dim, mlp_hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(mlp_hidden, num_buckets))
            for _ in range(self.num_experts)
        ])

    def forward(self, tokens, lengths, bracket_probs):
        emb = self.embed(tokens)
        packed = nn.utils.rnn.pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h_n = self.gru(packed)
        combined = torch.cat([h_n[-1], bracket_probs], dim=1)
        gate = bracket_probs  # [B, num_experts], fixed - already sums to ~1

        white_logits = torch.stack([e(combined) for e in self.white_experts], dim=1)  # [B,E,K]
        black_logits = torch.stack([e(combined) for e in self.black_experts], dim=1)
        white_probs = torch.softmax(white_logits, dim=-1)
        black_probs = torch.softmax(black_logits, dim=-1)
        white_mix = (gate.unsqueeze(-1) * white_probs).sum(dim=1).clamp_min(1e-8)  # [B,K]
        black_mix = (gate.unsqueeze(-1) * black_probs).sum(dim=1).clamp_min(1e-8)
        return white_mix, black_mix  # probabilities, not logits


def run_epoch_moe(model, loader, optimizer, criterion, train, grad_clip=1.0):
    model.train(train)
    total_loss, n = 0.0, 0
    correct_w, correct_b, adj_w, adj_b = 0, 0, 0, 0
    for tok, length, bprobs, white_y, black_y in loader:
        if train:
            optimizer.zero_grad()
        white_mix, black_mix = model(tok, length, bprobs)
        loss = criterion(torch.log(white_mix), white_y) + criterion(torch.log(black_mix), black_y)
        if train:
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        bs = len(white_y)
        total_loss += loss.item() * bs
        n += bs
        with torch.no_grad():
            pred_w, pred_b = white_mix.argmax(1), black_mix.argmax(1)
            correct_w += (pred_w == white_y).sum().item()
            correct_b += (pred_b == black_y).sum().item()
            adj_w += ((pred_w - white_y).abs() <= 1).sum().item()
            adj_b += ((pred_b - black_y).abs() <= 1).sum().item()
    return {"loss": total_loss / n, "acc_w": correct_w / n, "acc_b": correct_b / n,
            "adj_w": adj_w / n, "adj_b": adj_b / n}


def run_epoch_baseline(model, loader, optimizer, criterion, train, grad_clip=1.0):
    model.train(train)
    total_loss, n = 0.0, 0
    correct_w, correct_b, adj_w, adj_b = 0, 0, 0, 0
    for tok, length, bprobs, white_y, black_y in loader:
        if train:
            optimizer.zero_grad()
        white_logits, black_logits = model(tok, length, bprobs)
        loss = criterion(white_logits, white_y) + criterion(black_logits, black_y)
        if train:
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        bs = len(white_y)
        total_loss += loss.item() * bs
        n += bs
        with torch.no_grad():
            pred_w, pred_b = white_logits.argmax(1), black_logits.argmax(1)
            correct_w += (pred_w == white_y).sum().item()
            correct_b += (pred_b == black_y).sum().item()
            adj_w += ((pred_w - white_y).abs() <= 1).sum().item()
            adj_b += ((pred_b - black_y).abs() <= 1).sum().item()
    return {"loss": total_loss / n, "acc_w": correct_w / n, "acc_b": correct_b / n,
            "adj_w": adj_w / n, "adj_b": adj_b / n}


def fmt(m):
    return f"loss={m['loss']:.3f} acc(w/b)={m['acc_w']:.3f}/{m['acc_b']:.3f} adj_acc(w/b)={m['adj_w']:.3f}/{m['adj_b']:.3f}"


def band_breakdown(model, is_moe, tokens, lengths, bracket_probs, white_bucket, black_bucket, idx, prob_cols):
    """Per-band (using the argmax bracket_probs band as a proxy for true band) accuracy, to see where gains concentrate."""
    tok_t = torch.from_numpy(tokens[idx]).long()
    len_t = torch.from_numpy(lengths[idx]).long().clamp(min=1)
    bp_t = torch.from_numpy(bracket_probs[idx]).float()
    with torch.no_grad():
        if is_moe:
            white_mix, black_mix = model(tok_t, len_t, bp_t)
        else:
            wl, bl = model(tok_t, len_t, bp_t)
            white_mix, black_mix = torch.softmax(wl, dim=-1), torch.softmax(bl, dim=-1)
    pred_w = white_mix.argmax(1).numpy()
    pred_b = black_mix.argmax(1).numpy()
    true_w, true_b = white_bucket[idx], black_bucket[idx]
    band_of_game = bracket_probs[idx].argmax(axis=1)  # which existing-classifier band each game falls in

    rows = []
    for b, name in enumerate(prob_cols):
        mask = band_of_game == b
        if mask.sum() == 0:
            continue
        acc = np.concatenate([pred_w[mask] == true_w[mask], pred_b[mask] == true_b[mask]]).mean()
        adj = np.concatenate([np.abs(pred_w[mask] - true_w[mask]) <= 1, np.abs(pred_b[mask] - true_b[mask]) <= 1]).mean()
        rows.append((name, int(mask.sum()), acc, adj))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=120000)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--embed-dim", type=int, default=48)
    ap.add_argument("--hidden-dim", type=int, default=96)
    ap.add_argument("--mlp-hidden", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lr-min", type=float, default=1e-5)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--weight-power", type=float, default=0.5)
    ap.add_argument("--min-ply", type=int, default=10)
    args = ap.parse_args()

    torch.manual_seed(RANDOM_STATE)
    torch.set_num_threads(args.threads)
    print(f"torch using {torch.get_num_threads()} thread(s)")

    print(f"Loading {args.limit} games...")
    tokens, lengths, bracket_probs, white_bucket, black_bucket, prob_cols, _game_ids = load_data(args.min_ply, args.limit)
    vocab_path = Path(__file__).resolve().parent.parent / "data" / "processed" / "nn" / "vocab.json"
    vocab_size = len(json.loads(vocab_path.read_text()))

    n = len(white_bucket)
    rng = np.random.RandomState(RANDOM_STATE)
    perm = rng.permutation(n)
    n_test, n_val = int(n * 0.1), int(n * 0.1)
    test_idx, val_idx, train_idx = perm[:n_test], perm[n_test:n_test + n_val], perm[n_test + n_val:]
    print(f"train={len(train_idx):,} val={len(val_idx):,} test={len(test_idx):,}")

    combined_targets = np.concatenate([white_bucket[train_idx], black_bucket[train_idx]])
    counts = np.clip(np.bincount(combined_targets, minlength=NUM_BUCKETS).astype(np.float64), 1, None)
    weights = (counts.sum() / (NUM_BUCKETS * counts)) ** args.weight_power
    class_weights = torch.tensor(weights, dtype=torch.float32)

    def subset(idx):
        return ChessGameDataset(tokens[idx], lengths[idx], bracket_probs[idx], white_bucket[idx], black_bucket[idx])

    train_loader = DataLoader(subset(train_idx), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(subset(val_idx), batch_size=512)
    test_loader = DataLoader(subset(test_idx), batch_size=512)

    results = {}
    models = {}
    for name, is_moe in [("baseline (single head)", False), ("MoE (6 bracket experts)", True)]:
        print(f"\n=== {name} ===")
        if is_moe:
            model = EloMoEClassifier(vocab_size, args.embed_dim, args.hidden_dim, len(prob_cols), mlp_hidden=args.mlp_hidden)
            run_epoch = run_epoch_moe
            criterion = nn.NLLLoss(weight=class_weights)
        else:
            model = EloGRUClassifier(vocab_size=vocab_size, embed_dim=args.embed_dim, hidden_dim=args.hidden_dim,
                                      bracket_dim=len(prob_cols), mlp_hidden=args.mlp_hidden)
            run_epoch = run_epoch_baseline
            criterion = nn.CrossEntropyLoss(weight=class_weights)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr_min)
        best_val_loss, best_state = float("inf"), None
        for epoch in range(1, args.epochs + 1):
            t0 = time.time()
            train_m = run_epoch(model, train_loader, optimizer, criterion, train=True)
            with torch.no_grad():
                val_m = run_epoch(model, val_loader, optimizer, criterion, train=False)
            scheduler.step()
            improved = val_m["loss"] < best_val_loss
            print(f"  epoch {epoch}: train[{fmt(train_m)}] val[{fmt(val_m)}] ({time.time()-t0:.1f}s){' *' if improved else ''}", flush=True)
            if improved:
                best_val_loss = val_m["loss"]
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

        model.load_state_dict(best_state)
        with torch.no_grad():
            test_m = run_epoch(model, test_loader, optimizer, criterion, train=False)
        print(f"  FINAL {name} test: {fmt(test_m)}")
        results[name] = test_m
        models[name] = (model, is_moe)

    print("\n=== SUMMARY ===")
    for name, m in results.items():
        print(f"{name:28s} {fmt(m)}")

    print("\n=== Per-band breakdown (band = argmax of existing bracket_probs classifier) ===")
    band_results = {}
    for name, (model, is_moe) in models.items():
        rows = band_breakdown(model, is_moe, tokens, lengths, bracket_probs, white_bucket, black_bucket, test_idx, prob_cols)
        band_results[name] = rows
        print(f"\n{name}:")
        for band_name, n_band, acc, adj in rows:
            print(f"  {band_name:28s} n={n_band:6d} acc={acc:.3f} adj_acc={adj:.3f}")

    out_path = MODELS_DIR / "moe_experiment_results.json"
    out_path.write_text(json.dumps({
        "aggregate": results,
        "per_band": {name: [{"band": r[0], "n": r[1], "acc": r[2], "adj_acc": r[3]} for r in rows]
                     for name, rows in band_results.items()},
    }, indent=2))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
