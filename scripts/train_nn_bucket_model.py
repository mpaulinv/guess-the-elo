# scripts/train_nn_bucket_model.py - GRU predicting each player's Elo bucket
"""Same move-embedding + GRU backbone as train_nn_model.py, but:
  - predicts white_elo and black_elo separately (two classification heads)
    instead of a single avg_elo regression target
  - classifies into 200-wide Elo buckets (400-600, 600-800, ..., 3000+)
    rather than regressing a continuous value - matches the ~200 MAE the
    regression model already achieves, so exact-bucket accuracy is a
    meaningful number instead of being dominated by noise finer than the
    model can actually resolve
  - drops games with fewer than --min-ply plies (default 10): too short to
    plausibly carry a skill signal (see scripts/analyze_length_vs_error.py)

Reuses the existing tokens.npy/lengths.npy/vocab.json/meta.parquet built by
build_move_vocab.py - does NOT regenerate them (nn_v12 and other scripts
depend on those exact files). White/black Elo comes from a small side table
(scripts/build_player_elo.py -> data/processed/nn/player_elo.parquet)
joined onto meta.parquet by game_id.

Must run under the Python 3.10 env (torch DLL issue on 3.13 on this
machine):
  "C:\\Users\\mario\\AppData\\Local\\Programs\\Python\\Python310\\python.exe" scripts/train_nn_bucket_model.py [--limit N] [--epochs N]
"""
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

NN_DIR = Path(__file__).resolve().parent.parent / "data" / "processed" / "nn"
BRACKET_PROBS_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "bracket_probs_full.parquet"
PLAYER_ELO_PATH = NN_DIR / "player_elo.parquet"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
RANDOM_STATE = 42

BUCKET_LO = 400
BUCKET_WIDTH = 200
NUM_BUCKETS = 14  # 400-600, 600-800, ..., 2800-3000, 3000+ (open-ended)
BUCKET_MIDPOINTS = np.array([BUCKET_LO + BUCKET_WIDTH * i + BUCKET_WIDTH / 2 for i in range(NUM_BUCKETS)])


def elo_to_bucket(elo: np.ndarray) -> np.ndarray:
    return np.clip((elo - BUCKET_LO) // BUCKET_WIDTH, 0, NUM_BUCKETS - 1).astype(np.int64)


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


class EloGRUClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=24, hidden_dim=48, bracket_dim=6, num_buckets=NUM_BUCKETS,
                 mlp_hidden=64, gru_layers=1, bidirectional=False, dropout=0.2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(
            embed_dim, hidden_dim, num_layers=gru_layers, batch_first=True,
            bidirectional=bidirectional, dropout=dropout if gru_layers > 1 else 0.0,
        )
        gru_out_dim = hidden_dim * (2 if bidirectional else 1)
        self.trunk = nn.Sequential(
            nn.Linear(gru_out_dim + bracket_dim, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.white_head = nn.Linear(mlp_hidden, num_buckets)
        self.black_head = nn.Linear(mlp_hidden, num_buckets)
        self.num_directions = 2 if bidirectional else 1

    def forward(self, tokens, lengths, bracket_probs):
        emb = self.embed(tokens)
        packed = nn.utils.rnn.pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h_n = self.gru(packed)
        # h_n: [num_layers * num_directions, batch, hidden_dim] - take the last
        # layer's direction(s) and concat if bidirectional.
        h_n = h_n.view(-1, self.num_directions, h_n.size(1), h_n.size(2))[-1]
        h_last = h_n.transpose(0, 1).reshape(h_n.size(1), -1)
        trunk = self.trunk(torch.cat([h_last, bracket_probs], dim=1))
        return self.white_head(trunk), self.black_head(trunk)


def load_data(min_ply: int, limit=None):
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

    white_bucket = elo_to_bucket(merged["white_elo"].to_numpy())
    black_bucket = elo_to_bucket(merged["black_elo"].to_numpy())
    bracket_probs = merged.select(prob_cols).to_numpy().astype(np.float32)
    game_ids = merged["game_id"].to_numpy()

    return tokens_full[idx], lengths_full[idx], bracket_probs, white_bucket, black_bucket, prob_cols, game_ids


def run_epoch(model, loader, optimizer, criterion, train: bool, grad_clip: float = 0.0):
    model.train(train)
    total_loss, n = 0.0, 0
    correct_w, correct_b, adj_w, adj_b = 0, 0, 0, 0
    abs_err_w, abs_err_b = 0.0, 0.0
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
            pred_w = white_logits.argmax(dim=1)
            pred_b = black_logits.argmax(dim=1)
            correct_w += (pred_w == white_y).sum().item()
            correct_b += (pred_b == black_y).sum().item()
            adj_w += ((pred_w - white_y).abs() <= 1).sum().item()
            adj_b += ((pred_b - black_y).abs() <= 1).sum().item()
            midpoints = torch.from_numpy(BUCKET_MIDPOINTS).float()
            abs_err_w += (midpoints[pred_w] - midpoints[white_y]).abs().sum().item()
            abs_err_b += (midpoints[pred_b] - midpoints[black_y]).abs().sum().item()

    return {
        "loss": total_loss / n,
        "acc_white": correct_w / n, "acc_black": correct_b / n,
        "adj_acc_white": adj_w / n, "adj_acc_black": adj_b / n,
        "mae_white": abs_err_w / n, "mae_black": abs_err_b / n,
    }


def fmt(m):
    return (f"loss={m['loss']:.3f} acc(w/b)={m['acc_white']:.3f}/{m['acc_black']:.3f} "
            f"adj_acc(w/b)={m['adj_acc_white']:.3f}/{m['adj_acc_black']:.3f} "
            f"mae(w/b)={m['mae_white']:.1f}/{m['mae_black']:.1f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="subsample size, for quick prototyping")
    ap.add_argument("--min-ply", type=int, default=10, help="drop games with fewer than this many plies")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--embed-dim", type=int, default=24)
    ap.add_argument("--hidden-dim", type=int, default=48)
    ap.add_argument("--mlp-hidden", type=int, default=64)
    ap.add_argument("--gru-layers", type=int, default=1)
    ap.add_argument("--bidirectional", action="store_true")
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--class-weighted", action="store_true", help="inverse-frequency class weighting to counter regression-to-the-mean on the crowded middle buckets")
    ap.add_argument("--weight-power", type=float, default=1.0, help="exponent applied to the inverse-frequency weights (1.0=raw, 0.5=sqrt-softened, 0=uniform/off)")
    ap.add_argument("--max-len", type=int, default=None, help="further truncate sequences at load time")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lr-min", type=float, default=1e-5, help="cosine schedule floor")
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--model-tag", type=str, default="bucket_v1", help="output filename tag - use a NEW tag to avoid overwriting a model in use")
    args = ap.parse_args()

    torch.manual_seed(RANDOM_STATE)
    torch.set_num_threads(args.threads)
    print(f"torch using {torch.get_num_threads()} threads")

    print(f"Loading data (min_ply={args.min_ply})...")
    t0 = time.time()
    tokens, lengths, bracket_probs, white_bucket, black_bucket, prob_cols, _game_ids = load_data(args.min_ply, args.limit)
    if args.max_len is not None:
        tokens = tokens[:, :args.max_len]
        lengths = np.minimum(lengths, args.max_len)
    print(f"  {len(white_bucket):,} games loaded in {time.time()-t0:.1f}s (seq_len={tokens.shape[1]})")

    vocab = json.loads((NN_DIR / "vocab.json").read_text())
    vocab_size = len(vocab)

    n = len(white_bucket)
    rng = np.random.RandomState(RANDOM_STATE)
    perm = rng.permutation(n)
    n_test = int(n * 0.1)
    n_val = int(n * 0.1)
    test_idx, val_idx, train_idx = perm[:n_test], perm[n_test:n_test + n_val], perm[n_test + n_val:]

    def subset(idx):
        return ChessGameDataset(tokens[idx], lengths[idx], bracket_probs[idx], white_bucket[idx], black_bucket[idx])

    train_ds, val_ds, test_ds = subset(train_idx), subset(val_idx), subset(test_idx)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=512)
    test_loader = DataLoader(test_ds, batch_size=512)
    print(f"  train={len(train_ds):,} val={len(val_ds):,} test={len(test_ds):,}")

    model = EloGRUClassifier(
        vocab_size=vocab_size, embed_dim=args.embed_dim, hidden_dim=args.hidden_dim, bracket_dim=len(prob_cols),
        mlp_hidden=args.mlp_hidden, gru_layers=args.gru_layers, bidirectional=args.bidirectional, dropout=args.dropout,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr_min)

    class_weights = None
    if args.class_weighted:
        # Inverse-frequency ("balanced") weighting computed on the TRAIN split
        # only: w_c = (N / (K * n_c)) ** weight_power. ~70% of games sit in a
        # handful of middle buckets, so plain cross-entropy is minimized by
        # hedging toward them - this upweights the rare extreme buckets so
        # the loss stops being dominated by the crowded middle. Raw (power=1)
        # inverse frequency produces very extreme ratios (200x+) between the
        # rarest and most common buckets, which destabilizes training at
        # this data scale - weight_power < 1 (e.g. 0.5 = sqrt) softens that.
        combined = np.concatenate([white_bucket[train_idx], black_bucket[train_idx]])
        counts = np.clip(np.bincount(combined, minlength=NUM_BUCKETS).astype(np.float64), 1, None)
        weights = (counts.sum() / (NUM_BUCKETS * counts)) ** args.weight_power
        class_weights = torch.tensor(weights, dtype=torch.float32)
        print(f"  class weights (power={args.weight_power}, bucket 0-{NUM_BUCKETS-1}): {np.round(weights, 2).tolist()}", flush=True)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    ckpt_path = MODELS_DIR / f"nn_{args.model_tag}_best.pt"
    metrics_path = MODELS_DIR / f"nn_{args.model_tag}_metrics.json"
    print(f"checkpoint -> {ckpt_path}", flush=True)

    print(f"\nTraining for {args.epochs} epochs (batch_size={args.batch_size}, lr={args.lr}->{args.lr_min}, "
          f"weight_decay={args.weight_decay})...", flush=True)
    best_val_loss = float("inf")
    best_epoch = 0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_m = run_epoch(model, train_loader, optimizer, criterion, train=True, grad_clip=args.grad_clip)
        with torch.no_grad():
            val_m = run_epoch(model, val_loader, optimizer, criterion, train=False)
        cur_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        improved = val_m["loss"] < best_val_loss
        print(f"  epoch {epoch}: train[{fmt(train_m)}] val[{fmt(val_m)}] lr={cur_lr:.2e} "
              f"({time.time()-t0:.1f}s){' *' if improved else ''}", flush=True)
        if improved:
            best_val_loss = val_m["loss"]
            best_epoch = epoch
            torch.save(model.state_dict(), ckpt_path)

    model.load_state_dict(torch.load(ckpt_path, weights_only=True))
    with torch.no_grad():
        test_m = run_epoch(model, test_loader, optimizer, criterion, train=False)
    print(f"\nFinal test metrics (best val checkpoint, epoch {best_epoch}): {fmt(test_m)}", flush=True)

    metrics_path.write_text(json.dumps({
        "model": f"EloGRUClassifier_{args.model_tag}", "n_train": len(train_ds), "n_val": len(val_ds), "n_test": len(test_ds),
        "vocab_size": vocab_size, "epochs": args.epochs, "best_epoch": best_epoch, "min_ply": args.min_ply,
        "num_buckets": NUM_BUCKETS, "bucket_width": BUCKET_WIDTH, "bucket_lo": BUCKET_LO,
        "best_val_loss": round(best_val_loss, 4), "test_metrics": test_m,
        "lr": args.lr, "lr_min": args.lr_min, "weight_decay": args.weight_decay,
        "embed_dim": args.embed_dim, "hidden_dim": args.hidden_dim, "mlp_hidden": args.mlp_hidden,
        "gru_layers": args.gru_layers, "bidirectional": args.bidirectional, "dropout": args.dropout,
        "class_weighted": args.class_weighted, "weight_power": args.weight_power,
    }, indent=2))
    print(f"Saved: {metrics_path}", flush=True)


if __name__ == "__main__":
    main()
