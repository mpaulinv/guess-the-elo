# scripts/train_nn_bracket_experts_moe.py - 8 independent bracket experts + a
# trainable combiner, moves + clock-time only (no XGBoost/Lichess-analysis
# features anywhere)
"""8 fully independent GRU networks ("experts"), each its own embedding+GRU+
head, all consuming the same (move tokens, per-ply time-spent) input - no
bracket_probs, no engine features, nothing derived from Lichess's own
analysis. Each expert is trained with its LOSS MASKED to its own ~400-point
Elo bracket (with overlap to neighboring experts) - "expertise" comes purely
from which ground-truth examples get gradient, not from any pre-computed
routing/classifier. A trainable combiner then takes every expert's likelihood
distribution + derived point-estimate guess and learns the final prediction
end-to-end (gradient flows back into the experts too, not detached).

Brackets (400-wide, 350 stride, 50-point overlap):
  E0 400-800   E1 750-1150  E2 1100-1500 E3 1450-1850
  E4 1800-2200 E5 2150-2550 E6 2500-2900 E7 2850-3250

Usage: python scripts/train_nn_bracket_experts_moe.py [--limit N] [--epochs N]
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
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.time_features import parse_moves_with_clock, parse_time_control, _clk_to_seconds  # noqa: E402

NUM_BUCKETS = 14
BUCKET_LO, BUCKET_WIDTH = 400, 200
BUCKET_MIDPOINTS = np.array([BUCKET_LO + BUCKET_WIDTH * i + BUCKET_WIDTH / 2 for i in range(NUM_BUCKETS)])
MAX_LEN = 140
TIME_SCALE = 30.0
RANDOM_STATE = 42

NUM_EXPERTS = 8
BRACKET_WIDTH, BRACKET_STRIDE = 400, 350
BRACKET_RANGES = [(BUCKET_LO + i * BRACKET_STRIDE, BUCKET_LO + i * BRACKET_STRIDE + BRACKET_WIDTH) for i in range(NUM_EXPERTS)]


def elo_to_bucket(elo: np.ndarray) -> np.ndarray:
    return np.clip((elo - BUCKET_LO) // BUCKET_WIDTH, 0, NUM_BUCKETS - 1).astype(np.int64)


def bracket_masks(elo: np.ndarray) -> np.ndarray:
    """[N, NUM_EXPERTS] float32, 1.0 where elo falls in that expert's bracket."""
    masks = np.zeros((len(elo), NUM_EXPERTS), dtype=np.float32)
    for i, (lo, hi) in enumerate(BRACKET_RANGES):
        masks[:, i] = ((elo >= lo) & (elo < hi)).astype(np.float32)
    return masks


def compute_time_spent(move_sequence: str, time_control: str, max_len: int = MAX_LEN):
    out = np.zeros(max_len, dtype=np.float32)
    tc = parse_time_control(time_control)
    if tc is None:
        return out
    base_seconds, increment = tc
    if base_seconds <= 0:
        return out
    pairs = parse_moves_with_clock(move_sequence)[:max_len]
    clocks = [_clk_to_seconds(c) for _, c in pairs]
    prev = {True: base_seconds, False: base_seconds}
    for i, c in enumerate(clocks):
        if c is None:
            continue
        is_white = (i % 2 == 0)
        spent = max(0.0, prev[is_white] + increment - c)
        out[i] = spent / TIME_SCALE
        prev[is_white] = c
    return out


class BracketDataset(Dataset):
    def __init__(self, tokens, lengths, time_spent, white_bucket, black_bucket, white_masks, black_masks):
        self.tokens = torch.from_numpy(tokens).long()
        self.lengths = torch.from_numpy(lengths).long().clamp(min=1)
        self.time_spent = torch.from_numpy(time_spent).float()
        self.white_bucket = torch.from_numpy(white_bucket).long()
        self.black_bucket = torch.from_numpy(black_bucket).long()
        self.white_masks = torch.from_numpy(white_masks).float()
        self.black_masks = torch.from_numpy(black_masks).float()

    def __len__(self):
        return len(self.white_bucket)

    def __getitem__(self, idx):
        return (self.tokens[idx], self.lengths[idx], self.time_spent[idx],
                self.white_bucket[idx], self.black_bucket[idx],
                self.white_masks[idx], self.black_masks[idx])


class Expert(nn.Module):
    """One fully independent embedding+GRU+head. Input: tokens + per-ply time-spent."""
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_buckets=NUM_BUCKETS, mlp_hidden=48, dropout=0.2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim + 1, hidden_dim, batch_first=True)
        self.white_head = nn.Sequential(nn.Linear(hidden_dim, mlp_hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(mlp_hidden, num_buckets))
        self.black_head = nn.Sequential(nn.Linear(hidden_dim, mlp_hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(mlp_hidden, num_buckets))

    def forward(self, tokens, lengths, time_spent):
        emb = self.embed(tokens)
        x = torch.cat([emb, time_spent.unsqueeze(-1)], dim=-1)
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h_n = self.gru(packed)
        h = h_n[-1]
        return self.white_head(h), self.black_head(h)


class BracketExpertsMoE(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_experts=NUM_EXPERTS, num_buckets=NUM_BUCKETS,
                 expert_mlp_hidden=48, combiner_hidden=96, dropout=0.2):
        super().__init__()
        self.experts = nn.ModuleList([
            Expert(vocab_size, embed_dim, hidden_dim, num_buckets, expert_mlp_hidden, dropout)
            for _ in range(num_experts)
        ])
        self.num_experts = num_experts
        self.num_buckets = num_buckets
        combiner_in = num_experts * (num_buckets + 1)  # each expert: full likelihood + scalar guess
        self.register_buffer("midpoints", torch.from_numpy(BUCKET_MIDPOINTS).float())
        self.white_combiner = nn.Sequential(nn.Linear(combiner_in, combiner_hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(combiner_hidden, num_buckets))
        self.black_combiner = nn.Sequential(nn.Linear(combiner_in, combiner_hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(combiner_hidden, num_buckets))

    def forward(self, tokens, lengths, time_spent):
        white_logits_list, black_logits_list = [], []
        white_probs_list, black_probs_list = [], []
        for expert in self.experts:
            wl, bl = expert(tokens, lengths, time_spent)
            white_logits_list.append(wl)
            black_logits_list.append(bl)
            white_probs_list.append(torch.softmax(wl, dim=-1))
            black_probs_list.append(torch.softmax(bl, dim=-1))

        white_probs = torch.stack(white_probs_list, dim=1)  # [B, E, K]
        black_probs = torch.stack(black_probs_list, dim=1)
        white_guess = (white_probs * self.midpoints).sum(dim=-1)  # [B, E]
        black_guess = (black_probs * self.midpoints).sum(dim=-1)

        white_combiner_in = torch.cat([white_probs.flatten(start_dim=1), white_guess], dim=1)
        black_combiner_in = torch.cat([black_probs.flatten(start_dim=1), black_guess], dim=1)
        final_white = self.white_combiner(white_combiner_in)
        final_black = self.black_combiner(black_combiner_in)

        return final_white, final_black, white_logits_list, black_logits_list


def load_raw_sample(data_csvs: list[Path], min_ply: int, limit: int):
    cols = ["game_id", "move_sequence", "time_control", "white_elo", "black_elo",
            "is_quality_game", "time_class", "calculated_ply_count"]
    lazies = [
        pl.scan_csv(p).select(cols)
        .filter(pl.col("is_quality_game") & (pl.col("time_class") != "unknown") & (pl.col("calculated_ply_count") >= min_ply))
        for p in data_csvs
    ]
    lazy = pl.concat(lazies) if len(lazies) > 1 else lazies[0]
    df = lazy.collect(engine="streaming")
    if limit is not None and len(df) > limit:
        df = df.sample(n=limit, seed=RANDOM_STATE)
    return df


def tokenize_and_prepare(df, vocab, max_len=MAX_LEN):
    from src.move_features import parse_san_moves
    n = len(df)
    tokens = np.zeros((n, max_len), dtype=np.int64)
    lengths = np.zeros(n, dtype=np.int64)
    time_spent = np.zeros((n, max_len), dtype=np.float32)
    move_seqs = df["move_sequence"].to_list()
    time_controls = df["time_control"].to_list()
    for i, (seq, tc) in enumerate(zip(move_seqs, time_controls)):
        moves = parse_san_moves(seq)[:max_len]
        ids = [vocab.get(m, 1) for m in moves]  # 1 = <UNK>
        tokens[i, :len(ids)] = ids
        lengths[i] = max(1, len(ids))
        time_spent[i] = compute_time_spent(seq, tc, max_len)
    return tokens, lengths, time_spent


def build_vocab(df, max_vocab=20000):
    from collections import Counter
    from src.move_features import parse_san_moves
    counter = Counter()
    for seq in df["move_sequence"].to_list():
        counter.update(parse_san_moves(seq))
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for tok, _ in counter.most_common(max_vocab):
        vocab[tok] = len(vocab)
    return vocab


def run_epoch(model, loader, optimizer, criterion, train, expert_loss_weight=1.0, grad_clip=1.0):
    model.train(train)
    total_loss, n = 0.0, 0
    correct_w, correct_b, adj_w, adj_b = 0, 0, 0, 0
    for tok, length, tspent, white_y, black_y, white_masks, black_masks in loader:
        if train:
            optimizer.zero_grad()
        final_white, final_black, white_logits_list, black_logits_list = model(tok, length, tspent)

        combiner_loss = criterion(final_white, white_y) + criterion(final_black, black_y)

        expert_loss = 0.0
        ce_none = nn.CrossEntropyLoss(reduction="none")
        for i in range(model.num_experts):
            w_ce = ce_none(white_logits_list[i], white_y)
            b_ce = ce_none(black_logits_list[i], black_y)
            w_mask = white_masks[:, i]
            b_mask = black_masks[:, i]
            w_denom = w_mask.sum().clamp(min=1)
            b_denom = b_mask.sum().clamp(min=1)
            expert_loss = expert_loss + (w_ce * w_mask).sum() / w_denom + (b_ce * b_mask).sum() / b_denom
        expert_loss = expert_loss / model.num_experts  # keep on a comparable scale to combiner_loss, not 8x it

        loss = combiner_loss + expert_loss_weight * expert_loss
        if train:
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        bs = len(white_y)
        total_loss += loss.item() * bs
        n += bs
        with torch.no_grad():
            pred_w, pred_b = final_white.argmax(1), final_black.argmax(1)
            correct_w += (pred_w == white_y).sum().item()
            correct_b += (pred_b == black_y).sum().item()
            adj_w += ((pred_w - white_y).abs() <= 1).sum().item()
            adj_b += ((pred_b - black_y).abs() <= 1).sum().item()
    return {"loss": total_loss / n, "acc_w": correct_w / n, "acc_b": correct_b / n,
            "adj_w": adj_w / n, "adj_b": adj_b / n}


def fmt(m):
    return f"loss={m['loss']:.3f} acc(w/b)={m['acc_w']:.3f}/{m['acc_b']:.3f} adj_acc(w/b)={m['adj_w']:.3f}/{m['adj_b']:.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-csv", type=str, required=True, action="append", help="repeatable - pass multiple times to combine sources")
    ap.add_argument("--limit", type=int, default=20000)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--embed-dim", type=int, default=24)
    ap.add_argument("--hidden-dim", type=int, default=48)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lr-min", type=float, default=1e-5)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--weight-power", type=float, default=0.5)
    ap.add_argument("--expert-loss-weight", type=float, default=1.0, help="relative weight of the (now-normalized) per-expert bracket loss vs the combiner loss")
    ap.add_argument("--min-ply", type=int, default=10)
    ap.add_argument("--model-tag", type=str, default="bracket_moe_calib")
    args = ap.parse_args()

    torch.manual_seed(RANDOM_STATE)
    torch.set_num_threads(args.threads)
    print(f"torch using {torch.get_num_threads()} thread(s)")

    data_csvs = [Path(p) for p in args.data_csv]
    print(f"Loading up to {args.limit} games from {len(data_csvs)} source(s): {[str(p) for p in data_csvs]}...")
    t0 = time.time()
    df = load_raw_sample(data_csvs, args.min_ply, args.limit)
    print(f"  {len(df):,} games loaded in {time.time()-t0:.1f}s")

    print("Building vocab (fresh, this-run only)...")
    t0 = time.time()
    vocab = build_vocab(df)
    print(f"  vocab_size={len(vocab):,} in {time.time()-t0:.1f}s")

    print("Tokenizing + computing per-ply clock time...")
    t0 = time.time()
    tokens, lengths, time_spent = tokenize_and_prepare(df, vocab)
    print(f"  done in {time.time()-t0:.1f}s")

    white_elo = df["white_elo"].to_numpy().astype(np.float64)
    black_elo = df["black_elo"].to_numpy().astype(np.float64)
    white_bucket = elo_to_bucket(white_elo)
    black_bucket = elo_to_bucket(black_elo)
    white_masks = bracket_masks(white_elo)
    black_masks = bracket_masks(black_elo)
    print(f"  bracket coverage (white) per expert: {white_masks.sum(axis=0).astype(int).tolist()}")

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
        return BracketDataset(tokens[idx], lengths[idx], time_spent[idx], white_bucket[idx], black_bucket[idx],
                               white_masks[idx], black_masks[idx])

    train_loader = DataLoader(subset(train_idx), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(subset(val_idx), batch_size=512)
    test_loader = DataLoader(subset(test_idx), batch_size=512)

    model = BracketExpertsMoE(vocab_size=len(vocab), embed_dim=args.embed_dim, hidden_dim=args.hidden_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr_min)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    print(f"\nTraining {NUM_EXPERTS}-expert bracket MoE for {args.epochs} epochs...", flush=True)
    best_val_loss, best_state = float("inf"), None
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_m = run_epoch(model, train_loader, optimizer, criterion, train=True, expert_loss_weight=args.expert_loss_weight)
        with torch.no_grad():
            val_m = run_epoch(model, val_loader, optimizer, criterion, train=False, expert_loss_weight=args.expert_loss_weight)
        scheduler.step()
        improved = val_m["loss"] < best_val_loss
        print(f"  epoch {epoch}: train[{fmt(train_m)}] val[{fmt(val_m)}] ({time.time()-t0:.1f}s){' *' if improved else ''}", flush=True)
        if improved:
            best_val_loss = val_m["loss"]
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    with torch.no_grad():
        test_m = run_epoch(model, test_loader, optimizer, criterion, train=False, expert_loss_weight=args.expert_loss_weight)
    print(f"\nFinal test: {fmt(test_m)}", flush=True)

    from train_nn_bucket_model import MODELS_DIR
    metrics_path = MODELS_DIR / f"nn_{args.model_tag}_metrics.json"
    metrics_path.write_text(json.dumps({
        "model": f"BracketExpertsMoE_{args.model_tag}", "n_train": len(train_idx), "n_val": len(val_idx), "n_test": len(test_idx),
        "vocab_size": len(vocab), "epochs": args.epochs, "test_metrics": test_m,
        "embed_dim": args.embed_dim, "hidden_dim": args.hidden_dim, "num_experts": NUM_EXPERTS,
        "bracket_ranges": BRACKET_RANGES,
    }, indent=2))
    print(f"Saved: {metrics_path}")


if __name__ == "__main__":
    main()
