# scripts/train_nn_board_aware_experiment.py - does real board state help?
"""Small side experiment: the GRU only ever sees opaque SAN move tokens - it
never sees the actual board. "Nxe5" is the same vocab entry whether it
captures a pawn or a queen, so per-ply MATERIAL BALANCE (which requires
knowing what was actually on the target square) is genuinely new information
the token stream can't recover on its own - unlike e.g. "is this a capture",
which is already implicit in which token appears (a capture and non-capture
SAN string are different vocab entries).

Trains a token-only baseline and a token+material-balance variant on the same
small subsample, same epochs, for a direct comparison. Runs single-threaded
(--threads 1 default) so it doesn't compete with a larger run for CPU.

Usage: python scripts/train_nn_board_aware_experiment.py [--limit N] [--epochs N]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import chess
import numpy as np
import polars as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_nn_bucket_model import load_data, NUM_BUCKETS, NN_DIR, MODELS_DIR, RANDOM_STATE  # noqa: E402
from src.move_features import parse_san_moves, PIECE_VALUES  # noqa: E402

DATA_CSV = Path(__file__).resolve().parent.parent / "data" / "enhanced_extraction" / "enhanced_experiment_20250620_203308.csv"
MAX_LEN = 140


def compute_material_balance(move_sequence: str, max_len: int = MAX_LEN) -> np.ndarray:
    """Per-ply running material balance (White POV, pawn=1..queen=9), scaled
    down and padded/truncated to max_len. Requires actual board replay -
    board.piece_at(move.to_square) is the piece BEING captured, which the SAN
    token itself doesn't encode."""
    san_moves = parse_san_moves(move_sequence)[:max_len]
    board = chess.Board()
    balance = 0.0
    out = np.zeros(max_len, dtype=np.float32)
    try:
        for i, san in enumerate(san_moves):
            color = board.turn
            move = board.parse_san(san)
            if board.is_capture(move):
                if board.is_en_passant(move):
                    captured_value = PIECE_VALUES[chess.PAWN]
                else:
                    captured_piece = board.piece_at(move.to_square)
                    captured_value = PIECE_VALUES[captured_piece.piece_type] if captured_piece else 0
                balance += captured_value if color == chess.WHITE else -captured_value
            board.push(move)
            out[i] = balance
    except (ValueError, AssertionError):
        pass  # keep whatever was computed before the parse failure
    return out / 9.0


def get_move_sequences(game_ids: np.ndarray) -> list[str]:
    id_order = pl.DataFrame({"game_id": game_ids, "order": np.arange(len(game_ids))})
    seqs = (
        pl.scan_csv(DATA_CSV)
        .select(["game_id", "move_sequence"])
        .filter(pl.col("game_id").is_in(game_ids.tolist()))
        .collect(engine="streaming")
    )
    merged = id_order.join(seqs, on="game_id", how="left").sort("order")
    return merged["move_sequence"].to_list()


class ChessGameDataset(Dataset):
    def __init__(self, tokens, lengths, bracket_probs, material, white_bucket, black_bucket):
        self.tokens = torch.from_numpy(tokens).long()
        self.lengths = torch.from_numpy(lengths).long().clamp(min=1)
        self.bracket_probs = torch.from_numpy(bracket_probs).float()
        self.material = torch.from_numpy(material).float()
        self.white_bucket = torch.from_numpy(white_bucket).long()
        self.black_bucket = torch.from_numpy(black_bucket).long()

    def __len__(self):
        return len(self.white_bucket)

    def __getitem__(self, idx):
        return (self.tokens[idx], self.lengths[idx], self.bracket_probs[idx],
                self.material[idx], self.white_bucket[idx], self.black_bucket[idx])


class EloGRU(nn.Module):
    """extra_dim=0 -> token-only baseline. extra_dim=1 -> concatenates
    per-ply material balance onto the token embedding before the GRU."""
    def __init__(self, vocab_size, embed_dim, hidden_dim, bracket_dim, extra_dim=0, num_buckets=NUM_BUCKETS, mlp_hidden=64, dropout=0.2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.extra_dim = extra_dim
        self.gru = nn.GRU(embed_dim + extra_dim, hidden_dim, batch_first=True)
        self.trunk = nn.Sequential(nn.Linear(hidden_dim + bracket_dim, mlp_hidden), nn.ReLU(), nn.Dropout(dropout))
        self.white_head = nn.Linear(mlp_hidden, num_buckets)
        self.black_head = nn.Linear(mlp_hidden, num_buckets)

    def forward(self, tokens, lengths, bracket_probs, material):
        emb = self.embed(tokens)
        if self.extra_dim:
            emb = torch.cat([emb, material.unsqueeze(-1)], dim=-1)
        packed = nn.utils.rnn.pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h_n = self.gru(packed)
        trunk = self.trunk(torch.cat([h_n[-1], bracket_probs], dim=1))
        return self.white_head(trunk), self.black_head(trunk)


def run_epoch(model, loader, optimizer, criterion, use_material, train, grad_clip=1.0):
    model.train(train)
    total_loss, n = 0.0, 0
    correct_w, correct_b, adj_w, adj_b = 0, 0, 0, 0
    for tok, length, bprobs, material, white_y, black_y in loader:
        if not use_material:
            material = material * 0.0
        if train:
            optimizer.zero_grad()
        white_logits, black_logits = model(tok, length, bprobs, material)
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


def train_variant(name, use_material, tokens, lengths, bracket_probs, material, white_bucket, black_bucket,
                   prob_cols, vocab_size, train_idx, val_idx, test_idx, epochs, class_weights, args):
    def subset(idx):
        return ChessGameDataset(tokens[idx], lengths[idx], bracket_probs[idx], material[idx], white_bucket[idx], black_bucket[idx])

    train_loader = DataLoader(subset(train_idx), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(subset(val_idx), batch_size=512)
    test_loader = DataLoader(subset(test_idx), batch_size=512)

    model = EloGRU(vocab_size, args.embed_dim, args.hidden_dim, len(prob_cols), extra_dim=1 if use_material else 0)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=args.lr_min)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    print(f"\n=== {name} ===", flush=True)
    best_val_loss, best_state = float("inf"), None
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_m = run_epoch(model, train_loader, optimizer, criterion, use_material, train=True)
        with torch.no_grad():
            val_m = run_epoch(model, val_loader, optimizer, criterion, use_material, train=False)
        scheduler.step()
        improved = val_m["loss"] < best_val_loss
        print(f"  epoch {epoch}: train[{fmt(train_m)}] val[{fmt(val_m)}] ({time.time()-t0:.1f}s){' *' if improved else ''}", flush=True)
        if improved:
            best_val_loss = val_m["loss"]
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    with torch.no_grad():
        test_m = run_epoch(model, test_loader, optimizer, criterion, use_material, train=False)
    print(f"  FINAL {name} test: {fmt(test_m)}", flush=True)
    return test_m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=20000)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--threads", type=int, default=1)
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

    print("Computing per-ply material balance via board replay (single-threaded, will take a few minutes)...")
    t0 = time.time()
    move_sequences = get_move_sequences(game_ids)
    material = np.stack([compute_material_balance(seq) for seq in move_sequences])
    print(f"  done in {time.time()-t0:.1f}s")

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

    results = {}
    for name, use_material in [("token-only baseline", False), ("token + material-balance", True)]:
        results[name] = train_variant(
            name, use_material, tokens, lengths, bracket_probs, material, white_bucket, black_bucket,
            prob_cols, vocab_size, train_idx, val_idx, test_idx, args.epochs, class_weights, args,
        )

    print("\n=== SUMMARY ===")
    for name, m in results.items():
        print(f"{name:28s} {fmt(m)}")

    out_path = MODELS_DIR / "board_aware_experiment_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
