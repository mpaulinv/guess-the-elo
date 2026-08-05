# scripts/train_bracket_moe_gpu.py - GPU-ready, fully self-contained
"""8-expert bracket MoE (moves + clock-time only, no engine/XGBoost features),
adapted for GPU training on Colab or a rented GPU box. Fully self-contained -
no dependency on any other file in this repo. Loads pre-processed arrays
(built locally by scripts/preprocess_bracket_moe_data.py) rather than raw
CSVs - upload the data/processed/nn_bracket_moe/ directory alongside this
one file and you're ready to train.

Architecture (identical to scripts/train_nn_bracket_experts_moe.py, the
CPU-validated version this was derived from):
  - 8 fully independent GRU experts (own embedding + GRU + head each),
    each seeing every game's (move tokens, per-ply time-spent)
  - each expert's loss is MASKED to its own ~400-point Elo bracket (350
    stride, 50-point overlap) - "expertise" comes from which ground-truth
    examples get gradient, not from any pre-computed routing/classifier
  - a trainable combiner takes every expert's likelihood distribution +
    derived point-estimate guess and learns the final white/black Elo
    bucket prediction end-to-end (gradients flow back into the experts too)

Usage (Colab or SSH box with a CUDA GPU):
  python train_bracket_moe_gpu.py --data-dir data/processed/nn_bracket_moe --epochs 20

Checkpoints, per epoch, all in --out-dir:
  {tag}_last.pt        - overwritten every epoch (resume after any disconnect)
  {tag}_best.pt        - overwritten whenever val loss improves (used for final test eval)
  {tag}_epoch{NNN}.pt  - a standalone snapshot for THAT epoch specifically, never
                         overwritten - download whichever epoch you actually want
                         rather than whatever last/best happens to currently point
                         to, and resume/branch from any specific past epoch with
                         --resume path/to/{tag}_epoch007.pt
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset

NUM_BUCKETS = 14
BUCKET_LO, BUCKET_WIDTH = 400, 200
BUCKET_MIDPOINTS = np.array([BUCKET_LO + BUCKET_WIDTH * i + BUCKET_WIDTH / 2 for i in range(NUM_BUCKETS)])
NUM_EXPERTS = 8
RANDOM_STATE = 42


class BracketDataset(Dataset):
    def __init__(self, tokens, lengths, time_spent, white_bucket, black_bucket, white_masks, black_masks):
        # tokens stays int32 here (half the size of int64) - nn.Embedding only
        # needs int64 indices for the current batch, cast lazily in Expert.forward,
        # not for the whole multi-million-row array up front.
        self.tokens = torch.from_numpy(tokens)
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
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_buckets=NUM_BUCKETS, mlp_hidden=48, dropout=0.2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim + 1, hidden_dim, batch_first=True)
        self.white_head = nn.Sequential(nn.Linear(hidden_dim, mlp_hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(mlp_hidden, num_buckets))
        self.black_head = nn.Sequential(nn.Linear(hidden_dim, mlp_hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(mlp_hidden, num_buckets))

    def forward(self, tokens, lengths, time_spent):
        emb = self.embed(tokens.long())
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
        combiner_in = num_experts * (num_buckets + 1)
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

        white_probs = torch.stack(white_probs_list, dim=1)
        black_probs = torch.stack(black_probs_list, dim=1)
        white_guess = (white_probs * self.midpoints).sum(dim=-1)
        black_guess = (black_probs * self.midpoints).sum(dim=-1)

        white_combiner_in = torch.cat([white_probs.flatten(start_dim=1), white_guess], dim=1)
        black_combiner_in = torch.cat([black_probs.flatten(start_dim=1), black_guess], dim=1)
        final_white = self.white_combiner(white_combiner_in)
        final_black = self.black_combiner(black_combiner_in)

        return final_white, final_black, white_logits_list, black_logits_list


def run_epoch(model, loader, optimizer, criterion, device, train, expert_loss_weight=1.0, grad_clip=1.0):
    model.train(train)
    total_loss, n = 0.0, 0
    correct_w, correct_b, adj_w, adj_b = 0, 0, 0, 0
    ce_none = nn.CrossEntropyLoss(reduction="none")
    for tok, length, tspent, white_y, black_y, white_masks, black_masks in loader:
        tok, length, tspent = tok.to(device), length.to(device), tspent.to(device)
        white_y, black_y = white_y.to(device), black_y.to(device)
        white_masks, black_masks = white_masks.to(device), black_masks.to(device)

        if train:
            optimizer.zero_grad()
        final_white, final_black, white_logits_list, black_logits_list = model(tok, length, tspent)

        combiner_loss = criterion(final_white, white_y) + criterion(final_black, black_y)

        expert_loss = 0.0
        for i in range(model.num_experts):
            w_ce = ce_none(white_logits_list[i], white_y)
            b_ce = ce_none(black_logits_list[i], black_y)
            w_mask = white_masks[:, i]
            b_mask = black_masks[:, i]
            w_denom = w_mask.sum().clamp(min=1)
            b_denom = b_mask.sum().clamp(min=1)
            expert_loss = expert_loss + (w_ce * w_mask).sum() / w_denom + (b_ce * b_mask).sum() / b_denom
        expert_loss = expert_loss / model.num_experts

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


def _fixed_list_to_2d(record_batch, col_name, width, dtype):
    flat = record_batch.column(col_name).combine_chunks().flatten().to_numpy(zero_copy_only=False)
    return flat.reshape(-1, width).astype(dtype, copy=False)


def load_parquet_dataset(data_dir: Path, max_len: int, num_experts: int):
    """Reads scripts/preprocess_bracket_moe_data.py's dataset.parquet output
    into the same flat numpy arrays the rest of this script expects.

    Streams the file row-group by row-group (each ~50k rows, matching how
    preprocess_bracket_moe_data.py wrote it) straight into pre-allocated
    output arrays, instead of pq.read_table()-ing the whole file at once.
    A single read_table() + combine_chunks()/astype() on the full ~4M rows
    briefly held 3-4 full-size copies of the largest columns in memory at
    once (the source table, the combined-chunk copy, and the astype() copy
    - astype() always copies unless told not to) - comfortably past a 12GB
    Colab runtime even though the final arrays only total ~4.7GB. Streaming
    keeps only one row-group's small temporary buffers alive at a time."""
    needed_cols = ["tokens", "time_spent", "white_masks", "black_masks", "length", "white_bucket", "black_bucket"]
    pf = pq.ParquetFile(data_dir / "dataset.parquet")
    n = pf.metadata.num_rows

    tokens = np.empty((n, max_len), dtype=np.int32)
    time_spent = np.empty((n, max_len), dtype=np.float32)
    white_masks = np.empty((n, num_experts), dtype=np.float32)
    black_masks = np.empty((n, num_experts), dtype=np.float32)
    lengths = np.empty(n, dtype=np.int32)
    white_bucket = np.empty(n, dtype=np.int64)
    black_bucket = np.empty(n, dtype=np.int64)

    offset = 0
    for rg_idx in range(pf.num_row_groups):
        rg = pf.read_row_group(rg_idx, columns=needed_cols)
        m = rg.num_rows
        sl = slice(offset, offset + m)
        tokens[sl] = _fixed_list_to_2d(rg, "tokens", max_len, np.int32)
        time_spent[sl] = _fixed_list_to_2d(rg, "time_spent", max_len, np.float32)
        white_masks[sl] = _fixed_list_to_2d(rg, "white_masks", num_experts, np.float32)
        black_masks[sl] = _fixed_list_to_2d(rg, "black_masks", num_experts, np.float32)
        lengths[sl] = rg.column("length").to_numpy().astype(np.int32, copy=False)
        white_bucket[sl] = rg.column("white_bucket").to_numpy().astype(np.int64, copy=False)
        black_bucket[sl] = rg.column("black_bucket").to_numpy().astype(np.int64, copy=False)
        offset += m
        del rg
    return tokens, lengths, time_spent, white_bucket, black_bucket, white_masks, black_masks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=str, required=True, help="dir containing dataset.parquet + manifest.json from preprocess_bracket_moe_data.py")
    ap.add_argument("--out-dir", type=str, default="checkpoints")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--embed-dim", type=int, default=24)
    ap.add_argument("--hidden-dim", type=int, default=48)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lr-min", type=float, default=1e-5)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--weight-power", type=float, default=0.5)
    ap.add_argument("--expert-loss-weight", type=float, default=1.0)
    ap.add_argument("--model-tag", type=str, default="bracket_moe_gpu")
    ap.add_argument("--resume", type=str, default=None, help="path to a checkpoint .pt to resume from (last/best/any specific epoch snapshot)")
    ap.add_argument("--checkpoint-every", type=int, default=1, help="also save a standalone per-epoch snapshot every N epochs (in addition to the always-overwritten last/best), so any specific epoch can be resumed or downloaded later - not just whichever happened to be current when you last grabbed a file")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else " - WARNING: no GPU detected, this will be slow"))

    torch.manual_seed(RANDOM_STATE)
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads((data_dir / "manifest.json").read_text())
    vocab_size = manifest["vocab_size"]

    print(f"Loading pre-processed dataset from {data_dir}...")
    t0 = time.time()
    tokens, lengths, time_spent, white_bucket, black_bucket, white_masks, black_masks = load_parquet_dataset(
        data_dir, manifest["max_len"], manifest["num_experts"],
    )
    print(f"  {len(tokens):,} games, vocab_size={vocab_size:,}, loaded in {time.time()-t0:.1f}s")
    print(f"  white bracket coverage: {manifest['white_bracket_coverage']}")
    print(f"  black bracket coverage: {manifest['black_bracket_coverage']}")

    n = len(white_bucket)
    rng = np.random.RandomState(RANDOM_STATE)
    perm = rng.permutation(n)
    n_test, n_val = int(n * 0.1), int(n * 0.1)
    test_idx, val_idx, train_idx = perm[:n_test], perm[n_test:n_test + n_val], perm[n_test + n_val:]
    print(f"train={len(train_idx):,} val={len(val_idx):,} test={len(test_idx):,}")

    combined_targets = np.concatenate([white_bucket[train_idx], black_bucket[train_idx]])
    counts = np.clip(np.bincount(combined_targets, minlength=NUM_BUCKETS).astype(np.float64), 1, None)
    weights = (counts.sum() / (NUM_BUCKETS * counts)) ** args.weight_power
    class_weights = torch.tensor(weights, dtype=torch.float32, device=device)

    # One dataset over the full arrays, sliced via Subset (index-only, no
    # copy) instead of building train/val/test as three separately
    # fancy-indexed numpy copies - fancy indexing (tokens[idx]) allocates a
    # brand new array, so three eagerly-copied splits plus the original
    # full array would coexist in memory simultaneously, nearly doubling
    # the largest arrays' footprint for no reason.
    full_dataset = BracketDataset(tokens, lengths, time_spent, white_bucket, black_bucket, white_masks, black_masks)
    pin = device.type == "cuda"
    train_loader = DataLoader(Subset(full_dataset, train_idx), batch_size=args.batch_size, shuffle=True, pin_memory=pin, num_workers=2)
    val_loader = DataLoader(Subset(full_dataset, val_idx), batch_size=1024, pin_memory=pin, num_workers=2)
    test_loader = DataLoader(Subset(full_dataset, test_idx), batch_size=1024, pin_memory=pin, num_workers=2)

    model = BracketExpertsMoE(vocab_size=vocab_size, embed_dim=args.embed_dim, hidden_dim=args.hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr_min)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    start_epoch = 1
    best_val_loss = float("inf")
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scheduler.T_max = args.epochs  # keep THIS run's epoch count as the cosine horizon,
        # not whatever --epochs the checkpoint happened to be trained under (e.g. a short
        # calibration run) - otherwise a mismatched T_max silently wrecks the LR schedule
        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt["best_val_loss"]
        print(f"Resumed from {args.resume} at epoch {start_epoch}")

    ckpt_path = out_dir / f"{args.model_tag}_best.pt"
    last_path = out_dir / f"{args.model_tag}_last.pt"
    print(f"\nTraining {NUM_EXPERTS}-expert bracket MoE for {args.epochs} epochs...", flush=True)
    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        train_m = run_epoch(model, train_loader, optimizer, criterion, device, train=True, expert_loss_weight=args.expert_loss_weight)
        with torch.no_grad():
            val_m = run_epoch(model, val_loader, optimizer, criterion, device, train=False, expert_loss_weight=args.expert_loss_weight)
        scheduler.step()
        improved = val_m["loss"] < best_val_loss
        print(f"  epoch {epoch}: train[{fmt(train_m)}] val[{fmt(val_m)}] ({time.time()-t0:.1f}s){' *' if improved else ''}", flush=True)

        if improved:
            best_val_loss = val_m["loss"]
        # build ckpt_dict AFTER best_val_loss is finalized for this epoch, so
        # last_path/ckpt_path/the per-epoch snapshot below all agree on it -
        # saving last_path before this update left it one epoch stale, which
        # could make a resumed run's "is this a new best?" check wrong once.
        ckpt_dict = {
            "model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "epoch": epoch, "best_val_loss": best_val_loss,
        }
        # always save a "last" checkpoint so a disconnected Colab session can resume
        torch.save(ckpt_dict, last_path)
        if improved:
            torch.save(ckpt_dict, ckpt_path)
        # ALSO save a standalone per-epoch snapshot that never gets overwritten -
        # last_path/ckpt_path only ever hold the most-recent/best-so-far epoch, so if
        # you download one mid-run and training later improves past it, there's no way
        # to tell which epoch you actually have, or to deliberately resume/re-evaluate
        # an earlier specific epoch. This fixes that at a small, fixed disk cost
        # (~30MB/epoch for this model) - download epoch_XXX.pt whenever you want that
        # exact epoch, not whatever last_path/ckpt_path currently point to.
        if args.checkpoint_every > 0 and epoch % args.checkpoint_every == 0:
            torch.save(ckpt_dict, out_dir / f"{args.model_tag}_epoch{epoch:03d}.pt")

    best = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(best["model"])
    with torch.no_grad():
        test_m = run_epoch(model, test_loader, optimizer, criterion, device, train=False, expert_loss_weight=args.expert_loss_weight)
    print(f"\nFinal test (best val checkpoint, epoch {best['epoch']}): {fmt(test_m)}", flush=True)

    metrics_path = out_dir / f"{args.model_tag}_metrics.json"
    metrics_path.write_text(json.dumps({
        "model": f"BracketExpertsMoE_{args.model_tag}", "n_train": len(train_idx), "n_val": len(val_idx), "n_test": len(test_idx),
        "vocab_size": vocab_size, "epochs": args.epochs, "best_epoch": best["epoch"], "test_metrics": test_m,
        "embed_dim": args.embed_dim, "hidden_dim": args.hidden_dim, "num_experts": NUM_EXPERTS,
    }, indent=2))
    print(f"Saved: {metrics_path}")
    print(f"Checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
