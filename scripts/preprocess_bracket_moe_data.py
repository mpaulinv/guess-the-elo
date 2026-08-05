# scripts/preprocess_bracket_moe_data.py - one-time CPU preprocessing for the
# bracket-experts MoE, so the (paid) GPU run never has to touch raw CSVs.
"""Scans the combined dataset (original + 2023-04 extraction), tokenizes
every game, computes per-ply clock-time features, and derives per-color Elo
buckets + bracket masks - then writes everything to a single compressed
parquet file (plus small vocab.json/manifest.json sidecars).

This machine has only ~8GB RAM, often with several GB already used by other
running apps. Two earlier versions of this script each held far more in
memory at once than that allows (first: collecting the full ~4M-row dataset
before downsampling; second: pre-allocating full-size output numpy arrays
for the target count). This version follows the exact streaming pattern
already proven by scripts/build_move_vocab.py and scripts/data_extraction.py
- pandas `chunksize` reads, two passes over the raw CSVs - but pass 2 now
writes each chunk straight to a parquet file incrementally via
pyarrow.parquet.ParquetWriter, so peak memory is bounded to one chunk's
worth of games (~50k) regardless of total dataset size. Parquet's columnar
compression also makes the on-disk/upload size much smaller than raw .npy
for these token arrays, which are mostly padding zeros for shorter games.

Usage: python scripts/preprocess_bracket_moe_data.py [--limit N] [--chunk-size N]
"""
import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_nn_bracket_experts_moe import MAX_LEN, NUM_BUCKETS, elo_to_bucket, bracket_masks, compute_time_spent, BRACKET_RANGES, NUM_EXPERTS  # noqa: E402
from src.move_features import parse_san_moves  # noqa: E402

DATA_CSVS = [
    Path(r"C:\Users\mario\OneDrive\Documents\guess-the-elo\data\enhanced_extraction\enhanced_experiment_20250620_203308.csv"),
    Path(r"C:\Users\mario\OneDrive\Documents\guess-the-elo\data\enhanced_extraction_2023_04\enhanced_experiment_20260717_000231.csv"),
]
OUT_DIR = Path(r"C:\Users\mario\OneDrive\Documents\guess-the-elo\data\processed\nn_bracket_moe")
MIN_PLY = 10
MAX_VOCAB = 20000
KNOWN_TOTAL = 3_948_473  # confirmed by a prior full scan at --min-ply 10
USECOLS = ["game_id", "move_sequence", "time_control", "white_elo", "black_elo",
           "is_quality_game", "time_class", "calculated_ply_count"]

PARQUET_SCHEMA = pa.schema([
    pa.field("game_id", pa.string()),
    pa.field("white_elo", pa.float32()),
    pa.field("black_elo", pa.float32()),
    pa.field("white_bucket", pa.int64()),
    pa.field("black_bucket", pa.int64()),
    pa.field("length", pa.int32()),
    pa.field("tokens", pa.list_(pa.int32(), MAX_LEN)),
    pa.field("time_spent", pa.list_(pa.float32(), MAX_LEN)),
    pa.field("white_masks", pa.list_(pa.float32(), NUM_EXPERTS)),
    pa.field("black_masks", pa.list_(pa.float32(), NUM_EXPERTS)),
])


def _qualifies(chunk: pd.DataFrame, min_ply: int) -> pd.DataFrame:
    return chunk[chunk["is_quality_game"] & (chunk["time_class"] != "unknown") & (chunk["calculated_ply_count"] >= min_ply)]


def pass1_vocab_and_selection(min_ply: int, limit: int, chunk_size: int, max_vocab: int = MAX_VOCAB):
    """Streams every source CSV once. Builds the token vocab and records
    (game_id, white_elo, black_elo, time_control) for qualifying games only,
    up to --limit - never holds move_sequence text beyond the current chunk."""
    counter = Counter()
    selected_ids, selected_white_elo, selected_black_elo, selected_tc = [], [], [], []
    n_selected = 0
    t0 = time.time()

    for path in DATA_CSVS:
        if n_selected >= limit:
            break
        print(f"  pass 1: streaming {path.name}...", flush=True)
        for chunk in pd.read_csv(path, usecols=USECOLS, chunksize=chunk_size):
            quality = _qualifies(chunk, min_ply)
            if quality.empty:
                continue
            remaining = limit - n_selected
            if len(quality) > remaining:
                quality = quality.iloc[:remaining]
            for seq in quality["move_sequence"]:
                counter.update(parse_san_moves(seq))
            selected_ids.extend(quality["game_id"].tolist())
            selected_white_elo.extend(quality["white_elo"].tolist())
            selected_black_elo.extend(quality["black_elo"].tolist())
            selected_tc.extend(quality["time_control"].tolist())
            n_selected += len(quality)
            if n_selected % (chunk_size * 5) < chunk_size:
                print(f"    {n_selected:,}/{limit:,} selected ({time.time()-t0:.1f}s)", flush=True)
            if n_selected >= limit:
                break

    vocab = {"<PAD>": 0, "<UNK>": 1}
    for tok, _ in counter.most_common(max_vocab):
        vocab[tok] = len(vocab)

    selection = pd.DataFrame({
        "game_id": selected_ids, "white_elo": selected_white_elo,
        "black_elo": selected_black_elo, "time_control": selected_tc,
    })
    return vocab, selection


def pass2_tokenize_to_parquet(selection: pd.DataFrame, vocab: dict, min_ply: int, chunk_size: int,
                               out_path: Path, max_len: int = MAX_LEN):
    """Streams every source CSV a second time, keeping only rows whose
    game_id was selected in pass 1, tokenizing and writing each chunk
    straight to a parquet file - only one chunk of raw text and one chunk of
    output arrays exist in memory at any time, regardless of total size."""
    wanted = set(selection["game_id"])
    tc_by_id = dict(zip(selection["game_id"], selection["time_control"]))
    elo_by_id = dict(zip(selection["game_id"], zip(selection["white_elo"], selection["black_elo"])))
    n = len(selection)

    writer = pq.ParquetWriter(out_path, PARQUET_SCHEMA, compression="zstd")
    white_coverage = np.zeros(NUM_EXPERTS, dtype=np.int64)
    black_coverage = np.zeros(NUM_EXPERTS, dtype=np.int64)
    filled = 0
    t0 = time.time()

    def flush_batch(batch_ids, batch_tokens, batch_lengths, batch_time_spent):
        nonlocal filled
        if not batch_ids:
            return
        white_elo = np.array([elo_by_id[g][0] for g in batch_ids], dtype=np.float64)
        black_elo = np.array([elo_by_id[g][1] for g in batch_ids], dtype=np.float64)
        white_bucket = elo_to_bucket(white_elo)
        black_bucket = elo_to_bucket(black_elo)
        white_masks = bracket_masks(white_elo)
        black_masks = bracket_masks(black_elo)
        white_coverage[:] += white_masks.sum(axis=0).astype(np.int64)
        black_coverage[:] += black_masks.sum(axis=0).astype(np.int64)

        table = pa.table({
            "game_id": batch_ids,
            "white_elo": np.array(white_elo, dtype=np.float32),
            "black_elo": np.array(black_elo, dtype=np.float32),
            "white_bucket": white_bucket.astype(np.int64),
            "black_bucket": black_bucket.astype(np.int64),
            "length": np.array(batch_lengths, dtype=np.int32),
            "tokens": pa.FixedSizeListArray.from_arrays(np.array(batch_tokens, dtype=np.int32).reshape(-1), max_len),
            "time_spent": pa.FixedSizeListArray.from_arrays(np.array(batch_time_spent, dtype=np.float32).reshape(-1), max_len),
            "white_masks": pa.FixedSizeListArray.from_arrays(white_masks.astype(np.float32).reshape(-1), NUM_EXPERTS),
            "black_masks": pa.FixedSizeListArray.from_arrays(black_masks.astype(np.float32).reshape(-1), NUM_EXPERTS),
        }, schema=PARQUET_SCHEMA)
        writer.write_table(table)
        filled += len(batch_ids)

    for path in DATA_CSVS:
        if filled >= n:
            break
        print(f"  pass 2: streaming {path.name}...", flush=True)
        for chunk in pd.read_csv(path, usecols=["game_id", "move_sequence", "is_quality_game", "time_class", "calculated_ply_count"], chunksize=chunk_size):
            quality = _qualifies(chunk, min_ply)
            hit = quality[quality["game_id"].isin(wanted)]
            if hit.empty:
                continue

            batch_ids, batch_tokens, batch_lengths, batch_time_spent = [], [], [], []
            for gid, seq in zip(hit["game_id"], hit["move_sequence"]):
                moves = parse_san_moves(seq)[:max_len]
                ids = [vocab.get(m, 1) for m in moves]
                tok_row = np.zeros(max_len, dtype=np.int32)
                tok_row[:len(ids)] = ids
                batch_tokens.append(tok_row)
                batch_lengths.append(max(1, len(ids)))
                batch_time_spent.append(compute_time_spent(seq, tc_by_id[gid], max_len))
                batch_ids.append(gid)

            flush_batch(batch_ids, batch_tokens, batch_lengths, batch_time_spent)
            if filled % (chunk_size * 5) < chunk_size:
                print(f"    {filled:,}/{n:,} tokenized ({time.time()-t0:.1f}s)", flush=True)
            if filled >= n:
                break

    writer.close()
    print(f"  matched {filled:,}/{n:,} selected games in pass 2", flush=True)
    return filled, white_coverage, black_coverage


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=KNOWN_TOTAL, help="cap total games (default: the full known population)")
    ap.add_argument("--chunk-size", type=int, default=50_000, help="rows read per pandas chunksize batch")
    ap.add_argument("--min-ply", type=int, default=MIN_PLY)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "dataset.parquet"

    print(f"Pass 1/2: streaming {len(DATA_CSVS)} source(s) in chunks of {args.chunk_size:,} - building vocab + selecting up to {args.limit:,} games...")
    t0 = time.time()
    vocab, selection = pass1_vocab_and_selection(args.min_ply, args.limit, args.chunk_size)
    print(f"  {len(selection):,} games selected, vocab_size={len(vocab):,}, in {time.time()-t0:.1f}s")
    (OUT_DIR / "vocab.json").write_text(json.dumps(vocab))

    print(f"\nPass 2/2: streaming again to tokenize + write parquet incrementally for {len(selection):,} games...")
    t0 = time.time()
    filled, white_coverage, black_coverage = pass2_tokenize_to_parquet(selection, vocab, args.min_ply, args.chunk_size, out_path)
    print(f"  done in {time.time()-t0:.1f}s")

    manifest = {
        "n_games": filled, "vocab_size": len(vocab), "max_len": MAX_LEN, "min_ply": args.min_ply,
        "num_experts": NUM_EXPERTS, "num_buckets": NUM_BUCKETS, "bracket_ranges": BRACKET_RANGES,
        "white_bracket_coverage": white_coverage.tolist(),
        "black_bracket_coverage": black_coverage.tolist(),
        "sources": [str(p) for p in DATA_CSVS],
        "format": "parquet", "parquet_file": out_path.name,
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))

    size_mb = out_path.stat().st_size / 1e6
    print(f"\nDone. {filled:,} games written to {out_path} ({size_mb:.0f} MB, zstd-compressed)")
    print(f"White bracket coverage: {manifest['white_bracket_coverage']}")
    print(f"Black bracket coverage: {manifest['black_bracket_coverage']}")


if __name__ == "__main__":
    main()
