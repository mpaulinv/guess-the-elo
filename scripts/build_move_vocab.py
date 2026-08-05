# scripts/build_move_vocab.py - Tokenize move_sequence for the NN
"""Two passes over the full quality-game set:
  1. build a SAN-move vocabulary (capped at MAX_VOCAB most common tokens)
  2. encode every game's moves to a fixed-length, padded int array

Saves:
  data/processed/nn/vocab.json           token -> id
  data/processed/nn/tokens.npy           [N, MAX_LEN] int16, padded/truncated
  data/processed/nn/lengths.npy          [N] int16, true sequence length (capped)
  data/processed/nn/meta.parquet         game_id, avg_elo, time_class (row-aligned with tokens.npy)

Usage: python scripts/build_move_vocab.py
"""
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.move_features import parse_san_moves  # noqa: E402

DATA_CSV = Path(__file__).resolve().parent.parent / "data" / "enhanced_extraction" / "enhanced_experiment_20250620_203308.csv"
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "processed" / "nn"
USECOLS = ["game_id", "move_sequence", "avg_elo", "is_quality_game", "time_class"]
CHUNK_SIZE = 100_000

MAX_VOCAB = 20_000
MAX_LEN = 140  # covers ~97% of games per the sampled ply-length distribution
PAD_ID = 0
UNK_ID = 1


def iter_quality_chunks():
    reader = pd.read_csv(DATA_CSV, usecols=USECOLS, chunksize=CHUNK_SIZE)
    for chunk in reader:
        q = chunk[chunk["is_quality_game"] & (chunk["time_class"] != "unknown")]
        if not q.empty:
            yield q


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Pass 1/2: building vocabulary...")
    t0 = time.time()
    vocab_counter = Counter()
    n_games = 0
    for chunk in iter_quality_chunks():
        for seq in chunk["move_sequence"]:
            vocab_counter.update(parse_san_moves(seq))
        n_games += len(chunk)
        print(f"  ...{n_games:,} games scanned, vocab size so far {len(vocab_counter):,}", end="\r")
    print(f"\n  done in {time.time()-t0:.1f}s. {n_games:,} games, {len(vocab_counter):,} raw unique tokens")

    most_common = [tok for tok, _ in vocab_counter.most_common(MAX_VOCAB)]
    token_to_id = {"<PAD>": PAD_ID, "<UNK>": UNK_ID}
    for tok in most_common:
        token_to_id[tok] = len(token_to_id)
    print(f"  final vocab size (capped): {len(token_to_id):,}")
    (OUT_DIR / "vocab.json").write_text(json.dumps(token_to_id))

    print("\nPass 2/2: encoding games to padded token-id arrays...")
    tokens = np.zeros((n_games, MAX_LEN), dtype=np.int16)
    lengths = np.zeros(n_games, dtype=np.int16)
    meta_rows = []

    t0 = time.time()
    row = 0
    for chunk in iter_quality_chunks():
        for game_id, seq, avg_elo, time_class in zip(
            chunk["game_id"], chunk["move_sequence"], chunk["avg_elo"], chunk["time_class"]
        ):
            moves = parse_san_moves(seq)[:MAX_LEN]
            ids = [token_to_id.get(m, UNK_ID) for m in moves]
            tokens[row, :len(ids)] = ids
            lengths[row] = len(ids)
            meta_rows.append((game_id, avg_elo, time_class))
            row += 1
        print(f"  ...{row:,}/{n_games:,} encoded", end="\r")

    print(f"\n  done in {time.time()-t0:.1f}s")
    np.save(OUT_DIR / "tokens.npy", tokens)
    np.save(OUT_DIR / "lengths.npy", lengths)
    meta = pd.DataFrame(meta_rows, columns=["game_id", "avg_elo", "time_class"])
    meta.to_parquet(OUT_DIR / "meta.parquet")

    print(f"\nSaved to {OUT_DIR}:")
    print(f"  tokens.npy  {tokens.shape} {tokens.dtype}  ({tokens.nbytes/1e6:.1f} MB)")
    print(f"  lengths.npy {lengths.shape}")
    print(f"  meta.parquet {meta.shape}")
    print(f"  length distribution: mean={lengths.mean():.1f} p50={np.percentile(lengths,50):.0f} "
          f"p90={np.percentile(lengths,90):.0f} p99={np.percentile(lengths,99):.0f} "
          f"truncated={int((lengths==MAX_LEN).sum())} ({(lengths==MAX_LEN).mean()*100:.1f}%)")


if __name__ == "__main__":
    main()
