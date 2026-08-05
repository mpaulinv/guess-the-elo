# scripts/extract_move_features.py - Move-replay feature extraction (parallel)
"""Replays move_sequence for every quality game with python-chess to derive
development / material-swing / hanging-piece / book-depth features (see
src/move_features.py), in parallel across CPU cores. Writes one parquet part
per chunk to data/processed/move_features_parts/, so a run can be resumed/
inspected without re-doing finished chunks.

Usage:
  python scripts/extract_move_features.py [--limit N] [--workers K] [--chunk-size N]
"""
import argparse
import sys
import time
from pathlib import Path

import pandas as pd
from multiprocessing import Pool, cpu_count

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DATA_CSV = Path(__file__).resolve().parent.parent / "data" / "enhanced_extraction" / "enhanced_experiment_20250620_203308.csv"
PARTS_DIR = Path(__file__).resolve().parent.parent / "data" / "processed" / "move_features_parts"
USECOLS = ["game_id", "move_sequence", "avg_elo", "is_quality_game", "time_class", "result"]

_BOOK = None


def _init_worker():
    global _BOOK
    from src.opening_book import load_book_prefixes
    _BOOK = load_book_prefixes()


def _worker(args: tuple) -> dict:
    from src.move_features import extract_move_features
    move_sequence, result = args
    return extract_move_features(move_sequence, _BOOK, result=result)


def process_chunk(chunk: pd.DataFrame, workers: int) -> pd.DataFrame:
    task_args = list(zip(chunk["move_sequence"].tolist(), chunk["result"].tolist()))
    with Pool(processes=workers, initializer=_init_worker) as pool:
        feats = pool.map(_worker, task_args, chunksize=200)
    feat_df = pd.DataFrame(feats)
    out = pd.concat(
        [chunk[["game_id", "avg_elo", "time_class"]].reset_index(drop=True), feat_df],
        axis=1,
    )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="stop after this many quality games (for quick tests)")
    ap.add_argument("--workers", type=int, default=max(1, cpu_count() - 1))
    ap.add_argument("--chunk-size", type=int, default=50_000)
    args = ap.parse_args()

    PARTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"workers={args.workers} chunk_size={args.chunk_size} limit={args.limit}")

    reader = pd.read_csv(DATA_CSV, usecols=USECOLS, chunksize=args.chunk_size)
    total_done = 0
    t_start = time.time()

    for i, raw_chunk in enumerate(reader):
        part_path = PARTS_DIR / f"part_{i:04d}.parquet"
        if part_path.exists():
            n = len(pd.read_parquet(part_path, columns=["game_id"]))
            total_done += n
            print(f"chunk {i}: already done ({n} rows), skipping")
            if args.limit and total_done >= args.limit:
                break
            continue

        quality = raw_chunk[raw_chunk["is_quality_game"] & (raw_chunk["time_class"] != "unknown")]
        if args.limit is not None:
            remaining = args.limit - total_done
            if remaining <= 0:
                break
            quality = quality.head(remaining)
        if quality.empty:
            continue

        t0 = time.time()
        out = process_chunk(quality, args.workers)
        out.to_parquet(part_path)
        total_done += len(out)
        rate = len(out) / (time.time() - t0)
        elapsed = time.time() - t_start
        print(f"chunk {i}: {len(out):,} games in {time.time()-t0:.1f}s "
              f"({rate:.0f} games/s) | total {total_done:,} | elapsed {elapsed/60:.1f} min")

        if args.limit and total_done >= args.limit:
            break

    print(f"\nDone. {total_done:,} games processed in {(time.time()-t_start)/60:.1f} min")


if __name__ == "__main__":
    main()
