# src/opening_book.py - Loads the Lichess canonical opening reference
"""Reference: https://github.com/lichess-org/chess-openings (a-e.tsv, bundled
under data/reference/). This is the same table Lichess itself uses to derive
ECO/opening names, so it's a legitimate ground truth for "is this move still
known theory" - not something we're inventing from our own data.
"""
import re
from pathlib import Path

import pandas as pd

REFERENCE_DIR = Path(__file__).resolve().parent.parent / "data" / "reference"
_MOVE_NUM_RE = re.compile(r"\d+\.+")


def _tokenize(pgn_moves: str) -> tuple[str, ...]:
    """'1. e4 e5 2. Nf3' -> ('e4', 'e5', 'Nf3')"""
    tokens = _MOVE_NUM_RE.sub("", pgn_moves).split()
    return tuple(tokens)


def load_book_prefixes() -> frozenset[tuple[str, ...]]:
    """Every prefix of every known opening line, as a set of SAN-move tuples.
    A game's SAN move list is "still in book" through ply k iff moves[:k] is
    a member of this set."""
    prefixes = set()
    for f in sorted(REFERENCE_DIR.glob("*.tsv")):
        df = pd.read_csv(f, sep="\t")
        for pgn in df["pgn"]:
            moves = _tokenize(pgn)
            for k in range(1, len(moves) + 1):
                prefixes.add(moves[:k])
    return frozenset(prefixes)


def book_depth(san_moves: list[str], book_prefixes: frozenset) -> int:
    """Longest prefix of san_moves that matches known theory (in plies)."""
    depth = 0
    for k in range(1, len(san_moves) + 1):
        if tuple(san_moves[:k]) in book_prefixes:
            depth = k
        else:
            break
    return depth
