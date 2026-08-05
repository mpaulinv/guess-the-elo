# src/move_features.py - Move-by-move replay features
"""Replays each game's SAN move text with python-chess to derive richer,
skill-correlated signals that aren't present in the raw extraction counts:
piece development speed, material swings, hanging pieces, and how many
plies the game stayed inside known opening theory (see opening_book.py).
"""
import re

import chess

from src.signal_utils import longest_sustained_run

CLOCK_RE = re.compile(r"\{[^}]*\}")
MOVE_NUM_RE = re.compile(r"\d+\.+")
ANNOTATION_RE = re.compile(r"[!?]+$")  # engine-analysis glyphs, e.g. 'Nc6?', 'Rc4??'
RESULT_TOKENS = {"1-0", "0-1", "1/2-1/2", "*"}

PIECE_VALUES = {
    chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
    chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0,
}

DEVELOPMENT_PLY_HORIZON = 20  # first 10 full moves per side
HANGING_PIECE_TYPES = (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN)

# Starting squares for knights/bishops, keyed by color - used to detect
# "has this minor piece left its home square yet".
_HOME_SQUARES = {
    chess.WHITE: {chess.B1, chess.G1, chess.C1, chess.F1},
    chess.BLACK: {chess.B8, chess.G8, chess.C8, chess.F8},
}


def parse_san_moves(move_sequence: str) -> list[str]:
    text = CLOCK_RE.sub("", move_sequence)
    text = MOVE_NUM_RE.sub("", text)
    tokens = [ANNOTATION_RE.sub("", t) for t in text.split() if t not in RESULT_TOKENS]
    return [t for t in tokens if t]


# Minimum advantage (pawns-equivalent) to count as "a real advantage" for
# the conversion feature - roughly a minor piece, per the user's framing.
ADVANTAGE_THRESHOLD = 3.0
# The lead must hold for at least this many consecutive plies to count as
# "sustained" rather than a momentary blip mid-trade (e.g. capture, then
# recaptured 2 plies later nets back to ~even - that's not a real edge).
MIN_SUSTAIN_PLIES = 6


def _empty_features() -> dict:
    return {
        "book_depth_ply": 0,
        "material_std": 0.0,
        "material_max_swing": 0.0,
        "material_final_abs": 0.0,
        "material_num_big_swings": 0,
        "max_material_advantage": 0.0,
        "sustained_advantage_plies": 0,
        "advantage_converted": -1,
        "big_advantage_not_converted": 0,
        "white_developed_by_ply20": 0,
        "black_developed_by_ply20": 0,
        "white_castled_ply": -1,
        "black_castled_ply": -1,
        "white_pawn_moves_by_ply20": 0,
        "black_pawn_moves_by_ply20": 0,
        "hanging_piece_events": 0,
        "parse_ok": False,
        "parsed_ply_count": 0,
    }


def extract_move_features(move_sequence: str, book_prefixes: frozenset, result: str = "") -> dict:
    san_moves = parse_san_moves(move_sequence)
    feat = _empty_features()
    if not san_moves:
        return feat

    board = chess.Board()
    balance = 0.0
    balances = [0.0]
    home_left = {chess.WHITE: set(), chess.BLACK: set()}
    hanging_events = 0
    ply = 0

    try:
        for san in san_moves:
            color = board.turn
            move = board.parse_san(san)

            if board.is_capture(move):
                if board.is_en_passant(move):
                    captured_value = PIECE_VALUES[chess.PAWN]
                else:
                    captured_piece = board.piece_at(move.to_square)
                    captured_value = PIECE_VALUES[captured_piece.piece_type] if captured_piece else 0
                balance += captured_value if color == chess.WHITE else -captured_value

            is_castle = board.is_castling(move)
            piece = board.piece_at(move.from_square)

            board.push(move)
            ply += 1
            balances.append(balance)

            if ply <= DEVELOPMENT_PLY_HORIZON:
                if is_castle:
                    key = "white_castled_ply" if color == chess.WHITE else "black_castled_ply"
                    if feat[key] == -1:
                        feat[key] = ply
                elif piece is not None:
                    if piece.piece_type in (chess.KNIGHT, chess.BISHOP) and move.from_square in _HOME_SQUARES[color]:
                        home_left[color].add(move.from_square)
                    elif piece.piece_type == chess.PAWN:
                        pkey = "white_pawn_moves_by_ply20" if color == chess.WHITE else "black_pawn_moves_by_ply20"
                        feat[pkey] += 1

            # Hanging-piece check: pieces belonging to the side that just
            # moved, attacked by the opponent with zero defenders.
            mover_color = color
            for sq, pc in board.piece_map().items():
                if pc.color != mover_color or pc.piece_type not in HANGING_PIECE_TYPES:
                    continue
                if board.attackers(not mover_color, sq) and not board.attackers(mover_color, sq):
                    hanging_events += 1

        feat["parse_ok"] = True
    except (ValueError, AssertionError):
        # Malformed/truncated movetext - keep whatever was parsed so far.
        pass

    feat["parsed_ply_count"] = ply
    feat["book_depth_ply"] = _book_depth(san_moves[:ply], book_prefixes)
    feat["white_developed_by_ply20"] = len(home_left[chess.WHITE])
    feat["black_developed_by_ply20"] = len(home_left[chess.BLACK])
    feat["hanging_piece_events"] = hanging_events

    if len(balances) > 1:
        import numpy as np
        arr = np.array(balances)
        deltas = np.diff(arr)
        feat["material_std"] = float(arr.std())
        feat["material_max_swing"] = float(np.abs(deltas).max()) if len(deltas) else 0.0
        feat["material_final_abs"] = float(abs(arr[-1]))
        feat["material_num_big_swings"] = int((np.abs(deltas) >= 3).sum())

        feat["max_material_advantage"] = float(np.abs(arr).max())

        # Conversion: whichever side held a *sustained* material lead
        # (>= ADVANTAGE_THRESHOLD for >= MIN_SUSTAIN_PLIES in a row, not
        # just a one-ply blip from an in-progress trade) - did they go on
        # to actually win, or let it slip? Master-level games should
        # convert real advantages far more reliably than intermediate ones.
        run_len, run_sign = longest_sustained_run(arr, ADVANTAGE_THRESHOLD)
        feat["sustained_advantage_plies"] = run_len

        if run_len >= MIN_SUSTAIN_PLIES and run_sign != 0 and result in ("1-0", "0-1", "1/2-1/2"):
            leader_is_white = run_sign > 0
            leader_won = (leader_is_white and result == "1-0") or (not leader_is_white and result == "0-1")
            feat["advantage_converted"] = int(leader_won)
            if not leader_won:
                feat["big_advantage_not_converted"] = 1

    return feat


def _book_depth(san_moves: list[str], book_prefixes: frozenset) -> int:
    depth = 0
    for k in range(1, len(san_moves) + 1):
        if tuple(san_moves[:k]) in book_prefixes:
            depth = k
        else:
            break
    return depth
