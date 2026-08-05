# app/inference.py - self-contained inference pipeline for the bracket-MoE
# Elo predictor. Deliberately duplicates (not imports) the model classes from
# scripts/train_bracket_moe_gpu.py and the PGN/clock-time parsing helpers
# from src/move_features.py + src/time_features.py, so the deployed app has
# no dependency on the rest of this repo (no `chess` package, no scripts/ or
# src/ tree needed in the container) - matches the same self-contained
# philosophy already used for the GPU training script.
"""Given a raw PGN string (headers + movetext, exactly as exported by
Lichess - including [%clk ...] annotations), predicts white and black Elo
using the trained 8-expert bracket-MoE checkpoint.
"""
import json
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# ---- Config (overridable via env vars in main.py) ----
NUM_BUCKETS = 14
BUCKET_LO, BUCKET_WIDTH = 400, 200
BUCKET_MIDPOINTS = np.array([BUCKET_LO + BUCKET_WIDTH * i + BUCKET_WIDTH / 2 for i in range(NUM_BUCKETS)])
NUM_EXPERTS = 8
MAX_LEN = 140
TIME_SCALE = 30.0
EMBED_DIM, HIDDEN_DIM = 24, 48


# ---- PGN / move / clock-time parsing (matches src/move_features.py + src/time_features.py exactly) ----
TIME_CONTROL_RE = re.compile(r'\[TimeControl "([^"]+)"\]')
CLOCK_OR_EVAL_TOKEN_RE = re.compile(r"\d+\.+|\{[^}]*\}|\S+")
CLK_TAG_RE = re.compile(r"\[%clk\s+([^\]]+)\]")
CLOCK_RE = re.compile(r"\{[^}]*\}")
MOVE_NUM_RE = re.compile(r"\d+\.+")
ANNOTATION_RE = re.compile(r"[!?]+$")
RESULT_TOKENS = {"1-0", "0-1", "1/2-1/2", "*"}


class PGNParseError(ValueError):
    pass


def extract_time_control_and_movetext(pgn_text: str) -> tuple[str, str]:
    """Splits a raw PGN (headers + movetext) into (time_control, move_sequence).
    move_sequence keeps move numbers and [%clk] comments intact - exactly the
    same raw format compute_time_spent()/parse_san_moves() expect, matching
    what data_extraction.py stored during training."""
    tc_match = TIME_CONTROL_RE.search(pgn_text)
    if not tc_match:
        raise PGNParseError("No [TimeControl \"...\"] header found in PGN")
    time_control = tc_match.group(1)

    lines = pgn_text.splitlines()
    move_lines = []
    in_moves = False
    for line in lines:
        line = line.strip()
        if not line:
            if in_moves:
                break
            continue
        if line.startswith("["):
            continue
        in_moves = True
        move_lines.append(line)
    if not move_lines:
        raise PGNParseError("No movetext found in PGN")
    return time_control, " ".join(move_lines)


def parse_san_moves(move_sequence: str) -> list[str]:
    text = CLOCK_RE.sub("", move_sequence)
    text = MOVE_NUM_RE.sub("", text)
    tokens = [ANNOTATION_RE.sub("", t) for t in text.split() if t not in RESULT_TOKENS]
    return [t for t in tokens if t]


def parse_moves_with_clock(move_sequence: str) -> list[tuple[str, str | None]]:
    pairs = []
    pending_san = None
    for tok in CLOCK_OR_EVAL_TOKEN_RE.finditer(move_sequence):
        t = tok.group(0)
        if t[0].isdigit() and t.rstrip(".").isdigit():
            continue
        if t.startswith("{"):
            if pending_san is not None:
                m = CLK_TAG_RE.search(t)
                pairs.append((pending_san, m.group(1) if m else None))
                pending_san = None
            continue
        if t in RESULT_TOKENS:
            continue
        if pending_san is not None:
            pairs.append((pending_san, None))
        pending_san = ANNOTATION_RE.sub("", t)
    if pending_san is not None:
        pairs.append((pending_san, None))
    return pairs


def _clk_to_seconds(clk_str: str | None) -> float | None:
    if not clk_str:
        return None
    parts = clk_str.strip().split(":")
    try:
        parts = [float(p) for p in parts]
    except ValueError:
        return None
    while len(parts) < 3:
        parts.insert(0, 0.0)
    h, m, s = parts
    return h * 3600 + m * 60 + s


def parse_time_control(time_control: str) -> tuple[float, float] | None:
    if not isinstance(time_control, str) or "+" not in time_control:
        return None
    base, _, inc = time_control.partition("+")
    try:
        return float(base), float(inc)
    except ValueError:
        return None


def compute_time_spent(move_sequence: str, time_control: str, max_len: int = MAX_LEN) -> np.ndarray:
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


# ---- Model (identical architecture to scripts/train_bracket_moe_gpu.py) ----
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
        white_probs_list, black_probs_list = [], []
        for expert in self.experts:
            wl, bl = expert(tokens, lengths, time_spent)
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
        return final_white, final_black


class EloPredictor:
    """Loads the checkpoint + vocab once; call .predict(pgn_text) per request."""

    def __init__(self, checkpoint_path: str, vocab_path: str, min_ply: int = 10,
                 min_clock_coverage: float = 0.9, device: str = "cpu"):
        self.device = torch.device(device)
        self.vocab = json.loads(Path(vocab_path).read_text())
        self.vocab_size = len(self.vocab)
        self.min_ply = min_ply
        self.min_clock_coverage = min_clock_coverage

        self.model = BracketExpertsMoE(vocab_size=self.vocab_size, embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM).to(self.device)
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.midpoints = torch.from_numpy(BUCKET_MIDPOINTS).float()

    def predict(self, pgn_text: str) -> dict:
        time_control, move_sequence = extract_time_control_and_movetext(pgn_text)
        moves = parse_san_moves(move_sequence)
        if len(moves) < self.min_ply:
            raise PGNParseError(f"Game too short ({len(moves)} plies) - need at least {self.min_ply}")

        moves = moves[:MAX_LEN]

        # Every training game had real per-ply clock data - a game with little
        # or none feeds the model an out-of-distribution time_spent input it
        # essentially never saw, which produces severely wrong, falsely
        # confident predictions rather than gracefully degrading. Confirmed
        # empirically on two real games: 0% coverage predicted 2300+ Elo
        # against a true 1500/1599, and even 51% coverage (well above what
        # you'd guess is "mostly fine") predicted ~700-900 points high on a
        # game whose full-clock-data version was already off by "only"
        # ~100-370. The safe boundary between 51% and 100% is untested -
        # this threshold is a deliberately conservative guess given how fast
        # things degraded, not a calibrated cutoff.
        clock_pairs = parse_moves_with_clock(move_sequence)[:len(moves)]
        n_with_clock = sum(1 for _, c in clock_pairs if c is not None)
        clock_coverage = n_with_clock / len(moves) if moves else 0.0
        if clock_coverage < self.min_clock_coverage:
            raise PGNParseError(
                f"This game only has clock time data for {clock_coverage:.0%} of moves "
                f"(need at least {self.min_clock_coverage:.0%}). This model relies heavily on "
                "per-move clock time, and predictions on games with mostly-missing clock data "
                "have been confirmed to be severely unreliable - "
                "make sure you're uploading a PGN exported with clock times included."
            )

        ids = [self.vocab.get(m, 1) for m in moves]  # 1 = <UNK>
        tokens = np.zeros(MAX_LEN, dtype=np.int64)
        tokens[:len(ids)] = ids
        length = max(1, len(ids))
        time_spent = compute_time_spent(move_sequence, time_control, MAX_LEN)

        tok_t = torch.from_numpy(tokens).long().unsqueeze(0).to(self.device)
        len_t = torch.tensor([length]).long()
        ts_t = torch.from_numpy(time_spent).float().unsqueeze(0).to(self.device)

        with torch.no_grad():
            final_white, final_black = self.model(tok_t, len_t, ts_t)
            white_probs = torch.softmax(final_white, dim=1).squeeze(0)
            black_probs = torch.softmax(final_black, dim=1).squeeze(0)
            white_elo = float((white_probs * self.midpoints).sum().item())
            black_elo = float((black_probs * self.midpoints).sum().item())
            white_bucket = int(white_probs.argmax().item())
            black_bucket = int(black_probs.argmax().item())

        # Anything reaching here already cleared min_clock_coverage (the hard
        # cutoff above). This is just a lighter note for "not quite 100%" -
        # unlike the hard cutoff, this narrower band hasn't been specifically
        # tested, so the wording is deliberately softer/less certain.
        warning = None
        if clock_coverage < 1.0:
            warning = f"{clock_coverage:.0%} of moves had clock time data - predictions may be slightly less reliable than usual."

        return {
            "white_elo": round(white_elo),
            "black_elo": round(black_elo),
            "white_bucket_range": [BUCKET_LO + white_bucket * BUCKET_WIDTH, BUCKET_LO + (white_bucket + 1) * BUCKET_WIDTH],
            "black_bucket_range": [BUCKET_LO + black_bucket * BUCKET_WIDTH, BUCKET_LO + (black_bucket + 1) * BUCKET_WIDTH],
            "white_confidence": round(float(white_probs[white_bucket].item()), 3),
            "black_confidence": round(float(black_probs[black_bucket].item()), 3),
            "plies_used": min(len(moves), MAX_LEN),
            "clock_coverage": round(clock_coverage, 3),
            "warning": warning,
        }
