# scripts/analyze_length_vs_error.py - Does prediction error depend on game length?
"""Bins each model's held-out test predictions by true game length
(calculated_ply_count, not the NN's truncated input length) to check
whether short games are fundamentally harder to rate - and whether that's
confounded with something else (e.g. short games skew to certain brackets
or terminations).

Must run under Python 3.10 (torch): same as train_nn_model.py.
Usage: python scripts/analyze_length_vs_error.py
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.combined_features import build_combined_features, load_move_features  # noqa: E402
from src.features import load_quality_games  # noqa: E402

DATA_CSV = Path(__file__).resolve().parent.parent / "data" / "enhanced_extraction" / "enhanced_experiment_20250620_203308.csv"
PARTS_DIR = Path(__file__).resolve().parent.parent / "data" / "processed" / "move_features_parts"
NN_DIR = Path(__file__).resolve().parent.parent / "data" / "processed" / "nn"
BRACKET_PROBS_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "bracket_probs_full.parquet"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
RANDOM_STATE = 42

LENGTH_BINS = [0, 20, 30, 40, 50, 60, 80, 100, 140, 10_000]
LENGTH_LABELS = ["<20", "20-30", "30-40", "40-50", "50-60", "60-80", "80-100", "100-140", "140+"]

# calculated_ply_count = move_numbers * 2 (see data_extraction.py), so ply
# counts are always even - bin edges land on even numbers so no bucket is
# artificially empty. Covers the range the coarse "<20" bucket lumps together.
SHORT_LENGTH_BINS = [0, 4, 6, 8, 10, 12, 14, 16, 18, 20]
SHORT_LENGTH_LABELS = ["2-4", "4-6", "6-8", "8-10", "10-12", "12-14", "14-16", "16-18", "18-20"]

CUTOFF_CANDIDATES = [0, 4, 6, 8, 10, 12, 14, 16, 18, 20]

ENGINE_FEATURE_COLUMNS = [
    "eval_coverage", "white_acpl", "black_acpl", "avg_acpl",
    "white_blunders", "black_blunders", "white_mistakes", "black_mistakes",
    "white_inaccuracies", "black_inaccuracies", "max_loss_white", "max_loss_black",
    "first_blunder_ply", "eval_std", "opening_acpl", "middlegame_acpl", "endgame_acpl",
    "has_middlegame", "has_endgame", "balance_ratio", "sustained_eval_advantage_plies",
    "eval_advantage_converted", "big_eval_advantage_not_converted",
    "white_missed_mate_in_1", "black_missed_mate_in_1", "white_missed_mate_in_2", "black_missed_mate_in_2",
]
TIME_FEATURE_COLUMNS = [
    "clock_coverage", "white_time_per_move", "black_time_per_move", "white_time_std", "black_time_std",
    "white_low_time_moves", "black_low_time_moves", "white_fast_moves", "black_fast_moves",
    "white_first_low_time_ply", "black_first_low_time_ply", "white_opening_time_per_move",
    "black_opening_time_per_move", "white_pace_used_by_ply20", "black_pace_used_by_ply20",
]
KEY_MOMENT_FEATURE_COLUMNS = [
    "white_critical_time_ratio", "black_critical_time_ratio", "white_critical_loss", "black_critical_loss",
    "white_quiet_loss", "black_quiet_loss", "white_fast_critical_errors", "black_fast_critical_errors",
    "white_slow_critical_success", "black_slow_critical_success", "white_time_loss_corr",
    "black_time_loss_corr", "white_critical_moments", "black_critical_moments",
]


class EloGRU(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, bracket_dim, mlp_hidden=64):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim + bracket_dim, mlp_hidden), nn.ReLU(),
            nn.Dropout(0.2), nn.Linear(mlp_hidden, 1),
        )

    def forward(self, tokens, lengths, bracket_probs):
        emb = self.embed(tokens)
        packed = nn.utils.rnn.pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h_n = self.gru(packed)
        combined = torch.cat([h_n[-1], bracket_probs], dim=1)
        return self.head(combined).squeeze(-1)


def bin_report(df, pred_col, label, bins=LENGTH_BINS, labels=LENGTH_LABELS):
    df = df.copy()
    df["length_bin"] = pd.cut(df["ply_count"], bins=bins, labels=labels, right=False)
    df["abs_err"] = (df["avg_elo"] - df[pred_col]).abs()
    df["bias"] = df[pred_col] - df["avg_elo"]  # positive = model overestimates
    report = df.groupby("length_bin", observed=True).agg(
        n=("abs_err", "size"),
        mae=("abs_err", "mean"),
        bias=("bias", "mean"),
        avg_elo_mean=("avg_elo", "mean"),
        avg_elo_std=("avg_elo", "std"),
    ).round(1)
    print(f"\n=== {label}: MAE by game length ===")
    print(report)
    return report


def cutoff_sweep(df, pred_col, label, cutoffs=CUTOFF_CANDIDATES):
    """For each candidate minimum-ply cutoff, report how many games would be
    dropped, how bad they are in isolation, and what test MAE looks like on
    the remaining games - the direct "do I need to cap the low end" table."""
    df = df.copy()
    df["abs_err"] = (df["avg_elo"] - df[pred_col]).abs()
    n_total = len(df)
    rows = []
    for cutoff in cutoffs:
        below = df[df["ply_count"] < cutoff]
        remaining = df[df["ply_count"] >= cutoff]
        rows.append({
            "cutoff": cutoff,
            "n_removed": len(below),
            "pct_removed": round(100 * len(below) / n_total, 2),
            "mae_below": round(below["abs_err"].mean(), 1) if len(below) else float("nan"),
            "mae_remaining": round(remaining["abs_err"].mean(), 1) if len(remaining) else float("nan"),
        })
    report = pd.DataFrame(rows).set_index("cutoff")
    print(f"\n=== {label}: cutoff sweep (drop games with ply_count < cutoff) ===")
    print(report)
    return report


def main():
    print("Loading base data (calculated_ply_count, avg_elo)...")
    base_df = load_quality_games(DATA_CSV)[["game_id", "avg_elo", "calculated_ply_count", "termination"]]
    base_df = base_df.rename(columns={"calculated_ply_count": "ply_count"})

    # --- NN: reproduce its test split, predict on ALL 99,444 test games ---
    print("\nReproducing NN test split + predicting on its full test set...")
    meta = pd.read_parquet(NN_DIR / "meta.parquet")
    bracket_probs_df = pd.read_parquet(BRACKET_PROBS_PATH)
    merged = meta.reset_index().merge(bracket_probs_df, on="game_id", how="inner")
    n = len(merged)
    rng = np.random.RandomState(RANDOM_STATE)
    perm = rng.permutation(n)
    n_test = int(n * 0.1)
    test_positions = perm[:n_test]
    nn_test = merged.iloc[test_positions].reset_index(drop=True)

    tokens = np.load(NN_DIR / "tokens.npy")
    lengths = np.load(NN_DIR / "lengths.npy")
    prob_cols = [c for c in bracket_probs_df.columns if c.startswith("p_")]
    vocab = json.loads((NN_DIR / "vocab.json").read_text())
    nn_metrics = json.loads((MODELS_DIR / "nn_v12_metrics.json").read_text())
    elo_mean, elo_std = nn_metrics["elo_mean"], nn_metrics["elo_std"]

    rows = nn_test["index"].values
    tok_t = torch.from_numpy(tokens[rows]).long()
    len_t = torch.from_numpy(lengths[rows]).long().clamp(min=1)
    bp_t = torch.from_numpy(nn_test[prob_cols].values.astype(np.float32)).float()

    model = EloGRU(vocab_size=len(vocab), embed_dim=nn_metrics["embed_dim"], hidden_dim=nn_metrics["hidden_dim"], bracket_dim=len(prob_cols))
    model.load_state_dict(torch.load(MODELS_DIR / "nn_v12_best.pt", weights_only=True))
    model.eval()
    with torch.no_grad():
        preds_norm = model(tok_t, len_t, bp_t).numpy()
    nn_test["nn_pred"] = preds_norm * elo_std + elo_mean
    nn_test = nn_test.merge(base_df[["game_id", "ply_count", "termination"]], on="game_id", how="left")

    nn_report = bin_report(nn_test, "nn_pred", "NN (nn_v12), full 99,444-game test set")

    # Fine-grained breakdown of what the coarse "<20" bucket above is hiding,
    # plus the cutoff-sweep decision aid - both restricted to the NN's test
    # set since it's the only one with a meaningful population of very short
    # games (see note by xgb_report below).
    nn_short = nn_test[nn_test["ply_count"] < 20]
    nn_fine_report = bin_report(
        nn_short, "nn_pred", "NN (nn_v12), ply_count < 20 detail",
        bins=SHORT_LENGTH_BINS, labels=SHORT_LENGTH_LABELS,
    )
    nn_cutoff_report = cutoff_sweep(nn_test, "nn_pred", "NN (nn_v12), full test set")

    # --- XGBoost: reproduce its test split, predict on its 22,766-game test set ---
    print("\nReproducing XGBoost test split + predicting on its test set...")
    move_df = load_move_features(PARTS_DIR)
    engine_df = pd.read_parquet(Path(__file__).resolve().parent.parent / "data" / "processed" / "engine_features.parquet")
    time_df = pd.read_parquet(Path(__file__).resolve().parent.parent / "data" / "processed" / "time_features.parquet")
    key_df = pd.read_parquet(Path(__file__).resolve().parent.parent / "data" / "processed" / "time_quality_features.parquet")

    xgb_base = load_quality_games(DATA_CSV)
    xgb_base = xgb_base[
        xgb_base["game_id"].isin(engine_df["game_id"])
        & xgb_base["game_id"].isin(time_df["game_id"])
        & xgb_base["game_id"].isin(key_df["game_id"])
    ].reset_index(drop=True)

    X_basic, y = build_combined_features(xgb_base, move_df)
    merged_engine = xgb_base[["game_id"]].merge(engine_df, on="game_id", how="left")
    merged_time = xgb_base[["game_id"]].merge(time_df, on="game_id", how="left")
    merged_key = xgb_base[["game_id"]].merge(key_df, on="game_id", how="left")
    X = pd.concat([
        X_basic.reset_index(drop=True),
        merged_engine[ENGINE_FEATURE_COLUMNS].reset_index(drop=True),
        merged_time[TIME_FEATURE_COLUMNS].reset_index(drop=True),
        merged_key[KEY_MOMENT_FEATURE_COLUMNS].reset_index(drop=True),
    ], axis=1)

    idx_train, idx_test = train_test_split(range(len(X)), test_size=0.2, random_state=RANDOM_STATE)
    xgb_bundle = joblib.load(MODELS_DIR / "xgb_key_moments_v10.joblib")
    xgb_preds = xgb_bundle["model"].predict(X.iloc[idx_test][xgb_bundle["feature_columns"]])

    xgb_test = pd.DataFrame({
        "game_id": xgb_base["game_id"].values[idx_test],
        "avg_elo": y.iloc[idx_test].values,
        "xgb_pred": xgb_preds,
    }).merge(base_df[["game_id", "ply_count", "termination"]], on="game_id", how="left")

    # Note: xgb_test has almost no games under 60 plies (n=12 in the 20-30
    # bin, 0 below that) because it's restricted to games with engine/time-
    # quality features, which skew long. It's not a representative population
    # for the low-end cutoff decision - only the NN report/sweep above and
    # the short-game detail below should inform that call.
    xgb_report = bin_report(xgb_test, "xgb_pred", "XGBoost (xgb_key_moments_v10), 22,766-game test set")

    print("\n=== Termination reason by length bin (NN test set) - checking for confounds ===")
    nn_test["length_bin"] = pd.cut(nn_test["ply_count"], bins=LENGTH_BINS, labels=LENGTH_LABELS, right=False)
    print(pd.crosstab(nn_test["length_bin"], nn_test["termination"], normalize="index").round(2))

    print("\n=== Termination reason by length bin (NN test set, ply_count < 20 detail) ===")
    nn_short = nn_short.copy()
    nn_short["length_bin"] = pd.cut(nn_short["ply_count"], bins=SHORT_LENGTH_BINS, labels=SHORT_LENGTH_LABELS, right=False)
    print(pd.crosstab(nn_short["length_bin"], nn_short["termination"], normalize="index").round(2))

    out = {
        "nn_by_length": nn_report.to_dict(),
        "nn_by_length_fine": nn_fine_report.to_dict(),
        "cutoff_sweep": nn_cutoff_report.to_dict(),
        "xgb_by_length": xgb_report.to_dict(),
    }
    (MODELS_DIR / "length_vs_error_analysis.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"\nSaved: {MODELS_DIR / 'length_vs_error_analysis.json'}")


if __name__ == "__main__":
    main()
