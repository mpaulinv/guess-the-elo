# scripts/ensemble_predict.py - Ensemble the best XGBoost and NN models
"""xgb_key_moments_v10 (MAE 212.8, 114k engine-analyzed subset) and
nn_v7 (MAE 201.1, full 994k games) were trained with different splits on
different-sized datasets. To ensemble them fairly without leakage, this
reproduces each model's exact held-out test split, takes the INTERSECTION
(games neither model was trained on), and evaluates XGBoost alone, NN
alone, and their average on that shared, clean test set.

Must run under Python 3.10 (torch): see train_nn_model.py header.
Usage: python scripts/ensemble_predict.py
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.combined_features import build_combined_features, load_move_features  # noqa: E402
from src.features import elo_bracket, load_quality_games  # noqa: E402

DATA_CSV = Path(__file__).resolve().parent.parent / "data" / "enhanced_extraction" / "enhanced_experiment_20250620_203308.csv"
PARTS_DIR = Path(__file__).resolve().parent.parent / "data" / "processed" / "move_features_parts"
ENGINE_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "engine_features.parquet"
TIME_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "time_features.parquet"
KEY_MOMENT_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "time_quality_features.parquet"
NN_DIR = Path(__file__).resolve().parent.parent / "data" / "processed" / "nn"
BRACKET_PROBS_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "bracket_probs_full.parquet"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
RANDOM_STATE = 42

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


def get_xgb_test_game_ids():
    """Reproduces xgb_key_moments_v10's exact loading + split to recover
    which game_ids ended up in its held-out test set."""
    base_df = load_quality_games(DATA_CSV)
    move_df = load_move_features(PARTS_DIR)
    engine_df = pd.read_parquet(ENGINE_PATH)
    time_df = pd.read_parquet(TIME_PATH)
    key_df = pd.read_parquet(KEY_MOMENT_PATH)
    base_df = base_df[
        base_df["game_id"].isin(engine_df["game_id"])
        & base_df["game_id"].isin(time_df["game_id"])
        & base_df["game_id"].isin(key_df["game_id"])
    ].reset_index(drop=True)

    X_basic, y = build_combined_features(base_df, move_df)
    merged_engine = base_df[["game_id"]].merge(engine_df, on="game_id", how="left")
    merged_time = base_df[["game_id"]].merge(time_df, on="game_id", how="left")
    merged_key = base_df[["game_id"]].merge(key_df, on="game_id", how="left")
    X = pd.concat([
        X_basic.reset_index(drop=True),
        merged_engine[ENGINE_FEATURE_COLUMNS].reset_index(drop=True),
        merged_time[TIME_FEATURE_COLUMNS].reset_index(drop=True),
        merged_key[KEY_MOMENT_FEATURE_COLUMNS].reset_index(drop=True),
    ], axis=1)

    idx_train, idx_test = train_test_split(range(len(X)), test_size=0.2, random_state=RANDOM_STATE)
    game_ids = base_df["game_id"].values
    return X.iloc[idx_test], y.iloc[idx_test], game_ids[idx_test]


def get_nn_test_game_ids():
    """Reproduces train_nn_model.py's load_data() + split to recover which
    game_ids ended up in the NN's held-out test set."""
    meta = pd.read_parquet(NN_DIR / "meta.parquet")
    bracket_probs = pd.read_parquet(BRACKET_PROBS_PATH)
    merged = meta.reset_index().merge(bracket_probs, on="game_id", how="inner")

    n = len(merged)
    rng = np.random.RandomState(RANDOM_STATE)
    perm = rng.permutation(n)
    n_test = int(n * 0.1)
    test_positions = perm[:n_test]
    return merged.iloc[test_positions]["game_id"].values, merged.iloc[test_positions]["index"].values


def main():
    print("Reproducing XGBoost test split...")
    X_xgb_test, y_xgb_test, xgb_game_ids = get_xgb_test_game_ids()
    print(f"  XGBoost test set: {len(xgb_game_ids):,} games")

    print("Reproducing NN test split...")
    nn_game_ids, nn_row_positions = get_nn_test_game_ids()
    print(f"  NN test set: {len(nn_game_ids):,} games")

    shared_ids = np.intersect1d(xgb_game_ids, nn_game_ids)
    print(f"\nShared (leakage-free) ensemble test set: {len(shared_ids):,} games")

    xgb_mask = np.isin(xgb_game_ids, shared_ids)
    X_shared = X_xgb_test.iloc[xgb_mask]
    y_shared = y_xgb_test.iloc[xgb_mask].values
    shared_game_ids_ordered = xgb_game_ids[xgb_mask]

    print("\nScoring XGBoost (xgb_key_moments_v10)...")
    xgb_bundle = joblib.load(MODELS_DIR / "xgb_key_moments_v10.joblib")
    xgb_model = xgb_bundle["model"]
    xgb_preds = xgb_model.predict(X_shared[xgb_bundle["feature_columns"]])

    print("Scoring NN (nn_v12)...")
    tokens = np.load(NN_DIR / "tokens.npy")
    lengths = np.load(NN_DIR / "lengths.npy")
    bracket_probs_df = pd.read_parquet(BRACKET_PROBS_PATH)
    prob_cols = [c for c in bracket_probs_df.columns if c.startswith("p_")]
    vocab = json.loads((NN_DIR / "vocab.json").read_text())
    nn_metrics = json.loads((MODELS_DIR / "nn_v12_metrics.json").read_text())
    elo_mean, elo_std = nn_metrics["elo_mean"], nn_metrics["elo_std"]

    meta = pd.read_parquet(NN_DIR / "meta.parquet")
    id_to_row = {gid: i for i, gid in enumerate(meta["game_id"].values)}
    bp_by_id = bracket_probs_df.set_index("game_id")[prob_cols]

    rows = [id_to_row[gid] for gid in shared_game_ids_ordered]
    tok_t = torch.from_numpy(tokens[rows]).long()
    len_t = torch.from_numpy(lengths[rows]).long().clamp(min=1)
    bp_t = torch.from_numpy(bp_by_id.loc[shared_game_ids_ordered].values.astype(np.float32)).float()

    model = EloGRU(vocab_size=len(vocab), embed_dim=24, hidden_dim=48, bracket_dim=len(prob_cols))
    model.load_state_dict(torch.load(MODELS_DIR / "nn_v12_best.pt", weights_only=True))
    model.eval()
    with torch.no_grad():
        nn_preds_norm = model(tok_t, len_t, bp_t).numpy()
    nn_preds = nn_preds_norm * elo_std + elo_mean

    print("\n=== Individual model performance (shared test set) ===")
    for name, preds in (("XGBoost", xgb_preds), ("NN", nn_preds)):
        mae = mean_absolute_error(y_shared, preds)
        r2 = r2_score(y_shared, preds)
        print(f"{name:10s} MAE={mae:.1f}  R2={r2:.3f}")

    print("\n=== Ensemble (simple average) ===")
    ens_preds = (xgb_preds + nn_preds) / 2
    mae = mean_absolute_error(y_shared, ens_preds)
    r2 = r2_score(y_shared, ens_preds)
    print(f"Ensemble   MAE={mae:.1f}  R2={r2:.3f}")

    print("\n=== Weight sweep (diagnostic only, not model-selected) ===")
    for w in (0.3, 0.4, 0.5, 0.6, 0.7):
        blend = w * xgb_preds + (1 - w) * nn_preds
        m = mean_absolute_error(y_shared, blend)
        print(f"  xgb_weight={w:.1f}: MAE={m:.1f}")

    print("\n=== Ensemble MAE by Elo bracket ===")
    bracket_mae = (
        pd.DataFrame({"bracket": elo_bracket(pd.Series(y_shared)), "abs_err": np.abs(y_shared - ens_preds)})
        .groupby("bracket")["abs_err"].agg(["mean", "count"]).round(1)
    )
    print(bracket_mae)

    (MODELS_DIR / "ensemble_v13_metrics.json").write_text(json.dumps({
        "n_shared_test": len(shared_ids),
        "xgb_mae": round(mean_absolute_error(y_shared, xgb_preds), 1),
        "nn_mae": round(mean_absolute_error(y_shared, nn_preds), 1),
        "ensemble_mae": round(mae, 1),
        "ensemble_r2": round(r2, 3),
        "mae_by_bracket": bracket_mae["mean"].to_dict(),
    }, indent=2))
    print(f"\nSaved: {MODELS_DIR / 'ensemble_v13_metrics.json'}")


if __name__ == "__main__":
    main()
