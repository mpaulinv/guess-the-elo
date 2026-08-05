# scripts/train_balanced_model.py - Class imbalance: sample-weight vs. oversampling
"""Same 114k-game engine-analyzed subset, same train/test split, same
features as engine_rf_v3 (comparison_b_with_engine). Three variants:

  (B) unweighted             - the existing v3 result, as a control
  (C) inverse-frequency sample_weight - reweights the loss/bootstrap toward
      rare Elo brackets without duplicating any rows
  (D) literal oversampling   - duplicates rows in rare brackets up to a
      capped factor (the "traditional" oversampling approach)

Usage: python scripts/train_balanced_model.py
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.combined_features import build_combined_features, load_move_features  # noqa: E402
from src.features import elo_bracket, load_quality_games  # noqa: E402

DATA_CSV = Path(__file__).resolve().parent.parent / "data" / "enhanced_extraction" / "enhanced_experiment_20250620_203308.csv"
PARTS_DIR = Path(__file__).resolve().parent.parent / "data" / "processed" / "move_features_parts"
ENGINE_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "engine_features.parquet"
RANDOM_STATE = 42
MAX_OVERSAMPLE_FACTOR = 8  # cap duplication so tiny brackets don't get cloned 50-60x

ENGINE_FEATURE_COLUMNS = [
    "eval_coverage", "white_acpl", "black_acpl", "avg_acpl",
    "white_blunders", "black_blunders",
    "white_mistakes", "black_mistakes",
    "white_inaccuracies", "black_inaccuracies",
    "max_loss_white", "max_loss_black", "first_blunder_ply", "eval_std",
    "opening_acpl", "middlegame_acpl", "endgame_acpl",
    "has_middlegame", "has_endgame", "balance_ratio",
    "sustained_eval_advantage_plies", "eval_advantage_converted",
    "big_eval_advantage_not_converted",
    "white_missed_mate_in_1", "black_missed_mate_in_1",
    "white_missed_mate_in_2", "black_missed_mate_in_2",
]


def fit_eval(X_train, y_train, X_test, y_test, label, sample_weight=None):
    print(f"\n--- {label} (n_train={len(X_train):,}) ---")
    t0 = time.time()
    model = RandomForestRegressor(
        n_estimators=200, max_depth=20, min_samples_leaf=5,
        n_jobs=-1, random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train, sample_weight=sample_weight)
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    print(f"  trained in {time.time()-t0:.1f}s | MAE={mae:.1f} R2={r2:.3f}")
    bracket_mae = (
        pd.DataFrame({"bracket": elo_bracket(y_test), "abs_err": (y_test - preds).abs()})
        .groupby("bracket")["abs_err"].agg(["mean", "count"]).round(1)
    )
    print(bracket_mae)
    return {"mae": round(mae, 1), "r2": round(r2, 3), "mae_by_bracket": bracket_mae["mean"].to_dict()}


def main():
    print("Loading data...")
    base_df = load_quality_games(DATA_CSV)
    move_df = load_move_features(PARTS_DIR)
    engine_df = pd.read_parquet(ENGINE_PATH)
    base_df = base_df[base_df["game_id"].isin(engine_df["game_id"])].reset_index(drop=True)

    X_basic, y = build_combined_features(base_df, move_df)
    merged_engine = base_df[["game_id"]].merge(engine_df, on="game_id", how="left")
    X_engine = merged_engine[ENGINE_FEATURE_COLUMNS].reset_index(drop=True)
    X = pd.concat([X_basic.reset_index(drop=True), X_engine], axis=1)

    idx_train, idx_test = train_test_split(range(len(X)), test_size=0.2, random_state=RANDOM_STATE)
    X_train, X_test = X.iloc[idx_train].reset_index(drop=True), X.iloc[idx_test].reset_index(drop=True)
    y_train, y_test = y.iloc[idx_train].reset_index(drop=True), y.iloc[idx_test].reset_index(drop=True)
    bracket_train = elo_bracket(y_train)

    print("\nTrain bracket distribution:")
    print(bracket_train.value_counts())

    results = {}

    results["B_unweighted"] = fit_eval(X_train, y_train, X_test, y_test, "(B) unweighted (control)")

    # (C) inverse-frequency sample weights - no row duplication.
    freq = bracket_train.value_counts()
    weights = bracket_train.map(lambda b: 1.0 / freq[b])
    weights = weights / weights.mean()  # normalize around 1.0
    results["C_sample_weight"] = fit_eval(
        X_train, y_train, X_test, y_test, "(C) inverse-frequency sample_weight", sample_weight=weights.values
    )

    # (D) literal oversampling, capped at MAX_OVERSAMPLE_FACTOR x per bracket.
    max_count = freq.max()
    oversampled_positions = []
    for b, count in freq.items():
        positions_b = np.flatnonzero((bracket_train == b).values)
        target = min(max_count, count * MAX_OVERSAMPLE_FACTOR)
        if target > count:
            extra = resample(positions_b, replace=True, n_samples=int(target - count), random_state=RANDOM_STATE)
            positions_b = np.concatenate([positions_b, extra])
        oversampled_positions.append(positions_b)
    oversampled_positions = np.concatenate(oversampled_positions)
    print(f"\n(D) oversampled train size: {len(X_train):,} -> {len(oversampled_positions):,} "
          f"(cap {MAX_OVERSAMPLE_FACTOR}x per bracket)")
    results["D_oversampled"] = fit_eval(
        X_train.iloc[oversampled_positions], y_train.iloc[oversampled_positions],
        X_test, y_test, f"(D) oversampled (cap {MAX_OVERSAMPLE_FACTOR}x)",
    )

    print("\n=== Summary: overall MAE / R2 ===")
    for k, v in results.items():
        print(f"{k:20s} MAE={v['mae']:6.1f}  R2={v['r2']:.3f}")

    print("\n=== Summary: MAE by bracket ===")
    order = ["Sub-beginner (<800)", "Beginner (800-1200)", "Intermediate (1200-1600)",
              "Advanced (1600-2000)", "Expert (2000-2400)", "Master+ (2400+)"]
    table = pd.DataFrame({k: v["mae_by_bracket"] for k, v in results.items()}).reindex(order)
    print(table.round(1))


if __name__ == "__main__":
    main()
