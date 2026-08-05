# scripts/train_xgboost_full_time.py - Time features on the FULL dataset
"""Fair A/B on the full ~994k quality games (not just the 11% engine-
analyzed subset, since clock data covers nearly everyone):
  (A) basic + move-replay features only (control)
  (B) basic + move-replay + time-management features (treatment)
Same train/test split for both, so any MAE delta is attributable to the
time features themselves.

Usage: python scripts/train_xgboost_full_time.py
"""
import json
import sys
import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.combined_features import build_combined_features, load_move_features  # noqa: E402
from src.features import elo_bracket, load_quality_games  # noqa: E402

DATA_CSV = Path(__file__).resolve().parent.parent / "data" / "enhanced_extraction" / "enhanced_experiment_20250620_203308.csv"
PARTS_DIR = Path(__file__).resolve().parent.parent / "data" / "processed" / "move_features_parts"
TIME_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "time_features.parquet"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
RANDOM_STATE = 42
TARGET_MAE = 200

TIME_FEATURE_COLUMNS = [
    "clock_coverage",
    "white_time_per_move", "black_time_per_move",
    "white_time_std", "black_time_std",
    "white_low_time_moves", "black_low_time_moves",
    "white_fast_moves", "black_fast_moves",
    "white_first_low_time_ply", "black_first_low_time_ply",
    "white_opening_time_per_move", "black_opening_time_per_move",
    "white_pace_used_by_ply20", "black_pace_used_by_ply20",
]


def fit_eval(X_train, y_train, X_test, y_test, label):
    print(f"\n--- {label} ({X_train.shape[1]} features, n_train={len(X_train):,}) ---")
    X_fit, X_val, y_fit, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=RANDOM_STATE)
    t0 = time.time()
    model = XGBRegressor(
        n_estimators=2000, learning_rate=0.03, max_depth=6,
        min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
        reg_lambda=1.0, random_state=RANDOM_STATE, n_jobs=4,
        early_stopping_rounds=50, eval_metric="mae",
    )
    model.fit(X_fit, y_fit, eval_set=[(X_val, y_val)], verbose=False)
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    print(f"  trained in {time.time()-t0:.1f}s, best_iter={model.best_iteration} | MAE={mae:.1f} R2={r2:.3f}")
    bracket_mae = (
        pd.DataFrame({"bracket": elo_bracket(y_test), "abs_err": (y_test - preds).abs()})
        .groupby("bracket")["abs_err"].agg(["mean", "count"]).round(1)
    )
    print(bracket_mae)
    return model, {"mae": round(mae, 1), "r2": round(r2, 3), "mae_by_bracket": bracket_mae["mean"].to_dict()}


def main():
    print("Loading full dataset...")
    base_df = load_quality_games(DATA_CSV)
    move_df = load_move_features(PARTS_DIR)
    time_df = pd.read_parquet(TIME_PATH)
    base_df = base_df[base_df["game_id"].isin(time_df["game_id"])].reset_index(drop=True)

    X_basic, y = build_combined_features(base_df, move_df)
    merged_time = base_df[["game_id"]].merge(time_df, on="game_id", how="left")
    X_time = merged_time[TIME_FEATURE_COLUMNS].reset_index(drop=True)
    print(f"  {X_basic.shape[0]:,} games")

    idx_train, idx_test = train_test_split(range(len(X_basic)), test_size=0.2, random_state=RANDOM_STATE)
    y_train, y_test = y.iloc[idx_train], y.iloc[idx_test]

    model_a, metrics_a = fit_eval(
        X_basic.iloc[idx_train], y_train, X_basic.iloc[idx_test], y_test,
        "(A) basic+move-replay only, full dataset",
    )

    X_combined = pd.concat([X_basic.reset_index(drop=True), X_time.reset_index(drop=True)], axis=1)
    model_b, metrics_b = fit_eval(
        X_combined.iloc[idx_train], y_train, X_combined.iloc[idx_test], y_test,
        "(B) basic+move-replay+time features, full dataset",
    )

    print("\n=== Summary ===")
    print(f"(A) no time features:   MAE={metrics_a['mae']}  R2={metrics_a['r2']}")
    print(f"(B) with time features: MAE={metrics_b['mae']}  R2={metrics_b['r2']}  "
          f"(delta {metrics_b['mae']-metrics_a['mae']:+.1f})")

    print("\n=== Top 20 feature importances, model (B) ===")
    importances = (
        pd.Series(model_b.feature_importances_, index=X_combined.columns)
        .sort_values(ascending=False).head(20)
    )
    print(importances.round(4))

    joblib.dump({"model": model_b, "feature_columns": list(X_combined.columns)},
                MODELS_DIR / "xgb_time_v8.joblib")
    (MODELS_DIR / "xgb_time_v8_metrics.json").write_text(json.dumps({
        "model": "XGBRegressor_time_v8", "target_mae": TARGET_MAE,
        "passes_target": bool(metrics_b["mae"] < TARGET_MAE),
        "comparison_a_no_time": metrics_a, "comparison_b_with_time": metrics_b,
    }, indent=2))
    print(f"\nSaved model: {MODELS_DIR / 'xgb_time_v8.joblib'}")


if __name__ == "__main__":
    main()
