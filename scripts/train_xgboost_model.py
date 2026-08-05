# scripts/train_xgboost_model.py - XGBoost vs. RandomForest, same data/split
"""Same 114k-game engine-analyzed subset, same features, same train/test
split as engine_rf_v3 (MAE 242.6, R2 0.485) - swaps in XGBoost to see
whether gradient boosting beats bagged RandomForest on this feature set.

Usage: python scripts/train_xgboost_model.py
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
ENGINE_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "engine_features.parquet"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
RANDOM_STATE = 42
TARGET_MAE = 200

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
    print(f"  {X.shape[0]:,} games, {X.shape[1]} features")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    # Hold out a slice of train for early stopping.
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=RANDOM_STATE
    )

    print("\nTraining XGBRegressor...")
    t0 = time.time()
    model = XGBRegressor(
        n_estimators=2000,
        learning_rate=0.03,
        max_depth=6,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        early_stopping_rounds=50,
        eval_metric="mae",
    )
    model.fit(X_fit, y_fit, eval_set=[(X_val, y_val)], verbose=False)
    print(f"  Trained in {time.time() - t0:.1f}s, best_iteration={model.best_iteration}")

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = mean_squared_error(y_test, preds) ** 0.5
    r2 = r2_score(y_test, preds)

    print("\n=== Test set performance ===")
    print(f"MAE:  {mae:.1f}  (target: < {TARGET_MAE})  -> {'PASS' if mae < TARGET_MAE else 'FAIL'}")
    print(f"RMSE: {rmse:.1f}")
    print(f"R2:   {r2:.3f}")

    engine_v3_metrics_path = MODELS_DIR / "engine_rf_v3_metrics.json"
    if engine_v3_metrics_path.exists():
        rf_metrics = json.loads(engine_v3_metrics_path.read_text())["comparison_b_with_engine"]
        print(f"\n=== vs. RandomForest (engine_rf_v3, same subset/split) ===")
        print(f"MAE:  {rf_metrics['mae']:.1f} -> {mae:.1f}  ({mae - rf_metrics['mae']:+.1f})")
        print(f"R2:   {rf_metrics['r2']:.3f} -> {r2:.3f}")

    print("\n=== MAE by Elo bracket ===")
    bracket_mae = (
        pd.DataFrame({"bracket": elo_bracket(y_test), "abs_err": (y_test - preds).abs()})
        .groupby("bracket")["abs_err"].agg(["mean", "count"]).round(1)
    )
    print(bracket_mae)

    print("\n=== Top 20 feature importances ===")
    importances = (
        pd.Series(model.feature_importances_, index=X.columns)
        .sort_values(ascending=False).head(20)
    )
    print(importances.round(4))

    model_path = MODELS_DIR / "xgb_v4.joblib"
    joblib.dump({"model": model, "feature_columns": list(X.columns)}, model_path)
    print(f"\nSaved model: {model_path}")

    metrics = {
        "model": "XGBRegressor_v4",
        "n_train": len(X_fit), "n_val": len(X_val), "n_test": len(X_test),
        "best_iteration": int(model.best_iteration),
        "mae": round(mae, 1), "rmse": round(rmse, 1), "r2": round(r2, 3),
        "target_mae": TARGET_MAE, "passes_target": bool(mae < TARGET_MAE),
        "mae_by_bracket": bracket_mae["mean"].to_dict(),
    }
    (MODELS_DIR / "xgb_v4_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"Saved metrics: {MODELS_DIR / 'xgb_v4_metrics.json'}")


if __name__ == "__main__":
    main()
