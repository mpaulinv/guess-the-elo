# scripts/train_enhanced_model.py - Baseline + move-replay features
"""Trains the RandomForest Elo predictor on the combined feature set (basic
per-game counts + move-replay features: development, material swings,
hanging pieces, book depth, sustained-advantage conversion) and compares
against the v1 baseline.

Usage: python scripts/train_enhanced_model.py
"""
import json
import sys
import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.combined_features import build_combined_features, load_move_features  # noqa: E402
from src.features import elo_bracket, load_quality_games  # noqa: E402

DATA_CSV = Path(__file__).resolve().parent.parent / "data" / "enhanced_extraction" / "enhanced_experiment_20250620_203308.csv"
PARTS_DIR = Path(__file__).resolve().parent.parent / "data" / "processed" / "move_features_parts"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
BASELINE_METRICS = MODELS_DIR / "baseline_rf_v1_metrics.json"
RANDOM_STATE = 42
TARGET_MAE = 200


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading quality games + move-replay features...")
    t0 = time.time()
    base_df = load_quality_games(DATA_CSV)
    move_df = load_move_features(PARTS_DIR)
    print(f"  {len(base_df):,} quality games, {len(move_df):,} move-feature rows loaded in {time.time() - t0:.1f}s")

    X, y = build_combined_features(base_df, move_df)
    print(f"  Feature matrix after join: {X.shape} ({len(base_df) - len(X):,} games dropped - unparseable movetext)")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    print("\nTraining RandomForestRegressor...")
    t0 = time.time()
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=20,
        min_samples_leaf=5,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train)
    print(f"  Trained in {time.time() - t0:.1f}s")

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = mean_squared_error(y_test, preds) ** 0.5
    r2 = r2_score(y_test, preds)

    print("\n=== Test set performance ===")
    print(f"MAE:  {mae:.1f}  (target: < {TARGET_MAE})  -> {'PASS' if mae < TARGET_MAE else 'FAIL'}")
    print(f"RMSE: {rmse:.1f}")
    print(f"R2:   {r2:.3f}")

    if BASELINE_METRICS.exists():
        baseline = json.loads(BASELINE_METRICS.read_text())
        print(f"\n=== vs. baseline v1 (basic features only) ===")
        print(f"MAE:  {baseline['mae']:.1f} -> {mae:.1f}  ({mae - baseline['mae']:+.1f})")
        print(f"R2:   {baseline['r2']:.3f} -> {r2:.3f}")

    print("\n=== MAE by Elo bracket ===")
    bracket_mae = (
        pd.DataFrame({"bracket": elo_bracket(y_test), "abs_err": (y_test - preds).abs()})
        .groupby("bracket")["abs_err"]
        .agg(["mean", "count"])
        .round(1)
    )
    print(bracket_mae)

    print("\n=== Top 20 feature importances ===")
    importances = (
        pd.Series(model.feature_importances_, index=X.columns)
        .sort_values(ascending=False)
        .head(20)
    )
    print(importances.round(4))

    model_path = MODELS_DIR / "enhanced_rf_v2.joblib"
    joblib.dump({"model": model, "feature_columns": list(X.columns)}, model_path)
    print(f"\nSaved model: {model_path}")

    metrics = {
        "model": "RandomForestRegressor_enhanced_v2",
        "n_train": len(X_train),
        "n_test": len(X_test),
        "mae": round(mae, 1),
        "rmse": round(rmse, 1),
        "r2": round(r2, 3),
        "target_mae": TARGET_MAE,
        "passes_target": bool(mae < TARGET_MAE),
        "mae_by_bracket": bracket_mae["mean"].to_dict(),
    }
    metrics_path = MODELS_DIR / "enhanced_rf_v2_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"Saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
