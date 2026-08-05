# scripts/train_engine_model.py - Isolate the effect of engine (%eval) features
"""Fair A/B comparison on the SAME 114k-game engine-analyzed subset:
  (A) basic + move-replay features only  (same recipe as enhanced_rf_v2)
  (B) basic + move-replay + ACPL/blunder engine features
...using the same train/test split for both, so any MAE difference is
attributable to the engine features themselves, not to the subset being
smaller/different from the full 996k games.

Usage: python scripts/train_engine_model.py
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


def fit_eval(X_train, X_test, y_train, y_test, label):
    print(f"\n--- Training {label} ({X_train.shape[1]} features) ---")
    t0 = time.time()
    model = RandomForestRegressor(
        n_estimators=200, max_depth=20, min_samples_leaf=5,
        n_jobs=-1, random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = mean_squared_error(y_test, preds) ** 0.5
    r2 = r2_score(y_test, preds)
    print(f"  trained in {time.time()-t0:.1f}s | MAE={mae:.1f} RMSE={rmse:.1f} R2={r2:.3f}")

    bracket_mae = (
        pd.DataFrame({"bracket": elo_bracket(y_test), "abs_err": (y_test - preds).abs()})
        .groupby("bracket")["abs_err"].agg(["mean", "count"]).round(1)
    )
    print(bracket_mae)
    return model, {"mae": round(mae, 1), "rmse": round(rmse, 1), "r2": round(r2, 3),
                    "mae_by_bracket": bracket_mae["mean"].to_dict()}


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    base_df = load_quality_games(DATA_CSV)
    move_df = load_move_features(PARTS_DIR)
    engine_df = pd.read_parquet(ENGINE_PATH)

    # Restrict the whole comparison to games that have BOTH move-replay and
    # engine features, so (A) and (B) run on the exact same rows.
    base_df = base_df[base_df["game_id"].isin(engine_df["game_id"])].reset_index(drop=True)
    print(f"  engine-analyzed subset with full features: {len(base_df):,} games")

    X_basic, y = build_combined_features(base_df, move_df)
    merged_engine = base_df[["game_id"]].merge(engine_df, on="game_id", how="left")
    X_engine = merged_engine[ENGINE_FEATURE_COLUMNS].reset_index(drop=True)

    idx_train, idx_test = train_test_split(
        range(len(X_basic)), test_size=0.2, random_state=RANDOM_STATE
    )

    y_train, y_test = y.iloc[idx_train], y.iloc[idx_test]

    model_a, metrics_a = fit_eval(
        X_basic.iloc[idx_train], X_basic.iloc[idx_test], y_train, y_test,
        "(A) basic+move-replay only, engine-analyzed subset",
    )

    X_combined = pd.concat([X_basic.reset_index(drop=True), X_engine], axis=1)
    model_b, metrics_b = fit_eval(
        X_combined.iloc[idx_train], X_combined.iloc[idx_test], y_train, y_test,
        "(B) basic+move-replay+engine ACPL features",
    )

    print("\n=== Summary ===")
    print(f"(A) no engine features:  MAE={metrics_a['mae']}  R2={metrics_a['r2']}")
    print(f"(B) with engine features: MAE={metrics_b['mae']}  R2={metrics_b['r2']}  "
          f"(delta MAE {metrics_b['mae']-metrics_a['mae']:+.1f})")

    print("\n=== Top 20 feature importances, model (B) ===")
    importances = (
        pd.Series(model_b.feature_importances_, index=X_combined.columns)
        .sort_values(ascending=False).head(20)
    )
    print(importances.round(4))

    joblib.dump({"model": model_b, "feature_columns": list(X_combined.columns)},
                MODELS_DIR / "engine_rf_v3.joblib")
    (MODELS_DIR / "engine_rf_v3_metrics.json").write_text(json.dumps({
        "model": "RandomForestRegressor_engine_v3",
        "n_train": len(idx_train), "n_test": len(idx_test),
        "target_mae": TARGET_MAE, "passes_target": bool(metrics_b["mae"] < TARGET_MAE),
        "comparison_a_no_engine": metrics_a,
        "comparison_b_with_engine": metrics_b,
    }, indent=2))
    print(f"\nSaved model: {MODELS_DIR / 'engine_rf_v3.joblib'}")


if __name__ == "__main__":
    main()
