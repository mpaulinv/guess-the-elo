# scripts/train_full_classifier.py - Bracket classifier over ALL quality games
"""Unlike train_classifier_model.py (which used the 114k engine-analyzed
subset + engine features), this trains on basic+move-replay features across
the full ~994k quality games - engine features only cover 11% of games, but
the neural network needs a classifier prior for every game. Saves
per-game predicted bracket probabilities for use as NN input.

Usage: python scripts/train_full_classifier.py
"""
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 220)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.combined_features import build_combined_features, load_move_features  # noqa: E402
from src.features import ELO_BRACKETS, elo_bracket, load_quality_games  # noqa: E402

DATA_CSV = Path(__file__).resolve().parent.parent / "data" / "enhanced_extraction" / "enhanced_experiment_20250620_203308.csv"
PARTS_DIR = Path(__file__).resolve().parent.parent / "data" / "processed" / "move_features_parts"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
OUT_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "bracket_probs_full.parquet"
RANDOM_STATE = 42
BRACKET_ORDER = [name for _, _, name in ELO_BRACKETS]


def main():
    print("Loading full quality-game set (basic + move-replay features)...")
    base_df = load_quality_games(DATA_CSV)
    move_df = load_move_features(PARTS_DIR)
    X, y_elo = build_combined_features(base_df, move_df)
    game_ids = base_df.merge(move_df[["game_id"]], on="game_id")["game_id"].reset_index(drop=True)
    print(f"  {X.shape[0]:,} games, {X.shape[1]} features")

    bracket = elo_bracket(y_elo).reset_index(drop=True)
    le = LabelEncoder()
    le.fit(BRACKET_ORDER)
    y = pd.Series(le.transform(bracket), index=X.index)
    label_names = list(le.classes_)

    X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
        X, y, game_ids, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=RANDOM_STATE, stratify=y_train
    )

    print("\nTraining XGBClassifier on full dataset...")
    t0 = time.time()
    model = XGBClassifier(
        n_estimators=1000, learning_rate=0.05, max_depth=6,
        min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
        random_state=RANDOM_STATE, n_jobs=-1,
        early_stopping_rounds=50, eval_metric="mlogloss",
    )
    model.fit(X_fit, y_fit, eval_set=[(X_val, y_val)], verbose=False)
    print(f"  trained in {time.time()-t0:.1f}s, best_iteration={model.best_iteration}")

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"  test accuracy: {acc:.3f}")
    print(classification_report(y_test, preds, target_names=label_names, digits=3, zero_division=0))

    # Predicted bracket-probability vector for every game in the dataset -
    # this is what the NN will condition on.
    print("\nScoring all games for NN conditioning input...")
    all_probs = model.predict_proba(X)
    probs_df = pd.DataFrame(all_probs, columns=[f"p_{c}" for c in label_names])
    out = pd.concat([pd.DataFrame({"game_id": game_ids}), probs_df], axis=1)
    out.to_parquet(OUT_PATH)
    print(f"Saved bracket probabilities for {len(out):,} games: {OUT_PATH}")

    joblib.dump({"model": model, "feature_columns": list(X.columns), "label_classes": label_names},
                MODELS_DIR / "classifier_full_v6.joblib")
    print(f"Saved model: {MODELS_DIR / 'classifier_full_v6.joblib'}")


if __name__ == "__main__":
    main()
