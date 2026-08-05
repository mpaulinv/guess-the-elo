# scripts/train_classifier_model.py - Bracket classification (not exact Elo)
"""Same 114k-game engine-analyzed subset and feature set as xgb_v4, but
predicts the Elo bracket (6-class) instead of a point estimate. Trains both
an unweighted and a class-balanced XGBClassifier so we can see per-bracket
accuracy under both regimes, not just the (majority-class-dominated) overall
accuracy number.

Usage: python scripts/train_classifier_model.py
"""
import sys
import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 220)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.combined_features import build_combined_features, load_move_features  # noqa: E402
from src.features import ELO_BRACKETS, elo_bracket, load_quality_games  # noqa: E402

DATA_CSV = Path(__file__).resolve().parent.parent / "data" / "enhanced_extraction" / "enhanced_experiment_20250620_203308.csv"
PARTS_DIR = Path(__file__).resolve().parent.parent / "data" / "processed" / "move_features_parts"
ENGINE_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "engine_features.parquet"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
RANDOM_STATE = 42

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

BRACKET_ORDER = [name for _, _, name in ELO_BRACKETS]


def fit_eval(X_train, y_train, X_test, y_test, label_names, label, sample_weight=None):
    print(f"\n--- {label} ---")
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=RANDOM_STATE, stratify=y_train
    )
    w_fit = None
    if sample_weight is not None:
        w_fit = sample_weight.loc[X_fit.index] if hasattr(sample_weight, "loc") else sample_weight[X_fit.index]

    t0 = time.time()
    model = XGBClassifier(
        n_estimators=1000, learning_rate=0.05, max_depth=6,
        min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
        random_state=RANDOM_STATE, n_jobs=-1,
        early_stopping_rounds=50, eval_metric="mlogloss",
    )
    model.fit(X_fit, y_fit, sample_weight=w_fit, eval_set=[(X_val, y_val)], verbose=False)
    print(f"  trained in {time.time()-t0:.1f}s, best_iteration={model.best_iteration}")

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"  overall accuracy: {acc:.3f}")
    print(classification_report(y_test, preds, target_names=label_names, digits=3, zero_division=0))

    short = [n.split(" (")[0] for n in label_names]
    cm = confusion_matrix(y_test, preds, labels=range(len(label_names)))
    cm_df = pd.DataFrame(cm, index=[f"true:{n}" for n in short], columns=short)
    print("Confusion matrix (rows=true, cols=predicted):")
    print(cm_df)
    return model, acc


def main():
    print("Loading data...")
    base_df = load_quality_games(DATA_CSV)
    move_df = load_move_features(PARTS_DIR)
    engine_df = pd.read_parquet(ENGINE_PATH)
    base_df = base_df[base_df["game_id"].isin(engine_df["game_id"])].reset_index(drop=True)

    X_basic, y_elo = build_combined_features(base_df, move_df)
    merged_engine = base_df[["game_id"]].merge(engine_df, on="game_id", how="left")
    X_engine = merged_engine[ENGINE_FEATURE_COLUMNS].reset_index(drop=True)
    X = pd.concat([X_basic.reset_index(drop=True), X_engine], axis=1)

    bracket = elo_bracket(y_elo).reset_index(drop=True)
    le = LabelEncoder()
    le.fit(BRACKET_ORDER)
    y = pd.Series(le.transform(bracket), index=X.index)
    label_names = list(le.classes_)

    print(f"  {X.shape[0]:,} games, {X.shape[1]} features, {len(label_names)} brackets")
    print(bracket.value_counts().reindex(BRACKET_ORDER))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    model_a, acc_a = fit_eval(X_train, y_train, X_test, y_test, label_names, "(A) unweighted XGBClassifier")

    weights_train = pd.Series(
        compute_sample_weight("balanced", y_train), index=X_train.index
    )
    model_b, acc_b = fit_eval(
        X_train, y_train, X_test, y_test, label_names,
        "(B) class-balanced XGBClassifier", sample_weight=weights_train,
    )

    print(f"\n=== Summary ===")
    print(f"(A) unweighted:       accuracy={acc_a:.3f}")
    print(f"(B) class-balanced:   accuracy={acc_b:.3f}")

    joblib.dump({"model": model_b, "feature_columns": list(X.columns), "label_classes": label_names},
                MODELS_DIR / "classifier_v5.joblib")
    print(f"\nSaved model (B, balanced): {MODELS_DIR / 'classifier_v5.joblib'}")


if __name__ == "__main__":
    main()
