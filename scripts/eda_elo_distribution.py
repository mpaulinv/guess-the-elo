# scripts/eda_elo_distribution.py - Data review + Elo distribution check
"""Reviews the extracted games CSV and checks how Elo is represented in the
data: overall shape/nulls, bracket counts, skew across time controls, and
whether the "quality game" filter biases the Elo distribution.

Usage: python scripts/eda_elo_distribution.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.features import USECOLS, ELO_BRACKETS, elo_bracket  # noqa: E402

EDA_USECOLS = USECOLS + ["elo_diff"]

DATA_CSV = Path(__file__).resolve().parent.parent / "data" / "enhanced_extraction" / "enhanced_experiment_20250620_203308.csv"
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "processed" / "eda"


def bracket_table(elo: pd.Series) -> pd.DataFrame:
    counts = elo_bracket(elo).value_counts()
    order = [name for _, _, name in ELO_BRACKETS]
    counts = counts.reindex(order).fillna(0).astype(int)
    pct = (counts / counts.sum() * 100).round(1)
    return pd.DataFrame({"count": counts, "pct": pct})


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading", DATA_CSV.name)
    df = pd.read_csv(DATA_CSV, usecols=EDA_USECOLS)
    print(f"\n=== Shape: {df.shape[0]:,} rows x {df.shape[1]} cols ===")

    print("\n=== Nulls ===")
    print(df.isnull().sum()[df.isnull().sum() > 0])

    quality = df[df["is_quality_game"] & (df["time_class"] != "unknown")]
    dropped = len(df) - len(quality)
    print(f"\n=== Quality filter ===\n"
          f"kept {len(quality):,} / {len(df):,} ({len(quality)/len(df)*100:.1f}%), "
          f"dropped {dropped:,}")

    print("\n=== avg_elo describe: all vs quality-filtered ===")
    print(pd.DataFrame({
        "all_games": df["avg_elo"].describe(),
        "quality_games": quality["avg_elo"].describe(),
    }).round(1))

    print("\n=== Elo bracket distribution (quality-filtered games) ===")
    bt = bracket_table(quality["avg_elo"])
    print(bt)
    bt.to_csv(OUT_DIR / "elo_bracket_distribution.csv")

    print("\n=== Elo bracket distribution BY time_class ===")
    for tc, group in quality.groupby("time_class"):
        print(f"\n-- {tc} (n={len(group):,}) --")
        print(bracket_table(group["avg_elo"]))

    print("\n=== elo_diff (rating gap between opponents) describe ===")
    print(quality["elo_diff"].abs().describe().round(1))

    # White vs black Elo should be near-symmetric (Lichess pairs similar
    # ratings) - a large mean difference would flag a pairing/extraction bug.
    print("\n=== white_elo vs black_elo mean (sanity check for pairing bias) ===")
    print(quality[["white_elo", "black_elo"]].mean().round(1))

    # --- Plots ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].hist(quality["avg_elo"], bins=80, color="#3B6EA5", edgecolor="none")
    axes[0].set_title("avg_elo distribution (quality games)")
    axes[0].set_xlabel("avg_elo")
    axes[0].set_ylabel("count")

    for tc, group in quality.groupby("time_class"):
        axes[1].hist(group["avg_elo"], bins=60, alpha=0.5, label=f"{tc} (n={len(group):,})", density=True)
    axes[1].set_title("avg_elo density by time_class")
    axes[1].set_xlabel("avg_elo")
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    fig_path = OUT_DIR / "elo_distribution.png"
    fig.savefig(fig_path, dpi=130)
    print(f"\nSaved plot: {fig_path}")


if __name__ == "__main__":
    main()
