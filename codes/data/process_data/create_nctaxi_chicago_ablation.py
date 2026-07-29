"""Create nested, class-balanced Chicago Taxi auxiliary-table subsets."""

from pathlib import Path

import numpy as np
import pandas as pd


SEED = 42
TARGET_COL = "dropoff_community_area"
FRACTIONS = (0.25, 0.50, 0.75)
RAW_DIR = Path(__file__).resolve().parents[1] / "table_nctaxi" / "raw"
SOURCE_PATH = RAW_DIR / "chicago_taxi.csv"


def main() -> None:
    df = pd.read_csv(SOURCE_PATH, low_memory=False)
    if df[TARGET_COL].isna().any():
        raise ValueError(f"{TARGET_COL} contains missing values")

    rng = np.random.default_rng(SEED)
    shuffled_by_class = {}
    for label, indices in df.groupby(TARGET_COL, sort=True).groups.items():
        shuffled = np.asarray(indices, dtype=np.int64).copy()
        rng.shuffle(shuffled)
        shuffled_by_class[label] = shuffled

    previous_indices = set()
    for fraction in FRACTIONS:
        selected = []
        expected_per_class = {}
        for label, shuffled in shuffled_by_class.items():
            sample_size = int(len(shuffled) * fraction)
            selected.extend(shuffled[:sample_size])
            expected_per_class[label] = sample_size

        selected_set = set(selected)
        if not previous_indices.issubset(selected_set):
            raise RuntimeError("Generated subsets are not nested")
        previous_indices = selected_set

        subset = df.loc[sorted(selected)].reset_index(drop=True)
        actual_counts = subset[TARGET_COL].value_counts().to_dict()
        if actual_counts != expected_per_class:
            raise RuntimeError(f"Class-balance validation failed for {fraction:.0%}")

        pct = int(fraction * 100)
        output_path = RAW_DIR / f"chicago_taxi_{pct}pct.csv"
        subset.to_csv(output_path, index=False)
        print(
            f"Saved {output_path}: rows={len(subset)}, "
            f"classes={subset[TARGET_COL].nunique()}, "
            f"rows_per_class={next(iter(expected_per_class.values()))}"
        )


if __name__ == "__main__":
    main()
