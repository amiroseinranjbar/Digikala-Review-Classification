
import os
import pandas as pd

# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

INPUT_FILE = "train_balanced.csv"

OUTPUT_DIR = "digikala_rating_sampling_v1"
OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "digikala_rating_semi_supervised_600k_v1.csv"
)

TEXT_COL = "text"
RATE_COL = "rate"

SAMPLES_PER_CLASS = 200_000
RANDOM_STATE = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --------------------------------------------------
# 2. Load data
# --------------------------------------------------

df = pd.read_csv(INPUT_FILE)

required_columns = [TEXT_COL, RATE_COL]

for col in required_columns:
    if col not in df.columns:
        raise ValueError(f"Missing required column: {col}")

print("Original dataset:", len(df))

# --------------------------------------------------
# 3. Clean rate values
# --------------------------------------------------

df[RATE_COL] = pd.to_numeric(
    df[RATE_COL],
    errors="coerce"
)

# --------------------------------------------------
# 4. Assign rating bands
# --------------------------------------------------

df["rating_band"] = pd.NA

df.loc[df[RATE_COL].isin([1, 2]), "rating_band"] = "low"
df.loc[df[RATE_COL] == 3, "rating_band"] = "medium"
df.loc[df[RATE_COL].isin([4, 5]), "rating_band"] = "high"

# --------------------------------------------------
# 5. Assign label status
# --------------------------------------------------

df["label_status"] = "invalid"

df.loc[
    df[RATE_COL].isin([0]) | df[RATE_COL].isna(),
    "label_status"
] = "unlabeled"

df.loc[
    df["rating_band"].notna(),
    "label_status"
] = "labeled"

# --------------------------------------------------
# 6. Separate labeled classes
# --------------------------------------------------

low_df = df[df["rating_band"] == "low"]
medium_df = df[df["rating_band"] == "medium"]
high_df = df[df["rating_band"] == "high"]

class_data = {
    "low": low_df,
    "medium": medium_df,
    "high": high_df
}

# --------------------------------------------------
# 7. Check class sizes
# --------------------------------------------------

for class_name, class_df in class_data.items():
    print(f"{class_name}: {len(class_df):,}")

    if len(class_df) < SAMPLES_PER_CLASS:
        raise ValueError(
            f"Not enough samples in {class_name}: "
            f"{len(class_df):,} available, "
            f"{SAMPLES_PER_CLASS:,} required."
        )

# --------------------------------------------------
# 8. Random sampling
# --------------------------------------------------

sampled_classes = []

for class_name, class_df in class_data.items():

    sampled = class_df.sample(
        n=SAMPLES_PER_CLASS,
        replace=False,
        random_state=RANDOM_STATE
    )

    sampled_classes.append(sampled)

# --------------------------------------------------
# 9. Combine labeled samples
# --------------------------------------------------

labeled_data = pd.concat(
    sampled_classes,
    ignore_index=True
)

# --------------------------------------------------
# 10. Add all unlabeled comments
# --------------------------------------------------

unlabeled_data = df[
    df["label_status"] == "unlabeled"
].copy()

final_data = pd.concat(
    [labeled_data, unlabeled_data],
    ignore_index=True
)

# --------------------------------------------------
# 11. Encode labels
# --------------------------------------------------

label_mapping = {
    "low": 0,
    "medium": 1,
    "high": 2
}

final_data["labels"] = (
    final_data["rating_band"]
    .map(label_mapping)
    .astype("Int64")
)

# --------------------------------------------------
# 12. Shuffle final dataset
# --------------------------------------------------

final_data = final_data.sample(
    frac=1,
    random_state=RANDOM_STATE
).reset_index(drop=True)

# --------------------------------------------------
# 13. Save output
# --------------------------------------------------

if os.path.exists(OUTPUT_FILE):
    raise FileExistsError(
        f"Output already exists: {OUTPUT_FILE}"
    )

final_data.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)

# --------------------------------------------------
# 14. Final report
# --------------------------------------------------

print("\n" + "=" * 50)
print("SAMPLING COMPLETED")
print("=" * 50)

print("Labeled samples:")
print(
    final_data["rating_band"]
    .value_counts()
    .reindex(["low", "medium", "high"], fill_value=0)
)

print("\nUnlabeled samples:", len(unlabeled_data))
print("Final dataset size:", len(final_data))
print("Saved to:", OUTPUT_FILE)
