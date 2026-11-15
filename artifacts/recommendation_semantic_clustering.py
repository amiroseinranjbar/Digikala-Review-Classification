# ============================================================
# Digikala: Persistent ParsBERT + MiniBatchKMeans Clustering
#
# Two modes:
#   ACTION = "build"    -> Extract embeddings, fit KMeans, save
#                          cluster IDs + KMeans models, then sample.
#   ACTION = "resample" -> Reuse saved cluster IDs and sample again.
#
# IMPORTANT:
# - Run "build" once. The previous notebook did not save cluster IDs.
# - For later sampling, set ACTION = "resample".
# - Keep the source CSV unchanged for resampling.
# ============================================================

import os
import gc
import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import joblib

from sklearn.cluster import MiniBatchKMeans
from transformers import AutoTokenizer, AutoModel


# ------------------------------------------------------------
# 1. Configuration
# ------------------------------------------------------------

ACTION = "build"  # Change to "resample" after the first successful build.

SOURCE_DATA_PATH = "./train_balanced.csv"

TEXT_COLUMN = "text"
LABEL_COLUMN = "recommendation_status_clean"

TARGET_PER_CLASS = {
    "recommended": 166667,
    "no_idea": 166667,
    "not_recommended": 166666,
}

SEED = 42

MODEL_NAME = "HooshvareLab/bert-base-parsbert-uncased"
MAX_LENGTH = 256
EMBEDDING_BATCH_SIZE = 32
CLUSTER_BATCH_SIZE = 4096

# Keep 200 for now; see explanation below the code.
N_CLUSTERS = 200
MIN_SAMPLES_PER_CLUSTER = 10

ARTIFACT_DIR = Path("./digikala_cluster_artifacts")
CLASS_ARTIFACT_DIR = ARTIFACT_DIR / "classes"

SAMPLE_OUTPUT_PATH = Path("./digikala_balanced_cluster_sample.csv")
REPORT_OUTPUT_PATH = ARTIFACT_DIR / "sampling_report.csv"

ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
CLASS_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", DEVICE)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# ------------------------------------------------------------
# 2. Load and prepare the source data
# ------------------------------------------------------------

def load_source_data():
    print("\nLoading source CSV...")

    data = pd.read_csv(
        SOURCE_DATA_PATH,
        usecols=[TEXT_COLUMN, LABEL_COLUMN],
        encoding="utf-8-sig",
        dtype={
            TEXT_COLUMN: "string",
            LABEL_COLUMN: "string",
        },
    )

    # The original text is already normalized.
    # Only fill missing values, cast to string, and strip whitespace.
    data[TEXT_COLUMN] = (
        data[TEXT_COLUMN]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    data[LABEL_COLUMN] = (
        data[LABEL_COLUMN]
        .astype("string")
        .str.strip()
    )

    data = data[
        data[TEXT_COLUMN].ne("")
        & data[LABEL_COLUMN].notna()
        & data[LABEL_COLUMN].ne("")
    ].copy()

    # Detect conflicting labels for identical texts.
    print("Checking conflicting labels...")
    label_counts = (
        data.groupby(TEXT_COLUMN, sort=False)[LABEL_COLUMN]
        .nunique()
    )
    conflict_count = int((label_counts > 1).sum())
    del label_counts
    gc.collect()

    if conflict_count:
        raise ValueError(
            f"{conflict_count:,} texts have conflicting labels. "
            "Resolve them before clustering."
        )

    # Keep one copy of each text, as in the original notebook.
    data = (
        data.drop_duplicates(subset=[TEXT_COLUMN], keep="first")
        .reset_index(drop=True)
    )

    missing_classes = set(TARGET_PER_CLASS) - set(data[LABEL_COLUMN].unique())
    if missing_classes:
        raise ValueError(f"Missing classes in source data: {missing_classes}")

    print("Prepared rows:", f"{len(data):,}")
    print("\nClass counts:")
    print(data[LABEL_COLUMN].value_counts())

    return data


# ------------------------------------------------------------
# 3. Fingerprint the exact ordered data for safe artifact reuse
# ------------------------------------------------------------

def dataframe_fingerprint(frame):
    """
    Detects changes in text, labels, row order, or deduplication result.
    The same source data in the same order must be used for resampling.
    """
    hashed = pd.util.hash_pandas_object(
        frame[[TEXT_COLUMN, LABEL_COLUMN]],
        index=False,
    ).to_numpy(dtype=np.uint64)

    return hashlib.sha256(hashed.tobytes()).hexdigest()


# ------------------------------------------------------------
# 4. Load ParsBERT for embedding extraction
# ------------------------------------------------------------

def load_embedding_model():
    print("\nLoading ParsBERT for embeddings...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
    model.eval()

    return tokenizer, model


# ------------------------------------------------------------
# 5. Extract mean-pooled, L2-normalized embeddings
# ------------------------------------------------------------

def extract_embeddings(texts, output_path, tokenizer, model):
    n = len(texts)
    hidden_size = model.config.hidden_size

    embeddings = np.memmap(
        output_path,
        dtype="float16",
        mode="w+",
        shape=(n, hidden_size),
    )

    for start in range(0, n, EMBEDDING_BATCH_SIZE):
        end = min(start + EMBEDDING_BATCH_SIZE, n)
        batch_texts = texts[start:end]

        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )

        encoded = {
            key: value.to(DEVICE)
            for key, value in encoded.items()
        }

        with torch.inference_mode():
            outputs = model(**encoded)
            token_embeddings = outputs.last_hidden_state

            attention_mask = (
                encoded["attention_mask"]
                .unsqueeze(-1)
                .to(token_embeddings.dtype)
            )

            summed = (token_embeddings * attention_mask).sum(dim=1)
            counts = attention_mask.sum(dim=1).clamp(min=1)
            sentence_embeddings = summed / counts

            sentence_embeddings = torch.nn.functional.normalize(
                sentence_embeddings,
                p=2,
                dim=1,
            )

        embeddings[start:end] = (
            sentence_embeddings.cpu().numpy().astype(np.float16)
        )

        if end == n or end % 16000 == 0:
            print(f"Embedding progress: {end:,}/{n:,}")

    embeddings.flush()
    del embeddings
    gc.collect()

    return hidden_size


# ------------------------------------------------------------
# 6. Fit and persist KMeans + cluster IDs for one class
# ------------------------------------------------------------

def build_class_artifacts(class_df, class_name, tokenizer, embedding_model):
    class_df = class_df.reset_index(drop=True)
    n = len(class_df)

    class_dir = CLASS_ARTIFACT_DIR / class_name
    class_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 72)
    print("BUILDING CLASS:", class_name)
    print("Rows:", f"{n:,}")
    print("=" * 72)

    k = min(N_CLUSTERS, n)

    embedding_path = class_dir / "temporary_embeddings.dat"
    cluster_ids_path = class_dir / "cluster_ids.npy"
    kmeans_path = class_dir / "kmeans.joblib"
    metadata_path = class_dir / "metadata.json"

    # If this class was already completed in an earlier run,
    # reuse it instead of repeating the expensive embedding/KMeans work.
    if (
        cluster_ids_path.exists()
        and kmeans_path.exists()
        and metadata_path.exists()
    ):
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                old_metadata = json.load(f)

            current_fingerprint = dataframe_fingerprint(class_df)

            if (
                old_metadata.get("rows") == n
                and old_metadata.get("source_fingerprint") == current_fingerprint
                and old_metadata.get("n_clusters") == k
            ):
                print(
                    f"Saved artifacts for {class_name} already exist "
                    "and match the source data. Skipping rebuild."
                )
                return
        except Exception as exc:
            print("Existing artifact check failed; rebuilding:", exc)

    hidden_size = extract_embeddings(
        class_df[TEXT_COLUMN].tolist(),
        str(embedding_path),
        tokenizer,
        embedding_model,
    )

    embeddings = np.memmap(
        embedding_path,
        dtype="float16",
        mode="r",
        shape=(n, hidden_size),
    )

    kmeans = MiniBatchKMeans(
        n_clusters=k,
        random_state=SEED,
        batch_size=CLUSTER_BATCH_SIZE,
        n_init=3,
        max_iter=100,
        reassignment_ratio=0.01,
    )

    print("\nFitting MiniBatchKMeans...")
    for start in range(0, n, CLUSTER_BATCH_SIZE):
        end = min(start + CLUSTER_BATCH_SIZE, n)

        batch_embeddings = np.asarray(
            embeddings[start:end],
            dtype=np.float32,
        )

        kmeans.partial_fit(batch_embeddings)

        if end == n or end % 100000 < CLUSTER_BATCH_SIZE:
            print(f"KMeans progress: {end:,}/{n:,}")

    print("KMeans fitting completed.")

    # Assign a cluster ID to every row, in the exact source order.
    cluster_ids = np.empty(n, dtype=np.int32)

    print("\nPredicting cluster IDs...")
    for start in range(0, n, CLUSTER_BATCH_SIZE):
        end = min(start + CLUSTER_BATCH_SIZE, n)

        batch_embeddings = np.asarray(
            embeddings[start:end],
            dtype=np.float32,
        )

        cluster_ids[start:end] = kmeans.predict(batch_embeddings)

        if end == n or end % 100000 < CLUSTER_BATCH_SIZE:
            print(f"Prediction progress: {end:,}/{n:,}")

    cluster_counts = np.bincount(cluster_ids, minlength=k)

    # Persist cluster IDs and the fitted KMeans model.
    np.save(cluster_ids_path, cluster_ids)
    joblib.dump(kmeans, kmeans_path, compress=3)

    metadata = {
        "class_name": class_name,
        "rows": int(n),
        "n_clusters": int(k),
        "embedding_hidden_size": int(hidden_size),
        "model_name": MODEL_NAME,
        "max_length": int(MAX_LENGTH),
        "pooling": "attention-mask mean pooling",
        "normalization": "L2",
        "seed": int(SEED),
        "cluster_ids_file": "cluster_ids.npy",
        "kmeans_file": "kmeans.joblib",
        "source_fingerprint": dataframe_fingerprint(class_df),
    }

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    # Save cluster sizes separately for inspection.
    cluster_report = pd.DataFrame({
        "cluster_id": np.arange(k, dtype=np.int32),
        "cluster_size": cluster_counts,
    })
    cluster_report.to_csv(
        class_dir / "cluster_sizes.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\nSaved artifacts:")
    print(" -", cluster_ids_path)
    print(" -", kmeans_path)
    print(" -", metadata_path)
    print(" -", class_dir / "cluster_sizes.csv")
    print("Non-empty clusters:", int(np.count_nonzero(cluster_counts)))
    print("Largest cluster:", int(cluster_counts.max()))
    nonempty = cluster_counts[cluster_counts > 0]
    print("Smallest non-empty cluster:", int(nonempty.min()))

    # Clean up temporary embeddings only. Persistent artifacts remain.
    del embeddings
    del cluster_ids
    del cluster_counts
    del kmeans
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if embedding_path.exists():
        embedding_path.unlink()


# ------------------------------------------------------------
# 7. Allocate a target number of samples across clusters
# ------------------------------------------------------------

def allocate_cluster_samples(
    cluster_counts,
    target_samples,
    min_per_cluster=MIN_SAMPLES_PER_CLUSTER,
):
    cluster_counts = np.asarray(cluster_counts, dtype=np.int64)

    if target_samples > int(cluster_counts.sum()):
        raise ValueError(
            f"Target {target_samples:,} exceeds available "
            f"samples {int(cluster_counts.sum()):,}."
        )

    quotas = np.minimum(cluster_counts, min_per_cluster)
    remaining = int(target_samples - quotas.sum())

    if remaining < 0:
        raise ValueError("Minimum allocation exceeds target.")

    capacities = cluster_counts - quotas
    total_capacity = int(capacities.sum())

    if remaining > total_capacity:
        raise ValueError("Not enough cluster capacity.")

    if remaining == 0:
        return quotas

    if total_capacity == 0:
        raise ValueError("No remaining cluster capacity.")

    raw_allocation = remaining * capacities / total_capacity
    additional = np.floor(raw_allocation).astype(np.int64)
    quotas += additional

    leftover = int(target_samples - quotas.sum())
    fractions = raw_allocation - additional
    order = np.argsort(-fractions, kind="stable")

    for cluster_id in order:
        if leftover == 0:
            break
        if quotas[cluster_id] < cluster_counts[cluster_id]:
            quotas[cluster_id] += 1
            leftover -= 1

    if leftover != 0:
        raise RuntimeError("Could not distribute all remaining samples.")

    assert int(quotas.sum()) == int(target_samples)
    assert np.all(quotas <= cluster_counts)

    return quotas


# ------------------------------------------------------------
# 8. Re-sample from saved cluster IDs (no embeddings / no KMeans)
# ------------------------------------------------------------

def resample_from_saved_artifacts(data):
    sampled_classes = []
    all_reports = []

    for class_index, (class_name, target) in enumerate(TARGET_PER_CLASS.items()):
        print("\n" + "=" * 72)
        print("SAMPLING FROM SAVED CLUSTERS:", class_name)
        print("Target:", f"{target:,}")
        print("=" * 72)

        class_df = (
            data[data[LABEL_COLUMN] == class_name]
            .copy()
            .reset_index(drop=True)
        )

        class_dir = CLASS_ARTIFACT_DIR / class_name
        cluster_ids_path = class_dir / "cluster_ids.npy"
        metadata_path = class_dir / "metadata.json"

        if not cluster_ids_path.exists() or not metadata_path.exists():
            raise FileNotFoundError(
                f"Missing saved artifacts for {class_name}. "
                "Run ACTION='build' first."
            )

        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        current_fingerprint = dataframe_fingerprint(class_df)

        if len(class_df) != metadata["rows"]:
            raise ValueError(
                f"Row count changed for {class_name}: "
                f"saved={metadata['rows']:,}, current={len(class_df):,}. "
                "Do not reuse these cluster IDs with changed data."
            )

        if current_fingerprint != metadata["source_fingerprint"]:
            raise ValueError(
                f"Source data/order changed for {class_name}. "
                "The saved cluster IDs cannot safely be reused."
            )

        cluster_ids = np.load(cluster_ids_path, mmap_mode="r")

        if len(cluster_ids) != len(class_df):
            raise ValueError(
                f"Cluster ID count mismatch for {class_name}."
            )

        k = int(metadata["n_clusters"])
        cluster_counts = np.bincount(
            np.asarray(cluster_ids, dtype=np.int32),
            minlength=k,
        )

        quotas = allocate_cluster_samples(
            cluster_counts,
            target,
            MIN_SAMPLES_PER_CLUSTER,
        )

        rng = np.random.default_rng(SEED + class_index)
        selected_indices = []

        # Sort once by cluster ID instead of scanning the full array
        # separately for every cluster.
        sorted_indices = np.argsort(cluster_ids, kind="stable")
        cluster_boundaries = np.cumsum(cluster_counts)
        cluster_starts = np.concatenate(([0], cluster_boundaries[:-1]))

        for cluster_id in range(k):
            quota = int(quotas[cluster_id])
            if quota == 0:
                continue

            start = int(cluster_starts[cluster_id])
            end = int(cluster_boundaries[cluster_id])
            members = sorted_indices[start:end]

            chosen = rng.choice(members, size=quota, replace=False)
            selected_indices.extend(chosen.tolist())

        del sorted_indices

        selected_indices = np.asarray(selected_indices, dtype=np.int64)

        sampled_class = class_df.iloc[selected_indices].copy()
        sampled_class["cluster_id"] = np.asarray(
            cluster_ids[selected_indices],
            dtype=np.int32,
        )

        assert len(sampled_class) == target

        report = pd.DataFrame({
            "class": class_name,
            "cluster_id": np.arange(k, dtype=np.int32),
            "cluster_size": cluster_counts,
            "sample_quota": quotas,
            "selected_count": quotas,
        })

        sampled_classes.append(sampled_class)
        all_reports.append(report)

        report.to_csv(
            class_dir / "latest_sampling_report.csv",
            index=False,
            encoding="utf-8-sig",
        )

        print("Selected:", f"{len(sampled_class):,}")
        print("Non-empty clusters:", int(np.count_nonzero(cluster_counts)))

        del class_df, cluster_ids
        gc.collect()

    balanced_data = pd.concat(sampled_classes, ignore_index=True)

    balanced_data = balanced_data.sample(
        frac=1,
        random_state=SEED,
    ).reset_index(drop=True)

    combined_report = pd.concat(all_reports, ignore_index=True)

    balanced_data.to_csv(
        SAMPLE_OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    combined_report.to_csv(
        REPORT_OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n" + "=" * 72)
    print("SAMPLING COMPLETED")
    print("=" * 72)
    print("Total rows:", f"{len(balanced_data):,}")
    print("\nClass distribution:")
    print(balanced_data[LABEL_COLUMN].value_counts())
    print("\nSample saved:", SAMPLE_OUTPUT_PATH)
    print("Combined report saved:", REPORT_OUTPUT_PATH)
    print("\nThe cluster IDs and KMeans models remain saved in:")
    print(ARTIFACT_DIR.resolve())

    return balanced_data


# ------------------------------------------------------------
# 9. Main
# ------------------------------------------------------------

def main():
    data = load_source_data()

    if ACTION == "build":
        tokenizer, embedding_model = load_embedding_model()

        for class_name in TARGET_PER_CLASS:
            class_df = (
                data[data[LABEL_COLUMN] == class_name]
                .copy()
                .reset_index(drop=True)
            )

            build_class_artifacts(
                class_df=class_df,
                class_name=class_name,
                tokenizer=tokenizer,
                embedding_model=embedding_model,
            )

            del class_df
            gc.collect()

        del tokenizer, embedding_model
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("\nAll class clustering artifacts have been saved.")

    elif ACTION == "resample":
        print(
            "\nReusing saved cluster IDs. "
            "ParsBERT embeddings and KMeans will NOT be recomputed."
        )

    else:
        raise ValueError("ACTION must be 'build' or 'resample'.")

    # In both modes, produce a sample using the current targets.
    resample_from_saved_artifacts(data)


if __name__ == "__main__":
    main()
