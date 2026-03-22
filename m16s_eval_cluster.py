"""
Micro16S clustering evaluation configuration (configuration only).

Purpose
- Define a complete, implementation-ready clustering evaluation contract for the
  "Micro16S Evaluations" methods section in `ch4_m16s.md`.
- Keep all behavior controlled by constants in this file (no CLI flags).

Evaluation scope
- Backends: embedding checkpoints and k-mer baselines.
- Splits: train, test, excluded.
- Ranks: domain to genus (indices 0..5).
- Region analyses:
  - `mixed_random_per_sequence` (primary analysis),
  - `single_region_repeated` (region-invariance summary).
- Optional excluded+train mixed clustering with scoring restricted to excluded rows.

Implementation contract
1. Load split matrices and labels.
2. Build usable region pool:
   - start from all region indices,
   - remove index 0 (full-sequence representation),
   - if `REGION_IDS` is not None, keep only those IDs.
3. For each analysis run:
   - seed RNG with `GLOBAL_RANDOM_SEED + run_idx * INDEPENDENT_RUN_SEED_STRIDE`,
   - `mixed_random_per_sequence`: one random region per sequence,
   - `single_region_repeated`: one shared region per run, sampled without replacement;
     if requested runs exceed usable regions, raise an error.
4. For each split/rank/backend/model-or-k:
   - set `n_clusters` to the number of unique true taxa at that rank in the
     clustered dataset scope,
   - initialize cluster centers as the true-label centroids (mean vectors per taxon),
   - cluster selected vectors using `CLUSTER_DISTANCE_METRIC_BY_BACKEND`:
     - `"cosine"` uses custom NumPy spherical K-means,
     - `"euclidean"` and `"sqeuclidean"` use standard K-means,
   - report requested score(s) in `CLUSTERING_SCORE_NAMES`.
5. For `RUN_EXCLUDED_MIXED_WITH_TRAIN=True`, cluster on concatenated
   train+excluded vectors, set `n_clusters` from the same train+excluded scope,
   and score using `EXCLUDED_MIX_SCORING_SCOPE`.
6. For k-mer backends, choose reported K by `KMER_K_SELECTION_MODE` using
   `PRIMARY_SCORE_FOR_K_SELECTION`.

Expected outputs
- Root directory: `m16s_eval_results/cluster/`.
- Required files:
  - `scores_per_run.tsv`,
  - `scores_aggregated.tsv`,
  - `best_kmer_k.tsv` (when best-K search is used).
- Optional files:
  - `per_taxon_f1.tsv`,
  - `cluster_assignments.tsv`,
  - `contingency_matrices/*.csv`,
  - `skipped_cases.tsv`,
  - `config.json`,
  - `summary.txt`.

Notes
- No CLI arguments: all behavior must be controlled via constants in this file.
- Full sequences (index 0) are *always* excluded from this evaluation.
- Every sequence has every region (no missing-region handling is needed).
- `CLUSTER_DISTANCE_METRIC_BY_BACKEND` is the only backend clustering knob.
- `"cosine"` means custom NumPy spherical K-means.
- `"euclidean"` and `"sqeuclidean"` mean standard K-means.
- Region selectors accept either int indices (e.g., `4`) or region IDs from
  `region_indices.json` (e.g., `"V4-001"`), then normalize internally to indices.
- Export rows should always include both `region_idx` and `region_id`.
"""

from m16s_eval_utils import build_region_export_fields, load_region_index_mappings, normalize_region_selection, resolve_region_indices_json_path

# Required paths ---------------------------------------------------------------
# Dataset split used for all clustering evals unless overridden below.
DATASET_SPLIT_DIR = "/mnt/secondary/micro16s_dbs/16s_databases/m16s_db_004/seqs/split_001/"

# Auto-resolved from DATASET_SPLIT_DIR using simple string split/concatenation.
REGION_INDICES_JSON_PATH = resolve_region_indices_json_path(DATASET_SPLIT_DIR)

# region_indices.json mappings used for ID/index conversion.
REGION_IDX_TO_ID_MAPPING, REGION_ID_TO_IDX_MAPPING = load_region_index_mappings(REGION_INDICES_JSON_PATH)

# Default embedding checkpoint (application model by default).
MODEL_CHECKPOINT = "/home/haig/Repos/micro16s/models/m16s_002/ckpts/m16s_002_16000_batches.pth" # Application model

# Optional second checkpoint for validation-model clustering comparisons.
# Set to a real path when available; keep None to skip validation-model runs.
VALIDATION_MODEL_CHECKPOINT = "/home/haig/Repos/micro16s/models/m16s_001/ckpts/m16s_001_16000_batches.pth" # Validation model


# Output configuration ---------------------------------------------------------
# Root directory for all clustering outputs.
OUTPUT_ROOT_DIR = "m16s_eval_results/cluster"

# Prefix used when building run directory names.
RUN_DIR_PREFIX = "cluster"

# Timestamp format appended into run names.
RUN_DIR_TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"

# Numeric suffix width for uniqueness; e.g., "__001", "__002".
RUN_DIR_SUFFIX_DIGITS = 3

# Optional extra user tags appended to run name (short strings only).
RUN_NAME_EXTRA_TAGS = ()


# Runtime / reproducibility ----------------------------------------------------
# Global seed for numpy/torch/random in the future implementation.
GLOBAL_RANDOM_SEED = 24

# Seed offset between independent analysis runs.
INDEPENDENT_RUN_SEED_STRIDE = 1_000

# Use CUDA if available for embedding inference.
USE_CUDA_IF_AVAILABLE = True

# Embedding inference batch size (adjust by GPU memory).
INFERENCE_BATCH_SIZE = 1024

# CPU workers/threads for heavy preprocessing steps.
NUM_WORKERS = 12


# Representation backends ------------------------------------------------------
# Backends to run in this script.
# Allowed values: "embedding", "kmer".
REPRESENTATION_BACKENDS = (
    "embedding", 
    "kmer"
    )

# Named embedding model variants to evaluate.
EMBEDDING_MODEL_VARIANTS = {
    "application": MODEL_CHECKPOINT,
    "validation": VALIDATION_MODEL_CHECKPOINT,
}

# Candidate K values for k-mer baselines.
KMER_K_VALUES = (5, 6, 7)

# How to choose reported k-mer K from KMER_K_VALUES.
# "best_per_analysis": choose best K separately per split/rank/region-analysis.
# "fixed": use FORCE_KMER_K only.
KMER_K_SELECTION_MODE = "best_per_analysis"

# Set when KMER_K_SELECTION_MODE == "fixed".
FORCE_KMER_K = None

# Metric used to choose best K when KMER_K_SELECTION_MODE == "best_per_analysis".
PRIMARY_SCORE_FOR_K_SELECTION = "v_measure"


# Split / rank coverage --------------------------------------------------------
# Splits to evaluate.
EVAL_SPLITS = (
    "train",
    "test",
    "excluded",
)

# Taxonomic rank indices to evaluate.
# 0=domain, 1=phylum, 2=class, 3=order, 4=family, 5=genus.
CLUSTER_RANKS = (0, 1, 2, 3, 4, 5)

# Human-readable rank labels aligned to CLUSTER_RANKS.
CLUSTER_RANK_NAMES = ("domain", "phylum", "class", "order", "family", "genus")


# Excluded-set mixed clustering ------------------------------------------------
# Run clustering on excluded partition alone.
RUN_EXCLUDED_ALONE = True

# Run clustering on (train + excluded) combined representations.
RUN_EXCLUDED_MIXED_WITH_TRAIN = True

# When mixed-with-train is enabled, score only excluded sequences
# Allowed values: "excluded_only", "all".
EXCLUDED_MIX_SCORING_SCOPE = "excluded_only"


# Region handling --------------------------------------------------------------
# Optional explicit region selectors for both region analyses.
# Accepts int indices and/or region ID strings (e.g., "V4-001").
# None -> all usable subsequence regions.
REGION_IDS = None

# Normalized internal representation for filtering (always indices).
REGION_INDICES = normalize_region_selection(
    REGION_IDS,
    REGION_IDX_TO_ID_MAPPING,
    REGION_ID_TO_IDX_MAPPING,
    "REGION_IDS",
    REGION_INDICES_JSON_PATH,
)

# Region analyses to run:
# - "mixed_random_per_sequence": per sequence, pick a random region in pool.
# - "single_region_repeated": pick one region for all sequences per run, repeat N times (sampled without replacement).
REGION_ANALYSES = (
    "mixed_random_per_sequence",
    "single_region_repeated",
)

# Number of repeated runs for mixed-random analysis (usually 1 for primary table).
MIXED_RANDOM_NUM_RUNS = 30

# Number of repeated single-region runs for region-invariance summary.
SINGLE_REGION_NUM_RUNS = 29  # Sampled without replacement therefore must be <= number of regions

# If SINGLE_REGION_NUM_RUNS exceeds usable regions, raise an error.
ERROR_IF_SINGLE_REGION_RUNS_EXCEED_POOL = True


# Clustering metric / algorithm options ---------------------------------------
# Clustering quality metrics for reporting.
# Options:
# - "v_measure" (default): harmonic mean of homogeneity and completeness.
# - "adjusted_rand_index": chance-adjusted agreement with true taxon labels.
# - "adjusted_mutual_info": chance-adjusted mutual information with true labels.
CLUSTERING_SCORE_NAMES = (
    "v_measure", 
    "adjusted_rand_index", 
    "adjusted_mutual_info"
    )

# Distance metric per backend.
# Allowed values: "cosine", "euclidean", "sqeuclidean".
# Metric drives the clustering algorithm:
# - "cosine" -> custom NumPy spherical K-means.
# - "euclidean" or "sqeuclidean" -> standard K-means.
CLUSTER_DISTANCE_METRIC_BY_BACKEND = {
    "embedding": "cosine",
    "kmer": "sqeuclidean",
}

# If True, skip a split/rank/run combo when fewer than 2 unique labels are present.
SKIP_IF_LT_TWO_TAXA = True

# K-means controls (mirrors quick_test defaults).
KMEANS_MAX_ITER = 200
KMEANS_TOL = 5e-4
KMEANS_N_INIT = 1

# Initialization mode is fixed for this evaluation contract:
# - "label_centroids": seed centers from true-label means.
KMEANS_INIT_MODE = "label_centroids"


# Optional detailed exports ----------------------------------------------------
# Save per-taxon dominant-cluster precision/recall/F1 diagnostics.
SAVE_PER_TAXON_F1 = True

# Maximum taxa per rank to keep in per-taxon exports (largest taxa first).
MAX_TAXA_PER_RANK_DIAGNOSTIC = 200

# Save contingency matrices for selected rank/split combinations.
SAVE_CONTINGENCY_MATRICES = True

# Save predicted cluster assignments per sequence.
SAVE_CLUSTER_ASSIGNMENTS = True

# Save rows for skipped analyses (e.g., insufficient taxa in a rank).
SAVE_SKIPPED_CASES_TABLE = True

# Save run config snapshot JSON in each run directory.
SAVE_CONFIG_JSON = True

# Save per-run and aggregated TXT summaries.
SAVE_SUMMARY_TXT = True


# Implementation note ----------------------------------------------------------
# Add implementation below this config block (loading, inference, clustering, saving).
# Keep behavior fully driven by constants above; do not add CLI args.

import csv
import gc
import json
import os
import pickle
import random
import traceback
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, v_measure_score

from model import load_micro16s_model, run_inference


DEFAULT_RANK_NAMES = ("domain", "phylum", "class", "order", "family", "genus", "species")
SUPPORTED_BACKENDS = {"embedding", "kmer"}
SUPPORTED_REGION_ANALYSES = {"mixed_random_per_sequence", "single_region_repeated"}
SUPPORTED_KMER_SELECTION_MODES = {"best_per_analysis", "fixed"}
SUPPORTED_SCORE_NAMES = {"v_measure", "adjusted_rand_index", "adjusted_mutual_info"}
SUPPORTED_CLUSTER_DISTANCE_METRICS = {"cosine", "euclidean", "sqeuclidean"}
SPLIT_NAME_TO_FILESYSTEM_NAME = {"train": "train", "test": "test", "excluded": "excluded"}
SPLIT_NAME_TO_INDEX_FILENAME = {"train": "training_indices.txt", "test": "testing_indices.txt", "excluded": "excluded_taxa_indices.txt"}


def _set_seed(seed):
    """Seed all random generators used by this script."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _validate_config():
    """Validate config values early so failures are explicit."""
    if not REPRESENTATION_BACKENDS:
        raise ValueError("REPRESENTATION_BACKENDS must not be empty.")
    invalid_backends = [backend for backend in REPRESENTATION_BACKENDS if backend not in SUPPORTED_BACKENDS]
    if invalid_backends:
        raise ValueError(f"Unknown backends in REPRESENTATION_BACKENDS: {invalid_backends}.")

    invalid_region_analyses = [name for name in REGION_ANALYSES if name not in SUPPORTED_REGION_ANALYSES]
    if invalid_region_analyses:
        raise ValueError(f"Unknown region analyses in REGION_ANALYSES: {invalid_region_analyses}.")

    invalid_scores = [name for name in CLUSTERING_SCORE_NAMES if name not in SUPPORTED_SCORE_NAMES]
    if invalid_scores:
        raise ValueError(f"Unknown score names in CLUSTERING_SCORE_NAMES: {invalid_scores}.")

    if KMER_K_SELECTION_MODE not in SUPPORTED_KMER_SELECTION_MODES:
        raise ValueError(f"KMER_K_SELECTION_MODE must be one of {sorted(SUPPORTED_KMER_SELECTION_MODES)}.")
    if KMER_K_SELECTION_MODE == "fixed" and FORCE_KMER_K is None:
        raise ValueError("FORCE_KMER_K must be set when KMER_K_SELECTION_MODE == 'fixed'.")

    for backend in REPRESENTATION_BACKENDS:
        metric = CLUSTER_DISTANCE_METRIC_BY_BACKEND.get(backend)
        if metric not in SUPPORTED_CLUSTER_DISTANCE_METRICS:
            raise ValueError(f"CLUSTER_DISTANCE_METRIC_BY_BACKEND['{backend}'] must be one of {sorted(SUPPORTED_CLUSTER_DISTANCE_METRICS)}.")

    if len(CLUSTER_RANKS) != len(CLUSTER_RANK_NAMES):
        raise ValueError("CLUSTER_RANKS and CLUSTER_RANK_NAMES must have the same length.")

    invalid_splits = [split for split in EVAL_SPLITS if split not in SPLIT_NAME_TO_FILESYSTEM_NAME]
    if invalid_splits:
        raise ValueError(f"Unknown split names in EVAL_SPLITS: {invalid_splits}.")

    if EXCLUDED_MIX_SCORING_SCOPE not in ("excluded_only", "all"):
        raise ValueError("EXCLUDED_MIX_SCORING_SCOPE must be 'excluded_only' or 'all'.")

    if "kmer" in REPRESENTATION_BACKENDS:
        if KMER_K_SELECTION_MODE == "fixed" and FORCE_KMER_K not in KMER_K_VALUES:
            raise ValueError(f"FORCE_KMER_K={FORCE_KMER_K} is not in KMER_K_VALUES={KMER_K_VALUES}.")
        if PRIMARY_SCORE_FOR_K_SELECTION not in CLUSTERING_SCORE_NAMES:
            raise ValueError(f"PRIMARY_SCORE_FOR_K_SELECTION='{PRIMARY_SCORE_FOR_K_SELECTION}' must be present in CLUSTERING_SCORE_NAMES.")
    elif PRIMARY_SCORE_FOR_K_SELECTION not in CLUSTERING_SCORE_NAMES:
        raise ValueError(f"PRIMARY_SCORE_FOR_K_SELECTION='{PRIMARY_SCORE_FOR_K_SELECTION}' must be present in CLUSTERING_SCORE_NAMES.")


def _ensure_dir(path):
    """Create a directory if it does not already exist."""
    os.makedirs(path, exist_ok=True)
    return path


def _load_indices_file(path):
    """Load newline-delimited indices; allow empty files."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Index file not found: {path}")
    if os.path.getsize(path) == 0:
        return np.empty(0, dtype=np.int64)
    values = np.loadtxt(path, dtype=np.int64)
    return np.atleast_1d(values).astype(np.int64, copy=False)


def _resolve_encoded_dir(dataset_split_dir):
    """Resolve .../seqs/encoded/ from .../seqs/split_xxx/."""
    split_marker = "/split_"
    normalized = dataset_split_dir.rstrip("/")
    if split_marker not in normalized:
        raise ValueError(f"DATASET_SPLIT_DIR must contain '{split_marker}'. Got: {dataset_split_dir}")
    parent = normalized.split(split_marker)[0]
    return parent + "/encoded"


def _load_3bit_reps(encoded_dir):
    """Load 3-bit representations, supporting packed and unpacked files."""
    unpacked_path = os.path.join(encoded_dir, "3bit_seq_reps.npy")
    packed_path = os.path.join(encoded_dir, "3bit_seq_reps_packed.npy")
    if os.path.exists(unpacked_path):
        return np.load(unpacked_path, mmap_mode="r")
    if os.path.exists(packed_path):
        packed = np.load(packed_path, mmap_mode="r")
        unpacked = np.unpackbits(packed, axis=-1, bitorder="big")[..., :3]
        return unpacked.astype(bool, copy=False)
    raise FileNotFoundError(f"Could not find 3-bit representations at '{unpacked_path}' or '{packed_path}'.")


def _load_kmer_reps(encoded_dir, k_value):
    """Load one k-mer representation array."""
    path = os.path.join(encoded_dir, f"{k_value}-mer_seq_reps.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required k-mer file is missing: {path}")
    return np.load(path, mmap_mode="r")


def _safe_full_tax_label_lookup(label_dict, seq_id):
    """Resolve taxonomy labels from pickled dicts that may use int or str keys."""
    if seq_id in label_dict:
        return label_dict[seq_id]
    seq_id_int = int(seq_id)
    if seq_id_int in label_dict:
        return label_dict[seq_id_int]
    seq_id_str = str(seq_id_int)
    if seq_id_str in label_dict:
        return label_dict[seq_id_str]
    raise KeyError(f"Sequence ID {seq_id_int} was not found in full_tax_label_from_seq_id_dict.")


def _load_split_labels(dataset_split_dir, split_name, indices):
    """Load per-sequence full taxonomy labels aligned to split indices."""
    path = os.path.join(dataset_split_dir, "tax_objs", split_name, "full_tax_label_from_seq_id_dict.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing taxonomy labels file: {path}")
    with open(path, "rb") as handle:
        label_dict = pickle.load(handle)
    labels = [_safe_full_tax_label_lookup(label_dict, seq_id) for seq_id in indices]
    return labels


def _build_lazy_split_source(all_region_major_reps, split_indices):
    """Store one split as (full_array, split_indices) to avoid large eager copies."""
    return all_region_major_reps, np.asarray(split_indices, dtype=np.int64)


def _to_json_safe(obj):
    """Convert values recursively to JSON-safe primitives."""
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, (tuple, list)):
        return [_to_json_safe(v) for v in obj]
    if isinstance(obj, dict):
        return {str(k): _to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, set):
        return sorted(_to_json_safe(v) for v in obj)
    return str(obj)


def _collect_config_snapshot():
    """Capture all uppercase config constants for export."""
    snapshot = {}
    for key, value in globals().items():
        if key.isupper():
            snapshot[key] = _to_json_safe(value)
    return snapshot


def _build_run_dir():
    """Create a unique run directory under OUTPUT_ROOT_DIR."""
    _ensure_dir(OUTPUT_ROOT_DIR)
    timestamp = datetime.now().strftime(RUN_DIR_TIMESTAMP_FORMAT)
    tag_part = ""
    if RUN_NAME_EXTRA_TAGS:
        tag_part = "__" + "__".join(str(tag) for tag in RUN_NAME_EXTRA_TAGS)
    for suffix_idx in range(1, 10 ** RUN_DIR_SUFFIX_DIGITS):
        suffix = f"__{suffix_idx:0{RUN_DIR_SUFFIX_DIGITS}d}"
        run_name = f"{RUN_DIR_PREFIX}_{timestamp}{tag_part}{suffix}"
        run_dir = os.path.join(OUTPUT_ROOT_DIR, run_name)
        if not os.path.exists(run_dir):
            os.makedirs(run_dir, exist_ok=False)
            return run_dir
    raise RuntimeError(f"Could not allocate a unique run dir after {10 ** RUN_DIR_SUFFIX_DIGITS - 1} attempts.")


def _rank_name_for(rank):
    """Resolve a rank label from configured names with a safe fallback."""
    rank_to_name = {int(rank_idx): str(name) for rank_idx, name in zip(CLUSTER_RANKS, CLUSTER_RANK_NAMES)}
    if int(rank) in rank_to_name:
        return rank_to_name[int(rank)]
    if 0 <= int(rank) < len(DEFAULT_RANK_NAMES):
        return DEFAULT_RANK_NAMES[int(rank)]
    return f"rank_{int(rank)}"


def _extract_rank_labels(full_tax_labels, rank):
    """Extract one rank of labels from full taxonomy rows."""
    labels = []
    rank_int = int(rank)
    for row in full_tax_labels:
        value = "__unknown__"
        if isinstance(row, (list, tuple)) and rank_int < len(row):
            value = row[rank_int]
        if value is None or value == "":
            value = "__unknown__"
        labels.append(str(value))
    return np.asarray(labels, dtype=object)


def _l2_normalize_rows(array, eps=1e-8):
    """Row-wise L2 normalization used by spherical k-means."""
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    return array / np.clip(norms, eps, None)


def _init_centroids_from_labels(representations, labels):
    """Initialize one centroid per true label by averaging member vectors."""
    unique_labels, inverse, counts = np.unique(labels, return_inverse=True, return_counts=True)
    centroids = np.zeros((len(unique_labels), representations.shape[1]), dtype=np.float32)
    np.add.at(centroids, inverse, representations)
    centroids /= counts[:, None].astype(np.float32)
    return unique_labels, inverse, counts, centroids


def _run_spherical_kmeans(representations, init_centroids):
    """Run spherical k-means with deterministic label-centroid initialization."""
    reps_norm = _l2_normalize_rows(representations)
    centroids = _l2_normalize_rows(init_centroids)
    n_samples = reps_norm.shape[0]
    n_clusters = centroids.shape[0]
    predicted = np.full(n_samples, -1, dtype=np.int32)
    for _ in range(int(KMEANS_MAX_ITER)):
        similarities = reps_norm @ centroids.T
        new_predicted = np.argmax(similarities, axis=1).astype(np.int32)
        if np.array_equal(predicted, new_predicted):
            break
        predicted = new_predicted

        new_centroids = np.zeros_like(centroids)
        cluster_sizes = np.bincount(predicted, minlength=n_clusters)
        np.add.at(new_centroids, predicted, reps_norm)

        non_empty = cluster_sizes > 0
        if np.any(non_empty):
            new_centroids[non_empty] /= cluster_sizes[non_empty, None].astype(np.float32)
            new_centroids[non_empty] = _l2_normalize_rows(new_centroids[non_empty])
        if np.any(~non_empty):
            new_centroids[~non_empty] = centroids[~non_empty]

        centroid_shift = np.linalg.norm(new_centroids - centroids, axis=1).mean()
        centroids = new_centroids
        if centroid_shift <= float(KMEANS_TOL):
            break
    return predicted


def _run_euclidean_kmeans(representations, init_centroids):
    """Run standard Euclidean k-means with deterministic centroid initialization."""
    n_init_value = int(KMEANS_N_INIT)
    if KMEANS_INIT_MODE == "label_centroids":
        n_init_value = 1
    kmeans = KMeans(n_clusters=init_centroids.shape[0], init=init_centroids, n_init=n_init_value, max_iter=int(KMEANS_MAX_ITER), tol=float(KMEANS_TOL), random_state=int(GLOBAL_RANDOM_SEED), algorithm="lloyd")
    return kmeans.fit_predict(representations).astype(np.int32)


def _cluster_representations(representations, labels, distance_metric):
    """Cluster vectors according to configured metric."""
    reps = np.asarray(representations, dtype=np.float32)
    unique_labels, _, _, init_centroids = _init_centroids_from_labels(reps, labels)
    if distance_metric == "cosine":
        clusters = _run_spherical_kmeans(reps, init_centroids)
    elif distance_metric in ("euclidean", "sqeuclidean"):
        clusters = _run_euclidean_kmeans(reps, init_centroids.astype(np.float32, copy=False))
    else:
        raise ValueError(f"Unsupported clustering metric: {distance_metric}")
    return clusters, unique_labels


def _compute_requested_scores(true_labels, predicted_clusters):
    """Compute all requested clustering scores."""
    values = {}
    for score_name in CLUSTERING_SCORE_NAMES:
        if score_name == "v_measure":
            values[score_name] = float(v_measure_score(true_labels, predicted_clusters))
        elif score_name == "adjusted_rand_index":
            values[score_name] = float(adjusted_rand_score(true_labels, predicted_clusters))
        elif score_name == "adjusted_mutual_info":
            values[score_name] = float(adjusted_mutual_info_score(true_labels, predicted_clusters))
        else:
            raise ValueError(f"Unsupported score name: {score_name}")
    return values


def _compute_per_taxon_f1_rows(true_labels, predicted_clusters, max_taxa):
    """Compute dominant-cluster precision/recall/F1 per taxon."""
    unique_labels, counts = np.unique(true_labels, return_counts=True)
    cluster_sizes = np.bincount(predicted_clusters)
    sorted_indices = np.argsort(-counts)
    if max_taxa is not None:
        sorted_indices = sorted_indices[:int(max_taxa)]

    rows = []
    for idx in sorted_indices:
        taxon_label = unique_labels[idx]
        taxon_count = int(counts[idx])
        member_clusters = predicted_clusters[true_labels == taxon_label]
        if member_clusters.size == 0:
            continue
        bincount = np.bincount(member_clusters, minlength=max(len(cluster_sizes), int(member_clusters.max()) + 1))
        dominant_cluster = int(np.argmax(bincount))
        true_positive = float(bincount[dominant_cluster])
        recall = true_positive / max(float(taxon_count), 1.0)
        precision = true_positive / max(float(cluster_sizes[dominant_cluster]), 1.0)
        f1 = 2.0 * precision * recall / max(precision + recall, 1e-9)
        rows.append({"taxon_label": str(taxon_label), "taxon_count": taxon_count, "dominant_cluster": dominant_cluster, "precision": float(precision), "recall": float(recall), "f1": float(f1)})
    return rows


def _build_contingency_dataframe(true_labels, predicted_clusters, max_taxa):
    """Build a sorted contingency matrix dataframe."""
    unique_labels, label_counts = np.unique(true_labels, return_counts=True)
    if unique_labels.size == 0:
        return None
    order = np.argsort(-label_counts)
    if max_taxa is not None:
        order = order[:int(max_taxa)]
    selected_labels = unique_labels[order]
    if selected_labels.size == 0:
        return None

    label_to_col = {label: idx for idx, label in enumerate(selected_labels)}
    unique_clusters = np.unique(predicted_clusters)
    matrix = np.zeros((len(unique_clusters), len(selected_labels)), dtype=np.int64)
    for row_idx, cluster_id in enumerate(unique_clusters):
        member_labels = true_labels[predicted_clusters == cluster_id]
        for label in member_labels:
            col_idx = label_to_col.get(label)
            if col_idx is not None:
                matrix[row_idx, col_idx] += 1

    non_empty_rows = matrix.sum(axis=1) > 0
    if not np.any(non_empty_rows):
        return None
    matrix = matrix[non_empty_rows]

    dominant_cols = np.argmax(matrix, axis=1)
    dominant_counts = np.array([matrix[row_idx, dominant_cols[row_idx]] for row_idx in range(matrix.shape[0])], dtype=np.int64)
    row_order = np.lexsort((-dominant_counts, dominant_cols))
    matrix = matrix[row_order]

    row_labels = [f"cluster_{row_idx}" for row_idx in range(matrix.shape[0])]
    col_labels = [str(label) for label in selected_labels]
    return pd.DataFrame(matrix, index=row_labels, columns=col_labels)


def _sanitize_slug(value):
    """Create a conservative filesystem-safe slug."""
    text = str(value)
    cleaned = []
    for char in text:
        if char.isalnum() or char in ("_", "-", "."):
            cleaned.append(char)
        else:
            cleaned.append("_")
    return "".join(cleaned).strip("_")


def _build_usable_region_pool(n_regions):
    """Apply region pool rules from the contract."""
    region_pool = [idx for idx in range(int(n_regions)) if idx != 0]
    if REGION_INDICES is not None:
        allowed = set(int(idx) for idx in REGION_INDICES)
        region_pool = [idx for idx in region_pool if idx in allowed]
    missing_in_mapping = [idx for idx in region_pool if idx not in REGION_IDX_TO_ID_MAPPING]
    if missing_in_mapping:
        raise ValueError(f"Usable region indices are missing from REGION_IDX_TO_ID_MAPPING: {missing_in_mapping}")
    if not region_pool:
        raise ValueError("No usable subsequence regions remain after filtering. Index 0 is always excluded.")
    return tuple(region_pool)


def _build_region_run_plan(analysis_name, n_sequences, usable_regions):
    """Build deterministic region choices for one analysis."""
    if analysis_name == "mixed_random_per_sequence":
        num_runs = int(MIXED_RANDOM_NUM_RUNS)
    elif analysis_name == "single_region_repeated":
        num_runs = int(SINGLE_REGION_NUM_RUNS)
    else:
        raise ValueError(f"Unknown region analysis: {analysis_name}")

    if num_runs <= 0:
        return []

    region_pool = list(usable_regions)
    if analysis_name == "single_region_repeated" and num_runs > len(region_pool):
        message = f"SINGLE_REGION_NUM_RUNS={num_runs} exceeds usable region pool size={len(region_pool)}."
        if ERROR_IF_SINGLE_REGION_RUNS_EXCEED_POOL:
            raise ValueError(message)
        num_runs = len(region_pool)
        print(f"Warning: {message} Clipping runs to {num_runs}.")

    run_plan = []
    remaining_regions = list(region_pool)
    for run_idx in range(num_runs):
        run_seed = int(GLOBAL_RANDOM_SEED + run_idx * INDEPENDENT_RUN_SEED_STRIDE)
        rng = np.random.default_rng(run_seed)
        if analysis_name == "mixed_random_per_sequence":
            per_seq_region_idx = rng.choice(region_pool, size=int(n_sequences), replace=True).astype(np.int32, copy=False)
            shared_region_idx = None
        else:
            if not remaining_regions:
                break
            choice_idx = int(rng.integers(0, len(remaining_regions)))
            shared_region_idx = int(remaining_regions.pop(choice_idx))
            per_seq_region_idx = np.full(int(n_sequences), shared_region_idx, dtype=np.int32)
        run_plan.append({"run_idx": int(run_idx), "run_seed": run_seed, "analysis_name": analysis_name, "shared_region_idx": shared_region_idx, "per_seq_region_idx": per_seq_region_idx})
    return run_plan


def _build_eval_cases():
    """Build evaluation scopes from split config."""
    cases = []
    for split_name in EVAL_SPLITS:
        if split_name == "excluded" and not RUN_EXCLUDED_ALONE:
            continue
        cases.append({"split_name": split_name, "eval_scope": split_name, "source_splits": (split_name,), "scoring_scope": "all"})
    if RUN_EXCLUDED_MIXED_WITH_TRAIN:
        cases.append({"split_name": "excluded", "eval_scope": "excluded_mixed_with_train", "source_splits": ("train", "excluded"), "scoring_scope": EXCLUDED_MIX_SCORING_SCOPE})
    return cases


def _concat_case_labels(case, labels_by_split):
    """Concatenate labels for one evaluation case."""
    all_labels = []
    for split_name in case["source_splits"]:
        all_labels.extend(labels_by_split[split_name])
    return all_labels


def _concat_case_origins(case, counts_by_split):
    """Build per-sequence origin labels for one evaluation case."""
    origins = []
    for split_name in case["source_splits"]:
        origins.extend([split_name] * int(counts_by_split[split_name]))
    return np.asarray(origins, dtype=object)


def _select_vectors_from_case_arrays(region_major_arrays, case, per_seq_region_idx):
    """Select one vector per sequence according to region choices."""
    selected_blocks = []
    selected_region_blocks = []
    cursor = 0
    for split_name in case["source_splits"]:
        block = region_major_arrays[split_name]
        split_indices = None
        if isinstance(block, tuple):
            block, split_indices = block
            n_seq = int(len(split_indices))
        else:
            n_seq = int(block.shape[1])
        split_region_idx = per_seq_region_idx[cursor:cursor + n_seq]
        if split_indices is None:
            split_selected = block[split_region_idx, np.arange(n_seq), :]
        else:
            split_selected = block[split_region_idx, split_indices, :]
        selected_blocks.append(np.asarray(split_selected, dtype=np.float32))
        selected_region_blocks.append(np.asarray(split_region_idx, dtype=np.int32))
        cursor += n_seq
    all_vectors = np.concatenate(selected_blocks, axis=0)
    all_region_idx = np.concatenate(selected_region_blocks, axis=0)
    return all_vectors, all_region_idx


def _adapt_seq_len_for_model(seq_reps_3bit, target_seq_len):
    """Match input sequence length to a model's expected max_seq_len."""
    current_len = int(seq_reps_3bit.shape[-2])
    target_len = int(target_seq_len)
    if current_len == target_len:
        return seq_reps_3bit
    if current_len > target_len:
        return seq_reps_3bit[..., :target_len, :]
    pad_len = target_len - current_len
    pad_shape = tuple(seq_reps_3bit.shape[:-2]) + (pad_len, 3)
    padding = np.zeros(pad_shape, dtype=seq_reps_3bit.dtype)
    padding[..., 0] = 1
    return np.concatenate([seq_reps_3bit, padding], axis=-2)


def _infer_embeddings_for_variant(model_name, checkpoint_path, all_3bit_reps, split_indices):
    """Run embedding inference once per split for one checkpoint."""
    if checkpoint_path is None:
        return None
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Embedding checkpoint for variant '{model_name}' does not exist: {checkpoint_path}")

    print(f"Loading embedding model '{model_name}' from: {checkpoint_path}")
    model = load_micro16s_model(checkpoint_path)
    use_cuda = bool(USE_CUDA_IF_AVAILABLE and torch.cuda.is_available())
    device = torch.device("cuda" if use_cuda else "cpu")
    model = model.to(device)
    model.eval()

    outputs = {}
    for split_name, indices in split_indices.items():
        seq_reps_3bit = all_3bit_reps[:, np.asarray(indices, dtype=np.int64), :, :]
        adapted = _adapt_seq_len_for_model(seq_reps_3bit, model.max_seq_len)
        print(f"Running inference: model={model_name}, split={split_name}, shape={adapted.shape}")
        outputs[split_name] = run_inference(model, adapted, device=device, batch_size=int(INFERENCE_BATCH_SIZE), output_device="cpu", return_numpy=True, pin_inputs=use_cuda)

    model = model.cpu()
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return outputs


def _kmer_selection_group_key(row):
    """Grouping key for selecting best k separately per analysis context."""
    return (row["eval_scope"], row["split_name"], row["scoring_scope"], row["region_analysis"], int(row["rank_idx"]), row["rank_name"], row["distance_metric"])


def _select_best_kmer_per_group(run_rows):
    """Select best k per configured grouping using PRIMARY_SCORE_FOR_K_SELECTION."""
    kmer_rows = [row for row in run_rows if row.get("backend") == "kmer"]
    if not kmer_rows:
        return {}, []

    grouped = {}
    for row in kmer_rows:
        key = _kmer_selection_group_key(row)
        grouped.setdefault(key, {}).setdefault(int(row["kmer_k"]), []).append(row)

    selected_map = {}
    summary_rows = []
    for key, rows_per_k in grouped.items():
        best_k = None
        best_mean = None
        for k_value, rows_for_k in rows_per_k.items():
            scores = [float(row[PRIMARY_SCORE_FOR_K_SELECTION]) for row in rows_for_k if row.get(PRIMARY_SCORE_FOR_K_SELECTION) is not None and not np.isnan(row.get(PRIMARY_SCORE_FOR_K_SELECTION))]
            if not scores:
                continue
            score_mean = float(np.mean(scores))
            if best_mean is None or score_mean > best_mean or (score_mean == best_mean and int(k_value) < int(best_k)):
                best_k = int(k_value)
                best_mean = score_mean
        if best_k is None:
            continue
        selected_map[key] = best_k
        summary_rows.append({"eval_scope": key[0], "split_name": key[1], "scoring_scope": key[2], "region_analysis": key[3], "rank_idx": key[4], "rank_name": key[5], "distance_metric": key[6], "selected_k": best_k, f"mean_{PRIMARY_SCORE_FOR_K_SELECTION}": best_mean})
    return selected_map, summary_rows


def _filter_kmer_rows_by_selected_k(rows, selected_k_map):
    """Keep only selected k rows for k-mer backend."""
    filtered = []
    for row in rows:
        if row.get("backend") != "kmer":
            filtered.append(row)
            continue
        if KMER_K_SELECTION_MODE == "fixed":
            if int(row.get("kmer_k")) == int(FORCE_KMER_K):
                filtered.append(row)
            continue
        key = _kmer_selection_group_key(row)
        wanted_k = selected_k_map.get(key)
        if wanted_k is not None and int(row.get("kmer_k")) == int(wanted_k):
            filtered.append(row)
    return filtered


def _aggregate_run_scores(score_rows):
    """Aggregate per-run scores into mean/std tables."""
    if not score_rows:
        return []
    df = pd.DataFrame(score_rows)
    group_cols = ["backend", "model_variant", "kmer_k", "distance_metric", "split_name", "eval_scope", "scoring_scope", "region_analysis", "rank_idx", "rank_name"]
    aggregate_rows = []
    grouped = df.groupby(group_cols, dropna=False, sort=False)
    for key, group in grouped:
        row = {col: key[idx] for idx, col in enumerate(group_cols)}
        row["num_runs"] = int(group["run_idx"].nunique())
        row["n_sequences_clustered"] = int(group["n_sequences_clustered"].iloc[0])
        row["n_sequences_scored"] = int(group["n_sequences_scored"].iloc[0])
        row["n_clusters"] = int(group["n_clusters"].iloc[0])
        for score_name in CLUSTERING_SCORE_NAMES:
            values = group[score_name].astype(float).to_numpy()
            row[f"{score_name}_mean"] = float(np.mean(values))
            row[f"{score_name}_std"] = float(np.std(values, ddof=0))
        aggregate_rows.append(row)
    return aggregate_rows


def _write_tsv(path, rows):
    """Write a list of dict rows to TSV."""
    df = pd.DataFrame(rows)
    df.to_csv(path, sep="\t", index=False)


def _open_assignment_writer_if_needed(run_dir):
    """Create a streaming writer for assignment rows when enabled."""
    if not SAVE_CLUSTER_ASSIGNMENTS:
        return None, None
    path = os.path.join(run_dir, "cluster_assignments.tsv")
    handle = open(path, "w", newline="")
    fieldnames = ["backend", "model_variant", "kmer_k", "distance_metric", "split_name", "eval_scope", "scoring_scope", "region_analysis", "run_idx", "run_seed", "rank_idx", "rank_name", "sequence_idx_within_case", "origin_split", "is_scored", "true_label", "predicted_cluster", "region_idx", "region_id"]
    writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
    writer.writeheader()
    return handle, writer


def _write_summary_txt(path, run_dir, score_rows, aggregated_rows, skipped_rows, best_k_rows):
    """Write a compact human-readable summary."""
    with open(path, "w") as handle:
        handle.write("Micro16S clustering evaluation summary\n")
        handle.write(f"Run directory: {run_dir}\n")
        handle.write(f"Generated at: {datetime.now().isoformat()}\n\n")
        handle.write(f"Per-run score rows: {len(score_rows)}\n")
        handle.write(f"Aggregated score rows: {len(aggregated_rows)}\n")
        handle.write(f"Skipped rows: {len(skipped_rows)}\n")
        if best_k_rows is not None:
            handle.write(f"Best-k rows: {len(best_k_rows)}\n")
        handle.write("\n")
        if aggregated_rows:
            df = pd.DataFrame(aggregated_rows)
            sort_col = f"{PRIMARY_SCORE_FOR_K_SELECTION}_mean"
            if sort_col in df.columns:
                df = df.sort_values(sort_col, ascending=False)
            top_rows = df.head(10)
            handle.write("Top aggregated rows:\n")
            handle.write(top_rows.to_string(index=False))
            handle.write("\n")


def _append_per_taxon_rows(storage, base_row, per_taxon_rows):
    """Append per-taxon rows with shared metadata."""
    for row in per_taxon_rows:
        merged = dict(base_row)
        merged.update(row)
        storage.append(merged)


def _write_contingency_csv_if_needed(run_dir, base_row, true_labels, predicted_clusters):
    """Save a contingency matrix CSV for one clustering case."""
    if not SAVE_CONTINGENCY_MATRICES:
        return
    df = _build_contingency_dataframe(true_labels, predicted_clusters, MAX_TAXA_PER_RANK_DIAGNOSTIC)
    if df is None:
        return
    out_dir = _ensure_dir(os.path.join(run_dir, "contingency_matrices"))
    name_parts = [base_row["backend"], base_row.get("model_variant") or f"k{base_row.get('kmer_k')}", base_row["eval_scope"], base_row["region_analysis"], f"rank{base_row['rank_idx']}", f"run{base_row['run_idx']}"]
    filename = "_".join(_sanitize_slug(part) for part in name_parts if part is not None) + ".csv"
    df.to_csv(os.path.join(out_dir, filename))


def _evaluate_candidate(run_dir, candidate_backend, candidate_name, candidate_k, region_major_arrays_by_split, labels_by_split, eval_cases, usable_regions, collect_scores, collect_diagnostics, selected_k_map, score_rows, skipped_rows, per_taxon_rows, assignment_writer):
    """Evaluate one representation candidate (embedding variant or k value)."""
    metric = CLUSTER_DISTANCE_METRIC_BY_BACKEND[candidate_backend]

    for case_idx, case in enumerate(eval_cases):
        case_labels_list = _concat_case_labels(case, labels_by_split)
        case_labels_by_rank = {int(rank): _extract_rank_labels(case_labels_list, rank) for rank in CLUSTER_RANKS}
        counts_by_split = {split_name: int(len(labels_by_split[split_name])) for split_name in case["source_splits"]}
        case_origins = _concat_case_origins(case, counts_by_split)
        n_sequences_case = int(len(case_labels_list))
        print(f"      Case {case_idx + 1}/{len(eval_cases)}: {case['eval_scope']} (n_seq={n_sequences_case})")

        for analysis_idx, analysis_name in enumerate(REGION_ANALYSES):
            run_plan = _build_region_run_plan(analysis_name, n_sequences_case, usable_regions)
            print(f"        Analysis: {analysis_name} ({len(run_plan)} runs)")
            for run_info in run_plan:
                per_seq_region_idx = run_info["per_seq_region_idx"]
                selected_vectors, selected_region_idx = _select_vectors_from_case_arrays(region_major_arrays_by_split, case, per_seq_region_idx)
                run_idx_display = run_info["run_idx"] + 1
                if run_idx_display == 1 or run_idx_display == len(run_plan) or run_idx_display % 5 == 0:
                    print(f"          Run {run_idx_display}/{len(run_plan)}")

                for rank in CLUSTER_RANKS:
                    rank_name = _rank_name_for(rank)
                    true_labels = case_labels_by_rank[int(rank)]
                    n_unique_labels = int(np.unique(true_labels).size)
                    if SKIP_IF_LT_TWO_TAXA and n_unique_labels < 2:
                        skipped_rows.append({"backend": candidate_backend, "model_variant": candidate_name if candidate_backend == "embedding" else None, "kmer_k": int(candidate_k) if candidate_backend == "kmer" else None, "distance_metric": metric, "split_name": case["split_name"], "eval_scope": case["eval_scope"], "scoring_scope": case["scoring_scope"], "region_analysis": analysis_name, "run_idx": int(run_info["run_idx"]), "run_seed": int(run_info["run_seed"]), "rank_idx": int(rank), "rank_name": rank_name, "reason": "fewer_than_two_taxa"})
                        continue

                    if candidate_backend == "kmer" and selected_k_map is not None and KMER_K_SELECTION_MODE == "best_per_analysis":
                        kmer_row_probe = {"eval_scope": case["eval_scope"], "split_name": case["split_name"], "scoring_scope": case["scoring_scope"], "region_analysis": analysis_name, "rank_idx": int(rank), "rank_name": rank_name, "distance_metric": metric}
                        wanted_k = selected_k_map.get(_kmer_selection_group_key(kmer_row_probe))
                        if wanted_k is None or int(candidate_k) != int(wanted_k):
                            continue

                    predicted_clusters, _ = _cluster_representations(selected_vectors, true_labels, metric)

                    if case["eval_scope"] == "excluded_mixed_with_train" and case["scoring_scope"] == "excluded_only":
                        scored_mask = case_origins == "excluded"
                    else:
                        scored_mask = np.ones(true_labels.shape[0], dtype=bool)
                    n_scored = int(np.sum(scored_mask))
                    if n_scored == 0:
                        skipped_rows.append({"backend": candidate_backend, "model_variant": candidate_name if candidate_backend == "embedding" else None, "kmer_k": int(candidate_k) if candidate_backend == "kmer" else None, "distance_metric": metric, "split_name": case["split_name"], "eval_scope": case["eval_scope"], "scoring_scope": case["scoring_scope"], "region_analysis": analysis_name, "run_idx": int(run_info["run_idx"]), "run_seed": int(run_info["run_seed"]), "rank_idx": int(rank), "rank_name": rank_name, "reason": "no_scoring_rows"})
                        continue

                    scored_true_labels = true_labels[scored_mask]
                    scored_clusters = predicted_clusters[scored_mask]
                    scores = _compute_requested_scores(scored_true_labels, scored_clusters)

                    base_row = {"backend": candidate_backend, "model_variant": candidate_name if candidate_backend == "embedding" else None, "kmer_k": int(candidate_k) if candidate_backend == "kmer" else None, "distance_metric": metric, "split_name": case["split_name"], "eval_scope": case["eval_scope"], "scoring_scope": case["scoring_scope"], "region_analysis": analysis_name, "run_idx": int(run_info["run_idx"]), "run_seed": int(run_info["run_seed"]), "rank_idx": int(rank), "rank_name": rank_name, "n_sequences_clustered": int(true_labels.shape[0]), "n_sequences_scored": n_scored, "n_clusters": n_unique_labels}
                    if run_info.get("shared_region_idx") is not None:
                        shared_fields = build_region_export_fields(int(run_info["shared_region_idx"]), REGION_IDX_TO_ID_MAPPING)
                        base_row["region_idx"] = int(shared_fields["region_idx"])
                        base_row["region_id"] = str(shared_fields["region_id"])
                    else:
                        base_row["region_idx"] = -1
                        base_row["region_id"] = "mixed_per_sequence"

                    if collect_scores:
                        score_row = dict(base_row)
                        score_row.update(scores)
                        score_rows.append(score_row)

                    if not collect_diagnostics:
                        continue

                    if SAVE_PER_TAXON_F1:
                        taxon_rows = _compute_per_taxon_f1_rows(scored_true_labels, scored_clusters, MAX_TAXA_PER_RANK_DIAGNOSTIC)
                        _append_per_taxon_rows(per_taxon_rows, base_row, taxon_rows)

                    if assignment_writer is not None:
                        for seq_idx in range(true_labels.shape[0]):
                            region_fields = build_region_export_fields(int(selected_region_idx[seq_idx]), REGION_IDX_TO_ID_MAPPING)
                            assignment_writer.writerow({"backend": candidate_backend, "model_variant": candidate_name if candidate_backend == "embedding" else None, "kmer_k": int(candidate_k) if candidate_backend == "kmer" else None, "distance_metric": metric, "split_name": case["split_name"], "eval_scope": case["eval_scope"], "scoring_scope": case["scoring_scope"], "region_analysis": analysis_name, "run_idx": int(run_info["run_idx"]), "run_seed": int(run_info["run_seed"]), "rank_idx": int(rank), "rank_name": rank_name, "sequence_idx_within_case": int(seq_idx), "origin_split": str(case_origins[seq_idx]), "is_scored": bool(scored_mask[seq_idx]), "true_label": str(true_labels[seq_idx]), "predicted_cluster": int(predicted_clusters[seq_idx]), "region_idx": int(region_fields["region_idx"]), "region_id": str(region_fields["region_id"])})

                    _write_contingency_csv_if_needed(run_dir, base_row, scored_true_labels, scored_clusters)


def main():
    """Run clustering evaluation using only this file's constants."""
    print("Starting clustering evaluation...")
    _validate_config()
    print("Configuration validated.")
    _set_seed(int(GLOBAL_RANDOM_SEED))

    run_dir = _build_run_dir()
    print(f"Output directory: {run_dir}")

    if SAVE_CONFIG_JSON:
        config_path = os.path.join(run_dir, "config.json")
        with open(config_path, "w") as handle:
            json.dump(_collect_config_snapshot(), handle, indent=2)
        print("Saved: config.json")

    dataset_split_dir = DATASET_SPLIT_DIR.rstrip("/")
    encoded_dir = _resolve_encoded_dir(dataset_split_dir)
    print(f"Dataset split dir: {dataset_split_dir}")
    print(f"Encoded dir: {encoded_dir}")

    eval_cases = _build_eval_cases()
    print(f"Evaluation cases: {len(eval_cases)} ({[c['eval_scope'] for c in eval_cases]})")
    splits_needed = set()
    for case in eval_cases:
        for split_name in case["source_splits"]:
            splits_needed.add(split_name)
    splits_needed = tuple(split_name for split_name in ("train", "test", "excluded") if split_name in splits_needed)
    if not splits_needed:
        raise ValueError("No evaluation cases were enabled. Check EVAL_SPLITS and excluded toggles.")

    print("Loading split indices and labels...")
    split_indices = {}
    labels_by_split = {}
    for split_name in splits_needed:
        idx_path = os.path.join(dataset_split_dir, SPLIT_NAME_TO_INDEX_FILENAME[split_name])
        indices = _load_indices_file(idx_path)
        split_indices[split_name] = indices
        labels_by_split[split_name] = _load_split_labels(dataset_split_dir, SPLIT_NAME_TO_FILESYSTEM_NAME[split_name], indices)
        print(f"  Loaded split={split_name}: n_sequences={len(indices)}")

    all_3bit_reps = None
    if "embedding" in REPRESENTATION_BACKENDS:
        print("Loading 3-bit representations...")
        all_3bit_reps = _load_3bit_reps(encoded_dir)
        print(f"  Loaded 3-bit reps: shape={all_3bit_reps.shape}")
        for split_name in splits_needed:
            if len(split_indices[split_name]) != len(labels_by_split[split_name]):
                raise ValueError(f"Label count does not match embedding sequence count for split '{split_name}'.")
        n_regions = int(all_3bit_reps.shape[0])
    else:
        n_regions = None

    kmer_candidates = ()
    all_kmer_reps = {}
    split_kmer_sources = {}
    if "kmer" in REPRESENTATION_BACKENDS:
        if KMER_K_SELECTION_MODE == "fixed":
            kmer_candidates = (int(FORCE_KMER_K),)
        else:
            kmer_candidates = tuple(int(k) for k in KMER_K_VALUES)
        print(f"Loading k-mer representations (k={list(kmer_candidates)})...")
        for k_value in kmer_candidates:
            print(f"  Loading k-mer k={k_value}...")
            all_kmer_reps[k_value] = _load_kmer_reps(encoded_dir, k_value)
            split_kmer_sources[k_value] = {split_name: _build_lazy_split_source(all_kmer_reps[k_value], split_indices[split_name]) for split_name in splits_needed}
            for split_name in splits_needed:
                if len(split_indices[split_name]) != len(labels_by_split[split_name]):
                    raise ValueError(f"Label count does not match k-mer sequence count for split '{split_name}' at k={k_value}.")
            k_regions = int(all_kmer_reps[k_value].shape[0])
            print(f"  Loaded k-mer k={k_value}: shape={all_kmer_reps[k_value].shape}")
            if n_regions is not None and k_regions != n_regions:
                raise ValueError(f"Region-count mismatch between backends: embedding has {n_regions}, {k_value}-mer has {k_regions}.")
        if n_regions is None:
            n_regions = int(all_kmer_reps[kmer_candidates[0]].shape[0])

    if n_regions is None:
        raise ValueError("Could not determine number of regions. Enable at least one representation backend.")

    usable_regions = _build_usable_region_pool(n_regions)
    print(f"Usable region count: {len(usable_regions)}")
    print("Starting clustering evaluation phase...")

    score_rows = []
    skipped_rows = []
    per_taxon_rows = []
    assignment_handle, assignment_writer = _open_assignment_writer_if_needed(run_dir)

    best_k_rows = None
    selected_k_map = None

    if "embedding" in REPRESENTATION_BACKENDS:
        print("Evaluating embedding backends...")
        for model_variant, checkpoint_path in EMBEDDING_MODEL_VARIANTS.items():
            if checkpoint_path is None:
                print(f"  Skipping embedding variant '{model_variant}' because checkpoint is None.")
                continue
            print(f"  Embedding variant: {model_variant}")
            embedding_split_arrays = _infer_embeddings_for_variant(model_variant, checkpoint_path, all_3bit_reps, {split_name: split_indices[split_name] for split_name in splits_needed})
            _evaluate_candidate(run_dir, "embedding", model_variant, None, embedding_split_arrays, labels_by_split, eval_cases, usable_regions, True, True, None, score_rows, skipped_rows, per_taxon_rows, assignment_writer)
            print(f"  Completed embedding variant: {model_variant}")

    if "kmer" in REPRESENTATION_BACKENDS:
        print("Evaluating k-mer backends...")
        if KMER_K_SELECTION_MODE == "fixed":
            for k_value in kmer_candidates:
                print(f"  K-mer k={k_value} (fixed mode)")
                _evaluate_candidate(run_dir, "kmer", None, k_value, split_kmer_sources[k_value], labels_by_split, eval_cases, usable_regions, True, True, None, score_rows, skipped_rows, per_taxon_rows, assignment_writer)
        else:
            print("  K-mer pass 1: best-K selection...")
            kmer_score_rows_pass1 = []
            kmer_skipped_rows_pass1 = []
            for k_value in kmer_candidates:
                print(f"    K-mer k={k_value} (pass 1)")
                _evaluate_candidate(run_dir, "kmer", None, k_value, split_kmer_sources[k_value], labels_by_split, eval_cases, usable_regions, True, False, None, kmer_score_rows_pass1, kmer_skipped_rows_pass1, [], None)

            print("  Selecting best k per analysis group...")
            selected_k_map, best_k_rows = _select_best_kmer_per_group(kmer_score_rows_pass1)
            filtered_pass1_scores = _filter_kmer_rows_by_selected_k(kmer_score_rows_pass1, selected_k_map)
            filtered_pass1_skips = _filter_kmer_rows_by_selected_k(kmer_skipped_rows_pass1, selected_k_map)
            score_rows.extend(filtered_pass1_scores)
            skipped_rows.extend(filtered_pass1_skips)

            if SAVE_PER_TAXON_F1 or SAVE_CLUSTER_ASSIGNMENTS or SAVE_CONTINGENCY_MATRICES:
                selected_k_values = sorted(set(selected_k_map.values()))
                print("  K-mer pass 2: diagnostics for selected k values...")
                for k_value in selected_k_values:
                    print(f"    K-mer k={k_value} (pass 2: diagnostics)")
                    _evaluate_candidate(run_dir, "kmer", None, k_value, split_kmer_sources[k_value], labels_by_split, eval_cases, usable_regions, False, True, selected_k_map, [], [], per_taxon_rows, assignment_writer)

    if assignment_handle is not None:
        assignment_handle.close()

    if not score_rows:
        raise RuntimeError("No score rows were produced. Check dataset/config settings.")

    print("Aggregating scores...")
    aggregated_rows = _aggregate_run_scores(score_rows)

    print("Saving results...")
    _write_tsv(os.path.join(run_dir, "scores_per_run.tsv"), score_rows)
    print(f"  Saved: scores_per_run.tsv ({len(score_rows)} rows)")
    _write_tsv(os.path.join(run_dir, "scores_aggregated.tsv"), aggregated_rows)
    print(f"  Saved: scores_aggregated.tsv ({len(aggregated_rows)} rows)")

    if KMER_K_SELECTION_MODE == "best_per_analysis":
        if best_k_rows is None:
            best_k_rows = []
        _write_tsv(os.path.join(run_dir, "best_kmer_k.tsv"), best_k_rows)
        print(f"  Saved: best_kmer_k.tsv ({len(best_k_rows)} rows)")

    if SAVE_PER_TAXON_F1:
        _write_tsv(os.path.join(run_dir, "per_taxon_f1.tsv"), per_taxon_rows)
        print(f"  Saved: per_taxon_f1.tsv ({len(per_taxon_rows)} rows)")

    if SAVE_SKIPPED_CASES_TABLE:
        _write_tsv(os.path.join(run_dir, "skipped_cases.tsv"), skipped_rows)
        print(f"  Saved: skipped_cases.tsv ({len(skipped_rows)} rows)")

    if SAVE_SUMMARY_TXT:
        _write_summary_txt(os.path.join(run_dir, "summary.txt"), run_dir, score_rows, aggregated_rows, skipped_rows, best_k_rows)
        print("  Saved: summary.txt")

    print("Clustering evaluation complete.")
    print(f"Saved results to: {run_dir}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print("Clustering evaluation failed.")
        print(str(error))
        traceback.print_exc()
        raise
