"""
Micro16S subsequence congruency (SSC) evaluation configuration (configuration only).

Purpose
- Define a complete, implementation-ready contract for SSC evaluation used in the
  "Micro16S Evaluations" methods section in `ch4_m16s.md`.
- Keep all behavior controlled by constants in this file (no CLI flags).

SSC definition
- For each replicate:
  SSC = (mean_distance_other_sequences - mean_distance_same_sequence)
        / (mean_distance_other_sequences + SSC_EPSILON)
- Higher SSC means stronger region invariance; 1.0 is near-perfect congruency.

Implementation contract
1. Load split representations with shape `(n_regions, n_sequences, dim)`.
2. Build the usable region pool in this exact order:
   - start with all region indices,
   - remove index 0 (full-sequence representation),
   - if `SSC_REGION_IDS` is not None, keep only that intersection.
3. If usable regions are fewer than `SSC_MIN_NUM_REGIONS`, skip and record the case.
4. For each independent run (`SSC_NUM_INDEPENDENT_RUNS`):
   - seed RNG with `GLOBAL_RANDOM_SEED + run_idx * INDEPENDENT_RUN_SEED_STRIDE`,
   - run `SSC_N_REPLICATES` replicates,
   - per replicate, sample `2 * SSC_N_SAMPLES` distinct sequence IDs (no replacement),
   - if fewer than `2 * SSC_N_SAMPLES` sequences are available in the split, raise an error,
   - compute `mean_distance_same_sequence` from pairwise inter-region distances for
     each sampled sequence in the first half, then average across sequences,
   - compute `mean_distance_other_sequences` from all cross distances between
     flattened region vectors from first and second halves.
5. Aggregate replicate SSC values into run-level mean/std.
6. Aggregate run-level values into final mean/std per split/backend/model-or-k.
7. For k-mer backends, choose reported K by `KMER_K_SELECTION_MODE`:
   - `best_per_split`: maximize mean SSC over all runs/replicates for that split,
   - `fixed`: use `FORCE_KMER_K`.

Expected inputs
- `DATASET_SPLIT_DIR` with standard split metadata (`training_indices.txt`,
  `testing_indices.txt`, `excluded_taxa_indices.txt`, `tax_objs/*`, `labels/*`).
- Encoded sequence representations in parent `encoded/` directory:
  `3bit_seq_reps.npy` (or packed equivalent) and `{K}-mer_seq_reps.npy`.
- One or more embedding checkpoints from `EMBEDDING_MODEL_VARIANTS`.

Expected outputs
- Root directory: `m16s_eval_results/ssc/`.
- Required result files:
  - `ssc_per_run.tsv` (one row per independent run),
  - `ssc_aggregated.tsv` (mean/std across runs),
  - `best_kmer_k.tsv` (when k-mer search mode is "best_per_split").
- Optional files:
  - `ssc_replicate_values.tsv`,
  - `region_metadata.tsv`,
  - `skipped_cases.tsv`,
  - `config.json`,
  - `summary.txt`.

Notes
- No CLI arguments: all behavior must be controlled via constants in this file.
- Full sequences (index 0) are *always* excluded from this evaluation.
- Every sequence has every region (no missing-region handling is needed).
- Replicates always sample sequence IDs without replacement.
- Embeddings always use cosine distance; k-mer backends may use euclidean or sqeuclidean.
- Region selectors accept either int indices (e.g., `4`) or region IDs from
  `region_indices.json` (e.g., `"V4-001"`), then normalize internally to indices.
- Export rows should always include both `region_idx` and `region_id`.
"""

import csv
from datetime import datetime
import json
import os
import random
import re

import numpy as np
from scipy.spatial.distance import cdist
import torch

from model import load_micro16s_model, run_inference
from m16s_eval_utils import build_region_export_fields, load_region_index_mappings, normalize_region_selection, resolve_region_indices_json_path

# Required paths ---------------------------------------------------------------
# Dataset split used for all SSC evals unless overridden below.
DATASET_SPLIT_DIR = "/mnt/secondary/micro16s_dbs/16s_databases/m16s_db_004/seqs/split_001/"

# Auto-resolved from DATASET_SPLIT_DIR using simple string split/concatenation.
REGION_INDICES_JSON_PATH = resolve_region_indices_json_path(DATASET_SPLIT_DIR)

# region_indices.json mappings used for ID/index conversion.
REGION_IDX_TO_ID_MAPPING, REGION_ID_TO_IDX_MAPPING = load_region_index_mappings(REGION_INDICES_JSON_PATH)

# Default embedding checkpoint (application model by default).
MODEL_CHECKPOINT = "/home/haig/Repos/micro16s/models/m16s_002/ckpts/m16s_002_16000_batches.pth" # Application model

# Optional second checkpoint for validation-model SSC comparisons.
# Set to a real path when available; keep None to skip validation-model runs.
VALIDATION_MODEL_CHECKPOINT = "/home/haig/Repos/micro16s/models/m16s_001/ckpts/m16s_001_16000_batches.pth" # Validation model


# Output configuration ---------------------------------------------------------
# Root directory for all SSC outputs.
OUTPUT_ROOT_DIR = "m16s_eval_results/ssc"

# Prefix used when building run directory names.
RUN_DIR_PREFIX = "ssc"

# Timestamp format appended into run names.
RUN_DIR_TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"

# Numeric suffix width for uniqueness; e.g., "__001", "__002".
RUN_DIR_SUFFIX_DIGITS = 3

# Optional extra user tags appended to run name (short strings only).
RUN_NAME_EXTRA_TAGS = ()


# Runtime / reproducibility ----------------------------------------------------
# Global seed for numpy/torch/random in the future implementation.
GLOBAL_RANDOM_SEED = 24

# Seed offset between independent SSC runs.
INDEPENDENT_RUN_SEED_STRIDE = 1_000

# Use CUDA if available for embedding inference.
USE_CUDA_IF_AVAILABLE = True

# Embedding inference batch size (adjust by GPU memory).
INFERENCE_BATCH_SIZE = 1024

# CPU workers/threads for preprocessing/distance ops.
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
# "best_per_split": choose best K separately per split by maximizing mean SSC
#                   across all runs/replicates in that split.
# "fixed": use FORCE_KMER_K only.
KMER_K_SELECTION_MODE = "best_per_split"

# Set when KMER_K_SELECTION_MODE == "fixed".
FORCE_KMER_K = None


# Split / region coverage ------------------------------------------------------
# Splits to evaluate.
EVAL_SPLITS = (
    "train",
    "test",
    "excluded",
)

# Optional explicit region selectors to use (None means all allowed by strategy).
# Accepts int indices and/or region ID strings (e.g., "V4-001").
SSC_REGION_IDS = None

# Require at least this many regions after filtering to compute SSC.
SSC_MIN_NUM_REGIONS = 2

# Normalized internal representation for filtering (always indices).
SSC_REGION_INDICES = normalize_region_selection(
    SSC_REGION_IDS,
    REGION_IDX_TO_ID_MAPPING,
    REGION_ID_TO_IDX_MAPPING,
    "SSC_REGION_IDS",
    REGION_INDICES_JSON_PATH,
)


# SSC sampling parameters ------------------------------------------------------
# Number of sequences sampled for "same" and "other" groups per replicate.
SSC_N_SAMPLES = 500

# Number of stochastic replicates per run.
SSC_N_REPLICATES = 20

# Number of independent full SSC runs (new random samples each run).
# Useful for stable estimates and confidence intervals.
SSC_NUM_INDEPENDENT_RUNS = 20

# Numerical epsilon in SSC denominator.
SSC_EPSILON = 1e-5

# Distance metric per backend.
# Allowed metrics: "cosine", "euclidean", "sqeuclidean".
SSC_DISTANCE_METRIC_BY_BACKEND = {
    "embedding": "cosine",
    "kmer": "sqeuclidean",
}



# Optional detailed exports ----------------------------------------------------
# Save raw replicate-level SSC values (per run, per split, per backend).
SAVE_RAW_REPLICATE_VALUES = True

# Save config snapshot JSON in each run directory.
SAVE_CONFIG_JSON = True

# Save per-run and aggregated TXT summaries.
SAVE_SUMMARY_TXT = True

# Save region metadata used in each run for reproducibility.
SAVE_REGION_METADATA = True

# Save rows for skipped analyses (e.g., insufficient regions or sequences).
SAVE_SKIPPED_CASES_TABLE = True


# Implementation note ----------------------------------------------------------
# Add implementation below this config block (loading, inference, SSC computation, saving).
# Keep behavior fully driven by constants above; do not add CLI args.


def _validate_config():
    """Validate high-level config consistency before any heavy work."""
    allowed_backends = {"embedding", "kmer"}
    backend_set = set(REPRESENTATION_BACKENDS)
    unknown_backends = backend_set - allowed_backends
    if unknown_backends:
        raise ValueError(f"Unknown values in REPRESENTATION_BACKENDS: {sorted(unknown_backends)}")

    if not EVAL_SPLITS:
        raise ValueError("EVAL_SPLITS must not be empty.")

    if SSC_N_SAMPLES < 1 or SSC_N_REPLICATES < 1 or SSC_NUM_INDEPENDENT_RUNS < 1:
        raise ValueError("SSC_N_SAMPLES, SSC_N_REPLICATES, and SSC_NUM_INDEPENDENT_RUNS must all be >= 1.")

    if SSC_MIN_NUM_REGIONS < 2:
        raise ValueError("SSC_MIN_NUM_REGIONS must be >= 2.")

    if KMER_K_SELECTION_MODE not in {"best_per_split", "fixed"}:
        raise ValueError(f"KMER_K_SELECTION_MODE must be 'best_per_split' or 'fixed', got: {KMER_K_SELECTION_MODE}")

    if "kmer" in backend_set and KMER_K_SELECTION_MODE == "fixed":
        if FORCE_KMER_K is None:
            raise ValueError("FORCE_KMER_K must be set when KMER_K_SELECTION_MODE == 'fixed'.")
        if FORCE_KMER_K not in KMER_K_VALUES:
            raise ValueError(f"FORCE_KMER_K ({FORCE_KMER_K}) must be present in KMER_K_VALUES ({KMER_K_VALUES}).")

    allowed_metrics = {"cosine", "euclidean", "sqeuclidean"}
    for backend in backend_set:
        metric = SSC_DISTANCE_METRIC_BY_BACKEND.get(backend)
        if metric not in allowed_metrics:
            raise ValueError(f"Unsupported distance metric for backend '{backend}': {metric}")

    if "embedding" in backend_set and not EMBEDDING_MODEL_VARIANTS:
        raise ValueError("EMBEDDING_MODEL_VARIANTS is empty while 'embedding' backend is enabled.")


def _seed_everything(seed):
    """Seed python/numpy/torch for reproducible behavior."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _safe_tag(text):
    """Sanitize free-form tag text for directory names."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", str(text).strip())
    return cleaned.strip("-._")


def _make_run_dir():
    """Create a unique timestamped run directory under OUTPUT_ROOT_DIR."""
    os.makedirs(OUTPUT_ROOT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime(RUN_DIR_TIMESTAMP_FORMAT)
    parts = [RUN_DIR_PREFIX, timestamp]
    for tag in RUN_NAME_EXTRA_TAGS:
        safe = _safe_tag(tag)
        if safe:
            parts.append(safe)
    stem = "__".join(parts)
    max_tries = 10 ** RUN_DIR_SUFFIX_DIGITS
    for suffix in range(1, max_tries):
        run_name = f"{stem}__{suffix:0{RUN_DIR_SUFFIX_DIGITS}d}"
        run_dir = os.path.join(OUTPUT_ROOT_DIR, run_name)
        if not os.path.exists(run_dir):
            os.makedirs(run_dir, exist_ok=False)
            return run_dir
    raise RuntimeError(f"Could not allocate a unique run directory under {OUTPUT_ROOT_DIR}.")


def _write_tsv(path, rows, fieldnames):
    """Write list-of-dict rows to TSV with a stable header."""
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _to_jsonable(value):
    """Recursively convert config values into JSON-safe structures."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, torch.device):
        return str(value)
    return str(value)


def _build_config_snapshot():
    """Capture all uppercase constants as a config snapshot."""
    snapshot = {}
    for key, value in globals().items():
        if key.isupper():
            snapshot[key] = _to_jsonable(value)
    return snapshot


def _resolve_encoded_dir(dataset_split_dir):
    """Resolve `<database>/seqs/encoded` from a split path."""
    marker = "/seqs/split_"
    if marker not in dataset_split_dir:
        raise ValueError(f"DATASET_SPLIT_DIR must contain '{marker}'. Got: {dataset_split_dir}")
    return dataset_split_dir.split(marker)[0] + "/seqs/encoded"


def _load_indices_file(path):
    """Load newline-delimited integer indices, handling empty files."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing indices file: {path}")
    if os.path.getsize(path) == 0:
        return np.empty(0, dtype=np.int64)
    return np.atleast_1d(np.loadtxt(path, dtype=np.int64))


def _load_split_indices(dataset_split_dir):
    """Load train/test/excluded index arrays from split metadata."""
    split_to_filename = {"train": "training_indices.txt", "test": "testing_indices.txt", "excluded": "excluded_taxa_indices.txt"}
    split_indices = {}
    for split_name in EVAL_SPLITS:
        if split_name not in split_to_filename:
            raise ValueError(f"Unsupported split in EVAL_SPLITS: '{split_name}'. Allowed: train, test, excluded.")
        path = os.path.join(dataset_split_dir, split_to_filename[split_name])
        split_indices[split_name] = _load_indices_file(path)
    return split_indices


def _load_3bit_seq_reps(encoded_dir):
    """Load 3-bit representations, transparently unpacking packed files."""
    packed_path = os.path.join(encoded_dir, "3bit_seq_reps_packed.npy")
    unpacked_path = os.path.join(encoded_dir, "3bit_seq_reps.npy")
    if os.path.exists(packed_path):
        arr = np.load(packed_path)
        if arr.ndim != 4 or arr.shape[-1] != 1:
            raise ValueError(f"Packed 3-bit tensor must have shape [N_REGIONS, N_SEQS, MAX_SEQ_LEN, 1], got {arr.shape}")
        unpacked = np.unpackbits(arr, axis=-1, bitorder="big")[..., :3]
        return unpacked.astype(bool)
    if os.path.exists(unpacked_path):
        arr = np.load(unpacked_path)
        if arr.ndim != 4 or arr.shape[-1] != 3:
            raise ValueError(f"Unpacked 3-bit tensor must have shape [N_REGIONS, N_SEQS, MAX_SEQ_LEN, 3], got {arr.shape}")
        return arr.astype(bool)
    raise FileNotFoundError(f"Could not find 3-bit representations in {encoded_dir}.")


def _get_device():
    """Select inference device from config and availability."""
    if USE_CUDA_IF_AVAILABLE and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _align_3bit_seq_len(seq_reps_3bit, target_len):
    """Crop or pad 3-bit sequences to the model's required sequence length."""
    current_len = seq_reps_3bit.shape[-2]
    if current_len == target_len:
        return seq_reps_3bit
    if current_len > target_len:
        return seq_reps_3bit[..., :target_len, :]

    # Right-pad with mask bit = 1 and base bits = 0.
    pad_len = target_len - current_len
    pad_shape = seq_reps_3bit.shape[:-2] + (pad_len, 3)
    pad_block = np.zeros(pad_shape, dtype=bool)
    pad_block[..., 0] = True
    return np.concatenate([seq_reps_3bit, pad_block], axis=-2)


def _build_usable_region_indices(n_regions_total):
    """Apply the region-pool contract: all -> remove index 0 -> optional selector intersection."""
    usable = [idx for idx in range(n_regions_total) if idx != 0]
    if SSC_REGION_INDICES is not None:
        selected = set(SSC_REGION_INDICES)
        usable = [idx for idx in usable if idx in selected]
    return usable


def _safe_unit_rows(vectors):
    """Row-normalize vectors for cosine distance, protecting zero rows."""
    row_sq = np.einsum("ij,ij->i", vectors, vectors, dtype=np.float64)
    row_norm = np.sqrt(np.maximum(row_sq, 0.0))
    inv_norm = np.divide(1.0, row_norm, out=np.zeros_like(row_norm), where=row_norm > 1e-12)
    return vectors * inv_norm[:, None]


def _mean_within_seq_distance(region_vectors, metric):
    """Mean pairwise inter-region distance for one sequence."""
    n_regions = region_vectors.shape[0]
    if n_regions < 2:
        return np.nan
    n_pairs = n_regions * (n_regions - 1) / 2.0

    if metric == "sqeuclidean":
        sum_sq = float(np.einsum("ij,ij->", region_vectors, region_vectors, dtype=np.float64))
        sum_vec = region_vectors.sum(axis=0, dtype=np.float64)
        pair_sum = n_regions * sum_sq - float(np.dot(sum_vec, sum_vec))
        return max(0.0, pair_sum / n_pairs)

    if metric == "cosine":
        unit = _safe_unit_rows(region_vectors)
        sum_unit = unit.sum(axis=0, dtype=np.float64)
        pair_cos_sum = (float(np.dot(sum_unit, sum_unit)) - n_regions) / 2.0
        mean_cos = pair_cos_sum / n_pairs
        return float(np.clip(1.0 - mean_cos, 0.0, 2.0))

    if metric == "euclidean":
        dists = cdist(region_vectors, region_vectors, metric="euclidean")
        return float(np.mean(dists[np.triu_indices(n_regions, k=1)]))

    raise ValueError(f"Unsupported metric: {metric}")


def _cross_stats_sqeuclidean(reps, region_indices, seq_indices):
    """Aggregate mean-squared-norm and mean-vector stats for sqeuclidean cross distance."""
    n_regions = len(region_indices)
    n_points = n_regions * len(seq_indices)
    dim = reps.shape[2]
    sum_sq_total = 0.0
    sum_vec_total = np.zeros(dim, dtype=np.float64)

    for seq_idx in seq_indices:
        region_vectors = reps[region_indices, int(seq_idx), :]
        sum_sq_total += float(np.einsum("ij,ij->", region_vectors, region_vectors, dtype=np.float64))
        sum_vec_total += region_vectors.sum(axis=0, dtype=np.float64)

    mean_sq = sum_sq_total / float(n_points)
    mean_vec = sum_vec_total / float(n_points)
    return mean_sq, mean_vec


def _cross_stats_cosine(reps, region_indices, seq_indices):
    """Aggregate mean normalized vector for cosine cross distance."""
    n_regions = len(region_indices)
    n_points = n_regions * len(seq_indices)
    dim = reps.shape[2]
    sum_unit_total = np.zeros(dim, dtype=np.float64)

    for seq_idx in seq_indices:
        region_vectors = reps[region_indices, int(seq_idx), :]
        unit = _safe_unit_rows(region_vectors)
        sum_unit_total += unit.sum(axis=0, dtype=np.float64)

    mean_unit = sum_unit_total / float(n_points)
    return mean_unit


def _flatten_vectors(reps, region_indices, seq_indices):
    """Flatten `(regions, seqs, dim)` subset into `(regions*seqs, dim)` for fallback metrics."""
    blocks = []
    for seq_idx in seq_indices:
        blocks.append(reps[region_indices, int(seq_idx), :])
    return np.concatenate(blocks, axis=0)


def _mean_cross_distance(reps, region_indices, seq_indices_a, seq_indices_b, metric):
    """Mean distance between flattened region vectors from two disjoint sequence sets."""
    if metric == "sqeuclidean":
        mean_sq_a, mean_vec_a = _cross_stats_sqeuclidean(reps, region_indices, seq_indices_a)
        mean_sq_b, mean_vec_b = _cross_stats_sqeuclidean(reps, region_indices, seq_indices_b)
        value = mean_sq_a + mean_sq_b - 2.0 * float(np.dot(mean_vec_a, mean_vec_b))
        return float(max(0.0, value))

    if metric == "cosine":
        mean_unit_a = _cross_stats_cosine(reps, region_indices, seq_indices_a)
        mean_unit_b = _cross_stats_cosine(reps, region_indices, seq_indices_b)
        return float(np.clip(1.0 - float(np.dot(mean_unit_a, mean_unit_b)), 0.0, 2.0))

    if metric == "euclidean":
        vecs_a = _flatten_vectors(reps, region_indices, seq_indices_a)
        vecs_b = _flatten_vectors(reps, region_indices, seq_indices_b)
        chunk_size = max(32, min(1024, int(1e6 // max(1, vecs_b.shape[0]))))
        total_sum = 0.0
        total_count = 0
        for start in range(0, vecs_a.shape[0], chunk_size):
            end = min(start + chunk_size, vecs_a.shape[0])
            block = cdist(vecs_a[start:end], vecs_b, metric="euclidean")
            total_sum += float(np.sum(block, dtype=np.float64))
            total_count += block.size
        return total_sum / float(total_count)

    raise ValueError(f"Unsupported metric: {metric}")


def _evaluate_case(case_key, reps, split_name, backend, model_variant, k_value, distance_metric):
    """Run SSC for one concrete case and return all export rows."""
    if reps.ndim != 3:
        raise ValueError(f"Representations must have shape (n_regions, n_sequences, dim). Got: {reps.shape}")
    n_regions_total, n_sequences, dim = reps.shape
    if dim < 1:
        raise ValueError("Representation dim must be >= 1.")

    usable_region_indices = _build_usable_region_indices(n_regions_total)
    model_or_k = model_variant if backend == "embedding" else f"{k_value}-mer"

    # Collect region metadata rows for reproducibility.
    region_rows = []
    for region_idx in usable_region_indices:
        region_fields = build_region_export_fields(region_idx, REGION_IDX_TO_ID_MAPPING)
        region_rows.append({"split": split_name, "backend": backend, "model_variant": model_variant if model_variant is not None else "", "k_value": "" if k_value is None else int(k_value), "model_or_k": model_or_k, "region_idx": region_fields["region_idx"], "region_id": region_fields["region_id"]})

    # Skip case when too few usable regions remain.
    if len(usable_region_indices) < SSC_MIN_NUM_REGIONS:
        skip_row = {"split": split_name, "backend": backend, "model_variant": model_variant if model_variant is not None else "", "k_value": "" if k_value is None else int(k_value), "model_or_k": model_or_k, "distance_metric": distance_metric, "n_regions_total": int(n_regions_total), "n_regions_usable": int(len(usable_region_indices)), "n_sequences": int(n_sequences), "reason": f"usable_regions_lt_{SSC_MIN_NUM_REGIONS}"}
        return {"case_key": case_key, "split": split_name, "backend": backend, "model_variant": model_variant, "k_value": k_value, "model_or_k": model_or_k, "distance_metric": distance_metric, "region_rows": region_rows, "skipped": True, "skip_row": skip_row, "run_rows": [], "replicate_rows": [], "aggregated_row": None, "all_replicate_ssc": []}

    # Per contract, insufficient sequences is an error, not a skip.
    required_sequences = 2 * SSC_N_SAMPLES
    if n_sequences < required_sequences:
        raise ValueError(f"Split '{split_name}' with backend='{backend}' and model_or_k='{model_or_k}' has {n_sequences} sequences but needs at least {required_sequences} for SSC sampling.")

    run_rows = []
    replicate_rows = []
    run_level_means = []
    all_replicate_ssc = []
    same_seq_cache = {}

    for run_idx in range(SSC_NUM_INDEPENDENT_RUNS):
        run_seed = GLOBAL_RANDOM_SEED + run_idx * INDEPENDENT_RUN_SEED_STRIDE
        rng = np.random.default_rng(run_seed)
        ssc_values_this_run = []

        for replicate_idx in range(SSC_N_REPLICATES):
            sampled = rng.choice(n_sequences, size=required_sequences, replace=False)
            sample_indices_same = sampled[:SSC_N_SAMPLES]
            sample_indices_other = sampled[SSC_N_SAMPLES:]

            # Mean distance between regions of the same sequence.
            same_distances = []
            for seq_idx in sample_indices_same:
                seq_key = int(seq_idx)
                if seq_key not in same_seq_cache:
                    seq_region_vectors = reps[usable_region_indices, seq_key, :]
                    same_seq_cache[seq_key] = _mean_within_seq_distance(seq_region_vectors, distance_metric)
                same_distances.append(same_seq_cache[seq_key])
            mean_distance_same_sequence = float(np.mean(same_distances))

            # Mean distance between regions from different sampled sequence sets.
            mean_distance_other_sequences = _mean_cross_distance(reps, usable_region_indices, sample_indices_same, sample_indices_other, distance_metric)

            # SSC definition from the contract.
            ssc_value = (mean_distance_other_sequences - mean_distance_same_sequence) / (mean_distance_other_sequences + SSC_EPSILON)
            ssc_value = float(ssc_value)
            ssc_values_this_run.append(ssc_value)
            all_replicate_ssc.append(ssc_value)

            if SAVE_RAW_REPLICATE_VALUES:
                replicate_rows.append({"split": split_name, "backend": backend, "model_variant": model_variant if model_variant is not None else "", "k_value": "" if k_value is None else int(k_value), "model_or_k": model_or_k, "distance_metric": distance_metric, "run_idx": int(run_idx), "run_seed": int(run_seed), "replicate_idx": int(replicate_idx), "n_regions_total": int(n_regions_total), "n_regions_usable": int(len(usable_region_indices)), "n_sequences": int(n_sequences), "ssc": ssc_value, "mean_distance_same_sequence": mean_distance_same_sequence, "mean_distance_other_sequences": mean_distance_other_sequences})

        run_mean = float(np.mean(ssc_values_this_run))
        run_std = float(np.std(ssc_values_this_run))
        run_level_means.append(run_mean)
        run_rows.append({"split": split_name, "backend": backend, "model_variant": model_variant if model_variant is not None else "", "k_value": "" if k_value is None else int(k_value), "model_or_k": model_or_k, "distance_metric": distance_metric, "run_idx": int(run_idx), "run_seed": int(run_seed), "n_regions_total": int(n_regions_total), "n_regions_usable": int(len(usable_region_indices)), "n_sequences": int(n_sequences), "ssc_n_samples": int(SSC_N_SAMPLES), "ssc_n_replicates": int(SSC_N_REPLICATES), "ssc_mean": run_mean, "ssc_std": run_std})

    aggregated_row = {"split": split_name, "backend": backend, "model_variant": model_variant if model_variant is not None else "", "k_value": "" if k_value is None else int(k_value), "model_or_k": model_or_k, "distance_metric": distance_metric, "n_regions_total": int(n_regions_total), "n_regions_usable": int(len(usable_region_indices)), "n_sequences": int(n_sequences), "num_independent_runs": int(SSC_NUM_INDEPENDENT_RUNS), "ssc_mean": float(np.mean(run_level_means)), "ssc_std": float(np.std(run_level_means))}
    return {"case_key": case_key, "split": split_name, "backend": backend, "model_variant": model_variant, "k_value": k_value, "model_or_k": model_or_k, "distance_metric": distance_metric, "region_rows": region_rows, "skipped": False, "skip_row": None, "run_rows": run_rows, "replicate_rows": replicate_rows, "aggregated_row": aggregated_row, "all_replicate_ssc": all_replicate_ssc}


def _collect_embedding_cases(all_3bit_reps, split_indices, device):
    """Evaluate SSC cases for all embedding variants and splits."""
    results = []
    for model_variant, ckpt_path in EMBEDDING_MODEL_VARIANTS.items():
        if ckpt_path is None:
            continue
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Embedding checkpoint not found for model variant '{model_variant}': {ckpt_path}")

        print(f"[embedding] Loading model variant '{model_variant}' from: {ckpt_path}")
        model = load_micro16s_model(ckpt_path).to(device).eval()
        metric = SSC_DISTANCE_METRIC_BY_BACKEND["embedding"]

        for split_name, seq_indices in split_indices.items():
            print(f"[embedding] Inference for split='{split_name}' (n_seqs={len(seq_indices)})")
            split_3bit = all_3bit_reps[:, seq_indices, :, :]
            split_3bit = _align_3bit_seq_len(split_3bit, model.max_seq_len)
            reps = run_inference(model, split_3bit, device=device, batch_size=INFERENCE_BATCH_SIZE, return_numpy=True)
            case_key = (split_name, "embedding", model_variant, None)
            results.append(_evaluate_case(case_key, reps, split_name, "embedding", model_variant, None, metric))

        # Free model memory before loading the next checkpoint.
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return results


def _collect_kmer_cases(encoded_dir, split_indices):
    """Evaluate SSC cases for all requested k-mer K values and splits."""
    results = []
    if KMER_K_SELECTION_MODE == "fixed":
        k_values = (FORCE_KMER_K,)
    else:
        k_values = KMER_K_VALUES

    metric = SSC_DISTANCE_METRIC_BY_BACKEND["kmer"]
    for k_value in k_values:
        kmer_path = os.path.join(encoded_dir, f"{k_value}-mer_seq_reps.npy")
        if not os.path.exists(kmer_path):
            raise FileNotFoundError(f"Missing k-mer representation file: {kmer_path}")

        print(f"[kmer] Loading K={k_value} representations from: {kmer_path}")
        all_kmer_reps = np.load(kmer_path, mmap_mode="r")
        if all_kmer_reps.ndim != 3:
            raise ValueError(f"{k_value}-mer representations must have shape (n_regions, n_sequences, dim). Got: {all_kmer_reps.shape}")

        for split_name, seq_indices in split_indices.items():
            print(f"[kmer] Evaluating split='{split_name}' for K={k_value} (n_seqs={len(seq_indices)})")
            reps = np.asarray(all_kmer_reps[:, seq_indices, :])
            case_key = (split_name, "kmer", None, int(k_value))
            results.append(_evaluate_case(case_key, reps, split_name, "kmer", None, int(k_value), metric))

    return results


def _choose_reported_cases(case_results):
    """Apply K selection policy and return selected case keys + best-k export rows."""
    selected_keys = set()
    best_k_rows = []

    # Embedding rows are always reported.
    for case in case_results:
        if case["backend"] == "embedding" and not case["skipped"]:
            selected_keys.add(case["case_key"])

    kmer_cases = [case for case in case_results if case["backend"] == "kmer"]
    if not kmer_cases:
        return selected_keys, best_k_rows

    if KMER_K_SELECTION_MODE == "fixed":
        for case in kmer_cases:
            if case["k_value"] == FORCE_KMER_K and not case["skipped"]:
                selected_keys.add(case["case_key"])
        return selected_keys, best_k_rows

    # best_per_split: maximize mean SSC across all runs/replicates for each split.
    for split_name in EVAL_SPLITS:
        candidates = [case for case in kmer_cases if case["split"] == split_name and not case["skipped"] and case["all_replicate_ssc"]]
        if not candidates:
            continue

        scored_candidates = []
        for case in candidates:
            mean_ssc = float(np.mean(case["all_replicate_ssc"]))
            scored_candidates.append((mean_ssc, int(case["k_value"]), case))

        # Deterministic tiebreak: highest mean SSC, then smallest K.
        scored_candidates.sort(key=lambda x: (-x[0], x[1]))
        best_mean, best_k, best_case = scored_candidates[0]
        selected_keys.add(best_case["case_key"])

        for mean_ssc, k_value, case in sorted(scored_candidates, key=lambda x: x[1]):
            best_k_rows.append({"split": split_name, "k_value": int(k_value), "mean_ssc_across_runs_and_replicates": float(mean_ssc), "is_selected": int(k_value == best_k), "selection_mode": KMER_K_SELECTION_MODE})

        print(f"[kmer] best_per_split selected K={best_k} for split='{split_name}' with mean SSC={best_mean:.6f}")

    return selected_keys, best_k_rows


def _build_summary_text(run_dir, case_results, selected_case_keys, skipped_rows, best_k_rows):
    """Create a concise human-readable run summary."""
    lines = []
    lines.append("Micro16S SSC Evaluation Summary")
    lines.append("=" * 32)
    lines.append(f"Run directory: {run_dir}")
    lines.append(f"Dataset split dir: {DATASET_SPLIT_DIR}")
    lines.append(f"Region indices json: {REGION_INDICES_JSON_PATH}")
    lines.append(f"Backends: {', '.join(REPRESENTATION_BACKENDS)}")
    lines.append(f"Splits: {', '.join(EVAL_SPLITS)}")
    lines.append(f"SSC runs x replicates: {SSC_NUM_INDEPENDENT_RUNS} x {SSC_N_REPLICATES}")
    lines.append(f"SSC n_samples: {SSC_N_SAMPLES}")
    lines.append(f"Distance metrics: {SSC_DISTANCE_METRIC_BY_BACKEND}")
    lines.append(f"Total evaluated cases: {len(case_results)}")
    lines.append(f"Reported cases: {sum(1 for case in case_results if case['case_key'] in selected_case_keys)}")
    lines.append(f"Skipped cases: {len(skipped_rows)}")
    if best_k_rows:
        lines.append("")
        lines.append("Best k-mer K by split:")
        for row in best_k_rows:
            if row["is_selected"] == 1:
                lines.append(f"  - split={row['split']}: K={row['k_value']} (mean_ssc={row['mean_ssc_across_runs_and_replicates']:.6f})")
    return "\n".join(lines) + "\n"


def main():
    """Run the full SSC evaluation pipeline with constants from this file only."""
    _validate_config()
    _seed_everything(GLOBAL_RANDOM_SEED)

    # Respect CPU worker config when running on CPU-heavy workloads.
    if NUM_WORKERS is not None and NUM_WORKERS > 0:
        torch.set_num_threads(int(NUM_WORKERS))

    run_dir = _make_run_dir()
    print(f"[ssc] Writing outputs to: {run_dir}")

    encoded_dir = _resolve_encoded_dir(DATASET_SPLIT_DIR)
    split_indices = _load_split_indices(DATASET_SPLIT_DIR)
    device = _get_device()
    print(f"[ssc] Using device: {device}")

    case_results = []

    # Embedding backend cases.
    if "embedding" in REPRESENTATION_BACKENDS:
        all_3bit_reps = _load_3bit_seq_reps(encoded_dir)
        case_results.extend(_collect_embedding_cases(all_3bit_reps, split_indices, device))
        del all_3bit_reps
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # k-mer backend cases.
    if "kmer" in REPRESENTATION_BACKENDS:
        case_results.extend(_collect_kmer_cases(encoded_dir, split_indices))

    selected_case_keys, best_k_rows = _choose_reported_cases(case_results)

    # Flatten rows for selected report cases.
    per_run_rows = []
    aggregated_rows = []
    replicate_rows = []
    region_rows = []
    skipped_rows = []
    for case in case_results:
        if case["skipped"] and case["skip_row"] is not None:
            skipped_rows.append(case["skip_row"])
        if case["case_key"] not in selected_case_keys:
            continue
        per_run_rows.extend(case["run_rows"])
        if case["aggregated_row"] is not None:
            aggregated_rows.append(case["aggregated_row"])
        if SAVE_RAW_REPLICATE_VALUES:
            replicate_rows.extend(case["replicate_rows"])
        if SAVE_REGION_METADATA:
            region_rows.extend(case["region_rows"])

    per_run_columns = ["split", "backend", "model_variant", "k_value", "model_or_k", "distance_metric", "run_idx", "run_seed", "n_regions_total", "n_regions_usable", "n_sequences", "ssc_n_samples", "ssc_n_replicates", "ssc_mean", "ssc_std"]
    aggregated_columns = ["split", "backend", "model_variant", "k_value", "model_or_k", "distance_metric", "n_regions_total", "n_regions_usable", "n_sequences", "num_independent_runs", "ssc_mean", "ssc_std"]
    replicate_columns = ["split", "backend", "model_variant", "k_value", "model_or_k", "distance_metric", "run_idx", "run_seed", "replicate_idx", "n_regions_total", "n_regions_usable", "n_sequences", "ssc", "mean_distance_same_sequence", "mean_distance_other_sequences"]
    region_columns = ["split", "backend", "model_variant", "k_value", "model_or_k", "region_idx", "region_id"]
    skipped_columns = ["split", "backend", "model_variant", "k_value", "model_or_k", "distance_metric", "n_regions_total", "n_regions_usable", "n_sequences", "reason"]
    best_k_columns = ["split", "k_value", "mean_ssc_across_runs_and_replicates", "is_selected", "selection_mode"]

    # Required outputs.
    _write_tsv(os.path.join(run_dir, "ssc_per_run.tsv"), per_run_rows, per_run_columns)
    _write_tsv(os.path.join(run_dir, "ssc_aggregated.tsv"), aggregated_rows, aggregated_columns)

    # Optional outputs.
    if SAVE_RAW_REPLICATE_VALUES:
        _write_tsv(os.path.join(run_dir, "ssc_replicate_values.tsv"), replicate_rows, replicate_columns)
    if SAVE_REGION_METADATA:
        _write_tsv(os.path.join(run_dir, "region_metadata.tsv"), region_rows, region_columns)
    if SAVE_SKIPPED_CASES_TABLE:
        _write_tsv(os.path.join(run_dir, "skipped_cases.tsv"), skipped_rows, skipped_columns)
    if KMER_K_SELECTION_MODE == "best_per_split":
        _write_tsv(os.path.join(run_dir, "best_kmer_k.tsv"), best_k_rows, best_k_columns)
    if SAVE_CONFIG_JSON:
        with open(os.path.join(run_dir, "config.json"), "w") as f:
            json.dump(_build_config_snapshot(), f, indent=2)
    if SAVE_SUMMARY_TXT:
        summary_text = _build_summary_text(run_dir, case_results, selected_case_keys, skipped_rows, best_k_rows)
        with open(os.path.join(run_dir, "summary.txt"), "w") as f:
            f.write(summary_text)

    print("[ssc] Done.")


if __name__ == "__main__":
    main()
