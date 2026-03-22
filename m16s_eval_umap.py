"""
Micro16S UMAP evaluation configuration (configuration only).

Purpose
- Define an implementation-ready contract for UMAP panels in the "Micro16S
  Evaluations" methods section in `ch4_m16s.md`.
- Keep every behavior controlled by constants in this file (no CLI flags).

Panels (all from `EVAL_SPLIT`)
1. Top N phyla by sequence count; points colored by phylum.
2. Top N families within one order; color by family and shape/marker by genus.
3. All sequences in a target family; color by genus.
4. All subsequence regions for N random sequences in a target taxon at rank X;
   color by sequence.

Implementation contract
1. Load split arrays and taxonomy labels.
2. Build usable subsequence region pool:
   - start from all region indices,
   - remove index 0 (full-sequence representation),
   - if `TAXON_PANEL_REGION_IDS` is not None, keep only those IDs.
3. For panels 1-3, create one representation per sequence:
   - `mixed_random_per_sequence`: random region per sequence from pool,
   - `single_region`: one shared region (`TAXON_PANEL_FIXED_REGION_ID`) for all.
4. For panel 4, choose N random sequences in the target taxon at rank X, then emit one point per
   (sequence, region) across all subsequence regions.
5. Fit UMAP separately for each panel/backend/model-or-k combination using
   `UMAP_DISTANCE_METRIC_BY_BACKEND`.
6. Use one global run index across all panel/backend/model-or-k combinations in
   deterministic traversal order, then set
   `run_seed = GLOBAL_RANDOM_SEED + global_run_idx * RUN_SEED_STRIDE`.
   Use `run_seed` for random panel sampling and as the UMAP `random_state`.
7. Save points table and figures per panel, plus a panel summary row.
8. If `MAX_POINTS_PER_PANEL` is set and exceeded, randomly subsample points and
   print a warning.

Expected inputs
- `DATASET_SPLIT_DIR` split metadata and taxonomy objects.
- Encoded arrays in parent `encoded/` directory:
  `3bit_seq_reps.npy` (or packed equivalent) and `{K}-mer_seq_reps.npy`.
- Embedding checkpoint(s) listed in `EMBEDDING_MODEL_VARIANTS`.

Expected outputs
- Root directory: `m16s_eval_results/umap/`.
- Per-panel run directory containing:
  - `points.tsv`,
  - `plot.png` and optional `plot.svg`,
  - `panel_summary.tsv`,
  - optional metadata/config files.

Notes
- No CLI arguments: all behavior must be controlled via constants in this file.
- Full sequences (index 0) are *always* excluded from this evaluation.
- Every sequence has every region (no missing-region handling is needed).
- Panels are independent: panel-specific sequence/taxon/region filters are not reused
  across other panels.
- Embeddings always use cosine distance; k-mer backends may use euclidean or sqeuclidean.
- Region selectors accept either int indices (e.g., `4`) or region IDs from
  `region_indices.json` (e.g., `"V4-001"`), then normalize internally to indices.
- Export rows should always include both `region_idx` and `region_id`.
"""

import json
import os
import pickle
import random
import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import umap
from matplotlib.lines import Line2D

from m16s_eval_utils import build_region_export_fields, load_region_index_mappings, normalize_region_selection, normalize_single_region_value, resolve_region_indices_json_path
from model import load_micro16s_model, run_inference

# Required paths ---------------------------------------------------------------
# Dataset split used for all UMAP evals unless overridden below.
DATASET_SPLIT_DIR = "/mnt/secondary/micro16s_dbs/16s_databases/m16s_db_004/seqs/split_001/"

# Auto-resolved from DATASET_SPLIT_DIR using simple string split/concatenation.
REGION_INDICES_JSON_PATH = resolve_region_indices_json_path(DATASET_SPLIT_DIR)

# region_indices.json mappings used for ID/index conversion.
REGION_IDX_TO_ID_MAPPING, REGION_ID_TO_IDX_MAPPING = load_region_index_mappings(REGION_INDICES_JSON_PATH)

# Default embedding checkpoint (application model by default).
MODEL_CHECKPOINT = "/home/haig/Repos/micro16s/models/m16s_002/ckpts/m16s_002_16000_batches.pth" # Application model

# Optional second checkpoint for validation-model UMAP comparisons.
# Set to a real path when available; keep None to skip validation-model runs.
VALIDATION_MODEL_CHECKPOINT = "/home/haig/Repos/micro16s/models/m16s_001/ckpts/m16s_001_16000_batches.pth" # Validation model


# Output configuration ---------------------------------------------------------
# Root directory for all UMAP outputs.
OUTPUT_ROOT_DIR = "m16s_eval_results/umap"

# Prefix used when building run directory names.
RUN_DIR_PREFIX = "umap"

# Timestamp format appended into run names.
RUN_DIR_TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"

# Numeric suffix width for uniqueness; e.g., "__001", "__002".
RUN_DIR_SUFFIX_DIGITS = 3

# Optional extra user tags appended to run name (short strings only).
RUN_NAME_EXTRA_TAGS = ()


# Runtime / reproducibility ----------------------------------------------------
# Global seed for numpy/torch/random in the future implementation.
GLOBAL_RANDOM_SEED = 24

# Seed offset applied per panel/backend/model combination.
RUN_SEED_STRIDE = 1_000

# Use CUDA if available for embedding inference.
USE_CUDA_IF_AVAILABLE = True

# Embedding inference batch size (adjust by GPU memory).
INFERENCE_BATCH_SIZE = 1024

# CPU workers/threads for preprocessing/UMAP.
NUM_WORKERS = 12


# Representation backends ------------------------------------------------------
# Backends to run in this script.
# Allowed values: "embedding", "kmer".
REPRESENTATION_BACKENDS = ("embedding", "kmer")

# Named embedding model variants to evaluate.
EMBEDDING_MODEL_VARIANTS = {
    "application": MODEL_CHECKPOINT,
    "validation": VALIDATION_MODEL_CHECKPOINT,
}

# Candidate K values for k-mer baselines.
KMER_K_VALUES = (5, 6, 7)

# K selection policy for k-mer UMAPs:
# - "fixed": always use MANUAL_KMER_K.
# - "all": generate one panel per K in KMER_K_VALUES.
KMER_K_SELECTION_MODE = "all"

# Used when KMER_K_SELECTION_MODE == "fixed".
MANUAL_KMER_K = 6


# Split / region strategy ------------------------------------------------------
# UMAP panels are probably train-only
EVAL_SPLIT = "train"

# Optional explicit region selectors for panels 1-3.
# Accepts int indices and/or region ID strings (e.g., "V4-001").
TAXON_PANEL_REGION_IDS = None

# Region sampling mode for taxon-based panels:
# - "mixed_random_per_sequence": random region per sequence.
# - "single_region": one region for all sequences in panel.
TAXON_PANEL_REGION_MODE = "mixed_random_per_sequence"

# Region selector used when TAXON_PANEL_REGION_MODE == "single_region".
# Accepts either int index or region ID string.
TAXON_PANEL_FIXED_REGION_ID = None

# Normalized internal representations (always indices).
TAXON_PANEL_REGION_INDICES = normalize_region_selection(
    TAXON_PANEL_REGION_IDS,
    REGION_IDX_TO_ID_MAPPING,
    REGION_ID_TO_IDX_MAPPING,
    "TAXON_PANEL_REGION_IDS",
    REGION_INDICES_JSON_PATH,
)
TAXON_PANEL_FIXED_REGION_INDEX = normalize_single_region_value(
    TAXON_PANEL_FIXED_REGION_ID,
    REGION_IDX_TO_ID_MAPPING,
    REGION_ID_TO_IDX_MAPPING,
    "TAXON_PANEL_FIXED_REGION_ID",
    REGION_INDICES_JSON_PATH,
)

if TAXON_PANEL_REGION_MODE == "single_region" and TAXON_PANEL_FIXED_REGION_INDEX is None:
    raise ValueError(
        "TAXON_PANEL_FIXED_REGION_ID must be set when TAXON_PANEL_REGION_MODE == 'single_region'."
    )


# UMAP algorithm parameters ----------------------------------------------------
# 2D output for figure-ready ordinations.
UMAP_N_COMPONENTS = 2

# Distance metric used by each representation backend.
# Embeddings are fixed to cosine; k-mer can be euclidean or sqeuclidean.
UMAP_DISTANCE_METRIC_BY_BACKEND = {
    "embedding": "cosine",
    "kmer": "sqeuclidean",
}

# Other parameters for UMAP.
# Panel-specific neighbors/min_dist are configured in the panel config section.
UMAP_INIT = "spectral"
UMAP_N_JOBS = 12


# Panel toggles and panel-specific filters ------------------------------------
# Panel 1: top phyla by count.
# Plot encoding: colored by phylum; shaped by nothing (all dots).
RUN_PANEL_TOP_PHYLA = True
TOP_PHYLA_N = 6
TOP_PHYLA_COLOR_RANK = 1  # phylum
PANEL1_UMAP_N_NEIGHBORS = 20  # Small=tighter local clusters, large=smoother global structure and less fragmentation.
PANEL1_UMAP_MIN_DIST = 1.0    # Small=more compact clusters, large=more spread out clusters.

# Panel 2: top families within one order by count.
# Plot encoding: colored by family; shaped by genus.
RUN_PANEL_TOP_FAMILIES_IN_ORDER = True
TOP_FAMILIES_ORDER_LABEL = None  # None -> auto-select order with highest sequence count.
TOP_FAMILIES_N = 10
TOP_FAMILIES_COLOR_RANK = 4      # family
TOP_FAMILIES_SHAPE_RANK = 5      # genus (marker shape)
# Panel 2 marker-mode toggle:
# - False: marker shape encodes genus
# - True: force all points to dot markers
ALL_SHAPES_DOTS = True
PANEL2_UMAP_N_NEIGHBORS = 12  # Small=tighter local clusters, large=smoother global structure and less fragmentation.
PANEL2_UMAP_MIN_DIST = 0.4    # Small=more compact clusters, large=more spread out clusters.

# Panel 3: one family, colored by genus.
# Plot encoding: colored by genus; shaped by nothing (all dots).
RUN_PANEL_TARGET_FAMILY = True
TARGET_FAMILY_LABEL = "Enterobacteriaceae"
TARGET_FAMILY_COLOR_RANK = 5      # genus
PANEL3_UMAP_N_NEIGHBORS = 50  # Small=tighter local clusters, large=smoother global structure and less fragmentation.
PANEL3_UMAP_MIN_DIST = 0.08    # Small=more compact clusters, large=more spread out clusters.

# Panel 4: region invariance for N random sequences in a target taxon at rank X.
# Plot encoding: colored by sequence ID; shaped by nothing (all dots).
# Uses all subsequence regions for each selected sequence.
RUN_PANEL_REGION_INVARIANCE = True
REGION_INVARIANCE_TARGET_TAXON_RANK = 5
REGION_INVARIANCE_TARGET_TAXON_LABEL = "Enterobacteriaceae"
REGION_INVARIANCE_NUM_SEQS_FOR_PANEL = 5
PANEL4_UMAP_N_NEIGHBORS = 10  # Small=tighter local clusters, large=smoother global structure and less fragmentation.
PANEL4_UMAP_MIN_DIST = 0.3    # Small=more compact clusters, large=more spread out clusters.

# Optional explicit sequence IDs for panel 4. None -> auto-select.
REGION_INVARIANCE_SEQUENCE_IDS = None

# Sequence selection policy when REGION_INVARIANCE_SEQUENCE_IDS is None:
# - "random": random draw from eligible target-taxon sequences.
REGION_INVARIANCE_SEQUENCE_SELECTION_MODE = "random"

# Minimum usable region count required for a sequence to be eligible in panel 4.
REGION_INVARIANCE_MIN_NUM_REGIONS_PER_SEQUENCE = 2


# Plot/export configuration ----------------------------------------------------
# Optional point cap per panel for visual clarity and runtime control.
# If exceeded, randomly subsample and print a warning.
MAX_POINTS_PER_PANEL = None

# Save points table (coordinates + labels) for each panel/run.
SAVE_POINTS_TABLE = True

# Save raster and vector figures.
SAVE_PNG = True
SAVE_SVG = True

# Figure DPI for PNG exports.
PNG_DPI = 300

# Save run config snapshot JSON in each run directory.
SAVE_CONFIG_JSON = True

# Save panel-level metadata (filters, selected taxa/sequence IDs, region IDs).
SAVE_PANEL_METADATA_JSON = True

# Save summary TXT across panels/runs.
SAVE_SUMMARY_TXT = True


# Implementation note ----------------------------------------------------------
# Add implementation below this config block (loading, filtering, UMAP fit, plotting, saving).
# Keep behavior fully driven by constants above; do not add CLI args.


# Constants shared by implementation -------------------------------------------
_RANK_NAMES = ("domain", "phylum", "class", "order", "family", "genus", "species")
_SPLIT_TO_INDEX_FILE = {"train": "training_indices.txt", "test": "testing_indices.txt", "excluded": "excluded_taxa_indices.txt"}
_MARKERS = ("o", "s", "^", "v", "D", "P", "X", "h", "*", "<", ">", "8")
_CONTRAST_COLOR_PALETTE = (
    "#1f77b4",  # blue
    "#d62728",  # red
    "#2ca02c",  # green
    "#ff7f0e",  # orange
    "#9467bd",  # violet
    "#8c564b",  # brown
    "#e377c2",  # pink
    "#17becf",  # cyan
    "#bcbd22",  # olive
    "#7f7f7f",  # gray
)
_GLOBAL_COLOR_BY_LABEL = {}


def _set_global_seed(seed):
    # Keep every source of randomness deterministic.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _set_torch_threads(n_threads):
    # Keep CPU-side math predictable and aligned with config.
    if n_threads is None:
        return
    n_threads = int(n_threads)
    if n_threads <= 0:
        return
    torch.set_num_threads(n_threads)
    if hasattr(torch, "set_num_interop_threads"):
        try:
            torch.set_num_interop_threads(max(1, min(4, n_threads)))
        except RuntimeError:
            # Safe fallback when interop thread count is already fixed in this process.
            pass


def _sanitize_tag(value):
    # Restrict run-name chunks to safe filename characters.
    text = str(value).strip()
    if not text:
        return "na"
    chars = []
    for ch in text:
        if ch.isalnum() or ch in ("-", "_"):
            chars.append(ch)
        else:
            chars.append("_")
    out = "".join(chars).strip("_")
    return out if out else "na"


def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def _resolve_encoded_dir(dataset_split_dir):
    # DATASET_SPLIT_DIR points to .../seqs/split_xxx/, so encoded is its sibling.
    split_dir_clean = os.path.abspath(dataset_split_dir.rstrip("/"))
    seqs_dir = os.path.dirname(split_dir_clean)
    encoded_dir = os.path.join(seqs_dir, "encoded")
    if not os.path.isdir(encoded_dir):
        raise FileNotFoundError(f"Encoded directory not found: {encoded_dir}")
    return encoded_dir


def _load_indices(dataset_split_dir, split_name):
    # Read the sequence indices for the requested split.
    if split_name not in _SPLIT_TO_INDEX_FILE:
        raise ValueError(f"Unsupported split '{split_name}'. Expected one of {tuple(_SPLIT_TO_INDEX_FILE.keys())}.")
    path = os.path.join(dataset_split_dir, _SPLIT_TO_INDEX_FILE[split_name])
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing split index file: {path}")
    if os.path.getsize(path) == 0:
        return np.empty(0, dtype=np.int64)
    data = np.loadtxt(path, dtype=np.int64)
    return np.atleast_1d(data)


def _load_tax_label_dict(dataset_split_dir, split_name):
    # Taxonomy dictionary key is global sequence ID -> list of 7 labels.
    path = os.path.join(dataset_split_dir, "tax_objs", split_name, "full_tax_label_from_seq_id_dict.pkl")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing taxonomy dictionary: {path}")
    with open(path, "rb") as f:
        payload = pickle.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Taxonomy payload is not a dict: {path}")
    return payload


def _build_split_taxonomy_table(dataset_split_dir, split_name):
    # Build one row per split sequence with sequence ID + rank labels.
    seq_indices = _load_indices(dataset_split_dir, split_name)
    tax_dict = _load_tax_label_dict(dataset_split_dir, split_name)
    rows = []
    missing = 0
    bad = 0
    for seq_idx in seq_indices:
        labels = tax_dict.get(int(seq_idx))
        if labels is None:
            missing += 1
            continue
        if not isinstance(labels, (list, tuple)) or len(labels) != 7:
            bad += 1
            continue
        rows.append((int(seq_idx), str(labels[0]), str(labels[1]), str(labels[2]), str(labels[3]), str(labels[4]), str(labels[5]), str(labels[6])))
    if missing > 0:
        print(f"WARNING: {missing} sequence IDs in split '{split_name}' were missing in taxonomy dictionary.")
    if bad > 0:
        print(f"WARNING: {bad} sequence IDs in split '{split_name}' had invalid taxonomy label format.")
    if not rows:
        raise ValueError(f"No usable sequences were loaded for split '{split_name}'.")
    return pd.DataFrame(rows, columns=("seq_idx",) + _RANK_NAMES)


def _choose_device():
    # Embeddings can use CUDA, all other steps can stay on CPU.
    if USE_CUDA_IF_AVAILABLE and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _find_3bit_file(encoded_dir):
    # Prefer packed file when present because it is usually much smaller on disk.
    packed_path = os.path.join(encoded_dir, "3bit_seq_reps_packed.npy")
    plain_path = os.path.join(encoded_dir, "3bit_seq_reps.npy")
    if os.path.isfile(packed_path):
        return packed_path, True
    if os.path.isfile(plain_path):
        return plain_path, False
    raise FileNotFoundError(f"Could not find 3-bit sequence reps in {encoded_dir} (expected 3bit_seq_reps_packed.npy or 3bit_seq_reps.npy).")


def _init_3bit_cache(encoded_dir):
    # Keep a mmap view so we can gather only rows needed for each panel.
    path, packed = _find_3bit_file(encoded_dir)
    arr = np.load(path, mmap_mode="r")
    if arr.ndim != 4:
        raise ValueError(f"3-bit array must have 4 dimensions [n_regions, n_seqs, seq_len, bits], got {arr.shape} from {path}.")
    if packed and arr.shape[-1] != 1:
        raise ValueError(f"Packed 3-bit array should have trailing singleton axis, got {arr.shape} from {path}.")
    if not packed and arr.shape[-1] < 3:
        raise ValueError(f"Unpacked 3-bit array must have at least 3 bits in last axis, got {arr.shape} from {path}.")
    return {"path": path, "packed": packed, "array": arr}


def _gather_3bit_examples(cache, region_idx_arr, seq_idx_arr):
    # Use element-wise advanced indexing so each row maps to one (region, sequence).
    arr = cache["array"]
    region_idx_arr = np.asarray(region_idx_arr, dtype=np.int64)
    seq_idx_arr = np.asarray(seq_idx_arr, dtype=np.int64)
    if region_idx_arr.shape != seq_idx_arr.shape:
        raise ValueError("region_idx_arr and seq_idx_arr must have the same shape.")
    gathered = arr[region_idx_arr, seq_idx_arr]
    if cache["packed"]:
        bits = np.unpackbits(gathered, axis=-1, bitorder="big")[..., :3]
        return bits.astype(np.bool_, copy=False)
    return gathered[..., :3].astype(np.bool_, copy=False)


def _load_kmer_for_k(encoded_dir, k_value, kmer_cache):
    # K-mer tensors are loaded lazily per K and re-used across runs.
    if int(k_value) in kmer_cache:
        return kmer_cache[int(k_value)]
    path = os.path.join(encoded_dir, f"{int(k_value)}-mer_seq_reps.npy")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing k-mer tensor for k={k_value}: {path}")
    arr = np.load(path, mmap_mode="r")
    if arr.ndim != 3:
        raise ValueError(f"K-mer tensor for k={k_value} must have shape [n_regions, n_seqs, dim], got {arr.shape}.")
    kmer_cache[int(k_value)] = arr
    return arr


def _validate_config():
    # Validate toggles and enum-style settings up front.
    if EVAL_SPLIT not in _SPLIT_TO_INDEX_FILE:
        raise ValueError(f"EVAL_SPLIT must be one of {tuple(_SPLIT_TO_INDEX_FILE.keys())}, got '{EVAL_SPLIT}'.")
    if TAXON_PANEL_REGION_MODE not in ("mixed_random_per_sequence", "single_region"):
        raise ValueError("TAXON_PANEL_REGION_MODE must be 'mixed_random_per_sequence' or 'single_region'.")
    if KMER_K_SELECTION_MODE not in ("fixed", "all"):
        raise ValueError("KMER_K_SELECTION_MODE must be 'fixed' or 'all'.")
    if UMAP_N_COMPONENTS != 2:
        print("WARNING: This script and plotting helpers assume 2D UMAP output. Non-2D runs may fail plotting.")
    if RUN_DIR_SUFFIX_DIGITS < 1:
        raise ValueError("RUN_DIR_SUFFIX_DIGITS must be >= 1.")
    for backend in REPRESENTATION_BACKENDS:
        if backend not in ("embedding", "kmer"):
            raise ValueError(f"Unknown backend '{backend}'. Supported backends: ('embedding', 'kmer').")
        if backend not in UMAP_DISTANCE_METRIC_BY_BACKEND:
            raise ValueError(f"UMAP_DISTANCE_METRIC_BY_BACKEND missing key '{backend}'.")
    if MAX_POINTS_PER_PANEL is not None and int(MAX_POINTS_PER_PANEL) < 3:
        raise ValueError("MAX_POINTS_PER_PANEL must be >= 3 when set.")
    for panel_name, n_neighbors, min_dist in (
        ("panel1_top_phyla", PANEL1_UMAP_N_NEIGHBORS, PANEL1_UMAP_MIN_DIST),
        ("panel2_top_families_in_order", PANEL2_UMAP_N_NEIGHBORS, PANEL2_UMAP_MIN_DIST),
        ("panel3_target_family", PANEL3_UMAP_N_NEIGHBORS, PANEL3_UMAP_MIN_DIST),
        ("panel4_region_invariance", PANEL4_UMAP_N_NEIGHBORS, PANEL4_UMAP_MIN_DIST),
    ):
        if int(n_neighbors) < 2:
            raise ValueError(f"{panel_name} UMAP n_neighbors must be >= 2.")
        if float(min_dist) < 0.0:
            raise ValueError(f"{panel_name} UMAP min_dist must be >= 0.0.")
    if int(REGION_INVARIANCE_TARGET_TAXON_RANK) < 0 or int(REGION_INVARIANCE_TARGET_TAXON_RANK) >= len(_RANK_NAMES):
        raise ValueError(f"REGION_INVARIANCE_TARGET_TAXON_RANK must be between 0 and {len(_RANK_NAMES)-1}.")
    if int(REGION_INVARIANCE_NUM_SEQS_FOR_PANEL) <= 0:
        raise ValueError("REGION_INVARIANCE_NUM_SEQS_FOR_PANEL must be positive.")
    if REGION_INVARIANCE_SEQUENCE_SELECTION_MODE not in ("random",):
        raise ValueError("REGION_INVARIANCE_SEQUENCE_SELECTION_MODE must be 'random'.")


def _build_rep_specs():
    # Expand backend settings into explicit model/k targets in deterministic order.
    rep_specs = []
    for backend in REPRESENTATION_BACKENDS:
        if backend == "embedding":
            for variant_name, checkpoint_path in EMBEDDING_MODEL_VARIANTS.items():
                if checkpoint_path is None:
                    print(f"Skipping embedding variant '{variant_name}' because checkpoint is None.")
                    continue
                if not os.path.isfile(checkpoint_path):
                    print(f"Skipping embedding variant '{variant_name}' because checkpoint does not exist: {checkpoint_path}")
                    continue
                rep_specs.append({"backend": "embedding", "item_key": str(variant_name), "label": f"embedding-{_sanitize_tag(variant_name)}", "checkpoint_path": checkpoint_path})
        elif backend == "kmer":
            if KMER_K_SELECTION_MODE == "fixed":
                if MANUAL_KMER_K is None:
                    raise ValueError("MANUAL_KMER_K must be set when KMER_K_SELECTION_MODE == 'fixed'.")
                k_list = [int(MANUAL_KMER_K)]
            else:
                k_list = [int(k) for k in KMER_K_VALUES]
            seen = set()
            ordered_unique = []
            for k in k_list:
                if k not in seen:
                    seen.add(k)
                    ordered_unique.append(k)
            for k in ordered_unique:
                rep_specs.append({"backend": "kmer", "item_key": str(k), "label": f"kmer-k{k}", "k_value": int(k)})
    if not rep_specs:
        raise ValueError("No representation specs were produced. Check REPRESENTATION_BACKENDS, checkpoints, and K settings.")
    return rep_specs


def _build_panel_specs():
    # Keep panel traversal order stable for deterministic global run indexing.
    panel_specs = []
    if RUN_PANEL_TOP_PHYLA:
        panel_specs.append({
            "panel_key": "panel1_top_phyla",
            "title": "Top Phyla",
            "umap_n_neighbors": int(PANEL1_UMAP_N_NEIGHBORS),
            "umap_min_dist": float(PANEL1_UMAP_MIN_DIST),
        })
    if RUN_PANEL_TOP_FAMILIES_IN_ORDER:
        panel_specs.append({
            "panel_key": "panel2_top_families_in_order",
            "title": "Top Families In One Order",
            "umap_n_neighbors": int(PANEL2_UMAP_N_NEIGHBORS),
            "umap_min_dist": float(PANEL2_UMAP_MIN_DIST),
        })
    if RUN_PANEL_TARGET_FAMILY:
        panel_specs.append({
            "panel_key": "panel3_target_family",
            "title": "Target Family",
            "umap_n_neighbors": int(PANEL3_UMAP_N_NEIGHBORS),
            "umap_min_dist": float(PANEL3_UMAP_MIN_DIST),
        })
    if RUN_PANEL_REGION_INVARIANCE:
        panel_specs.append({
            "panel_key": "panel4_region_invariance",
            "title": "Region Invariance",
            "umap_n_neighbors": int(PANEL4_UMAP_N_NEIGHBORS),
            "umap_min_dist": float(PANEL4_UMAP_MIN_DIST),
        })
    if not panel_specs:
        raise ValueError("No panel toggles are enabled. Enable at least one RUN_PANEL_* setting.")
    return panel_specs


def _build_run_specs(panel_specs, rep_specs):
    # Cross product panel x representation with one global run index and seed.
    run_specs = []
    global_run_idx = 0
    for panel in panel_specs:
        for rep in rep_specs:
            run_seed = int(GLOBAL_RANDOM_SEED + global_run_idx * RUN_SEED_STRIDE)
            run_specs.append({"global_run_idx": global_run_idx, "run_seed": run_seed, "panel_key": panel["panel_key"], "panel_title": panel["title"], "umap_n_neighbors": int(panel["umap_n_neighbors"]), "umap_min_dist": float(panel["umap_min_dist"]), "backend": rep["backend"], "item_key": rep["item_key"], "label": rep["label"], "checkpoint_path": rep.get("checkpoint_path"), "k_value": rep.get("k_value")})
            global_run_idx += 1
    return run_specs


def _build_region_pools():
    # Panel 1-3 use taxon pool (full index 0 removed + optional selector), panel 4 uses all subseqs.
    all_indices_sorted = sorted(int(i) for i in REGION_IDX_TO_ID_MAPPING.keys())
    if not all_indices_sorted:
        raise ValueError("REGION_IDX_TO_ID_MAPPING is empty.")
    all_subseq_regions = [idx for idx in all_indices_sorted if idx != 0]
    if not all_subseq_regions:
        raise ValueError("No subsequence regions are available after excluding index 0.")
    taxon_pool = list(all_subseq_regions)
    if TAXON_PANEL_REGION_INDICES is not None:
        allowed = set(int(i) for i in TAXON_PANEL_REGION_INDICES)
        taxon_pool = [idx for idx in taxon_pool if idx in allowed]
    if not taxon_pool:
        raise ValueError("Taxon panel region pool is empty after applying TAXON_PANEL_REGION_IDS and full-sequence exclusion.")
    if TAXON_PANEL_REGION_MODE == "single_region":
        fixed = int(TAXON_PANEL_FIXED_REGION_INDEX)
        if fixed == 0:
            raise ValueError("TAXON_PANEL_FIXED_REGION_ID cannot be full-sequence index 0 for this evaluation.")
        if fixed not in taxon_pool:
            raise ValueError(f"TAXON_PANEL_FIXED_REGION_ID resolves to index {fixed}, which is not in the usable taxon region pool {taxon_pool}.")
    return taxon_pool, all_subseq_regions


def _assign_taxon_panel_regions(n_rows, rng, taxon_pool):
    # Panels 1-3 map one region per sequence using the configured strategy.
    if n_rows <= 0:
        return np.empty(0, dtype=np.int64)
    if TAXON_PANEL_REGION_MODE == "mixed_random_per_sequence":
        return rng.choice(np.asarray(taxon_pool, dtype=np.int64), size=n_rows, replace=True).astype(np.int64, copy=False)
    if TAXON_PANEL_REGION_MODE == "single_region":
        return np.full(n_rows, int(TAXON_PANEL_FIXED_REGION_INDEX), dtype=np.int64)
    raise ValueError(f"Unsupported TAXON_PANEL_REGION_MODE: {TAXON_PANEL_REGION_MODE}")


def _cap_panel_points(df, rng, panel_key, run_label):
    # Optional hard cap keeps UMAP runtime and plot density manageable.
    if MAX_POINTS_PER_PANEL is None:
        return df
    cap = int(MAX_POINTS_PER_PANEL)
    if len(df) <= cap:
        return df
    print(f"WARNING: {run_label} {panel_key} has {len(df)} points; subsampling to MAX_POINTS_PER_PANEL={cap}.")
    keep_positions = rng.choice(np.arange(len(df), dtype=np.int64), size=cap, replace=False)
    keep_positions = np.sort(keep_positions)
    return df.iloc[keep_positions].reset_index(drop=True)


def _make_panel1_points(split_df, rng, taxon_pool):
    # Panel 1: top N phyla in split, colored by phylum.
    rank_name = _RANK_NAMES[TOP_PHYLA_COLOR_RANK]
    counts = split_df[rank_name].value_counts()
    top_labels = counts.head(int(TOP_PHYLA_N)).index.tolist()
    if not top_labels:
        raise ValueError("Panel 1 selection produced zero phyla.")
    panel_df = split_df[split_df[rank_name].isin(top_labels)].copy().reset_index(drop=True)
    if panel_df.empty:
        raise ValueError("Panel 1 has no points after filtering.")
    panel_df["region_idx"] = _assign_taxon_panel_regions(len(panel_df), rng, taxon_pool)
    panel_df["color_label"] = panel_df[rank_name]
    panel_df["shape_label"] = ""
    panel_df["panel_name"] = "top_phyla"
    metadata = {"panel_key": "panel1_top_phyla", "color_rank": int(TOP_PHYLA_COLOR_RANK), "color_rank_name": rank_name, "selected_phyla": top_labels}
    return panel_df, metadata


def _make_panel2_points(split_df, rng, taxon_pool):
    # Panel 2: top N families within one order, family color + genus marker.
    order_rank = _RANK_NAMES[3]
    family_rank = _RANK_NAMES[TOP_FAMILIES_COLOR_RANK]
    genus_rank = _RANK_NAMES[TOP_FAMILIES_SHAPE_RANK]
    order_label = TOP_FAMILIES_ORDER_LABEL
    if order_label is None:
        order_counts = split_df[order_rank].value_counts()
        if order_counts.empty:
            raise ValueError("Panel 2 cannot auto-select an order because no order labels are present.")
        order_label = str(order_counts.index[0])
    order_df = split_df[split_df[order_rank] == str(order_label)].copy().reset_index(drop=True)
    if order_df.empty:
        raise ValueError(f"Panel 2 has no sequences for order '{order_label}'.")
    family_counts = order_df[family_rank].value_counts()
    top_families = family_counts.head(int(TOP_FAMILIES_N)).index.tolist()
    panel_df = order_df[order_df[family_rank].isin(top_families)].copy().reset_index(drop=True)
    if panel_df.empty:
        raise ValueError(f"Panel 2 has no sequences in top families within order '{order_label}'.")
    panel_df["region_idx"] = _assign_taxon_panel_regions(len(panel_df), rng, taxon_pool)
    panel_df["color_label"] = panel_df[family_rank]
    panel_df["shape_label"] = panel_df[genus_rank]
    panel_df["panel_name"] = "top_families_in_order"
    metadata = {"panel_key": "panel2_top_families_in_order", "order_label": str(order_label), "color_rank": int(TOP_FAMILIES_COLOR_RANK), "shape_rank": int(TOP_FAMILIES_SHAPE_RANK), "top_families": top_families}
    return panel_df, metadata


def _select_region_invariance_seq_ids(taxon_df, rng, all_subseq_regions):
    # Panel 4 sequence selector supports explicit IDs or deterministic random sampling.
    min_regions = int(REGION_INVARIANCE_MIN_NUM_REGIONS_PER_SEQUENCE)
    available_regions = len(all_subseq_regions)
    if available_regions < min_regions:
        raise ValueError(f"Panel 4 needs at least {min_regions} regions per sequence, but only {available_regions} subsequence regions are available.")
    taxon_seq_ids = taxon_df["seq_idx"].astype(np.int64).to_numpy()
    target_rank_name = _RANK_NAMES[int(REGION_INVARIANCE_TARGET_TAXON_RANK)]
    if taxon_seq_ids.size == 0:
        raise ValueError(
            f"Panel 4 target taxon '{REGION_INVARIANCE_TARGET_TAXON_LABEL}' at rank '{target_rank_name}' has zero sequences in split '{EVAL_SPLIT}'."
        )
    if REGION_INVARIANCE_SEQUENCE_IDS is not None:
        requested = np.array([int(x) for x in REGION_INVARIANCE_SEQUENCE_IDS], dtype=np.int64)
        allowed = set(taxon_seq_ids.tolist())
        selected = [int(x) for x in requested.tolist() if int(x) in allowed]
        if not selected:
            raise ValueError(
                f"None of REGION_INVARIANCE_SEQUENCE_IDS are present in split '{EVAL_SPLIT}' for taxon '{REGION_INVARIANCE_TARGET_TAXON_LABEL}' at rank '{target_rank_name}'."
            )
        if len(selected) > int(REGION_INVARIANCE_NUM_SEQS_FOR_PANEL):
            selected = selected[:int(REGION_INVARIANCE_NUM_SEQS_FOR_PANEL)]
        return selected
    n_wanted = int(REGION_INVARIANCE_NUM_SEQS_FOR_PANEL)
    if n_wanted <= 0:
        raise ValueError("REGION_INVARIANCE_NUM_SEQS_FOR_PANEL must be positive.")
    if taxon_seq_ids.size < n_wanted:
        raise ValueError(
            f"Panel 4 requested {n_wanted} sequences but only {taxon_seq_ids.size} are available for "
            f"taxon '{REGION_INVARIANCE_TARGET_TAXON_LABEL}' at rank '{target_rank_name}'."
        )
    if REGION_INVARIANCE_SEQUENCE_SELECTION_MODE == "random":
        chosen = rng.choice(taxon_seq_ids, size=n_wanted, replace=False)
        return [int(x) for x in chosen.tolist()]
    raise ValueError(f"Unsupported REGION_INVARIANCE_SEQUENCE_SELECTION_MODE: {REGION_INVARIANCE_SEQUENCE_SELECTION_MODE}")


def _make_panel3_points(split_df, rng, taxon_pool):
    # Panel 3: target family, color by genus.
    family_rank = _RANK_NAMES[4]
    genus_rank = _RANK_NAMES[TARGET_FAMILY_COLOR_RANK]
    panel_df = split_df[split_df[family_rank] == str(TARGET_FAMILY_LABEL)].copy().reset_index(drop=True)
    if panel_df.empty:
        raise ValueError(f"Panel 3 target family '{TARGET_FAMILY_LABEL}' has zero sequences in split '{EVAL_SPLIT}'.")
    panel_df["region_idx"] = _assign_taxon_panel_regions(len(panel_df), rng, taxon_pool)
    panel_df["color_label"] = panel_df[genus_rank]
    panel_df["shape_label"] = ""
    panel_df["panel_name"] = "target_family"
    metadata = {"panel_key": "panel3_target_family", "target_family": str(TARGET_FAMILY_LABEL), "color_rank": int(TARGET_FAMILY_COLOR_RANK), "color_rank_name": genus_rank}
    return panel_df, metadata


def _make_panel4_points(split_df, rng, all_subseq_regions):
    # Panel 4: one point per (selected sequence, subsequence region), color by sequence.
    target_rank_name = _RANK_NAMES[int(REGION_INVARIANCE_TARGET_TAXON_RANK)]
    target_label = str(REGION_INVARIANCE_TARGET_TAXON_LABEL)
    taxon_df = split_df[
        split_df[target_rank_name].astype(str) == target_label
    ].copy().reset_index(drop=True)
    selected_seq_ids = _select_region_invariance_seq_ids(taxon_df, rng, all_subseq_regions)
    source = split_df.set_index("seq_idx")
    rows = []
    for seq_idx in selected_seq_ids:
        if seq_idx not in source.index:
            continue
        row = source.loc[seq_idx]
        for region_idx in all_subseq_regions:
            rows.append((int(seq_idx), row["domain"], row["phylum"], row["class"], row["order"], row["family"], row["genus"], row["species"], int(region_idx), REGION_IDX_TO_ID_MAPPING[int(region_idx)], int(seq_idx), ""))
    if not rows:
        raise ValueError("Panel 4 produced zero points after sequence selection.")
    panel_df = pd.DataFrame(rows, columns=("seq_idx", "domain", "phylum", "class", "order", "family", "genus", "species", "region_idx", "region_id", "color_label", "shape_label"))
    panel_df["panel_name"] = "region_invariance"
    metadata = {
        "panel_key": "panel4_region_invariance",
        "target_taxon_rank": int(REGION_INVARIANCE_TARGET_TAXON_RANK),
        "target_taxon_rank_name": target_rank_name,
        "target_taxon_label": target_label,
        "selected_sequence_ids": [int(x) for x in selected_seq_ids],
        "selection_mode": str(REGION_INVARIANCE_SEQUENCE_SELECTION_MODE),
        "n_regions_per_sequence": int(len(all_subseq_regions)),
    }
    return panel_df, metadata


def _build_panel_points(panel_key, split_df, rng, taxon_pool, all_subseq_regions):
    # Dispatch table for panel builders.
    if panel_key == "panel1_top_phyla":
        return _make_panel1_points(split_df, rng, taxon_pool)
    if panel_key == "panel2_top_families_in_order":
        return _make_panel2_points(split_df, rng, taxon_pool)
    if panel_key == "panel3_target_family":
        return _make_panel3_points(split_df, rng, taxon_pool)
    if panel_key == "panel4_region_invariance":
        return _make_panel4_points(split_df, rng, all_subseq_regions)
    raise ValueError(f"Unknown panel key: {panel_key}")


def _attach_region_fields(panel_df):
    # Ensure exports always include both region_idx and region_id.
    if "region_id" in panel_df.columns:
        return panel_df
    region_ids = []
    for idx in panel_df["region_idx"].astype(np.int64).tolist():
        region_ids.append(build_region_export_fields(int(idx), REGION_IDX_TO_ID_MAPPING)["region_id"])
    panel_df = panel_df.copy()
    panel_df["region_id"] = region_ids
    return panel_df


def _load_or_get_model(checkpoint_path, device, model_cache):
    # Cache models by checkpoint path to avoid reloading weights repeatedly.
    if checkpoint_path in model_cache:
        return model_cache[checkpoint_path]
    print(f"Loading embedding model checkpoint: {checkpoint_path}")
    model = load_micro16s_model(checkpoint_path)
    model = model.to(device)
    model.eval()
    model_cache[checkpoint_path] = model
    return model


def _extract_features(run_spec, panel_df, data_cache, encoded_dir, device, model_cache, kmer_cache):
    # Turn panel point table into a matrix suitable for UMAP.
    seq_idx_arr = panel_df["seq_idx"].astype(np.int64).to_numpy()
    region_idx_arr = panel_df["region_idx"].astype(np.int64).to_numpy()
    if run_spec["backend"] == "embedding":
        model = _load_or_get_model(run_spec["checkpoint_path"], device, model_cache)
        seq_reps = _gather_3bit_examples(data_cache["three_bit"], region_idx_arr, seq_idx_arr)
        feats = run_inference(model, seq_reps, device=device, batch_size=INFERENCE_BATCH_SIZE, output_device="cpu", return_numpy=True, pin_inputs=(device.type == "cuda"))
        return np.asarray(feats, dtype=np.float32)
    if run_spec["backend"] == "kmer":
        k_value = int(run_spec["k_value"])
        arr = _load_kmer_for_k(encoded_dir, k_value, kmer_cache)
        feats = arr[region_idx_arr, seq_idx_arr, :]
        return np.asarray(feats, dtype=np.float32)
    raise ValueError(f"Unknown backend '{run_spec['backend']}'.")


def _fit_umap(features, metric, run_seed, panel_n_neighbors, panel_min_dist):
    # Fit a fresh UMAP model for each panel/backend/model-or-k combination.
    n_points = int(features.shape[0])
    if n_points < 3:
        raise ValueError(f"UMAP requires at least 3 points, got {n_points}.")
    n_neighbors = int(min(max(2, int(panel_n_neighbors)), n_points - 1))
    min_dist = float(panel_min_dist)
    reducer = umap.UMAP(n_components=UMAP_N_COMPONENTS, n_neighbors=n_neighbors, min_dist=min_dist, metric=metric, init=UMAP_INIT, n_jobs=UMAP_N_JOBS, random_state=int(run_seed))
    coords = reducer.fit_transform(features)
    if coords.ndim != 2 or coords.shape[1] < 2:
        raise ValueError(f"UMAP output has unexpected shape {coords.shape}.")
    return np.asarray(coords[:, :2], dtype=np.float32), n_neighbors, min_dist


def _build_color_map(values):
    # Reuse one fixed high-contrast palette across all plots.
    labels = sorted(set(str(v) for v in values))
    for label in labels:
        if label in _GLOBAL_COLOR_BY_LABEL:
            continue
        color_idx = len(_GLOBAL_COLOR_BY_LABEL) % len(_CONTRAST_COLOR_PALETTE)
        _GLOBAL_COLOR_BY_LABEL[label] = _CONTRAST_COLOR_PALETTE[color_idx]
    return {label: _GLOBAL_COLOR_BY_LABEL[label] for label in labels}


def _plot_panel(run_spec, points_df, run_dir):
    # Draw a panel plot with simple styling and readable legends.
    fig, ax = plt.subplots(figsize=(10, 8))
    panel_key = run_spec["panel_key"]
    if panel_key == "panel2_top_families_in_order":
        family_colors = _build_color_map(points_df["color_label"].astype(str).tolist())
        genera = pd.Series(points_df["shape_label"].astype(str).tolist(), dtype=str).value_counts().index.tolist()
        genus_markers = {label: "o" for label in genera} if ALL_SHAPES_DOTS else {label: _MARKERS[i % len(_MARKERS)] for i, label in enumerate(genera)}
        for (family, genus), subset in points_df.groupby(["color_label", "shape_label"], sort=False):
            ax.scatter(subset["umap_x"].to_numpy(), subset["umap_y"].to_numpy(), s=14, alpha=0.78, c=[family_colors[str(family)]], marker=genus_markers[str(genus)], linewidths=0)
        if len(family_colors) <= 25:
            family_handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=family_colors[label], markersize=6, label=label) for label in family_colors.keys()]
            legend1 = ax.legend(handles=family_handles, title="Family", loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
            ax.add_artist(legend1)
        if not ALL_SHAPES_DOTS and len(genus_markers) <= 18:
            genus_handles = [Line2D([0], [0], marker=genus_markers[label], color="black", linestyle="", markersize=6, label=label) for label in genus_markers.keys()]
            ax.legend(handles=genus_handles, title="Genus", loc="upper left", bbox_to_anchor=(1.01, 0.5), frameon=False)
    else:
        colors = _build_color_map(points_df["color_label"].astype(str).tolist())
        color_values = [colors[str(v)] for v in points_df["color_label"].astype(str).tolist()]
        ax.scatter(points_df["umap_x"].to_numpy(), points_df["umap_y"].to_numpy(), s=14, alpha=0.78, c=color_values, linewidths=0)
        if len(colors) <= 25:
            handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=color, markersize=6, label=label) for label, color in colors.items()]
            legend_title = "Sequence" if panel_key == "panel4_region_invariance" else "Label"
            ax.legend(handles=handles, title=legend_title, loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.set_title(f"{run_spec['panel_title']} | {run_spec['label']} | split={EVAL_SPLIT}")
    ax.grid(False)
    fig.tight_layout()
    if SAVE_PNG:
        fig.savefig(os.path.join(run_dir, "plot.png"), dpi=PNG_DPI, bbox_inches="tight")
    if SAVE_SVG:
        fig.savefig(os.path.join(run_dir, "plot.svg"), bbox_inches="tight")
    plt.close(fig)


def _to_jsonable(value):
    # Convert arbitrary config payloads into JSON-safe values.
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, set):
        return sorted(_to_jsonable(v) for v in value)
    if isinstance(value, dict):
        out = {}
        for key, val in value.items():
            out[str(key)] = _to_jsonable(val)
        return out
    if isinstance(value, torch.device):
        return str(value)
    return str(value)


def _collect_config_snapshot():
    # Save only uppercase names from this module as config.
    snapshot = {}
    for name, value in globals().items():
        if name.isupper():
            snapshot[name] = _to_jsonable(value)
    return snapshot


def _make_run_dir(output_root, run_spec, timestamp_str):
    # Name format: prefix__timestamp__panel__backend_item__[tags]__NNN
    base_parts = [RUN_DIR_PREFIX, timestamp_str, _sanitize_tag(run_spec["panel_key"]), _sanitize_tag(run_spec["label"])]
    for tag in RUN_NAME_EXTRA_TAGS:
        base_parts.append(_sanitize_tag(tag))
    base = "__".join(base_parts)
    for i in range(1, 10 ** RUN_DIR_SUFFIX_DIGITS):
        candidate_name = f"{base}__{i:0{RUN_DIR_SUFFIX_DIGITS}d}"
        candidate_path = os.path.join(output_root, candidate_name)
        if not os.path.exists(candidate_path):
            _ensure_dir(candidate_path)
            return candidate_path
    raise RuntimeError(f"Could not allocate a unique run directory for base '{base}'.")


def _write_points_table(points_df, run_dir):
    # Keep point export order fixed and explicit.
    export_cols = ("seq_idx", "region_idx", "region_id", "domain", "phylum", "class", "order", "family", "genus", "species", "color_label", "shape_label", "umap_x", "umap_y")
    to_save = points_df.loc[:, export_cols].copy()
    to_save.to_csv(os.path.join(run_dir, "points.tsv"), sep="\t", index=False)


def _write_panel_summary(summary_row, run_dir):
    # One TSV row per run for quick downstream parsing.
    pd.DataFrame([summary_row]).to_csv(os.path.join(run_dir, "panel_summary.tsv"), sep="\t", index=False)


def _write_json(path, payload):
    # Small helper for config/metadata payloads.
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def _run_single_eval(run_spec, split_df, taxon_pool, all_subseq_regions, data_cache, encoded_dir, device, model_cache, kmer_cache, output_root, timestamp_str, config_snapshot):
    # Execute one panel/backend/model-or-k run from start to finish.
    run_label = f"run#{run_spec['global_run_idx']} seed={run_spec['run_seed']} {run_spec['panel_key']} {run_spec['label']}"
    print(f"Starting {run_label}")
    rng = np.random.default_rng(int(run_spec["run_seed"]))
    panel_df, panel_metadata = _build_panel_points(run_spec["panel_key"], split_df, rng, taxon_pool, all_subseq_regions)
    panel_df = _attach_region_fields(panel_df)
    panel_df = _cap_panel_points(panel_df, rng, run_spec["panel_key"], run_label)
    if panel_df.empty:
        raise ValueError("Panel had no points after MAX_POINTS_PER_PANEL filtering.")
    features = _extract_features(run_spec, panel_df, data_cache, encoded_dir, device, model_cache, kmer_cache)
    metric = str(UMAP_DISTANCE_METRIC_BY_BACKEND[run_spec["backend"]])
    coords, used_neighbors, used_min_dist = _fit_umap(features, metric, int(run_spec["run_seed"]), int(run_spec["umap_n_neighbors"]), float(run_spec["umap_min_dist"]))
    panel_df = panel_df.copy()
    panel_df["umap_x"] = coords[:, 0]
    panel_df["umap_y"] = coords[:, 1]
    run_dir = _make_run_dir(output_root, run_spec, timestamp_str)
    if SAVE_POINTS_TABLE:
        _write_points_table(panel_df, run_dir)
    _plot_panel(run_spec, panel_df, run_dir)
    summary_row = {"global_run_idx": int(run_spec["global_run_idx"]), "run_seed": int(run_spec["run_seed"]), "panel_key": str(run_spec["panel_key"]), "panel_title": str(run_spec["panel_title"]), "backend": str(run_spec["backend"]), "item_key": str(run_spec["item_key"]), "label": str(run_spec["label"]), "split": str(EVAL_SPLIT), "umap_metric": metric, "umap_n_neighbors": int(used_neighbors), "umap_min_dist": float(used_min_dist), "umap_init": str(UMAP_INIT), "n_points": int(len(panel_df)), "n_labels": int(panel_df["color_label"].astype(str).nunique()), "n_unique_sequences": int(panel_df["seq_idx"].nunique()), "n_unique_regions": int(panel_df["region_idx"].nunique()), "run_dir": run_dir, "status": "ok", "error": ""}
    _write_panel_summary(summary_row, run_dir)
    if SAVE_PANEL_METADATA_JSON:
        _write_json(os.path.join(run_dir, "panel_metadata.json"), {"run_spec": _to_jsonable(run_spec), "panel_metadata": _to_jsonable(panel_metadata), "taxon_panel_region_pool": _to_jsonable(taxon_pool), "all_subsequence_regions": _to_jsonable(all_subseq_regions)})
    if SAVE_CONFIG_JSON:
        _write_json(os.path.join(run_dir, "config_snapshot.json"), config_snapshot)
    print(f"Finished {run_label} -> {run_dir}")
    return summary_row


def _write_root_outputs(output_root, all_summary_rows, started_at, finished_at):
    # Save aggregate tables and optional text summary at output root.
    summary_df = pd.DataFrame(all_summary_rows)
    if not summary_df.empty:
        summary_df.to_csv(os.path.join(output_root, "all_panel_summaries.tsv"), sep="\t", index=False)
    if not SAVE_SUMMARY_TXT:
        return
    total = len(all_summary_rows)
    ok = sum(1 for row in all_summary_rows if row.get("status") == "ok")
    failed = total - ok
    lines = []
    lines.append("Micro16S UMAP evaluation summary")
    lines.append(f"Started: {started_at}")
    lines.append(f"Finished: {finished_at}")
    lines.append(f"Split: {EVAL_SPLIT}")
    lines.append(f"Total runs: {total}")
    lines.append(f"Successful: {ok}")
    lines.append(f"Failed: {failed}")
    lines.append(f"Output root: {os.path.abspath(output_root)}")
    lines.append("")
    if failed > 0:
        lines.append("Failed runs:")
        for row in all_summary_rows:
            if row.get("status") != "ok":
                lines.append(f"- idx={row.get('global_run_idx')} panel={row.get('panel_key')} label={row.get('label')} error={row.get('error')}")
    with open(os.path.join(output_root, "summary.txt"), "w") as f:
        f.write("\n".join(lines).rstrip() + "\n")


def main():
    # High-level script orchestration with no CLI args by design.
    t0 = time.time()
    started_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("Micro16S UMAP evaluation starting...")
    _validate_config()
    _set_global_seed(int(GLOBAL_RANDOM_SEED))
    _set_torch_threads(NUM_WORKERS)
    device = _choose_device()
    print(f"Using device: {device}")
    split_df = _build_split_taxonomy_table(DATASET_SPLIT_DIR, EVAL_SPLIT)
    print(f"Loaded split '{EVAL_SPLIT}' with {len(split_df)} sequences.")
    taxon_pool, all_subseq_regions = _build_region_pools()
    encoded_dir = _resolve_encoded_dir(DATASET_SPLIT_DIR)
    data_cache = {"three_bit": _init_3bit_cache(encoded_dir)}
    panel_specs = _build_panel_specs()
    rep_specs = _build_rep_specs()
    run_specs = _build_run_specs(panel_specs, rep_specs)
    output_root = _ensure_dir(OUTPUT_ROOT_DIR)
    timestamp_str = datetime.now().strftime(RUN_DIR_TIMESTAMP_FORMAT)
    config_snapshot = _collect_config_snapshot()
    model_cache = {}
    kmer_cache = {}
    all_summary_rows = []
    for run_spec in run_specs:
        try:
            summary_row = _run_single_eval(run_spec, split_df, taxon_pool, all_subseq_regions, data_cache, encoded_dir, device, model_cache, kmer_cache, output_root, timestamp_str, config_snapshot)
            all_summary_rows.append(summary_row)
        except Exception as exc:
            msg = str(exc)
            print(f"ERROR in run#{run_spec['global_run_idx']} ({run_spec['panel_key']} | {run_spec['label']}): {msg}")
            all_summary_rows.append({"global_run_idx": int(run_spec["global_run_idx"]), "run_seed": int(run_spec["run_seed"]), "panel_key": str(run_spec["panel_key"]), "panel_title": str(run_spec["panel_title"]), "backend": str(run_spec["backend"]), "item_key": str(run_spec["item_key"]), "label": str(run_spec["label"]), "split": str(EVAL_SPLIT), "umap_metric": str(UMAP_DISTANCE_METRIC_BY_BACKEND.get(run_spec["backend"], "")), "umap_n_neighbors": int(run_spec["umap_n_neighbors"]), "umap_min_dist": float(run_spec["umap_min_dist"]), "umap_init": str(UMAP_INIT), "n_points": 0, "n_labels": 0, "n_unique_sequences": 0, "n_unique_regions": 0, "run_dir": "", "status": "failed", "error": msg})
    finished_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    _write_root_outputs(output_root, all_summary_rows, started_at, finished_at)
    elapsed = time.time() - t0
    n_ok = sum(1 for row in all_summary_rows if row.get("status") == "ok")
    print(f"Done. Successful runs: {n_ok}/{len(all_summary_rows)}. Elapsed: {elapsed:.2f}s. Output root: {os.path.abspath(output_root)}")


if __name__ == "__main__":
    main()
