"""
Micro16S Dataset Loader Module

This module streamlines the process of loading a Micro16S training dataset.
Its primary function, `load_micro16s_dataset`, takes the path to a specific dataset
directory (containing index files and taxonomic object pickles) and populates a
centralized set of global variables defined in the `globals_config` module.

Crucially, this loading process is designed to happen **once**. After `load_micro16s_dataset`
completes successfully, the populated global variables in `globals_config` should be
treated as **read-only constants** by the rest of the application.

By populating the `globals_config` variables, this module makes the entire dataset
readily available to other parts of the application (like model training, triplet/pair
selection, and evaluation) simply by importing `globals_config`. This avoids the need
to pass numerous dataset components as arguments and ensures that the dataset is loaded
only once, preventing redundancy.

An optional taxa cap (TAXA_CAP_MAX_NUM_TAXA) can be configured to limit the dataset to
the top N most populous taxa at a given rank. When enabled, sequences outside the selected
taxa are filtered out of the training, testing, and excluded sets after initial loading.


Assumed Directory Structure:

seqs/                        <-- DATABASE_DIR
    |-- encoded/             <-- ENCODED_SEQS_DIR
        |-- 3bit_seq_reps.npy (or 3bit_seq_reps_packed.npy)
        |-- {K}-mer_seq_reps.npy (where K is a positive integer - there could be any number of these)
    |-- split_{n}/           <-- DATASET_SPLIT_DIR
        |-- excluded_taxa_indices.txt
        |-- testing_indices.txt
        |-- training_indices.txt
        |-- tax_objs/
        |   |-- train/
        |   |   |-- full_tax_label_from_seq_id_dict.pkl
        |   |   |-- list_of_seq_indices_in_taxon_at_rank_dict.pkl
        |   |   |-- list_of_taxon_labels_in_taxon_at_rank_dict.pkl
        |   |   |-- list_of_taxon_labels_at_rank_dict.pkl
        |   |   |-- nested_list_of_seq_indices.pkl
        |   |   |-- nested_dicts_of_taxa.pkl
        |   |   |-- taxon_label_to_taxon_id.pkl
        |   |   └── taxon_id_to_taxon_label.pkl
        |   |-- test/
        |   |   └── (the equivalent .pkl files)
        |   └── excluded/
        |       └── (the equivalent .pkl files)
        └── labels/
        |   |-- train/
        |   |-- seq_taxon_ids.npy
        |   |-- pairwise_ranks.npy
        |   |-- pairwise_pos_masks.npy
        |   |-- pairwise_neg_masks.npy
        |   |-- pairwise_mrca_taxon_ids.npy
        |   |-- pairwise_distances.npy
        |   |-- distances_lookup_array.npy
        |   └── distance_between_domains.npy
        |-- test/
        |   └── (the equivalent .npy files)
        └── excluded/
            └── (the equivalent .npy files)

# Note: The RedTree files loaded by load_red_trees are assumed to be at fixed paths
# relative to the execution directory or accessible via the Python path by default:
# /home/haig/Repos/micro16s/redvals/decorated_trees/ar53_r226_decorated.pkl
# /home/haig/Repos/micro16s/redvals/decorated_trees/bac120_r226_decorated.pkl
# /home/haig/Repos/micro16s/redvals/taxon_mappings/taxon_to_node_mapping_r226.pkl

"""

# Imports
import numpy as np
import os
import pickle
import re
import gc
import math
import time

# Local Imports
from redvals.redvals import RedTree
from encode_seqs import read_3bit_seq_reps
import globals_config as gb
from utils import parent_dir
from quick_test import load_kmer_qt_cache


_RANK_PREFIXES = {
    0: "d__",
    1: "p__",
    2: "c__",
    3: "o__",
    4: "f__",
    5: "g__",
    6: "s__",
}


def _get_taxon_id_at_rank(taxon_label_to_taxon_id, rank, taxon_label):
    """
    Resolve a taxon ID from the mapping using rank-qualified keys, with legacy fallback.
    """
    ranked_key = f"{_RANK_PREFIXES[rank]}{taxon_label}"
    if ranked_key in taxon_label_to_taxon_id:
        return taxon_label_to_taxon_id[ranked_key]
    if taxon_label in taxon_label_to_taxon_id:
        return taxon_label_to_taxon_id[taxon_label]
    raise KeyError(
        f"Could not find taxon ID for rank={rank} taxon='{taxon_label}'. "
        f"Tried keys '{ranked_key}' and '{taxon_label}'."
    )


def _validate_rank_qualified_taxon_mapping(taxon_label_to_taxon_id, list_of_taxon_labels_at_rank_dict, split_name="train"):
    """
    Guard against legacy unqualified mappings that can corrupt MRCA distance lookup.
    """
    ranked_prefixes = tuple(_RANK_PREFIXES.values())
    has_ranked_keys = any(
        isinstance(key, str) and key.startswith(ranked_prefixes)
        for key in taxon_label_to_taxon_id.keys()
    )
    if has_ranked_keys:
        return

    # Legacy mapping style detected (plain labels only). This is safe only when labels are
    # unique across ranks. If labels repeat across ranks, IDs can be overwritten.
    label_to_first_rank = {}
    n_collisions = 0
    for rank in sorted(list_of_taxon_labels_at_rank_dict.keys()):
        for taxon_label in list_of_taxon_labels_at_rank_dict[rank]:
            first_rank = label_to_first_rank.get(taxon_label)
            if first_rank is None:
                label_to_first_rank[taxon_label] = rank
            elif first_rank != rank:
                n_collisions += 1

    if n_collisions > 0:
        raise ValueError(
            f"Legacy unqualified taxon mapping detected for split '{split_name}' with "
            f"{n_collisions} cross-rank label collisions. Rebuild this dataset split "
            f"with the patched construct_dataset.py (rank-qualified taxon IDs)."
        )


def _limit_region_axis(array, max_num_subseqs, array_label, log_change=False):
    """
    Limit the number of subsequence regions retained along axis 0 (region axis).
    
    Args:
        array (np.ndarray): Tensor with regions on axis 0.
        max_num_subseqs (int or None): Maximum subsequences to keep (excluding the full sequence at index 0).
        array_label (str): Description used when logging changes.
        log_change (bool): Print a message when the array is truncated.
    
    Returns:
        np.ndarray: Array with at most (1 + max_num_subseqs) regions.
    """
    if array is None or max_num_subseqs is None:
        return array
    if array.shape[0] <= 1:
        return array
    subseq_count = array.shape[0] - 1
    if subseq_count <= max_num_subseqs:
        return array
    n_regions_to_keep = 1 + max_num_subseqs
    if log_change:
        print(f"Limiting {array_label} to {max_num_subseqs} subseqs from {subseq_count} (not including full sequences)")
    return array[:n_regions_to_keep]


def _load_indices_file(path):
    """Load a newline-delimited index file, allowing for empty files."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Index file not found: {path}")
    if os.path.getsize(path) == 0:
        return np.empty(0, dtype=np.int64)
    data = np.loadtxt(path, dtype=np.int64)
    return np.atleast_1d(data)


def compute_train_taxon_counts_and_baseline(seq_taxon_ids):
    """
    Compute per-sequence taxon sizes and per-rank baseline taxon sizes for the training set.

    Baselines are computed per-taxon (not per-seq) using the configured statistic.
    Invalid taxon IDs (< 0) raise an error.
    
    Uses:
        gb.TAXON_SIZE_MINING_BIAS_BASELINE_STAT ("median", "mean", or "percentile")
        gb.TAXON_SIZE_MINING_BIAS_BASELINE_PERCENTILE (only when percentile is chosen)
    """
    if seq_taxon_ids is None:
        raise ValueError("seq_taxon_ids is None. Cannot compute taxon counts.")
    if len(seq_taxon_ids.shape) != 2 or seq_taxon_ids.shape[1] != 7:
        raise ValueError(f"seq_taxon_ids must have shape (n_seqs, 7), got {seq_taxon_ids.shape}")

    n_seqs, n_ranks = seq_taxon_ids.shape
    counts_per_seq = np.zeros((n_seqs, n_ranks), dtype=np.int32)
    baseline_per_rank = np.zeros(n_ranks, dtype=np.float32)

    rank_labels = ["domain", "phylum", "class", "order", "family", "genus", "species"]
    baseline_stat = getattr(gb, "TAXON_SIZE_MINING_BIAS_BASELINE_STAT", None) or "median"
    baseline_stat = baseline_stat.lower()
    baseline_percentile = getattr(gb, "TAXON_SIZE_MINING_BIAS_BASELINE_PERCENTILE", None)
    if baseline_percentile is None:
        baseline_percentile = 50.0
    if baseline_stat == "percentile":
        if not isinstance(baseline_percentile, (int, float)):
            raise ValueError("TAXON_SIZE_MINING_BIAS_BASELINE_PERCENTILE must be a number.")
        if not (0.0 <= float(baseline_percentile) <= 100.0):
            raise ValueError("TAXON_SIZE_MINING_BIAS_BASELINE_PERCENTILE must be in [0, 100].")

    for rank in range(n_ranks):
        ids = seq_taxon_ids[:, rank]
        valid_mask = ids >= 0

        if not np.all(valid_mask):
            n_invalid = int(np.count_nonzero(~valid_mask))
            raise ValueError(f"Invalid taxon IDs at rank '{rank_labels[rank]}' (count={n_invalid}).")

        if not np.any(valid_mask):
            print(f"WARNING: No valid taxon IDs found at rank '{rank_labels[rank]}'. Setting baseline to 0.")
            continue

        valid_ids = ids[valid_mask]
        counts = np.bincount(valid_ids)
        counts_per_seq[valid_mask, rank] = counts[valid_ids].astype(np.int32)

        nonzero_counts = counts[counts > 0]
        if nonzero_counts.size == 0:
            print(f"WARNING: No non-zero taxon counts found at rank '{rank_labels[rank]}'. Setting baseline to 0.")
            continue

        if baseline_stat == "median":
            baseline_per_rank[rank] = float(np.median(nonzero_counts))
        elif baseline_stat == "mean":
            baseline_per_rank[rank] = float(np.mean(nonzero_counts))
        elif baseline_stat == "percentile":
            baseline_per_rank[rank] = float(np.percentile(nonzero_counts, baseline_percentile))
        else:
            raise ValueError(f"Unsupported TAXON_SIZE_MINING_BIAS_BASELINE_STAT: {baseline_stat}")

    return counts_per_seq, baseline_per_rank


def apply_taxa_cap():
    """
    Apply the taxa cap filter to limit the number of taxa in the dataset.

    When TAXA_CAP_MAX_NUM_TAXA is set (not None), this function filters the training,
    testing, and excluded sets to only include sequences belonging to the top N taxa
    (ranked by training-set count) at the rank specified by TAXA_CAP_RANK.

    This must be called after taxonomy objects and label arrays are loaded, but before
    derived quantities (adjusted distances, taxon counts, etc.) are computed.
    ALL_3BIT_SEQ_REPS and ALL_KMER_SEQ_REPS must still be available for re-slicing.
    """

    cap = gb.TAXA_CAP_MAX_NUM_TAXA
    if cap is None:
        return

    rank = gb.TAXA_CAP_RANK
    verbose = gb.TAXA_CAP_VERBOSE
    rank_names_plural = ["Domains", "Phyla", "Classes", "Orders", "Families", "Genera"]

    if verbose:
        t0 = time.time()
        print(f"\nApplying taxa cap...")
        print(f"  Selecting a maximum of {cap} {rank_names_plural[rank]}")

    # Count training sequences per taxon at the cap rank
    taxon_counts = {}
    for seq_i in gb.TRAINING_INDICES:
        label = gb.TRAIN_FULL_TAX_LABEL_FROM_SEQ_ID_DICT[seq_i][rank]
        taxon_counts[label] = taxon_counts.get(label, 0) + 1

    n_taxa_before = len(taxon_counts)
    counts_before = list(taxon_counts.values())

    # Select top N taxa by training-set count
    sorted_taxa = sorted(taxon_counts.items(), key=lambda x: x[1], reverse=True)
    top_taxa = set(t for t, _ in sorted_taxa[:cap])

    if verbose:
        print(f"  Taxa at rank before filter: {n_taxa_before}")
        print(f"  Seqs/taxon before: min={min(counts_before)}, median={int(np.median(counts_before))}, max={max(counts_before)}")

    # If the cap is >= the number of existing taxa, no filtering needed
    if len(top_taxa) >= n_taxa_before:
        if verbose:
            print(f"  No filtering needed (cap {cap} >= existing {n_taxa_before})")
        return

    # Build keep masks for each partition
    train_keep_mask = np.array([
        gb.TRAIN_FULL_TAX_LABEL_FROM_SEQ_ID_DICT[seq_i][rank] in top_taxa
        for seq_i in gb.TRAINING_INDICES
    ], dtype=bool)
    train_keep_indices = np.where(train_keep_mask)[0]

    test_keep_mask = np.array([
        gb.TEST_FULL_TAX_LABEL_FROM_SEQ_ID_DICT[seq_i][rank] in top_taxa
        for seq_i in gb.TESTING_INDICES
    ], dtype=bool)

    if len(gb.EXCLUDED_TAXA_INDICES) > 0:
        excl_keep_mask = np.array([
            gb.EXCLUDED_FULL_TAX_LABEL_FROM_SEQ_ID_DICT[seq_i][rank] in top_taxa
            for seq_i in gb.EXCLUDED_TAXA_INDICES
        ], dtype=bool)
    else:
        excl_keep_mask = np.ones(0, dtype=bool)

    # Filter indices
    gb.TRAINING_INDICES = gb.TRAINING_INDICES[train_keep_mask]
    gb.TESTING_INDICES = gb.TESTING_INDICES[test_keep_mask]
    gb.EXCLUDED_TAXA_INDICES = gb.EXCLUDED_TAXA_INDICES[excl_keep_mask]
    gb.HAS_EXCLUDED_SET = gb.EXCLUDED_TAXA_INDICES.size > 0

    # Ensure we still have training sequences
    if len(gb.TRAINING_INDICES) == 0:
        raise ValueError("Taxa cap filtered out all training sequences.")

    # Re-slice 3-bit representations
    gb.TRAINING_3BIT_SEQ_REPS = gb.ALL_3BIT_SEQ_REPS[:, gb.TRAINING_INDICES, :, :]
    gb.TRAINING_3BIT_SEQ_REPS_TRANSPOSED = gb.TRAINING_3BIT_SEQ_REPS.transpose(1, 0, 2, 3)
    gb.TESTING_3BIT_SEQ_REPS = gb.ALL_3BIT_SEQ_REPS[:, gb.TESTING_INDICES, :, :]
    gb.EXCLUDED_3BIT_SEQ_REPS = gb.ALL_3BIT_SEQ_REPS[:, gb.EXCLUDED_TAXA_INDICES, :, :]

    # Re-slice k-mer representations
    for k in list(gb.ALL_KMER_SEQ_REPS.keys()):
        gb.TRAINING_KMER_SEQ_REPS[k] = gb.ALL_KMER_SEQ_REPS[k][:, gb.TRAINING_INDICES, :]
        gb.TESTING_KMER_SEQ_REPS[k] = gb.ALL_KMER_SEQ_REPS[k][:, gb.TESTING_INDICES, :]
        gb.EXCLUDED_KMER_SEQ_REPS[k] = gb.ALL_KMER_SEQ_REPS[k][:, gb.EXCLUDED_TAXA_INDICES, :]

    # Rebuild full taxonomic labels
    gb.TRAIN_FULL_TAX_LABELS = [gb.TRAIN_FULL_TAX_LABEL_FROM_SEQ_ID_DICT[seq_i] for seq_i in gb.TRAINING_INDICES]
    gb.TEST_FULL_TAX_LABELS = [gb.TEST_FULL_TAX_LABEL_FROM_SEQ_ID_DICT[seq_i] for seq_i in gb.TESTING_INDICES]
    gb.EXCLUDED_FULL_TAX_LABELS = [gb.EXCLUDED_FULL_TAX_LABEL_FROM_SEQ_ID_DICT[seq_i] for seq_i in gb.EXCLUDED_TAXA_INDICES]

    # Filter training label arrays (indexed by local position within the training partition)
    gb.TRAIN_PAIRWISE_RANKS = gb.TRAIN_PAIRWISE_RANKS[np.ix_(train_keep_indices, train_keep_indices)]
    gb.TRAIN_SEQ_TAXON_IDS = gb.TRAIN_SEQ_TAXON_IDS[train_keep_indices]
    if gb.TRAIN_PAIRWISE_DISTANCES is not None:
        gb.TRAIN_PAIRWISE_DISTANCES = gb.TRAIN_PAIRWISE_DISTANCES[np.ix_(train_keep_indices, train_keep_indices)]

    # Rebuild list_of_taxon_labels_at_rank_dict for each partition
    for r in range(7):
        gb.TRAIN_LIST_OF_TAXON_LABELS_AT_RANK_DICT[r] = sorted(set(lbl[r] for lbl in gb.TRAIN_FULL_TAX_LABELS))
    if gb.TEST_FULL_TAX_LABELS:
        for r in range(7):
            gb.TEST_LIST_OF_TAXON_LABELS_AT_RANK_DICT[r] = sorted(set(lbl[r] for lbl in gb.TEST_FULL_TAX_LABELS))
    if gb.EXCLUDED_FULL_TAX_LABELS:
        for r in range(7):
            gb.EXCLUDED_LIST_OF_TAXON_LABELS_AT_RANK_DICT[r] = sorted(set(lbl[r] for lbl in gb.EXCLUDED_FULL_TAX_LABELS))

    if verbose:
        n_taxa_after = len(top_taxa)
        kept_counts = sorted([taxon_counts[t] for t in top_taxa], reverse=True)
        print(f"  Taxa at rank after filter: {n_taxa_after}")
        print(f"  Seqs/taxon after: min={min(kept_counts)}, median={int(np.median(kept_counts))}, max={max(kept_counts)}")
        print(f"  Training seqs: {len(train_keep_mask)} -> {int(train_keep_mask.sum())}")
        print(f"  Testing seqs: {len(test_keep_mask)} -> {int(test_keep_mask.sum())}")
        if len(gb.TESTING_INDICES) == 0:
            print("  NOTICE: No testing sequences remain after taxa cap. Test-set quick tests will be skipped.")
        if excl_keep_mask.size > 0:
            print(f"  Excluded seqs: {len(excl_keep_mask)} -> {int(excl_keep_mask.sum())}")
        print(f"  Time: {time.time() - t0:.2f}s")


def unload_micro16s_dataset():
    """Release dataset globals so repeated training runs don't carry stale arrays."""

    if not gb.DATASET_IS_LOADED:
        return
    gb.HAS_EXCLUDED_SET = False

    # List the heavy globals we want to drop between runs
    array_like_attrs = [
        "EXCLUDED_TAXA_INDICES", "TESTING_INDICES", "TRAINING_INDICES", "EXCLUDED_3BIT_SEQ_REPS", "TESTING_3BIT_SEQ_REPS", "TRAINING_3BIT_SEQ_REPS", "TRAINING_3BIT_SEQ_REPS_TRANSPOSED", 
        "EXCLUDED_FULL_TAX_LABELS", "TEST_FULL_TAX_LABELS", "TRAIN_FULL_TAX_LABELS", "TRAIN_SEQ_TAXON_IDS", "TRAIN_PAIRWISE_RANKS", "TRAIN_PAIRWISE_POS_MASKS", "TRAIN_PAIRWISE_NEG_MASKS", 
        "TRAIN_PAIRWISE_MRCA_TAXON_IDS", "ADJUSTED_TRAIN_PAIRWISE_DISTANCES", "TRAIN_DISTANCES_LOOKUP_ARRAY", "TRAIN_DISTANCE_BETWEEN_DOMAINS", "TRAIN_TAXON_COUNTS_PER_SEQ", "TRAIN_TAXON_BASELINE_COUNT_PER_RANK", "ALL_KMER_SEQ_REPS", 
        "EXCLUDED_KMER_SEQ_REPS", "TESTING_KMER_SEQ_REPS", "TRAINING_KMER_SEQ_REPS", "MEAN_PER_KMERS", "STD_PER_KMERS", "TRAIN_TAXON_LABEL_TO_TAXON_ID", "TRAIN_TAXON_ID_TO_TAXON_LABEL", 
        "TEST_TAXON_LABEL_TO_TAXON_ID", "EXCLUDED_TAXON_LABEL_TO_TAXON_ID", "RED_TREES",
        "TRAIN_BACTERIA_INDICES", "TRAIN_ARCHAEA_INDICES"
        ]
    for attr in array_like_attrs:
        if hasattr(gb, attr):
            setattr(gb, attr, None)

    # Reset caches / helper dicts
    gb.ORDINATION_LABEL_COLOR_CACHE = {}
    gb.KMER_QT_CACHE = None

    # Flag as unloaded and let GC reclaim things promptly
    gb.DATASET_IS_LOADED = False
    gc.collect()


def load_micro16s_dataset(dataset_split_dir, needed_k_values):
    """
    Ensure that all the shared globals for dataset loading are loaded.

    If a taxa cap is configured (TAXA_CAP_MAX_NUM_TAXA is not None), sequences are
    filtered after loading to include only those belonging to the top N taxa at the
    rank specified by TAXA_CAP_RANK, ranked by training-set count.
    """

    # Use the globals from globals
    if not gb.DATASET_IS_LOADED:
        print("\nLoading dataset --------------------------")

        # Set the directories
        gb.DATABASE_DIR = parent_dir(dataset_split_dir)
        gb.ENCODED_SEQS_DIR = os.path.join(gb.DATABASE_DIR, "encoded")

        # Load the k-mer quick test cache
        gb.KMER_QT_CACHE = load_kmer_qt_cache(dataset_split_dir)

        # Load the RedTrees object
        gb.RED_TREES = load_red_trees()
        
        # Load indices from text files
        gb.EXCLUDED_TAXA_INDICES = _load_indices_file(os.path.join(dataset_split_dir, "excluded_taxa_indices.txt"))
        gb.TESTING_INDICES = _load_indices_file(os.path.join(dataset_split_dir, "testing_indices.txt"))
        gb.TRAINING_INDICES = _load_indices_file(os.path.join(dataset_split_dir, "training_indices.txt"))
        gb.HAS_EXCLUDED_SET = gb.EXCLUDED_TAXA_INDICES.size > 0

        # Set the database directory and 3-bit representations path
        if os.path.exists(os.path.join(gb.ENCODED_SEQS_DIR, "3bit_seq_reps_packed.npy")):
            gb.ALL_3BIT_SEQ_REPS_PATH = os.path.join(gb.ENCODED_SEQS_DIR, "3bit_seq_reps_packed.npy")
        elif os.path.exists(os.path.join(gb.ENCODED_SEQS_DIR, "3bit_seq_reps.npy")):
            gb.ALL_3BIT_SEQ_REPS_PATH = os.path.join(gb.ENCODED_SEQS_DIR, "3bit_seq_reps.npy")
        else:
            raise FileNotFoundError(f"3-bit representations not found in {gb.ENCODED_SEQS_DIR}")

        gb.ALL_3BIT_SEQ_REPS = read_3bit_seq_reps(gb.ALL_3BIT_SEQ_REPS_PATH)

        # Trim ALL_3BIT_SEQ_REPS if a maximum length cap is configured for imported sequences
        if gb.MAX_IMPORTED_SEQ_LEN is not None and gb.ALL_3BIT_SEQ_REPS.shape[-2] > gb.MAX_IMPORTED_SEQ_LEN:
            gb.ALL_3BIT_SEQ_REPS = gb.ALL_3BIT_SEQ_REPS[..., :gb.MAX_IMPORTED_SEQ_LEN, :]

        # Limit subsequences if requested
        gb.ALL_3BIT_SEQ_REPS = _limit_region_axis(
            gb.ALL_3BIT_SEQ_REPS,
            getattr(gb, "MAX_NUM_SUBSEQS", None),
            "3-bit representations",
            log_change=True
        )

        # Check if there is masking before the sequence
        masking_bits = gb.ALL_3BIT_SEQ_REPS[..., 0, 0]
        if np.any(masking_bits):
            raise ValueError("There is masking before the sequence.")
        
        # Validate the 3-bit representations and indices
        if len(gb.ALL_3BIT_SEQ_REPS.shape) != 4:
            raise ValueError(f"3-bit representations have {len(gb.ALL_3BIT_SEQ_REPS.shape)} dimensions, but 4 are expected")
        n_regions, n_seqs, max_seq_len, n_bits = gb.ALL_3BIT_SEQ_REPS.shape
        total_seq_count = len(gb.EXCLUDED_TAXA_INDICES) + len(gb.TESTING_INDICES) + len(gb.TRAINING_INDICES)
        if n_seqs != total_seq_count:
            raise ValueError(f"3-bit representations have {n_seqs} sequences, but dataset has {total_seq_count}")
        if n_bits != 3:
            raise ValueError(f"3-bit representations have {n_bits} bits per base, expected 3")
        if ((len(gb.EXCLUDED_TAXA_INDICES) > 0 and max(gb.EXCLUDED_TAXA_INDICES) >= n_seqs) or
            (len(gb.TESTING_INDICES) > 0 and max(gb.TESTING_INDICES) >= n_seqs) or
            (len(gb.TRAINING_INDICES) > 0 and max(gb.TRAINING_INDICES) >= n_seqs)):
            raise ValueError(f"One or more index values are out of range for {n_seqs} sequences")
        if len(gb.TRAINING_INDICES) == 0:
            raise ValueError("Training set must have sequences.")
        if len(gb.TESTING_INDICES) == 0:
            print("NOTICE: No testing sequences found. Test-set quick tests will be skipped.")

        # Validate region availability for subsequence pair mining (if needed)
        enforce_cross_region = getattr(gb, "SUBSEQUENCES_ALWAYS_CROSS_REGION", True)
        if gb.PAIR_RANKS[8] and enforce_cross_region:
            min_region_idx = 0 if gb.USE_FULL_SEQS else 1
            max_region_idx = n_regions if gb.USE_SUB_SEQS else 1
            n_available_regions = max_region_idx - min_region_idx
            if n_available_regions < 2:
                raise ValueError(
                    f"PAIR_RANKS[8]=True requires at least 2 available regions for subsequence pairs. "
                    f"Got {n_available_regions} with USE_FULL_SEQS={gb.USE_FULL_SEQS}, USE_SUB_SEQS={gb.USE_SUB_SEQS}, N_REGIONS={n_regions}"
                )

        # Slice the 3-bit representations into their respective sets
        gb.EXCLUDED_3BIT_SEQ_REPS = gb.ALL_3BIT_SEQ_REPS[:, gb.EXCLUDED_TAXA_INDICES, :, :]
        gb.TESTING_3BIT_SEQ_REPS = gb.ALL_3BIT_SEQ_REPS[:, gb.TESTING_INDICES, :, :]
        gb.TRAINING_3BIT_SEQ_REPS = gb.ALL_3BIT_SEQ_REPS[:, gb.TRAINING_INDICES, :, :]
        gb.TRAINING_3BIT_SEQ_REPS_TRANSPOSED = gb.TRAINING_3BIT_SEQ_REPS.transpose(1, 0, 2, 3)

        # Load the k-mer representations
        gb.ALL_KMER_SEQ_REPS = {}
        gb.EXCLUDED_KMER_SEQ_REPS = {}
        gb.TESTING_KMER_SEQ_REPS = {}
        gb.TRAINING_KMER_SEQ_REPS = {}
        gb.MEAN_PER_KMERS = {}
        gb.STD_PER_KMERS = {}
        kmer_file_pattern = re.compile(r"^(\d+)-mer_seq_reps\.npy$")

        loaded_k_values = []
        for filename in os.listdir(gb.ENCODED_SEQS_DIR):
            match = kmer_file_pattern.match(filename)
            if match:
                k = int(match.group(1))
                # Don't load unneeded k-mer representations
                if k not in needed_k_values:
                    continue
                else:
                    loaded_k_values.append(k)
                    kmer_filepath = os.path.join(gb.ENCODED_SEQS_DIR, filename)
                    all_kmer_reps_for_k = np.load(kmer_filepath)

                    # Limit subsequences on k-mer tensors to match 3-bit tensors
                    all_kmer_reps_for_k = _limit_region_axis(
                        all_kmer_reps_for_k,
                        getattr(gb, "MAX_NUM_SUBSEQS", None),
                        f"{k}-mer representations",
                        log_change=False
                    )
                    
                    # Validate k-mer representations
                    if len(all_kmer_reps_for_k.shape) != 3: # Expecting [N_REGIONS+1, N_SEQS, 4**K]
                        raise ValueError(f"{k}-mer representations have {len(all_kmer_reps_for_k.shape)} dimensions, expected 3. Like: [N_REGIONS+1, N_SEQS, 4**K]")
                    if all_kmer_reps_for_k.shape[1] != n_seqs:
                        raise ValueError(f"{k}-mer representations have {all_kmer_reps_for_k.shape[1]} sequences, but dataset has {n_seqs}")
                    if all_kmer_reps_for_k.shape[2] != 4**k:
                        raise ValueError(f"{k}-mer representations have {all_kmer_reps_for_k.shape[2]} dimensions, expected {4**k}")

                    # Store the loaded representations
                    gb.ALL_KMER_SEQ_REPS[k] = all_kmer_reps_for_k
                    
                    # Slice the k-mer representations into their respective sets
                    gb.EXCLUDED_KMER_SEQ_REPS[k] = all_kmer_reps_for_k[:, gb.EXCLUDED_TAXA_INDICES, :]
                    gb.TESTING_KMER_SEQ_REPS[k] = all_kmer_reps_for_k[:, gb.TESTING_INDICES, :]
                    gb.TRAINING_KMER_SEQ_REPS[k] = all_kmer_reps_for_k[:, gb.TRAINING_INDICES, :]

                    # Store the mean and std of the k-mer representations
                    gb.MEAN_PER_KMERS[k] = np.mean(all_kmer_reps_for_k, axis=(0, 1))
                    gb.STD_PER_KMERS[k] = np.std(all_kmer_reps_for_k, axis=(0, 1))
        
        # If we didnt get all the needed k-mer representations, raise an error
        if any(k not in loaded_k_values for k in needed_k_values):
            unfound_k_values = [k for k in needed_k_values if k not in loaded_k_values]
            raise ValueError(f"Did not load all needed k-mer representations: {unfound_k_values}")

        # Load taxonomic datastructures for 'train', 'test', and 'excluded'
        (gb.TRAIN_FULL_TAX_LABEL_FROM_SEQ_ID_DICT, gb.TRAIN_LIST_OF_SEQ_INDICES_IN_TAXON_AT_RANK_DICT, 
         gb.TRAIN_LIST_OF_TAXON_LABELS_IN_TAXON_AT_RANK_DICT, gb.TRAIN_LIST_OF_TAXON_LABELS_AT_RANK_DICT, 
         gb.TRAIN_NESTED_LIST_OF_SEQ_INDICES, gb.TRAIN_NESTED_DICTS_OF_TAXA,
         gb.TRAIN_TAXON_LABEL_TO_TAXON_ID, gb.TRAIN_TAXON_ID_TO_TAXON_LABEL) = load_tax_objs(dataset_split_dir, "train")

        (gb.TEST_FULL_TAX_LABEL_FROM_SEQ_ID_DICT, gb.TEST_LIST_OF_SEQ_INDICES_IN_TAXON_AT_RANK_DICT, 
         gb.TEST_LIST_OF_TAXON_LABELS_IN_TAXON_AT_RANK_DICT, gb.TEST_LIST_OF_TAXON_LABELS_AT_RANK_DICT, 
         gb.TEST_NESTED_LIST_OF_SEQ_INDICES, gb.TEST_NESTED_DICTS_OF_TAXA,
         gb.TEST_TAXON_LABEL_TO_TAXON_ID, gb.TEST_TAXON_ID_TO_TAXON_LABEL) = load_tax_objs(dataset_split_dir, "test")

        (gb.EXCLUDED_FULL_TAX_LABEL_FROM_SEQ_ID_DICT, gb.EXCLUDED_LIST_OF_SEQ_INDICES_IN_TAXON_AT_RANK_DICT, 
         gb.EXCLUDED_LIST_OF_TAXON_LABELS_IN_TAXON_AT_RANK_DICT, gb.EXCLUDED_LIST_OF_TAXON_LABELS_AT_RANK_DICT, 
         gb.EXCLUDED_NESTED_LIST_OF_SEQ_INDICES, gb.EXCLUDED_NESTED_DICTS_OF_TAXA,
         gb.EXCLUDED_TAXON_LABEL_TO_TAXON_ID, gb.EXCLUDED_TAXON_ID_TO_TAXON_LABEL) = load_tax_objs(dataset_split_dir, "excluded")

        _validate_rank_qualified_taxon_mapping(
            gb.TRAIN_TAXON_LABEL_TO_TAXON_ID,
            gb.TRAIN_LIST_OF_TAXON_LABELS_AT_RANK_DICT,
            split_name="train",
        )
        
        # Build full taxonomic labels for each set based on the loaded dictionaries and indices
        gb.TRAIN_FULL_TAX_LABELS = [gb.TRAIN_FULL_TAX_LABEL_FROM_SEQ_ID_DICT[seq_i] for seq_i in gb.TRAINING_INDICES]
        gb.TEST_FULL_TAX_LABELS = [gb.TEST_FULL_TAX_LABEL_FROM_SEQ_ID_DICT[seq_i] for seq_i in gb.TESTING_INDICES]
        gb.EXCLUDED_FULL_TAX_LABELS = [gb.EXCLUDED_FULL_TAX_LABEL_FROM_SEQ_ID_DICT[seq_i] for seq_i in gb.EXCLUDED_TAXA_INDICES]

        # Load label arrays for 'train', 'test', and 'excluded'
        (gb.TRAIN_PAIRWISE_RANKS, gb.TRAIN_SEQ_TAXON_IDS, 
         gb.TRAIN_PAIRWISE_POS_MASKS, gb.TRAIN_PAIRWISE_NEG_MASKS, 
         gb.TRAIN_PAIRWISE_MRCA_TAXON_IDS, gb.TRAIN_PAIRWISE_DISTANCES,
         gb.TRAIN_DISTANCES_LOOKUP_ARRAY, gb.TRAIN_DISTANCE_BETWEEN_DOMAINS) = load_labels_arrays(dataset_split_dir, "train")
        # (gb.TEST_PAIRWISE_RANKS, gb.TEST_SEQ_TAXON_IDS, 
        #  gb.TEST_PAIRWISE_POS_MASKS, gb.TEST_PAIRWISE_NEG_MASKS, 
        #  gb.TEST_PAIRWISE_MRCA_TAXON_IDS, gb.TEST_PAIRWISE_DISTANCES,
        #  gb.TEST_DISTANCES_LOOKUP_ARRAY, gb.TEST_DISTANCE_BETWEEN_DOMAINS) = load_labels_arrays(dataset_split_dir, "test") # Unused now
        # (gb.EXCLUDED_PAIRWISE_RANKS, gb.EXCLUDED_SEQ_TAXON_IDS, 
        #  gb.EXCLUDED_PAIRWISE_POS_MASKS, gb.EXCLUDED_PAIRWISE_NEG_MASKS, 
        #  gb.EXCLUDED_PAIRWISE_MRCA_TAXON_IDS, gb.EXCLUDED_PAIRWISE_DISTANCES,
        #  gb.EXCLUDED_DISTANCES_LOOKUP_ARRAY, gb.EXCLUDED_DISTANCE_BETWEEN_DOMAINS) = load_labels_arrays(dataset_split_dir, "excluded") # Unused now

        # Apply taxa cap filter (if configured)
        apply_taxa_cap()

        # Compute per-seq taxon sizes and per-rank baseline taxon sizes (train set)
        gb.TRAIN_TAXON_COUNTS_PER_SEQ, gb.TRAIN_TAXON_BASELINE_COUNT_PER_RANK = compute_train_taxon_counts_and_baseline(gb.TRAIN_SEQ_TAXON_IDS)

        # Compute domain-adjusted pairwise distances for training set
        # This is done once at load time to avoid recomputing it for every batch during training
        gb.ADJUSTED_TRAIN_PAIRWISE_DISTANCES = compute_adjusted_pairwise_distances(
            gb.TRAIN_PAIRWISE_DISTANCES,
            gb.TRAIN_SEQ_TAXON_IDS,
            gb.TRAIN_TAXON_LABEL_TO_TAXON_ID,
            gb.BACTERIA_DISTANCE_FACTOR,
            gb.ARCHEA_DISTANCE_FACTOR,
            gb.RED_DISTANCE_BETWEEN_DOMAINS,
            gb.DISTANCE_GAMMA_CORRECTION_GAMMA
        )

        # Precompute bacteria and archaea indices for fast subsampling during mining
        # This avoids repeated np.where() calls every batch
        bacteria_taxon_id = _get_taxon_id_at_rank(gb.TRAIN_TAXON_LABEL_TO_TAXON_ID, rank=0, taxon_label='Bacteria')
        archaea_taxon_id = _get_taxon_id_at_rank(gb.TRAIN_TAXON_LABEL_TO_TAXON_ID, rank=0, taxon_label='Archaea')
        domain_ids = gb.TRAIN_SEQ_TAXON_IDS[:, 0]
        gb.TRAIN_BACTERIA_INDICES = np.where(domain_ids == bacteria_taxon_id)[0]
        gb.TRAIN_ARCHAEA_INDICES = np.where(domain_ids == archaea_taxon_id)[0]

        # We dont need the original pairwise distances anymore (free some memory)
        del gb.TRAIN_PAIRWISE_DISTANCES
        gb.TRAIN_PAIRWISE_DISTANCES = None
        
        # We dont need the full array of sequence representations anymore (free some memory)
        del gb.ALL_3BIT_SEQ_REPS
        gb.ALL_3BIT_SEQ_REPS = None

        # Set the flag indicating that the dataset has been loaded
        gb.DATASET_IS_LOADED = True
        print("Done loading dataset.")


def load_red_trees(arc_decorated_tree_path=None, 
                   bac_decorated_tree_path=None, 
                   precomputed_mapping_path=None): 
    """
    Load the RedTrees object from the dataset directory.

    Example Args:
        arc_decorated_tree_path = "/home/haig/Repos/micro16s/redvals/decorated_trees/ar53_r226_decorated.pkl"
        bac_decorated_tree_path = "/home/haig/Repos/micro16s/redvals/decorated_trees/bac120_r226_decorated.pkl"
        precomputed_mapping_path = "/home/haig/Repos/micro16s/redvals/taxon_mappings/taxon_to_node_mapping_r226.pkl"

    """

    if arc_decorated_tree_path is None:
        arc_decorated_tree_path = gb.REDVALS_DIR + "/decorated_trees/ar53_r226_decorated.pkl"
    if bac_decorated_tree_path is None:
        bac_decorated_tree_path = gb.REDVALS_DIR + "/decorated_trees/bac120_r226_decorated.pkl"
    if precomputed_mapping_path is None:
        precomputed_mapping_path = gb.REDVALS_DIR + "/taxon_mappings/taxon_to_node_mapping_r226.pkl"

    # Initialise (already decorated) RedTree object
    # We use the decorated trees (.pkl files) as input
    red_trees = RedTree(bac_decorated_tree_path, arc_decorated_tree_path, verbose=False)

    # The trees must already be decorated with RED values
    assert red_trees.is_decorated()

    # Load the taxon to node mappings
    red_trees.load_taxa_to_node_mapping(precomputed_mapping_path)

    return red_trees


def load_tax_objs(dataset_split_dir, name):
    """Load all taxonomic datastructures for a given dataset partition.
    
    Args:
        dataset_split_dir (str): Directory containing the dataset files
        name (str): Name of the partition (e.g., 'train', 'test', 'excluded')
    
    Returns:
        tuple: Contains the following datastructures:
            - full_tax_label_from_seq_id_dict
            - list_of_seq_indices_in_taxon_at_rank_dict
            - list_of_taxon_labels_in_taxon_at_rank_dict
            - list_of_taxon_labels_at_rank_dict
            - nested_list_of_seq_indices
            - nested_dicts_of_taxa
            - taxon_label_to_taxon_id
            - taxon_id_to_taxon_label
    """
    path = os.path.join(dataset_split_dir, "tax_objs", name, "")
    return (
        pickle.load(open(path + "full_tax_label_from_seq_id_dict.pkl", "rb")),
        None, # UNUSED: pickle.load(open(path + "list_of_seq_indices_in_taxon_at_rank_dict.pkl", "rb")),
        None, # UNUSED: pickle.load(open(path + "list_of_taxon_labels_in_taxon_at_rank_dict.pkl", "rb")),
        pickle.load(open(path + "list_of_taxon_labels_at_rank_dict.pkl", "rb")),
        None, # UNUSED: pickle.load(open(path + "nested_list_of_seq_indices.pkl", "rb")),
        None, # UNUSED: pickle.load(open(path + "nested_dicts_of_taxa.pkl", "rb"))
        pickle.load(open(path + "taxon_label_to_taxon_id.pkl", "rb")),
        None, # UNUSED: pickle.load(open(path + "taxon_id_to_taxon_label.pkl", "rb")),
    )

def load_labels_arrays(dataset_split_dir, name):
    """Load all label arrays for a given dataset partition.
    
    Args:
        dataset_split_dir (str): Directory containing the dataset files
        name (str): Name of the partition (e.g., 'train', 'test', 'excluded')
    
    Returns:
        tuple: (pairwise_ranks, seq_taxon_ids, pairwise_pos_masks, pairwise_neg_masks, pairwise_mrca_taxon_ids, pairwise_distances, distances_lookup_array, distance_between_domains)
            - pairwise_ranks (np.ndarray): Matrix of shape (n_seqs, n_seqs) with dtype int8
            - seq_taxon_ids (np.ndarray): Matrix of shape (n_seqs, 7) with dtype int32
            - pairwise_pos_masks (np.ndarray): Matrix of shape (7, n_seqs, n_seqs) with dtype bool
            - pairwise_neg_masks (np.ndarray): Matrix of shape (7, n_seqs, n_seqs) with dtype bool
            - pairwise_mrca_taxon_ids (np.ndarray): Matrix of shape (n_seqs, n_seqs) with dtype int32
            - pairwise_distances (np.ndarray): Matrix of shape (n_seqs, n_seqs) with dtype float32
            - distances_lookup_array (np.ndarray): 1D array of shape (max_taxon_id + 1,) with dtype float32
            - distance_between_domains (float): Scalar distance value for different domains
    """
    path = os.path.join(dataset_split_dir, "labels", name, "")
    return (
        np.load(path + "pairwise_ranks.npy"), 
        np.load(path + "seq_taxon_ids.npy"), 
        None, # UNUSED: np.load(path + "pairwise_pos_masks.npy"), 
        None, # UNUSED: np.load(path + "pairwise_neg_masks.npy"), 
        None, # UNUSED: np.load(path + "pairwise_mrca_taxon_ids.npy"), 
        np.load(path + "pairwise_distances.npy"), 
        None, # UNUSED: np.load(path + "distances_lookup_array.npy"), 
        None # UNUSED: np.load(path + "distance_between_domains.npy")[0] # Extract scalar from single-element array
    )


def compute_adjusted_pairwise_distances(pairwise_distances, seq_taxon_ids, taxon_label_to_taxon_id, 
                                         bacteria_distance_factor, archaea_distance_factor, 
                                         red_distance_between_domains, distance_gamma_correction_gamma):
    """
    Compute domain-adjusted pairwise distances based on domain-specific distance factors.
    
    This function modifies the true pairwise distances by:
    1. Multiplying distances by domain-specific factors when both sequences are in the same domain
    2. Using the RED distance between domains when sequences are from different domains
    3. Applying gamma correction for non-linear compression of the embedding space
    
    The adjustment accounts for the fact that evolutionary distance scales differently
    within Bacteria vs Archaea domains.
    
    Args:
        pairwise_distances (np.ndarray): The true pairwise RED distances | shape: (N_SEQS, N_SEQS)
        seq_taxon_ids (np.ndarray): The taxon IDs for all sequences | shape: (N_SEQS, 7) | dtype: int32
        taxon_label_to_taxon_id (dict): Dict mapping rank-qualified taxon label (str) -> taxon ID (int)
        bacteria_distance_factor (float): The distance scaling factor for bacterial pairs
        archaea_distance_factor (float): The distance scaling factor for archaeal pairs
        red_distance_between_domains (float): The RED distance between Bacteria and Archaea domains
        distance_gamma_correction_gamma (float): Gamma factor for gamma correction
    
    Returns:
        adjusted_distances (np.ndarray): The adjusted pairwise distances with domain-specific scaling applied | shape: (N_SEQS, N_SEQS)
    """
    
    # Get taxon IDs for Bacteria and Archaea domains
    bacteria_taxon_id = _get_taxon_id_at_rank(taxon_label_to_taxon_id, rank=0, taxon_label='Bacteria')
    archaea_taxon_id = _get_taxon_id_at_rank(taxon_label_to_taxon_id, rank=0, taxon_label='Archaea')
    
    # Get domain taxon IDs for all sequences
    # seq_taxon_ids[:, 0] extracts the domain (rank 0) for all sequences
    # domain_ids.shape: (N_SEQS,)
    domain_ids = seq_taxon_ids[:, 0]
    
    # Create domain membership arrays
    is_bacteria = domain_ids == bacteria_taxon_id  # shape: (N_SEQS,)
    is_archaea = domain_ids == archaea_taxon_id    # shape: (N_SEQS,)
    
    # Precompute row views for 2D broadcasting
    is_bacteria_row = is_bacteria[:, np.newaxis]  # shape: (N_SEQS, 1)
    is_archaea_row = is_archaea[:, np.newaxis]    # shape: (N_SEQS, 1)
    
    # Start with a copy of the original distances
    adjusted_distances = pairwise_distances.copy()
    # adjusted_distances.shape: (N_SEQS, N_SEQS)
    
    # Apply domain-specific distance factors using in-place operations
    # Multiply bacterial pairs by bacteria distance factor
    np.multiply(adjusted_distances, bacteria_distance_factor, 
                out=adjusted_distances, where=is_bacteria_row & is_bacteria)
    
    # Multiply archaeal pairs by archaea distance factor
    np.multiply(adjusted_distances, archaea_distance_factor, 
                out=adjusted_distances, where=is_archaea_row & is_archaea)
    
    # Set cross-domain distances to the RED distance between domains
    cross_domain_mask = (is_bacteria_row & is_archaea) | (is_archaea_row & is_bacteria)
    adjusted_distances[cross_domain_mask] = red_distance_between_domains

    # Apply gamma correction to all distances
    # distance = γ * distance ** log2(2 / γ)
    if distance_gamma_correction_gamma != 1.0:
        log2_2_gamma = math.log2(2 / distance_gamma_correction_gamma)
        adjusted_distances = distance_gamma_correction_gamma * (adjusted_distances ** log2_2_gamma)
    
    return adjusted_distances
