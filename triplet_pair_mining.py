"""
This file contains functions for online mining of pairs and triplets from a Micro16S dataset.

Taxonomic Ranks:
    The taxonomic ranks (pairwise_ranks values) are as follows:
        - Ignore Pair    <->  -2  (pair is skipped during mining)
        - Domain (d__)   <->  -1  (different domains)
        - Phylum (p__)   <->   0  (same domain, different phyla)
        - Class (c__)    <->   1  (same phylum, different classes)
        - Order (o__)    <->   2  (same class, different orders)
        - Family (f__)   <->   3  (same order, different families)
        - Genus (g__)    <->   4  (same family, different genera)
        - Species (s__)  <->   5  (same genus, different species)
        - Sequence       <->   6  (same species, different sequences)
        - Subsequence    <->   7  (duplicate pairs: same sequence, region relationship optional)

Upper Triangle Filtering (Duplicate Removal):
    The pairwise distance and rank matrices are symmetric: entry (i,j) == entry (j,i).
    This means pairs (seq1, seq2) and (seq2, seq1) are identical duplicates.
    Similarly, anchor-positive pairs (A, P) and (P, A) represent the same relationship.
    
    To avoid wasting batch size and compute on duplicates:
    - Pair mining filters to upper triangle (row < col) before sampling
    - Triplet mining keeps one AP orientation only (chosen randomly each run: anchor < positive OR anchor > positive)
    
    This halves the candidate pools while retaining all unique relationships.

Mining Process:

    ---
    The main function mine_pairs_and_triplets() performs the following steps:
    1. Gets raw training sequences
    2. Subsamples sequences (bacteria/archaea) and matrices to reduce compute
    3. Adds duplicate sequences for subsequence mining (rank 8) if enabled (optionally forcing cross-region pairs via SUBSEQUENCES_ALWAYS_CROSS_REGION)
    4. Performs rank-based subsampling (setting some pairs to -2)
    5. Performs taxon-size-based subsampling to reduce dominance of large taxa
    6. Applies region selection and variations (mutations, shifts, etc.)
    7. Runs inference to get embeddings
    8. Computes pairwise embedding distances
    9. Mines triplets (if requested) - keeps one AP orientation (randomly anchor < positive or anchor > positive)
    10. Mines pairs (if requested) - filters to upper triangle to avoid duplicates
    11. Writes mining log (if enabled)
    
    ---
    Pair Mining:
    Pairs are mined using percentile-bucket sampling based on relative squared error:
        - rel_sq_error = ((d_pred - d_true)^2) / (d_true + eps)
        - eps is gb.RELATIVE_ERROR_EPSILONS_PAIR_MINING to avoid division by zero
    
    Pairs are selected from ranks specified by gb.PAIR_RANKS (boolean list).
    
    Pairs at rank r satisfy:
        - sequence_1 and sequence_2 share classification at rank (r-1), but differ at rank r
        - For example, rank 4 (family) pairs share the same order but belong to different families
    
    Percentile Bucket Sampling:
        - Pairs are ranked by relative squared error and placed into percentile buckets defined by gb.PAIR_MINING_BUCKETS
        - For example, if PAIR_MINING_BUCKETS specifies gaps of 0.2 (20%) and 0.7 (70%), we get buckets:
            - Bucket 0: bottom 20% of errors
            - Bucket 1: next 70% of errors (20th to 90th percentile)
            - Bucket 2: top 10% of errors (remaining)
        - Pairs are sampled from each bucket according to the sampling proportion defined in gb.PAIR_MINING_BUCKETS
        - The 'any' bucket (gap=None) allows random sampling from the entire pool.
    
    Sign Tracking:
        - We track whether each pair has positive error (d_pred > d_true, too far) or negative error (d_pred < d_true, too close)
        - This is useful for debugging but does not affect sampling

    ---
    Triplet Mining:
    Triplets are mined using separate per-rank percentile-bucket sampling to remove bias 
    from ranks with large volumes of triplets. Each rank gets:
    - Its own representative set (up to TRIPLET_MINING_REPRESENTATIVE_SET_SIZES[rank] candidates)
    - Its own bucket boundaries computed from that representative set
    - A budget proportional to its EMA-smoothed hardness metric
    
    Structure:
        - Anchor (A)
        - Positive (P)
        - Negative (N)
    
    For triplets at rank r (where r is an index in gb.TRIPLET_RANKS):
        - Anchor and Positive share the same taxon at rank r (pairwise_ranks[A, P] >= r)
          e.g. same domain if r=0, same phylum if r=1 (may also share deeper taxonomy)
        - Anchor and Negative share rank r-1 (pairwise_ranks[A, N] == r-1)
          e.g. different domains if r=0, different phyla if r=1
    
    Hardness Metric (per rank):
        - Based on satisfaction proportions:
          metric = hard_triplet_prop * gb.TRIPLET_EMA_HARD_WEIGHT + moderate_triplet_prop * gb.TRIPLET_EMA_MODERATE_WEIGHT
        - hard_triplet_prop: Proportion of triplets where AP >= AN
        - moderate_triplet_prop: Proportion of triplets where AP < AN < AP + M
        - Computed BEFORE zero-loss filtering to reflect overall rank difficulty
        - Updated via EMA after each batch using gb.TRIPLET_MINING_EMA_ALPHA
    
    Budget Allocation:
        - Batch size is divided across ranks according to EMA hardness
        - Each rank's proportion is capped by gb.TRIPLET_PER_RANK_BATCH_PROPORTION_MAX
        - Excess budget is redistributed to uncapped ranks preserving relative ratios
    
    Metric (Normalised Violation):
        - margin = (true_dist(A, N) - true_dist(A, P)) * gb.TRIPLET_MARGIN_EPSILON  (or manual fixed margin)
        - violation = pred_dist(A, P) - pred_dist(A, N) + margin
        - normalised_violation = violation / (true_dist(A, N) - true_dist(A, P))
    
    Sampling:
        - Candidates are gathered from valid ranks, subsampled to per-rank representative set sizes.
        - Normalised violations are computed.
        - EMA hardness buffers are updated with per-rank mean violations (BEFORE filtering).
        - (Optional) Filter to triplets with positive violation (controlled by gb.FILTER_ZERO_LOSS_TRIPLETS).
          Triplets with violation <= 0 already satisfy the constraint and produce zero loss.
          The per-rank minimum after filtering is derived from
          gb.MIN_TRIPLETS_AFTER_FILTER_PER_RANK_BATCH_SIZE_FACTOR * per_rank_batch_size
        - Per-rank bucket boundaries are computed from each rank's pool.
        - Triplets are sampled from each rank's buckets according to its budget.

    ---

"""

# Imports
import time
import numpy as np
import torch
from numba import njit, prange

# Local Imports
from pair_mining import mine_pairs, get_effective_pair_ranks
from triplet_mining import mine_triplets, get_effective_triplet_ranks
from generate_seq_variants import gen_seq_variants
from model import run_inference
import globals_config as gb
from utils import synchronize_if_cuda
from logging_utils import log_arc_bac_counts, write_mining_log, write_mining_phylum_log

# Re-use a global generator so we don't re-seed every call
_RNG = np.random.default_rng()


def _get_domain_taxon_id(taxon_label_to_taxon_id, domain_label):
    """
    Resolve a domain taxon ID from either rank-qualified or legacy plain mapping keys.
    """
    ranked_key = f"d__{domain_label}"
    if ranked_key in taxon_label_to_taxon_id:
        return taxon_label_to_taxon_id[ranked_key]
    if domain_label in taxon_label_to_taxon_id:
        return taxon_label_to_taxon_id[domain_label]
    raise KeyError(
        f"Could not resolve domain taxon ID for '{domain_label}'. "
        f"Tried keys '{ranked_key}' and '{domain_label}'."
    )


def _maybe_pin_cpu_tensor(tensor):
    """
    Pin a CPU tensor when PIN_MEMORY_FOR_MINING is enabled.
    """
    if tensor is None or not gb.PIN_MEMORY_FOR_MINING:
        return tensor
    if not isinstance(tensor, torch.Tensor):
        return tensor
    if tensor.device.type != 'cpu' or tensor.is_pinned():
        return tensor
    return tensor.pin_memory()


def _shuffle_aligned_cpu_tensors(*tensors):
    """
    Shuffle aligned CPU tensors along the first dimension using one shared permutation.

    All non-None tensors must be torch tensors with the same size in dimension 0.
    """
    if not tensors:
        return tuple()

    # Find a reference tensor to infer length/permutation.
    reference_tensor = None
    for tensor in tensors:
        if tensor is not None:
            reference_tensor = tensor
            break

    if reference_tensor is None:
        return tensors
    if not isinstance(reference_tensor, torch.Tensor):
        raise TypeError("Aligned shuffle expects torch tensors or None.")

    n_items = int(reference_tensor.shape[0])
    if n_items <= 1:
        return tensors

    permutation = torch.from_numpy(_RNG.permutation(n_items).astype(np.int64, copy=False))

    shuffled_tensors = []
    for tensor in tensors:
        if tensor is None:
            shuffled_tensors.append(None)
            continue
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("Aligned shuffle expects torch tensors or None.")
        if int(tensor.shape[0]) != n_items:
            raise ValueError("Aligned shuffle received tensors with mismatched leading dimensions.")
        shuffled_tensors.append(tensor.index_select(0, permutation))

    return tuple(shuffled_tensors)


def get_train_sequences():
    """
    Get all training sequences in their raw form (all regions, no variations).
    
    This function:
    1. Returns the pre-computed transposed training sequences from gb.TRAINING_3BIT_SEQ_REPS_TRANSPOSED
    2. Returns them with all regions intact, without region selection or variations
    
    Returns:
        train_sequences: np.ndarray of shape (N_TRAIN_SEQUENCES, N_REGIONS, MAX_IMPORTED_SEQ_LEN, 3)
            The raw 3-bit encoded training sequences with all regions.
    """
    
    # Check if dataset is loaded before proceeding
    if not gb.DATASET_IS_LOADED:
        raise RuntimeError("Dataset has not been loaded. Call load_micro16s_dataset first.")
    
    # Return the pre-computed transposed training sequences
    # gb.TRAINING_3BIT_SEQ_REPS_TRANSPOSED.shape: (N_TRAIN_SEQS, N_REGIONS, MAX_IMPORTED_SEQ_LEN, 3)
    return gb.TRAINING_3BIT_SEQ_REPS_TRANSPOSED


@njit(parallel=True, cache=True)
def _extract_submatrix_float32(full_matrix, indices):
    """
    Extract a square submatrix from a full matrix using parallel processing.
    
    Parallelizes over rows, with each thread extracting one complete row of the output.
    Indices should be sorted for better cache locality.
    
    Args:
        full_matrix: np.ndarray of shape (N, N) with dtype float32
        indices: np.ndarray of shape (n,) with dtype int64, indices to extract
    
    Returns:
        submatrix: np.ndarray of shape (n, n) with dtype float32
    """
    n = indices.shape[0]
    out = np.empty((n, n), dtype=np.float32)
    for i in prange(n):
        row_idx = indices[i]
        for j in range(n):
            col_idx = indices[j]
            out[i, j] = full_matrix[row_idx, col_idx]
    return out


@njit(parallel=True, cache=True)
def _extract_submatrix_int8(full_matrix, indices):
    """
    Extract a square submatrix from a full matrix using parallel processing.
    
    Parallelizes over rows, with each thread extracting one complete row of the output.
    Indices should be sorted for better cache locality.
    
    Args:
        full_matrix: np.ndarray of shape (N, N) with dtype int8
        indices: np.ndarray of shape (n,) with dtype int64, indices to extract
    
    Returns:
        submatrix: np.ndarray of shape (n, n) with dtype int8
    """
    n = indices.shape[0]
    out = np.empty((n, n), dtype=np.int8)
    for i in prange(n):
        row_idx = indices[i]
        for j in range(n):
            col_idx = indices[j]
            out[i, j] = full_matrix[row_idx, col_idx]
    return out


def _count_pairwise_rank_bins(pairwise_ranks):
    """
    Count pairwise rank occurrences, including ignored pairs.

    Returns:
        counts: np.ndarray of shape (10,) dtype int64
            counts[0] = ignored (-2), counts[1]..counts[9] = ranks -1..7
    """
    flat_ranks = pairwise_ranks.ravel()
    if flat_ranks.size == 0:
        return np.zeros(10, dtype=np.int64)
    adjusted = np.add(flat_ranks, 2, dtype=np.int16)
    return np.bincount(adjusted, minlength=10)


@njit(cache=True)
def _count_active_pair_endpoints_per_phylum_code(pairwise_ranks, phylum_codes, n_phyla, target_pairwise_rank=0):
    """
    Count active pair endpoints by phylum using the upper triangle (i < j).

    Returns:
        endpoint_counts: np.ndarray shape (n_phyla,), where each active pair contributes
                         +1 to each endpoint phylum.
        n_active_pairs: int
    """
    endpoint_counts = np.zeros(n_phyla, dtype=np.int64)
    n_active_pairs = 0
    n_seqs = pairwise_ranks.shape[0]

    for i in range(n_seqs):
        phylum_i = phylum_codes[i]
        for j in range(i + 1, n_seqs):
            if pairwise_ranks[i, j] != target_pairwise_rank:
                continue
            phylum_j = phylum_codes[j]
            endpoint_counts[phylum_i] += 1
            endpoint_counts[phylum_j] += 1
            n_active_pairs += 1

    return endpoint_counts, n_active_pairs


def _get_local_phylum_metadata(subsample_indices):
    """
    Build local phylum metadata aligned to subsampled train indices.

    Returns:
        phylum_codes: np.ndarray shape (n_subsampled,), contiguous codes [0..n_phyla-1]
        phylum_labels: list[str] aligned to phylum_codes
    """
    # Preferred path: derive names directly from full labels (robust even when
    # taxon_id_to_label is intentionally not loaded to save memory).
    if gb.TRAIN_FULL_TAX_LABELS is not None and len(gb.TRAIN_FULL_TAX_LABELS) > 0:
        n_train = len(gb.TRAIN_FULL_TAX_LABELS)
        if len(subsample_indices) > 0 and int(np.max(subsample_indices)) >= n_train:
            # Safety guard for unexpected index mismatches.
            return None, None

        local_phylum_labels = []
        for local_idx in subsample_indices:
            tax_labels = gb.TRAIN_FULL_TAX_LABELS[int(local_idx)]
            if tax_labels is None or len(tax_labels) < 2 or tax_labels[1] is None:
                local_phylum_labels.append("unknown_phylum")
            else:
                local_phylum_labels.append(str(tax_labels[1]))

        unique_labels = sorted(set(local_phylum_labels))
        label_to_code = {label: code for code, label in enumerate(unique_labels)}
        phylum_codes = np.array([label_to_code[label] for label in local_phylum_labels], dtype=np.int32)
        return phylum_codes, unique_labels

    # Fallback path: use numeric IDs if labels are unavailable.
    if (
        gb.TRAIN_SEQ_TAXON_IDS is not None
        and gb.TRAIN_SEQ_TAXON_IDS.ndim == 2
        and gb.TRAIN_SEQ_TAXON_IDS.shape[1] >= 2
    ):
        local_phylum_ids = gb.TRAIN_SEQ_TAXON_IDS[subsample_indices, 1].astype(np.int32, copy=False)
        if local_phylum_ids.size == 0:
            return np.empty((0,), dtype=np.int32), []

        unique_phylum_ids = np.unique(local_phylum_ids)
        phylum_codes = np.searchsorted(unique_phylum_ids, local_phylum_ids).astype(np.int32, copy=False)
        phylum_labels = [f"phylum_id_{int(phylum_id)}" for phylum_id in unique_phylum_ids]
        return phylum_codes, phylum_labels

    return None, None


def _build_step_phylum_stats(step_name, pairwise_ranks, phylum_codes, n_phyla, target_pairwise_rank=0):
    """
    Collect per-step phylum counts for mining diagnostics.
    """
    if phylum_codes is None:
        return None
    seq_counts = np.bincount(phylum_codes, minlength=n_phyla).astype(np.int64, copy=False)
    pair_endpoint_counts, n_active_pairs = _count_active_pair_endpoints_per_phylum_code(
        pairwise_ranks, phylum_codes, n_phyla, target_pairwise_rank=target_pairwise_rank
    )
    return {
        "step": step_name,
        "seq_counts": seq_counts,
        "pair_endpoint_counts": pair_endpoint_counts,
        "n_active_pairs": int(n_active_pairs),
    }


def _count_final_pair_phylum_combos(mined_pair_local_indices, mined_pair_ranks, phylum_codes, n_train_seqs, target_pair_rank=1):
    """
    Count final mined pair phylum combinations (unordered).

    Subsequence pairs are represented with duplicate indices (>= n_train_seqs). For those
    entries, the duplicate's phylum is identical to the first index phylum.
    """
    combo_counts = {}
    if mined_pair_local_indices is None or len(mined_pair_local_indices) == 0:
        return combo_counts

    if mined_pair_ranks is None:
        mined_pair_ranks = np.empty((len(mined_pair_local_indices),), dtype=np.int64)
        mined_pair_ranks.fill(target_pair_rank)
    else:
        mined_pair_ranks = np.asarray(mined_pair_ranks, dtype=np.int64)
    local_indices = np.asarray(mined_pair_local_indices, dtype=np.int64)

    n_items = min(len(local_indices), len(mined_pair_ranks))
    for row_idx in range(n_items):
        if int(mined_pair_ranks[row_idx]) != int(target_pair_rank):
            continue
        idx_a_raw, idx_b_raw = local_indices[row_idx]
        idx_a = int(idx_a_raw)
        idx_b = int(idx_b_raw)
        code_a = int(phylum_codes[idx_a])
        if idx_b < n_train_seqs:
            code_b = int(phylum_codes[idx_b])
        else:
            code_b = code_a
        if code_a <= code_b:
            key = (code_a, code_b)
        else:
            key = (code_b, code_a)
        combo_counts[key] = combo_counts.get(key, 0) + 1

    return combo_counts


def _count_final_triplet_an_phylum_combos(mined_triplet_local_indices, mined_triplet_ranks, phylum_codes, target_triplet_rank=1):
    """
    Count final mined triplet A-N phylum combinations (ordered).
    """
    combo_counts = {}
    if mined_triplet_local_indices is None or len(mined_triplet_local_indices) == 0:
        return combo_counts

    if mined_triplet_ranks is None:
        mined_triplet_ranks = np.empty((len(mined_triplet_local_indices),), dtype=np.int64)
        mined_triplet_ranks.fill(target_triplet_rank)
    else:
        mined_triplet_ranks = np.asarray(mined_triplet_ranks, dtype=np.int64)
    local_indices = np.asarray(mined_triplet_local_indices, dtype=np.int64)

    n_items = min(len(local_indices), len(mined_triplet_ranks))
    for row_idx in range(n_items):
        if int(mined_triplet_ranks[row_idx]) != int(target_triplet_rank):
            continue
        anchor_idx_raw, _, negative_idx_raw = local_indices[row_idx]
        anchor_idx = int(anchor_idx_raw)
        negative_idx = int(negative_idx_raw)
        key = (int(phylum_codes[anchor_idx]), int(phylum_codes[negative_idx]))
        combo_counts[key] = combo_counts.get(key, 0) + 1

    return combo_counts


def subsample_train_sequences_and_matrices(train_sequences):
    """
    Randomly subsample training sequences and their corresponding pairwise matrices.
    
    This function reduces compute by selecting a random fraction of training sequences
    separately for Bacteria and Archaea based on gb.MINING_TRAIN_SEQUENCES_SUB_SAMPLE_BACTERIA
    and gb.MINING_TRAIN_SEQUENCES_SUB_SAMPLE_ARCHAEA.
    
    It also extracts the corresponding rows and columns from the global pairwise distance 
    and rank matrices, and slices the per-sequence taxon counts for taxon-size-based subsampling.
    
    The subsampling is done without replacement, and indices are sorted for better cache locality.
    
    Args:
        train_sequences: np.ndarray of shape (N_TRAIN_SEQUENCES, N_REGIONS, MAX_IMPORTED_SEQ_LEN, 3)
            The raw training sequences with all regions.

    Returns:
        subsampled_sequences: np.ndarray of shape (n_seqs_to_use, N_REGIONS, MAX_IMPORTED_SEQ_LEN, 3)
            The randomly selected training sequences with all regions.
        adjusted_distances_subsampled: np.ndarray of shape (n_seqs_to_use, n_seqs_to_use)
            The pairwise distances matrix for the subsampled sequences.
        pairwise_ranks_subsampled: np.ndarray of shape (n_seqs_to_use, n_seqs_to_use)
            The pairwise ranks matrix for the subsampled sequences.
        taxon_counts_subsampled: np.ndarray of shape (n_seqs_to_use, 7)
            The per-sequence taxon counts for the subsampled sequences.
        domain_counts: dict with keys 'bacteria' and 'archaea' containing the sampled counts.
        subsample_indices: np.ndarray of shape (n_seqs_to_use,)
            Global training set indices (0..n_train-1) of the subsampled sequences.
    
    Global variables used:
        gb.MINING_TRAIN_SEQUENCES_SUB_SAMPLE_BACTERIA: Fraction of bacteria to sample (0.0-1.0) | float
        gb.MINING_TRAIN_SEQUENCES_SUB_SAMPLE_ARCHAEA: Fraction of archaea to sample (0.0-1.0) | float
        gb.ADJUSTED_TRAIN_PAIRWISE_DISTANCES: Full pairwise distances matrix | shape: (N_TRAIN_SEQUENCES, N_TRAIN_SEQUENCES)
        gb.TRAIN_PAIRWISE_RANKS: Full pairwise ranks matrix | shape: (N_TRAIN_SEQUENCES, N_TRAIN_SEQUENCES)
        gb.TRAIN_BACTERIA_INDICES: Precomputed indices of bacteria sequences | shape: (n_bacteria,)
        gb.TRAIN_ARCHAEA_INDICES: Precomputed indices of archaea sequences | shape: (n_archaea,)
        gb.TRAIN_TAXON_COUNTS_PER_SEQ: Per-sequence taxon sizes at each rank | shape: (N_TRAIN_SEQUENCES, 7)
    """

    # Use precomputed bacteria and archaea indices (computed once at dataset load time)
    bacteria_indices = gb.TRAIN_BACTERIA_INDICES
    archaea_indices = gb.TRAIN_ARCHAEA_INDICES
    
    # Determine number of sequences to sample for each domain
    n_bacteria = len(bacteria_indices)
    n_archaea = len(archaea_indices)
    
    n_bacteria_to_use = max(0, int(round(gb.MINING_TRAIN_SEQUENCES_SUB_SAMPLE_BACTERIA * n_bacteria)))
    n_archaea_to_use = max(0, int(round(gb.MINING_TRAIN_SEQUENCES_SUB_SAMPLE_ARCHAEA * n_archaea)))

    # Randomly select indices without replacement
    if n_bacteria_to_use > 0:
        bacteria_subsample_indices = _RNG.choice(bacteria_indices, size=n_bacteria_to_use, replace=False)
    else:
        bacteria_subsample_indices = np.array([], dtype=int)
        
    if n_archaea_to_use > 0:
        archaea_subsample_indices = _RNG.choice(archaea_indices, size=n_archaea_to_use, replace=False)
    else:
        archaea_subsample_indices = np.array([], dtype=int)
    
    # Combine indices (these are global training set indices: 0..n_train-1)
    subsample_indices = np.concatenate([bacteria_subsample_indices, archaea_subsample_indices])

    subsample_indices = np.sort(subsample_indices)  # Sort for better cache locality
    # subsample_indices.shape: (n_seqs_to_use,)

    # Subsample the sequences
    subsampled_sequences = train_sequences[subsample_indices]
    # subsampled_sequences.shape: (n_seqs_to_use, N_REGIONS, MAX_IMPORTED_SEQ_LEN, 3)
    
    # Subsample the global pairwise matrices to match the subsampled sequences
    # Extract rows and columns corresponding to the subsampled sequence indices
    # Use parallel Numba kernels for efficient submatrix extraction
    # Ensure indices are int64 for numba kernels (no-op if already int64)
    if subsample_indices.dtype != np.int64:
        subsample_indices = subsample_indices.astype(np.int64)
    adjusted_distances_subsampled = _extract_submatrix_float32(gb.ADJUSTED_TRAIN_PAIRWISE_DISTANCES, subsample_indices)
    pairwise_ranks_subsampled = _extract_submatrix_int8(gb.TRAIN_PAIRWISE_RANKS, subsample_indices)
    # adjusted_distances_subsampled.shape: (n_seqs_to_use, n_seqs_to_use)
    # pairwise_ranks_subsampled.shape: (n_seqs_to_use, n_seqs_to_use)

    # Slice the per-sequence taxon counts for the subsampled sequences
    taxon_counts_subsampled = gb.TRAIN_TAXON_COUNTS_PER_SEQ[subsample_indices]
    # taxon_counts_subsampled.shape: (n_seqs_to_use, 7)

    domain_counts = {
        'bacteria': len(bacteria_subsample_indices),
        'archaea': len(archaea_subsample_indices),
    }

    return subsampled_sequences, adjusted_distances_subsampled, pairwise_ranks_subsampled, taxon_counts_subsampled, domain_counts, subsample_indices


def add_duplicate_sequences(train_sequences, proportion):
    """
    Generate duplicate sequences for subsequence pair mining (rank 8).
    
    This creates copies of a proportion of sequences that can optionally be forced
    onto different regions and always receive independent sequence variations,
    enabling mining of subsequence pairs.
    
    Unlike the previous implementation, this does NOT modify the input arrays or expand
    the pairwise matrices. Instead, it returns separate duplicate sequence data. 
    The implicit understanding is:
        - Rank between a duplicate and its original = 7 (same sequence, subsequence relationship configurable)
        - Distance between a duplicate and its original = 0
    
    Args:
        train_sequences: np.ndarray of shape (n_seqs, N_REGIONS, MAX_IMPORTED_SEQ_LEN, 3)
        proportion: float (0.0-1.0), proportion of sequences to duplicate
    
    Returns:
        duplicate_sequences: np.ndarray of shape (n_duplicates, N_REGIONS, MAX_IMPORTED_SEQ_LEN, 3)
            Copies of the selected original sequences.
        duplicate_indices: np.ndarray of shape (n_duplicates,)
            Index into train_sequences for each duplicate's original sequence.
            i.e. duplicate_indices[i] is the original index for duplicate i.
    """
    n_seqs = train_sequences.shape[0]
    n_duplicates = int(n_seqs * proportion)
    
    if n_duplicates == 0:
        # Return empty arrays with correct shapes
        empty_seqs = np.empty((0,) + train_sequences.shape[1:], dtype=train_sequences.dtype)
        empty_indices = np.empty((0,), dtype=np.int64)
        return empty_seqs, empty_indices
    
    # Randomly select which sequences to duplicate (each original is duplicated at most once)
    source_indices = _RNG.choice(n_seqs, size=n_duplicates, replace=False)
    
    # Create duplicate sequences (copies of the originals so downstream augmentations run independently)
    duplicate_sequences = train_sequences[source_indices].copy()
    # duplicate_sequences.shape: (n_duplicates, N_REGIONS, MAX_IMPORTED_SEQ_LEN, 3)
    
    # Store the original indices
    duplicate_indices = source_indices.astype(np.int64)
    # duplicate_indices.shape: (n_duplicates,)
    
    return duplicate_sequences, duplicate_indices


@njit(parallel=True, cache=True)
def _rank_based_subsampling_kernel(flat_ranks, rank_proportions):
    """
    Numba JIT-compiled parallel kernel for rank-based subsampling.
    
    Single-pass parallel algorithm that modifies flat_ranks in-place.
    Generates random numbers lazily only when needed (avoiding expensive
    pre-allocation of random array).
    
    Args:
        flat_ranks: np.ndarray of shape (n_elements,) with dtype int8
            Flattened pairwise ranks matrix, modified in-place.
        rank_proportions: np.ndarray of shape (9,) with dtype float64
            Proportion to KEEP for each internal rank (-1 to 7, indexed as 0 to 8).
    """
    n_elements = flat_ranks.shape[0]
    
    for i in prange(n_elements):
        rank = flat_ranks[i]
        
        # Skip if rank is -2 (already ignored) or out of expected range
        if rank < -1 or rank > 7:
            continue
        
        # Get proportion for this rank (offset by 1 since internal ranks are -1 to 7)
        proportion = rank_proportions[rank + 1]
        
        # Optimization: Skip RNG completely if keeping all (proportion >= 0.999)
        if proportion >= 0.999:
            continue
        
        # Drop if proportion is effectively 0
        if proportion <= 0.0:
            flat_ranks[i] = -2
            continue
        
        # Only generate random number if strictly necessary
        if np.random.random() > proportion:
            flat_ranks[i] = -2


def rank_based_subsampling(pairwise_ranks):
    """
    Randomly downsample pairs at specific ranks by setting them to -2 (Ignore Pair).
    
    This allows us to reduce the number of pairs mined at abundant ranks (e.g. phylum)
    without losing the hard negatives completely (as we do if we just turn off mining for that rank).

    Uses a Numba JIT-compiled parallel kernel.
    
    Args:
        pairwise_ranks: np.ndarray of shape (n_seqs, n_seqs)
            The pairwise ranks matrix (typically subsampled).
            
    Returns:
        pairwise_ranks: np.ndarray of shape (n_seqs, n_seqs)
            The modified pairwise ranks matrix.
            
    Global variables:
        gb.DOWNSAMPLE_PAIRS_AT_RANK: List of (rank, proportion) tuples.
            rank: 0=domain, ..., 8=subsequence
            proportion: 0.0-1.0 (proportion of pairs to KEEP)
    """
    # If not configured or empty, return as is
    if not gb.DOWNSAMPLE_PAIRS_AT_RANK:
        return pairwise_ranks
    
    # Build lookup array: index = internal_rank + 1, value = proportion to KEEP
    rank_proportions = np.ones(9, dtype=np.float64)
    
    for rank_idx, proportion in gb.DOWNSAMPLE_PAIRS_AT_RANK:
        # Map config rank to lookup index (0-8)
        if 0 <= rank_idx <= 8:
            rank_proportions[rank_idx] = max(0.0, proportion)
    
    # Check if any downsampling is actually needed
    if np.all(rank_proportions >= 0.999):
        return pairwise_ranks
    
    # Get flattened view
    flat_ranks = pairwise_ranks.ravel()
    
    # Sync Numba's RNG with our global _RNG
    current_seed = _RNG.integers(0, 2**31 - 1) # We use 2**31 - 1 to stay safely within int32 range for legacy compatibility.
    np.random.seed(current_seed)
    
    # Call parallel JIT kernel (no pre-generated randoms needed)
    _rank_based_subsampling_kernel(flat_ranks, rank_proportions)
    
    return pairwise_ranks


@njit(parallel=True, cache=True)
def _taxon_size_based_subsampling_kernel(flat_ranks, n_seqs, taxon_counts, baselines, 
                                          alphas, min_keeps, warmup_phase):
    """
    Numba JIT-compiled parallel kernel for taxon-size-based subsampling.
    
    Single-pass parallel algorithm that probabilistically drops pairs from large taxa
    by setting their rank to -2 (Ignore Pair). Modifies flat_ranks in-place.
    
    Row and column indices are computed on-the-fly from the flat index to avoid
    creating large index arrays (saves ~2.4GB allocation for 10k x 10k matrices).
    
    For each pair (i, j) at pairwise rank r in [-1, 5]:
      - taxon_idx = r + 1 (maps pairwise rank to taxon counts index 0-6)
      - size_pair = sqrt(count_i * count_j) where counts are from taxon_counts[:, taxon_idx]
      - p = min(1, (baseline / size_pair) ** alpha)
      - p = max(p, min_keep)  # optional floor
      - During warmup, blend: p = warmup_phase * 1.0 + (1 - warmup_phase) * p
      - Drop pair (set rank to -2) with probability (1 - p)
    
    Args:
        flat_ranks: np.ndarray of shape (n_elements,) dtype int8
            Flattened pairwise ranks matrix, modified in-place.
        n_seqs: int
            Number of sequences (matrix dimension). Used to compute row/col indices.
        taxon_counts: np.ndarray of shape (n_seqs, 7) dtype int32
            Per-sequence taxon sizes at each rank (domain=0, ..., species=6).
        baselines: np.ndarray of shape (7,) dtype float32
            Per-rank baseline taxon size for the keep probability formula.
        alphas: np.ndarray of shape (7,) dtype float64
            Per-rank alpha exponent (0=off, 0.5=gentle, 1=strong).
        min_keeps: np.ndarray of shape (7,) dtype float64
            Per-rank minimum keep probability floor (use 0.0 to disable).
        warmup_phase: float
            Warmup blend factor in [0, 1]. 1.0 = full warmup (no bias), 0.0 = full bias.
    """
    n_elements = flat_ranks.shape[0]
    
    for idx in prange(n_elements):
        rank = flat_ranks[idx]
        
        # Skip if rank is outside the bias range [-1, 5] or already ignored
        # Ranks 6 (sequence) and 7 (subsequence) get no bias
        if rank < -1 or rank > 5:
            continue
        
        # Map pairwise rank to taxon counts index: -1 -> 0 (domain), 0 -> 1 (phylum), etc.
        taxon_idx = rank + 1
        
        alpha = alphas[taxon_idx]
        
        # Skip if alpha is 0 (bias disabled for this rank)
        if alpha <= 0.0:
            continue
        
        # Compute row and column indices on-the-fly from flat index
        # This avoids allocating large index arrays (2.4GB for 10k x 10k matrix)
        row_idx = idx // n_seqs
        col_idx = idx % n_seqs
        
        # Get taxon counts for both sequences at this rank
        count_i = taxon_counts[row_idx, taxon_idx]
        count_j = taxon_counts[col_idx, taxon_idx]
        
        # Compute pair taxon size: sqrt(count_i * count_j)
        size_pair = np.sqrt(float(count_i) * float(count_j))
        
        # Get baseline for this rank
        baseline = baselines[taxon_idx]
        
        # Compute keep probability: p = min(1, (baseline / size_pair) ** alpha)
        if size_pair <= 0.0 or baseline <= 0.0:
            # Edge case: if size is 0 or baseline is 0, keep the pair
            keep_prob = 1.0
        else:
            ratio = baseline / size_pair
            if ratio >= 1.0:
                keep_prob = 1.0
            else:
                keep_prob = ratio ** alpha
        
        # Apply minimum keep floor
        min_keep = min_keeps[taxon_idx]
        if keep_prob < min_keep:
            keep_prob = min_keep
        
        # Apply warmup blending: during warmup, blend toward p=1 (no bias)
        if warmup_phase > 0.0:
            keep_prob = warmup_phase * 1.0 + (1.0 - warmup_phase) * keep_prob
        
        # Skip RNG if keeping all
        if keep_prob >= 0.999:
            continue
        
        # Drop pair with probability (1 - keep_prob)
        if np.random.random() > keep_prob:
            flat_ranks[idx] = -2


def taxon_size_based_subsampling(pairwise_ranks, taxon_counts, warmup_phase=0.0):
    """
    Probabilistically downsample pairs from large taxa by setting them to -2 (Ignore Pair).
    
    This step biases mining away from dominant taxa by giving pairs from large taxa a lower
    keep probability. The bias uses a power-law formula around a per-rank baseline:
        p = min(1, (baseline / size_pair) ** alpha)
    where:
        - size_pair = sqrt(count_i * count_j) for the pair's sequences
        - baseline is the per-rank baseline taxon size (computed at dataset load time)
        - alpha controls strength (0=off, 0.5=gentle, 1=strong)
    
    The warmup_phase parameter blends the keep probability toward 1.0 (no bias) during
    early training, allowing the bias to gradually take effect.
    
    Args:
        pairwise_ranks: np.ndarray of shape (n_seqs, n_seqs) dtype int8
            The pairwise ranks matrix. Modified in-place.
        taxon_counts: np.ndarray of shape (n_seqs, 7) dtype int32
            Per-sequence taxon sizes at each rank (domain through species).
        warmup_phase: float in [0, 1]
            Blend factor for warmup. 1.0 = full warmup (no bias), 0.0 = full bias.
    
    Returns:
        pairwise_ranks: np.ndarray of shape (n_seqs, n_seqs) dtype int8
            The modified pairwise ranks matrix.
    
    Global variables used:
        gb.USE_TAXON_SIZE_MINING_BIAS: bool - whether to apply this bias
        gb.TAXON_SIZE_MINING_BIAS_ALPHAS_PER_RANK: list/tuple of 7 floats - per-rank alpha values
        gb.TAXON_SIZE_MINING_BIAS_MIN_KEEP_PER_RANK: list/tuple of 7 floats or None - per-rank min keep floors
        gb.TRAIN_TAXON_BASELINE_COUNT_PER_RANK: np.ndarray of shape (7,) - per-rank baselines
    """
    # Skip if disabled
    if not gb.USE_TAXON_SIZE_MINING_BIAS:
        return pairwise_ranks
    
    # Check if all alphas are 0 (bias disabled for all ranks)
    alphas = np.array(gb.TAXON_SIZE_MINING_BIAS_ALPHAS_PER_RANK, dtype=np.float64)
    if np.all(alphas <= 0.0):
        return pairwise_ranks
    
    # Get baselines (computed at dataset load time)
    baselines = gb.TRAIN_TAXON_BASELINE_COUNT_PER_RANK.astype(np.float32)
    
    # Get min keep floors (default to 0.0 if not specified)
    if gb.TAXON_SIZE_MINING_BIAS_MIN_KEEP_PER_RANK is not None:
        min_keeps = np.array(gb.TAXON_SIZE_MINING_BIAS_MIN_KEEP_PER_RANK, dtype=np.float64)
    else:
        min_keeps = np.zeros(7, dtype=np.float64)
    
    # Ensure taxon_counts is int32 for the kernel
    if taxon_counts.dtype != np.int32:
        taxon_counts = taxon_counts.astype(np.int32)
    
    # Get matrix dimensions and flattened view
    n_seqs = pairwise_ranks.shape[0]
    flat_ranks = pairwise_ranks.ravel()
    
    # Sync Numba's RNG with our global _RNG
    current_seed = _RNG.integers(0, 2**31 - 1)
    np.random.seed(current_seed)
    
    # Call parallel JIT kernel
    # Note: Row/col indices are computed on-the-fly in the kernel to avoid
    # allocating large index arrays (saves ~2.4GB for 10k x 10k matrices)
    _taxon_size_based_subsampling_kernel(
        flat_ranks, n_seqs, taxon_counts, baselines, alphas, min_keeps, warmup_phase
    )
    
    return pairwise_ranks


def apply_region_selection_and_variations(train_sequences, duplicate_sequences=None, duplicate_indices=None):
    """
    Apply region selection and sequence variations to training and duplicate sequences.
    
    This function performs efficient region sampling and data augmentation for subsequence pair mining:
    1. Randomly selects one region per sequence (either full sequence or a subsequence)
    2. Enforces cross-region sampling for duplicates (when gb.SUBSEQUENCES_ALWAYS_CROSS_REGION=True)
    3. Extracts selected regions using vectorized numpy indexing (avoids expensive tensor copying)
    4. Applies independent variations (mutations, truncations, shifts) to each sequence
    5. Converts to torch tensors and optionally pins memory for fast GPU transfer
    
    Region Selection Configuration (via globals):
        gb.USE_FULL_SEQS: If True, can sample the full sequence (region index 0)
        gb.USE_SUB_SEQS: If True, can sample subsequences (region indices 1 to N_REGIONS-1)
        gb.SUBSEQUENCES_ALWAYS_CROSS_REGION: If True, duplicate sequences are forced to select
            a different region than their original sequence (requires >= 2 available regions)
    
    Region Indexing:
        - Region 0: Full sequence (entire sequence without truncation)
        - Regions 1+: Subsequences (various subsequence extractions/positions)
        - min_region_idx: First eligible region (0 if USE_FULL_SEQS, else 1)
        - max_region_idx: One past last eligible region (N_REGIONS if USE_SUB_SEQS, else 1)
    
    Args:
        train_sequences: np.ndarray of shape (N_TRAIN_SEQS, N_REGIONS, MAX_IMPORTED_SEQ_LEN, 3)
            Raw 3-bit encoded training sequences with all regions pre-computed.
            Each sequence has N_REGIONS variants (1 full + N_REGIONS-1 subsequences).
        duplicate_sequences: np.ndarray of shape (N_DUPLICATES, N_REGIONS, MAX_IMPORTED_SEQ_LEN, 3) or None
            Duplicate sequences for subsequence pair mining. These are copies of sequences
            from train_sequences that will be sampled with different regions to create
            positive pairs within the same original sequence.
        duplicate_indices: np.ndarray of shape (N_DUPLICATES,) dtype=int, or None
            Maps each duplicate back to its original sequence in train_sequences.
            duplicate_sequences[i] is a copy of train_sequences[duplicate_indices[i]].
            Used for cross-region enforcement.
    
    Returns:
        train_sequences: torch.Tensor of shape (N_TRAIN_SEQS, MAX_MODEL_SEQ_LEN, 3)
            Training sequences with region selection and variations applied.
            Lives on CPU and is pinned if gb.PIN_MEMORY_FOR_MINING is True.
        duplicate_sequences: torch.Tensor of shape (N_DUPLICATES, MAX_MODEL_SEQ_LEN, 3) or None
            Duplicate sequences with region selection and variations applied.
            Lives on CPU and is pinned if gb.PIN_MEMORY_FOR_MINING is True.
        train_regions: np.ndarray of shape (N_TRAIN_SEQS,) dtype=int
            Selected region index for each training sequence (values in [min_region_idx, max_region_idx)).
        duplicate_regions: np.ndarray of shape (N_DUPLICATES,) dtype=int, or None
            Selected region index for each duplicate sequence (values in [min_region_idx, max_region_idx)).
    """
    # Get number of training sequences
    # train_sequences.shape: (N_TRAIN_SEQS, N_REGIONS, MAX_IMPORTED_SEQ_LEN, 3)
    n_train = train_sequences.shape[0]
    has_duplicates = duplicate_sequences is not None and len(duplicate_sequences) > 0
    
    # 1. Determine eligible region range ---
    # Region 0 = full sequence, regions 1+ = subsequences
    n_regions = train_sequences.shape[1]
    min_region_idx = 0 if gb.USE_FULL_SEQS else 1  # First eligible region
    max_region_idx = n_regions if gb.USE_SUB_SEQS else 1  # One past last eligible region
    
    # Validate configuration: need at least one eligible region
    if max_region_idx <= min_region_idx:
        raise ValueError(f"No regions to sample from with given configuration. "
                         f"use_full_seqs={gb.USE_FULL_SEQS}, use_sub_seqs={gb.USE_SUB_SEQS}, n_regions={n_regions}")

    # 2. Randomly select regions for training sequences ---
    # Sample random region indices for each training sequence
    # train_region_indices.shape: (N_TRAIN_SEQS,) with values in [min_region_idx, max_region_idx)
    train_region_indices = _RNG.integers(min_region_idx, max_region_idx, size=n_train)
    
    # 3. Select regions for duplicate sequences ---
    if has_duplicates:
        # duplicate_sequences.shape: (N_DUPLICATES, N_REGIONS, MAX_IMPORTED_SEQ_LEN, 3)
        n_duplicates = duplicate_sequences.shape[0]
        
        # Initially sample random regions for duplicates (may conflict with originals)
        # duplicate_region_indices.shape: (N_DUPLICATES,) with values in [min_region_idx, max_region_idx)
        duplicate_region_indices = _RNG.integers(min_region_idx, max_region_idx, size=n_duplicates)
        
        # Enforce cross-region constraint: duplicates must use different regions than their originals
        if gb.SUBSEQUENCES_ALWAYS_CROSS_REGION:
            # Need at least 2 regions to enforce different regions
            if (max_region_idx - min_region_idx) < 2:
                raise ValueError("Cannot enforce different regions when available regions < 2")
            
            # Vectorized conflict detection and resolution
            # For each duplicate, get the region selected by its original sequence
            # original_regions.shape: (N_DUPLICATES,)
            original_regions = train_region_indices[duplicate_indices]
            
            # Find which duplicates conflict (same region as original)
            # conflicts.shape: (N_DUPLICATES,) dtype=bool
            conflicts = duplicate_region_indices == original_regions
            
            # Resample conflicting duplicates until no conflicts remain
            # Usually converges in 1-2 iterations (probability of conflict = 1/(max-min))
            while np.any(conflicts):
                n_conflicts = np.sum(conflicts)
                # Resample only the conflicting entries
                duplicate_region_indices[conflicts] = _RNG.integers(min_region_idx, max_region_idx, size=n_conflicts)
                # Recheck for conflicts
                conflicts = duplicate_region_indices == original_regions
            
    else:
        # No duplicates: create empty arrays for consistent return types
        n_duplicates = 0
        duplicate_region_indices = np.array([], dtype=int)
        
    # 4. Extract selected regions using advanced indexing ---
    # Use numpy's advanced indexing to select one region per sequence
    
    # For training sequences: select train_region_indices[i] from train_sequences[i]
    # Advanced indexing: arr[batch_indices, region_indices] extracts arr[i, region_indices[i]]
    # train_seqs_selected.shape: (N_TRAIN_SEQS, MAX_IMPORTED_SEQ_LEN, 3)
    train_seqs_selected = train_sequences[np.arange(n_train), train_region_indices]
    
    if has_duplicates:
        # For duplicate sequences: select duplicate_region_indices[i] from duplicate_sequences[i]
        # duplicate_seqs_selected.shape: (N_DUPLICATES, MAX_IMPORTED_SEQ_LEN, 3)
        duplicate_seqs_selected = duplicate_sequences[np.arange(n_duplicates), duplicate_region_indices]
        
        # Concatenate train and duplicate sequences for joint augmentation
        # Only concatenating selected regions (small) instead of all regions (large)
        # combined_seqs.shape: (N_TRAIN_SEQS + N_DUPLICATES, MAX_IMPORTED_SEQ_LEN, 3)
        combined_seqs = np.concatenate([train_seqs_selected, duplicate_seqs_selected], axis=0)
        # combined_regions.shape: (N_TRAIN_SEQS + N_DUPLICATES,)
        combined_regions = np.concatenate([train_region_indices, duplicate_region_indices])
    else:
        # No duplicates: use only training sequences
        combined_seqs = train_seqs_selected
        combined_regions = train_region_indices

    # 5. Apply independent data augmentation to each sequence ---
    # Generate variations (mutations, truncations, shifts) for all sequences
    # Each sequence gets independent random augmentation, so even if a duplicate and its
    # original happened to sample the same region (when cross-region not enforced), they
    # will still differ due to different mutations/truncations/shifts.
    # 
    # Augmentation operations:
    # - Mutations: Random nucleotide substitutions at rate gb.MUTATION_RATE
    # - Truncations: Random start/end truncation within [MIN_TRUNC_START, MAX_TRUNC_START] 
    #                and [MIN_TRUNC_END, MAX_TRUNC_END]
    # - Shifts: Random left/right shifts applied to gb.PROP_SHIFT_SEQS proportion of sequences
    # - Padding/truncation to target length: Ensures output is exactly MAX_MODEL_SEQ_LEN
    combined_seqs = gen_seq_variants(combined_seqs, mutation_rate=gb.MUTATION_RATE, trunc_prop=gb.PROP_TRUNC, min_trunc_start=gb.MIN_TRUNC_START, max_trunc_start=gb.MAX_TRUNC_START, min_trunc_end=gb.MIN_TRUNC_END, max_trunc_end=gb.MAX_TRUNC_END, shift_prop=gb.PROP_SHIFT_SEQS, target_seq_len=gb.MAX_MODEL_SEQ_LEN)
    # combined_seqs.shape: (N_TRAIN_SEQS + N_DUPLICATES, MAX_MODEL_SEQ_LEN, 3)
    # Note: MAX_MODEL_SEQ_LEN may differ from MAX_IMPORTED_SEQ_LEN (e.g., after truncation/padding)

    # 6. Convert to torch tensors ---
    # gen_seq_variants may return numpy or torch depending on implementation
    if isinstance(combined_seqs, np.ndarray):
        combined_seqs = torch.from_numpy(combined_seqs)
    
    # 7. Split combined sequences back into train and duplicate ---
    # Slice the concatenated tensor back into separate train and duplicate tensors
    # train_seqs_out.shape: (N_TRAIN_SEQS, MAX_MODEL_SEQ_LEN, 3)
    train_seqs_out = combined_seqs[:n_train]
    # train_regions.shape: (N_TRAIN_SEQS,)
    train_regions = combined_regions[:n_train]
    
    if has_duplicates:
        # duplicate_seqs_out.shape: (N_DUPLICATES, MAX_MODEL_SEQ_LEN, 3)
        duplicate_seqs_out = combined_seqs[n_train:]
        # duplicate_regions.shape: (N_DUPLICATES,)
        duplicate_regions = combined_regions[n_train:]
    else:
        # No duplicates case: return None for duplicate outputs
        duplicate_seqs_out = None
        duplicate_regions = None
    
    # 8. Pin memory for efficient GPU transfers ---
    # If gb.PIN_MEMORY_FOR_MINING is True, pin CPU tensors for faster H2D transfers
    # Pinned memory allows asynchronous DMA transfers without CPU page locking overhead
    train_seqs_out = _maybe_pin_cpu_tensor(train_seqs_out)
    if duplicate_seqs_out is not None:
        duplicate_seqs_out = _maybe_pin_cpu_tensor(duplicate_seqs_out)
    
    return train_seqs_out, duplicate_seqs_out, train_regions, duplicate_regions


def get_seq_embeddings(model, sequences):
    """
    Compute embeddings for sequences using the model.
    
    This function:
    1. Determines the device the model is currently on
    2. Runs batched inference
    3. Returns the embeddings as a torch tensor on the model's device
    
    Args:
        model: The Micro16S model
        sequences: torch.Tensor
            - shape (N_SEQUENCES, MAX_MODEL_SEQ_LEN, 3)
    
    Returns:
        embeddings: torch.Tensor of shape (N_SEQUENCES, EMBED_DIMS)
            The computed embeddings for all sequences on the model's current device.
    """
    
    # Get the device the model is currently on
    device = next(model.parameters()).device
    
    # Ensure the sequences are tensors for inference and pin if desired
    if not isinstance(sequences, torch.Tensor):
        sequences = torch.from_numpy(sequences)
    sequences = _maybe_pin_cpu_tensor(sequences)
    
    # Run inference
    # Sequences are 3-bit encoded
    # sequences.shape: (N_SEQUENCES, MAX_MODEL_SEQ_LEN, 3)
    embeddings = run_inference(
        model,
        sequences,
        device=device,
        batch_size=gb.MINING_BATCH_SIZE,
        output_device=device,
        return_numpy=False,
        pin_inputs=gb.PIN_MEMORY_FOR_MINING
    )
    # embeddings.shape: (N_SEQUENCES, EMBED_DIMS)
    
    return embeddings


def get_seq_embeddings_seqless_mode(model, indices):
    """
    Get embeddings for sequences in sequenceless mode using their indices.
    
    In sequenceless mode, the model is a lookup table of embeddings indexed by
    training set indices. This function converts indices to the appropriate format
    and retrieves the corresponding embeddings from the model.
    
    Args:
        model: The SequencelessMicro16S model (embedding lookup table)
        indices: np.ndarray of shape (N_SEQUENCES,)
            Global training set indices (0..n_train-1) for which to retrieve embeddings.
    
    Returns:
        embeddings: torch.Tensor of shape (N_SEQUENCES, EMBED_DIMS)
            The embeddings corresponding to the given indices, on the model's device.
    """
    # Get the device the model is currently on
    device = next(model.parameters()).device
    
    # Convert indices to torch tensor if needed
    if not isinstance(indices, torch.Tensor):
        indices = torch.from_numpy(indices)
    
    # Ensure indices are long (int64) for embedding lookup
    if indices.dtype != torch.long:
        indices = indices.long()
    
    # Move indices to model's device
    indices = indices.to(device)
    
    # Get embeddings from the lookup table
    # model(indices) returns L2-normalized embeddings
    with torch.no_grad():
        embeddings = model(indices)
    # embeddings.shape: (N_SEQUENCES, EMBED_DIMS)
    
    return embeddings


def compute_seq_embeddings_distances(train_embeddings):
    """
    Compute pairwise cosine distances between all training sequence embeddings.
    
    This function:
    1. Converts embeddings to torch tensors on the best available device
    2. L2-normalizes the embeddings
    3. Computes pairwise cosine similarities via matrix multiplication
    4. Converts similarities to distances (distance = 1 - similarity)
    5. Returns the distance matrix as a numpy array
    
    The cosine distance is computed as: d(u, v) = 1 - cos_sim(u, v)
    where cos_sim(u, v) = (u · v) / (||u|| ||v||)
    
    After L2-normalization, the cosine similarity simplifies to: cos_sim(u, v) = u · v
    
    Args:
        train_embeddings: torch.Tensor or np.ndarray of shape (N_TRAIN_SEQUENCES, EMBED_DIMS)
            The embeddings for all training sequences. Tensors can reside on CPU or GPU.
    
    Returns:
        distance_matrix: np.ndarray of shape (N_TRAIN_SEQUENCES, N_TRAIN_SEQUENCES)
            The pairwise cosine distance matrix where distance_matrix[i, j] is the
            cosine distance between sequence i and sequence j.
    """
    
    # Convert inputs to a torch tensor on the best available device
    if isinstance(train_embeddings, torch.Tensor):
        embeddings_torch = train_embeddings.detach()
    elif isinstance(train_embeddings, np.ndarray):
        embeddings_torch = torch.from_numpy(train_embeddings)
    else:
        raise TypeError(f"train_embeddings must be a torch.Tensor or np.ndarray, got {type(train_embeddings)}")

    if embeddings_torch.dtype != torch.float32:
        embeddings_torch = embeddings_torch.float()
    
    target_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if embeddings_torch.device != target_device:
        embeddings_torch = embeddings_torch.to(target_device, non_blocking=True)
    
    compute_start = time.time()

    # L2-normalize the embeddings
    # After normalization, cosine similarity = dot product
    epsilon = 1e-8
    embeddings_normalized = torch.nn.functional.normalize(embeddings_torch, p=2, dim=1, eps=epsilon)
    
    # Compute pairwise cosine similarities using matrix multiplication
    # similarity_matrix[i, j] = embeddings_normalized[i] · embeddings_normalized[j]
    # embeddings_normalized: (N_TRAIN_SEQUENCES, EMBED_DIMS)
    # embeddings_normalized.T: (EMBED_DIMS, N_TRAIN_SEQUENCES)
    # similarity_matrix: (N_TRAIN_SEQUENCES, N_TRAIN_SEQUENCES)
    similarity_matrix = torch.matmul(embeddings_normalized, embeddings_normalized.T)
    
    # Convert similarities to distances
    # cosine_distance = 1 - cosine_similarity
    distance_matrix = 1.0 - similarity_matrix
    
    # Optionally convert to float16 to halve transfer time
    if gb.USE_FLOAT_16_FOR_MINING_COSINE_DISTANCES and target_device.type == 'cuda':
        distance_matrix = distance_matrix.to(torch.float16)

    num_bytes = distance_matrix.numel() * distance_matrix.element_size()

    # Ensure GPU kernels are finished before reporting compute time
    synchronize_if_cuda(distance_matrix)
    compute_time = time.time() - compute_start
    
    transfer_start = time.time()
    # Move back to CPU and convert to numpy
    distance_matrix_np = distance_matrix.cpu().numpy()
    transfer_time = time.time() - transfer_start

    # Print timing/size info for the GPU->CPU transfer.
    device_desc = 'cpu' if target_device.type != 'cuda' else f"cuda:{target_device.index if target_device.index is not None else torch.cuda.current_device()}"
    if gb.VERBOSE_MINING_TIMING:    
        print(f"    > Distance matrix computation ({device_desc}): {compute_time:.4f} seconds")
        print(f"    > Distance matrix CPU transfer: {transfer_time:.4f} seconds ({num_bytes / (1024 ** 2):.1f} MB)")
    
    return distance_matrix_np


def mine_pairs_and_triplets(model, n_triplets_to_mine, n_pairs_to_mine, verbose_pairs=False, verbose_triplets=False, log_pairs=False, log_triplets=False, log=False, batch_num=None, logs_dir=None, triplet_satisfaction_df=None, pair_distances_df=None, triplet_error_metrics_df=None, pair_error_metrics_df=None):
    """
    Mine pairs and triplets for a single batch.
    
    Returns CPU tensors that are pinned when gb.PIN_MEMORY_FOR_MINING is enabled so the training
    step can stream them to the GPU without blocking.
    
    Duplicate sequences (for subsequence pair mining, rank 8) are handled separately from the
    main training sequences.
    
    Args:
        log: Optional bool to enable mining-level logging for this run.
        triplet_satisfaction_df: Optional pd.DataFrame for tracking triplet satisfaction over time.
                                 If provided, will be populated with satisfaction metrics during logging.
        pair_distances_df: Optional pd.DataFrame for tracking pair distances over time.
                          If provided, will be populated with distance statistics during logging.
        triplet_error_metrics_df: Optional pd.DataFrame for tracking triplet per-rank hardness metrics.
                                  If provided, will be populated with the EMA input metric during logging.
        pair_error_metrics_df: Optional pd.DataFrame for tracking pair per-rank error metrics.
                               If provided, will be populated with the EMA input metric during logging.
    """
    func_start_time = time.time()

    # Calculate Triplet Warmup Phase
    triplet_uniform_duration = gb.TRIPLET_MINING_WARMUP_UNIFORM_DURATION or 0
    triplet_transition_duration = gb.TRIPLET_MINING_WARMUP_TRANSITION_DURATION or 0
    triplet_total_warmup = triplet_uniform_duration + triplet_transition_duration
    
    if batch_num is None or triplet_total_warmup == 0:
        triplet_warmup_phase = 0.0
    elif batch_num <= triplet_uniform_duration:
        triplet_warmup_phase = 1.0
    elif batch_num <= triplet_total_warmup:
        triplet_warmup_phase = 1.0 - (batch_num - triplet_uniform_duration) / max(triplet_transition_duration, 1)
    else:
        triplet_warmup_phase = 0.0

    # Calculate Pair Warmup Phase
    pair_uniform_duration = gb.PAIR_MINING_WARMUP_UNIFORM_DURATION or 0
    pair_transition_duration = gb.PAIR_MINING_WARMUP_TRANSITION_DURATION or 0
    pair_total_warmup = pair_uniform_duration + pair_transition_duration

    if batch_num is None or pair_total_warmup == 0:
        pair_warmup_phase = 0.0
    elif batch_num <= pair_uniform_duration:
        pair_warmup_phase = 1.0
    elif batch_num <= pair_total_warmup:
        pair_warmup_phase = 1.0 - (batch_num - pair_uniform_duration) / max(pair_transition_duration, 1)
    else:
        pair_warmup_phase = 0.0

    do_phylum_logging = (
        getattr(gb, "LOG_MINING_PHYLA_COUNTS", False)
        and logs_dir is not None
        and batch_num is not None
        and (batch_num == 1 or batch_num % gb.LOG_EVERY_N_BATCHES == 0)
    )
    phylum_step_stats = []
    local_phylum_codes = None
    phylum_labels = None


    # 1. Get the raw training sequences (all regions, no variations) --------------------------------------------------
    part_start_time = time.time()
    train_sequences = get_train_sequences()
    # train_sequences.shape: (N_TRAIN_SEQUENCES, N_REGIONS, MAX_IMPORTED_SEQ_LEN, 3)
    if gb.VERBOSE_MINING_TIMING:
        print(f"  > Time taken to get raw training sequences: {time.time() - part_start_time:.4f} seconds")

 
    # 2. Subsample training sequences to reduce compute --------------------------------------------------
    part_start_time = time.time()
    # Note: From this point onwards, pairwise_ranks may contain -2 ("Ignore Pair") values.
    train_sequences, adjusted_distances_subsampled, pairwise_ranks_subsampled, taxon_counts_subsampled, domain_counts, subsample_indices = subsample_train_sequences_and_matrices(train_sequences)
    # train_sequences.shape: (n_seqs_to_use, N_REGIONS, MAX_IMPORTED_SEQ_LEN, 3)
    # adjusted_distances_subsampled.shape: (n_seqs_to_use, n_seqs_to_use)
    # pairwise_ranks_subsampled.shape: (n_seqs_to_use, n_seqs_to_use)
    # taxon_counts_subsampled.shape: (n_seqs_to_use, 7)
    # subsample_indices.shape: (n_seqs_to_use,) - global training set indices (0..n_train-1)
    # pairwise_ranks_subsampled contains ranks as integers: -2 to 7, where:
    #   - -2 = Ignore Pair (excluded from mining)
    #   - -1 = different domains (Bacteria and Archaea), both prokaryotes
    #   -  0 = same domain, different phyla
    #   -  1 = same phylum, different classes
    #   -  2 = same class, different orders
    #   -  3 = same order, different families
    #   -  4 = same family, different genera
    #   -  5 = same genus, different species
    #   -  6 = same species, different sequences
    #   -  7 = duplicate pairs (same sequence, region relationship optional; not present yet because duplicates are held separately)
    if gb.VERBOSE_MINING_TIMING:
        print(f"  > Time taken to subsample training sequences: {time.time() - part_start_time:.4f} seconds")

    n_bacteria_subsampled = domain_counts.get('bacteria', 0)
    n_archaea_subsampled = domain_counts.get('archaea', 0)

    if do_phylum_logging:
        local_phylum_codes, phylum_labels = _get_local_phylum_metadata(subsample_indices)
        if local_phylum_codes is None or phylum_labels is None:
            do_phylum_logging = False
        else:
            n_phyla = len(phylum_labels)
            phylum_step_stats.append(
                _build_step_phylum_stats(
                    "post_sequence_subsample",
                    pairwise_ranks_subsampled,
                    local_phylum_codes,
                    n_phyla,
                )
            )

    if gb.LOG_MINING_ARC_BAC_COUNTS:
        should_log_arc_bac_counts = (
            logs_dir is not None
            and batch_num is not None
            and (batch_num == 1 or batch_num % gb.LOG_EVERY_N_BATCHES == 0)
        )
        if should_log_arc_bac_counts:
            log_arc_bac_counts(batch_num, logs_dir, num_archaea=n_archaea_subsampled, num_bacteria=n_bacteria_subsampled)

    n_train_seqs = train_sequences.shape[0]
    # print("  > n_seqs_to_use = ", n_train_seqs)

    # Correct taxon sizes based on domain-based downsampling for representative-set balancing.
    # The raw taxon counts reflect the full training set, but mining only sees a domain-stratified
    # subsample (e.g. 15% of Bacteria, 100% of Archaea). Scaling each sequence's counts by its
    # domain's subsample fraction gives weights that reflect actual mining pool composition.
    if gb.USE_REPRESENTATIVE_TAXON_SIZE_BALANCING:
        seq_domain_ids = gb.TRAIN_SEQ_TAXON_IDS[subsample_indices, 0]
        bacteria_id = _get_domain_taxon_id(gb.TRAIN_TAXON_LABEL_TO_TAXON_ID, "Bacteria")
        archaea_id = _get_domain_taxon_id(gb.TRAIN_TAXON_LABEL_TO_TAXON_ID, "Archaea")
        domain_scales = np.ones(len(subsample_indices), dtype=np.float32)
        domain_scales[seq_domain_ids == bacteria_id] = gb.MINING_TRAIN_SEQUENCES_SUB_SAMPLE_BACTERIA
        domain_scales[seq_domain_ids == archaea_id] = gb.MINING_TRAIN_SEQUENCES_SUB_SAMPLE_ARCHAEA
        taxon_counts_for_rep_balancing = taxon_counts_subsampled.astype(np.float32) * domain_scales[:, np.newaxis]
    else:
        taxon_counts_for_rep_balancing = taxon_counts_subsampled


    # 3. Generate duplicate sequences for subsequence pair mining (rank 8) --------------------------------------------------
    # Duplicates are copies of the same sequence that can be forced onto different regions and always get independent variations.
    # The implicit data for duplicates is:
    #   - Rank between duplicate and its original = 7 (same sequence, subsequence relationship configurable)
    #   - Distance between duplicate and its original = 0
    duplicate_sequences = None
    duplicate_indices = None
    effective_pair_ranks = get_effective_pair_ranks(batch_num=batch_num)
    if gb.N_PAIRS_PER_BATCH > 0 and effective_pair_ranks[8] and gb.PROPORTION_IDENTICAL_SEQUENCES_TO_GENERATE > 0:
        part_start_time = time.time()
        duplicate_sequences, duplicate_indices = add_duplicate_sequences(train_sequences, gb.PROPORTION_IDENTICAL_SEQUENCES_TO_GENERATE)
        # duplicate_sequences.shape: (n_duplicates, N_REGIONS, MAX_IMPORTED_SEQ_LEN, 3)
        # duplicate_indices.shape: (n_duplicates,) - indices into train_sequences for each duplicate's original
        n_duplicates = len(duplicate_indices) if duplicate_indices is not None else 0
        if gb.VERBOSE_MINING_TIMING:
            print(f"  > Time taken to generate duplicate sequences: {time.time() - part_start_time:.4f} seconds (n_duplicates={n_duplicates})")


    # 4. Perform rank-based subsampling --------------------------------------------------
    part_start_time = time.time()
    # This sets pairs at specific ranks to -2 (Ignore Pair) based on gb.DOWNSAMPLE_PAIRS_AT_RANK. Does not affect duplicates.
    pairwise_ranks_subsampled = rank_based_subsampling(pairwise_ranks_subsampled)
    # pairwise_ranks_subsampled.shape: (n_seqs_to_use, n_seqs_to_use)
    if gb.VERBOSE_MINING_TIMING:
        print(f"  > Time taken to perform rank-based subsampling: {time.time() - part_start_time:.4f} seconds")
    if do_phylum_logging:
        phylum_step_stats.append(
            _build_step_phylum_stats(
                "post_rank_subsample",
                pairwise_ranks_subsampled,
                local_phylum_codes,
                len(phylum_labels),
            )
        )


    # 5. Perform taxon-size-based subsampling --------------------------------------------------
    # This probabilistically drops pairs from large taxa to reduce their dominance in mining.
    # The warmup phase follows the max of pair/triplet warmup phases (bias off during warmup).
    pre_taxon_subsampling_ignored = None
    taxon_subsampling_counts_pre = None
    taxon_subsampling_counts_post = None
    if log and gb.USE_TAXON_SIZE_MINING_BIAS:
        taxon_subsampling_counts_pre = _count_pairwise_rank_bins(pairwise_ranks_subsampled)
        pre_taxon_subsampling_ignored = int(taxon_subsampling_counts_pre[0])
    if gb.USE_TAXON_SIZE_MINING_BIAS:
        part_start_time = time.time()
        # Use the maximum warmup phase so bias is off when either pair or triplet warmup is active
        taxon_bias_warmup_phase = max(pair_warmup_phase, triplet_warmup_phase)
        pairwise_ranks_subsampled = taxon_size_based_subsampling(pairwise_ranks_subsampled, taxon_counts_subsampled, warmup_phase=taxon_bias_warmup_phase)
        # pairwise_ranks_subsampled.shape: (n_seqs_to_use, n_seqs_to_use)
        if gb.VERBOSE_MINING_TIMING:
            print(f"  > Time taken for taxon-size-based subsampling: {time.time() - part_start_time:.4f} seconds")
        if log and taxon_subsampling_counts_pre is not None:
            taxon_subsampling_counts_post = _count_pairwise_rank_bins(pairwise_ranks_subsampled)
    if do_phylum_logging:
        step_name = "post_taxon_size_subsample" if gb.USE_TAXON_SIZE_MINING_BIAS else "post_taxon_size_subsample_disabled"
        phylum_step_stats.append(
            _build_step_phylum_stats(
                step_name,
                pairwise_ranks_subsampled,
                local_phylum_codes,
                len(phylum_labels),
            )
        )


    # 6. Apply region selection and sequence variations (skip in sequenceless mode) --------------------------------------------------
    if not gb.SEQLESS_MODE:
        part_start_time = time.time()
        # This combines train and duplicate sequences internally for joint region selection,
        # optionally ensuring duplicates get different regions from their originals, then splits them back.
        train_sequences, duplicate_sequences, train_regions, duplicate_regions = apply_region_selection_and_variations(
            train_sequences, duplicate_sequences, duplicate_indices
        )
        # train_sequences.shape: (n_seqs_to_use, MAX_MODEL_SEQ_LEN, 3)
        # duplicate_sequences.shape: (n_duplicates, MAX_MODEL_SEQ_LEN, 3) or None
        # train_regions.shape: (n_seqs_to_use,)
        # duplicate_regions.shape: (n_duplicates,) or None
        if gb.VERBOSE_MINING_TIMING:
            print(f"  > Time taken for region selection and variations: {time.time() - part_start_time:.4f} seconds")
    else:
        # In sequenceless mode, skip region selection and variations (not applicable)
        # We don't use regions or sequences in the forward pass, only indices
        # However, mining functions still need sequences and regions to construct outputs
        # Create dummy regions (all set to 0) for mining functions to use
        train_regions = np.zeros(n_train_seqs, dtype=np.int16)
        duplicate_regions = None
        
        # Convert numpy arrays to torch tensors (normally done by apply_region_selection_and_variations)
        # Select the first region (index 0 = full sequence) for all sequences
        train_sequences = torch.from_numpy(train_sequences[:, 0, :, :].astype(np.int8))
        if duplicate_sequences is not None and len(duplicate_sequences) > 0:
            duplicate_sequences = torch.from_numpy(duplicate_sequences[:, 0, :, :].astype(np.int8))
        
        # Pin memory if configured for faster GPU transfer (even though we won't use sequences)
        if gb.PIN_MEMORY_FOR_MINING:
            train_sequences = train_sequences.pin_memory()
            if duplicate_sequences is not None:
                duplicate_sequences = duplicate_sequences.pin_memory()


    # 7. Run inference on sequences (or get embeddings from indices in sequenceless mode) --------------------------------------------------
    part_start_time = time.time()
    
    if not gb.SEQLESS_MODE:
        # Standard mode: run inference on sequences
        # Concatenate sequences if we have duplicates
        has_duplicates = duplicate_sequences is not None and len(duplicate_sequences) > 0
        if has_duplicates:
            combined_sequences = torch.cat([train_sequences, duplicate_sequences], dim=0)
        else:
            combined_sequences = train_sequences
        
        # Run an inference pass
        combined_embeddings = get_seq_embeddings(model, combined_sequences)
        # combined_embeddings.shape: (n_train_seqs + n_duplicates, EMBED_DIMS)

        synchronize_if_cuda(combined_embeddings)

        # Split embeddings back into their respective groups
        train_embeddings = combined_embeddings[:n_train_seqs]
        duplicate_embeddings = combined_embeddings[n_train_seqs:] if has_duplicates else None
    else:
        # Sequenceless mode: get embeddings from indices (no sequence processing)
        # Concatenate indices if we have duplicates
        has_duplicates = duplicate_indices is not None and len(duplicate_indices) > 0
        if has_duplicates:
            # Map duplicate_indices (which are local indices into train_sequences) to global indices
            duplicate_global_indices = subsample_indices[duplicate_indices]
            combined_indices = np.concatenate([subsample_indices, duplicate_global_indices])
        else:
            combined_indices = subsample_indices
        
        # Get embeddings from the sequenceless model (lookup table)
        combined_embeddings = get_seq_embeddings_seqless_mode(model, combined_indices)
        # combined_embeddings.shape: (n_train_seqs + n_duplicates, EMBED_DIMS)

        synchronize_if_cuda(combined_embeddings)

        # Split embeddings back into their respective groups
        train_embeddings = combined_embeddings[:n_train_seqs]
        duplicate_embeddings = combined_embeddings[n_train_seqs:] if has_duplicates else None
    
    if gb.VERBOSE_MINING_TIMING:
        print(f"  > Time taken to run inference: {time.time() - part_start_time:.4f} seconds")


    # 8. Compute distances between training sequence embeddings --------------------------------------------------
    part_start_time = time.time()
    if gb.VERBOSE_MINING_TIMING:
        print("  > Starting distance matrix computation...")
    # Note: We only compute distances for original training sequences. For duplicates, we know their distances are 0
    seq_embeddings_distances = compute_seq_embeddings_distances(train_embeddings)
    # seq_embeddings_distances.shape: (n_seqs_to_use, n_seqs_to_use)
    if gb.VERBOSE_MINING_TIMING:
        print(f"  > Time taken to compute distances: {time.time() - part_start_time:.4f} seconds")

    # Note: This can take ~500ms to run
    # print(f"  > Embedding distances: mean: {seq_embeddings_distances.mean():.7f}, std: {seq_embeddings_distances.std():.7f}")


    # 9. If mining triplets --------------------------------------------------
    if n_triplets_to_mine > 0:

        part_start_time = time.time()
        if gb.VERBOSE_MINING_TIMING:
            print("  > Starting triplet mining...")
        # Mine triplets
        mined_triplets, mined_triplet_margins, mined_triplet_ranks, mined_triplet_buckets, mined_triplet_local_indices = mine_triplets(
            train_sequences,
            n_triplets_to_mine,
            seq_embeddings_distances,
            adjusted_distances_subsampled,
            pairwise_ranks_subsampled,
            taxon_counts_subsampled=taxon_counts_for_rep_balancing,
            verbose=verbose_triplets,
            log=log_triplets,
            batch_num=batch_num,
            logs_dir=logs_dir,
            warmup_phase=triplet_warmup_phase,
            triplet_satisfaction_df=triplet_satisfaction_df,
            triplet_error_metrics_df=triplet_error_metrics_df,
            phylum_codes_for_logging=local_phylum_codes if do_phylum_logging else None,
            phylum_step_stats_out=phylum_step_stats if do_phylum_logging else None,
        )
        # mined_triplets.shape: (n_triplets_to_mine, 3, 4**K)  OR  (n_triplets_to_mine, 3, MAX_MODEL_SEQ_LEN, 3)
        # mined_triplet_margins.shape: (n_triplets_to_mine)
        # mined_triplet_ranks.shape: (n_triplets_to_mine)
        # mined_triplet_buckets.shape: (n_triplets_to_mine)
        # mined_triplet_local_indices.shape: (n_triplets_to_mine, 3) - local indices (0..n_seqs_to_use-1)
        
        # Convert local indices to global training set indices (0..n_train-1)
        mined_triplet_indices = subsample_indices[mined_triplet_local_indices]
        # mined_triplet_indices.shape: (n_triplets_to_mine, 3) dtype: int64
        mined_triplet_indices = torch.from_numpy(mined_triplet_indices.astype(np.int64))
        
        if gb.VERBOSE_MINING_TIMING:
            print(f"  > Time taken to mine triplets: {time.time() - part_start_time:.4f} seconds")

    else:
        # No triplets to mine - create empty tensors with appropriate shapes
        # train_sequences.shape is (n_seqs_to_use, MAX_MODEL_SEQ_LEN, 3)
        mined_triplets = torch.empty((0, 3, train_sequences.shape[1], train_sequences.shape[2]), dtype=train_sequences.dtype)
        mined_triplet_margins = torch.empty((0,), dtype=torch.float32)
        mined_triplet_ranks = torch.empty((0,), dtype=torch.int32)
        mined_triplet_buckets = torch.empty((0,), dtype=torch.long)
        mined_triplet_local_indices = np.empty((0, 3), dtype=np.int64)
        mined_triplet_indices = torch.empty((0, 3), dtype=torch.long)
    
    # Pin triplets CPU tensors
    mined_triplets = _maybe_pin_cpu_tensor(mined_triplets)
    mined_triplet_margins = _maybe_pin_cpu_tensor(mined_triplet_margins)
    mined_triplet_ranks = _maybe_pin_cpu_tensor(mined_triplet_ranks)
    mined_triplet_buckets = _maybe_pin_cpu_tensor(mined_triplet_buckets)
    mined_triplet_indices = _maybe_pin_cpu_tensor(mined_triplet_indices)


    # 10. If mining pairs --------------------------------------------------
    if n_pairs_to_mine > 0:

        part_start_time = time.time()
        if gb.VERBOSE_MINING_TIMING:
            print("  > Starting pair mining...")
        # Mine pairs (including subsequence pairs if duplicates exist)
        mined_pairs, mined_pair_distances, mined_pair_ranks, mined_pair_buckets, mined_pair_region_pairs, mined_pair_local_indices = mine_pairs(
            train_sequences,
            n_pairs_to_mine,
            seq_embeddings_distances,
            train_regions,
            adjusted_distances_subsampled,
            pairwise_ranks_subsampled,
            taxon_counts_subsampled=taxon_counts_for_rep_balancing,
            train_embeddings=train_embeddings,
            duplicate_sequences=duplicate_sequences,
            duplicate_indices=duplicate_indices,
            duplicate_embeddings=duplicate_embeddings,
            duplicate_regions=duplicate_regions,
            verbose=verbose_pairs,
            log=log_pairs,
            batch_num=batch_num,
            logs_dir=logs_dir,
            warmup_phase=pair_warmup_phase,
            pair_distances_df=pair_distances_df,
            pair_error_metrics_df=pair_error_metrics_df,
            phylum_codes_for_logging=local_phylum_codes if do_phylum_logging else None,
            phylum_step_stats_out=phylum_step_stats if do_phylum_logging else None,
        )
        # mined_pairs.shape: (n_pairs_to_mine, 2, 4**K)  OR  (n_pairs_to_mine, 2, MAX_MODEL_SEQ_LEN, 3)
        # mined_pair_distances.shape: (n_pairs_to_mine)
        # mined_pair_ranks.shape: (n_pairs_to_mine)
        # mined_pair_buckets.shape: (n_pairs_to_mine)
        # mined_pair_local_indices.shape: (n_pairs_to_mine, 2) - local indices (0..n_seqs_to_use-1)
        #   Note: For subsequence pairs, second index is n_train_seqs + duplicate_idx
        
        # Convert local indices to global training set indices (0..n_train-1)
        # Handle regular pairs (both indices < n_train_seqs) and subsequence pairs separately
        mined_pair_indices = np.zeros_like(mined_pair_local_indices)
        for i in range(len(mined_pair_local_indices)):
            idx1, idx2 = mined_pair_local_indices[i]
            # Regular pair: both indices are < n_train_seqs
            if idx2 < n_train_seqs:
                mined_pair_indices[i, 0] = subsample_indices[idx1]
                mined_pair_indices[i, 1] = subsample_indices[idx2]
            else:
                # Subsequence pair: idx1 is original, idx2 is duplicate (encoded as n_train_seqs + duplicate_idx)
                # For subsequence pairs, both refer to the same original sequence
                mined_pair_indices[i, 0] = subsample_indices[idx1]
                mined_pair_indices[i, 1] = subsample_indices[idx1]  # Same sequence
        mined_pair_indices = torch.from_numpy(mined_pair_indices.astype(np.int64))
        # mined_pair_indices.shape: (n_pairs_to_mine, 2) dtype: int64
        
        if gb.VERBOSE_MINING_TIMING:
            print(f"  > Time taken to mine pairs: {time.time() - part_start_time:.4f} seconds")

    else:
        # No pairs to mine - create empty tensors with appropriate shapes
        # train_sequences.shape is (n_seqs_to_use, MAX_MODEL_SEQ_LEN, 3)
        mined_pairs = torch.empty((0, 2, train_sequences.shape[1], train_sequences.shape[2]), dtype=train_sequences.dtype)
        mined_pair_distances = torch.empty((0,), dtype=torch.float32)
        mined_pair_ranks = torch.empty((0,), dtype=torch.int32)
        mined_pair_buckets = torch.empty((0,), dtype=torch.long)
        mined_pair_region_pairs = torch.empty((0, 2), dtype=torch.int16)
        mined_pair_local_indices = np.empty((0, 2), dtype=np.int64)
        mined_pair_indices = torch.empty((0, 2), dtype=torch.long)

    # Pin pairs CPU tensors
    mined_pairs = _maybe_pin_cpu_tensor(mined_pairs)
    mined_pair_distances = _maybe_pin_cpu_tensor(mined_pair_distances)
    mined_pair_ranks = _maybe_pin_cpu_tensor(mined_pair_ranks)
    mined_pair_buckets = _maybe_pin_cpu_tensor(mined_pair_buckets)
    mined_pair_region_pairs = _maybe_pin_cpu_tensor(mined_pair_region_pairs)
    mined_pair_indices = _maybe_pin_cpu_tensor(mined_pair_indices)


    # 11. Mining logging --------------------------------------------------
    if do_phylum_logging:
        final_pair_combo_counts = _count_final_pair_phylum_combos(
            mined_pair_local_indices,
            mined_pair_ranks,
            local_phylum_codes,
            n_train_seqs,
        )
        final_triplet_an_combo_counts = _count_final_triplet_an_phylum_combos(
            mined_triplet_local_indices,
            mined_triplet_ranks,
            local_phylum_codes,
        )
        write_mining_phylum_log(
            batch_num=batch_num,
            logs_dir=logs_dir,
            phylum_labels=phylum_labels,
            step_stats=phylum_step_stats,
            final_pair_combo_counts=final_pair_combo_counts,
            final_triplet_an_combo_counts=final_triplet_an_combo_counts,
            max_lines_per_table=getattr(gb, "MINING_PHYLA_LOG_MAX_LINES_PER_TABLE", 8),
        )

    if log:
        write_mining_log(
            batch_num,
            logs_dir,
            pairwise_ranks_subsampled,
            pre_taxon_subsampling_ignored,
            gb.USE_TAXON_SIZE_MINING_BIAS,
            taxon_subsampling_counts_pre,
            taxon_subsampling_counts_post,
        )


    # 12. Shuffle mined outputs (optional) --------------------------------------------------
    if gb.SHUFFLE_TRIPLETS_AND_PAIRS_AFTER_MINING:
        mined_triplets, mined_triplet_margins, mined_triplet_ranks, mined_triplet_buckets, mined_triplet_indices = _shuffle_aligned_cpu_tensors(
            mined_triplets,
            mined_triplet_margins,
            mined_triplet_ranks,
            mined_triplet_buckets,
            mined_triplet_indices,
        )
        mined_pairs, mined_pair_distances, mined_pair_ranks, mined_pair_buckets, mined_pair_region_pairs, mined_pair_indices = _shuffle_aligned_cpu_tensors(
            mined_pairs,
            mined_pair_distances,
            mined_pair_ranks,
            mined_pair_buckets,
            mined_pair_region_pairs,
            mined_pair_indices,
        )

        # index_select creates new tensors, so re-pin when enabled.
        mined_triplets = _maybe_pin_cpu_tensor(mined_triplets)
        mined_triplet_margins = _maybe_pin_cpu_tensor(mined_triplet_margins)
        mined_triplet_ranks = _maybe_pin_cpu_tensor(mined_triplet_ranks)
        mined_triplet_buckets = _maybe_pin_cpu_tensor(mined_triplet_buckets)
        mined_triplet_indices = _maybe_pin_cpu_tensor(mined_triplet_indices)
        mined_pairs = _maybe_pin_cpu_tensor(mined_pairs)
        mined_pair_distances = _maybe_pin_cpu_tensor(mined_pair_distances)
        mined_pair_ranks = _maybe_pin_cpu_tensor(mined_pair_ranks)
        mined_pair_buckets = _maybe_pin_cpu_tensor(mined_pair_buckets)
        mined_pair_region_pairs = _maybe_pin_cpu_tensor(mined_pair_region_pairs)
        mined_pair_indices = _maybe_pin_cpu_tensor(mined_pair_indices)


    # 13. Free large intermediates --------------------------------------------------
    if gb.FREE_LARGE_INTERMEDIATES_WHEN_MINING:
        part_start_time = time.time()
        del train_sequences, adjusted_distances_subsampled, pairwise_ranks_subsampled, train_embeddings, seq_embeddings_distances
        del duplicate_sequences, duplicate_indices, duplicate_embeddings, duplicate_regions
        if gb.VERBOSE_MINING_TIMING:
            print(f"  > Time taken to free large intermediates: {time.time() - part_start_time:.4f} seconds")

    if gb.VERBOSE_MINING_TIMING:
        print(f"Time taken to mine {n_triplets_to_mine} triplets and {n_pairs_to_mine} pairs: {time.time() - func_start_time:.4f} seconds")

    return (mined_triplets, mined_pairs, mined_triplet_margins, mined_pair_distances, mined_triplet_ranks, mined_pair_ranks, mined_pair_region_pairs, mined_pair_buckets, mined_triplet_buckets, mined_triplet_indices, mined_pair_indices)


def mine(batch_num, model, logs_dir, triplet_satisfaction_df, pair_distances_df, triplet_error_metrics_df, pair_error_metrics_df):
    """
    A wrapper around mine_pairs_and_triplets that handles the mining of multiple batches at once.
    It uses a global cache (gb.MINED_CACHE) to store mined data and dispenses it batch by batch.
    """

    # Respect rank-introduction schedules at runtime. If an objective has no enabled ranks
    # for this batch, request zero samples for it until its intro batch is reached.
    has_triplet_ranks_this_batch = any(get_effective_triplet_ranks(batch_num=batch_num))
    has_pair_ranks_this_batch = any(get_effective_pair_ranks(batch_num=batch_num))
    requested_triplets_per_batch = gb.N_TRIPLETS_PER_BATCH if has_triplet_ranks_this_batch else 0
    requested_pairs_per_batch = gb.N_PAIRS_PER_BATCH if has_pair_ranks_this_batch else 0

    # Determine if we need to mine new data.
    # We need to mine if the cache is None or if it doesn't have enough data for the
    # objectives that are active for this batch.
    need_to_mine = gb.MINED_CACHE is None
    if not need_to_mine:
        # Check if we have enough triplets (if mining triplets)
        if requested_triplets_per_batch > 0 and len(gb.MINED_CACHE[0]) < requested_triplets_per_batch:
            need_to_mine = True
        # Check if we have enough pairs (if mining pairs)
        elif requested_pairs_per_batch > 0 and len(gb.MINED_CACHE[1]) < requested_pairs_per_batch:
            need_to_mine = True

    if need_to_mine:
        mining_start_time = time.time()
        
        # Determine logging verbosity for this mining run
        do_verbose_pairs = gb.VERBOSE_PAIR_MINING and (batch_num == 1 or batch_num % gb.VERBOSE_EVERY_N_BATCHES == 0)
        do_verbose_triplets = gb.VERBOSE_TRIPLET_MINING and (batch_num == 1 or batch_num % gb.VERBOSE_EVERY_N_BATCHES == 0)
        do_log_pairs = gb.LOG_PAIR_MINING and (batch_num == 1 or batch_num % gb.LOG_EVERY_N_BATCHES == 0)
        do_log_triplets = gb.LOG_TRIPLET_MINING and (batch_num == 1 or batch_num % gb.LOG_EVERY_N_BATCHES == 0)
        do_log = gb.LOG_MINING and (batch_num == 1 or batch_num % gb.LOG_EVERY_N_BATCHES == 0)

        # Mine multiple batches worth of data
        n_triplets_to_mine = requested_triplets_per_batch * gb.NUM_BATCHES_PER_MINING_RUN
        n_pairs_to_mine = requested_pairs_per_batch * gb.NUM_BATCHES_PER_MINING_RUN
        
        # Run mining
        gb.MINED_CACHE = mine_pairs_and_triplets(
            model, 
            n_triplets_to_mine, 
            n_pairs_to_mine, 
            verbose_pairs=do_verbose_pairs, 
            verbose_triplets=do_verbose_triplets, 
            log_pairs=do_log_pairs, 
            log_triplets=do_log_triplets, 
            log=do_log, 
            batch_num=batch_num, 
            logs_dir=logs_dir, 
            triplet_satisfaction_df=triplet_satisfaction_df, 
            pair_distances_df=pair_distances_df, 
            triplet_error_metrics_df=triplet_error_metrics_df, 
            pair_error_metrics_df=pair_error_metrics_df
        )
        
        if gb.VERBOSE_TRAINING_TIMING:
            print(f"> Time taken to mine training pairs/triplets ({gb.NUM_BATCHES_PER_MINING_RUN} batches): {time.time() - mining_start_time:.4f} seconds")

    # Take the data for this batch from the cache
    # mined_cache contains: (triplets, pairs, triplet_margins, pair_distances, triplet_ranks, pair_ranks, pair_region_pairs, pair_buckets, triplet_buckets, triplet_indices, pair_indices)
    
    # Extract triplets
    n_triplets = requested_triplets_per_batch
    triplets = gb.MINED_CACHE[0][:n_triplets]
    triplet_margins = gb.MINED_CACHE[2][:n_triplets]
    triplet_ranks = gb.MINED_CACHE[4][:n_triplets]
    triplet_buckets = gb.MINED_CACHE[8][:n_triplets] if gb.MINED_CACHE[8] is not None else None
    triplet_indices = gb.MINED_CACHE[9][:n_triplets]

    # Extract pairs
    n_pairs = requested_pairs_per_batch
    pairs = gb.MINED_CACHE[1][:n_pairs]
    pair_distances = gb.MINED_CACHE[3][:n_pairs]
    pair_ranks = gb.MINED_CACHE[5][:n_pairs]
    pair_region_pairs = gb.MINED_CACHE[6][:n_pairs]
    pair_buckets = gb.MINED_CACHE[7][:n_pairs] if gb.MINED_CACHE[7] is not None else None
    pair_indices = gb.MINED_CACHE[10][:n_pairs]

    # Update cache (remove taken items)
    gb.MINED_CACHE = (
        gb.MINED_CACHE[0][n_triplets:],
        gb.MINED_CACHE[1][n_pairs:],
        gb.MINED_CACHE[2][n_triplets:],
        gb.MINED_CACHE[3][n_pairs:],
        gb.MINED_CACHE[4][n_triplets:],
        gb.MINED_CACHE[5][n_pairs:],
        gb.MINED_CACHE[6][n_pairs:],
        gb.MINED_CACHE[7][n_pairs:] if gb.MINED_CACHE[7] is not None else None,
        gb.MINED_CACHE[8][n_triplets:] if gb.MINED_CACHE[8] is not None else None,
        gb.MINED_CACHE[9][n_triplets:],
        gb.MINED_CACHE[10][n_pairs:]
    )

    return triplets, pairs, triplet_margins, pair_distances, triplet_ranks, pair_ranks, pair_region_pairs, pair_buckets, triplet_buckets, triplet_indices, pair_indices
