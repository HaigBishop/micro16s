"""
This script contains functions for online mining of **pairs** from a Micro16S dataset.

The primary function is mine_pairs(), which is only ever called by triplet_pair_mining.py. 
Please see that file for more context and details.


Per-Rank Pair Mining:
    Pairs are mined using **separate per-rank** percentile-bucket sampling to remove bias 
    from ranks with large volumes of pairs. Each rank gets:
    - Its own representative set (up to PAIR_MINING_REPRESENTATIVE_SET_SIZES[rank] candidates)
    - Its own bucket boundaries computed from that representative set
    - A budget proportional to its EMA-smoothed hardness metric
    
    Hardness Metric (per rank):
        - Composite metric based on Mean and Quartiles of relative squared error:
          metric = mean_err * gb.PAIR_EMA_MEAN_WEIGHT + (p25_err + p75_err) * gb.PAIR_EMA_QUARTILES_WEIGHT
        - eps is gb.RELATIVE_ERROR_EPSILONS_PAIR_MINING[rank] to avoid division by zero (per-rank epsilon)
        - Updated via EMA after each batch using gb.PAIR_MINING_EMA_ALPHA
    
    Budget Allocation:
        - Batch size is divided across ranks according to EMA hardness
        - Each rank's proportion is capped by gb.PAIR_PER_RANK_BATCH_PROPORTION_MAX
        - Excess budget is redistributed to uncapped ranks preserving relative ratios
        - gb.PAIR_MINING_EMA_WEIGHT_EXPONENT controls how strongly hardness differences are amplified
    
    Pairs at rank r satisfy:
        - sequence_1 and sequence_2 share classification at rank (r-1), but differ at rank r
        - For example, rank 4 (family) pairs share the same order but belong to different families
    
    Upper Triangle Filtering (Duplicate Removal):
        - The pairwise distance and rank matrices are symmetric: (i,j) == (j,i).
        - Pairs (seq1, seq2) and (seq2, seq1) are identical - same distance, same rank.
        - To avoid wasting batch size on duplicates, we filter to upper triangle (row < col).
        - This halves the candidate pool while retaining all unique pair relationships.
    
    Per-Rank Percentile Bucket Sampling:
        - For each rank, pairs are placed into percentile buckets defined by gb.PAIR_MINING_BUCKETS
        - Bucket boundaries are computed separately for each rank from its representative set
        - Pairs are sampled from each bucket according to the sampling proportion
        - The 'any' bucket (gap=None) allows random sampling from the rank's entire pool
        - When a bucket has insufficient pairs, borrows from adjacent buckets
    
    Sign Tracking:
        - We track whether each pair has positive error (d_pred > d_true, too far) or
          negative error (d_pred < d_true, too close)
        - Optional sign-bias can upweight/downweight hardness by sign per rank via
          gb.PAIR_SIGN_BIAS_BETA_PER_RANK

"""

# Imports
import math
import time
import numpy as np
import torch
from numba import njit, prange

# Local Imports
import globals_config as gb
from logging_utils import print_pair_mining_stats, write_pair_mining_log, log_bucket_thresholds

# Re-use a global generator so we don't re-seed every call
_RNG = np.random.default_rng()
# Rank metadata: internal rank values -1 (domain) to 7 (subseq) map to PAIR_RANKS indices 0-8
_PAIR_RANK_LABELS = {
    -1: "domain",
    0: "phylum",
    1: "class",
    2: "order",
    3: "family",
    4: "genus",
    5: "species",
    6: "sequence",
    7: "subseq",
}
# Number of pair ranks (domain through subsequence)
_N_PAIR_RANKS = 9
_PAIR_SIGN_BIAS_CONFIG_WARNED = False


def is_rank_enabled_for_batch(rank_idx, batch_num=None):
    """
    Return True if a pair rank is introduced for the current batch.
    """
    if batch_num is None:
        return True

    intro_batches = gb.INTRODUCE_RANK_AT_BATCHES
    if intro_batches is None:
        return True
    if rank_idx < 0 or rank_idx >= len(intro_batches):
        return True

    try:
        introduced_at = int(intro_batches[rank_idx])
    except Exception:
        return True

    return int(batch_num) >= introduced_at


def get_effective_pair_ranks(batch_num=None):
    """
    Combine PAIR_RANKS with INTRODUCE_RANK_AT_BATCHES for the current batch.
    """
    effective = [False] * _N_PAIR_RANKS
    if gb.PAIR_RANKS is None:
        return effective

    for rank_idx in range(min(len(gb.PAIR_RANKS), _N_PAIR_RANKS)):
        if gb.PAIR_RANKS[rank_idx] and is_rank_enabled_for_batch(rank_idx, batch_num=batch_num):
            effective[rank_idx] = True
    return effective


def _get_pair_sign_bias_betas():
    """
    Get per-rank sign-bias betas for pair mining hardness.

    Returns:
        betas: np.ndarray shape (9,), dtype float32
        active: bool, True if any beta is non-zero
    """
    global _PAIR_SIGN_BIAS_CONFIG_WARNED
    config = getattr(gb, "PAIR_SIGN_BIAS_BETA_PER_RANK", None)
    if config is None:
        return np.zeros(_N_PAIR_RANKS, dtype=np.float32), False

    try:
        betas = np.asarray(config, dtype=np.float32)
    except Exception:
        betas = np.array([], dtype=np.float32)

    if betas.shape != (_N_PAIR_RANKS,) or not np.all(np.isfinite(betas)):
        if not _PAIR_SIGN_BIAS_CONFIG_WARNED:
            print("WARNING: PAIR_SIGN_BIAS_BETA_PER_RANK must be a finite length-9 list. Ignoring sign bias.")
            _PAIR_SIGN_BIAS_CONFIG_WARNED = True
        return np.zeros(_N_PAIR_RANKS, dtype=np.float32), False

    return betas, bool(np.any(np.abs(betas) > 1e-12))


def _apply_pair_sign_bias_to_hardness(relative_errors, signed_errors, flat_ranks, betas):
    """
    Apply per-rank sign bias to relative squared error hardness.

    hardness = rel_sq_error * exp(beta_r * sign)
    sign = +1 (too far), -1 (too close), 0 (exact match)
    """
    if relative_errors.size == 0:
        return relative_errors

    signs = np.sign(signed_errors).astype(np.float32, copy=False)
    rank_indices = (flat_ranks + 1).astype(np.int64, copy=False)
    np.clip(rank_indices, 0, _N_PAIR_RANKS - 1, out=rank_indices)
    beta_per_pair = betas[rank_indices]

    exp_args = beta_per_pair * signs
    np.clip(exp_args, -20.0, 20.0, out=exp_args)

    multipliers = np.empty_like(exp_args, dtype=np.float32)
    np.exp(exp_args, out=multipliers)

    return relative_errors * multipliers


def _build_pair_representative_phylum_step_stats(phylum_codes, n_train_seqs, regular_flat_indices, subseq_orig_indices, pairwise_ranks, target_pairwise_rank=0):
    """
    Build phylum step stats for pair representative-set subsampling.
    """
    if phylum_codes is None:
        return None

    phylum_codes = np.asarray(phylum_codes, dtype=np.int32)
    if phylum_codes.ndim != 1 or len(phylum_codes) != n_train_seqs:
        return None
    if len(phylum_codes) == 0:
        return None
    if np.min(phylum_codes) < 0:
        return None

    n_phyla = int(np.max(phylum_codes)) + 1
    if n_phyla <= 0:
        return None

    seq_counts = np.bincount(phylum_codes, minlength=n_phyla).astype(np.int64, copy=False)
    endpoint_counts = np.zeros(n_phyla, dtype=np.int64)

    regular_flat_indices = np.asarray(regular_flat_indices, dtype=np.int64)
    if pairwise_ranks is not None and regular_flat_indices.size > 0:
        flat_ranks = np.asarray(pairwise_ranks, dtype=np.int8).ravel()[regular_flat_indices]
        regular_flat_indices = regular_flat_indices[flat_ranks == int(target_pairwise_rank)]
    if regular_flat_indices.size > 0:
        seq_i = regular_flat_indices // n_train_seqs
        seq_j = regular_flat_indices % n_train_seqs
        if np.max(seq_i) >= len(phylum_codes) or np.max(seq_j) >= len(phylum_codes):
            return None
        endpoint_phyla = np.concatenate([phylum_codes[seq_i], phylum_codes[seq_j]])
        endpoint_counts += np.bincount(endpoint_phyla, minlength=n_phyla).astype(np.int64, copy=False)

    # Subsequence pairs are always sequence/subsequence rank, not phylum rank.
    n_active_pairs = int(len(regular_flat_indices))

    return {
        "step": "post_pair_representative_subsample",
        "seq_counts": seq_counts,
        "pair_endpoint_counts": endpoint_counts,
        "n_active_pairs": n_active_pairs,
    }


def _compute_per_rank_budgets(ema_hardness, total_budget, cap_max, enabled_ranks, weight_exponent=1.0, cap_min=0.0):
    """
    Compute integer per-rank batch allocations from EMA hardness values.
    
    Uses the hardness-proportional allocation algorithm with iterative min/max capping:
    1. Normalise hardness values to sum to 1.0
    2. Iteratively apply MIN floors (raise ranks below minimum)
    3. Iteratively apply MAX caps (cap ranks above maximum)
    4. Redistribute budget to unfrozen ranks preserving their relative ratios
    5. Convert continuous proportions to integers, fixing rounding drift
    
    Args:
        ema_hardness: np.ndarray of shape (9,) - EMA hardness values per rank
        total_budget: int - Total number of pairs to allocate
        cap_max: float - Maximum proportion any single rank can receive (0, 1]
        enabled_ranks: list/array of bool (length 9) - Which ranks are enabled
        weight_exponent: float - Exponent applied to hardness before normalising (>=0)
        cap_min: float - Minimum proportion any single enabled rank must receive [0, 1]
    
    Returns:
        budgets: np.ndarray of shape (9,) dtype int64 - Integer allocations per rank
        proportions: np.ndarray of shape (9,) dtype float64 - Continuous proportions per rank
    """
    n_ranks = len(ema_hardness)
    
    # Start with hardness values, zeroing disabled ranks
    weights = np.array(ema_hardness, dtype=np.float64)
    for i in range(n_ranks):
        if not enabled_ranks[i]:
            weights[i] = 0.0

    exponent = 1.0 if weight_exponent is None else float(weight_exponent)
    if exponent != 1.0:
        positive_mask = weights > 0
        if exponent == 0.0:
            weights[positive_mask] = 1.0
        else:
            weights[positive_mask] = np.power(weights[positive_mask], exponent)
        weights[~positive_mask] = 0.0
    
    # Handle edge case: all weights are zero
    total_weight = weights.sum()
    if total_weight <= 0:
        return np.zeros(n_ranks, dtype=np.int64), np.zeros(n_ranks, dtype=np.float64)
    
    # Count enabled ranks for feasibility check
    n_enabled = sum(1 for i in range(n_ranks) if enabled_ranks[i] and weights[i] > 0)
    
    # Feasibility check: can we satisfy all minimums?
    # If cap_min * n_enabled > 1.0, we cannot give every rank its minimum
    if cap_min > 0 and n_enabled > 0 and cap_min * n_enabled > 1.0:
        effective_min = 1.0 / n_enabled
        print(f"WARNING: PAIR_PER_RANK_BATCH_PROPORTION_MIN ({cap_min}) * n_enabled_ranks ({n_enabled}) > 1.0. "
              f"Using equal distribution ({effective_min:.4f}) instead.")
        # Use equal distribution since we can't satisfy all minimums
        proportions = np.zeros(n_ranks, dtype=np.float64)
        for i in range(n_ranks):
            if enabled_ranks[i] and weights[i] > 0:
                proportions[i] = effective_min
    else:
        # Normalise to sum to 1.0
        proportions = weights / total_weight
        
        # Iterative min/max capping: freeze ranks at floor or cap, redistribute to remaining
        frozen_at_min = np.zeros(n_ranks, dtype=bool)
        frozen_at_max = np.zeros(n_ranks, dtype=bool)
        
        max_iterations = n_ranks * 2 + 10  # Safety limit to prevent infinite loops
        for _ in range(max_iterations):
            changes_made = False
            frozen = frozen_at_min | frozen_at_max
            
            # Check for ranks below MIN (that aren't already frozen and are enabled with positive weight)
            if cap_min > 0:
                below_min = (proportions < cap_min - 1e-9) & ~frozen & np.array([enabled_ranks[i] and weights[i] > 0 for i in range(n_ranks)])
                if np.any(below_min):
                    for i in np.where(below_min)[0]:
                        frozen_at_min[i] = True
                        proportions[i] = cap_min
                    changes_made = True
            
            # Check for ranks above MAX (that aren't already frozen)
            above_max = (proportions > cap_max + 1e-9) & ~frozen_at_min & ~frozen_at_max
            if np.any(above_max):
                for i in np.where(above_max)[0]:
                    frozen_at_max[i] = True
                    proportions[i] = cap_max
                changes_made = True
            
            if not changes_made:
                break
            
            # Renormalise unfrozen ranks to use remaining budget
            frozen = frozen_at_min | frozen_at_max
            frozen_sum = proportions[frozen].sum()
            remaining_budget = 1.0 - frozen_sum
            
            unfrozen_mask = ~frozen & (weights > 0)
            unfrozen_sum = weights[unfrozen_mask].sum()
            if unfrozen_sum > 0 and remaining_budget > 1e-9:
                for i in np.where(unfrozen_mask)[0]:
                    proportions[i] = (weights[i] / unfrozen_sum) * remaining_budget
            elif remaining_budget <= 1e-9:
                # All budget allocated to frozen ranks
                for i in np.where(unfrozen_mask)[0]:
                    proportions[i] = 0.0
                break

    # If proportions don't sum to 1.0 (can happen if all ranks frozen or floating point drift), fix it
    current_sum = proportions.sum()
    leftover = max(0.0, 1.0 - current_sum)
    if leftover > 1e-9:
        eligible = np.where((weights > 0) & np.array(enabled_ranks))[0]
        if eligible.size > 0:
            # Give the remainder to the hardest enabled rank to avoid losing budget
            idx = eligible[np.argmax(weights[eligible])]
            proportions[idx] += leftover
            rank_label = _PAIR_RANK_LABELS.get(idx - 1, str(idx - 1))
            print(f"WARNING: Relaxing per-rank cap for '{rank_label}' to absorb surplus batch budget.")
    
    # Convert to integer allocations
    float_allocations = proportions * total_budget
    int_allocations = np.floor(float_allocations).astype(np.int64)
    
    # Fix rounding drift by distributing remainder to ranks with largest fractional parts
    remainder = total_budget - int_allocations.sum()
    if remainder > 0:
        fractional_parts = float_allocations - int_allocations
        # Only consider enabled ranks for adjustment
        for i in range(n_ranks):
            if not enabled_ranks[i]:
                fractional_parts[i] = -1.0  # Exclude from consideration
        
        # Give +1 to ranks with largest fractional parts
        indices_by_frac = np.argsort(-fractional_parts)
        for i in range(int(remainder)):
            idx = indices_by_frac[i]
            if enabled_ranks[idx]:
                int_allocations[idx] += 1
    
    return int_allocations, proportions


def _update_pair_ema_hardness(per_rank_metrics, alpha):
    """
    Update the global pair mining EMA hardness buffer.
    
    EMA update formula: ema = alpha * new_value + (1 - alpha) * ema
    
    Args:
        per_rank_metrics: np.ndarray of shape (9,) - Hardness metric per rank
            (Weighted sum of mean and quartile relative squared errors). NaN values indicate no samples for that rank (EMA unchanged).
        alpha: float - EMA smoothing factor in (0, 1]
    """
    if gb.PAIR_MINING_EMA_HARDNESS is None:
        return
    
    for i in range(_N_PAIR_RANKS):
        if not np.isnan(per_rank_metrics[i]) and per_rank_metrics[i] >= 0:
            gb.PAIR_MINING_EMA_HARDNESS[i] = (
                alpha * per_rank_metrics[i] + 
                (1 - alpha) * gb.PAIR_MINING_EMA_HARDNESS[i]
            )


@njit(parallel=True, cache=True, fastmath=True)
def _build_valid_pair_indices_by_rank(pairwise_ranks, rank_lookup_array):
    """
    Build valid upper-triangle pair indices grouped by rank using parallel processing.
    
    This replaces the sequential reservoir sampling approach with a parallel two-pass
    algorithm (count then fill), similar to _build_neighbor_data_numba in triplet_mining.py.
    
    Args:
        pairwise_ranks: np.ndarray of shape (n_seqs, n_seqs), dtype int8
            2D pairwise ranks matrix (NOT flattened - better cache locality).
        rank_lookup_array: np.ndarray of shape (9,), dtype bool
            Which ranks are enabled. Indexed as rank + 1 (so -1..7 -> 0..8).
    
    Returns:
        counts_per_rank: np.ndarray (9,) dtype int64 - total valid pairs per rank
        rank_starts: np.ndarray (9,) dtype int64 - start offset in pairs_flat for each rank
        pairs_flat: np.ndarray (total_pairs,) dtype int64 - flat indices grouped by rank
    """
    n_seqs = pairwise_ranks.shape[0]
    n_ranks = 9
    
    # Pass 1: Count pairs per (row, rank) in parallel
    # Direct 2D access with constant row, sequential col has excellent cache locality
    row_counts = np.zeros((n_seqs, n_ranks), dtype=np.int32)
    
    for row in prange(n_seqs):
        for col in range(row + 1, n_seqs):  # Upper triangle only
            rank = pairwise_ranks[row, col]  # Direct 2D access - cache friendly
            if rank < -1 or rank > 7:
                continue
            rank_idx = rank + 1
            if rank_lookup_array[rank_idx]:
                row_counts[row, rank_idx] += 1
    
    # Compute per-rank totals and starts (sequential but O(n_seqs * n_ranks) - very fast)
    counts_per_rank = np.zeros(n_ranks, dtype=np.int64)
    for row in range(n_seqs):
        for rank_idx in range(n_ranks):
            counts_per_rank[rank_idx] += row_counts[row, rank_idx]
    
    rank_starts = np.zeros(n_ranks, dtype=np.int64)
    total = np.int64(0)
    for rank_idx in range(n_ranks):
        rank_starts[rank_idx] = total
        total += counts_per_rank[rank_idx]
    
    # Compute per-(row, rank) write offsets using prefix sums within each rank's section
    row_offsets = np.zeros((n_seqs, n_ranks), dtype=np.int64)
    for rank_idx in range(n_ranks):
        offset = rank_starts[rank_idx]
        for row in range(n_seqs):
            row_offsets[row, rank_idx] = offset
            offset += row_counts[row, rank_idx]
    
    # Pass 2: Fill pair indices in parallel (each row writes to non-overlapping segments)
    # Flat index computed only here, not in counting pass
    pairs_flat = np.empty(total, dtype=np.int64)
    
    for row in prange(n_seqs):
        # Local write positions for this row
        write_pos = np.empty(n_ranks, dtype=np.int64)
        for rank_idx in range(n_ranks):
            write_pos[rank_idx] = row_offsets[row, rank_idx]
        
        # Precompute row offset for flat index calculation
        row_offset = row * n_seqs
        
        for col in range(row + 1, n_seqs):
            rank = pairwise_ranks[row, col]  # Direct 2D access
            if rank < -1 or rank > 7:
                continue
            rank_idx = rank + 1
            if rank_lookup_array[rank_idx]:
                pairs_flat[write_pos[rank_idx]] = row_offset + col  # Flat index only computed when storing
                write_pos[rank_idx] += 1
    
    return counts_per_rank, rank_starts, pairs_flat


@njit(parallel=True, cache=True, fastmath=True)
def _compute_gumbel_keys_for_weighted_sampling(
    flat_indices, n_seqs, taxon_counts, rank_idx,
    baseline_count, lam, eps, log_weight_cap, uniform_samples, keys_out
):
    """
    Fused computation of Gumbel-top-k keys for weighted sampling without replacement.
    
    High-level purpose:
        Sample pairs from the representative set with weights that counteract combinatorial
        taxon-size effects. Pairs involving sequences from rarer taxa (smaller taxon counts)
        get higher sampling weights, preventing the training set from being dominated by
        pairs from common taxa. This ensures balanced representation across the taxonomic
        diversity of the dataset.
    
    Implementation:
        Computes key_i = log(w_i) + Gumbel(0,1) in a single parallel pass per candidate,
        eliminating all temporary array allocations from the naive multi-pass NumPy approach.
        The top-k keys correspond to weighted sampling without replacement.
    
    Weight formula:  w(i,j,r) = ((b_r^2) / (c_i * c_j + eps)) ^ lambda
    Log-weight:      log(w)   = lambda * (log(b_r^2) - log(c_i * c_j + eps))
    Gumbel noise:    G        = -log(-log(U)),  U ~ Uniform(0,1)
    
    Args:
        flat_indices: (n_candidates,) int64 - flat indices into (n_seqs, n_seqs)
        n_seqs: int64 - matrix side length
        taxon_counts: (n_seqs, 7) int32 or float32 - per-sequence taxon sizes
            (float32 when corrected for domain-based downsampling)
        rank_idx: int64 - column index into taxon_counts (0=domain, ..., 6=species)
        baseline_count: float64 - b_r for this rank
        lam: float64 - balancing strength (0=uniform, 1=full counteraction)
        eps: float64 - numerical safety
        log_weight_cap: float64 - log(weight_clip), or inf if no clipping
        uniform_samples: (n_candidates,) float64 - pre-drawn U ~ Uniform(0,1)
        keys_out: (n_candidates,) float64 - output buffer for Gumbel keys
    """
    log_b_sq = math.log(max(baseline_count * baseline_count, 1e-300))
    n = len(flat_indices)
    
    for i in prange(n):
        flat_idx = flat_indices[i]
        row = flat_idx // n_seqs
        col = flat_idx - row * n_seqs  # Faster than modulo
        
        c_i = float(taxon_counts[row, rank_idx])
        c_j = float(taxon_counts[col, rank_idx])
        
        log_w = lam * (log_b_sq - math.log(c_i * c_j + eps))
        
        # Apply weight clip in log-space
        if log_w > log_weight_cap:
            log_w = log_weight_cap
        
        # Gumbel noise: -log(-log(u))
        u = uniform_samples[i]
        keys_out[i] = log_w - math.log(-math.log(u))


def _compute_representative_weights(flat_indices, n_seqs, taxon_counts, baseline_counts,
                                     rank_idx, lam, eps, weight_clip):
    """
    Compute per-candidate taxon-size-balancing weights for representative-set subsampling.
    
    Weight formula:  w(i,j,r) = ((b_r^2) / (c_i * c_j + eps)) ^ lambda
    where:
        c_i, c_j = taxon counts for sequences i and j at rank r
        b_r      = baseline taxon count at rank r
        lambda   = balancing strength (0=uniform, 1=full counteraction)
    
    Args:
        flat_indices: np.ndarray of shape (n_candidates,) dtype int64 - flat indices into (n_seqs, n_seqs)
        n_seqs: int - number of sequences (matrix side length)
        taxon_counts: np.ndarray of shape (n_seqs, 7) dtype int32 or float32
            Per-sequence taxon sizes (float32 when corrected for domain-based downsampling).
        baseline_counts: np.ndarray of shape (7,) - per-rank baseline taxon sizes
        rank_idx: int - PAIR_RANKS index (0=domain, ..., 6=species). Must be in [0, 6].
        lam: float - lambda in [0, 1]
        eps: float - epsilon for numerical safety
        weight_clip: float or None - optional cap for extreme weights
    
    Returns:
        weights: np.ndarray of shape (n_candidates,) dtype float64 - normalised sampling probabilities
    """
    n = len(flat_indices)
    
    # Decode row and column indices from flat indices
    rows = flat_indices // n_seqs
    cols = flat_indices % n_seqs
    
    # Get taxon counts for both sequences at this rank
    c_i = taxon_counts[rows, rank_idx].astype(np.float64)
    c_j = taxon_counts[cols, rank_idx].astype(np.float64)
    
    # Baseline count for this rank
    b_r = float(baseline_counts[rank_idx])
    
    # Compute raw weights: ((b_r^2) / (c_i * c_j + eps)) ^ lambda
    numerator = b_r * b_r
    denominator = c_i * c_j + eps
    ratio = numerator / denominator
    
    # Clip ratio to [0, large_value] before exponentiation to avoid overflow
    np.clip(ratio, 0.0, 1e30, out=ratio)
    
    if lam == 1.0:
        weights = ratio
    else:
        weights = np.power(ratio, lam)
    
    # Clip extreme weights if configured
    if weight_clip is not None and weight_clip > 0:
        np.clip(weights, 0.0, weight_clip, out=weights)
    
    # Normalise to probability distribution
    total = weights.sum()
    if total <= 0 or not np.isfinite(total):
        # Degenerate case: fall back to uniform
        return np.ones(n, dtype=np.float64) / n
    
    weights /= total
    return weights


def _get_valid_pair_indices_per_rank(pairwise_ranks, rank_lookup_array, max_samples_per_rank,
                                     seed, taxon_counts=None, shortage_recorder=None):
    """
    Get valid pair indices with per-rank subsampling up to representative set sizes.
    
    Uses parallel index building followed by per-rank random subsampling.
    When representative taxon-size balancing is enabled (gb.USE_REPRESENTATIVE_TAXON_SIZE_BALANCING
    and lambda > 0), subsampling uses weighted sampling without replacement to counteract
    combinatorial taxon-size effects. Otherwise falls back to uniform sampling.
    
    Args:
        pairwise_ranks: np.ndarray of shape (n_seqs, n_seqs), dtype int8
            2D pairwise ranks matrix (NOT flattened - better cache locality).
        rank_lookup_array: np.ndarray of shape (9,), dtype bool - which ranks are enabled
        max_samples_per_rank: np.ndarray of shape (9,), dtype int64 - max samples per rank
        seed: int for reproducibility
        taxon_counts: np.ndarray of shape (n_seqs, 7), dtype int32 or float32, or None
            Per-sequence taxon sizes. May be float32 when corrected for domain-based
            downsampling. Required when representative taxon-size balancing is enabled.
        shortage_recorder: list or None - accumulates shortage warnings
    
    Returns:
        valid_indices: np.ndarray of flat indices (combined across all ranks, subsampled, upper triangle only)
        valid_ranks: np.ndarray of internal rank values (-1 to 7) for each index
        total_valid_per_rank: np.ndarray of shape (9,) - total count before subsampling
    """
    sub_sub_part_start_time = time.time()
    n_seqs = pairwise_ranks.shape[0]
    
    # Build all valid pair indices in parallel (grouped by rank)
    counts_per_rank, rank_starts, pairs_flat = _build_valid_pair_indices_by_rank(
        pairwise_ranks, rank_lookup_array
    )

    if gb.VERBOSE_MINING_TIMING:
        print(f"           > Time taken for sub part #1.1: {time.time() - sub_sub_part_start_time:.4f} seconds")
    sub_sub_part_start_time = time.time()
    
    total_valid_per_rank = counts_per_rank.copy()
    
    if counts_per_rank.sum() == 0:
        return (np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.int8),
                total_valid_per_rank)
    
    # Determine if weighted sampling should be used
    use_weighted = (
        gb.USE_REPRESENTATIVE_TAXON_SIZE_BALANCING
        and gb.PAIR_REPRESENTATIVE_TAXON_SIZE_BALANCE_LAMBDA > 0
        and taxon_counts is not None
    )
    lam = float(gb.PAIR_REPRESENTATIVE_TAXON_SIZE_BALANCE_LAMBDA) if use_weighted else 0.0
    eps = float(gb.REPRESENTATIVE_TAXON_SIZE_BALANCE_EPS) if use_weighted else 1e-12
    weight_clip = gb.REPRESENTATIVE_TAXON_SIZE_BALANCE_WEIGHT_CLIP if use_weighted else None
    baseline_counts = gb.TRAIN_TAXON_BASELINE_COUNT_PER_RANK if use_weighted else None
    
    if gb.VERBOSE_MINING_TIMING:
        print(f"           > Time taken for sub part #1.2: {time.time() - sub_sub_part_start_time:.4f} seconds")
    sub_sub_part_start_time = time.time()
    
    # Subsample each rank to its representative set size
    rng = np.random.default_rng(seed)
    subsampled_indices_list = []
    subsampled_ranks_list = []
    
    for rank_idx in range(9):
        if not rank_lookup_array[rank_idx]:
            continue
        
        internal_rank = rank_idx - 1  # Convert 0..8 back to -1..7
        total_for_rank = int(counts_per_rank[rank_idx])
        cap_for_rank = int(max_samples_per_rank[rank_idx])
        
        if cap_for_rank > 0 and total_for_rank < cap_for_rank:
            rank_label = _PAIR_RANK_LABELS.get(internal_rank, str(internal_rank))
            print(f"WARNING: Pair mining rank '{rank_label}' has {total_for_rank} candidates, "
                  f"fewer than representative set size {cap_for_rank}.")
            if shortage_recorder is not None:
                shortage_recorder.append({
                    'rank_name': rank_label,
                    'requested': cap_for_rank,
                    'available': total_for_rank,
                })
        
        if total_for_rank == 0:
            continue
        
        # Get slice of pairs_flat for this rank
        start = int(rank_starts[rank_idx])
        end = start + total_for_rank
        rank_pairs = pairs_flat[start:end]
        
        # Subsample if needed
        if cap_for_rank > 0 and total_for_rank > cap_for_rank:
            # Use weighted sampling when enabled and taxon counts cover this rank (0-6)
            if use_weighted and rank_idx <= 6 and baseline_counts is not None:
                # Representative taxon-size balancing: weight each pair inversely by the
                # product of its sequences' taxon counts. This counteracts the combinatorial
                # explosion of pairs from common taxa (e.g., if taxon A has 100 sequences
                # and taxon B has 10, A×A yields 4950 pairs vs B×B yielding 45 pairs).
                # By upweighting rarer pairs, we ensure balanced representation across the
                # full taxonomic diversity rather than oversampling from abundant taxa.
                #
                # Fused Gumbel-top-k: compute log-weight + Gumbel noise in a single
                # parallel Numba pass, then select top-k via argpartition.
                # Statistically equivalent to rng.choice(..., p=weights) but avoids
                # all temporary array allocations from multi-pass NumPy weight computation.
                b_r = float(baseline_counts[rank_idx])
                log_wc = math.log(weight_clip) if (weight_clip is not None and weight_clip > 0) else math.inf
                u = rng.random(total_for_rank)
                np.clip(u, 1e-300, 1.0, out=u)  # Guard against log(0) in Gumbel noise
                keys = np.empty(total_for_rank, dtype=np.float64)
                _compute_gumbel_keys_for_weighted_sampling(
                    rank_pairs, np.int64(n_seqs), taxon_counts, np.int64(rank_idx),
                    b_r, lam, eps, log_wc, u, keys
                )
                chosen_local = np.argpartition(keys, -cap_for_rank)[-cap_for_rank:]
            else:
                chosen_local = rng.choice(total_for_rank, size=cap_for_rank, replace=False)
            chosen_local.sort()  # Preserve cache locality for downstream gathers
            rank_pairs = rank_pairs[chosen_local]
        
        subsampled_indices_list.append(rank_pairs.copy())
        subsampled_ranks_list.append(np.full(len(rank_pairs), internal_rank, dtype=np.int8))
    
    if gb.VERBOSE_MINING_TIMING:
        print(f"           > Time taken for sub part #1.3: {time.time() - sub_sub_part_start_time:.4f} seconds")
    sub_sub_part_start_time = time.time()
    
    if not subsampled_indices_list:
        return (np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.int8),
                total_valid_per_rank)
    
    valid_indices = np.concatenate(subsampled_indices_list)
    valid_ranks = np.concatenate(subsampled_ranks_list)
    
    if gb.VERBOSE_MINING_TIMING:
        print(f"           > Time taken for sub part #1.4: {time.time() - sub_sub_part_start_time:.4f} seconds")
    
    return valid_indices, valid_ranks, total_valid_per_rank


def _compute_per_rank_bucket_thresholds(relative_errors, pool_ranks, percentile_gaps):
    """
    Compute bucket error thresholds separately for each rank.
    
    Args:
        relative_errors: np.ndarray of shape (n_pool,) - relative squared errors for pool
        pool_ranks: np.ndarray of shape (n_pool,) - internal ranks (-1 to 7) for pool
        percentile_gaps: list of floats - bucket percentile gaps (e.g. [0.25, 0.25, 0.25, 0.25])
    
    Returns:
        per_rank_thresholds: dict mapping internal_rank -> list of threshold values
            Each list has len(percentile_gaps) - 1 thresholds
    """
    # Compute cumulative percentiles from gaps
    cumulative_percentiles = np.cumsum(percentile_gaps)[:-1]  # Exclude the last (100%)
    
    per_rank_thresholds = {}
    
    for rank_idx in range(_N_PAIR_RANKS):
        internal_rank = rank_idx - 1  # 0..8 -> -1..7
        rank_mask = (pool_ranks == internal_rank)
        
        if not np.any(rank_mask):
            per_rank_thresholds[internal_rank] = []
            continue
        
        rank_errors = relative_errors[rank_mask]
        
        if len(rank_errors) == 0:
            per_rank_thresholds[internal_rank] = []
            continue
        
        # Compute threshold values at each percentile
        thresholds = np.quantile(rank_errors, cumulative_percentiles).tolist()
        per_rank_thresholds[internal_rank] = thresholds
    
    return per_rank_thresholds


@njit(parallel=True, cache=True)
def _assign_buckets_per_rank_kernel(relative_errors, pool_ranks, thresholds_array, n_thresholds_per_rank):
    """
    Numba kernel to assign each pair to its bucket based on per-rank thresholds.
    
    Parallelizes over the pool to assign bucket indices efficiently.
    
    Args:
        relative_errors: np.ndarray of shape (n_pool,) dtype float64
        pool_ranks: np.ndarray of shape (n_pool,) dtype int16 - internal ranks (-1 to 7)
        thresholds_array: np.ndarray of shape (9, max_thresholds) dtype float64 - thresholds per rank
        n_thresholds_per_rank: np.ndarray of shape (9,) dtype int32 - number of valid thresholds per rank
    
    Returns:
        bucket_assignments: np.ndarray of shape (n_pool,) dtype int32
    """
    n_pool = len(relative_errors)
    bucket_assignments = np.zeros(n_pool, dtype=np.int32)
    
    for i in prange(n_pool):
        rank = pool_ranks[i]
        rank_idx = rank + 1  # Map -1..7 to 0..8
        
        if rank_idx < 0 or rank_idx >= 9:
            continue
        
        error = relative_errors[i]
        n_thresholds = n_thresholds_per_rank[rank_idx]
        
        # Find bucket: bucket 0 if below first threshold, bucket k if >= threshold k-1
        bucket = 0
        for t_idx in range(n_thresholds):
            if error >= thresholds_array[rank_idx, t_idx]:
                bucket = t_idx + 1
        
        bucket_assignments[i] = bucket
    
    return bucket_assignments


def _assign_buckets_per_rank(relative_errors, pool_ranks, per_rank_thresholds):
    """
    Assign each pair to its bucket based on per-rank thresholds.
    
    Uses a Numba-accelerated parallel kernel for performance.
    
    Args:
        relative_errors: np.ndarray of shape (n_pool,)
        pool_ranks: np.ndarray of shape (n_pool,) - internal ranks (-1 to 7)
        per_rank_thresholds: dict from _compute_per_rank_bucket_thresholds
    
    Returns:
        bucket_assignments: np.ndarray of shape (n_pool,) dtype int32
    """
    n_pool = len(relative_errors)
    
    if n_pool == 0:
        return np.zeros(0, dtype=np.int32)
    
    # Find max number of thresholds across all ranks
    max_thresholds = 0
    for thresholds in per_rank_thresholds.values():
        if len(thresholds) > max_thresholds:
            max_thresholds = len(thresholds)
    
    if max_thresholds == 0:
        # No thresholds means everything goes to bucket 0
        return np.zeros(n_pool, dtype=np.int32)
    
    # Build arrays from dict for Numba kernel
    thresholds_array = np.zeros((9, max_thresholds), dtype=np.float64)
    n_thresholds_per_rank = np.zeros(9, dtype=np.int32)
    
    for internal_rank, thresholds in per_rank_thresholds.items():
        rank_idx = internal_rank + 1  # Map -1..7 to 0..8
        if 0 <= rank_idx < 9:
            n_thresholds_per_rank[rank_idx] = len(thresholds)
            for t_idx, t_val in enumerate(thresholds):
                thresholds_array[rank_idx, t_idx] = t_val
    
    # Ensure correct dtypes for Numba kernel
    if relative_errors.dtype != np.float64:
        relative_errors = relative_errors.astype(np.float64)
    if pool_ranks.dtype != np.int16:
        pool_ranks = pool_ranks.astype(np.int16)
    
    return _assign_buckets_per_rank_kernel(
        relative_errors,
        pool_ranks,
        thresholds_array,
        n_thresholds_per_rank
    )


def _sample_from_rank_buckets(n_to_sample, rank_mask, bucket_assignments, relative_errors,
                              n_buckets, target_proportions, available_mask, internal_rank):
    """
    Sample pairs from a single rank's buckets with overflow handling.
    
    When a bucket has insufficient pairs, borrows from adjacent buckets:
    - Right-most bucket first, borrow from left
    - Left-most bucket can borrow from 'any' (all remaining)
    
    Args:
        n_to_sample: int - target number of pairs for this rank
        rank_mask: np.ndarray bool - mask for pairs of this rank
        bucket_assignments: np.ndarray int32 - bucket assignment for each pair
        relative_errors: np.ndarray - for sorting when borrowing
        n_buckets: int - number of regular buckets (excluding 'any')
        target_proportions: np.ndarray - target proportion for each bucket + 'any'
        available_mask: np.ndarray bool - which pairs are still available (modified in-place)
        internal_rank: int - internal rank id (-1..7) for warning messages
    
    Returns:
        sampled_indices: list of int - indices into the pool
        deficit: int - how many pairs short of target
        bucket_stats: dict describing per-bucket fulfillment (targets/sample/borrow/deficit)
    """
    if n_to_sample <= 0:
        return [], 0, None
    
    # Compute deterministic target counts via floor + remainder distribution
    float_targets = target_proportions * n_to_sample
    bucket_targets = np.floor(float_targets).astype(np.int32)
    remainder = int(n_to_sample - bucket_targets.sum())
    if remainder > 0:
        fractional_parts = float_targets - bucket_targets
        order = np.argsort(-fractional_parts)
        for idx in order[:remainder]:
            bucket_targets[idx] += 1
    elif remainder < 0:
        # Should not happen, but guard by removing from smallest fractional parts
        fractional_parts = float_targets - bucket_targets
        order = np.argsort(fractional_parts)
        to_remove = min(-remainder, len(order))
        for idx in order[:to_remove]:
            if bucket_targets[idx] > 0:
                bucket_targets[idx] -= 1
        remainder = 0  # best effort safeguard
    
    rank_label = _PAIR_RANK_LABELS.get(internal_rank, str(internal_rank))
    
    diag_entries = [{
        'target': int(bucket_targets[i]),
        'sampled': 0,
        'borrowed_from': {},
        'residual_deficit': 0,
    } for i in range(len(bucket_targets))]
    
    sampled_indices = []
    
    # Sample from each regular bucket (right to left for deficit handling)
    bucket_order = list(range(n_buckets - 1, -1, -1))  # Start from hardest bucket
    
    for b in bucket_order:
        target = bucket_targets[b]
        if target <= 0:
            continue
        
        # Get available pairs in this bucket
        rank_available = rank_mask & available_mask
        bucket_mask = (bucket_assignments == b) & rank_available
        bucket_indices = np.where(bucket_mask)[0]
        n_available = len(bucket_indices)
        
        if n_available >= target:
            # Enough pairs - sample randomly
            chosen = _RNG.choice(bucket_indices, size=target, replace=False)
            sampled_indices.extend(chosen.tolist())
            available_mask[chosen] = False
            diag_entries[b]['sampled'] += int(target)
        else:
            # Not enough - take all and note deficit
            if n_available > 0:
                sampled_indices.extend(bucket_indices.tolist())
                available_mask[bucket_indices] = False
            diag_entries[b]['sampled'] += int(n_available)
            
            deficit = target - n_available
            if deficit > 0:
                print(f"WARNING: Pair bucket {b} for rank '{rank_label}' only has "
                      f"{n_available}/{target} pairs. Borrowing from easier buckets.")
                left_bucket = b - 1
                # Keep shifting left until deficit is cleared or we run out of buckets
                while deficit > 0 and left_bucket >= 0:
                    rank_available = rank_mask & available_mask
                    left_mask = (bucket_assignments == left_bucket) & rank_available
                    left_indices = np.where(left_mask)[0]
                    n_left = len(left_indices)
                    
                    if n_left > 0:
                        borrow_count = min(deficit, n_left)
                        # Sort by error (descending) to get hardest from this bucket
                        sorted_idx = np.argsort(-relative_errors[left_indices])
                        borrowed = left_indices[sorted_idx[:borrow_count]]
                        sampled_indices.extend(borrowed.tolist())
                        available_mask[borrowed] = False
                        deficit -= borrow_count
                        diag_entries[b]['sampled'] += int(borrow_count)
                        prev = diag_entries[b]['borrowed_from'].get(left_bucket, 0)
                        diag_entries[b]['borrowed_from'][left_bucket] = prev + int(borrow_count)
                    left_bucket -= 1
                
                diag_entries[b]['residual_deficit'] = int(deficit)
                if deficit > 0:
                    print(f"WARNING: Pair bucket {b} for rank '{rank_label}' still short by "
                          f"{deficit} pairs after borrowing. Marking shortfall for 'any' bucket.")
                    # Push remaining deficit into 'any' bucket so it can attempt to fill it
                    bucket_targets[-1] += deficit
                    diag_entries[-1]['target'] += int(deficit)
    
    # Handle 'any' bucket - sample from remaining available pairs of this rank
    any_target = bucket_targets[-1]
    any_taken = 0
    if any_target > 0:
        remaining_mask = rank_mask & available_mask
        remaining_indices = np.where(remaining_mask)[0]
        n_remaining = len(remaining_indices)
        
        if n_remaining > 0:
            take_count = min(any_target, n_remaining)
            chosen = _RNG.choice(remaining_indices, size=take_count, replace=False)
            sampled_indices.extend(chosen.tolist())
            available_mask[chosen] = False
            any_taken = int(take_count)
        else:
            any_taken = 0
    diag_entries[-1]['sampled'] += int(any_taken)
    diag_entries[-1]['residual_deficit'] = int(max(any_target - any_taken, 0))
    
    # Calculate final deficit
    actual_sampled = len(sampled_indices)
    deficit = n_to_sample - actual_sampled
    
    bucket_stats = {
        'targets': [int(entry['target']) for entry in diag_entries],
        'sampled': [int(entry['sampled']) for entry in diag_entries],
        'residual_deficit': [int(entry['residual_deficit']) for entry in diag_entries],
        'borrowed_from': [
            {int(k): int(v) for k, v in entry['borrowed_from'].items()}
            for entry in diag_entries
        ],
    }
    
    return sampled_indices, deficit, bucket_stats


def mine_pairs(train_sequences, n_pairs_to_mine, seq_embeddings_distances, train_regions, adjusted_distances, pairwise_ranks, taxon_counts_subsampled=None, train_embeddings=None, duplicate_sequences=None, duplicate_indices=None, duplicate_embeddings=None, duplicate_regions=None, warmup_phase=0.0, verbose=False, log=False, batch_num=None, logs_dir=None, pair_distances_df=None, pair_error_metrics_df=None, phylum_codes_for_logging=None, phylum_step_stats_out=None):
    """
    Mine pairs using per-rank percentile-bucket sampling based on relative squared error.
    
    This function implements separate mining per taxonomic rank to remove bias from ranks
    with large volumes of pairs. Each rank gets its own bucket boundaries computed from
    its representative set, and the batch budget is divided across ranks according to
    EMA-smoothed hardness metrics.
    
    Algorithm:
    1. Determine enabled ranks and prepare lookup structures
    2. Compute per-rank batch budgets using EMA hardness with capped proportions
    3. Gather per-rank representative sets (up to PAIR_MINING_REPRESENTATIVE_SET_SIZES[rank])
    4. Compute relative squared errors for all pairs in the representative sets
    5. Update EMA hardness buffers with per-rank mean errors
    6. Compute per-rank bucket thresholds from each rank's representative set
    7. Sample pairs from each rank according to its budget
    8. Construct output tensors

    Args:
        train_sequences: torch.Tensor of shape (N_TRAIN_SEQUENCES, MAX_MODEL_SEQ_LEN, 3) OR (N_TRAIN_SEQUENCES, 4**K)
        n_pairs_to_mine: int
        seq_embeddings_distances: np.ndarray of shape (N_TRAIN_SEQUENCES, N_TRAIN_SEQUENCES)
        train_regions: np.ndarray of shape (N_TRAIN_SEQUENCES,)
        adjusted_distances: np.ndarray of shape (N_TRAIN_SEQUENCES, N_TRAIN_SEQUENCES)
            Domain-adjusted pairwise distances (typically subsampled).
        pairwise_ranks: np.ndarray of shape: (N_TRAIN_SEQUENCES, N_TRAIN_SEQUENCES) | dtype: int8
            Pairwise ranks matrix (typically subsampled). 
            The ranks of the pairs (the rank they *share*):
                - -2 = Ignore Pair (explicitly excluded from mining)
                - -1 = different domains (both Prokaryota)
                -  0 = same domain, different phyla
                -  1 = same phylum, different classes
                -  2 = same class, different orders
                -  3 = same order, different families
                -  4 = same family, different genera
                -  5 = same genus, different species
                -  6 = same species, different sequences
            Note: Rank 7 (subsequence) pairs are handled separately via duplicate_* parameters.
        taxon_counts_subsampled: np.ndarray of shape (N_TRAIN_SEQUENCES, 7) or None
            Per-sequence taxon sizes for the subsampled training set, used for
            representative-set taxon-size balancing when enabled. May be float32 when
            counts have been corrected for domain-based downsampling (see
            mine_pairs_and_triplets() in triplet_pair_mining.py).
        train_phylum_labels: Optional np.ndarray/list of shape (N_TRAIN_SEQUENCES,)
            Phylum labels aligned to local train indices (used only for taxonomy logging diagnostics).
        train_class_labels: Optional np.ndarray/list of shape (N_TRAIN_SEQUENCES,)
            Class labels aligned to local train indices (used only for taxonomy logging diagnostics).
        train_embeddings: torch.Tensor of shape (N_TRAIN_SEQUENCES, EMBED_DIMS) or None
            Embeddings for training sequences. Required if mining subsequence pairs.
        duplicate_sequences: torch.Tensor of shape (N_DUPLICATES, MAX_MODEL_SEQ_LEN, 3) or None
            Duplicate sequences for subsequence pair mining (region selection + augmentation already applied).
        duplicate_indices: np.ndarray of shape (N_DUPLICATES,) or None
            Index into train_sequences for each duplicate's original sequence.
        duplicate_embeddings: torch.Tensor of shape (N_DUPLICATES, EMBED_DIMS) or None
            Embeddings for duplicate sequences.
        duplicate_regions: np.ndarray of shape (N_DUPLICATES,) or None
            Selected region indices for each duplicate sequence (may match originals when SUBSEQUENCES_ALWAYS_CROSS_REGION=False).
        warmup_phase: float in [0,1]
            Proportion of sampling weight to keep on the uniform 'any' bucket (0 = normal bucketed mining, 1 = fully uniform).
        verbose: bool
        log: bool - If True, writes statistics to log and CSV files.
        batch_num: int - Batch number for logging (required if log=True).
        logs_dir: str - Directory for log files (required if log=True).
        pair_distances_df: pd.DataFrame or None
            Optional dataframe for tracking pair distances over time. If provided, will be populated with 
            distance statistics (box plot data) during logging.
        pair_error_metrics_df: pd.DataFrame or None
            Optional dataframe for tracking per-rank error metrics over time. If provided, will be populated
            with the composite error metric (mean_error * PAIR_EMA_MEAN_WEIGHT + quartiles * PAIR_EMA_QUARTILES_WEIGHT)
            computed from relative squared errors.
        phylum_codes_for_logging: np.ndarray of shape (N_TRAIN_SEQUENCES,) or None
            Optional local phylum code per sequence. When provided with phylum_step_stats_out, this function
            appends representative-set phylum diagnostics for pair mining.
        phylum_step_stats_out: list or None
            Optional list to append phylum step dictionaries into.

    Returns:
        mined_pairs: torch.Tensor of shape (n_pairs_mined, 2, MAX_MODEL_SEQ_LEN, 3) OR (n_pairs_mined, 2, 4**K)
        mined_pair_distances: torch.Tensor of shape (n_pairs_mined,)
        mined_pair_ranks: torch.Tensor of shape (n_pairs_mined,)
        mined_pair_buckets: torch.Tensor of shape (n_pairs_mined,)
        mined_pair_region_pairs: torch.Tensor of shape (n_pairs_mined, 2) with sorted region indices

    Global variables:
        gb.PAIR_RANKS: Boolean list for which ranks to mine [domain, phylum, class, order, family, genus, species, sequence, subsequence]
        gb.PAIR_MINING_BUCKETS: List of tuples (percentile_gap, sampling_proportion) defining buckets and sampling rates.
        gb.PAIR_MINING_REPRESENTATIVE_SET_SIZES: List of ints (length 9) - max candidates per rank for percentile calculations.
        gb.PAIR_MINING_EMA_ALPHA: EMA smoothing factor for hardness metrics.
        gb.PAIR_PER_RANK_BATCH_PROPORTION_MAX: Max proportion of batch any single rank can receive.
        gb.RELATIVE_ERROR_EPSILONS_PAIR_MINING: Per-rank list of small constants to avoid division by zero (length 9)
    """
    
    # Initialize per-rank mining stats for logging (will be populated throughout the function)
    per_rank_stats = {
        'ema_hardness_pre': None,        # EMA values before this batch's update
        'ema_hardness_budget': None,     # EMA values after warmup blending (used for budgets)
        'ema_hardness_post': None,       # EMA values after this batch's update
        'hardness_metrics': np.full(_N_PAIR_RANKS, np.nan, dtype=np.float64),
        'sign_bias_active': False,       # Whether sign-bias is active for pair hardness
        'sign_bias_betas': None,         # Per-rank sign-bias betas (domain->subseq)
        'budgets': None,                 # Requested budget per rank
        'proportions': None,             # Budget proportions per rank
        'sampled_counts': np.zeros(_N_PAIR_RANKS, dtype=np.int64),  # Actual sampled per rank
        'pool_counts': np.zeros(_N_PAIR_RANKS, dtype=np.int64),     # Pool size per rank
        'deficits': np.zeros(_N_PAIR_RANKS, dtype=np.int64),        # Deficit per rank
        'per_rank_thresholds': {},       # Bucket thresholds per rank
        'warmup_phase': warmup_phase,    # Current warmup phase
        'representative_shortages': [],  # Representative set shortfalls for logging
        'bucket_proportions_base': None,     # Raw bucket proportions from config
        'bucket_proportions_target': None,   # Post-warmup blended bucket proportions
        'bucket_diagnostics': [None] * _N_PAIR_RANKS,  # Per-rank bucket fulfillment stats
    }

    func_start_time = time.time()

    def _empty_pair_return(reason=None):
        warning_msg = "WARNING: Empty pair tensors being returned by mine_pairs()."
        if reason:
            warning_msg += f" Reason: {reason}."
        print(warning_msg)
        if len(train_sequences.shape) == 3:
            empty_pairs = torch.empty((0, 2, train_sequences.shape[1], train_sequences.shape[2]), dtype=train_sequences.dtype)
        else:
            empty_pairs = torch.empty((0, 2, train_sequences.shape[1]), dtype=train_sequences.dtype)
        empty_distances = torch.empty((0,), dtype=torch.float32)
        empty_ranks = torch.empty((0,), dtype=torch.int32)
        empty_buckets = torch.empty((0,), dtype=torch.long)
        empty_regions = torch.empty((0, 2), dtype=torch.int16)
        if gb.VERBOSE_MINING_TIMING:
            suffix = f" ({reason})" if reason else ""
            print(f"     > Time taken to mine pairs{suffix}: {time.time() - func_start_time:.4f} seconds")
        return empty_pairs, empty_distances, empty_ranks, empty_buckets, empty_regions

    if n_pairs_to_mine <= 0:
        return _empty_pair_return("Zero pairs requested")

    
    n_train_seqs = train_sequences.shape[0]
    
    part_start_time = time.time()
    # 1. Determine enabled ranks and prepare for per-rank mining -------------------------------------------------
    # Effective pair ranks combine the static PAIR_RANKS config with
    # INTRODUCE_RANK_AT_BATCHES for the current batch.
    # Subsequence pairs (rank 7, index 8) are handled separately via duplicate_* parameters.
    
    # Build lookup array for which ranks are enabled (index = internal_rank + 1)
    rank_lookup_array = np.array(get_effective_pair_ranks(batch_num=batch_num), dtype=np.bool_)
    
    # Check if we should include subsequence pairs
    has_subsequence_pairs = (
        rank_lookup_array[8] and
        duplicate_indices is not None and len(duplicate_indices) > 0
    )
    # Keep subsequence rank disabled when there are no duplicate candidates.
    if not has_subsequence_pairs:
        rank_lookup_array[8] = False
    
    if gb.VERBOSE_MINING_TIMING:
        print(f"     > Time taken to get target ranks: {time.time() - part_start_time:.4f} seconds")
    if not np.any(rank_lookup_array):
        return _empty_pair_return("No enabled pair ranks")
    
    # 2. Compute per-rank batch budgets using EMA hardness -------------------------------------------------
    part_start_time = time.time()
    
    # Get EMA hardness values (or use equal weights if not initialized)
    if gb.PAIR_MINING_EMA_HARDNESS is not None:
        ema_hardness = gb.PAIR_MINING_EMA_HARDNESS.copy()
        per_rank_stats['ema_hardness_pre'] = ema_hardness.copy()  # Capture pre-update EMA
    else:
        ema_hardness = np.ones(_N_PAIR_RANKS, dtype=np.float64)
        per_rank_stats['ema_hardness_pre'] = ema_hardness.copy()
    
    warmup_value = 0.0 if warmup_phase is None else warmup_phase
    warmup_blend = float(np.clip(warmup_value, 0.0, 1.0))
    if warmup_blend > 0:
        # Blend EMA hardness with uniform weights so early batches stay balanced.
        uniform_weights = rank_lookup_array.astype(np.float64)
        ema_hardness = (
            (1.0 - warmup_blend) * ema_hardness +
            warmup_blend * uniform_weights
        )
    per_rank_stats['ema_hardness_budget'] = ema_hardness.copy()
    
    # Compute per-rank budgets
    per_rank_budgets, per_rank_proportions = _compute_per_rank_budgets(
        ema_hardness, 
        n_pairs_to_mine, 
        gb.PAIR_PER_RANK_BATCH_PROPORTION_MAX,
        rank_lookup_array,
        gb.PAIR_MINING_EMA_WEIGHT_EXPONENT,
        cap_min=gb.PAIR_PER_RANK_BATCH_PROPORTION_MIN
    )
    
    # Store budgets and proportions for logging
    per_rank_stats['budgets'] = per_rank_budgets.copy()
    per_rank_stats['proportions'] = per_rank_proportions.copy()
    
    if gb.VERBOSE_MINING_TIMING:
        print(f"     > Time taken to compute per-rank budgets: {time.time() - part_start_time:.4f} seconds")
        budget_str = ", ".join([f"{_PAIR_RANK_LABELS[r-1]}={per_rank_budgets[r]}" 
                                for r in range(_N_PAIR_RANKS) if rank_lookup_array[r]])
        print(f"     > Per-rank budgets: {budget_str}")
    

    part_start_time = time.time()
    # 3. Get flat indices of valid pairs with per-rank representative set limits -------------------------------------------------
    
    sub_part_start_time = time.time()
    
    # Get per-rank representative set sizes
    max_samples_per_rank = np.array(gb.PAIR_MINING_REPRESENTATIVE_SET_SIZES, dtype=np.int64)
    # Don't include subsequence in regular pair gathering (handled separately)
    max_samples_per_rank[8] = 0
    
    if gb.VERBOSE_MINING_TIMING:
        print(f"        > Time taken for sub part #0: {time.time() - sub_part_start_time:.4f} seconds")
    
    # Build rank lookup for regular pairs only (exclude subsequence)
    regular_rank_lookup = rank_lookup_array.copy()
    regular_rank_lookup[8] = False  # Subsequence handled separately
    
    representative_shortages = []
    if np.any(regular_rank_lookup):
        # Get seed for reproducibility
        current_seed = _RNG.integers(0, 2**31 - 1)
        
        # Get valid indices per rank with per-rank subsampling
        # Pass 2D pairwise_ranks directly for better cache locality
        valid_flat_indices, valid_flat_ranks, total_valid_per_rank = _get_valid_pair_indices_per_rank(
            pairwise_ranks, regular_rank_lookup, max_samples_per_rank, current_seed,
            taxon_counts=taxon_counts_subsampled,
            shortage_recorder=representative_shortages
        )
        
    else:
        valid_flat_indices = np.array([], dtype=np.int64)
        valid_flat_ranks = np.array([], dtype=np.int8)
        total_valid_per_rank = np.zeros(_N_PAIR_RANKS, dtype=np.int64)
        representative_shortages = []

    if gb.VERBOSE_MINING_TIMING:
        print(f"        > Time taken for sub part #1: {time.time() - sub_part_start_time:.4f} seconds")
    
    per_rank_stats['representative_shortages'] = representative_shortages
    
    # Store total available counts before subsampling for logging
    # (This is what we could have sampled, not what we actually sampled)
    # Note: total_valid_per_rank covers ranks 0-7 (indices 0-7), subsequence (rank 8) is added later
    per_rank_stats['available_counts'] = total_valid_per_rank.copy()
    
    # valid_flat_indices.shape: (n_processing_regular,) 
    n_regular_pairs = len(valid_flat_indices)
    
    # Compute subsequence pair data if applicable
    n_subseq_pairs = 0
    subseq_pred_distances = None
    subseq_true_distances = None
    
    if has_subsequence_pairs:
        sub_part_start_time = time.time()
        n_subseq_pairs = len(duplicate_indices)
        # Store original count before subsampling for available_counts logging
        original_n_subseq_pairs = n_subseq_pairs
        max_subseq_candidates = int(gb.PAIR_MINING_REPRESENTATIVE_SET_SIZES[8])
        if max_subseq_candidates <= 0:
            if n_subseq_pairs > 0:
                print("WARNING: Subsequence representative set size is 0. Skipping subsequence pairs.")
            n_subseq_pairs = 0
            has_subsequence_pairs = False
            if len(per_rank_budgets) > 8:
                per_rank_budgets[8] = 0
            rank_lookup_array[8] = False
        elif n_subseq_pairs > max_subseq_candidates:
            chosen = _RNG.choice(n_subseq_pairs, size=max_subseq_candidates, replace=False)
            chosen.sort()
            torch_indexer = torch.as_tensor(chosen, dtype=torch.long)
            duplicate_indices = duplicate_indices[chosen]
            if duplicate_sequences is not None:
                if isinstance(duplicate_sequences, torch.Tensor):
                    duplicate_sequences = duplicate_sequences[torch_indexer]
                else:
                    duplicate_sequences = duplicate_sequences[chosen]
            if duplicate_embeddings is not None:
                if isinstance(duplicate_embeddings, torch.Tensor):
                    duplicate_embeddings = duplicate_embeddings[torch_indexer]
                else:
                    duplicate_embeddings = duplicate_embeddings[chosen]
            if duplicate_regions is not None:
                duplicate_regions = duplicate_regions[chosen]
            n_subseq_pairs = max_subseq_candidates
        elif n_subseq_pairs < max_subseq_candidates:
            print(f"WARNING: Subsequence rank has {n_subseq_pairs} candidates, "
                  f"fewer than representative set size {max_subseq_candidates}.")
        
        # Store original subsequence count in available_counts (before subsampling)
        per_rank_stats['available_counts'][8] = original_n_subseq_pairs
        
        # Compute predicted distances for subsequence pairs
        # duplicate_embeddings[i] <-> train_embeddings[duplicate_indices[i]]
        # Cosine distance = 1 - cosine_similarity
        
        # Convert to numpy if needed
        if isinstance(duplicate_embeddings, torch.Tensor):
            dup_emb_np = duplicate_embeddings.detach().cpu().numpy()
        else:
            dup_emb_np = duplicate_embeddings
            
        if isinstance(train_embeddings, torch.Tensor):
            train_emb_np = train_embeddings.detach().cpu().numpy()
        else:
            train_emb_np = train_embeddings
        
        if gb.VERBOSE_MINING_TIMING:
            print(f"        > Time taken for sub part #2: {time.time() - sub_part_start_time:.4f} seconds")
        sub_part_start_time = time.time()
        
        # Get embeddings for the original sequence of each duplicate
        orig_emb_np = train_emb_np[duplicate_indices]
        # orig_emb_np.shape: (n_subseq_pairs, EMBED_DIMS)
        # dup_emb_np.shape: (n_subseq_pairs, EMBED_DIMS)

        if gb.VERBOSE_MINING_TIMING:
            print(f"        > Time taken for sub part #3: {time.time() - sub_part_start_time:.4f} seconds")
        sub_part_start_time = time.time()
        
        # L2 normalize embeddings
        epsilon = 1e-8
        orig_norms = np.linalg.norm(orig_emb_np, axis=1, keepdims=True) + epsilon
        dup_norms = np.linalg.norm(dup_emb_np, axis=1, keepdims=True) + epsilon
        orig_normalized = orig_emb_np / orig_norms
        dup_normalized = dup_emb_np / dup_norms
        
        if gb.VERBOSE_MINING_TIMING:
            print(f"        > Time taken for sub part #4: {time.time() - sub_part_start_time:.4f} seconds")
        sub_part_start_time = time.time()
        
        # Compute cosine similarity (row-wise dot product)
        cos_sim = np.sum(orig_normalized * dup_normalized, axis=1)
        
        if gb.VERBOSE_MINING_TIMING:
            print(f"        > Time taken for sub part #5: {time.time() - sub_part_start_time:.4f} seconds")
        sub_part_start_time = time.time()
        
        # Cosine distance = 1 - cosine_similarity
        subseq_pred_distances = 1.0 - cos_sim
        # subseq_pred_distances.shape: (n_subseq_pairs,)

        if gb.VERBOSE_MINING_TIMING:
            print(f"        > Time taken for sub part #6: {time.time() - sub_part_start_time:.4f} seconds")
        sub_part_start_time = time.time()
        
        # True distance for subsequence pairs is configurable (allows a relaxed target)
        subseq_true_distances = np.full(n_subseq_pairs, gb.SUB_SEQUENCE_TRUE_DISTANCE, dtype=np.float32)
        # subseq_true_distances.shape: (n_subseq_pairs,)

        if gb.VERBOSE_MINING_TIMING:
            print(f"        > Time taken for sub part #7: {time.time() - sub_part_start_time:.4f} seconds")


    sub_part_start_time = time.time()
    
    # Handle edge case with no valid pairs from either source
    if n_regular_pairs == 0 and n_subseq_pairs == 0:
        if phylum_step_stats_out is not None:
            rep_step_stats = _build_pair_representative_phylum_step_stats(
                phylum_codes=phylum_codes_for_logging,
                n_train_seqs=n_train_seqs,
                regular_flat_indices=np.empty((0,), dtype=np.int64),
                subseq_orig_indices=None,
                pairwise_ranks=pairwise_ranks,
            )
            if rep_step_stats is not None:
                phylum_step_stats_out.append(rep_step_stats)
        return _empty_pair_return("No valid pairs (regular or subsequence)")

    if gb.VERBOSE_MINING_TIMING:
        print(f"        > Time taken for sub part #8: {time.time() - sub_part_start_time:.4f} seconds")

    if phylum_step_stats_out is not None:
        rep_step_stats = _build_pair_representative_phylum_step_stats(
            phylum_codes=phylum_codes_for_logging,
            n_train_seqs=n_train_seqs,
            regular_flat_indices=valid_flat_indices,
            subseq_orig_indices=duplicate_indices if n_subseq_pairs > 0 else None,
            pairwise_ranks=pairwise_ranks,
        )
        if rep_step_stats is not None:
            phylum_step_stats_out.append(rep_step_stats)

    
    # Log representative-set taxon-size balancing status
    rep_balance_active = (
        gb.USE_REPRESENTATIVE_TAXON_SIZE_BALANCING
        and gb.PAIR_REPRESENTATIVE_TAXON_SIZE_BALANCE_LAMBDA > 0
        and taxon_counts_subsampled is not None
    )
    per_rank_stats['rep_taxon_balance_active'] = rep_balance_active
    per_rank_stats['rep_taxon_balance_lambda'] = gb.PAIR_REPRESENTATIVE_TAXON_SIZE_BALANCE_LAMBDA if rep_balance_active else 0.0
    
    if gb.VERBOSE_MINING_TIMING:
        print(f"     > Time taken to get valid flat indices: {time.time() - part_start_time:.4f} seconds")
        if rep_balance_active:
            clip_str = f", clip={gb.REPRESENTATIVE_TAXON_SIZE_BALANCE_WEIGHT_CLIP}" if gb.REPRESENTATIVE_TAXON_SIZE_BALANCE_WEIGHT_CLIP else ""
            print(f"     > Representative taxon-size balancing: active (lambda={gb.PAIR_REPRESENTATIVE_TAXON_SIZE_BALANCE_LAMBDA}{clip_str})")
    

    part_start_time = time.time()
    # 4. Compute relative squared error + optional sign-biased hardness for valid pairs ----------------
    # rel_sq_error = ((d_pred - d_true)^2) / (d_true + eps)
    # Where eps = gb.RELATIVE_ERROR_EPSILONS_PAIR_MINING[rank]
    # Also track signed error for debugging: d_pred - d_true (positive = too far, negative = too close)
    
    # Compute for regular pairs (from pairwise matrix)
    if n_regular_pairs > 0:
        regular_pred_distances = seq_embeddings_distances.ravel()[valid_flat_indices]
        regular_true_distances = adjusted_distances.ravel()[valid_flat_indices]
        regular_pair_ranks = pairwise_ranks.ravel()[valid_flat_indices].astype(np.int16)
        regular_signed_errors = regular_pred_distances - regular_true_distances
        # Use rank-specific epsilons
        epsilons_arr = np.array(gb.RELATIVE_ERROR_EPSILONS_PAIR_MINING, dtype=np.float32)
        regular_epsilons = epsilons_arr[regular_pair_ranks + 1] # Map internal ranks (-1..7) to indices (0..8)
        regular_relative_errors = (regular_signed_errors ** 2) / (regular_true_distances + regular_epsilons)
    else:
        regular_pred_distances = np.array([], dtype=np.float32)
        regular_true_distances = np.array([], dtype=np.float32)
        regular_pair_ranks = np.array([], dtype=np.int16)
        regular_signed_errors = np.array([], dtype=np.float32)
        regular_relative_errors = np.array([], dtype=np.float32)
    
    # Compute for subsequence pairs (from duplicate data)
    if n_subseq_pairs > 0:
        subseq_signed_errors = subseq_pred_distances - subseq_true_distances
        subseq_pair_ranks = np.full(n_subseq_pairs, 7, dtype=np.int16)
        subseq_epsilon = gb.RELATIVE_ERROR_EPSILONS_PAIR_MINING[8]
        subseq_relative_errors = (subseq_signed_errors ** 2) / (subseq_true_distances + subseq_epsilon)
    else:
        subseq_signed_errors = np.array([], dtype=np.float32)
        subseq_pair_ranks = np.array([], dtype=np.int16)
        subseq_relative_errors = np.array([], dtype=np.float32)
        subseq_pred_distances = np.array([], dtype=np.float32)
        subseq_true_distances = np.array([], dtype=np.float32)
    
    # Combine regular and subsequence pairs into unified pool
    # Pool layout: [regular_pairs..., subseq_pairs...]
    flat_pred_distances = np.concatenate([regular_pred_distances, subseq_pred_distances])
    flat_true_distances = np.concatenate([regular_true_distances, subseq_true_distances])
    signed_errors = np.concatenate([regular_signed_errors, subseq_signed_errors])
    relative_errors = np.concatenate([regular_relative_errors, subseq_relative_errors])
    flat_ranks = np.concatenate([regular_pair_ranks, subseq_pair_ranks])
    # flat_pred_distances.shape: (n_processing,) where n_processing = n_regular_pairs + n_subseq_pairs
    # flat_true_distances.shape: (n_processing,)
    # signed_errors.shape: (n_processing,)
    # relative_errors.shape: (n_processing,)
    
    n_processing = len(relative_errors)
    available_mask = np.ones(n_processing, dtype=bool)

    # Optional sign-aware hardness bias:
    #   hardness = rel_sq_error * exp(beta_r * sign), sign=+1 (too far), -1 (too close)
    pair_sign_betas, sign_bias_active = _get_pair_sign_bias_betas()
    if sign_bias_active:
        hardness_errors = _apply_pair_sign_bias_to_hardness(
            relative_errors, signed_errors, flat_ranks, pair_sign_betas
        )
    else:
        hardness_errors = relative_errors

    per_rank_stats['sign_bias_active'] = sign_bias_active
    per_rank_stats['sign_bias_betas'] = pair_sign_betas.copy()

    if gb.VERBOSE_MINING_TIMING:
        print(f"     > Time taken to compute relative squared errors: {time.time() - part_start_time:.4f} seconds")
        if sign_bias_active:
            print("     > Pair sign-bias hardness: active")
    
    # 5. Compute per-rank hardness metrics and update EMA -------------------------------------------------
    part_start_time = time.time()
    
    # Compute hardness metrics per rank for EMA update
    per_rank_metrics = np.full(_N_PAIR_RANKS, np.nan, dtype=np.float64)
    for rank_idx in range(_N_PAIR_RANKS):
        internal_rank = rank_idx - 1  # 0..8 -> -1..7
        rank_mask = (flat_ranks == internal_rank)
        if np.any(rank_mask):
            rank_errors = hardness_errors[rank_mask]
            
            # Mean error
            mean_err = np.mean(rank_errors)
            
            # Quartile errors
            p25_err = np.percentile(rank_errors, 25)
            p75_err = np.percentile(rank_errors, 75)
            
            # Calculate metric
            metric = (mean_err * gb.PAIR_EMA_MEAN_WEIGHT) + ((p25_err + p75_err) * gb.PAIR_EMA_QUARTILES_WEIGHT)
            
            per_rank_metrics[rank_idx] = metric
    
    # Store raw hardness metrics for logging (these values feed the EMA update)
    per_rank_stats['hardness_metrics'] = per_rank_metrics.copy()
    
    # Update EMA with current batch's hardness metrics
    _update_pair_ema_hardness(per_rank_metrics, gb.PAIR_MINING_EMA_ALPHA)
    
    # Capture post-update EMA for logging
    if gb.PAIR_MINING_EMA_HARDNESS is not None:
        per_rank_stats['ema_hardness_post'] = gb.PAIR_MINING_EMA_HARDNESS.copy()
    
    if gb.VERBOSE_MINING_TIMING:
        print(f"     > Time taken to update EMA hardness: {time.time() - part_start_time:.4f} seconds")


    part_start_time = time.time()
    # 6. Compute per-rank bucket thresholds -------------------------------------------------
    # Parse PAIR_MINING_BUCKETS to get gaps and proportions
    bucket_configs = [b for b in gb.PAIR_MINING_BUCKETS if b[0] is not None]
    any_bucket_config = [b for b in gb.PAIR_MINING_BUCKETS if b[0] is None]
    
    # Get bucket gaps and sampling proportions
    percentile_gaps = [b[0] for b in bucket_configs]
    n_buckets = len(percentile_gaps)
    
    # Get target proportions with warmup adjustment
    base_proportions = [b[1] for b in bucket_configs]
    if any_bucket_config:
        base_proportions.append(any_bucket_config[0][1])
    else:
        base_proportions.append(0.0)
    base_proportions = np.array(base_proportions, dtype=np.float64)
    
    if warmup_phase > 0 and len(base_proportions) == n_buckets + 1:
        # Shift proportion toward uniform sampling via the 'any' bucket
        uniform = np.zeros_like(base_proportions)
        uniform[-1] = 1.0
        target_proportions = (1 - warmup_phase) * base_proportions + warmup_phase * uniform
    else:
        target_proportions = base_proportions
    
    per_rank_stats['bucket_proportions_base'] = base_proportions.copy()
    per_rank_stats['bucket_proportions_target'] = target_proportions.copy()
    
    use_uniform_buckets = warmup_phase >= 0.999 or n_processing == 0
    if use_uniform_buckets:
        # During full warmup, use uniform sampling (single bucket)
        per_rank_thresholds = {r: [] for r in range(-1, 8)}
        bucket_assignments = np.zeros(n_processing, dtype=np.int32)
        percentile_thresholds = []
    else:
        # Compute per-rank bucket thresholds from each rank's representative set
        per_rank_thresholds = _compute_per_rank_bucket_thresholds(
            hardness_errors, flat_ranks, percentile_gaps
        )
        
        # Assign buckets per-rank
        bucket_assignments = _assign_buckets_per_rank(
            hardness_errors, flat_ranks, per_rank_thresholds
        )
        
        # For logging compatibility, compute global percentile thresholds
        cumulative = np.cumsum(percentile_gaps)
        percentile_thresholds = cumulative[:-1].tolist() if len(cumulative) > 1 else []
    
    # Store per-rank thresholds for logging
    per_rank_stats['per_rank_thresholds'] = per_rank_thresholds.copy()
    
    # Update pool counts before logging so threshold logs capture the actual pool sizes
    pool_counts = per_rank_stats.get('pool_counts')
    if pool_counts is not None:
        pool_counts[:] = 0
        if n_processing > 0:
            # Map internal ranks (-1..7) to indices (0..8) and count occurrences
            rank_indices = (flat_ranks + 1).astype(np.int64, copy=False)
            valid_mask = (rank_indices >= 0) & (rank_indices < _N_PAIR_RANKS)
            if np.any(valid_mask):
                counts = np.bincount(rank_indices[valid_mask], minlength=_N_PAIR_RANKS)
                pool_counts[:_N_PAIR_RANKS] = counts[:_N_PAIR_RANKS]

    # Log bucket thresholds if enabled
    if log and getattr(gb, "LOG_MINING_BUCKET_THRESHOLDS", False) and batch_num is not None and logs_dir is not None:
         log_bucket_thresholds(
             batch_num=batch_num,
             logs_dir=logs_dir,
             mining_type="pair",
             per_rank_thresholds=per_rank_thresholds,
             per_rank_stats=per_rank_stats,
             rank_map=_PAIR_RANK_LABELS,
             bucket_configs=gb.PAIR_MINING_BUCKETS
         )
    
    if gb.VERBOSE_MINING_TIMING:
        print(f"     > Time taken to compute per-rank bucket thresholds: {time.time() - part_start_time:.4f} seconds")
    

    part_start_time = time.time()
    # 7. Sample pairs from each rank according to per-rank budgets -------------------------------------------------
    sampled_local_indices = []
    total_deficit = 0
    
    # Sample from each enabled rank
    for rank_idx in range(_N_PAIR_RANKS):
        if not rank_lookup_array[rank_idx]:
            continue
        
        internal_rank = rank_idx - 1  # 0..8 -> -1..7
        rank_budget = int(per_rank_budgets[rank_idx])
        
        if rank_budget <= 0:
            continue
        
        # Get mask for this rank
        rank_mask = (flat_ranks == internal_rank)
        rank_pool_count = np.sum(rank_mask)
        per_rank_stats['pool_counts'][rank_idx] = rank_pool_count
        
        if not np.any(rank_mask):
            total_deficit += rank_budget
            per_rank_stats['deficits'][rank_idx] = rank_budget
            rank_label = _PAIR_RANK_LABELS.get(internal_rank, str(internal_rank))
            print(f"WARNING: Pair mining rank '{rank_label}' has no candidates in pool, "
                  f"budget of {rank_budget} pairs unfulfilled.")
            continue
        
        # Sample from this rank's buckets
        rank_sampled, rank_deficit, bucket_diag = _sample_from_rank_buckets(
            rank_budget, rank_mask, bucket_assignments, hardness_errors,
            n_buckets, target_proportions, available_mask, internal_rank
        )
        if bucket_diag is not None:
            per_rank_stats['bucket_diagnostics'][rank_idx] = bucket_diag
        
        sampled_local_indices.extend(rank_sampled)
        total_deficit += rank_deficit
        
        # Store per-rank stats for logging
        per_rank_stats['sampled_counts'][rank_idx] = len(rank_sampled)
        per_rank_stats['deficits'][rank_idx] = rank_deficit
        
        if rank_deficit > 0:
            rank_label = _PAIR_RANK_LABELS.get(internal_rank, str(internal_rank))
            print(f"WARNING: Pair mining rank '{rank_label}' short by {rank_deficit} pairs "
                  f"(sampled {len(rank_sampled)}/{rank_budget}).")
    
    # Report overall deficit if any
    if total_deficit > 0:
        print(f"WARNING: Pair mining total deficit: {total_deficit} pairs "
              f"(sampled {len(sampled_local_indices)}/{n_pairs_to_mine}).")

    # Convert to numpy array
    sampled_local_indices = np.array(sampled_local_indices, dtype=np.int64)
    
    # Handle case where we couldn't sample enough pairs
    n_pairs_sampled = len(sampled_local_indices)
    if n_pairs_sampled == 0:
        return _empty_pair_return("Sampling step returned zero pairs")
    if gb.VERBOSE_MINING_TIMING:
        print(f"     > Time taken to sample pairs: {time.time() - part_start_time:.4f} seconds")



    part_start_time = time.time()
    # 8. Construct output tensors -------------------------------------------------
    # Pool layout: [regular_pairs (indices 0 to n_regular_pairs-1), subseq_pairs (indices n_regular_pairs to n_processing-1)]
    # sampled_local_indices contains indices into this combined pool
    
    # Separate sampled indices into regular and subsequence pairs
    regular_mask = sampled_local_indices < n_regular_pairs
    subseq_mask = ~regular_mask
    
    sampled_regular_indices = sampled_local_indices[regular_mask]
    sampled_subseq_indices = sampled_local_indices[subseq_mask] - n_regular_pairs  # Convert to local duplicate index
    
    n_regular_sampled = len(sampled_regular_indices)
    n_subseq_sampled = len(sampled_subseq_indices)
    
    # Initialize output lists
    pairs_list = []
    distances_list = []
    ranks_list = []
    region_pairs_list = []
    indices_list = []  # Track local indices (into train_sequences) for each pair
    
    # Process regular pairs
    if n_regular_sampled > 0:
        # Map sampled local indices back to original global flat indices
        selected_flat_indices = valid_flat_indices[sampled_regular_indices]
        
        # Convert flat indices to 2D indices
        seq_i_indices = selected_flat_indices // n_train_seqs
        seq_j_indices = selected_flat_indices % n_train_seqs
        
        # Extract sequence pairs
        seq_pairs_i = train_sequences[seq_i_indices]
        seq_pairs_j = train_sequences[seq_j_indices]
        regular_pairs = torch.stack([seq_pairs_i, seq_pairs_j], dim=1)
        # regular_pairs.shape: (n_regular_sampled, 2, MAX_MODEL_SEQ_LEN, 3) or (n_regular_sampled, 2, 4**K)
        
        # Track local indices for regular pairs
        regular_pair_indices = np.stack([seq_i_indices, seq_j_indices], axis=1)
        # regular_pair_indices.shape: (n_regular_sampled, 2) dtype: int64
        
        # Extract true distances
        regular_distances = regular_true_distances[sampled_regular_indices]
        
        # Get ranks for regular pairs
        flat_ranks_full = pairwise_ranks.ravel()
        sampled_shared_ranks = flat_ranks_full[selected_flat_indices]
        # The pair rank_idx = shared_rank + 1 (since shared_rank is (rank_idx - 1))
        regular_ranks = sampled_shared_ranks + 1
        
        # Track region combinations (sorted so R1-R2 == R2-R1)
        regular_region_pairs = np.stack([train_regions[seq_i_indices], train_regions[seq_j_indices]], axis=1)
        regular_region_pairs.sort(axis=1)

        pairs_list.append(regular_pairs)
        distances_list.append(regular_distances)
        ranks_list.append(regular_ranks)
        region_pairs_list.append(regular_region_pairs.astype(np.int16, copy=False))
        indices_list.append(regular_pair_indices)
    
    # Process subsequence pairs
    if n_subseq_sampled > 0:
        # For subsequence pairs, pair is (original, duplicate)
        # duplicate_indices[sampled_subseq_indices] gives original sequence index
        # sampled_subseq_indices gives duplicate sequence index
        orig_indices = duplicate_indices[sampled_subseq_indices]
        
        seq_pairs_orig = train_sequences[orig_indices]
        seq_pairs_dup = duplicate_sequences[sampled_subseq_indices]
        subseq_pairs = torch.stack([seq_pairs_orig, seq_pairs_dup], dim=1)
        # subseq_pairs.shape: (n_subseq_sampled, 2, MAX_MODEL_SEQ_LEN, 3) or (n_subseq_sampled, 2, 4**K)
        
        # Track local indices for subsequence pairs (original and duplicate)
        # Note: Duplicates aren't part of train_sequences, so we use n_train_seqs + duplicate_idx as a convention
        # The caller will need to handle duplicate indices separately
        subseq_pair_indices = np.stack([orig_indices, n_train_seqs + sampled_subseq_indices], axis=1)
        # subseq_pair_indices.shape: (n_subseq_sampled, 2) dtype: int64
        
        # True distances for subsequence pairs mirror mining config (allows relaxed targets)
        subseq_distances = np.full(n_subseq_sampled, gb.SUB_SEQUENCE_TRUE_DISTANCE, dtype=np.float32)
        
        # Ranks for subsequence pairs are 8 (shared_rank=7 -> rank_idx=8) regardless of whether the regions match.
        subseq_ranks = np.full(n_subseq_sampled, 8, dtype=np.int32)

        # Region pairs (original + duplicate)
        if duplicate_regions is None:
            raise RuntimeError("duplicate_regions must be provided when mining subsequence pairs.")
        subseq_region_pairs = np.stack([train_regions[orig_indices], duplicate_regions[sampled_subseq_indices]], axis=1)
        subseq_region_pairs.sort(axis=1)
        
        pairs_list.append(subseq_pairs)
        distances_list.append(subseq_distances)
        ranks_list.append(subseq_ranks)
        region_pairs_list.append(subseq_region_pairs.astype(np.int16, copy=False))
        indices_list.append(subseq_pair_indices)
    
    # Concatenate results (maintaining the order: regular then subsequence)
    if len(pairs_list) > 1:
        mined_pairs = torch.cat(pairs_list, dim=0)
        mined_pair_distances = np.concatenate(distances_list)
        mined_pair_ranks = np.concatenate(ranks_list)
        mined_pair_indices = np.concatenate(indices_list, axis=0)
    elif len(pairs_list) == 1:
        mined_pairs = pairs_list[0]
        mined_pair_distances = distances_list[0]
        mined_pair_ranks = ranks_list[0]
        mined_pair_indices = indices_list[0]
    else:
        # Should not happen since we checked n_pairs_sampled > 0 earlier
        if len(train_sequences.shape) == 3:
            mined_pairs = torch.empty((0, 2, train_sequences.shape[1], train_sequences.shape[2]), dtype=train_sequences.dtype)
        else:
            mined_pairs = torch.empty((0, 2, train_sequences.shape[1]), dtype=train_sequences.dtype)
        mined_pair_distances = np.array([], dtype=np.float32)
        mined_pair_ranks = np.array([], dtype=np.int32)
        region_pairs_list = []
        mined_pair_indices = np.empty((0, 2), dtype=np.int64)
    
    # Convert to tensors
    mined_pair_distances = torch.from_numpy(mined_pair_distances).float()
    mined_pair_ranks = torch.from_numpy(mined_pair_ranks.astype(np.int32))
    if region_pairs_list:
        region_pairs_np = np.concatenate(region_pairs_list, axis=0)
    else:
        region_pairs_np = np.empty((0, 2), dtype=np.int16)
    mined_pair_region_pairs = torch.from_numpy(region_pairs_np.astype(np.int16, copy=False))
    # mined_pair_indices.shape: (n_pairs_sampled, 2) dtype: int64
    # Local indices into train_sequences (0..n_seqs_to_use-1) for each pair
    # Note: For subsequence pairs, the second index is n_train_seqs + duplicate_idx
    # mined_pairs.shape: (n_pairs_sampled, 2, MAX_MODEL_SEQ_LEN, 3) or (n_pairs_sampled, 2, 4**K)
    # mined_pair_distances.shape: (n_pairs_sampled,)
    # mined_pair_ranks.shape: (n_pairs_sampled,)
    
    # Extract bucket indices (reorder to match [regular, subsequence] output order)
    if n_regular_sampled > 0 and n_subseq_sampled > 0:
        regular_buckets = bucket_assignments[sampled_local_indices[regular_mask]]
        subseq_buckets = bucket_assignments[sampled_local_indices[subseq_mask]]
        mined_pair_buckets = np.concatenate([regular_buckets, subseq_buckets])
    else:
        mined_pair_buckets = bucket_assignments[sampled_local_indices]
    mined_pair_buckets = torch.from_numpy(mined_pair_buckets).long()
    # mined_pair_buckets.shape: (n_pairs_sampled,)
    if gb.VERBOSE_MINING_TIMING:
        print(f"     > Time taken to construct output tensors: {time.time() - part_start_time:.4f} seconds")



    part_start_time = time.time()
    # Prepare data for logging functions
    # The logging functions expect pool_ranks to be computable from valid_flat_indices and pairwise_ranks.
    # Since our pool now includes subsequence pairs, we need to create synthetic structures.
    if verbose or log:
        n_total_seqs_for_logging = n_train_seqs + n_subseq_pairs
        
        # Build combined regions array for logging
        if n_subseq_pairs > 0 and duplicate_regions is not None:
            combined_regions = np.concatenate([train_regions, duplicate_regions])
        else:
            combined_regions = train_regions

        # Create synthetic pairwise_ranks for logging (n_total x n_total matrix)
        pairwise_ranks_for_logging = np.full((n_total_seqs_for_logging, n_total_seqs_for_logging), -2, dtype=np.int8)
        pairwise_ranks_for_logging[:n_train_seqs, :n_train_seqs] = pairwise_ranks
        
        # Add subsequence pairs at (orig_idx, combined_idx) positions
        if n_subseq_pairs > 0:
            for dup_idx in range(n_subseq_pairs):
                orig_idx = duplicate_indices[dup_idx]
                combined_idx = n_train_seqs + dup_idx
                pairwise_ranks_for_logging[orig_idx, combined_idx] = 7
                pairwise_ranks_for_logging[combined_idx, orig_idx] = 7
        
        # Convert regular flat indices from (n_train_seqs x n_train_seqs) space to (n_total x n_total) space
        if n_regular_pairs > 0:
            orig_seq_i = valid_flat_indices // n_train_seqs
            orig_seq_j = valid_flat_indices % n_train_seqs
            valid_flat_indices_converted = orig_seq_i * n_total_seqs_for_logging + orig_seq_j
        else:
            valid_flat_indices_converted = np.array([], dtype=np.int64)
        
        # Compute subsequence flat indices in the (n_total x n_total) space
        # Each subsequence pair (dup_idx) corresponds to position (orig_idx, n_train_seqs + dup_idx)
        if n_subseq_pairs > 0:
            subseq_flat_indices = np.array([
                duplicate_indices[i] * n_total_seqs_for_logging + (n_train_seqs + i)
                for i in range(n_subseq_pairs)
            ], dtype=np.int64)
        else:
            subseq_flat_indices = np.array([], dtype=np.int64)
        
        valid_flat_indices_for_logging = np.concatenate([valid_flat_indices_converted, subseq_flat_indices])
    if gb.VERBOSE_MINING_TIMING:
        print(f"     > Time taken to prepare data for logging: {time.time() - part_start_time:.4f} seconds")


    if gb.VERBOSE_MINING_TIMING:
        print(f"     > Time taken to mine pairs: {time.time() - func_start_time:.4f} seconds")

    # Print mining stats
    if verbose:
        print_pair_mining_stats(hardness_errors, signed_errors, bucket_assignments, sampled_local_indices, valid_flat_indices_for_logging, percentile_thresholds, flat_true_distances,
            flat_pred_distances, combined_regions, n_total_seqs_for_logging, pairwise_ranks_for_logging, per_rank_stats=per_rank_stats)

    # Log mining stats to files
    if log:
        if batch_num is not None and logs_dir is not None:
            write_pair_mining_log(batch_num, logs_dir, hardness_errors, signed_errors, bucket_assignments, sampled_local_indices, valid_flat_indices_for_logging,
                percentile_thresholds, flat_true_distances, flat_pred_distances, combined_regions, n_total_seqs_for_logging, pairwise_ranks_for_logging, pair_distances_df=pair_distances_df, per_rank_stats=per_rank_stats, pair_error_metrics_df=pair_error_metrics_df)

    return mined_pairs, mined_pair_distances, mined_pair_ranks, mined_pair_buckets, mined_pair_region_pairs, mined_pair_indices
