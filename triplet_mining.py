"""
This script contains functions for online mining of **triplets** from a Micro16S dataset.

The primary function is mine_triplets(), which is only ever called by triplet_pair_mining.py.
Please see that file for more context and details.


Per-Rank Triplet Mining:
    Triplets are mined using **separate per-rank** percentile-bucket sampling to remove bias 
    from ranks with large volumes of triplets. Each rank gets:
    - Its own representative set (up to TRIPLET_MINING_REPRESENTATIVE_SET_SIZES[rank] candidates)
    - Its own bucket boundaries computed from that representative set
    - A budget proportional to its EMA-smoothed hardness metric
    
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
        - gb.TRIPLET_MINING_EMA_WEIGHT_EXPONENT controls how strongly hardness differences are amplified

Structure:
    - Anchor (A)
    - Positive (P)
    - Negative (N)

For triplets at rank r (where r is an index in gb.TRIPLET_RANKS):
    - Anchor and Positive share the same taxon at rank r (pairwise_ranks[A, P] >= r)
      e.g. same domain if r=0, same phylum if r=1 (may also share deeper taxonomy)
    - Anchor and Negative share rank r-1 (pairwise_ranks[A, N] == r-1)
      e.g. different domains if r=0, different phyla if r=1

Upper Triangle Filtering (Duplicate Removal):
    - The pairwise rank matrix is symmetric: (A, P) at rank r == (P, A) at rank r.
    - AP pairs (anchor, positive) and (positive, anchor) represent the same relationship.
    - To avoid oversampling the same AP relationship in a batch, we keep one orientation only.
    - The kept orientation is chosen randomly per mining run: anchor < positive OR anchor > positive.
    - This halves the AP candidate pool while retaining all unique relationships.
    - Each unique AP pair is expanded into K triplets with independently chosen negatives
      (K = gb.TRIPLET_NEGATIVES_PER_AP, default 1).

Negative Sampling:
    For each (A, P) candidate, K negatives are sampled from valid negatives at rank r-1.
    Sampling is done with replacement (independent draws).
    Two modes are supported, selected at runtime:
    
    Uniform (default):
        - Each valid negative is equally likely.
        - Active when USE_TRIPLET_MINING_NEGATIVE_SELECTION_BIAS is False or beta_eff <= 0.
    
    Softmax-Biased (optional):
        - Closer (harder) negatives are sampled with higher probability.
        - For candidate set C = {n : pairwise_rank(A, n) = r-1}:
            z_n  = -beta_eff * d_pred(A, n)
            p(n) = softmax(z)  =  exp(z_n - max(z)) / sum_m exp(z_m - max(z))
        - beta_eff = beta_cfg * (1 - triplet_warmup_phase), so bias ramps in after warmup.
        - beta_eff = 0 gives uniform; larger beta_eff gives stronger hard-negative preference.
        - Falls back to uniform when candidate count < 2 or softmax weights collapse.
        - Controlled by:
            gb.USE_TRIPLET_MINING_NEGATIVE_SELECTION_BIAS (bool)
            gb.TRIPLET_MINING_NEGATIVE_SELECTION_BIAS_BETA (float >= 0)
            gb.TRIPLET_MINING_NEGATIVE_SELECTION_BIAS_EPS (float > 0, denominator guard)

Positive Selection Bias (optional):
    Before negative sampling, AP candidates can be re-sampled within each (anchor, triplet_rank)
    group to upweight harder positives (larger predicted anchor-positive distance).
    For positive candidates P_a,r = {p : pairwise_rank(a, p) >= r} already present in the
    representative AP set for that group:
        z_p  = +beta_eff * d_pred(a, p)
        p(p) = softmax(z)
    - beta_eff = beta_cfg * (1 - triplet_warmup_phase), so bias ramps in after warmup.
    - beta_eff = 0 gives no bias; larger beta_eff gives stronger hard-positive preference.
    - Sampling is with replacement and preserves AP pool size.
    - Falls back to uniform within-group sampling when weights are degenerate.
    - Controlled by:
        gb.USE_TRIPLET_MINING_POSITIVE_SELECTION_BIAS (bool)
        gb.TRIPLET_MINING_POSITIVE_SELECTION_BIAS_BETA (float >= 0)
        gb.TRIPLET_MINING_POSITIVE_SELECTION_BIAS_EPS (float > 0, denominator guard)

Metric (Normalised Violation):
    - margin = (true_dist(A, N) - true_dist(A, P)) * gb.TRIPLET_MARGIN_EPSILON  (or manual fixed margin)
    - violation = pred_dist(A, P) - pred_dist(A, N) + margin
    - normalised_violation = violation / (true_dist(A, N) - true_dist(A, P))

Per-Rank Percentile Bucket Sampling:
    - For each rank, triplets are placed into percentile buckets defined by gb.TRIPLET_MINING_BUCKETS
    - Bucket boundaries are computed separately for each rank from its representative set
    - Triplets are sampled from each bucket according to the sampling proportion
    - The 'any' bucket (gap=None) allows random sampling from the rank's entire pool
    - When a bucket has insufficient triplets, borrows from adjacent buckets

Zero-Loss Filtering (when gb.FILTER_ZERO_LOSS_TRIPLETS is True):
    - Triplets with violation <= 0 already satisfy the margin constraint and produce zero loss.
    - These are filtered out (per rank) to focus mining on triplets that provide a training signal.
    - Per-rank minimum after filtering: MIN_TRIPLETS_AFTER_FILTER_PER_RANK_BATCH_SIZE_FACTOR * per_rank_budget
    - If fewer than the minimum remain, the hardest non-hard triplets are added back to top up the pool.

"""

# Imports
import time
import numpy as np
import torch
from numba import njit, prange

# Local Imports
import globals_config as gb
from logging_utils import print_triplet_mining_stats, write_triplet_mining_log, log_bucket_thresholds

# Re-use a global generator so we don't re-seed every call
_RNG = np.random.default_rng()
# Rank metadata helpers (domain->genus for triplets)
_TRIPLET_RANK_VALUE_ORDER = [0, 1, 2, 3, 4, 5]
_TRIPLET_RANK_LABELS = {
    0: "domain",
    1: "phylum",
    2: "class",
    3: "order",
    4: "family",
    5: "genus",
}
# Number of triplet ranks
_N_TRIPLET_RANKS = 6


def is_rank_enabled_for_batch(rank_idx, batch_num=None):
    """
    Return True if a triplet rank is introduced for the current batch.
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


def get_effective_triplet_ranks(batch_num=None):
    """
    Combine TRIPLET_RANKS with INTRODUCE_RANK_AT_BATCHES for the current batch.
    """
    effective = [False] * _N_TRIPLET_RANKS
    if gb.TRIPLET_RANKS is None:
        return effective

    for rank_idx in range(min(len(gb.TRIPLET_RANKS), _N_TRIPLET_RANKS)):
        if gb.TRIPLET_RANKS[rank_idx] and is_rank_enabled_for_batch(rank_idx, batch_num=batch_num):
            effective[rank_idx] = True
    return effective


def _build_triplet_representative_phylum_step_stats(phylum_codes, representative_ap_array):
    """
    Build phylum step stats for triplet representative-set subsampling.
    """
    if phylum_codes is None:
        return None

    phylum_codes = np.asarray(phylum_codes, dtype=np.int32)
    if phylum_codes.ndim != 1 or len(phylum_codes) == 0:
        return None
    if np.min(phylum_codes) < 0:
        return None

    n_phyla = int(np.max(phylum_codes)) + 1
    if n_phyla <= 0:
        return None

    seq_counts = np.bincount(phylum_codes, minlength=n_phyla).astype(np.int64, copy=False)
    endpoint_counts = np.zeros(n_phyla, dtype=np.int64)

    representative_ap_array = np.asarray(representative_ap_array, dtype=np.int64)
    if representative_ap_array.size > 0 and representative_ap_array.shape[1] >= 3:
        # For phylum logging, only include triplets mined at the phylum rank.
        representative_ap_array = representative_ap_array[representative_ap_array[:, 2] == 1]
    n_active_pairs = int(len(representative_ap_array))
    if representative_ap_array.size > 0:
        anchors = representative_ap_array[:, 0]
        positives = representative_ap_array[:, 1]
        if np.max(anchors) >= len(phylum_codes) or np.max(positives) >= len(phylum_codes):
            return None
        endpoint_phyla = np.concatenate([phylum_codes[anchors], phylum_codes[positives]])
        endpoint_counts += np.bincount(endpoint_phyla, minlength=n_phyla).astype(np.int64, copy=False)

    return {
        "step": "post_triplet_representative_subsample",
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
        ema_hardness: np.ndarray of shape (6,) - EMA hardness values per rank
        total_budget: int - Total number of triplets to allocate
        cap_max: float - Maximum proportion any single rank can receive (0, 1]
        enabled_ranks: list/array of bool (length 6) - Which ranks are enabled
        weight_exponent: float - Exponent applied to hardness before normalising (>=0)
        cap_min: float - Minimum proportion any single enabled rank must receive [0, 1]
    
    Returns:
        budgets: np.ndarray of shape (6,) dtype int64 - Integer allocations per rank
        proportions: np.ndarray of shape (6,) dtype float64 - Continuous proportions per rank
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
        print(f"WARNING: TRIPLET_PER_RANK_BATCH_PROPORTION_MIN ({cap_min}) * n_enabled_ranks ({n_enabled}) > 1.0. "
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
            rank_label = _TRIPLET_RANK_LABELS.get(idx, str(idx))
            print(f"WARNING: Relaxing per-rank cap for triplet rank '{rank_label}' to absorb surplus batch budget.")
    
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


def _update_triplet_ema_hardness(per_rank_metrics, alpha):
    """
    Update the global triplet mining EMA hardness buffer.
    
    EMA update formula: ema = alpha * new_value + (1 - alpha) * ema
    
    Args:
        per_rank_metrics: np.ndarray of shape (6,) - Hardness metric per rank
            (0 to 1 range). NaN values indicate no samples for that rank (EMA unchanged).
        alpha: float - EMA smoothing factor in (0, 1]
    """
    if gb.TRIPLET_MINING_EMA_HARDNESS is None:
        return
    
    for i in range(_N_TRIPLET_RANKS):
        value = per_rank_metrics[i]
        if not np.isnan(value) and value >= 0:
            gb.TRIPLET_MINING_EMA_HARDNESS[i] = (
                alpha * value + 
                (1 - alpha) * gb.TRIPLET_MINING_EMA_HARDNESS[i]
            )
        else:
            # No observations for this rank this batch: decay toward zero so budgets
            # naturally migrate away from empty ranks while keeping long-term history.
            gb.TRIPLET_MINING_EMA_HARDNESS[i] *= (1 - alpha)


@njit(parallel=True, cache=True)
def _build_neighbor_data_numba(pairwise_ranks, needed_ranks_mask):
    """
    Build neighbor data structures in a single pass through the pairwise_ranks matrix.
    
    This replaces the slow Python loop that called np.flatnonzero for each (anchor, rank) pair.
    Uses two passes: first to count neighbors, then to fill the flat array.
    The counting pass is parallelized across rows for efficiency.
    
    Args:
        pairwise_ranks: np.ndarray (n_seqs, n_seqs), dtype int8
            2D pairwise ranks matrix.
        needed_ranks_mask: np.ndarray (9,), dtype bool
            Boolean mask where needed_ranks_mask[rank+1] = True if we need neighbors at that rank.
            Maps ranks -1 to 7 -> indices 0 to 8.
    
    Returns:
        neighbor_counts: np.ndarray (n_seqs, 9), dtype int32
            Count of neighbors for each (anchor, rank). Indexed as [anchor, rank+1].
        neighbor_starts: np.ndarray (n_seqs, 9), dtype int64
            Start offset in neighbors_flat for each (anchor, rank). Indexed as [anchor, rank+1].
        neighbors_flat: np.ndarray (total_neighbors,), dtype int64
            Flat array of all neighbor indices, concatenated across all (anchor, rank) pairs.
    """
    n_seqs = pairwise_ranks.shape[0]
    
    # First pass: count neighbors per (anchor, rank) - parallel by row
    neighbor_counts = np.zeros((n_seqs, 9), dtype=np.int32)
    
    for i in prange(n_seqs):
        for j in range(n_seqs):
            rank = pairwise_ranks[i, j]
            if rank < -1 or rank > 7:
                continue
            rank_idx = rank + 1  # Map -1..7 to 0..8
            if needed_ranks_mask[rank_idx]:
                neighbor_counts[i, rank_idx] += 1
    
    # Compute per-(anchor, rank) start offsets using prefix sum
    # This is sequential but O(n_seqs * 9) which is very fast
    neighbor_starts = np.empty((n_seqs, 9), dtype=np.int64)
    offset = np.int64(0)
    for i in range(n_seqs):
        for k in range(9):
            neighbor_starts[i, k] = offset
            offset += neighbor_counts[i, k]
    
    total = offset
    
    # Second pass: fill neighbor indices - parallel by row
    # Each row writes to its own non-overlapping segment of neighbors_flat
    neighbors_flat = np.empty(total, dtype=np.int64)
    
    for i in prange(n_seqs):
        # Local write positions for this row (thread-local in prange)
        write_pos = np.empty(9, dtype=np.int64)
        for k in range(9):
            write_pos[k] = neighbor_starts[i, k]
        
        for j in range(n_seqs):
            rank = pairwise_ranks[i, j]
            if rank < -1 or rank > 7:
                continue
            rank_idx = rank + 1
            if needed_ranks_mask[rank_idx]:
                neighbors_flat[write_pos[rank_idx]] = j
                write_pos[rank_idx] += 1
    
    return neighbor_counts, neighbor_starts, neighbors_flat


@njit(parallel=True, cache=True)
def _collect_ap_from_neighbors(neighbor_counts, neighbor_starts, neighbors_flat, 
                               enabled_ranks_lookup, use_upper):
    """
    Collect valid (anchor, positive, triplet_rank) tuples by iterating neighbor lists.
    
    Reuses neighbor data from Step 3 instead of rescanning the pairwise_ranks matrix.
    This is O(neighbors) instead of O(n²), providing significant speedup.
    
    Also applies the upper/lower triangle filter inline to avoid a separate filtering pass.
    
    Positive condition: pairwise_rank >= triplet_rank (same taxon at triplet rank level).
    This allows anchor-positive pairs that share the same taxon even when only one child
    taxon exists at the rank below (e.g. a phylum with a single class).
    
    Args:
        neighbor_counts: np.ndarray (n_seqs, 9), dtype int32
            Count of neighbors for each (anchor, rank). Indexed as [anchor, pw_rank+1].
        neighbor_starts: np.ndarray (n_seqs, 9), dtype int64
            Start offset in neighbors_flat for each (anchor, rank).
        neighbors_flat: np.ndarray (total_neighbors,), dtype int64
            Flat array of all neighbor indices.
        enabled_ranks_lookup: np.ndarray (8,), dtype bool
            enabled_ranks_lookup[rank] = True if rank is an enabled triplet rank (0-7).
        use_upper: bool
            If True, keep only anchor < positive; if False, keep only anchor > positive.
    
    Returns:
        result: np.ndarray (n_valid, 3), dtype int64
            Columns: [anchor, positive, triplet_rank]
    """
    n_seqs = neighbor_counts.shape[0]
    
    # First pass: count valid pairs per anchor (parallel)
    row_counts = np.zeros(n_seqs, dtype=np.int64)
    
    for anchor in prange(n_seqs):
        count = 0
        # Iterate over triplet ranks 0-5 (domain to genus)
        for triplet_rank in range(6):
            if not enabled_ranks_lookup[triplet_rank]:
                continue
            
            # Negatives are at pairwise_rank = triplet_rank - 1
            # Index mapping: pw_rank -> pw_rank + 1, so neg_idx = triplet_rank
            neg_idx = triplet_rank
            if neighbor_counts[anchor, neg_idx] == 0:
                continue  # No negatives available for this triplet rank
            
            # Positives are at pairwise_rank >= triplet_rank (indices triplet_rank+1 to 8)
            for pos_idx in range(triplet_rank + 1, 9):
                pos_start = neighbor_starts[anchor, pos_idx]
                pos_count = neighbor_counts[anchor, pos_idx]
                
                # Count positives that pass the triangle filter
                for k in range(pos_count):
                    positive = neighbors_flat[pos_start + k]
                    if use_upper:
                        if anchor < positive:
                            count += 1
                    else:
                        if anchor > positive:
                            count += 1
        
        row_counts[anchor] = count
    
    # Compute prefix sum to get row offsets (sequential - O(n_seqs), very fast)
    total = np.int64(0)
    row_offsets = np.empty(n_seqs + 1, dtype=np.int64)
    for i in range(n_seqs):
        row_offsets[i] = total
        total += row_counts[i]
    row_offsets[n_seqs] = total
    
    # Allocate result array
    result = np.empty((total, 3), dtype=np.int64)
    
    # Second pass: fill results (parallel)
    for anchor in prange(n_seqs):
        idx = row_offsets[anchor]
        
        for triplet_rank in range(6):
            if not enabled_ranks_lookup[triplet_rank]:
                continue
            
            # Check negatives exist at pairwise_rank = triplet_rank - 1
            neg_idx = triplet_rank
            if neighbor_counts[anchor, neg_idx] == 0:
                continue
            
            # Positives at pairwise_rank >= triplet_rank (indices triplet_rank+1 to 8)
            for pos_idx in range(triplet_rank + 1, 9):
                pos_start = neighbor_starts[anchor, pos_idx]
                pos_count = neighbor_counts[anchor, pos_idx]
                
                # Emit (anchor, positive, triplet_rank) for positives passing filter
                for k in range(pos_count):
                    positive = neighbors_flat[pos_start + k]
                    if use_upper:
                        if anchor < positive:
                            result[idx, 0] = anchor
                            result[idx, 1] = positive
                            result[idx, 2] = triplet_rank
                            idx += 1
                    else:
                        if anchor > positive:
                            result[idx, 0] = anchor
                            result[idx, 1] = positive
                            result[idx, 2] = triplet_rank
                            idx += 1
    
    return result


@njit(parallel=True, cache=True)
def _sample_negatives_parallel_kernel(anchors, neg_pairwise_ranks, 
                                      neighbor_starts, neighbor_counts, neighbors_flat):
    """
    Parallel numba kernel to sample one negative per (anchor, neg_pairwise_rank) pair.
    
    Uses 2D-indexed neighbor data structures for direct access.
    Random number generation is thread-safe in numba's prange.
    
    Args:
        anchors: np.ndarray of shape (n_processing,), dtype int64
            Anchor indices for each triplet candidate.
        neg_pairwise_ranks: np.ndarray of shape (n_processing,), dtype int64
            Pairwise ranks for negative sampling (triplet_rank - 1).
        neighbor_starts: np.ndarray of shape (n_seqs, 9), dtype int64
            Starting index in neighbors_flat for each (anchor, rank).
            Indexed as [anchor, pw_rank+1] to map ranks -1..7 to indices 0..8.
        neighbor_counts: np.ndarray of shape (n_seqs, 9), dtype int32
            Number of neighbors for each (anchor, rank).
            Indexed as [anchor, pw_rank+1] to map ranks -1..7 to indices 0..8.
        neighbors_flat: np.ndarray of shape (total_neighbors,), dtype int64
            Flattened array of all neighbor indices.
    
    Returns:
        negatives: np.ndarray of shape (n_processing,), dtype int64
            Sampled negative index for each triplet candidate.
    """
    n_processing = len(anchors)
    negatives = np.empty(n_processing, dtype=np.int64)
    
    for i in prange(n_processing):
        a = anchors[i]
        pw_rank = neg_pairwise_ranks[i]
        rank_idx = pw_rank + 1  # Map -1..7 to 0..8
        
        start = neighbor_starts[a, rank_idx]
        count = neighbor_counts[a, rank_idx]
        
        # Random index within the candidate range
        random_idx = np.random.randint(0, count)
        negatives[i] = neighbors_flat[start + random_idx]
    
    return negatives


def _sample_negatives_biased_kernel(anchors, neg_pairwise_ranks,
                                    neighbor_starts, neighbor_counts, neighbors_flat,
                                    sampling_distances, beta_eff, eps):
    """
    Sample one negative per (anchor, neg_pairwise_rank) pair using softmax-biased
    selection over predicted embedding distances.
    
    Closer (harder) negatives are sampled with higher probability, controlled by beta_eff.
    
    For each anchor's candidate negative set C = {n : pairwise_rank(anchor, n) = neg_pw_rank}:
        z_n  = -beta_eff * d_pred(anchor, n)      (closer negatives get higher logits)
        z'_n = z_n - max(z)                        (subtract max for numerical stability)
        w_n  = exp(z'_n)                           (softmax weights)
        p_n  = w_n / sum(w)                        (categorical probabilities)
        sample n ~ Categorical(p)                  (inverse-CDF sampling)
    
    Implementation detail:
        We group rows by (anchor, neg_pairwise_rank) and compute softmax weights once
        per unique group, then draw all samples for that group using vectorized inverse-CDF.
        This is mathematically equivalent to independently re-running the per-row softmax
        sampler, but avoids repeating identical work for duplicate groups.
    
    Falls back to uniform sampling when:
        - candidate count < 2 (no meaningful bias possible)
        - softmax weights are non-finite or sum(w) <= eps (degenerate numeric case)
    
    Args:
        anchors: np.ndarray (n_processing,) dtype int64
        neg_pairwise_ranks: np.ndarray (n_processing,) dtype int64
        neighbor_starts: np.ndarray (n_seqs, 9) dtype int64
        neighbor_counts: np.ndarray (n_seqs, 9) dtype int32
        neighbors_flat: np.ndarray (total_neighbors,) dtype int64
        sampling_distances: np.ndarray (n_seqs, n_seqs) dtype float32
            Predicted pairwise embedding distances between sequences.
        beta_eff: float
            Effective inverse-temperature. Must be > 0 (caller guarantees this).
        eps: float
            Guard for degenerate softmax (sum_w <= eps triggers uniform fallback).
    
    Returns:
        negatives: np.ndarray (n_processing,) dtype int64
    """
    n_processing = len(anchors)
    negatives = np.empty(n_processing, dtype=np.int64)

    if n_processing == 0:
        return negatives

    # Rank indexing: map pairwise ranks -1..7 to 0..8 for neighbor tensors.
    rank_idxs = neg_pairwise_ranks + 1
    starts = neighbor_starts[anchors, rank_idxs]
    counts = neighbor_counts[anchors, rank_idxs].astype(np.int64)

    # Group by (anchor, rank_idx) so repeated rows re-use one softmax build.
    # 9 ranks total -> unique key can be encoded as anchor * 9 + rank_idx.
    group_keys = anchors * 9 + rank_idxs
    order = np.argsort(group_keys, kind='mergesort')
    sorted_keys = group_keys[order]

    i = 0
    while i < n_processing:
        j = i + 1
        key = sorted_keys[i]
        while j < n_processing and sorted_keys[j] == key:
            j += 1

        batch_indices = order[i:j]
        m = j - i  # Number of draws needed for this (anchor, rank_idx) group.

        ref_idx = batch_indices[0]
        a = anchors[ref_idx]
        start = int(starts[ref_idx])
        count = int(counts[ref_idx])

        # With fewer than 2 candidates, bias is meaningless -> uniform.
        if count < 2:
            chosen_k = np.random.randint(0, count, size=m)
            negatives[batch_indices] = neighbors_flat[start + chosen_k]
            i = j
            continue

        candidates = neighbors_flat[start:start + count]

        # Compute stable softmax weights over candidates for this anchor/rank.
        logits = -beta_eff * sampling_distances[a, candidates]
        max_logit = np.max(logits)
        weights = np.exp(logits - max_logit)
        sum_w = weights.sum(dtype=np.float64)

        # Degenerate fallback for numeric issues.
        if (not np.isfinite(sum_w)) or (sum_w <= eps):
            chosen_k = np.random.randint(0, count, size=m)
            negatives[batch_indices] = neighbors_flat[start + chosen_k]
            i = j
            continue

        # Draw all samples for this group from one CDF.
        cdf = np.cumsum(weights, dtype=np.float64)
        u = np.random.random(size=m) * sum_w
        chosen_k = np.searchsorted(cdf, u, side='left')
        chosen_k = np.minimum(chosen_k, count - 1)  # Floating-point safety.

        negatives[batch_indices] = candidates[chosen_k]
        i = j

    return negatives


def _sample_positives_biased_kernel(anchors, positives, triplet_ranks_arr,
                                    sampling_distances, beta_eff, eps):
    """
    Re-sample one positive per AP row using softmax-biased probabilities within each
    (anchor, triplet_rank) group.

    Harder positives (farther from the anchor in embedding space) are sampled with
    higher probability, controlled by beta_eff.

    For a group G(a, r) with candidate positives P:
        z_p  = +beta_eff * d_pred(a, p)      (farther positives get higher logits)
        z'_p = z_p - max(z)                  (numerical stability)
        w_p  = exp(z'_p)                     (softmax weights)
        p_p  = w_p / sum(w)
        sample p ~ Categorical(p)

    The number of sampled rows per group is preserved (sampling with replacement), so
    the AP pool size remains unchanged.

    Args:
        anchors: np.ndarray (n_processing,) dtype int64
        positives: np.ndarray (n_processing,) dtype int64
        triplet_ranks_arr: np.ndarray (n_processing,) dtype int64
        sampling_distances: np.ndarray (n_seqs, n_seqs) dtype float32
            Predicted pairwise embedding distances between sequences.
        beta_eff: float
            Effective inverse-temperature. Must be > 0 (caller guarantees this).
        eps: float
            Guard for degenerate softmax (sum_w <= eps triggers uniform fallback).

    Returns:
        sampled_positives: np.ndarray (n_processing,) dtype int64
    """
    n_processing = len(anchors)
    sampled_positives = np.empty(n_processing, dtype=np.int64)

    if n_processing == 0:
        return sampled_positives

    # Group by (anchor, triplet_rank) so repeated rows share one softmax build.
    group_keys = anchors * _N_TRIPLET_RANKS + triplet_ranks_arr
    order = np.argsort(group_keys, kind='mergesort')
    sorted_keys = group_keys[order]

    i = 0
    while i < n_processing:
        j = i + 1
        key = sorted_keys[i]
        while j < n_processing and sorted_keys[j] == key:
            j += 1

        batch_indices = order[i:j]
        m = j - i  # Number of draws for this (anchor, triplet_rank) group.

        ref_idx = batch_indices[0]
        a = anchors[ref_idx]
        candidates = positives[batch_indices]
        count = len(candidates)

        # With fewer than 2 candidates, bias is meaningless -> keep existing positives.
        if count < 2:
            sampled_positives[batch_indices] = candidates
            i = j
            continue

        # Compute stable softmax weights over positives for this anchor/rank group.
        logits = beta_eff * sampling_distances[a, candidates]
        max_logit = np.max(logits)
        weights = np.exp(logits - max_logit)
        sum_w = weights.sum(dtype=np.float64)

        # Degenerate fallback for numeric issues.
        if (not np.isfinite(sum_w)) or (sum_w <= eps):
            chosen_k = np.random.randint(0, count, size=m)
            sampled_positives[batch_indices] = candidates[chosen_k]
            i = j
            continue

        # Draw all samples for this group from one CDF.
        cdf = np.cumsum(weights, dtype=np.float64)
        u = np.random.random(size=m) * sum_w
        chosen_k = np.searchsorted(cdf, u, side='left')
        chosen_k = np.minimum(chosen_k, count - 1)  # Floating-point safety.

        sampled_positives[batch_indices] = candidates[chosen_k]
        i = j

    return sampled_positives


@njit(parallel=True, cache=True)
def _compute_triplet_distances_kernel(anchors, positives, negatives, adjusted_distances, seq_embeddings_distances):
    """
    Parallel numba kernel to compute true and predicted distances for triplets.
    
    Replaces advanced indexing (which can be slow) with a single parallel pass.
    """
    n = anchors.shape[0]
    
    # We output float32 for consistency and speed in downstream ops
    true_ap = np.empty(n, dtype=np.float32)
    true_an = np.empty(n, dtype=np.float32)
    pred_ap = np.empty(n, dtype=np.float32)
    pred_an = np.empty(n, dtype=np.float32)
    
    for i in prange(n):
        a = anchors[i]
        p = positives[i]
        n_idx = negatives[i]
        
        # Access distances
        true_ap[i] = adjusted_distances[a, p]
        true_an[i] = adjusted_distances[a, n_idx]
        pred_ap[i] = seq_embeddings_distances[a, p]
        pred_an[i] = seq_embeddings_distances[a, n_idx]
        
    return true_ap, true_an, pred_ap, pred_an


@njit(parallel=True, cache=True)
def _group_ap_by_rank_parallel(all_ap_array):
    """
    Group AP pairs by rank using parallel chunk-based processing.
    
    Two-pass algorithm that parallelizes over chunks to avoid race conditions:
    1. Count pairs per (chunk, rank) - each chunk is independent
    2. Compute per-chunk write offsets from counts
    3. Fill grouped index array - each chunk writes to non-overlapping segments
    
    Args:
        all_ap_array: np.ndarray (n_pairs, 3) with columns [anchor, positive, triplet_rank]
    
    Returns:
        counts_per_rank: np.ndarray (6,) - total count per rank
        rank_starts: np.ndarray (6,) - start offset in indices_flat for each rank
        indices_flat: np.ndarray (n_valid_pairs,) - row indices grouped by rank
    """
    n_pairs = all_ap_array.shape[0]
    n_ranks = 6
    
    if n_pairs == 0:
        return (np.zeros(n_ranks, dtype=np.int64), 
                np.zeros(n_ranks, dtype=np.int64), 
                np.empty(0, dtype=np.int64))
    
    # Determine chunk count for parallel processing (balance parallelism vs overhead)
    n_chunks = min(64, max(1, n_pairs // 10000))
    chunk_size = (n_pairs + n_chunks - 1) // n_chunks
    
    # Pass 1: Count per (chunk, rank) in parallel
    # Each chunk writes to its own row, so no race conditions
    chunk_counts = np.zeros((n_chunks, n_ranks), dtype=np.int64)
    
    for chunk_idx in prange(n_chunks):
        start = chunk_idx * chunk_size
        end = min(start + chunk_size, n_pairs)
        for i in range(start, end):
            rank = all_ap_array[i, 2]
            if 0 <= rank < n_ranks:
                chunk_counts[chunk_idx, rank] += 1
    
    # Sum chunk counts to get totals per rank (sequential, O(n_chunks * n_ranks) = small)
    counts_per_rank = np.zeros(n_ranks, dtype=np.int64)
    for c in range(n_chunks):
        for r in range(n_ranks):
            counts_per_rank[r] += chunk_counts[c, r]
    
    # Compute rank start offsets (sequential, O(n_ranks) = 6)
    rank_starts = np.zeros(n_ranks, dtype=np.int64)
    offset = np.int64(0)
    for r in range(n_ranks):
        rank_starts[r] = offset
        offset += counts_per_rank[r]
    
    total = offset
    if total == 0:
        return counts_per_rank, rank_starts, np.empty(0, dtype=np.int64)
    
    # Compute per-chunk write offsets for each rank
    # chunk_offsets[c, r] = where chunk c starts writing for rank r
    chunk_offsets = np.zeros((n_chunks, n_ranks), dtype=np.int64)
    for r in range(n_ranks):
        running_offset = rank_starts[r]
        for c in range(n_chunks):
            chunk_offsets[c, r] = running_offset
            running_offset += chunk_counts[c, r]
    
    # Pass 2: Fill indices in parallel (each chunk writes to non-overlapping segments)
    indices_flat = np.empty(total, dtype=np.int64)
    
    for chunk_idx in prange(n_chunks):
        start = chunk_idx * chunk_size
        end = min(start + chunk_size, n_pairs)
        
        # Local write positions for this chunk (thread-local in prange)
        write_pos = np.empty(n_ranks, dtype=np.int64)
        for r in range(n_ranks):
            write_pos[r] = chunk_offsets[chunk_idx, r]
        
        for i in range(start, end):
            rank = all_ap_array[i, 2]
            if 0 <= rank < n_ranks:
                indices_flat[write_pos[rank]] = i
                write_pos[rank] += 1
    
    return counts_per_rank, rank_starts, indices_flat


def _compute_ap_representative_weights(all_ap_array, row_indices, taxon_counts, baseline_counts,
                                        rank_idx, lam, eps, weight_clip):
    """
    Compute per-candidate taxon-size-balancing weights for AP representative-set subsampling.
    
    Weight formula:  w(a,p,r) = ((b_r^2) / (c_a * c_p + eps)) ^ lambda
    where:
        c_a, c_p = taxon counts for anchor and positive at triplet rank r
        b_r      = baseline taxon count at rank r
        lambda   = balancing strength (0=uniform, 1=full counteraction)
    
    Since anchor and positive share the same taxon at rank r, c_a and c_p are typically
    equal, making the weight effectively ((b_r / c_a)^2)^lambda = (b_r / c_a)^(2*lambda).
    
    Args:
        all_ap_array: np.ndarray of shape (n_total_ap, 3) - [anchor, positive, triplet_rank]
        row_indices: np.ndarray of shape (n_candidates,) - indices into all_ap_array for this rank
        taxon_counts: np.ndarray of shape (n_seqs, 7), dtype int32 or float32
            Per-sequence taxon sizes (float32 when corrected for domain-based downsampling).
        baseline_counts: np.ndarray of shape (7,) - per-rank baseline taxon sizes
        rank_idx: int - triplet rank (0=domain, ..., 5=genus)
        lam: float - lambda in [0, 1]
        eps: float - epsilon for numerical safety
        weight_clip: float or None - optional cap for extreme weights
    
    Returns:
        weights: np.ndarray of shape (n_candidates,) dtype float64 - normalised sampling probabilities
    """
    n = len(row_indices)
    
    # Get anchor and positive indices
    anchors = all_ap_array[row_indices, 0]
    positives = all_ap_array[row_indices, 1]
    
    # Get taxon counts for both sequences at this rank
    c_a = taxon_counts[anchors, rank_idx].astype(np.float64)
    c_p = taxon_counts[positives, rank_idx].astype(np.float64)
    
    # Baseline count for this rank
    b_r = float(baseline_counts[rank_idx])
    
    # Compute raw weights: ((b_r^2) / (c_a * c_p + eps)) ^ lambda
    numerator = b_r * b_r
    denominator = c_a * c_p + eps
    ratio = numerator / denominator
    
    # Clip ratio before exponentiation to avoid overflow
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


def _get_valid_ap_indices_per_rank(all_ap_array, rank_lookup_array, max_samples_per_rank, seed,
                                    taxon_counts=None, shortage_recorder=None):
    """
    Subsample AP pairs per rank to representative set sizes using parallel grouping.
    
    Uses a parallel two-pass algorithm to group pairs by rank, then subsamples
    each rank using numpy's optimized random.choice. When representative taxon-size
    balancing is enabled (gb.USE_REPRESENTATIVE_TAXON_SIZE_BALANCING and lambda > 0),
    subsampling uses weighted sampling without replacement. Otherwise falls back to
    uniform sampling.
    
    Args:
        all_ap_array: np.ndarray of shape (n_pairs, 3) with columns [anchor, positive, triplet_rank]
        rank_lookup_array: np.ndarray of shape (6,), dtype bool - which triplet ranks are enabled
        max_samples_per_rank: np.ndarray of shape (6,), dtype int64 - max samples per rank
        seed: int for reproducibility
        taxon_counts: np.ndarray of shape (n_seqs, 7), dtype int32 or float32, or None
            Per-sequence taxon sizes. May be float32 when corrected for domain-based
            downsampling. Required when representative taxon-size balancing is enabled.
        shortage_recorder: list or None - accumulates shortage warnings
    
    Returns:
        subsampled_ap_array: np.ndarray of shape (n_subsampled, 3)
        total_valid_per_rank: np.ndarray of shape (6,) - total count before subsampling
    """
    # Group pairs by rank in parallel
    counts_per_rank, rank_starts, indices_flat = _group_ap_by_rank_parallel(all_ap_array)
    
    total_valid_per_rank = counts_per_rank.copy()
    
    if counts_per_rank.sum() == 0:
        return np.empty((0, 3), dtype=np.int64), total_valid_per_rank
    
    # Determine if weighted sampling should be used
    use_weighted = (
        gb.USE_REPRESENTATIVE_TAXON_SIZE_BALANCING
        and gb.TRIPLET_REPRESENTATIVE_TAXON_SIZE_BALANCE_LAMBDA > 0
        and taxon_counts is not None
    )
    lam = float(gb.TRIPLET_REPRESENTATIVE_TAXON_SIZE_BALANCE_LAMBDA) if use_weighted else 0.0
    eps = float(gb.REPRESENTATIVE_TAXON_SIZE_BALANCE_EPS) if use_weighted else 1e-12
    weight_clip = gb.REPRESENTATIVE_TAXON_SIZE_BALANCE_WEIGHT_CLIP if use_weighted else None
    baseline_counts = gb.TRAIN_TAXON_BASELINE_COUNT_PER_RANK if use_weighted else None
    
    # Subsample each rank using numpy's optimized random.choice
    rng = np.random.default_rng(seed)
    subsampled_indices_list = []
    
    for rank_idx in range(_N_TRIPLET_RANKS):
        if not rank_lookup_array[rank_idx]:
            continue

        total_for_rank = int(counts_per_rank[rank_idx])
        cap_for_rank = int(max_samples_per_rank[rank_idx])
        
        # Emit warning if fewer candidates than requested
        if cap_for_rank > 0 and total_for_rank < cap_for_rank:
            rank_label = _TRIPLET_RANK_LABELS.get(rank_idx, str(rank_idx))
            print(f"WARNING: Triplet mining rank '{rank_label}' has {total_for_rank} AP candidates, "
                  f"fewer than representative set size {cap_for_rank}.")
            if shortage_recorder is not None:
                shortage_recorder.append({
                    'rank_name': rank_label,
                    'requested': cap_for_rank,
                    'available': total_for_rank,
                })
        
        if total_for_rank == 0:
            continue
        
        # Get slice of indices_flat for this rank
        start = int(rank_starts[rank_idx])
        end = start + total_for_rank
        rank_indices = indices_flat[start:end]
        
        # Subsample if needed
        if cap_for_rank > 0 and total_for_rank > cap_for_rank:
            # Use weighted sampling when enabled and taxon counts cover this rank (0-5 maps to taxon indices 0-5)
            if use_weighted and rank_idx <= 6 and baseline_counts is not None:
                weights = _compute_ap_representative_weights(
                    all_ap_array, rank_indices, taxon_counts, baseline_counts,
                    rank_idx, lam, eps, weight_clip
                )
                chosen_local = rng.choice(total_for_rank, size=cap_for_rank, replace=False, p=weights)
            else:
                chosen_local = rng.choice(total_for_rank, size=cap_for_rank, replace=False)
            chosen_local.sort()  # Preserve cache locality for downstream gathers
            rank_indices = rank_indices[chosen_local]
        
        subsampled_indices_list.append(rank_indices.copy())
    
    if not subsampled_indices_list:
        return np.empty((0, 3), dtype=np.int64), total_valid_per_rank
    
    # Concatenate all selected indices
    all_indices = np.concatenate(subsampled_indices_list)
    
    # Single sort for cache locality
    all_indices.sort()
    
    # Single gather operation
    subsampled_ap_array = all_ap_array[all_indices]
    
    return subsampled_ap_array, total_valid_per_rank


def _compute_per_rank_bucket_thresholds(normalised_violations, pool_ranks, percentile_gaps):
    """
    Compute bucket violation thresholds separately for each rank.
    
    Args:
        normalised_violations: np.ndarray of shape (n_pool,) - normalised violations for pool
        pool_ranks: np.ndarray of shape (n_pool,) - triplet ranks (0 to 5) for pool
        percentile_gaps: list of floats - bucket percentile gaps (e.g. [0.25, 0.25, 0.25, 0.25])
    
    Returns:
        per_rank_thresholds: dict mapping triplet_rank -> list of threshold values
            Each list has len(percentile_gaps) - 1 thresholds
    """
    # Compute cumulative percentiles from gaps
    cumulative_percentiles = np.cumsum(percentile_gaps)[:-1]  # Exclude the last (100%)
    
    per_rank_thresholds = {}
    
    for rank_idx in range(_N_TRIPLET_RANKS):
        rank_mask = (pool_ranks == rank_idx)
        
        if not np.any(rank_mask):
            per_rank_thresholds[rank_idx] = []
            continue
        
        rank_violations = normalised_violations[rank_mask]
        
        if len(rank_violations) == 0:
            per_rank_thresholds[rank_idx] = []
            continue
        
        # Compute threshold values at each percentile
        thresholds = np.quantile(rank_violations, cumulative_percentiles).tolist()
        per_rank_thresholds[rank_idx] = thresholds
    
    return per_rank_thresholds


@njit(parallel=True, cache=True)
def _assign_buckets_per_rank_kernel(normalised_violations, pool_ranks, thresholds_array, n_thresholds_per_rank):
    """
    Numba kernel to assign each triplet to its bucket based on per-rank thresholds.
    
    Parallelizes over the pool to assign bucket indices efficiently.
    
    Args:
        normalised_violations: np.ndarray of shape (n_pool,) dtype float64
        pool_ranks: np.ndarray of shape (n_pool,) dtype int32 - triplet ranks (0 to 5)
        thresholds_array: np.ndarray of shape (6, max_thresholds) dtype float64 - thresholds per rank
        n_thresholds_per_rank: np.ndarray of shape (6,) dtype int32 - number of valid thresholds per rank
    
    Returns:
        bucket_assignments: np.ndarray of shape (n_pool,) dtype int32
    """
    n_pool = len(normalised_violations)
    bucket_assignments = np.zeros(n_pool, dtype=np.int32)
    
    for i in prange(n_pool):
        rank = pool_ranks[i]
        
        if rank < 0 or rank >= 6:
            continue
        
        violation = normalised_violations[i]
        n_thresholds = n_thresholds_per_rank[rank]
        
        # Find bucket: bucket 0 if below first threshold, bucket k if >= threshold k-1
        bucket = 0
        for t_idx in range(n_thresholds):
            if violation >= thresholds_array[rank, t_idx]:
                bucket = t_idx + 1
        
        bucket_assignments[i] = bucket
    
    return bucket_assignments


def _assign_buckets_per_rank(normalised_violations, pool_ranks, per_rank_thresholds):
    """
    Assign each triplet to its bucket based on per-rank thresholds.
    
    Uses a Numba-accelerated parallel kernel for performance.
    
    Args:
        normalised_violations: np.ndarray of shape (n_pool,)
        pool_ranks: np.ndarray of shape (n_pool,) - triplet ranks (0 to 5)
        per_rank_thresholds: dict from _compute_per_rank_bucket_thresholds
    
    Returns:
        bucket_assignments: np.ndarray of shape (n_pool,) dtype int32
    """
    n_pool = len(normalised_violations)
    
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
    thresholds_array = np.zeros((6, max_thresholds), dtype=np.float64)
    n_thresholds_per_rank = np.zeros(6, dtype=np.int32)
    
    for rank_idx, thresholds in per_rank_thresholds.items():
        if 0 <= rank_idx < 6:
            n_thresholds_per_rank[rank_idx] = len(thresholds)
            for t_idx, t_val in enumerate(thresholds):
                thresholds_array[rank_idx, t_idx] = t_val
    
    # Ensure correct dtypes for Numba kernel
    if normalised_violations.dtype != np.float64:
        normalised_violations = normalised_violations.astype(np.float64)
    if pool_ranks.dtype != np.int32:
        pool_ranks = pool_ranks.astype(np.int32)
    
    return _assign_buckets_per_rank_kernel(
        normalised_violations,
        pool_ranks,
        thresholds_array,
        n_thresholds_per_rank
    )


def _sample_from_rank_buckets(n_to_sample, rank_mask, bucket_assignments, normalised_violations,
                              n_buckets, target_proportions, available_mask, triplet_rank):
    """
    Sample triplets from a single rank's buckets with overflow handling.
    
    When a bucket has insufficient triplets, borrows from adjacent buckets:
    - Right-most bucket first, borrow from left
    - Left-most bucket can borrow from 'any' (all remaining)
    
    Args:
        n_to_sample: int - target number of triplets for this rank
        rank_mask: np.ndarray bool - mask for triplets of this rank
        bucket_assignments: np.ndarray int32 - bucket assignment for each triplet
        normalised_violations: np.ndarray - for sorting when borrowing
        n_buckets: int - number of regular buckets (excluding 'any')
        target_proportions: np.ndarray - target proportion for each bucket + 'any'
        available_mask: np.ndarray bool - which triplets are still available (modified in-place)
        triplet_rank: int - triplet rank id (0..5) for warning messages
    
    Returns:
        sampled_indices: list of int - indices into the pool
        deficit: int - how many triplets short of target
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
    
    rank_label = _TRIPLET_RANK_LABELS.get(triplet_rank, str(triplet_rank))
    
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
        
        # Get available triplets in this bucket
        rank_available = rank_mask & available_mask
        bucket_mask = (bucket_assignments == b) & rank_available
        bucket_indices = np.where(bucket_mask)[0]
        n_available = len(bucket_indices)
        
        if n_available >= target:
            # Enough triplets - sample randomly
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
                print(f"WARNING: Triplet bucket {b} for rank '{rank_label}' only has "
                      f"{n_available}/{target} triplets. Borrowing from easier buckets.")
                left_bucket = b - 1
                # Keep shifting left until deficit is cleared or we run out of buckets
                while deficit > 0 and left_bucket >= 0:
                    rank_available = rank_mask & available_mask
                    left_mask = (bucket_assignments == left_bucket) & rank_available
                    left_indices = np.where(left_mask)[0]
                    n_left = len(left_indices)
                    
                    if n_left > 0:
                        borrow_count = min(deficit, n_left)
                        # Sort by violation (descending) to get hardest from this bucket
                        sorted_idx = np.argsort(-normalised_violations[left_indices])
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
                    print(f"WARNING: Triplet bucket {b} for rank '{rank_label}' still short by "
                          f"{deficit} triplets after borrowing. Marking shortfall for 'any' bucket.")
                    # Push remaining deficit into 'any' bucket so it can attempt to fill it
                    bucket_targets[-1] += deficit
                    diag_entries[-1]['target'] += int(deficit)
    
    # Handle 'any' bucket - sample from remaining available triplets of this rank
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


def mine_triplets(train_sequences, n_triplets_to_mine, seq_embeddings_distances, adjusted_distances, pairwise_ranks, taxon_counts_subsampled=None, warmup_phase=0.0, verbose=False, log=False, batch_num=None, logs_dir=None, triplet_satisfaction_df=None, triplet_error_metrics_df=None, phylum_codes_for_logging=None, phylum_step_stats_out=None):
    """
    Mine triplets using per-rank percentile-bucket sampling based on normalised violation.

    This function implements separate mining per taxonomic rank to remove bias from ranks
    with large volumes of triplets. Each rank gets its own bucket boundaries computed from
    its representative set, and the batch budget is divided across ranks according to
    EMA-smoothed hardness metrics.

    Algorithm:
    1. Determine enabled ranks and prepare lookup structures
    2. Compute per-rank batch budgets using EMA hardness with capped proportions
    3. Build neighbor lists for valid positives and negatives at each rank
    4. Gather per-rank representative AP sets (up to TRIPLET_MINING_REPRESENTATIVE_SET_SIZES[rank])
    5. Optionally apply positive selection bias to AP candidates, then expand by K negatives per AP
       and compute normalised violations
    6. Update EMA hardness buffers with per-rank mean violations (BEFORE filtering)
    7. (Optional) Filter to triplets with positive violation, per rank
    8. Compute per-rank bucket thresholds from each rank's pool
    9. Sample triplets from each rank according to its budget
    10. Construct output tensors

    For triplets at rank r (where r is an index in gb.TRIPLET_RANKS):
        - Anchor and Positive share the same taxon at rank r (pairwise_ranks[anchor, positive] >= r)
        - Anchor and Negative share rank r-1 (pairwise_ranks[anchor, negative] = r-1)
    
    Hardness Metric (Normalised Violation):
        - margin: (true_an - true_ap) * gb.TRIPLET_MARGIN_EPSILON (or manual fixed margin)
        - violation = pred_ap - pred_an + margin
        - normalised_violation = violation / (true_an - true_ap)
    
    Zero-Loss Filtering (when gb.FILTER_ZERO_LOSS_TRIPLETS is True):
        - Triplets with violation <= 0 already satisfy the margin constraint and produce zero loss.
        - These are filtered out to focus mining on triplets that provide a training signal.
        - Per-rank minimum: MIN_TRIPLETS_AFTER_FILTER_PER_RANK_BATCH_SIZE_FACTOR * per_rank_budget
        - If fewer remain, the hardest non-hard triplets (sorted by normalised_violation desc) are added back.

    Args:
        train_sequences: torch.Tensor (N_TRAIN_SEQS, MAX_MODEL_SEQ_LEN, 3)
        n_triplets_to_mine: int
        seq_embeddings_distances: np.ndarray (N_TRAIN_SEQS, N_TRAIN_SEQS)
        adjusted_distances: np.ndarray (N_TRAIN_SEQS, N_TRAIN_SEQS)
        pairwise_ranks: np.ndarray (N_TRAIN_SEQS, N_TRAIN_SEQS)
            -2=Ignore, -1=Domain, 0=Phylum, ...
        taxon_counts_subsampled: np.ndarray (N_TRAIN_SEQS, 7) or None
            Per-sequence taxon sizes for the subsampled training set, used for
            representative-set taxon-size balancing when enabled. May be float32 when
            counts have been corrected for domain-based downsampling (see
            mine_pairs_and_triplets() in triplet_pair_mining.py).
        train_phylum_labels: Optional np.ndarray/list of shape (N_TRAIN_SEQS,)
            Phylum labels aligned to local train indices (used only for taxonomy logging diagnostics).
        train_class_labels: Optional np.ndarray/list of shape (N_TRAIN_SEQS,)
            Class labels aligned to local train indices (used only for taxonomy logging diagnostics).
        warmup_phase: float in [0,1] - proportion of sampling weight to route through the 'any' bucket (0=normal buckets, 1=uniform).
        verbose: bool
        log: bool
        batch_num: int
        logs_dir: str
        triplet_satisfaction_df: Optional pd.DataFrame for tracking triplet satisfaction over time.
                                 If provided, will be populated with satisfaction metrics during logging.
        triplet_error_metrics_df: Optional pd.DataFrame for tracking per-rank hardness metrics over time.
                                  If provided, will be populated with the composite metric
                                  (hard_triplet_prop * TRIPLET_EMA_HARD_WEIGHT + moderate_triplet_prop * TRIPLET_EMA_MODERATE_WEIGHT).
        phylum_codes_for_logging: np.ndarray of shape (N_TRAIN_SEQS,) or None
            Optional local phylum code per sequence. When provided with phylum_step_stats_out, this function
            appends representative-set phylum diagnostics for triplet mining.
        phylum_step_stats_out: list or None
            Optional list to append phylum step dictionaries into.

    Returns:
        mined_triplets: torch.Tensor (n_triplets, 3, MAX_MODEL_SEQ_LEN, 3) - [anchor, positive, negative]
        mined_triplet_margins: torch.Tensor (n_triplets,)
        mined_triplet_ranks: torch.Tensor (n_triplets,) - The triplet rank (0-5)
        mined_triplet_buckets: torch.Tensor (n_triplets,)

    Global variables:
        gb.TRIPLET_RANKS: Boolean list for which ranks (0-5) to mine triplets at
        gb.TRIPLET_MINING_BUCKETS: List of tuples (percentile_gap, sampling_proportion)
        gb.TRIPLET_MINING_REPRESENTATIVE_SET_SIZES: List of ints (length 6) - max candidates per rank
        gb.TRIPLET_MARGIN_EPSILON: Factor for dynamic margin computation
        gb.MANUAL_TRIPLET_MARGINS: Whether to use manual fixed margins
        gb.MANUAL_TRIPLET_MARGINS_PER_RANK: Dict mapping rank -> fixed margin value
        gb.FILTER_ZERO_LOSS_TRIPLETS: Whether to filter out triplets that would produce zero loss
        gb.MIN_TRIPLETS_AFTER_FILTER_PER_RANK_BATCH_SIZE_FACTOR: Factor for per-rank min triplets after filter
        gb.TRIPLET_MINING_EMA_ALPHA: EMA smoothing factor for hardness metrics
        gb.TRIPLET_PER_RANK_BATCH_PROPORTION_MAX: Max proportion of batch any single rank can receive
        gb.USE_TRIPLET_MINING_POSITIVE_SELECTION_BIAS: Whether to use softmax-biased positive re-sampling
        gb.TRIPLET_MINING_POSITIVE_SELECTION_BIAS_BETA: Inverse-temperature for positive softmax bias (>= 0)
        gb.TRIPLET_MINING_POSITIVE_SELECTION_BIAS_EPS: Denominator guard for degenerate softmax
        gb.USE_TRIPLET_MINING_NEGATIVE_SELECTION_BIAS: Whether to use softmax-biased negative sampling
        gb.TRIPLET_MINING_NEGATIVE_SELECTION_BIAS_BETA: Inverse-temperature for softmax bias (>= 0)
        gb.TRIPLET_MINING_NEGATIVE_SELECTION_BIAS_EPS: Denominator guard for degenerate softmax
        gb.TRIPLET_NEGATIVES_PER_AP: Number of negatives sampled per (anchor, positive) candidate (>= 1)
        NOTE: bias scores are computed from predicted embedding distances (seq_embeddings_distances).
    """
    
    func_start_time = time.time()
    
    # Initialize per-rank mining stats for logging (will be populated throughout the function)
    per_rank_stats = {
        'ema_hardness_pre': None,        # EMA values before this batch's update
        'ema_hardness_budget': None,     # EMA values after warmup blending (used for budgets)
        'ema_hardness_post': None,       # EMA values after this batch's update
        'hardness_metrics': np.full(_N_TRIPLET_RANKS, np.nan, dtype=np.float64),
        'budgets': None,                 # Requested budget per rank
        'proportions': None,             # Budget proportions per rank
        'sampled_counts': np.zeros(_N_TRIPLET_RANKS, dtype=np.int64),  # Actual sampled per rank
        'pool_counts': np.zeros(_N_TRIPLET_RANKS, dtype=np.int64),     # Pool size per rank (after filter)
        'pool_counts_pre_filter': np.zeros(_N_TRIPLET_RANKS, dtype=np.int64),  # Pool size per rank (before filter)
        'deficits': np.zeros(_N_TRIPLET_RANKS, dtype=np.int64),        # Deficit per rank
        'per_rank_thresholds': {},       # Bucket thresholds per rank
        'warmup_phase': warmup_phase,    # Current warmup phase
        'representative_shortages': [],  # Representative set shortfalls for logging
        'filter_hard_counts': np.zeros(_N_TRIPLET_RANKS, dtype=np.int64),   # Violating triplets surviving zero-loss filter
        'filter_topup_counts': np.zeros(_N_TRIPLET_RANKS, dtype=np.int64),  # Non-violating triplets used for top-up
        'filter_minimums': np.zeros(_N_TRIPLET_RANKS, dtype=np.int64),      # Target min per rank after filter
        'bucket_proportions_base': None,     # Raw bucket proportions from config
        'bucket_proportions_target': None,   # Post-warmup blended bucket proportions
        'bucket_diagnostics': [None] * _N_TRIPLET_RANKS,  # Per-rank bucket fulfillment stats
    }

    
    n_seqs = train_sequences.shape[0]
    
    # Helper for empty returns
    def _empty_triplet_return(reason=None):
        warning_msg = "WARNING: Empty triplet tensors being returned by mine_triplets()."
        if reason:
            warning_msg += f" Reason: {reason}."
        print(warning_msg)
        empty_triplets = torch.empty((0, 3, train_sequences.shape[1], train_sequences.shape[2]), dtype=train_sequences.dtype)
        empty_margins = torch.empty((0,), dtype=torch.float32)
        empty_ranks = torch.empty((0,), dtype=torch.int32)
        empty_buckets = torch.empty((0,), dtype=torch.long)
        empty_indices = np.empty((0, 3), dtype=np.int64)
        if gb.VERBOSE_MINING_TIMING:
            suffix = f" ({reason})" if reason else ""
            print(f"     > Time taken to mine triplets{suffix}: {time.time() - func_start_time:.4f} seconds")
        return empty_triplets, empty_margins, empty_ranks, empty_buckets, empty_indices
    
    part_start_time = time.time()
    # 1. Get enabled ranks and build lookup -------------------------------------------------
    # Effective triplet ranks combine static TRIPLET_RANKS with
    # INTRODUCE_RANK_AT_BATCHES for the current batch.
    rank_lookup_array = np.array(get_effective_triplet_ranks(batch_num=batch_num), dtype=np.bool_)
    enabled_ranks = [r for r in range(_N_TRIPLET_RANKS) if rank_lookup_array[r]]

    if not enabled_ranks or n_triplets_to_mine <= 0:
        return _empty_triplet_return("No enabled triplet ranks for this batch or zero triplets requested")
    
    if gb.VERBOSE_MINING_TIMING:
        print(f"     > Time taken to get enabled ranks: {time.time() - part_start_time:.4f} seconds")

    # 2. Compute per-rank batch budgets using EMA hardness -------------------------------------------------
    part_start_time = time.time()
    
    # Get EMA hardness values (or use equal weights if not initialized)
    if gb.TRIPLET_MINING_EMA_HARDNESS is not None:
        ema_hardness = gb.TRIPLET_MINING_EMA_HARDNESS.copy()
        per_rank_stats['ema_hardness_pre'] = ema_hardness.copy()  # Capture pre-update EMA
    else:
        ema_hardness = np.ones(_N_TRIPLET_RANKS, dtype=np.float64)
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
        n_triplets_to_mine, 
        gb.TRIPLET_PER_RANK_BATCH_PROPORTION_MAX,
        rank_lookup_array,
        gb.TRIPLET_MINING_EMA_WEIGHT_EXPONENT,
        cap_min=gb.TRIPLET_PER_RANK_BATCH_PROPORTION_MIN
    )
    
    # Store budgets and proportions for logging
    per_rank_stats['budgets'] = per_rank_budgets.copy()
    per_rank_stats['proportions'] = per_rank_proportions.copy()
    
    if gb.VERBOSE_MINING_TIMING:
        print(f"     > Time taken to compute per-rank budgets: {time.time() - part_start_time:.4f} seconds")
        budget_str = ", ".join([f"{_TRIPLET_RANK_LABELS[r]}={per_rank_budgets[r]}" 
                                for r in range(_N_TRIPLET_RANKS) if rank_lookup_array[r]])
        print(f"     > Per-rank budgets: {budget_str}")

    part_start_time = time.time()
    # 3. Build neighbor lists per rank -------------------------------------------------
    # For each anchor and each triplet rank, store valid positive and negative indices
    # Positives: pairwise_ranks[anchor, pos] >= triplet_rank (same taxon at triplet rank level)
    # Negatives: pairwise_ranks[anchor, neg] == triplet_rank - 1
    #
    # Note on rank semantics for triplets:
    #   - triplet_rank r means A-P have pairwise_rank >= r, A-N have pairwise_rank = r-1
    #   - e.g. triplet_rank=1 (phylum): A-P same phylum (pairwise_rank>=1), A-N diff phyla (pairwise_rank=0)
    
    # Collect needed pairwise_ranks values for building neighbor lists
    needed_pairwise_ranks = set()
    for r in enabled_ranks:
        # Positive pairwise_ranks: all ranks >= r (same taxon at triplet rank level)
        for pr in range(r, 8):
            needed_pairwise_ranks.add(pr)
        needed_pairwise_ranks.add(r - 1)  # negative pairwise_rank
    if gb.VERBOSE_MINING_TIMING:
        print(f"     > Time taken to collect needed pairwise ranks: {time.time() - part_start_time:.4f} seconds")
    
    part_start_time = time.time()
    # Build neighbor data using numba kernel (single pass through matrix)
    # This replaces the slow Python loop that called np.flatnonzero for each (anchor, rank) pair.
    needed_ranks_mask = np.zeros(9, dtype=np.bool_)  # 9 for ranks -1 to 7, indexed as 0 to 8
    for pw_rank in needed_pairwise_ranks:
        if -1 <= pw_rank <= 7:
            needed_ranks_mask[pw_rank + 1] = True
    
    neighbor_counts, neighbor_starts, neighbors_flat = _build_neighbor_data_numba(pairwise_ranks, needed_ranks_mask)
    # neighbor_counts.shape: (n_seqs, 9) - count per (anchor, rank), indexed as [anchor, rank+1]
    # neighbor_starts.shape: (n_seqs, 9) - start offset per (anchor, rank)
    # neighbors_flat.shape: (total_neighbors,) - flat array of all neighbor indices
    
    if gb.VERBOSE_MINING_TIMING:
        print(f"     > Time taken to build neighbor lists per rank: {time.time() - part_start_time:.4f} seconds")

    part_start_time = time.time()
    # 4. Collect valid (anchor, positive) pairs with their triplet ranks -------------------------------------------------
    # Reuses neighbor data from Step 3 instead of rescanning the pairwise_ranks matrix.
    # This is O(neighbors) instead of O(n²), providing significant speedup.
    # Also applies the upper/lower triangle filter inline to avoid a separate filtering pass.
    
    # Keep only one AP orientation per run to avoid duplicate (A,P)/(P,A) pairs.
    # We flip this randomly each run so mining stays fair over time:
    #   - True  -> keep anchor < positive
    #   - False -> keep anchor > positive
    use_upper = bool(_RNG.integers(0, 2))
    
    # Build enabled_ranks_lookup: True if that rank is an enabled triplet rank
    enabled_ranks_lookup = np.zeros(8, dtype=np.bool_)  # 8 for ranks 0-7
    for r in enabled_ranks:
        if 0 <= r <= 7:
            enabled_ranks_lookup[r] = True
    
    # Call optimized kernel that iterates neighbor lists (O(neighbors) instead of O(n²))
    all_ap_array = _collect_ap_from_neighbors(
        neighbor_counts, neighbor_starts, neighbors_flat,
        enabled_ranks_lookup, use_upper
    )
    # all_ap_array.shape: (n_total_ap_pairs, 3) -> (anchor, positive, triplet_rank)
    
    if len(all_ap_array) == 0:
        return _empty_triplet_return("No valid (anchor, positive) pairs available")
    
    if gb.VERBOSE_MINING_TIMING:
        print(f"     > Time taken to collect valid (anchor, positive) pairs with their triplet ranks: {time.time() - part_start_time:.4f} seconds")

    part_start_time = time.time()
    # 5. Subsample to per-rank representative set sizes -------------------------------------------------
    # Get per-rank representative set sizes
    max_samples_per_rank = np.array(gb.TRIPLET_MINING_REPRESENTATIVE_SET_SIZES, dtype=np.int64)
    
    # Get seed for reproducibility
    current_seed = _RNG.integers(0, 2**31 - 1)
    
    # Subsample each rank to its representative set size
    representative_shortages = []
    all_ap_array, total_valid_per_rank = _get_valid_ap_indices_per_rank(
        all_ap_array, rank_lookup_array, max_samples_per_rank, current_seed,
        taxon_counts=taxon_counts_subsampled,
        shortage_recorder=representative_shortages
    )
    per_rank_stats['representative_shortages'] = representative_shortages
    
    # Store total available counts before subsampling for logging
    # (This is what we could have sampled, not what we actually sampled)
    per_rank_stats['available_counts'] = total_valid_per_rank.copy()

    if phylum_step_stats_out is not None:
        rep_step_stats = _build_triplet_representative_phylum_step_stats(
            phylum_codes=phylum_codes_for_logging,
            representative_ap_array=all_ap_array,
        )
        if rep_step_stats is not None:
            phylum_step_stats_out.append(rep_step_stats)

    if len(all_ap_array) == 0:
        return _empty_triplet_return("Representative set sampling removed all (anchor, positive) pairs")
    
    # Log representative-set taxon-size balancing status
    rep_balance_active = (
        gb.USE_REPRESENTATIVE_TAXON_SIZE_BALANCING
        and gb.TRIPLET_REPRESENTATIVE_TAXON_SIZE_BALANCE_LAMBDA > 0
        and taxon_counts_subsampled is not None
    )
    per_rank_stats['rep_taxon_balance_active'] = rep_balance_active
    per_rank_stats['rep_taxon_balance_lambda'] = gb.TRIPLET_REPRESENTATIVE_TAXON_SIZE_BALANCE_LAMBDA if rep_balance_active else 0.0
    
    if gb.VERBOSE_MINING_TIMING:
        print(f"     > Time taken to subsample to representative set size: {time.time() - part_start_time:.4f} seconds")
        if rep_balance_active:
            clip_str = f", clip={gb.REPRESENTATIVE_TAXON_SIZE_BALANCE_WEIGHT_CLIP}" if gb.REPRESENTATIVE_TAXON_SIZE_BALANCE_WEIGHT_CLIP else ""
            print(f"     > Representative taxon-size balancing: active (lambda={gb.TRIPLET_REPRESENTATIVE_TAXON_SIZE_BALANCE_LAMBDA}{clip_str})")

    part_start_time = time.time()
    # 6. Optionally bias positive selection, then sample K negatives -------------------------------------------------
    # Positive bias (optional): within each (anchor, triplet_rank) group, positives are
    # softmax-sampled using +beta_eff * d_pred(a, p), so farther/harder positives are favored.
    # Negative bias (optional): for each AP candidate, negatives are softmax-sampled using
    # -beta_eff * d_pred(a, n), so closer/harder negatives are favored.
    # Both biases use beta_eff = beta_cfg * (1 - warmup_phase) to ramp in after warmup.
    n_base_ap = len(all_ap_array)
    n_negatives_per_ap = int(getattr(gb, "TRIPLET_NEGATIVES_PER_AP", 1) or 1)
    anchors_base = all_ap_array[:, 0].astype(np.int64)
    positives_base = all_ap_array[:, 1].astype(np.int64)
    triplet_ranks_base = all_ap_array[:, 2].astype(np.int64)
    neg_pairwise_ranks_base = triplet_ranks_base - 1

    warmup_val = 0.0 if warmup_phase is None else float(np.clip(warmup_phase, 0.0, 1.0))

    # Optionally bias positive selection toward larger predicted A-P distances.
    use_positive_bias = bool(getattr(gb, "USE_TRIPLET_MINING_POSITIVE_SELECTION_BIAS", False))
    pos_beta_cfg = float(getattr(gb, "TRIPLET_MINING_POSITIVE_SELECTION_BIAS_BETA", 0.0)) if use_positive_bias else 0.0
    pos_beta_eff = pos_beta_cfg * (1.0 - warmup_val)
    use_biased_positives = use_positive_bias and pos_beta_eff > 0
    if use_biased_positives:
        # Sync numpy RNG for reproducible sampling inside the kernel.
        current_seed = _RNG.integers(0, 2**31 - 1)
        np.random.seed(current_seed)
        positives_base = _sample_positives_biased_kernel(
            anchors_base,
            positives_base,
            triplet_ranks_base,
            seq_embeddings_distances,
            pos_beta_eff,
            float(getattr(gb, "TRIPLET_MINING_POSITIVE_SELECTION_BIAS_EPS", 1e-12)),
        )

    if n_negatives_per_ap > 1:
        # Vectorized expansion: each AP candidate contributes K independent negative draws.
        anchors = np.repeat(anchors_base, n_negatives_per_ap)
        positives = np.repeat(positives_base, n_negatives_per_ap)
        triplet_ranks_arr = np.repeat(triplet_ranks_base, n_negatives_per_ap)
        neg_pairwise_ranks = np.repeat(neg_pairwise_ranks_base, n_negatives_per_ap)
    else:
        anchors = anchors_base
        positives = positives_base
        triplet_ranks_arr = triplet_ranks_base
        neg_pairwise_ranks = neg_pairwise_ranks_base

    n_processing = len(anchors)
    
    # Sync numpy/numba RNG with global _RNG for negative sampling.
    current_seed = _RNG.integers(0, 2**31 - 1)
    np.random.seed(current_seed)
    
    # Compute effective beta for negative selection bias
    # beta_eff = beta_cfg * (1 - warmup_phase): near-uniform during warmup, full bias after
    beta_cfg = float(gb.TRIPLET_MINING_NEGATIVE_SELECTION_BIAS_BETA) if gb.USE_TRIPLET_MINING_NEGATIVE_SELECTION_BIAS else 0.0
    beta_eff = beta_cfg * (1.0 - warmup_val)
    use_biased_negatives = gb.USE_TRIPLET_MINING_NEGATIVE_SELECTION_BIAS and beta_eff > 0
    
    if use_biased_negatives:
        # Biased sampling: closer negatives in embedding space are sampled with higher probability
        negatives = _sample_negatives_biased_kernel(
            anchors, neg_pairwise_ranks,
            neighbor_starts, neighbor_counts, neighbors_flat,
            seq_embeddings_distances, beta_eff, float(gb.TRIPLET_MINING_NEGATIVE_SELECTION_BIAS_EPS)
        )
    else:
        # Uniform sampling: each valid negative is equally likely
        negatives = _sample_negatives_parallel_kernel(
            anchors, neg_pairwise_ranks,
            neighbor_starts, neighbor_counts, neighbors_flat
        )

    if gb.VERBOSE_MINING_TIMING:
        pos_bias_str = f"biased (beta_eff={pos_beta_eff:.4f})" if use_biased_positives else "uniform/original"
        bias_str = f"biased (beta_eff={beta_eff:.4f})" if use_biased_negatives else "uniform"
        print(
            f"     > Time taken for AP positive bias [{pos_bias_str}] and sampling "
            f"{n_negatives_per_ap} negative(s) per AP [{bias_str}]: {time.time() - part_start_time:.4f} seconds"
        )
        if n_negatives_per_ap > 1:
            print(f"     > Expanded AP candidate pool from {n_base_ap} to {n_processing} triplet candidates")
    
    part_start_time = time.time()
    # 7. Compute distances -------------------------------------------------
    # Use parallel numba kernel for fast distance lookup
    true_ap, true_an, pred_ap, pred_an = _compute_triplet_distances_kernel(
        anchors, positives, negatives, adjusted_distances, seq_embeddings_distances
    )

    if gb.VERBOSE_MINING_TIMING:
        print(f"     > Time taken to compute distances: {time.time() - part_start_time:.4f} seconds")
    
    part_start_time = time.time()
    # 8. Compute margins -------------------------------------------------
    delta_true = true_an - true_ap

    # There are a small number of triplets where true_an == true_ap, which for now we ignore.
    valid_mask = delta_true > 0
    
    # Check we have any valid triplets
    if not np.any(valid_mask):
        if verbose:
            print("Skipping triplet mining step: no valid triplets with true_an > true_ap.")
        return _empty_triplet_return("No valid triplets with true_an > true_ap")
    
    # Filter bad triplets
    if not np.all(valid_mask):
        n_filtered = np.count_nonzero(~valid_mask)
        if verbose:
            print(f"Filtered {n_filtered} triplets with true_an <= true_ap.")
        anchors = anchors[valid_mask]
        positives = positives[valid_mask]
        negatives = negatives[valid_mask]
        triplet_ranks_arr = triplet_ranks_arr[valid_mask]
        true_ap = true_ap[valid_mask]
        true_an = true_an[valid_mask]
        pred_ap = pred_ap[valid_mask]
        pred_an = pred_an[valid_mask]
        delta_true = delta_true[valid_mask]

    n_processing = len(anchors)
    
    if gb.MANUAL_TRIPLET_MARGINS:
        # Use fixed margins per triplet rank
        margins = np.array([gb.MANUAL_TRIPLET_MARGINS_PER_RANK[r] for r in triplet_ranks_arr], dtype=np.float32)
    else:
        # Dynamic margins: margin = (true_an - true_ap) * epsilon
        margins = delta_true * gb.TRIPLET_MARGIN_EPSILON
    if gb.VERBOSE_MINING_TIMING:
        print(f"     > Time taken to compute margins: {time.time() - part_start_time:.4f} seconds")
    
    part_start_time = time.time()
    # 9. Compute normalized violations (hardness metric) -------------------------------------------------
    # violation = pred_ap - pred_an + margin
    # normalised_violation = violation / delta_true
    violations = pred_ap - pred_an + margins
    normalised_violations = violations / delta_true
    # normalised_violations.shape: (n_processing,)

    if gb.VERBOSE_MINING_TIMING:
        print(f"     > Time taken to compute normalized violations: {time.time() - part_start_time:.4f} seconds")
    
    part_start_time = time.time()
    # 10. Compute per-rank hardness metrics and update EMA (BEFORE zero-loss filtering) -------------------------------------------------
    # The EMA reflects overall rank difficulty, so we compute it before filtering
    # Metric: hard_prop * HARD_WEIGHT + moderate_prop * MODERATE_WEIGHT
    per_rank_metrics = np.full(_N_TRIPLET_RANKS, np.nan, dtype=np.float64)
    
    # Pre-compute masks for hard and moderate triplets
    # Hard: AP >= AN  (pred_ap >= pred_an)
    # Moderate: AP < AN < AP + M  (pred_ap < pred_an AND violation > 0)
    is_hard = pred_ap >= pred_an
    is_moderate = (pred_ap < pred_an) & (violations > 0)
    
    for rank_idx in range(_N_TRIPLET_RANKS):
        rank_mask = (triplet_ranks_arr == rank_idx)
        if np.any(rank_mask):
            total_for_rank = np.sum(rank_mask)
            # Store pre-filter pool count for logging
            per_rank_stats['pool_counts_pre_filter'][rank_idx] = total_for_rank
            
            # Compute proportions
            n_hard = np.sum(is_hard & rank_mask)
            n_moderate = np.sum(is_moderate & rank_mask)
            
            prop_hard = n_hard / total_for_rank
            prop_moderate = n_moderate / total_for_rank
            
            # Compute metric
            metric = (prop_hard * gb.TRIPLET_EMA_HARD_WEIGHT) + (prop_moderate * gb.TRIPLET_EMA_MODERATE_WEIGHT)
            per_rank_metrics[rank_idx] = metric
            
    # Store the raw hardness metric feeding the EMA for logging
    per_rank_stats['hardness_metrics'] = per_rank_metrics.copy()
    
    # Update EMA with current batch's hardness metrics
    _update_triplet_ema_hardness(per_rank_metrics, gb.TRIPLET_MINING_EMA_ALPHA)
    
    # Capture post-update EMA for logging
    if gb.TRIPLET_MINING_EMA_HARDNESS is not None:
        per_rank_stats['ema_hardness_post'] = gb.TRIPLET_MINING_EMA_HARDNESS.copy()
    
    if gb.VERBOSE_MINING_TIMING:
        print(f"     > Time taken to update EMA hardness: {time.time() - part_start_time:.4f} seconds")
    
    part_start_time = time.time()
    # 11. Filter to triplets with positive violation (would produce non-zero loss) -------------------------------------------------
    # Triplets with violation <= 0 already satisfy the margin constraint, so they produce zero loss.
    # Filtering these out focuses mining on triplets that actually provide a training signal.
    
    # Save unfiltered pool data for logging/verbose printing
    if verbose or (log and batch_num is not None and logs_dir is not None):
        pool_triplet_ranks_arr = triplet_ranks_arr.copy()
        pool_true_ap = true_ap.copy()
        pool_true_an = true_an.copy()
        pool_pred_ap = pred_ap.copy()
        pool_pred_an = pred_an.copy()
        pool_margins = margins.copy()
        pool_normalised_violations = normalised_violations.copy()
    
    if gb.FILTER_ZERO_LOSS_TRIPLETS:
        # Identify triplets with positive violation (would produce loss)
        hard_mask = violations > 0
        
        # Compute per-rank minimums after filtering
        factor = gb.MIN_TRIPLETS_AFTER_FILTER_PER_RANK_BATCH_SIZE_FACTOR
        hard_counts_arr = per_rank_stats['filter_hard_counts']
        topup_counts_arr = per_rank_stats['filter_topup_counts']
        min_counts_arr = per_rank_stats['filter_minimums']
        
        # Process each rank separately for minimum pool maintenance
        for rank_idx in range(_N_TRIPLET_RANKS):
            if not rank_lookup_array[rank_idx]:
                continue
            
            rank_budget = int(per_rank_budgets[rank_idx])
            if rank_budget <= 0:
                continue
            
            min_triplets_for_rank = int(factor * rank_budget)
            min_counts_arr[rank_idx] = min_triplets_for_rank
            rank_mask = (triplet_ranks_arr == rank_idx)
            rank_hard_mask = hard_mask & rank_mask
            n_hard_for_rank = np.count_nonzero(rank_hard_mask)
            hard_counts_arr[rank_idx] = n_hard_for_rank
            fallback_count = 0
            
            # If we have fewer than minimum hard triplets, top up with hardest non-hard
            if n_hard_for_rank < min_triplets_for_rank:
                n_needed = min_triplets_for_rank - n_hard_for_rank
                rank_label = _TRIPLET_RANK_LABELS.get(rank_idx, str(rank_idx))
                
                # Get indices of non-hard triplets for this rank
                non_hard_rank_mask = ~hard_mask & rank_mask
                non_hard_indices = np.where(non_hard_rank_mask)[0]
                
                if len(non_hard_indices) > 0:
                    # Sort by normalised_violation descending (highest violation = hardest)
                    non_hard_sorted_indices = non_hard_indices[np.argsort(normalised_violations[non_hard_indices])[::-1]]
                    # Take the top n_needed
                    fallback_count = min(n_needed, len(non_hard_sorted_indices))
                    fallback_indices = non_hard_sorted_indices[:fallback_count]
                    # Mark these as included in the pool
                    hard_mask[fallback_indices] = True
                    
                    if fallback_count < n_needed:
                        print(f"WARNING: Triplet rank '{rank_label}' only has {n_hard_for_rank + fallback_count} "
                              f"triplets after zero-loss filter top-up (target: {min_triplets_for_rank}).")
                else:
                    print(f"WARNING: Triplet rank '{rank_label}' has {n_hard_for_rank} hard triplets, "
                          f"fewer than minimum {min_triplets_for_rank}. No non-hard triplets available to top up.")
            
            topup_counts_arr[rank_idx] = fallback_count
        
        # Apply the filter to all arrays
        keep_indices = np.where(hard_mask)[0]
        anchors = anchors[keep_indices]
        positives = positives[keep_indices]
        negatives = negatives[keep_indices]
        triplet_ranks_arr = triplet_ranks_arr[keep_indices]
        true_ap = true_ap[keep_indices]
        true_an = true_an[keep_indices]
        pred_ap = pred_ap[keep_indices]
        pred_an = pred_an[keep_indices]
        delta_true = delta_true[keep_indices]
        margins = margins[keep_indices]
        violations = violations[keep_indices]
        normalised_violations = normalised_violations[keep_indices]
        n_processing = len(anchors)

    if gb.VERBOSE_MINING_TIMING:
        print(f"     > Time taken to filter to triplets with positive violation: {time.time() - part_start_time:.4f} seconds")
    
    if n_processing == 0:
        return _empty_triplet_return("No triplets remaining after zero-loss filtering")
    
    part_start_time = time.time()
    # 12. Compute per-rank bucket thresholds -------------------------------------------------
    # Parse TRIPLET_MINING_BUCKETS to get gaps and proportions
    bucket_configs = [b for b in gb.TRIPLET_MINING_BUCKETS if b[0] is not None]
    any_bucket_config = [b for b in gb.TRIPLET_MINING_BUCKETS if b[0] is None]
    
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
        per_rank_thresholds = {r: [] for r in range(_N_TRIPLET_RANKS)}
        bucket_assignments = np.zeros(n_processing, dtype=np.int32)
        percentile_thresholds = []
    else:
        # Compute per-rank bucket thresholds from each rank's pool
        per_rank_thresholds = _compute_per_rank_bucket_thresholds(
            normalised_violations, triplet_ranks_arr, percentile_gaps
        )
        
        # Assign buckets per-rank
        bucket_assignments = _assign_buckets_per_rank(
            normalised_violations, triplet_ranks_arr, per_rank_thresholds
        )
        
        # For logging compatibility, compute global percentile thresholds
        cumulative = np.cumsum(percentile_gaps)
        percentile_thresholds = cumulative[:-1].tolist() if len(cumulative) > 1 else []
    
    # Store per-rank thresholds for logging
    per_rank_stats['per_rank_thresholds'] = per_rank_thresholds.copy()
    
    # Update pool counts prior to logging so bucket logs reflect the actual representative pool sizes
    pool_counts = per_rank_stats.get('pool_counts')
    if pool_counts is not None:
        pool_counts[:] = 0
        if n_processing > 0:
            rank_indices = triplet_ranks_arr.astype(np.int64, copy=False)
            valid_mask = (rank_indices >= 0) & (rank_indices < _N_TRIPLET_RANKS)
            if np.any(valid_mask):
                counts = np.bincount(rank_indices[valid_mask], minlength=_N_TRIPLET_RANKS)
                pool_counts[:_N_TRIPLET_RANKS] = counts[:_N_TRIPLET_RANKS]

    # Log bucket thresholds if enabled
    if log and getattr(gb, "LOG_MINING_BUCKET_THRESHOLDS", False) and batch_num is not None and logs_dir is not None:
         log_bucket_thresholds(
             batch_num=batch_num,
             logs_dir=logs_dir,
             mining_type="triplet",
             per_rank_thresholds=per_rank_thresholds,
             per_rank_stats=per_rank_stats,
             rank_map=_TRIPLET_RANK_LABELS,
             bucket_configs=gb.TRIPLET_MINING_BUCKETS
         )

    if gb.VERBOSE_MINING_TIMING:
        print(f"     > Time taken to compute per-rank bucket thresholds: {time.time() - part_start_time:.4f} seconds")
    
    part_start_time = time.time()
    # 13. Sample triplets from each rank according to per-rank budgets -------------------------------------------------
    available_mask = np.ones(n_processing, dtype=bool)
    sampled_local_indices = []
    total_deficit = 0
    
    # Sample from each enabled rank
    for rank_idx in range(_N_TRIPLET_RANKS):
        if not rank_lookup_array[rank_idx]:
            continue
        
        rank_budget = int(per_rank_budgets[rank_idx])
        
        if rank_budget <= 0:
            continue
        
        # Get mask for this rank
        rank_mask = (triplet_ranks_arr == rank_idx)
        rank_pool_count = np.sum(rank_mask)
        per_rank_stats['pool_counts'][rank_idx] = rank_pool_count
        
        if not np.any(rank_mask):
            total_deficit += rank_budget
            per_rank_stats['deficits'][rank_idx] = rank_budget
            rank_label = _TRIPLET_RANK_LABELS.get(rank_idx, str(rank_idx))
            print(f"WARNING: Triplet mining rank '{rank_label}' has no candidates in pool, "
                  f"budget of {rank_budget} triplets unfulfilled.")
            continue
        
        # Sample from this rank's buckets
        rank_sampled, rank_deficit, bucket_diag = _sample_from_rank_buckets(
            rank_budget, rank_mask, bucket_assignments, normalised_violations,
            n_buckets, target_proportions, available_mask, rank_idx
        )
        if bucket_diag is not None:
            per_rank_stats['bucket_diagnostics'][rank_idx] = bucket_diag
        
        sampled_local_indices.extend(rank_sampled)
        total_deficit += rank_deficit
        
        # Store per-rank stats for logging
        per_rank_stats['sampled_counts'][rank_idx] = len(rank_sampled)
        per_rank_stats['deficits'][rank_idx] = rank_deficit
        
        if rank_deficit > 0:
            rank_label = _TRIPLET_RANK_LABELS.get(rank_idx, str(rank_idx))
            print(f"WARNING: Triplet mining rank '{rank_label}' short by {rank_deficit} triplets "
                  f"(sampled {len(rank_sampled)}/{rank_budget}).")
    
    # Report overall deficit if any
    if total_deficit > 0:
        print(f"WARNING: Triplet mining total deficit: {total_deficit} triplets "
              f"(sampled {len(sampled_local_indices)}/{n_triplets_to_mine}).")

    # Convert to numpy array
    sampled_local_indices = np.array(sampled_local_indices, dtype=np.int64)
    
    # Handle case where we couldn't sample enough triplets
    n_triplets_sampled = len(sampled_local_indices)
    if n_triplets_sampled == 0:
        return _empty_triplet_return("Sampling step returned zero triplets")

    if gb.VERBOSE_MINING_TIMING:
        print(f"     > Time taken to sample triplets from each bucket: {time.time() - part_start_time:.4f} seconds")
    
    part_start_time = time.time()
    # 14. Construct output tensors -------------------------------------------------
    sampled_anchors = anchors[sampled_local_indices]
    sampled_positives = positives[sampled_local_indices]
    sampled_negatives = negatives[sampled_local_indices]
    sampled_ranks = triplet_ranks_arr[sampled_local_indices]
    sampled_margins = margins[sampled_local_indices]
    sampled_buckets = bucket_assignments[sampled_local_indices]
    
    # Extract sequence triplets: [anchor, positive, negative]
    anchor_seqs = train_sequences[sampled_anchors]
    positive_seqs = train_sequences[sampled_positives]
    negative_seqs = train_sequences[sampled_negatives]
    
    mined_triplets = torch.stack([anchor_seqs, positive_seqs, negative_seqs], dim=1)
    # mined_triplets.shape: (n_triplets_sampled, 3, MAX_MODEL_SEQ_LEN, 3)
    
    # Also prepare index triplets (local indices into train_sequences: 0..n_seqs_to_use-1)
    # These will be converted to global indices (0..n_train-1) by the caller
    mined_triplet_indices = np.stack([sampled_anchors, sampled_positives, sampled_negatives], axis=1)
    # mined_triplet_indices.shape: (n_triplets_sampled, 3) dtype: int64
    
    mined_triplet_margins = torch.from_numpy(sampled_margins.astype(np.float32))
    mined_triplet_ranks = torch.from_numpy(sampled_ranks.astype(np.int32))
    mined_triplet_buckets = torch.from_numpy(sampled_buckets.astype(np.int64))

    if gb.VERBOSE_MINING_TIMING:
        print(f"     > Time taken to construct output tensors: {time.time() - part_start_time:.4f} seconds")
    
    # Print mining stats
    if verbose:
        print_triplet_mining_stats(normalised_violations, bucket_assignments, sampled_local_indices,
                                   percentile_thresholds, triplet_ranks_arr, true_ap, true_an,
                                   pred_ap, pred_an, margins,
                                   pool_normalised_violations, pool_triplet_ranks_arr, pool_true_ap,
                                   pool_true_an, pool_pred_ap, pool_pred_an, pool_margins,
                                   per_rank_stats=per_rank_stats)
    
    # Log mining stats to files
    if log and batch_num is not None and logs_dir is not None:
        write_triplet_mining_log(batch_num, logs_dir, normalised_violations, bucket_assignments,
                                  sampled_local_indices, percentile_thresholds, triplet_ranks_arr,
                                  true_ap, true_an, pred_ap, pred_an, margins,
                                  pool_normalised_violations, pool_triplet_ranks_arr, pool_true_ap,
                                  pool_true_an, pool_pred_ap, pool_pred_an, pool_margins,
                                  triplet_satisfaction_df=triplet_satisfaction_df, per_rank_stats=per_rank_stats,
                                  triplet_error_metrics_df=triplet_error_metrics_df)

    if gb.VERBOSE_MINING_TIMING:
        print(f"     > Time taken to mine triplets: {time.time() - func_start_time:.4f} seconds")

    return mined_triplets, mined_triplet_margins, mined_triplet_ranks, mined_triplet_buckets, mined_triplet_indices
