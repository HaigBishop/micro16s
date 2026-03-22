"""
generate_seq_variants.py

This module provides utilities for generating variants of DNA sequences encoded in 
a the 3-bit format. This is for data augmentation during Micro16S training.

Variations:
- **Point Mutations:** Randomly mutate bases within the sequence at a specified rate, simulating natural variation.
- **End Truncation:** Randomly shorten sequences from either end to introduce length variability.
- **Shifting:** Shifts sequences randomly within the tensor adjusting the masking. This doesn't affect the DNA sequence itself.
- **Maximum Length Capping:** Optionally enforces a maximum sequence length (MAX_MODEL_SEQ_LEN).

3-bit Encoding Scheme:
- The last dimension of the input tensor represents the 3-bit encoding for each base:
    - 1st bit: Mask flag (0 = not masked, 1 = masked)
    - 2nd & 3rd bits: DNA base (00 = A, 01 = C, 10 = G, 11 = T)
- Input tensors may have arbitrary leading dimensions, but the last two must be [sequence_length, 3].

Functions:
- gen_seq_variants: Generate variants of all provided sequences.
    - The shape of the tensor is preserved unless we use max_len
    - Input Shape: (N_REGIONS + 1, N_SEQS, MAX_SEQ_LEN, 3)
    - Output Shape: (N_REGIONS + 1, N_SEQS, MAX_SEQ_LEN, 3)
- select_regions: Select a single region for all provided sequences.
    - The specified region dimension is collapsed to 1
    - Input Shape: (N_REGIONS, N_SEQS, MAX_SEQ_LEN, 3)
    - Output Shape: (N_SEQS, MAX_SEQ_LEN, 3)
"""

# Imports
import numpy as np
import time
from typing import Optional
from numba import njit, prange

# Re-use a global generator so we don't re-seed every call
_RNG = np.random.default_rng()


@njit(parallel=True, cache=True)
def _truncate_sequences_kernel(arr, s_trunc, e_trunc, unmasked_len):
    """
    Numba kernel to truncate sequences in parallel.
    
    Masks the regions [0, s_trunc) and [unmasked_len - e_trunc, unmasked_len).
    
    Args:
        arr: np.ndarray of shape (n_seqs, L, 3), dtype bool - modified in-place
        s_trunc: np.ndarray of shape (n_seqs,), dtype int32
        e_trunc: np.ndarray of shape (n_seqs,), dtype int32
        unmasked_len: np.ndarray of shape (n_seqs,), dtype int32 (original unmasked lengths)
    """
    n_seqs = arr.shape[0]
    
    for i in prange(n_seqs):
        s = s_trunc[i]
        e = e_trunc[i]
        ulen = unmasked_len[i]
        
        # Skip if no truncation needed
        if s == 0 and e == 0:
            continue
            
        # Mask start: [0, s)
        for j in range(s):
            arr[i, j, 0] = True
            arr[i, j, 1] = False
            arr[i, j, 2] = False
            
        # Mask end: [ulen - e, ulen)
        start_mask_end = ulen - e
        # Ensure we don't go backwards if logic elsewhere failed 
        if start_mask_end < s:
            start_mask_end = s
            
        for j in range(start_mask_end, ulen):
            arr[i, j, 0] = True
            arr[i, j, 1] = False
            arr[i, j, 2] = False


@njit(parallel=True, cache=True)
def _shift_sequences_kernel(arr, unmasked_len, s_trunc, do_shift, shifts):
    """
    Numba kernel to shift sequences in parallel.
    
    Moves the unmasked block from [s_trunc, s_trunc + unmasked_len) to
    [shifts, shifts + unmasked_len) for each sequence where do_shift is True.
    
    Args:
        arr: np.ndarray of shape (n_seqs, L, 3), dtype bool - modified in-place
        unmasked_len: np.ndarray of shape (n_seqs,), dtype int32
        s_trunc: np.ndarray of shape (n_seqs,), dtype int32
        do_shift: np.ndarray of shape (n_seqs,), dtype bool
        shifts: np.ndarray of shape (n_seqs,), dtype int32 - new starting positions
            (values at indices where do_shift is False are ignored)
    """
    n_seqs = arr.shape[0]
    L = arr.shape[1]
    
    for i in prange(n_seqs):
        if not do_shift[i]:
            continue
        
        new_start = shifts[i]
        old_start = s_trunc[i]
        seq_len = unmasked_len[i]
        
        # Skip if no movement needed or empty sequence
        if new_start == old_start or seq_len <= 0:
            continue
        
        # Temporary buffer for the DNA bits (only for this sequence)
        temp_bits1 = np.empty(seq_len, dtype=np.bool_)
        temp_bits2 = np.empty(seq_len, dtype=np.bool_)
        
        # Copy the unmasked region to temp
        for j in range(seq_len):
            temp_bits1[j] = arr[i, old_start + j, 1]
            temp_bits2[j] = arr[i, old_start + j, 2]
        
        # Clear the entire array (set all positions to masked with zeroed bits)
        for j in range(L):
            arr[i, j, 0] = True
            arr[i, j, 1] = False
            arr[i, j, 2] = False
        
        # Write the unmasked region to new position
        for j in range(seq_len):
            arr[i, new_start + j, 0] = False
            arr[i, new_start + j, 1] = temp_bits1[j]
            arr[i, new_start + j, 2] = temp_bits2[j]


def gen_seq_variants(
    seqs_tensor      : np.ndarray,
    mutation_rate    : float = 0.0,
    min_trunc_start  : int   = 0,
    max_trunc_start  : int   = 0,
    min_trunc_end    : int   = 0,
    max_trunc_end    : int   = 0,
    trunc_prop       : float = 1.0,
    shift_prop       : float = 1.0,
    target_seq_len   : Optional[int] = None,
    use_accelerated_shifting : bool = True,
    use_accelerated_trunc    : bool = True,
    rng=_RNG):
    """
    Generate variants of 3-bit encoded DNA sequences.

    seqs_tensor: 3-bit encoded sequences (np.ndarray, dtype=bool)
        - There may be any number of dimensions >= 2
        - The final dimension is the 3-bit encoding, which is always size 3
        - The second to last dimension is the maximum sequence length, which can be any positive integer
        - These two last dimensions are for each sequence representations
        - The other dimensions (if any) are for all the different sequences
        - For example, (2, 135, 1600, 3) is 2x135 sequences, each with up to 1600 bases, each with 3-bit encoding
        - Or for example, (2000, 3) is a single sequence with 2000 bases, each with 3-bit encoding

    The 3-bit encoding is as follows:
        The 0th bit encodes the mask:
            - 0 = Not masked
            - 1 = Masked
        The 1st and 2nd bits encode the DNA base:
            - 00 = A
            - 01 = C
            - 10 = G
            - 11 = T

    So, if there is no masking for a sequence, the sequence is the maximum sequence length. But if there is masking, the sequence is the length of the unmasked region and the masked regions are ignored.

    min_trunc_start and max_trunc_start: int
        - Randomly truncate the start of sequences between min_trunc_start and max_trunc_start
        - This is done by zeroing out random numbers of bases at the start of the sequence (the unmasked region)

    min_trunc_end and max_trunc_end: int
        - Randomly truncate the end of sequences between min_trunc_end and max_trunc_end
        - This is done by zeroing out random numbers of bases at the end of the sequence (the unmasked region)

    trunc_prop: float
        - The proportion of sequences to randomly truncate
        - If 1.0, all sequences are truncated (subject to the start/end truncation ranges)
        - If 0.0, no sequences are truncated

    mutation_rate: float, between 0 and 1
        - The probability of point mutation per base
        - If a base if mutated, the base is changed to another base (3 options - each equally likely)
        - This mutation is done by flipping either: 
            - Only the 1st bit (A <-> C)
            - Only the 2nd bit (G <-> T)
            - Both bits (A <-> T or C <-> G)
        - The probability of each mutation type is equal

    shift_prop: float
        - Likelihood ∈ [0, 1] that any given sequence is shifted within its slack.
        - Shifting is done by randomly choosing a new starting position for the unmasked region.
        - So for example in a maximum sequence length of 1500, if the last 500 bases are masked, then the sequence is 1000 bases long. With shift_prop=0.5 the sequence has a 50% chance of being randomly shifted so that it starts at an index between 0 and 500; otherwise it remains left-aligned.

    target_seq_len: int or None
        - If provided, enforces a maximum length after truncation/mutation but before shifting.
        - This allows datasets imported with a higher MAX_IMPORTED_SEQ_LEN to keep their length
          until after augmentation, only cropping to MAX_MODEL_SEQ_LEN right before shifting.

    use_accelerated_shifting: bool
        - If True, use the numba-accelerated kernel for shifting (faster).
        - If False, use the original numpy implementation.

    use_accelerated_trunc: bool
        - If True, use the numba-accelerated kernel for truncation (faster).
        - If False, use the original numpy implementation.

    """

    # Validate shift proportion early
    if not (0.0 <= shift_prop <= 1.0):
        raise ValueError("shift_prop must be in [0, 1].")
    if not (0.0 <= trunc_prop <= 1.0):
        raise ValueError("trunc_prop must be in [0, 1].")

    # -----------------------------------------------------------------
    # Basic sanity checks – cheap, but catch silent bugs early
    # -----------------------------------------------------------------
    # If empty, return empty
    if seqs_tensor.size == 0:
        return seqs_tensor
    if seqs_tensor.dtype != np.bool_:
        raise TypeError("seqs_tensor must be boolean.")
    if seqs_tensor.shape[-1] != 3:
        raise ValueError("last dim must be size-3 (mask, bit1, bit2).")
    if not (0.0 <= mutation_rate <= 1.0):
        raise ValueError("mutation_rate ∉ [0, 1].")
    if min_trunc_start < 0 or max_trunc_start < 0 or min_trunc_start > max_trunc_start:
        raise ValueError("invalid start truncation range.")
    if min_trunc_end < 0 or max_trunc_end < 0 or min_trunc_end > max_trunc_end:
        raise ValueError("invalid end truncation range.")
    # mask bit at position 0 *must* be zero (no leading masking)
    if np.any(seqs_tensor[..., 0, 0]):
        raise ValueError("masking found before first base.")

    # -----------------------------------------------------------------
    # Flatten everything except (seq_len, 3) so we can work in 3-D
    # -----------------------------------------------------------------
    L             = seqs_tensor.shape[-2]                # max seq length
    outer_shape   = seqs_tensor.shape[:-2]
    n_seqs        = int(np.prod(outer_shape, dtype=np.int64))

    arr           = seqs_tensor.reshape(n_seqs, L, 3)    # view, no copy
    mask          = arr[..., 0]                          # (n, L)  bool
    bits          = arr[..., 1:]                         # (n, L, 2) bool

    # current unmasked length for every sequence  (trailing mask only)
    unmasked_len  = L - mask.sum(axis=-1, dtype=np.int32)   # (n,)


    # -----------------------------------------------------------------
    # 1. Random truncation
    # -----------------------------------------------------------------
    if (max_trunc_start > 0 or max_trunc_end > 0) and trunc_prop > 0.0:
        # Determine which sequences to truncate
        if trunc_prop >= 1.0:
            do_trunc = np.ones(n_seqs, dtype=bool)
        else:
            do_trunc = rng.random(n_seqs) < trunc_prop

        s_trunc = np.zeros(n_seqs, dtype=np.int32)
        e_trunc = np.zeros(n_seqs, dtype=np.int32)

        if np.any(do_trunc):
            n_trunc = np.sum(do_trunc)
            if max_trunc_start > 0:
                s_trunc[do_trunc] = rng.integers(min_trunc_start, max_trunc_start + 1, size=n_trunc, dtype=np.int32)
            if max_trunc_end > 0:
                e_trunc[do_trunc] = rng.integers(min_trunc_end, max_trunc_end + 1, size=n_trunc, dtype=np.int32)

        # never truncate past the actual sequence
        over    = s_trunc + e_trunc >= unmasked_len
        e_trunc[over] = np.maximum(unmasked_len[over] - s_trunc[over] - 1, 0)

        if use_accelerated_trunc:
            _truncate_sequences_kernel(arr, s_trunc, e_trunc, unmasked_len)
            
            # updated effective lengths
            unmasked_len -= s_trunc + e_trunc
        else:
            idx     = np.arange(L, dtype=np.int32)                         # (L,)
            to_mask = (idx < s_trunc[:, None]) | (idx >= (unmasked_len - e_trunc)[:, None])

            # set new mask + wipe the corresponding base bits
            newly   = to_mask & (~mask)
            mask   |= to_mask
            bits[newly] = False

            # updated effective lengths
            unmasked_len -= s_trunc + e_trunc
    else:
        s_trunc = np.zeros(n_seqs, dtype=np.int32)


    # -----------------------------------------------------------------
    # 2. Point mutations
    # -----------------------------------------------------------------
    if mutation_rate:
        unmasked = ~mask
        mut_hits = rng.random(unmasked.shape, dtype=np.float32) < mutation_rate
        mut_hits &= unmasked

        if mut_hits.any():
            mut_type = rng.integers(0, 3, size=mut_hits.shape, dtype=np.int8)
            flip1    = mut_hits & ((mut_type == 0) | (mut_type == 2))   # bit-1 flips
            flip2    = mut_hits & ((mut_type == 1) | (mut_type == 2))   # bit-2 flips

            bits[..., 0] ^= flip1
            bits[..., 1] ^= flip2


    # -----------------------------------------------------------------
    # 3. Optional max-length clamp (after truncation, before shifting)
    # -----------------------------------------------------------------
    if target_seq_len is not None:
        target_seq_len = int(target_seq_len)
        if target_seq_len <= 0:
            raise ValueError("target_seq_len must be positive.")
        if target_seq_len > L:
            raise ValueError("target_seq_len cannot exceed the current sequence length.")
        if target_seq_len < L:
            trim_total = L - target_seq_len
            start_drop = np.minimum(s_trunc, trim_total).astype(np.int32)
            # Remaining positions to drop after removing start slack
            remaining = trim_total - start_drop
            end_mask = L - (s_trunc + unmasked_len)
            end_drop_mask = np.minimum(end_mask, remaining)
            remaining -= end_drop_mask
            # Gather target_seq_len contiguous positions starting after the dropped prefix
            gather_idx = start_drop[:, None] + np.arange(target_seq_len, dtype=np.int32)[None, :]
            arr = np.take_along_axis(arr, gather_idx[..., None], axis=1)
            mask = arr[..., 0]
            bits = arr[..., 1:]
            s_trunc = (s_trunc - start_drop).astype(np.int32)
            unmasked_len = (unmasked_len - remaining).astype(np.int32)
            L = target_seq_len


    # -----------------------------------------------------------------
    # 4. Random shift of the *contiguous* unmasked block (per sequence probability)
    # -----------------------------------------------------------------
    if shift_prop > 0.0:
        if shift_prop >= 1.0:
            do_shift = np.ones(n_seqs, dtype=bool)
        else:
            do_shift = rng.random(n_seqs) < shift_prop

        if np.any(do_shift):
            if use_accelerated_shifting:
                # Compute slack (available space to shift into)
                slack = L - unmasked_len  # (n_seqs,)
                
                # Generate random shift amounts for shifted sequences
                shifts = np.zeros(n_seqs, dtype=np.int32)
                shifts[do_shift] = rng.integers(0, slack[do_shift] + 1)
                
                # Apply shifts using numba kernel (modifies arr in-place)
                _shift_sequences_kernel(arr, unmasked_len.astype(np.int32), 
                                        s_trunc.astype(np.int32), do_shift, shifts)
            else:
                # slack remaining for the unmasked block
                slack       = (L - unmasked_len)[do_shift]                      # (n_shift,)
                # sample new offsets as before
                shifts      = rng.integers(0, slack + 1)                        # (n_shift,)
                # gather indices:  j ↦ (j - shift + s_trunc)
                idx         = np.arange(L, dtype=np.int32)
                sub_s_trunc = s_trunc[do_shift]
                sub_lengths = unmasked_len[do_shift]

                gather_map  = idx[None, :] - shifts[:, None] + sub_s_trunc[:, None]
                gather_map  = np.clip(gather_map, 0, L - 1)

                # shifted copy of the full (mask + bits)
                shifted     = np.take_along_axis(arr[do_shift], gather_map[..., None], axis=1)

                # rebuild mask so positions outside [shift, shift+len) are masked
                keep        = (idx[None, :] >= shifts[:, None]) & (idx[None, :] < (shifts + sub_lengths)[:, None])
                shifted[..., 0] = ~keep
                # zero DNA bits wherever masked
                masked_pos  = shifted[..., 0]
                shifted[..., 1][masked_pos] = False
                shifted[..., 2][masked_pos] = False

                # final in-place swap (mem-efficient view assignment)
                arr[do_shift] = shifted

    return arr.reshape(outer_shape + (L, 3))


# DEPRECATED: Now using custom code inside apply_region_selection_and_variations
def select_regions(seqs_tensor, use_full_seqs, use_sub_seqs, region_dim=1, remove_region_dim=True, ranks=None, enforce_different_pairs=None):
    """
    Select a region from the available regions of each sequence.

    seqs_tensor: bool 3-bit encoded sequences
        - This holds single sequences, pairs, or triplets of sequences
           - (each sequence is a (MAX_SEQ_LEN, 3) tensor, but they can have multiple regions/representations)
        - For example,   (N_SEQS, N_REGIONS + 1, 1, MAX_SEQ_LEN, 3)    [single sequences]
        - Or for example,  (N_PAIRS, N_REGIONS + 1, 2, MAX_SEQ_LEN, 3)   [pairs]
        - Or for example, (N_TRIPETS, N_REGIONS + 1, 3, MAX_SEQ_LEN, 3) [triplets]
        - Where:
           - N_SEQS, N_PAIRS, or N_TRIPETS is any positive integer
           - N_REGIONS is any positive integer
           - MAX_SEQ_LEN is any positive integer
           - 3 is the number of bits in the 3-bit encoding, so it is always 3
    use_full_seqs: bool
        - If True, sample from the full sequences (index 0 of region dimension)
    use_sub_seqs: bool
        - If True, sample from the subsequences (indices > 0 of region dimension)
    region_dim: int
        - The dimension to select the region from (i.e. the N_REGIONS dimension)
    remove_region_dim: bool
        - If True, remove the region dimension from the output (which will have size 1)
    ranks: np.ndarray, optional
        - An array of ranks (length N_PAIRS or N_TRIPETS), used for special handling of certain ranks (e.g., rank 8 for pairs).
    enforce_different_pairs: list of (idx1, idx2) tuples, optional
        - For single sequences, ensures that the sequences at idx1 and idx2 get different
          regions. Used for subsequence pair mining where the same sequence needs to
          appear with different regions.
    
    Note: Index 0 on the region dimension is always the full sequence, and indices > 0 are the subsequences.
    """
    
    if not use_full_seqs and not use_sub_seqs:
        raise ValueError("At least one of use_full_seqs or use_sub_seqs must be True.")

    # How many regions are there? (including the full sequence)
    n_regions = seqs_tensor.shape[region_dim]

    # Determine sampling range
    min_region_idx = 0 if use_full_seqs else 1
    max_region_idx = n_regions if use_sub_seqs else 1

    if max_region_idx <= min_region_idx:
        raise ValueError(f"No regions to sample from with given configuration. "
                         f"use_full_seqs={use_full_seqs}, use_sub_seqs={use_sub_seqs}, n_regions={n_regions}")

    # Are there single sequences, pairs, or triplets?
    pair_triplet_dim = region_dim + 1
    n_seqs_in_pair_or_triplet = seqs_tensor.shape[pair_triplet_dim]
    is_single = n_seqs_in_pair_or_triplet == 1
    is_pairs = n_seqs_in_pair_or_triplet == 2
    is_triplets = n_seqs_in_pair_or_triplet == 3

    if not is_single and not is_pairs and not is_triplets:
        raise ValueError(f"seqs_tensor must have 1, 2, or 3 in dimension {pair_triplet_dim}")

    # How many single sequences/pairs/triplets are there?
    n_pairs_or_triplets = seqs_tensor.shape[0]

    # We need to select random regions for each sequence
    random_indices = _RNG.integers(min_region_idx, max_region_idx, size=(n_pairs_or_triplets, n_seqs_in_pair_or_triplet))

    # Enforce different regions for specified pairs of single sequences (used when subsequence pair mining requests cross-region duplicates)
    # This ensures that those duplicate sequences get different regions from their originals
    if enforce_different_pairs is not None:
        if not is_single:
            raise ValueError("enforce_different_pairs is only supported for single sequences")
        if (max_region_idx - min_region_idx) < 2:
            raise ValueError("Cannot enforce different regions when available regions < 2")
        for idx1, idx2 in enforce_different_pairs:
            # For single sequences, each has one region at position [idx, 0]
            while random_indices[idx1, 0] == random_indices[idx2, 0]:
                random_indices[idx2, 0] = _RNG.integers(min_region_idx, max_region_idx)

    # Here, enforce that the regions are *different* for pairs with rank 8
    if is_pairs and ranks is not None:
        # Check if we have enough regions to select different ones for rank 8 pairs
        if (max_region_idx - min_region_idx) < 2 and np.any(ranks == 8):
            raise ValueError("Cannot select two different regions when available regions < 2 for rank 8 pairs")
        
        rank_8_indices = np.where(ranks == 8)[0]
        for i in rank_8_indices:
            while random_indices[i, 0] == random_indices[i, 1]:
                random_indices[i, 1] = _RNG.integers(min_region_idx, max_region_idx)
    
    # The random indices are now like:
    # (N_SEQS, 1) or (N_PAIRS, 2) or (N_TRIPETS, 3)

    # Now use the random indices to select the regions for each sequence reducing the region dimension to 1
    # e.g. (N_PAIRS, N_REGIONS + 1, 2, MAX_SEQ_LEN, 3) -> (N_PAIRS, 1, 2, MAX_SEQ_LEN, 3)
    
    idx0 = np.arange(n_pairs_or_triplets)[:, np.newaxis]
    idx2 = np.arange(n_seqs_in_pair_or_triplet)[np.newaxis, :]

    # This advanced indexing selects the region given by random_indices for
    # each sequence.
    # It is equivalent to:
    # selected[i, j, ...] = seqs_tensor[i, random_indices[i, j], j, ...]
    selected = seqs_tensor[idx0, random_indices, idx2]

    # The indexing operation removes the region dimension. We add it back
    # here with size 1 to conform to the expected output shape.
    selected = np.expand_dims(selected, axis=region_dim)

    # Remove the region dimension if requested
    if remove_region_dim:
        selected = np.squeeze(selected, axis=region_dim)
    
    return selected, random_indices
