"""
encode_seqs.py - A script for encoding 16S rRNA sequences into 3-bit and (optionally) k-mer representations.

Input: Directory specified by INPUT_DIR, which must contain:
    - "FULL_seqs.fasta": Full-length 16S rRNA sequences.
    - Optionally, one or more "*_seqs.fasta" files (e.g., "V3-V4_seqs.fasta", "V4_seqs.fasta") 
      representing variable regions extracted from the full sequences.

Output: Adds files to the INPUT_DIR:
    - If SAVE_3BIT is True:
        - "3bit_seq_reps.npy": 3-bit representations 
            (OR "3bit_seq_reps_packed.npy" if PACK_3BIT_OUTPUT is True)
        - "about_3bit_encodings.txt": Metadata about the 3-bit encoding process.
    - If ENCODE_KMER is True:
        - "{K}-mer_seq_reps.npy": k-mer count representations derived from the 3-bit data (e.g., "7-mer_seq_reps.npy").
        - "about_{K}-mer_encodings.txt": Metadata about the k-mer encoding process.

3-bit encoding scheme:
    - Produces a single boolean tensor of shape [N_REGIONS + 1, N_SEQS, MAX_SEQ_LEN, 3]
    - N_REGIONS is the number of '*_seqs.fasta' files found (0 if none).
    - The first dimension (index 0) corresponds to the full-length sequences.
    - Subsequent indices correspond to the variable region files found (sorted alphabetically).
    - MAX_SEQ_LEN is the length of the longest full sequence (capped by MAX_SEQ_LEN constant).
    - The final dimension holds the 3-bit encoding for each base:
        - Bit 1 (Mask): 0 = Valid base, 1 = Masked/Padding
        - Bits 2, 3 (Base): 00=A, 01=C, 10=G, 11=T
        - Examples: 000=A, 001=C, 010=G, 011=T, 100=Masked/Padding
    - Masking are essentially just padding to handle variable length sequences
    - Sequences shorter than MAX_SEQ_LEN are right-padded with the mask encoding [1, 0, 0].
    - If PACK_3BIT_OUTPUT is True, the last dimension (3 bits) is packed into a single byte using np.packbits for efficient storage, resulting in a shape of [N_REGIONS + 1, N_SEQS, MAX_SEQ_LEN, 1].

3-bit encoding packing details:
    - By default, numpy stores each bit as a single byte (8 bits), which is inefficient
    - When packing is enabled (PACK_BITS = True):
        - The 3 bits for each base are packed into a single byte
        - This reduces the file size by approximately 3x
        - The packed format uses numpy's packbits/unpackbits functions
    - Example shapes:
        - Original shape: (3, 56932, 2000, 3)
        - Packed shape: (3, 56932, 2000, 1)
    - The packing is handled automatically by the write_3bit_seq_reps() and 
      read_3bit_seq_reps() functions

k-mer encoding scheme (optional, if ENCODE_KMER is True):
    - Input: The in-memory 3-bit boolean tensor [N_REGIONS + 1, N_SEQS, MAX_SEQ_LEN, 3].
    - Output: An int16 tensor [N_REGIONS + 1, N_SEQS, 4**K].
    - For each sequence (across regions and sequence IDs):
        - A sliding window of size K moves across the sequence.
        - k-mers containing any masked bases are ignored.
        - Valid k-mers are converted to a base-4 index (0 to 4**K - 1).
        - Counts of each k-mer index are tallied into a histogram of length 4**K.

Conceptual explanation of K-mer representations:
    - A K-mer is a contiguous subsequence of length K in a DNA sequence.
    - e.g. in the sequence "ATCGTTAG", the K-mer "ATCG" is a subsequence of length 4, called a 4-mer.
    - All of the 4-mers in the sequence are: "ATCG", "TCGT", "CGTT", "GTTA", "TTAG"
    - There are 4**K possible k-mers of length K, so there are 256 possible 4-mers: "AAAA", "AAAT", "AAAC", etc...
    - For any DNA sequence, we can encode it as a K-mer representation by counting the number of times each k-mer appears in the sequence.
    - Therefore, K-mer representations are vectors of positive integers of length 4**K.
    - IMPORTANT: The order of K-mers is important. It uses the order: A, C, G, T...
    - So for example the order of 3-mers is: AAAA, AAAC, AAAG, AAAT, AACA, AACC, AACG, AACT...
"""

from Bio import SeqIO
import numpy as np
import numba
import os
import glob
import datetime
import json

# --- Configuration ---

# Input directory containing the "*_seqs.fasta" files
INPUT_DIR = "/mnt/secondary/micro16s_dbs/16s_databases/m16s_db_004/seqs/" 
OUTPUT_DIR = "/mnt/secondary/micro16s_dbs/16s_databases/m16s_db_004/seqs/encoded/" 

# Maximum sequence length to consider for padding/truncation
# 1600-1800 is a good range for maximum lengths of full length 16S rRNA sequences
# If we are not using full sequences, then we can use a much shorter length like 400
MAX_SEQ_LEN = 600  

# 3-bit Encoding Settings
SAVE_3BIT = True        # Save the 3-bit representation to a .npy file?
PACK_3BIT_OUTPUT = False # Pack the 3-bit output file to save space? (Ignored if SAVE_3BIT is False)

# k-mer Encoding Settings
ENCODE_KMER = True  # Generate and save k-mer representations?
K_VALS = [4, 5, 6, 7]               # k-mer size (Ignored if ENCODE_KMER is False)
# Rough File Sizes:
#   - 5-mer encoding: ~250 MB
#   - 6-mer encoding: ~1 GB
#   - 7-mer encoding: ~4 GB
#   - 8-mer encoding: ~16 GB

# Validation Settings
VALIDATE_3BIT = False  # Perform a validation check on the 3-bit data? (It probably works anyway, but this is a good sanity check)

# --- Output File Paths ---
OUTPUT_3BIT_FILENAME = "3bit_seq_reps_packed.npy" if PACK_3BIT_OUTPUT else "3bit_seq_reps.npy"
OUTPUT_3BIT_PATH = os.path.join(OUTPUT_DIR, OUTPUT_3BIT_FILENAME) if SAVE_3BIT else None
ABOUT_3BIT_PATH = os.path.join(OUTPUT_DIR, "about_3bit_encodings.txt") if SAVE_3BIT else None

OUTPUT_KMER_PATHS = []
ABOUT_KMER_PATHS = []
for K in K_VALS:
    OUTPUT_KMER_FILENAME = f"{K}-mer_seq_reps.npy"
    OUTPUT_KMER_PATHS.append(os.path.join(OUTPUT_DIR, OUTPUT_KMER_FILENAME) if ENCODE_KMER else None)
    ABOUT_KMER_PATHS.append(os.path.join(OUTPUT_DIR, f"about_{K}-mer_encodings.txt") if ENCODE_KMER else None)

# --- Helper Data ---
DNA_BASE_REPLACE_DICT = {
    "N": ["A", "C", "G", "T"], # Ambiguous base
    "R": ["A", "G"], # Purine
    "Y": ["C", "T"], # Pyrimidine
    "S": ["G", "C"], # Strong
    "W": ["A", "T"], # Weak
    "K": ["G", "T"], # Keto
    "M": ["A", "C"], # Amino
    "B": ["C", "G", "T"], # Not A
    "D": ["A", "G", "T"], # Not C
    "H": ["A", "C", "T"], # Not G
    "V": ["A", "C", "G"]  # Not T
}


# --- 3-bit Encoding Functions ---

def encode_base(base):
    """Convert a DNA base to its 3-bit representation [mask, b1, b0]."""
    base = base.upper()
    if base == 'A':
        return [0, 0, 0]
    elif base == 'C':
        return [0, 0, 1]
    elif base == 'G':
        return [0, 1, 0]
    elif base == 'T':
        return [0, 1, 1]
    else:
        raise ValueError(f"Invalid base: {base}")
        #return [1, 0, 0]  # Masked/invalid/padding base

def replace_non_standard_bases(base):
    """Randomly replace an ambiguous DNA base code with one of its standard possibilities."""
    base = base.upper()
    if base in DNA_BASE_REPLACE_DICT:
        return np.random.choice(DNA_BASE_REPLACE_DICT[base])
    elif base not in 'ACGT':
        # This case should ideally not happen if the input is DNA,
        # but handle it just in case and issue a warning.
        print(f"WARNING: Found unexpected non-standard base '{base}'. Treating as 'N'.")
        return np.random.choice(DNA_BASE_REPLACE_DICT["N"])
    else:
        # Base is already A, C, G, or T
        return base

def encode_sequence_3bit(seq_str, max_length):
    """Convert a DNA sequence string to its padded 3-bit boolean representation."""
    # Truncate if longer than max_length
    seq_str = seq_str[:max_length]
    
    # Handle non-standard bases (N, R, Y, etc.)
    processed_seq = "".join(replace_non_standard_bases(b) for b in seq_str)
    
    # Initialize encoding array (padded with mask)
    encoding = np.full((max_length, 3), [1, 0, 0], dtype=bool)
    
    # Encode the actual sequence bases
    for i, base in enumerate(processed_seq):
        encoding[i] = encode_base(base)
        
    return encoding

def write_3bit_seq_reps(file_path, seq_reps, packed=False):
    """Write 3-bit sequence representations to a numpy file."""
    print(f"Saving 3-bit representations to {os.path.basename(file_path)} (Packed: {packed})...")
    if packed:
        # Pack along the last dimension (the 3 bits)
        # 'big' endian ensures the mask bit (most significant) comes first in the byte
        packed_seqs = np.packbits(seq_reps, axis=-1, bitorder='big')
        # Keep only the first byte since we only need 3 bits. Result shape ends in 1.
        packed_seqs = packed_seqs[..., :1] 
        np.save(file_path, packed_seqs)
    else:
        np.save(file_path, seq_reps)
    print(f"Saved: {file_path} ({os.path.getsize(file_path):,} bytes)")
    return file_path

def read_3bit_seq_reps(file_path, packed=None):
    """Read 3-bit sequence representations from a numpy file.
    
    Args:
        file_path (str): Path to the numpy file
        packed (bool, optional): Whether the file contains packed bits. 
            If None, inferred from filename.
    
    Returns:
        np.ndarray: Array of shape [N_REGIONS, N_SEQS, MAX_SEQ_LEN, 3] containing boolean values
    """
    # Infer packed status from filename if not specified
    if packed is None:
        packed = '_packed.npy' in file_path
    
    # Load the array
    arr = np.load(file_path)
    
    # Unpack if necessary
    if packed:
        # Remove the last singleton dimension and reshape for sequence length
        arr = arr.squeeze(-1)
        arr = arr.reshape(*arr.shape[:-1], MAX_SEQ_LEN, -1)
        
        # Unpack bits and take only the first 3 bits per base
        arr = np.unpackbits(arr, axis=-1, bitorder='big')[..., :3]
    
    return arr.astype(bool)





# --- k-mer Encoding Functions ---

# Pure NumPy approach (~10x slower than Numba)
def encode_sequences_kmer_numpy(seq_reps_3bit, k_value):
    """
    Convert 3-bit encoded sequences to k-mer count representations with high performance.

    This function is optimized for speed, using vectorized NumPy operations to process
    large batches of sequences efficiently.

    Input:
        seq_reps_3bit (np.ndarray): A boolean NumPy array of shape [..., SEQ_LEN, 3].
                                    The dimensions "..." can be any number of batch dimensions.
        k_value (int):              The size of the k-mer (e.g., 6 for 6-mers).

    Output:
        np.ndarray: An int16 NumPy array of shape [..., 4**k_value], containing the
                    k-mer counts for each sequence.
    """
    # --- 1. Input Validation and Shape Setup ---
    if not isinstance(seq_reps_3bit, np.ndarray):
        raise TypeError("Input seq_reps_3bit must be a NumPy ndarray.")
    if seq_reps_3bit.ndim < 2:
        raise ValueError("Input 3-bit sequence representations must have at least 2 dimensions (sequence_length, 3-bits).")
    if seq_reps_3bit.shape[-1] != 3:
        raise ValueError("Input 3-bit sequence representations must have a final dimension of size 3.")
    if not isinstance(k_value, int) or k_value < 0:
        raise ValueError("k_value must be a non-negative integer.")

    # This algorithm relies on 64-bit integer arithmetic to calculate k-mer indices.
    # For k >= 32, 4**k exceeds the maximum value of a 64-bit integer, causing
    # numpy to use floats, which breaks the indexing for bincount.
    if k_value >= 32:
        raise ValueError(
            f"k_value ({k_value}) is too large for this vectorized algorithm. "
            f"The internal representation of k-mer indices overflows 64-bit integers "
            f"when k >= 32. Please use a smaller k_value."
        )

    original_shape = seq_reps_3bit.shape
    seq_len = original_shape[-2]

    if k_value > seq_len:
        raise ValueError(f"k_value ({k_value}) cannot be larger than the sequence length ({seq_len}).")

    kmer_dim = 4**k_value
    
    if k_value == 0:
        flat_masks = seq_reps_3bit[..., 0].reshape(-1, seq_len)
        is_valid_sequence = ~np.all(flat_masks, axis=1)
        output_counts = is_valid_sequence.astype(np.int16).reshape(*original_shape[:-2], 1)
        return output_counts

    batch_shape = original_shape[:-2]
    num_sequences = np.prod(batch_shape, dtype=int) if batch_shape else 1
    
    # Check if the total number of bins for the bincount trick will overflow.
    # The offsets and the minlength for bincount must fit within a 64-bit integer.
    total_bins = float(num_sequences) * kmer_dim
    if total_bins > np.iinfo(np.int64).max:
        raise ValueError(
            f"The combination of number of sequences ({num_sequences}) and k_value ({k_value}) "
            f"results in a total number of bins ({total_bins:.2e}) that exceeds the "
            f"maximum 64-bit integer size. Please use smaller batches or a smaller k_value."
        )

    flat_seqs = seq_reps_3bit.reshape(num_sequences, seq_len, 3)

    # --- 2. Vectorized Base and Mask Conversion ---
    masks = flat_seqs[:, :, 0]
    # Explicitly cast to int64 to ensure all subsequent arithmetic is 64-bit.
    int_bases = (flat_seqs[:, :, 1] * 2 + flat_seqs[:, :, 2]).astype(np.int64)

    # --- 3. Create Sliding Windows (Memory-Efficient Views) ---
    num_windows = seq_len - k_value + 1
    window_shape = (num_sequences, num_windows, k_value)
    base_strides = (int_bases.strides[0], int_bases.strides[1], int_bases.strides[1])
    mask_strides = (masks.strides[0], masks.strides[1], masks.strides[1])

    windowed_bases = np.lib.stride_tricks.as_strided(int_bases, shape=window_shape, strides=base_strides)
    windowed_masks = np.lib.stride_tricks.as_strided(masks, shape=window_shape, strides=mask_strides)

    # --- 4. Calculate K-mer Indices ---
    # Use np.int64 for the powers to support k-values up to 31 without overflow.
    # The original code's use of int32 was a bug for k >= 16.
    powers_of_4 = 4 ** np.arange(k_value - 1, -1, -1, dtype=np.int64)
    kmer_indices = windowed_bases @ powers_of_4

    # --- 5. Filter Invalid K-mers and Count ---
    is_valid_kmer = ~windowed_masks.any(axis=2)
    
    # Explicitly use int64 for offsets to prevent overflow when multiplying by kmer_dim.
    offsets = np.arange(num_sequences, dtype=np.int64) * kmer_dim
    offset_indices = kmer_indices + offsets[:, np.newaxis]
    final_indices_to_count = offset_indices[is_valid_kmer]

    # The minlength must be an integer.
    total_counts = np.bincount(final_indices_to_count, minlength=int(total_bins))

    # --- 6. Reshape to Final Output Format ---
    seq_reps_kmer_flat = total_counts.reshape(num_sequences, kmer_dim)
    output_shape = batch_shape + (kmer_dim,)
    seq_reps_kmer = seq_reps_kmer_flat.reshape(output_shape).astype(np.int16)

    return seq_reps_kmer


# Numba approach (~10x faster than pure NumPy)
def encode_sequences_kmer_numba(seq_reps_3bit, k_value):
    """
    Convert 3-bit encoded sequences to k-mer count representations with high performance.

    This function is optimized for speed, using a Numba-accelerated, parallel
    algorithm to process large batches of sequences efficiently.

    Input:
        seq_reps_3bit (np.ndarray): A boolean NumPy array of shape [..., SEQ_LEN, 3].
                                    The dimensions "..." can be any number of batch dimensions.
        k_value (int):              The size of the k-mer (e.g., 6 for 6-mers).

    Output:
        np.ndarray: An int16 NumPy array of shape [..., 4**k_value], containing the
                    k-mer counts for each sequence.
    """
    # --- 1. Input Validation and Shape Setup ---
    if not isinstance(seq_reps_3bit, np.ndarray):
        raise TypeError("Input seq_reps_3bit must be a NumPy ndarray.")
    if seq_reps_3bit.ndim < 2:
        raise ValueError("Input 3-bit sequence representations must have at least 2 dimensions (sequence_length, 3-bits).")
    if seq_reps_3bit.shape[-1] != 3:
        raise ValueError("Input 3-bit sequence representations must have a final dimension of size 3.")
    if not isinstance(k_value, int) or k_value < 0:
        raise ValueError("k_value must be a non-negative integer.")
    
    # Numba approach does not have the same 64-bit integer overflow issues as the
    # previous numpy-only approach, but k-mer dimensions can still become enormous.
    # A practical limit is useful. k=16 -> 4GB array per sequence.
    if k_value > 15:
        raise ValueError(f"k_value ({k_value}) is likely too large, leading to excessive memory usage.")

    original_shape = seq_reps_3bit.shape
    seq_len = original_shape[-2]

    if k_value > seq_len:
        raise ValueError(f"k_value ({k_value}) cannot be larger than the sequence length ({seq_len}).")

    kmer_dim = 4**k_value
    
    if k_value == 0:
        flat_masks = seq_reps_3bit[..., 0].reshape(-1, seq_len)
        is_valid_sequence = ~np.all(flat_masks, axis=1)
        output_counts = is_valid_sequence.astype(np.int16).reshape(*original_shape[:-2], 1)
        return output_counts

    batch_shape = original_shape[:-2]
    num_sequences = np.prod(batch_shape, dtype=int) if batch_shape else 1
    flat_seqs = seq_reps_3bit.reshape(num_sequences, seq_len, 3)

    # --- 2. Vectorized Base and Mask Conversion ---
    masks = flat_seqs[:, :, 0]
    # Using uint8 is sufficient for bases 0-3 and is memory friendly
    int_bases = (flat_seqs[:, :, 1] * 2 + flat_seqs[:, :, 2]).astype(np.uint8)

    # --- 3. Numba-Accelerated K-mer Counting ---
    # Call the JIT-compiled, parallel function for the main computation.
    seq_reps_kmer_flat = _numba_kmer_count(int_bases, masks, k_value, kmer_dim)

    # --- 4. Reshape to Final Output Format ---
    output_shape = batch_shape + (kmer_dim,)
    seq_reps_kmer = seq_reps_kmer_flat.reshape(output_shape)

    return seq_reps_kmer

@numba.jit(nopython=True, parallel=True, cache=True)
def _numba_kmer_count(int_bases, masks, k_value, kmer_dim):
    """
    Numba-jitted kernel for high-speed, parallel k-mer counting.
    
    This function is the performance-critical core. `nopython=True` ensures it
    compiles to pure machine code without Python interpreter overhead. `parallel=True`
    enables automatic parallelization across CPU cores. `cache=True` saves the
    compiled function to disk to speed up subsequent script runs.
    """
    num_sequences, seq_len = int_bases.shape
    counts = np.zeros((num_sequences, kmer_dim), dtype=np.int16)
    num_windows = seq_len - k_value + 1

    # numba.prange parallelizes this outer loop. Each sequence is processed
    # independently on a separate thread, which is safe as each thread writes
    # to its own unique row `counts[i, :]`.
    for i in numba.prange(num_sequences):
        # Loop over all possible start positions for a k-mer in the sequence
        for j in range(num_windows):
            
            # --- 1. Check if the k-mer window is valid (contains no masks) ---
            is_window_valid = True
            for l in range(k_value):
                if masks[i, j + l]:
                    is_window_valid = False
                    break
            
            # --- 2. If valid, calculate the k-mer index and increment count ---
            if is_window_valid:
                # This simple loop calculates the base-4 index. Numba compiles
                # this into highly efficient machine code.
                kmer_index = 0
                for l in range(k_value):
                    kmer_index = kmer_index * 4 + int_bases[i, j + l]
                
                counts[i, kmer_index] += 1
                
    return counts
















# --- Main Execution Block ---

if __name__ == "__main__":

    # --- 0. Initialize Output Directory ---
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"Starting sequence encoding process for directory: {INPUT_DIR}")
    start_time = datetime.datetime.now()

    # --- 1. Find Input Sequence Files ---
    full_seqs_path = os.path.join(INPUT_DIR, "FULL_seqs.fasta")
    if not os.path.exists(full_seqs_path):
        raise FileNotFoundError(f"Required file not found: {full_seqs_path}")
    
    # Find optional region files, sort them for consistent order
    region_files = sorted(glob.glob(os.path.join(INPUT_DIR, "*_seqs.fasta")))
    region_files = [f for f in region_files if not f.endswith("FULL_seqs.fasta") and "failed" not in os.path.basename(f)] # Exclude full and failed seqs
    region_names = [os.path.basename(f).replace("_seqs.fasta", "") for f in region_files]
    
    print(f"Found full sequences file: {os.path.basename(full_seqs_path)}")
    if region_files:
        print(f"Found {len(region_files)} region files: {', '.join(region_names)}")
    else:
        print("No additional region files found.")
        
    # --- 2. Load Full Sequences & Determine Dimensions ---
    print("Loading full sequences to get IDs and max length...")
    full_seqs = list(SeqIO.parse(full_seqs_path, "fasta"))
    if not full_seqs:
         raise ValueError(f"No sequences found in {full_seqs_path}")
         
    seq_ids = [seq.id for seq in full_seqs]
    # Determine max length based on actual data, capped by the constant
    actual_max_len = max(len(seq.seq) for seq in full_seqs)
    max_length = min(actual_max_len, MAX_SEQ_LEN) 
    print(f"Found {len(seq_ids)} sequences. Actual max length: {actual_max_len}. Using max length: {max_length}")
    
    # --- 3. Initialize 3-bit Output Tensor ---
    n_regions_total = len(region_files) + 1 # +1 for full sequences
    n_seqs = len(seq_ids)
    # Shape: [N_REGIONS_TOTAL, N_SEQS, MAX_LENGTH, 3]
    encoded_seqs_3bit = np.zeros((n_regions_total, n_seqs, max_length, 3), dtype=bool)
    print(f"Initialized 3-bit tensor with shape: {encoded_seqs_3bit.shape}")

    # --- 4. Perform 3-bit Encoding ---
    # Encode full sequences (index 0)
    print("Encoding full sequences (Region index 0)...")
    for i, seq_record in enumerate(full_seqs):
        if i % 5000 == 0 and i > 0:
             print(f"  Processed {i}/{n_seqs} full sequences...")
        encoded_seqs_3bit[0, i] = encode_sequence_3bit(str(seq_record.seq), max_length)
    print(f"  Finished encoding {n_seqs} full sequences.")
    
    # Encode each region (indices 1 to N)
    for region_idx, region_file in enumerate(region_files, 1):
        region_name = region_names[region_idx-1]
        print(f"Encoding {region_name} region (Region index {region_idx})...")
        # Load region seqs into a dictionary for quick lookup
        region_seqs_dict = {seq.id: seq for seq in SeqIO.parse(region_file, "fasta")}
        
        processed_count = 0
        missing_count = 0
        for seq_idx, seq_id in enumerate(seq_ids):
            if seq_id in region_seqs_dict:
                encoded_seqs_3bit[region_idx, seq_idx] = encode_sequence_3bit(
                    str(region_seqs_dict[seq_id].seq), max_length)
                processed_count += 1
            else:
                # If a sequence ID from full_seqs is missing in a region file, 
                # fill its entry with mask padding.
                encoded_seqs_3bit[region_idx, seq_idx] = np.array([[1, 0, 0]] * max_length, dtype=bool)
                missing_count += 1
        print(f"  Finished encoding {region_name}. Processed {processed_count} sequences, {missing_count} missing (filled with mask).")

    # --- 4.5 Save Region Indices ---
    region_indices_path = os.path.join(OUTPUT_DIR, "region_indices.json")
    print(f"Saving region indices to {os.path.basename(region_indices_path)}...")
    
    region_indices_data = {"region_indices": {0: "FULL"}}
    for idx, name in enumerate(region_names, 1):
        region_indices_data["region_indices"][idx] = name
        
    with open(region_indices_path, 'w') as f:
        json.dump(region_indices_data, f, indent=4)
    print(f"Saved: {region_indices_path}")

    # --- 5. Save 3-bit Encodings & Metadata (Optional) ---
    if SAVE_3BIT and OUTPUT_3BIT_PATH and ABOUT_3BIT_PATH:
        output_3bit_filepath = write_3bit_seq_reps(OUTPUT_3BIT_PATH, encoded_seqs_3bit, packed=PACK_3BIT_OUTPUT)
        
        print(f"Writing 3-bit encoding metadata to {os.path.basename(ABOUT_3BIT_PATH)}...")
        with open(ABOUT_3BIT_PATH, 'w') as f: # Overwrite or create
            f.write("3-bit Sequence Encoding Metadata\n")
            f.write("================================\n")
            f.write(f"Processing date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Input directory: {INPUT_DIR}\n\n")
            f.write(f"Output directory: {OUTPUT_DIR}\n\n")
            
            f.write("Tensor information:\n")
            f.write(f"Output file: {os.path.basename(output_3bit_filepath)}\n")
            # Determine shape based on whether it was packed
            saved_shape = np.load(output_3bit_filepath).shape if os.path.exists(output_3bit_filepath) else "File not found"
            f.write(f"Shape in file: {saved_shape}\n")
            f.write(f"Original (in-memory) shape: {encoded_seqs_3bit.shape}\n")
            f.write(f"Number of sequences: {n_seqs}\n")
            f.write(f"Maximum sequence length (padded to): {max_length}\n")
            f.write(f"Total regions (including full): {n_regions_total}\n\n")
            
            f.write("Region indices in tensor:\n")
            f.write("  0: Full length sequences\n")
            for idx, name in enumerate(region_names, 1):
                f.write(f"  {idx}: {name} region\n")
            
            f.write("\nEncoding scheme (per base):\n")
            f.write("- 3 bits [mask, bit1, bit0] (boolean)\n")
            f.write("- 000: A (unmasked)\n")
            f.write("- 001: C (unmasked)\n")
            f.write("- 010: G (unmasked)\n")
            f.write("- 011: T (unmasked)\n")
            f.write("- 100: Masked/padded/ambiguous position\n")
            f.write("- Other combinations ([1,0,1], [1,1,0], [1,1,1]) are unused by this script.\n\n")
            
            f.write("Storage details:\n")
            if PACK_3BIT_OUTPUT:
                f.write("- Data type: uint8 (packed bits)\n")
                f.write("- Storage: 3 bits per base (packed into 1 byte)\n")
            else:
                f.write("- Data type: boolean\n")
                f.write("- Storage: 3 bytes per base (unpacked)\n")
            f.write(f"- File size: {os.path.getsize(output_3bit_filepath):,} bytes\n")
        print(f"Finished writing {os.path.basename(ABOUT_3BIT_PATH)}")

    # --- 6. Perform k-mer Encoding (Optional) ---
    kmer_seq_reps = None # Initialize in case k-mer encoding is skipped
    if ENCODE_KMER:
        for K, OUTPUT_KMER_PATH, ABOUT_KMER_PATH in zip(K_VALS, OUTPUT_KMER_PATHS, ABOUT_KMER_PATHS):
            if not isinstance(K, int) or K <= 0:
                raise ValueError(f"K must be a positive integer, but got {K}")
            kmer_seq_reps = encode_sequences_kmer_numba(encoded_seqs_3bit, K)
            
            # --- 7. Save k-mer Encodings & Metadata (Optional) ---
            if OUTPUT_KMER_PATH and ABOUT_KMER_PATH:
                print(f"Saving {K}-mer representations to {os.path.basename(OUTPUT_KMER_PATH)}...")
                np.save(OUTPUT_KMER_PATH, kmer_seq_reps)
                print(f"Saved: {OUTPUT_KMER_PATH} ({os.path.getsize(OUTPUT_KMER_PATH):,} bytes)")

                print(f"Writing {K}-mer encoding metadata to {os.path.basename(ABOUT_KMER_PATH)}...")
                with open(ABOUT_KMER_PATH, 'w') as f: # Overwrite or create
                    f.write(f"{K}-mer Sequence Encoding Metadata\n")
                    f.write("=================================\n")
                    f.write(f"Processing date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"Input directory: {INPUT_DIR}\n")
                    f.write(f"Output directory: {OUTPUT_DIR}\n")
                    if SAVE_3BIT and OUTPUT_3BIT_PATH:
                        f.write(f"Based on 3-bit data from: {os.path.basename(OUTPUT_3BIT_PATH)}\n\n")
                    else:
                        f.write(f"Based on 3-bit data generated in memory.\n\n")

                    f.write("k-mer settings:\n")
                    f.write(f"- k = {K}\n")
                    f.write(f"- Vector length = 4^{K} = {4**K}\n\n")

                    f.write("Tensor information:\n")
                    f.write(f"Output file: {os.path.basename(OUTPUT_KMER_PATH)}\n")
                    f.write(f"Shape: {kmer_seq_reps.shape}\n")
                    f.write(f"Data type: {kmer_seq_reps.dtype}\n")
                    f.write(f"Number of sequences: {n_seqs}\n")
                    f.write(f"Total regions (including full): {n_regions_total}\n\n")

                    f.write("Region indices correspond to the 3-bit encoding:\n")
                    f.write("  0: Full length sequences\n")
                    for idx, name in enumerate(region_names, 1):
                        f.write(f"  {idx}: {name} region\n")
                    
                    f.write("\nEncoding process:\n")
                    f.write(f"- Uses the {encoded_seqs_3bit.shape} 3-bit boolean tensor as input.\n")
                    f.write(f"- Slides a window of size K={K} across each sequence (dim=-2).\n")
                    f.write("- Ignores k-mers containing any masked base (where the first bit is 1).\n")
                    f.write("- Converts each valid k-mer into a base-4 index (0 to 4**K - 1).\n")
                    f.write(f"- Counts occurrences of each index, storing the result as a vector of length {4**K}.\n")
                    f.write("- Resulting tensor shape: [N_REGIONS+1, N_SEQS, 4**K].\n\n")

                    f.write("Storage details:\n")
                    f.write(f"- Data type: {kmer_seq_reps.dtype}\n")
                    f.write(f"- File size: {os.path.getsize(OUTPUT_KMER_PATH):,} bytes\n")
                print(f"Finished writing {os.path.basename(ABOUT_KMER_PATH)}")

    # --- 8. Completion ---
    end_time = datetime.datetime.now()
    print(f"\nEncoding complete. Total time: {end_time - start_time}")
    if SAVE_3BIT and OUTPUT_3BIT_PATH:
        print(f"  - 3-bit output saved: {os.path.basename(OUTPUT_3BIT_PATH)}")
        print(f"  - 3-bit metadata saved: {os.path.basename(ABOUT_3BIT_PATH)}")
    if ENCODE_KMER and OUTPUT_KMER_PATHS:
        for K, OUTPUT_KMER_PATH, ABOUT_KMER_PATH in zip(K_VALS, OUTPUT_KMER_PATHS, ABOUT_KMER_PATHS):
            print(f"  - {K}-mer output saved: {os.path.basename(OUTPUT_KMER_PATH)}")
            print(f"  - {K}-mer metadata saved: {os.path.basename(ABOUT_KMER_PATH)}")

    # --- 9. Validation Check ---
    if VALIDATE_3BIT and SAVE_3BIT and OUTPUT_3BIT_PATH:
        print("\nPerforming validation check on saved 3-bit data...")
        try:
            # Read the saved data back
            loaded_data = read_3bit_seq_reps(OUTPUT_3BIT_PATH, packed=PACK_3BIT_OUTPUT)
            
            # Check shape
            if loaded_data.shape != encoded_seqs_3bit.shape:
                print(f"  WARNING: Shape mismatch! Original: {encoded_seqs_3bit.shape}, Loaded: {loaded_data.shape}")
            else:
                print(f"  Shape check passed: {loaded_data.shape}")
            
            # Check content (data equivalence)
            if np.array_equal(loaded_data, encoded_seqs_3bit):
                print("  Content validation PASSED: Saved data matches original in-memory data.")
            else:
                print("  WARNING: Content validation FAILED: Saved data differs from original in-memory data!")
                # Calculate percentage of matching elements
                match_percentage = np.mean(loaded_data == encoded_seqs_3bit) * 100
                print(f"  Match percentage: {match_percentage:.2f}%")
        except Exception as e:
            print(f"  ERROR during validation: {str(e)}")
