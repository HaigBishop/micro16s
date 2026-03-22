"""
Classification Benchmarking Script for 16S rRNA Sequences for RDP

This is script 1 out of 3:
1. classify_bench_01_rdp.py
2. classify_bench_02_m16s.py
3. classify_bench_03_analyse.py

This script benchmarks taxonomy classification using DADA2's naive Bayesian classifier
via an R script wrapper (classify_rdp.R). It evaluates classification performance across
multiple 16S rRNA gene regions by:

1. Loading training sequences from the FULL_seqs.fasta reference
2. Loading test sequences from multiple region-specific FASTA files
3. Subsampling regions per sequence via MAX_REGIONS_PER_SEQUENCE_TEST
4. Running DADA2 taxonomy assignment through classify_rdp.R
5. Outputting predicted and true classifications with region metadata

Inputs:
    - DATASET_SPLIT_DIR_PATH: Directory containing the dataset split with:
        - testing_indices.txt: Line-separated indices for test sequences
        - training_indices.txt: Line-separated indices for training sequences
        - seqs/: Subdirectory containing region FASTA files (*_seqs.fasta)

Outputs:
    - train_seqs_rdp.fna: Training sequences (from FULL_seqs.fasta)
    - train_seqs_m16s.fna: Training sequences with random regions (for Micro16S)
    - test_seqs.fna: Test sequences (from selected regions)
    - predicted_classes_rdp.csv: DADA2 predicted taxonomies with region metadata
    - true_classes.csv: Ground truth taxonomies with region metadata

Training Sequences:
    Two versions of training sequences are generated to reflect different classifier needs:
    
    - train_seqs_rdp.fna: Uses FULL sequences for all training indices
      (RDP classifier trains on full-length 16S sequences)
      
    - train_seqs_m16s.fna: Uses randomly selected regions (1 per sequence) for training
      (Micro16S classifier trains on region-specific sequences to match test conditions)
      Each training sequence gets a single randomly selected region (excluding FULL).
      Same sequences and order as train_seqs_rdp.fna, just different regions.

Note on FASTA files:
    All `*_seqs.fasta` files in a `seqs` directory (excluding those starting with
    "failed") are guaranteed to mirror each other. This means they have:
    - The same order of sequences
    - The same headers (containing taxonomy)
    - The same number of sequences
    
    This mirroring property allows us to:
    - Use any region file to extract taxonomy information
    - Select the same sequence index across different region files
    - Combine sequences from multiple regions while tracking their origins

ASV ID Format:
    Each sequence is assigned a unique ASV_ID combining its index and region:
    - Test sequences: "{index}_{region}" (e.g., "42_V4-004", "42_V3-V4-002")
    - Train sequences (RDP): "{index}_FULL" (e.g., "42_FULL")
    - Train sequences (Micro16S): "{index}_{region}" (e.g., "42_V4-003")
"""

import subprocess
import os
import random
from glob import glob
import time


# =============================================================================
# Configuration
# =============================================================================

# Input/Output paths
DATABASE_DIR_PATH = "/mnt/secondary/micro16s_dbs/16s_databases/m16s_db_004"
DATASET_SPLIT_DIR_PATH = "/mnt/secondary/micro16s_dbs/16s_databases/m16s_db_004/seqs/split_001"
OUTPUT_DIR_PATH = DATASET_SPLIT_DIR_PATH + "/classification_eval_excluded"
PREDICTED_CLASSES_RDP_PATH = OUTPUT_DIR_PATH + "/predicted_classes_rdp.csv"
TRUE_CLASSES_PATH = OUTPUT_DIR_PATH + "/true_classes.csv"

# Options
USE_ALL_SEQUENCES_FOR_TRAIN = False  # Whether to use all sequences for the train set (i.e. join the train, test and excluded sets)
TEST_OR_EXCLUDED_TAXA = "excluded"  # What set to test on?    "test" or "excluded"
USE_FULL_SEQS_FOR_TEST = False  # Whether to include FULL_seqs.fasta in the test set
MAX_REGIONS_PER_SEQUENCE_TEST = 10  # Max regions per sequence for the test set (None = use all available)
ONLY_USE_FIRST_N_SEQUENCES_FOR_TEST = None  # Only use the first N sequences for the test set (None = use all)
MIN_SEQUENCE_LENGTH = 40  # Minimum sequence length to allow
ALLOW_AMBIGUOUS_BASES_IN_FULL_SEQS_FOR_TRAINING = True  # Whether to allow ambiguous bases in FULL_seqs.fasta - FYI, in assignTaxonomy, k-mers that contains ambiguous bases are simply ignored
ASSIGN_TAXONOMY_BATCH_SIZE = 1000  # Batch size for assignTaxonomy (too many sequences at once can cause errors)
TRUNCATE_N_BP_FOR_SUBSEQUENCES = 30 # Number of bases to truncate from either end of every single non-FULL sequence

# =============================================================================
# Helper Functions
# =============================================================================

def parse_fasta_file(fasta_path, allow_ambiguous=False, truncate_bp_from_ends=0):
    """
    Parse a FASTA file and return a list of (header, sequence) tuples.
    
    Args:
        fasta_path: Path to the FASTA file
        allow_ambiguous: Whether to allow ambiguous bases (non-ATCG)
        truncate_bp_from_ends: Number of bases to remove from start and end of sequence
        
    Returns:
        List of (header, sequence) tuples, where header includes the '>'
    """
    sequences = []
    current_header = None
    current_seq_lines = []

    def validate_and_add_sequence(header, seq_lines):
        sequence = "".join(seq_lines)
        
        # Truncate sequences if requested (before validation)
        if truncate_bp_from_ends > 0:
            # If sequence is shorter than 2*truncate, this results in empty string
            if len(sequence) <= 2 * truncate_bp_from_ends:
                sequence = ""
            else:
                sequence = sequence[truncate_bp_from_ends:-truncate_bp_from_ends]
        
        # Validation 1: Length
        if len(sequence) < MIN_SEQUENCE_LENGTH:
            raise ValueError(f"Sequence length ({len(sequence)}) is less than MIN_SEQUENCE_LENGTH ({MIN_SEQUENCE_LENGTH}) in file: {fasta_path}\nHeader: {header}")
            
        # Validation 2: Characters
        # Check for any characters that are not A, T, C, or G
        if not allow_ambiguous:
            invalid_chars = set(sequence) - set("ATCG")
            if invalid_chars:
                raise ValueError(f"Sequence contains invalid characters {invalid_chars} in file: {fasta_path}\nHeader: {header}")
            
        sequences.append((header, sequence))
    
    with open(fasta_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                # Save previous sequence if exists
                if current_header is not None:
                    validate_and_add_sequence(current_header, current_seq_lines)
                # Start new sequence
                current_header = line
                current_seq_lines = []
            else:
                current_seq_lines.append(line)
        
        # Save last sequence
        if current_header is not None:
            validate_and_add_sequence(current_header, current_seq_lines)
    
    return sequences


def parse_taxonomy_from_header(header):
    """
    Parse taxonomy string from a FASTA header.
    
    Args:
        header: FASTA header line (with or without '>')
        
    Returns:
        List of 7 taxonomy levels [Domain, Phylum, Class, Order, Family, Genus, Species]
    """
    try:
        # Header format: >{ID} d__...;p__...;... [metadata]
        # Remove '>' if present
        header = header.lstrip('>')
        
        # Get taxonomy part (after first space, before '[')
        tax_str = header.split(' ', 1)[1].split('[')[0].strip()
        levels = tax_str.split(';')
        
        # Pad to 7 levels if needed
        levels += [''] * (7 - len(levels))
        return levels[:7]
    except Exception:
        return [''] * 7


def discover_region_files(seqs_dir, use_full_for_test):
    """
    Discover available region FASTA files in the seqs directory.
    
    Args:
        seqs_dir: Path to the seqs directory
        use_full_for_test: Whether to include FULL_seqs.fasta
        
    Returns:
        Dictionary mapping region names to file paths
    """
    region_files = {}
    
    # Find all *_seqs.fasta files
    pattern = os.path.join(seqs_dir, "*_seqs.fasta")
    for fasta_path in glob(pattern):
        filename = os.path.basename(fasta_path)
        
        # Skip failed files
        if filename.startswith("failed_"):
            continue
        
        # Extract region name from filename (e.g., "V4-004" from "V4-004_seqs.fasta")
        region = filename.replace("_seqs.fasta", "")
        
        # Handle FULL file based on setting
        if region == "FULL":
            if use_full_for_test:
                region_files[region] = fasta_path
        else:
            region_files[region] = fasta_path
    
    return region_files


def load_indices(file_paths):
    """
    Load sequence indices from a file.
    
    Args:
        file_paths: List of paths to the indices files (one index per line)
        
    Returns:
        Set of integer indices
    """
    indices = set()
    for file_path in file_paths:
        with open(file_path, 'r') as f:
            indices.update(int(line.strip()) for line in f)
    return indices


# =============================================================================
# Main Functions
# =============================================================================

def load_all_region_data(seqs_dir, use_full_for_test):
    """
    Load sequence data from all region files.
    
    Args:
        seqs_dir: Path to the seqs directory
        use_full_for_test: Whether to include FULL_seqs.fasta for testing
        
    Returns:
        Tuple of (region_data, full_data) where:
        - region_data: Dict mapping region names to list of (header, sequence) tuples
        - full_data: List of (header, sequence) tuples from FULL_seqs.fasta
    """
    # Discover region files for test set
    region_files = discover_region_files(seqs_dir, use_full_for_test)
    print(f"Found {len(region_files)} region files for testing: {list(region_files.keys())}")
    
    # Load region data for test set
    region_data = {}
    for region, fasta_path in region_files.items():
        # Truncate subsequences if configured (but not FULL sequences)
        truncation = TRUNCATE_N_BP_FOR_SUBSEQUENCES if region != "FULL" else 0
        region_data[region] = parse_fasta_file(fasta_path, truncate_bp_from_ends=truncation)
    
    # Load FULL sequences for training
    full_path = os.path.join(seqs_dir, "FULL_seqs.fasta")
    
    # Check if we should allow ambiguous bases in FULL_seqs.fasta
    allow_ambiguous_in_full = ALLOW_AMBIGUOUS_BASES_IN_FULL_SEQS_FOR_TRAINING and not use_full_for_test
    full_data = parse_fasta_file(full_path, allow_ambiguous=allow_ambiguous_in_full)
    
    return region_data, full_data


def select_regions_for_sequences(indices, available_regions, max_regions):
    """
    Select which regions to use for each sequence index.
    
    Args:
        indices: Set of sequence indices
        available_regions: List of available region names
        max_regions: Maximum regions per sequence (None = all)
        
    Returns:
        Dict mapping each index to a list of selected region names
    """
    selection = {}
    
    for idx in indices:
        if max_regions is None or max_regions >= len(available_regions):
            # Use all regions
            selection[idx] = list(available_regions)
        else:
            # Randomly select regions for this sequence
            selection[idx] = random.sample(available_regions, max_regions)
    
    return selection


def select_train_regions_for_m16s(indices, available_regions):
    """
    Select a single random region for each training sequence (for Micro16S training).
    
    Each training sequence gets exactly 1 randomly selected region (excluding FULL).
    This matches the Micro16S classifier's need to train on region-specific sequences.
    
    Args:
        indices: Set of sequence indices
        available_regions: List of available region names (should not include FULL)
        
    Returns:
        Dict mapping each index to a single selected region name
    """
    selection = {}
    
    for idx in indices:
        # Select exactly 1 random region for this sequence
        selection[idx] = random.choice(available_regions)
    
    return selection


def prepare_sequences(region_data, full_data, test_indices, train_indices, 
                      test_region_selection, train_region_selection, output_dir):
    """
    Prepare training and test sequence files and track metadata.
    
    Args:
        region_data: Dict mapping region names to sequence data
        full_data: Sequence data from FULL_seqs.fasta
        test_indices: Set of test sequence indices
        train_indices: Set of training sequence indices
        test_region_selection: Dict mapping test indices to selected regions
        train_region_selection: Dict mapping train indices to selected region (single region per sequence)
        output_dir: Directory to write output files
        
    Returns:
        Tuple of (test_metadata, train_metadata_rdp, train_metadata_m16s) where each is a list of dicts
        containing ASV_ID, Seq_Index, Region, and taxonomy info
    """
    # Output file paths
    train_rdp_path = os.path.join(output_dir, "train_seqs_rdp.fna")
    train_m16s_path = os.path.join(output_dir, "train_seqs_m16s.fna")
    test_path = os.path.join(output_dir, "test_seqs.fna")
    
    test_metadata = []
    train_metadata_rdp = []
    train_metadata_m16s = []
    
    # Write training sequences for RDP (always from FULL)
    with open(train_rdp_path, 'w') as f_train_rdp:
        for idx in sorted(train_indices):
            # Indices from split files are 0-based and map directly to FASTA row order
            if idx < 0 or idx >= len(full_data):
                continue
                
            header, sequence = full_data[idx]
            asv_id = str(idx) + "_FULL"
            taxonomy = parse_taxonomy_from_header(header)
            
            # Write sequence with full header (ASV_ID + original header content)
            original_header_content = header.lstrip('>')
            original_header_content_no_brackets = "".join(original_header_content.split('}')[1:])
            f_train_rdp.write(f">{{{asv_id}}}{original_header_content_no_brackets}\n{sequence}\n")
            
            # Track metadata
            train_metadata_rdp.append({
                'ASV_ID': asv_id,
                'Seq_Index': idx,
                'Region': 'FULL',
                'Taxonomy': taxonomy
            })
    
    # Write training sequences for Micro16S (from selected regions)
    with open(train_m16s_path, 'w') as f_train_m16s:
        for idx in sorted(train_indices):
            selected_region = train_region_selection.get(idx)
            
            if selected_region is None or selected_region not in region_data:
                continue
            # Indices from split files are 0-based and map directly to FASTA row order
            if idx < 0 or idx >= len(region_data[selected_region]):
                continue
                
            header, sequence = region_data[selected_region][idx]
            asv_id = str(idx) + "_" + selected_region
            taxonomy = parse_taxonomy_from_header(header)
            
            # Write sequence with full header (ASV_ID + original header content)
            original_header_content = header.lstrip('>')
            original_header_content_no_brackets = "".join(original_header_content.split('}')[1:])
            f_train_m16s.write(f">{{{asv_id}}}{original_header_content_no_brackets}\n{sequence}\n")
            
            # Track metadata
            train_metadata_m16s.append({
                'ASV_ID': asv_id,
                'Seq_Index': idx,
                'Region': selected_region,
                'Taxonomy': taxonomy
            })
    
    # Write test sequences (from selected regions)
    with open(test_path, 'w') as f_test:
        for idx in sorted(test_indices):
            selected_regions = test_region_selection.get(idx, [])
            
            for region in selected_regions:
                if region not in region_data:
                    continue
                # Indices from split files are 0-based and map directly to FASTA row order
                if idx < 0 or idx >= len(region_data[region]):
                    continue
                    
                header, sequence = region_data[region][idx]
                asv_id = str(idx) + "_" + region
                taxonomy = parse_taxonomy_from_header(header)
                
                # Write sequence with full header (ASV_ID + original header content)
                original_header_content = header.lstrip('>')
                original_header_content_no_brackets = "".join(original_header_content.split('}')[1:])
                f_test.write(f">{{{asv_id}}}{original_header_content_no_brackets}\n{sequence}\n")
                
                # Track metadata
                test_metadata.append({
                    'ASV_ID': asv_id,
                    'Seq_Index': idx,
                    'Region': region,
                    'Taxonomy': taxonomy
                })
    print("Wrote sequences:")
    print(f"  Training (RDP): {len(train_metadata_rdp)}")
    print(f"  Training (Micro16S): {len(train_metadata_m16s)}")
    print(f"  Test: {len(test_metadata)}")
    
    return test_metadata, train_metadata_rdp, train_metadata_m16s


def write_true_classes(test_metadata, output_path):
    """
    Write the true classifications CSV file.
    
    Args:
        test_metadata: List of metadata dicts for test sequences
        output_path: Path to output CSV file
    """
    with open(output_path, 'w') as f:
        # Write header
        f.write("ASV_ID,Seq_Index,Region_Train,Region_Test,Domain,Phylum,Class,Order,Family,Genus,Species\n")
        
        # Write each test sequence
        for meta in test_metadata:
            taxonomy = meta['Taxonomy']
            # Region_Train is "NA" for true classes as specified
            # Region_Test is the sequence's region
            f.write(f"{meta['ASV_ID']},{meta['Seq_Index']},NA,{meta['Region']},{','.join(taxonomy)}\n")


def run_dada2_taxonomy(query_fasta, ref_fasta, output_csv, batch_size):
    """
    Run DADA2 taxonomy classification via R script.
    
    Args:
        query_fasta: Path to query sequences (test)
        ref_fasta: Path to reference sequences (train)
        output_csv: Path to output CSV file
        batch_size: Batch size for assignTaxonomy
        
    Returns:
        True if successful, None if failed
    """
    # Construct the shell command to run the R script
    command = [
        "Rscript", "classify_rdp.R",
        query_fasta,
        ref_fasta,
        output_csv,
        str(batch_size)
    ]
    
    # Run the command
    print(f"Running DADA2 taxonomy assignment with batch size {batch_size}...")
    result = subprocess.run(command, capture_output=True, text=True)
    
    if result.returncode == 0:
        return True
    else:
        print(f"R Error: {result.stderr}")
        return None


# =============================================================================
# Main Execution
# =============================================================================

if __name__ == "__main__":

    print("Starting classification benchmarking...")

    # Setup paths
    seqs_dir = os.path.join(DATABASE_DIR_PATH, "seqs")
    train_idx_path = os.path.join(DATASET_SPLIT_DIR_PATH, "training_indices.txt")
    test_idx_path = os.path.join(DATASET_SPLIT_DIR_PATH, "testing_indices.txt")
    excluded_idx_path = os.path.join(DATASET_SPLIT_DIR_PATH, "excluded_taxa_indices.txt")
    
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR_PATH, exist_ok=True)
    
    # Internal output file paths
    train_rdp_fna_path = os.path.join(OUTPUT_DIR_PATH, "train_seqs_rdp.fna")
    train_m16s_fna_path = os.path.join(OUTPUT_DIR_PATH, "train_seqs_m16s.fna")
    test_fna_path = os.path.join(OUTPUT_DIR_PATH, "test_seqs.fna")
    
    
    # Step 1: Load indices
    test_indices = load_indices([test_idx_path]) if TEST_OR_EXCLUDED_TAXA == "test" else load_indices([excluded_idx_path])
    train_indices = load_indices([train_idx_path, test_idx_path, excluded_idx_path]) if USE_ALL_SEQUENCES_FOR_TRAIN else load_indices([train_idx_path]) 
    
    # Apply ONLY_USE_FIRST_N_SEQUENCES_FOR_TEST for the test set if set (useful for quick testing)
    if ONLY_USE_FIRST_N_SEQUENCES_FOR_TEST is not None:
        test_indices = set(sorted(test_indices)[:ONLY_USE_FIRST_N_SEQUENCES_FOR_TEST])
        print(f"Limited to first {ONLY_USE_FIRST_N_SEQUENCES_FOR_TEST} sequences per split")
    
    print(f"Test indices: {len(test_indices)}")
    print(f"Train indices: {len(train_indices)}")
    
    
    # Step 2: Load all FASTA data
    print("Loading FASTA Data...")
    region_data, full_data = load_all_region_data(seqs_dir, USE_FULL_SEQS_FOR_TEST)
    print(f"Total sequences in reference: {len(full_data)}")
    
    
    # Step 3: Select regions for each test sequence
    print("Selecting Regions...")
    available_regions = list(region_data.keys())
    test_region_selection = select_regions_for_sequences(
        test_indices, available_regions, MAX_REGIONS_PER_SEQUENCE_TEST
    )
    
    total_test_seqs = sum(len(regions) for regions in test_region_selection.values())
    print(f"Available regions: {len(available_regions)}")
    print(f"MAX_REGIONS_PER_SEQUENCE_TEST: {MAX_REGIONS_PER_SEQUENCE_TEST}")
    print(f"Total test sequences to classify: {total_test_seqs}")
    
    # Select regions for training sequences (for Micro16S)
    # Get regions excluding FULL (available_regions already excludes FULL unless USE_FULL_SEQS_FOR_TEST is True)
    train_available_regions = [r for r in available_regions if r != 'FULL']
    train_region_selection = select_train_regions_for_m16s(
        train_indices, train_available_regions
    )
    print(f"Training regions available (excluding FULL): {len(train_available_regions)}")
    
    
    # Step 4: Prepare sequences
    print("Preparing Sequences...")
    test_metadata, train_metadata_rdp, train_metadata_m16s = prepare_sequences(
        region_data, full_data, test_indices, train_indices,
        test_region_selection, train_region_selection, OUTPUT_DIR_PATH
    )
    
    # Step 5: Write true classifications
    print("Writing True Classifications...")
    write_true_classes(test_metadata, TRUE_CLASSES_PATH)
    
    
    # Step 6: Run DADA2 classification
    print("Running Classification...")
    classification_start_time = time.time()
    results = run_dada2_taxonomy(test_fna_path, train_rdp_fna_path, PREDICTED_CLASSES_RDP_PATH, ASSIGN_TAXONOMY_BATCH_SIZE)
    
    if results is not None:
        print("\nClassification completed successfully!")
        print(f"Classification time: {time.time() - classification_start_time} seconds")
        print(f"Predicted classes written to: {PREDICTED_CLASSES_RDP_PATH}")
        print(f"True classes written to: {TRUE_CLASSES_PATH}")
    else:
        print("Classification failed!")
