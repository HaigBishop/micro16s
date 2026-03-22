"""
Classification Benchmarking Analysis Script for 16S rRNA Sequences

This is script 3 out of 3:
1. classify_bench_01_rdp.py
2. classify_bench_02_m16s.py
3. classify_bench_03_analyse.py

Important: You must have already run scripts 1 and 2 to have these files in the classification_eval directory:
    - predicted_classes_rdp.csv
    - predicted_classes_m16s.csv
    - true_classes.csv
    - train_seqs_rdp.fna
    - train_seqs_m16s.fna
    - test_seqs.fna

This script analyzes the classification results from RDP and Micro16S classifiers by:
1. Computing per-rank accuracy and macro-averaged accuracy
2. Analyzing confidence score distributions and calibration
3. Evaluating accuracy by region, taxon size, and other factors
4. Identifying top misclassifications and absent taxa
5. Generating summary statistics and visualizations

Handling of Taxa Absent from Training Set:
    For classification metrics (accuracy, confidence, calibration, etc.), sequences are
    excluded from analysis at any rank where their true taxon is not present in the
    training set. This is because a classifier cannot be expected to correctly classify
    to a taxon it has never seen. However, the SAME sequence may still be included in
    metrics at higher (less specific) ranks where its true taxon IS present in training.
    
    For example, if a test sequence has true Species "X" which is absent from training,
    but its true Genus "Y" is present in training, then:
    - The sequence is excluded from Species-level accuracy/confidence metrics
    - The sequence is included in Genus-level (and higher) accuracy/confidence metrics
    
    The absent taxa analysis outputs (ref_seq_absence_frequency.csv, all_absent_ref_seqs.csv,
    absent_taxa_sequences_plot.png) are specifically designed to investigate these cases
    and are not affected by this filtering.

Outputs (in results/ subdirectory):
    - Multiple CSV files with detailed metrics including:
      - recall_by_{rank}_size.csv: Per-taxon recall (sensitivity) vs training set size
      - precision_by_{rank}_size.csv: Per-taxon precision vs training set size
      - f1_by_{rank}_size.csv: Per-taxon F1 score vs training set size
    - Multiple PNG plots for visualization including:
      - recall_by_taxon_size.png: Combined plot of recall for all ranks
      - precision_by_taxon_size.png: Combined plot of precision for all ranks
      - f1_by_taxon_size.png: Combined plot of F1 for all ranks
    - results_summary.txt with overall summary
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots



# =============================================================================
# Configuration
# =============================================================================

# Input/Output paths
DATASET_SPLIT_DIR_PATH = "/mnt/secondary/micro16s_dbs/16s_databases/m16s_db_004/seqs/split_001"
# EVAL_DIR_PATH = DATASET_SPLIT_DIR_PATH + "/classification_eval_test_validation"
EVAL_DIR_PATH = DATASET_SPLIT_DIR_PATH + "/classification_eval_test_application"
# EVAL_DIR_PATH = DATASET_SPLIT_DIR_PATH + "/classification_eval_excluded_validation"
# EVAL_DIR_PATH = DATASET_SPLIT_DIR_PATH + "/classification_eval_excluded_application"
RESULTS_DIR_PATH = EVAL_DIR_PATH + "/results"

# Input file paths
PREDICTED_CLASSES_RDP_PATH = EVAL_DIR_PATH + "/predicted_classes_rdp.csv"
PREDICTED_CLASSES_M16S_PATH = EVAL_DIR_PATH + "/predicted_classes_m16s.csv"
TRUE_CLASSES_PATH = EVAL_DIR_PATH + "/true_classes.csv"
TRAIN_SEQS_RDP_PATH = EVAL_DIR_PATH + "/train_seqs_rdp.fna"
TRAIN_SEQS_M16S_PATH = EVAL_DIR_PATH + "/train_seqs_m16s.fna"
TEST_SEQS_PATH = EVAL_DIR_PATH + "/test_seqs.fna"

# Options
TAXONOMY_RANKS = ["Domain", "Phylum", "Class", "Order", "Family", "Genus", "Species"]
NUM_BINS_CONFIDENCE_CALIBRATION = 100  # Number of confidence bins (e.g., 10 = 0-10%, 10-20%, etc.)
MIN_NUM_SEQ_PER_BIN_CONFIDENCE_CALIBRATION = 10  # Minimum samples per bin, bins below this are skipped
BINNING_METHOD_CONFIDENCE_CALIBRATION = "percentiles"  # "equal" or "percentiles"
TOP_N_MISCLASSIFICATIONS = 20
TOP_N_TAXA_ACCURACY = 50
MAX_TAXA_PER_HEATMAP = 50


# =============================================================================
# Helper Functions
# =============================================================================

def load_csv_as_df(csv_path):
    """
    Load a CSV file as a pandas DataFrame.
    
    Args:
        csv_path: Path to the CSV file
        
    Returns:
        pandas DataFrame
    """
    return pd.read_csv(csv_path, dtype=str, na_filter=False)


def parse_fasta_headers(fasta_path):
    """
    Parse FASTA file headers to extract ASV_ID and taxonomy.
    
    Args:
        fasta_path: Path to the FASTA file
        
    Returns:
        List of dicts with ASV_ID, Seq_Index, Region, and Taxonomy
    """
    records = []
    
    with open(fasta_path, 'r') as f:
        for line in f:
            if line.startswith(">"):
                header = line.strip().lstrip('>')
                
                # Extract ASV_ID from braces
                if header.startswith("{") and "}" in header:
                    asv_id = header[1:header.index("}")]
                else:
                    asv_id = header.split()[0].strip("{}")
                
                # Parse Seq_Index and Region from ASV_ID
                if "_" in asv_id:
                    seq_index_str, region = asv_id.split("_", 1)
                    try:
                        seq_index = int(seq_index_str)
                    except ValueError:
                        seq_index = None
                else:
                    seq_index = None
                    region = None
                
                # Parse taxonomy from header
                try:
                    if ' ' in header:
                        tax_str = header.split(' ', 1)[1].split('[')[0].strip()
                        taxonomy = tax_str.split(';') if tax_str else []
                        taxonomy += [''] * (7 - len(taxonomy))
                        taxonomy = taxonomy[:7]
                    else:
                        taxonomy = [''] * 7
                except Exception:
                    taxonomy = [''] * 7
                
                records.append({
                    'ASV_ID': asv_id,
                    'Seq_Index': seq_index,
                    'Region': region,
                    'Taxonomy': taxonomy
                })
    
    return records


def get_confidence_columns(method):
    """
    Get confidence column names for a method.
    
    Args:
        method: 'RDP' or 'M16S'
        
    Returns:
        List of column names for confidence scores
    """
    if method == 'RDP':
        return [f"{rank}_Boot" for rank in TAXONOMY_RANKS]
    else:
        return [f"{rank}_Conf" for rank in TAXONOMY_RANKS]


def normalize_confidence(conf_value, method):
    """
    Normalize confidence to [0, 1] range.
    
    Args:
        conf_value: Raw confidence value
        method: 'RDP' or 'M16S'
        
    Returns:
        Normalized confidence in [0, 1]
    """
    try:
        val = float(conf_value)
        if method == 'RDP':
            return val / 100.0  # Bootstrap values are 0-100
        return val  # M16S confidence is already 0-1
    except (ValueError, TypeError):
        return np.nan


def compute_percentiles(values):
    """
    Compute summary statistics for a list of values.
    
    Args:
        values: List or array of numeric values
        
    Returns:
        Dict with min, p5, p25, median, p75, p95, max
    """
    if len(values) == 0:
        return {k: np.nan for k in ['min', 'p5', 'p25', 'median', 'p75', 'p95', 'max']}
    
    arr = np.array(values)
    return {
        'min': float(np.min(arr)),
        'p5': float(np.percentile(arr, 5)),
        'p25': float(np.percentile(arr, 25)),
        'median': float(np.median(arr)),
        'p75': float(np.percentile(arr, 75)),
        'p95': float(np.percentile(arr, 95)),
        'max': float(np.max(arr))
    }


def save_csv(df, path, float_format='%.6f'):
    """
    Save DataFrame to CSV with consistent formatting.
    
    Args:
        df: pandas DataFrame
        path: Output path
        float_format: Format string for floats
    """
    df.to_csv(path, index=False, float_format=float_format)
    print(f"  Saved: {os.path.basename(path)}")


def save_plot(fig, path, dpi=150):
    """
    Save matplotlib figure to file.
    
    Args:
        fig: matplotlib Figure
        path: Output path
        dpi: Resolution
    """
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {os.path.basename(path)}")


def build_train_taxa_sets(train_taxa):
    """
    Build a dictionary mapping each rank to the set of taxa present in training.
    
    Args:
        train_taxa: List of training taxonomy dicts from parse_fasta_headers()
        
    Returns:
        Dict mapping rank name -> set of taxa names present in training
    """
    train_taxa_sets = {rank: set() for rank in TAXONOMY_RANKS}
    
    for record in train_taxa:
        for rank_idx, rank in enumerate(TAXONOMY_RANKS):
            taxon = record['Taxonomy'][rank_idx]
            if taxon:
                train_taxa_sets[rank].add(taxon)
    
    return train_taxa_sets


def add_in_train_columns(merged_df, train_taxa_sets):
    """
    Add boolean columns indicating whether each sequence's true taxon at each rank
    is present in the training set.
    
    Args:
        merged_df: Merged DataFrame with True_{Rank} columns
        train_taxa_sets: Dict from build_train_taxa_sets()
        
    Returns:
        DataFrame with added In_Train_{Rank} columns
    """
    df = merged_df.copy()
    
    for rank in TAXONOMY_RANKS:
        true_col = f"True_{rank}"
        in_train_col = f"In_Train_{rank}"
        # A taxon is "in train" if it's non-empty AND exists in the training set
        df[in_train_col] = (df[true_col] != '') & df[true_col].isin(train_taxa_sets[rank])
    
    return df


# =============================================================================
# Data Loading Functions
# =============================================================================

def load_prediction_data():
    """
    Load all prediction data and merge with true classes.
    
    Returns:
        Tuple of (df_rdp, df_m16s, df_true, merged_df)
    """
    print("Loading prediction data...")
    
    df_rdp = load_csv_as_df(PREDICTED_CLASSES_RDP_PATH)
    df_m16s = load_csv_as_df(PREDICTED_CLASSES_M16S_PATH)
    df_true = load_csv_as_df(TRUE_CLASSES_PATH)
    
    print(f"  RDP predictions: {len(df_rdp)}")
    print(f"  M16S predictions: {len(df_m16s)}")
    print(f"  True classes: {len(df_true)}")
    
    # Rename columns in true_classes for clarity
    true_rename = {rank: f"True_{rank}" for rank in TAXONOMY_RANKS}
    true_rename['Region_Test'] = 'Region_Test_True'
    df_true_renamed = df_true.rename(columns=true_rename)
    
    # Merge RDP predictions with true classes
    rdp_rename = {rank: f"Pred_{rank}_RDP" for rank in TAXONOMY_RANKS}
    for rank in TAXONOMY_RANKS:
        boot_col = f"{rank}_Boot"
        if boot_col in df_rdp.columns:
            rdp_rename[boot_col] = f"Conf_{rank}_RDP"
    df_rdp_renamed = df_rdp.rename(columns=rdp_rename)
    
    # Merge M16S predictions with true classes
    m16s_rename = {rank: f"Pred_{rank}_M16S" for rank in TAXONOMY_RANKS}
    m16s_rename['Region_Train'] = 'Region_Train_M16S'
    for rank in TAXONOMY_RANKS:
        conf_col = f"{rank}_Conf"
        if conf_col in df_m16s.columns:
            m16s_rename[conf_col] = f"Conf_{rank}_M16S"
    df_m16s_renamed = df_m16s.rename(columns=m16s_rename)
    
    # Merge all DataFrames on ASV_ID
    merged = df_true_renamed.merge(
        df_rdp_renamed[['ASV_ID'] + [c for c in df_rdp_renamed.columns if c != 'ASV_ID' and c not in ['Seq_Index', 'Region_Test']]],
        on='ASV_ID', how='left'
    )
    merged = merged.merge(
        df_m16s_renamed[['ASV_ID'] + [c for c in df_m16s_renamed.columns if c != 'ASV_ID' and c not in ['Seq_Index', 'Region_Test']]],
        on='ASV_ID', how='left'
    )
    
    # Rename Region_Test_True back to Region_Test
    merged = merged.rename(columns={'Region_Test_True': 'Region_Test'})
    
    print(f"  Merged dataset: {len(merged)} samples")
    
    return df_rdp, df_m16s, df_true, merged


def load_training_taxonomy():
    """
    Load taxonomy from training FASTA files.
    
    Returns:
        Tuple of (train_taxa_rdp, train_taxa_m16s) as lists of dicts
    """
    print("Loading training taxonomy...")
    
    train_rdp = parse_fasta_headers(TRAIN_SEQS_RDP_PATH)
    train_m16s = parse_fasta_headers(TRAIN_SEQS_M16S_PATH)
    
    print(f"  RDP training sequences: {len(train_rdp)}")
    print(f"  M16S training sequences: {len(train_m16s)}")
    
    return train_rdp, train_m16s


# =============================================================================
# Accuracy Analysis Functions
# =============================================================================

def compute_overall_accuracy(merged_df):
    """
    Compute overall accuracy per rank for both methods.
    
    Only includes sequences whose true taxon at that rank is present in training.
    
    Args:
        merged_df: Merged DataFrame with true and predicted classes (must have In_Train_{Rank} columns)
        
    Returns:
        DataFrame with columns: Rank, Accuracy_RDP, Accuracy_M16S, N_Valid_RDP, N_Valid_M16S
    """
    results = []
    
    for rank in TAXONOMY_RANKS:
        true_col = f"True_{rank}"
        pred_rdp = f"Pred_{rank}_RDP"
        pred_m16s = f"Pred_{rank}_M16S"
        in_train_col = f"In_Train_{rank}"
        
        # Filter valid samples: true taxon in training, both true and predicted present
        valid_rdp = merged_df[merged_df[in_train_col] & (merged_df[pred_rdp] != '')]
        valid_m16s = merged_df[merged_df[in_train_col] & (merged_df[pred_m16s] != '')]
        
        acc_rdp = (valid_rdp[true_col] == valid_rdp[pred_rdp]).mean() if len(valid_rdp) > 0 else np.nan
        acc_m16s = (valid_m16s[true_col] == valid_m16s[pred_m16s]).mean() if len(valid_m16s) > 0 else np.nan
        
        results.append({
            'Rank': rank,
            'Accuracy_RDP': acc_rdp,
            'Accuracy_M16S': acc_m16s,
            'N_Valid_RDP': len(valid_rdp),
            'N_Valid_M16S': len(valid_m16s)
        })
    
    return pd.DataFrame(results)


def compute_macro_accuracy(merged_df):
    """
    Compute macro-averaged accuracy per rank (average accuracy across labels).
    
    Only includes sequences whose true taxon at that rank is present in training.
    
    Args:
        merged_df: Merged DataFrame with true and predicted classes (must have In_Train_{Rank} columns)
        
    Returns:
        DataFrame with columns: Rank, Macro_Accuracy_RDP, Macro_Accuracy_M16S, N_Taxa_RDP, N_Taxa_M16S
    """
    results = []
    
    for rank in TAXONOMY_RANKS:
        true_col = f"True_{rank}"
        pred_rdp = f"Pred_{rank}_RDP"
        pred_m16s = f"Pred_{rank}_M16S"
        in_train_col = f"In_Train_{rank}"
        
        # Only consider sequences whose true taxon is in training
        in_train_df = merged_df[merged_df[in_train_col]]
        
        # Get unique taxa (that are in training)
        unique_taxa = in_train_df[true_col].unique()
        
        accuracies_rdp = []
        accuracies_m16s = []
        
        for taxon in unique_taxa:
            taxon_df = in_train_df[in_train_df[true_col] == taxon]
            
            # RDP accuracy for this taxon
            valid_rdp = taxon_df[taxon_df[pred_rdp] != '']
            if len(valid_rdp) > 0:
                acc = (valid_rdp[true_col] == valid_rdp[pred_rdp]).mean()
                accuracies_rdp.append(acc)
            
            # M16S accuracy for this taxon
            valid_m16s = taxon_df[taxon_df[pred_m16s] != '']
            if len(valid_m16s) > 0:
                acc = (valid_m16s[true_col] == valid_m16s[pred_m16s]).mean()
                accuracies_m16s.append(acc)
        
        macro_rdp = np.mean(accuracies_rdp) if accuracies_rdp else np.nan
        macro_m16s = np.mean(accuracies_m16s) if accuracies_m16s else np.nan
        
        results.append({
            'Rank': rank,
            'Macro_Accuracy_RDP': macro_rdp,
            'Macro_Accuracy_M16S': macro_m16s,
            'N_Taxa_RDP': len(accuracies_rdp),
            'N_Taxa_M16S': len(accuracies_m16s)
        })
    
    return pd.DataFrame(results)


def compute_deepest_correct_rank(merged_df):
    """
    For each test sequence, find the deepest rank where all ranks up to that point are correct.
    
    Only considers ranks where the true taxon is present in training. If a rank's true taxon
    is absent from training, that rank is skipped (not counted as correct or incorrect).
    
    Args:
        merged_df: Merged DataFrame (must have In_Train_{Rank} columns)
        
    Returns:
        DataFrame with columns: Method, Deepest_Rank, N_Count, Percent
    """
    results = []
    total = len(merged_df)
    
    for method in ['RDP', 'M16S']:
        counts = defaultdict(int)
        
        for _, row in merged_df.iterrows():
            deepest = 'None'
            
            for rank in TAXONOMY_RANKS:
                true_col = f"True_{rank}"
                pred_col = f"Pred_{rank}_{method}"
                in_train_col = f"In_Train_{rank}"
                
                true_val = row[true_col]
                pred_val = row[pred_col]
                
                # Skip if either is empty
                if true_val == '' or pred_val == '':
                    break
                
                # Skip ranks where true taxon is not in training
                if not row[in_train_col]:
                    break
                
                # Check if correct at this rank
                if true_val == pred_val:
                    deepest = rank
                else:
                    break
            
            counts[deepest] += 1
        
        # Convert to results format
        for rank in ['None'] + TAXONOMY_RANKS:
            n_count = counts.get(rank, 0)
            results.append({
                'Method': method,
                'Deepest_Rank': rank,
                'N_Count': n_count,
                'Percent': n_count / total if total > 0 else 0.0
            })
    
    return pd.DataFrame(results)


# =============================================================================
# Region Analysis Functions
# =============================================================================

def compute_region_counts(merged_df):
    """
    Compute basic region coverage stats from the test set.
    
    Args:
        merged_df: Merged DataFrame
        
    Returns:
        DataFrame with columns: Region_Test, N_ASVs, N_Seq_Indices
    """
    region_groups = merged_df.groupby('Region_Test')
    
    results = []
    for region, group in region_groups:
        results.append({
            'Region_Test': region,
            'N_ASVs': len(group),
            'N_Seq_Indices': group['Seq_Index'].nunique()
        })
    
    return pd.DataFrame(results).sort_values('N_ASVs', ascending=False)


def compute_accuracy_by_region(merged_df):
    """
    Compute per-region accuracy breakdown for each rank.
    
    Only includes sequences whose true taxon at that rank is present in training.
    
    Args:
        merged_df: Merged DataFrame (must have In_Train_{Rank} columns)
        
    Returns:
        DataFrame with columns: Region_Test, Rank, N_Valid_RDP, N_Valid_M16S, 
                               Accuracy_RDP, Accuracy_M16S, Delta_M16S_minus_RDP
    """
    results = []
    
    for region in merged_df['Region_Test'].unique():
        region_df = merged_df[merged_df['Region_Test'] == region]
        
        for rank in TAXONOMY_RANKS:
            true_col = f"True_{rank}"
            pred_rdp = f"Pred_{rank}_RDP"
            pred_m16s = f"Pred_{rank}_M16S"
            in_train_col = f"In_Train_{rank}"
            
            # Filter: true taxon in training, prediction non-empty
            valid_rdp = region_df[region_df[in_train_col] & (region_df[pred_rdp] != '')]
            valid_m16s = region_df[region_df[in_train_col] & (region_df[pred_m16s] != '')]
            
            n_valid_rdp = len(valid_rdp)
            n_valid_m16s = len(valid_m16s)
            
            acc_rdp = (valid_rdp[true_col] == valid_rdp[pred_rdp]).mean() if n_valid_rdp > 0 else np.nan
            acc_m16s = (valid_m16s[true_col] == valid_m16s[pred_m16s]).mean() if n_valid_m16s > 0 else np.nan
            
            delta = acc_m16s - acc_rdp if not (np.isnan(acc_rdp) or np.isnan(acc_m16s)) else np.nan
            
            results.append({
                'Region_Test': region,
                'Rank': rank,
                'N_Valid_RDP': n_valid_rdp,
                'N_Valid_M16S': n_valid_m16s,
                'Accuracy_RDP': acc_rdp,
                'Accuracy_M16S': acc_m16s,
                'Delta_M16S_minus_RDP': delta
            })
    
    return pd.DataFrame(results)


def compute_accuracy_by_train_region_m16s(merged_df):
    """
    Compute Micro16S accuracy by training region only.
    
    Only includes sequences whose true taxon at that rank is present in training.
    
    Args:
        merged_df: Merged DataFrame (must have In_Train_{Rank} columns)
        
    Returns:
        DataFrame with columns: Region_Train, Rank, N_Valid_M16S, Accuracy_M16S
    """
    results = []
    
    if 'Region_Train_M16S' not in merged_df.columns:
        return pd.DataFrame(columns=['Region_Train', 'Rank', 'N_Valid_M16S', 'Accuracy_M16S'])
    
    for region in merged_df['Region_Train_M16S'].unique():
        if pd.isna(region) or region == '':
            continue
        region_df = merged_df[merged_df['Region_Train_M16S'] == region]
        
        for rank in TAXONOMY_RANKS:
            true_col = f"True_{rank}"
            pred_col = f"Pred_{rank}_M16S"
            in_train_col = f"In_Train_{rank}"
            
            # Filter: true taxon in training, prediction non-empty
            valid = region_df[region_df[in_train_col] & (region_df[pred_col] != '')]
            n_valid = len(valid)
            acc = (valid[true_col] == valid[pred_col]).mean() if n_valid > 0 else np.nan
            
            results.append({
                'Region_Train': region,
                'Rank': rank,
                'N_Valid_M16S': n_valid,
                'Accuracy_M16S': acc
            })
    
    return pd.DataFrame(results)


def compute_accuracy_by_region_pair_m16s(merged_df):
    """
    Compute Micro16S accuracy by train/test region pairing.
    
    Only includes sequences whose true taxon at that rank is present in training.
    
    Args:
        merged_df: Merged DataFrame (must have In_Train_{Rank} columns)
        
    Returns:
        DataFrame with columns: Region_Train, Region_Test, Rank, N_Valid_M16S, Accuracy_M16S
    """
    results = []
    
    if 'Region_Train_M16S' not in merged_df.columns:
        return pd.DataFrame(columns=['Region_Train', 'Region_Test', 'Rank', 'N_Valid_M16S', 'Accuracy_M16S'])
    
    for (region_train, region_test), group in merged_df.groupby(['Region_Train_M16S', 'Region_Test']):
        if pd.isna(region_train) or region_train == '':
            continue
            
        for rank in TAXONOMY_RANKS:
            true_col = f"True_{rank}"
            pred_col = f"Pred_{rank}_M16S"
            in_train_col = f"In_Train_{rank}"
            
            # Filter: true taxon in training, prediction non-empty
            valid = group[group[in_train_col] & (group[pred_col] != '')]
            n_valid = len(valid)
            acc = (valid[true_col] == valid[pred_col]).mean() if n_valid > 0 else np.nan
            
            results.append({
                'Region_Train': region_train,
                'Region_Test': region_test,
                'Rank': rank,
                'N_Valid_M16S': n_valid,
                'Accuracy_M16S': acc
            })
    
    return pd.DataFrame(results)


# =============================================================================
# Confidence Analysis Functions
# =============================================================================

def compute_confidence_distributions(merged_df, method):
    """
    Compute confidence score distributions for correct and incorrect classifications.
    
    Only includes sequences whose true taxon at that rank is present in training.
    
    Args:
        merged_df: Merged DataFrame (must have In_Train_{Rank} columns)
        method: 'RDP' or 'M16S'
        
    Returns:
        DataFrame with confidence percentiles for positive and negative classifications
    """
    results = []
    
    for rank in TAXONOMY_RANKS:
        true_col = f"True_{rank}"
        pred_col = f"Pred_{rank}_{method}"
        conf_col = f"Conf_{rank}_{method}"
        in_train_col = f"In_Train_{rank}"
        
        if conf_col not in merged_df.columns:
            continue
        
        # Get valid samples: true taxon in training, prediction non-empty
        valid = merged_df[merged_df[in_train_col] & (merged_df[pred_col] != '')]
        
        # Separate correct and incorrect
        correct_mask = valid[true_col] == valid[pred_col]
        
        # Normalize confidence values
        conf_values = valid[conf_col].apply(lambda x: normalize_confidence(x, method))
        
        conf_pos = conf_values[correct_mask].dropna().values
        conf_neg = conf_values[~correct_mask].dropna().values
        
        stats_pos = compute_percentiles(conf_pos)
        stats_neg = compute_percentiles(conf_neg)
        
        result = {'Rank': rank}
        for key in ['min', 'p5', 'p25', 'median', 'p75', 'p95', 'max']:
            result[f'{key.capitalize()}_Conf_Pos'] = stats_pos[key]
            result[f'{key.capitalize()}_Conf_Neg'] = stats_neg[key]
        
        results.append(result)
    
    return pd.DataFrame(results)


def compute_confidence_calibration(merged_df):
    """
    Compute reliability table: bin confidences and compute observed accuracy per bin.
    
    Only includes sequences whose true taxon at that rank is present in training.
    Bins with fewer than MIN_NUM_SEQ_PER_BIN_CONFIDENCE_CALIBRATION samples are skipped
    with a warning printed.
    
    Args:
        merged_df: Merged DataFrame (must have In_Train_{Rank} columns)
        
    Returns:
        DataFrame with columns: Method, Rank, Conf_Bin_Low, Conf_Bin_High, N_Samples, Mean_Conf, Accuracy
    """
    results = []
    
    # Generate fixed confidence bins for "equal" method
    # For "percentiles", bins are generated dynamically per rank/method
    fixed_bins = None
    if BINNING_METHOD_CONFIDENCE_CALIBRATION == "equal":
        bin_width = 1.0 / NUM_BINS_CONFIDENCE_CALIBRATION
        fixed_bins = [(i * bin_width, (i + 1) * bin_width) for i in range(NUM_BINS_CONFIDENCE_CALIBRATION)]
    
    for method in ['RDP', 'M16S']:
        for rank in TAXONOMY_RANKS:
            true_col = f"True_{rank}"
            pred_col = f"Pred_{rank}_{method}"
            conf_col = f"Conf_{rank}_{method}"
            in_train_col = f"In_Train_{rank}"
            
            if conf_col not in merged_df.columns:
                continue
            
            # Filter: true taxon in training, prediction non-empty
            valid = merged_df[merged_df[in_train_col] & (merged_df[pred_col] != '')].copy()
            if len(valid) == 0:
                continue
            
            # Normalize confidence
            valid['conf_norm'] = valid[conf_col].apply(lambda x: normalize_confidence(x, method))
            valid = valid.dropna(subset=['conf_norm'])
            
            if len(valid) == 0:
                continue

            # Define bins/chunks based on method
            chunks_to_process = []
            
            if BINNING_METHOD_CONFIDENCE_CALIBRATION == "equal":
                # Use fixed bins
                for bin_low, bin_high in fixed_bins:
                    bin_mask = (valid['conf_norm'] >= bin_low) & (valid['conf_norm'] < bin_high)
                    if bin_high == 1.0:  # Include 1.0 in last bin
                        bin_mask = (valid['conf_norm'] >= bin_low) & (valid['conf_norm'] <= bin_high)
                    
                    bin_df = valid[bin_mask]
                    chunks_to_process.append({
                        'df': bin_df,
                        'low': bin_low,
                        'high': bin_high
                    })
            
            elif BINNING_METHOD_CONFIDENCE_CALIBRATION == "percentiles":
                # Split into N equal-sized chunks
                valid_sorted = valid.sort_values('conf_norm')
                if len(valid_sorted) < NUM_BINS_CONFIDENCE_CALIBRATION:
                    # Not enough samples to split into desired bins
                    # Just treat as one single bin (or fewer splits)
                    splits = [valid_sorted]
                else:
                    # Avoid numpy calling DataFrame.swapaxes (deprecated in pandas)
                    index_splits = np.array_split(np.arange(len(valid_sorted)), NUM_BINS_CONFIDENCE_CALIBRATION)
                    splits = [valid_sorted.iloc[idx] for idx in index_splits if len(idx) > 0]
                
                for chunk in splits:
                    if len(chunk) == 0:
                        continue
                    chunks_to_process.append({
                        'df': chunk,
                        'low': chunk['conf_norm'].min(),
                        'high': chunk['conf_norm'].max()
                    })

            # Process chunks
            for item in chunks_to_process:
                bin_df = item['df']
                bin_low = item['low']
                bin_high = item['high']
                
                n_samples = len(bin_df)
                
                # Skip bins below minimum sample threshold
                if n_samples < MIN_NUM_SEQ_PER_BIN_CONFIDENCE_CALIBRATION:
                    if BINNING_METHOD_CONFIDENCE_CALIBRATION == "equal":
                        print(f"  Warning: Skipping bin [{bin_low:.0%}-{bin_high:.0%}] for {method} {rank}: "
                              f"only {n_samples} samples (min={MIN_NUM_SEQ_PER_BIN_CONFIDENCE_CALIBRATION})")
                    continue
                
                mean_conf = bin_df['conf_norm'].mean()
                accuracy = (bin_df[true_col] == bin_df[pred_col]).mean()
                
                results.append({
                    'Method': method,
                    'Rank': rank,
                    'Conf_Bin_Low': bin_low,
                    'Conf_Bin_High': bin_high,
                    'N_Samples': n_samples,
                    'Mean_Conf': mean_conf,
                    'Accuracy': accuracy
                })
    
    return pd.DataFrame(results)


def compute_confidence_correlation(merged_df):
    """
    Compute correlation between confidence and correctness.
    
    Only includes sequences whose true taxon at that rank is present in training.
    
    Args:
        merged_df: Merged DataFrame (must have In_Train_{Rank} columns)
        
    Returns:
        DataFrame with columns: Method, Rank, N_Samples, Pearson_r, Spearman_r
    """
    from scipy import stats as scipy_stats
    
    results = []
    
    for method in ['RDP', 'M16S']:
        for rank in TAXONOMY_RANKS:
            true_col = f"True_{rank}"
            pred_col = f"Pred_{rank}_{method}"
            conf_col = f"Conf_{rank}_{method}"
            in_train_col = f"In_Train_{rank}"
            
            if conf_col not in merged_df.columns:
                continue
            
            # Filter: true taxon in training, prediction non-empty
            valid = merged_df[merged_df[in_train_col] & (merged_df[pred_col] != '')].copy()
            if len(valid) < 3:
                continue
            
            # Normalize confidence
            valid['conf_norm'] = valid[conf_col].apply(lambda x: normalize_confidence(x, method))
            valid = valid.dropna(subset=['conf_norm'])
            
            # Compute correctness (1=correct, 0=incorrect)
            valid['correct'] = (valid[true_col] == valid[pred_col]).astype(int)
            
            n_samples = len(valid)
            
            # Compute correlations
            pearson_r = np.nan
            pearson_p = np.nan
            try:
                pearson_r, pearson_p = scipy_stats.pearsonr(valid['conf_norm'], valid['correct'])
            except Exception:
                pass
            
            spearman_r = np.nan
            spearman_p = np.nan
            try:
                spearman_r, spearman_p = scipy_stats.spearmanr(valid['conf_norm'], valid['correct'])
            except Exception:
                pass
            
            results.append({
                'Method': method,
                'Rank': rank,
                'N_Samples': n_samples,
                'Pearson_r': pearson_r,
                'Pearson_p': pearson_p,
                'Spearman_r': spearman_r,
                'Spearman_p': spearman_p
            })
    
    return pd.DataFrame(results)


# =============================================================================
# Misclassification Analysis Functions
# =============================================================================

def compute_top_misclassifications(merged_df):
    """
    Find top confusion pairs per rank.
    
    Only includes sequences whose true taxon at that rank is present in training.
    
    Args:
        merged_df: Merged DataFrame (must have In_Train_{Rank} columns)
        
    Returns:
        DataFrame with columns: Method, Rank, True_Taxon, Pred_Taxon, N_Count, Percent
    """
    results = []
    
    for method in ['RDP', 'M16S']:
        for rank in TAXONOMY_RANKS:
            true_col = f"True_{rank}"
            pred_col = f"Pred_{rank}_{method}"
            in_train_col = f"In_Train_{rank}"
            
            # Get misclassified samples: true taxon in training, prediction non-empty
            valid = merged_df[merged_df[in_train_col] & (merged_df[pred_col] != '')]
            misclassified = valid[valid[true_col] != valid[pred_col]]
            
            if len(misclassified) == 0:
                continue
            
            # Count confusion pairs
            confusion_counts = misclassified.groupby([true_col, pred_col]).size().reset_index(name='count')
            confusion_counts = confusion_counts.sort_values('count', ascending=False).head(TOP_N_MISCLASSIFICATIONS)
            
            total_misclassified = len(misclassified)
            
            for _, row in confusion_counts.iterrows():
                results.append({
                    'Method': method,
                    'Rank': rank,
                    'True_Taxon': row[true_col],
                    'Pred_Taxon': row[pred_col],
                    'N_Count': row['count'],
                    'Percent': row['count'] / total_misclassified
                })
    
    return pd.DataFrame(results)


def compute_taxon_accuracy_top(merged_df):
    """
    Compute per-taxon accuracy for the top-N most frequent taxa.
    
    Only includes sequences whose true taxon at that rank is present in training.
    
    Args:
        merged_df: Merged DataFrame (must have In_Train_{Rank} columns)
        
    Returns:
        DataFrame with columns: Method, Rank, Taxon, N_Samples, Accuracy
    """
    results = []
    
    for method in ['RDP', 'M16S']:
        for rank in TAXONOMY_RANKS:
            true_col = f"True_{rank}"
            pred_col = f"Pred_{rank}_{method}"
            in_train_col = f"In_Train_{rank}"
            
            # Filter: true taxon in training, prediction non-empty
            valid = merged_df[merged_df[in_train_col] & (merged_df[pred_col] != '')]
            
            # Get top taxa by frequency
            taxon_counts = valid[true_col].value_counts()
            top_taxa = taxon_counts.head(TOP_N_TAXA_ACCURACY).index.tolist()
            
            for taxon in top_taxa:
                taxon_df = valid[valid[true_col] == taxon]
                n_samples = len(taxon_df)
                accuracy = (taxon_df[true_col] == taxon_df[pred_col]).mean()
                
                results.append({
                    'Method': method,
                    'Rank': rank,
                    'Taxon': taxon,
                    'N_Samples': n_samples,
                    'Accuracy': accuracy
                })
    
    return pd.DataFrame(results)


# =============================================================================
# Taxon Size Analysis Functions
# =============================================================================

def compute_taxon_metrics_by_size(merged_df, train_taxa, rank):
    """
    Compute recall, precision, and F1 for each taxon vs number of train sequences.
    
    Only includes taxa that are present in the training set (Train_Count > 0).
    
    Definitions:
    - Recall (Sensitivity): TP / (TP + FN) = correct predictions / total true samples for taxon
    - Precision: TP / (TP + FP) = correct predictions / total predicted as taxon
    - F1: 2 * (Precision * Recall) / (Precision + Recall)
    
    Args:
        merged_df: Merged DataFrame (must have In_Train_{Rank} columns)
        train_taxa: List of training taxonomy dicts
        rank: Taxonomy rank to analyze
        
    Returns:
        DataFrame with columns: Taxon, Train_Count, Test_Count,
                               Recall_RDP, Recall_M16S,
                               Precision_RDP, Precision_M16S,
                               F1_RDP, F1_M16S
    """
    rank_idx = TAXONOMY_RANKS.index(rank)
    in_train_col = f"In_Train_{rank}"
    true_col = f"True_{rank}"
    pred_rdp = f"Pred_{rank}_RDP"
    pred_m16s = f"Pred_{rank}_M16S"
    
    # Count training sequences per taxon
    train_counts = defaultdict(int)
    for record in train_taxa:
        taxon = record['Taxonomy'][rank_idx]
        if taxon:
            train_counts[taxon] += 1
    
    # Only consider sequences whose true taxon is in training
    in_train_df = merged_df[merged_df[in_train_col]]
    
    # Get unique taxa (that are in training)
    unique_taxa = in_train_df[true_col].unique()
    
    # Count predictions per taxon and correct predictions for precision calculation
    # For RDP
    pred_counts_rdp = defaultdict(int)
    pred_correct_rdp = defaultdict(int)
    for _, row in in_train_df.iterrows():
        if row[pred_rdp] != '':
            pred_counts_rdp[row[pred_rdp]] += 1
            if row[true_col] == row[pred_rdp]:
                pred_correct_rdp[row[pred_rdp]] += 1
    
    # For M16S
    pred_counts_m16s = defaultdict(int)
    pred_correct_m16s = defaultdict(int)
    for _, row in in_train_df.iterrows():
        if row[pred_m16s] != '':
            pred_counts_m16s[row[pred_m16s]] += 1
            if row[true_col] == row[pred_m16s]:
                pred_correct_m16s[row[pred_m16s]] += 1
    
    results = []
    
    for taxon in unique_taxa:
        taxon_df = in_train_df[in_train_df[true_col] == taxon]
        test_count = len(taxon_df)
        train_count = train_counts.get(taxon, 0)
        
        # ----- RDP metrics -----
        # Recall counts no-calls as false negatives by using all true samples in the denominator.
        valid_rdp = taxon_df[taxon_df[pred_rdp] != '']
        tp_rdp = (valid_rdp[true_col] == valid_rdp[pred_rdp]).sum()
        
        # Recall: TP / (TP + FN), with FN including incorrect predictions and no-calls.
        recall_denom_rdp = len(taxon_df)
        recall_rdp = tp_rdp / recall_denom_rdp if recall_denom_rdp > 0 else np.nan
        
        # Precision: TP / (TP + FP) where FP = predicted as taxon but wrong
        precision_denom_rdp = pred_counts_rdp.get(taxon, 0)
        precision_rdp = pred_correct_rdp.get(taxon, 0) / precision_denom_rdp if precision_denom_rdp > 0 else np.nan
        
        # F1
        if np.isnan(recall_rdp) or np.isnan(precision_rdp) or (recall_rdp + precision_rdp) == 0:
            f1_rdp = np.nan
        else:
            f1_rdp = 2 * precision_rdp * recall_rdp / (precision_rdp + recall_rdp)
        
        # ----- M16S metrics -----
        valid_m16s = taxon_df[taxon_df[pred_m16s] != '']
        tp_m16s = (valid_m16s[true_col] == valid_m16s[pred_m16s]).sum()
        
        # Recall: TP / (TP + FN), with FN including incorrect predictions and no-calls.
        recall_denom_m16s = len(taxon_df)
        recall_m16s = tp_m16s / recall_denom_m16s if recall_denom_m16s > 0 else np.nan
        
        # Precision
        precision_denom_m16s = pred_counts_m16s.get(taxon, 0)
        precision_m16s = pred_correct_m16s.get(taxon, 0) / precision_denom_m16s if precision_denom_m16s > 0 else np.nan
        
        # F1
        if np.isnan(recall_m16s) or np.isnan(precision_m16s) or (recall_m16s + precision_m16s) == 0:
            f1_m16s = np.nan
        else:
            f1_m16s = 2 * precision_m16s * recall_m16s / (precision_m16s + recall_m16s)
        
        results.append({
            'Taxon': taxon,
            'Train_Count': train_count,
            'Test_Count': test_count,
            'Recall_RDP': recall_rdp,
            'Recall_M16S': recall_m16s,
            'Precision_RDP': precision_rdp,
            'Precision_M16S': precision_m16s,
            'F1_RDP': f1_rdp,
            'F1_M16S': f1_m16s
        })

    output_columns = [
        'Taxon',
        'Train_Count',
        'Test_Count',
        'Recall_RDP',
        'Recall_M16S',
        'Precision_RDP',
        'Precision_M16S',
        'F1_RDP',
        'F1_M16S'
    ]
    
    if not results:
        # Can happen in excluded-taxa evaluations when no test taxa at this rank
        # are present in training (e.g., family/genus/species).
        return pd.DataFrame(columns=output_columns)
    
    return pd.DataFrame(results, columns=output_columns).sort_values('Train_Count', ascending=False)


# =============================================================================
# Absent Taxa Analysis Functions
# =============================================================================

def compute_ref_seq_absence_frequency(merged_df, train_taxa):
    """
    Count test sequences whose true taxon is absent from the training set.
    
    Args:
        merged_df: Merged DataFrame
        train_taxa: List of training taxonomy dicts
        
    Returns:
        DataFrame with columns: Rank, N_Test_Seqs_Absent, Percent_Test_Seqs_Absent, N_Taxa_Absent
    """
    results = []
    total_test = len(merged_df)
    
    for rank_idx, rank in enumerate(TAXONOMY_RANKS):
        true_col = f"True_{rank}"
        
        # Get taxa present in training set
        train_taxa_set = set()
        for record in train_taxa:
            taxon = record['Taxonomy'][rank_idx]
            if taxon:
                train_taxa_set.add(taxon)
        
        # Find test sequences with absent taxa
        valid_test = merged_df[merged_df[true_col] != '']
        absent_mask = ~valid_test[true_col].isin(train_taxa_set)
        
        n_absent_seqs = absent_mask.sum()
        absent_taxa = valid_test[absent_mask][true_col].unique()
        
        results.append({
            'Rank': rank,
            'N_Test_Seqs_Absent': n_absent_seqs,
            'Percent_Test_Seqs_Absent': n_absent_seqs / total_test if total_test > 0 else 0.0,
            'N_Taxa_Absent': len(absent_taxa)
        })
    
    return pd.DataFrame(results)


def compute_all_absent_ref_seqs(merged_df, train_taxa):
    """
    List taxa observed in the test set but absent from the training set.
    
    Args:
        merged_df: Merged DataFrame
        train_taxa: List of training taxonomy dicts
        
    Returns:
        DataFrame with columns: Rank, Taxon, Num_Test_Seqs
    """
    results = []
    
    for rank_idx, rank in enumerate(TAXONOMY_RANKS):
        true_col = f"True_{rank}"
        
        # Get taxa present in training set
        train_taxa_set = set()
        for record in train_taxa:
            taxon = record['Taxonomy'][rank_idx]
            if taxon:
                train_taxa_set.add(taxon)
        
        # Find absent taxa and their counts
        valid_test = merged_df[merged_df[true_col] != '']
        absent_mask = ~valid_test[true_col].isin(train_taxa_set)
        absent_df = valid_test[absent_mask]
        
        taxon_counts = absent_df[true_col].value_counts()
        
        for taxon, count in taxon_counts.items():
            results.append({
                'Rank': rank,
                'Taxon': taxon,
                'Num_Test_Seqs': count
            })
    
    return pd.DataFrame(results)


# =============================================================================
# Plotting Functions
# =============================================================================

def plot_accuracy_bars(accuracy_df, output_path, title='Accuracy by Rank'):
    """
    Create bar plot comparing RDP and M16S accuracy per rank.
    
    Args:
        accuracy_df: DataFrame with Rank, Accuracy_RDP, Accuracy_M16S columns
        output_path: Path to save the plot
        title: Plot title
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(accuracy_df))
    width = 0.35
    
    bars_rdp = ax.bar(x - width/2, accuracy_df['Accuracy_RDP'], width, label='RDP', color='red', alpha=0.7)
    bars_m16s = ax.bar(x + width/2, accuracy_df['Accuracy_M16S'], width, label='Micro16S', color='blue', alpha=0.7)
    
    ax.set_xlabel('Rank')
    ax.set_ylabel('Accuracy')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(accuracy_df['Rank'], rotation=45, ha='right')
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    save_plot(fig, output_path)


def plot_macro_accuracy_bars(macro_df, output_path):
    """
    Create bar plot comparing macro-averaged accuracy per rank.
    
    Args:
        macro_df: DataFrame with Rank, Macro_Accuracy_RDP, Macro_Accuracy_M16S columns
        output_path: Path to save the plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(macro_df))
    width = 0.35
    
    bars_rdp = ax.bar(x - width/2, macro_df['Macro_Accuracy_RDP'], width, label='RDP', color='red', alpha=0.7)
    bars_m16s = ax.bar(x + width/2, macro_df['Macro_Accuracy_M16S'], width, label='Micro16S', color='blue', alpha=0.7)
    
    ax.set_xlabel('Rank')
    ax.set_ylabel('Macro Accuracy')
    ax.set_title('Macro-Averaged Accuracy by Rank')
    ax.set_xticks(x)
    ax.set_xticklabels(macro_df['Rank'], rotation=45, ha='right')
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    save_plot(fig, output_path)


def plot_accuracy_delta(accuracy_df, output_path):
    """
    Create bar plot of Delta_M16S_minus_RDP per rank.

    Args:
        accuracy_df: DataFrame with Rank, Accuracy_RDP, Accuracy_M16S columns
        output_path: Path to save the plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    delta = accuracy_df['Accuracy_M16S'] - accuracy_df['Accuracy_RDP']
    colors = ['green' if d >= 0 else 'red' for d in delta]
    
    x = np.arange(len(accuracy_df))
    ax.bar(x, delta, color=colors, alpha=0.7)
    
    ax.set_xlabel('Rank')
    ax.set_ylabel('Accuracy Delta (M16S - RDP)')
    ax.set_title('Accuracy Difference: Micro16S vs RDP')
    ax.set_xticks(x)
    ax.set_xticklabels(accuracy_df['Rank'], rotation=45, ha='right')
    
    # Make y-axis symmetric around 0
    max_abs = max(abs(delta.min()), abs(delta.max())) * 1.1
    ax.set_ylim(-max_abs, max_abs)
    
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    save_plot(fig, output_path)


def plot_deepest_correct_rank(deepest_df, output_path):
    """
    Create side-by-side bar plots showing the deepest correct rank per method.

    Args:
        deepest_df: DataFrame with Method, Deepest_Rank, Percent columns
        output_path: Path to save the plot
    """
    methods = ['RDP', 'M16S']
    colors = {'RDP': 'red', 'M16S': 'blue'}
    order = ['None'] + TAXONOMY_RANKS

    fig, axes = plt.subplots(1, len(methods), figsize=(12, 5), sharey=True)
    if len(methods) == 1:
        axes = [axes]

    for ax, method in zip(axes, methods):
        subset = deepest_df[deepest_df['Method'] == method].set_index('Deepest_Rank')
        subset = subset.reindex(order).reset_index()
        percents = subset['Percent'].fillna(0) * 100

        ax.bar(order, percents, color=colors[method], alpha=0.7)
        ax.set_title(method)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(order, rotation=45, ha='right')
        ax.set_xlabel('Deepest Rank')
        ax.set_ylim(0, 100)
        ax.grid(axis='y', alpha=0.3)

    axes[0].set_ylabel('% Test samples')

    plt.tight_layout()
    save_plot(fig, output_path)


def plot_accuracy_by_region_heatmap_combined(accuracy_by_region_df, output_path):
    """
    Create a combined heatmap of per-region accuracy for all ranks.
    
    Generates a grid with methods on rows and ranks on columns.
    Domain is excluded to focus on lower ranks.
    
    Args:
        accuracy_by_region_df: DataFrame with Region_Test, Rank, Accuracy_RDP, Accuracy_M16S
        output_path: Path to save the plot
    """
    plot_ranks = [rank for rank in TAXONOMY_RANKS if rank != 'Domain']
    n_ranks = len(plot_ranks)
    methods = [('RDP', 'Reds'), ('M16S', 'Blues')]
    n_methods = len(methods)
    
    # Determine regions to plot (consistent across all subplots)
    all_regions = sorted(accuracy_by_region_df['Region_Test'].unique())
    if len(all_regions) > MAX_TAXA_PER_HEATMAP:
        np.random.seed(42)
        regions_to_plot = sorted(np.random.choice(all_regions, MAX_TAXA_PER_HEATMAP, replace=False))
    else:
        regions_to_plot = all_regions
        
    if not regions_to_plot:
        return
        
    fig, axes = plt.subplots(n_methods, n_ranks, figsize=(3.5 * n_ranks, max(16, len(regions_to_plot) * 0.5)))
    
    # Ensure axes is 2D even if n_ranks = 1
    if n_ranks == 1:
        axes = axes.reshape(n_methods, 1)
        
    for rank_idx, rank in enumerate(plot_ranks):
        rank_df = accuracy_by_region_df[accuracy_by_region_df['Rank'] == rank]
        
        for m_idx, (method, cmap) in enumerate(methods):
            ax = axes[m_idx, rank_idx]
            col = f'Accuracy_{method}'
            
            # Pivot for this method and rank
            if len(rank_df) > 0:
                pivot_df = rank_df.pivot_table(index='Region_Test', values=col, aggfunc='mean')
                # Reindex to ensure consistent regions across all plots
                pivot_df = pivot_df.reindex(regions_to_plot)
                vals = pivot_df.values.reshape(-1, 1)
            else:
                vals = np.full((len(regions_to_plot), 1), np.nan)
            
            im = ax.imshow(vals, cmap=cmap, aspect='auto', vmin=0, vmax=1)
            
            # Y-labels only on the first column
            if rank_idx == 0:
                ax.set_yticks(range(len(regions_to_plot)))
                ax.set_yticklabels(regions_to_plot, fontsize=7)
                ax.set_ylabel('Region', fontsize=9)
            else:
                ax.set_yticks([])
                
            ax.set_xticks([])
            ax.set_title(f'{method} - {rank}', fontsize=9)
            
            # Colorbar only on the last column for each method
            if rank_idx == n_ranks - 1:
                plt.colorbar(im, ax=ax, label='Accuracy')
                
    plt.tight_layout()
    save_plot(fig, output_path)


def plot_confidence_violin(merged_df, output_path):
    """
    Create violin plots of confidence distributions for correct vs incorrect classifications.
    
    Only includes sequences whose true taxon at that rank is present in training.
    
    Args:
        merged_df: Merged DataFrame (must have In_Train_{Rank} columns)
        output_path: Path to save the plot
    """
    fig, axes = plt.subplots(2, len(TAXONOMY_RANKS), figsize=(20, 8))
    
    for method_idx, method in enumerate(['RDP', 'M16S']):
        for rank_idx, rank in enumerate(TAXONOMY_RANKS):
            ax = axes[method_idx, rank_idx]
            
            true_col = f"True_{rank}"
            pred_col = f"Pred_{rank}_{method}"
            conf_col = f"Conf_{rank}_{method}"
            in_train_col = f"In_Train_{rank}"
            
            if conf_col not in merged_df.columns:
                ax.set_visible(False)
                continue
            
            # Filter: true taxon in training, prediction non-empty
            valid = merged_df[merged_df[in_train_col] & (merged_df[pred_col] != '')].copy()
            if len(valid) == 0:
                ax.set_visible(False)
                continue
            
            valid['conf_norm'] = valid[conf_col].apply(lambda x: normalize_confidence(x, method))
            valid = valid.dropna(subset=['conf_norm'])
            
            correct_mask = valid[true_col] == valid[pred_col]
            
            conf_correct = valid[correct_mask]['conf_norm'].values
            conf_incorrect = valid[~correct_mask]['conf_norm'].values
            
            data_to_plot = []
            labels = []
            if len(conf_correct) > 0:
                data_to_plot.append(conf_correct)
                labels.append('Correct')
            if len(conf_incorrect) > 0:
                data_to_plot.append(conf_incorrect)
                labels.append('Incorrect')
            
            if len(data_to_plot) > 0:
                parts = ax.violinplot(data_to_plot, showmeans=True, showmedians=True)
                ax.set_xticks(range(1, len(labels) + 1))
                ax.set_xticklabels(labels, fontsize=8)
            
            ax.set_ylim(0, 1)
            ax.set_title(f'{method} - {rank}', fontsize=9)
            
            if rank_idx == 0:
                ax.set_ylabel('Confidence')
    
    plt.tight_layout()
    save_plot(fig, output_path)


def plot_confidence_calibration(calibration_df, output_path):
    """
    Create reliability curves of accuracy vs confidence for all ranks.
    
    Plots all ranks in a 2-row grid with RDP (red) and M16S (blue) lines per rank.
    X-axis shows confidence bin centers, y-axis shows accuracy.
    
    Args:
        calibration_df: DataFrame with calibration data
        output_path: Path to save the plot
    """
    n_ranks = len(TAXONOMY_RANKS)
    n_cols = (n_ranks + 1) // 2  # 2 rows, compute columns needed
    fig, axes = plt.subplots(2, n_cols, figsize=(4 * n_cols, 8))
    axes = axes.flatten()  # Flatten for easy indexing
    
    for rank_idx, rank in enumerate(TAXONOMY_RANKS):
        ax = axes[rank_idx]
        
        for method, color in [('RDP', 'red'), ('M16S', 'blue')]:
            method_rank_df = calibration_df[(calibration_df['Method'] == method) & (calibration_df['Rank'] == rank)]
            
            if len(method_rank_df) == 0:
                continue
            
            # Sort by bin for proper line plot
            method_rank_df = method_rank_df.sort_values('Conf_Bin_Low')
            
            # Use bin center as x-axis position
            bin_centers = (method_rank_df['Conf_Bin_Low'] + method_rank_df['Conf_Bin_High']) / 2
            
            ax.plot(bin_centers, method_rank_df['Accuracy'], 
                   'o-', label=method, color=color, alpha=0.7, linewidth=1.5, markersize=4)
        
        # Add perfect calibration line
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect')
        
        ax.set_xlabel('Confidence')
        ax.set_ylabel('Accuracy')
        ax.set_title(rank)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    
    # Hide any unused subplots
    for idx in range(n_ranks, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    save_plot(fig, output_path)


def plot_metric_by_taxon_size(taxon_size_df, rank, metric_name, output_path):
    """
    Create scatter plots of a metric vs train count for a specific rank.
    
    Args:
        taxon_size_df: DataFrame with Taxon, Train_Count, and {metric_name}_RDP, {metric_name}_M16S
        rank: Taxonomy rank being plotted
        metric_name: Name of metric (e.g., 'Recall', 'Precision', 'F1')
        output_path: Path to save the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    for idx, (method, color, ax) in enumerate([('RDP', 'red', axes[0]), ('M16S', 'blue', axes[1])]):
        col = f'{metric_name}_{method}'
        valid = taxon_size_df.dropna(subset=[col]).copy()
        valid['Train_Count'] = pd.to_numeric(valid['Train_Count'], errors='coerce')
        valid[col] = pd.to_numeric(valid[col], errors='coerce')
        valid = valid.dropna(subset=['Train_Count', col])
        valid = valid[valid['Train_Count'] > 0]
        
        ax.scatter(valid['Train_Count'], valid[col], c=color, alpha=0.5, s=20)
        
        ax.set_xlabel('Training Count')
        ax.set_ylabel(metric_name)
        ax.set_title(f'{method} - {rank}')
        ax.set_ylim(0, 1.05)
        if len(valid) > 0:
            ax.set_xscale('log')
        else:
            ax.text(0.5, 0.5, 'No positive train counts', ha='center', va='center',
                    transform=ax.transAxes, fontsize=8, alpha=0.7)
        ax.grid(True, which="both", ls="-", alpha=0.3)
    
    plt.tight_layout()
    save_plot(fig, output_path)


def plot_metric_by_taxon_size_combined(taxon_metrics_by_rank, metric_name, output_path):
    """
    Create combined scatter plots of a metric vs train count for all ranks.
    
    Generates a 2-row grid with methods on rows and ranks on columns.
    Domain is excluded to keep focus on lower ranks.
    
    Args:
        taxon_metrics_by_rank: Dict mapping rank -> DataFrame with Taxon, Train_Count, 
                               and {metric_name}_RDP, {metric_name}_M16S
        metric_name: Name of metric (e.g., 'Recall', 'Precision', 'F1')
        output_path: Path to save the plot
    """
    plot_ranks = [rank for rank in TAXONOMY_RANKS if rank != 'Domain']
    n_ranks = len(plot_ranks)
    fig, axes = plt.subplots(2, n_ranks, figsize=(3.5 * n_ranks, 7))
    
    # Handle case where axes is 1D
    if n_ranks == 1:
        axes = axes.reshape(2, 1)
    
    for rank_idx, rank in enumerate(plot_ranks):
        taxon_size_df = taxon_metrics_by_rank[rank]
        
        for method_idx, (method, color) in enumerate([('RDP', 'red'), ('M16S', 'blue')]):
            ax = axes[method_idx, rank_idx]
            col = f'{metric_name}_{method}'
            valid = taxon_size_df.dropna(subset=[col]).copy()
            valid['Train_Count'] = pd.to_numeric(valid['Train_Count'], errors='coerce')
            valid[col] = pd.to_numeric(valid[col], errors='coerce')
            valid = valid.dropna(subset=['Train_Count', col])
            valid = valid[valid['Train_Count'] > 0]
            
            ax.scatter(valid['Train_Count'], valid[col], c=color, alpha=0.5, s=20)
            
            ax.set_ylabel(metric_name)
            ax.set_title(f'{method} - {rank}')
            ax.set_ylim(0, 1.05)
            if len(valid) > 0:
                ax.set_xscale('log')
            else:
                ax.text(0.5, 0.5, 'No positive train counts', ha='center', va='center',
                        transform=ax.transAxes, fontsize=7, alpha=0.7)
            ax.grid(True, which="both", ls="-", alpha=0.3)
            
            # Only show x-axis label on bottom row
            if method_idx == 1:
                ax.set_xlabel('Training Count')
    
    plt.tight_layout()
    save_plot(fig, output_path)


def plot_absent_taxa_sequences(absence_df, output_path):
    """
    Create bar plot of number of test sequences whose taxon is absent from training.
    
    Args:
        absence_df: DataFrame with Rank, N_Test_Seqs_Absent
        output_path: Path to save the plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(absence_df))
    ax.bar(x, absence_df['N_Test_Seqs_Absent'], color='orange', alpha=0.7)
    
    ax.set_xlabel('Rank')
    ax.set_ylabel('Number of Test Sequences')
    ax.set_title('Test Sequences with Taxa Absent from Training Set')
    ax.set_xticks(x)
    ax.set_xticklabels(absence_df['Rank'], rotation=45, ha='right')
    ax.set_ylim(bottom=0)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    save_plot(fig, output_path)


# =============================================================================
# Summary Functions
# =============================================================================

def write_summary(results_dir, merged_df, accuracy_df, macro_df, train_taxa):
    """
    Write a summary text file with key statistics.
    
    Args:
        results_dir: Output directory
        merged_df: Merged DataFrame (must have In_Train_{Rank} columns)
        accuracy_df: Overall accuracy DataFrame
        macro_df: Macro accuracy DataFrame
        train_taxa: List of training taxonomy dicts
    """
    summary_path = os.path.join(results_dir, 'results_summary.txt')
    
    with open(summary_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("Classification Benchmarking Analysis Summary\n")
        f.write("=" * 80 + "\n\n")
        
        # Timestamp and paths
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"EVAL_DIR_PATH: {EVAL_DIR_PATH}\n\n")
        
        # Note about filtering
        f.write("NOTE: Accuracy and confidence metrics EXCLUDE sequences whose true taxon\n")
        f.write("at that rank is absent from the training set. A sequence may be included\n")
        f.write("at higher ranks where its taxon IS present. See all_absent_ref_seqs.csv\n")
        f.write("and ref_seq_absence_frequency.csv for details on absent taxa.\n\n")
        
        # Input files
        f.write("Input Files:\n")
        input_files = [
            PREDICTED_CLASSES_RDP_PATH,
            PREDICTED_CLASSES_M16S_PATH,
            TRUE_CLASSES_PATH,
            TRAIN_SEQS_RDP_PATH,
            TRAIN_SEQS_M16S_PATH,
            TEST_SEQS_PATH
        ]
        for fpath in input_files:
            exists = "EXISTS" if os.path.exists(fpath) else "MISSING"
            f.write(f"  - {os.path.basename(fpath)}: {exists}\n")
        f.write("\n")
        
        # Dataset statistics
        f.write("Dataset Statistics:\n")
        f.write(f"  Total test ASVs: {len(merged_df)}\n")
        f.write(f"  Unique Seq_Index count: {merged_df['Seq_Index'].nunique()}\n")
        f.write(f"  Number of test regions: {merged_df['Region_Test'].nunique()}\n")
        f.write(f"  Training sequences: {len(train_taxa)}\n")
        f.write("\n")
        
        # Regions
        f.write("Test Regions:\n")
        region_counts = merged_df['Region_Test'].value_counts()
        for region, count in region_counts.head(10).items():
            f.write(f"  - {region}: {count} ASVs\n")
        if len(region_counts) > 10:
            f.write(f"  ... and {len(region_counts) - 10} more regions\n")
        f.write("\n")
        
        # Overall accuracy (with sample counts)
        f.write("Overall Accuracy (excluding sequences with absent taxa at each rank):\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Rank':<12} {'RDP':>12} {'M16S':>12} {'Delta':>12} {'N_RDP':>10} {'N_M16S':>10}\n")
        f.write("-" * 80 + "\n")
        for _, row in accuracy_df.iterrows():
            delta = row['Accuracy_M16S'] - row['Accuracy_RDP']
            f.write(f"{row['Rank']:<12} {row['Accuracy_RDP']:>12.4f} {row['Accuracy_M16S']:>12.4f} {delta:>+12.4f} {row['N_Valid_RDP']:>10} {row['N_Valid_M16S']:>10}\n")
        f.write("\n")
        
        # Macro accuracy
        f.write("Macro-Averaged Accuracy (excluding sequences with absent taxa at each rank):\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Rank':<12} {'RDP':>12} {'M16S':>12} {'Delta':>12} {'N_Taxa_RDP':>10} {'N_Taxa_M16S':>12}\n")
        f.write("-" * 80 + "\n")
        for _, row in macro_df.iterrows():
            delta = row['Macro_Accuracy_M16S'] - row['Macro_Accuracy_RDP']
            f.write(f"{row['Rank']:<12} {row['Macro_Accuracy_RDP']:>12.4f} {row['Macro_Accuracy_M16S']:>12.4f} {delta:>+12.4f} {row['N_Taxa_RDP']:>10} {row['N_Taxa_M16S']:>12}\n")
        f.write("\n")
        
        f.write("=" * 80 + "\n")
        f.write("End of Summary\n")
        f.write("=" * 80 + "\n")
    
    print(f"  Saved: results_summary.txt")


# =============================================================================
# Main Execution
# =============================================================================

if __name__ == "__main__":
    
    print("=" * 60)
    print("Classification Benchmarking Analysis")
    print("=" * 60)
    
    # Create results directory
    os.makedirs(RESULTS_DIR_PATH, exist_ok=True)
    print(f"Results directory: {RESULTS_DIR_PATH}\n")
    
    # Load data
    df_rdp, df_m16s, df_true, merged_df = load_prediction_data()
    train_taxa_rdp, train_taxa_m16s = load_training_taxonomy()
    
    # Use M16S training data for taxon size analysis (same underlying sequences for RDP and M16S)
    train_taxa = train_taxa_m16s
    
    # Build training taxa sets and add filtering columns to merged_df
    # These columns indicate whether each sequence's true taxon at each rank is in training
    print("\nBuilding training taxa sets...")
    train_taxa_sets = build_train_taxa_sets(train_taxa)
    for rank in TAXONOMY_RANKS:
        print(f"  {rank}: {len(train_taxa_sets[rank])} unique taxa in training")
    
    merged_df = add_in_train_columns(merged_df, train_taxa_sets)
    
    print("\n" + "=" * 60)
    print("Computing Metrics...")
    print("=" * 60 + "\n")
    
    # ----- Accuracy Tables -----
    print("Accuracy metrics...")
    accuracy_df = compute_overall_accuracy(merged_df)
    save_csv(accuracy_df, os.path.join(RESULTS_DIR_PATH, 'accuracy.csv'))
    
    macro_df = compute_macro_accuracy(merged_df)
    save_csv(macro_df, os.path.join(RESULTS_DIR_PATH, 'accuracy_macro.csv'))
    
    deepest_df = compute_deepest_correct_rank(merged_df)
    save_csv(deepest_df, os.path.join(RESULTS_DIR_PATH, 'deepest_correct_rank.csv'))
    
    # ----- Region Tables -----
    print("\nRegion metrics...")
    region_counts_df = compute_region_counts(merged_df)
    save_csv(region_counts_df, os.path.join(RESULTS_DIR_PATH, 'region_counts.csv'))
    
    accuracy_by_region_df = compute_accuracy_by_region(merged_df)
    save_csv(accuracy_by_region_df, os.path.join(RESULTS_DIR_PATH, 'accuracy_by_region.csv'))
    
    accuracy_by_train_region_df = compute_accuracy_by_train_region_m16s(merged_df)
    save_csv(accuracy_by_train_region_df, os.path.join(RESULTS_DIR_PATH, 'accuracy_by_train_region_m16s.csv'))
    
    accuracy_by_region_pair_df = compute_accuracy_by_region_pair_m16s(merged_df)
    save_csv(accuracy_by_region_pair_df, os.path.join(RESULTS_DIR_PATH, 'accuracy_by_region_pair_m16s.csv'))
    
    # ----- Confidence Tables -----
    print("\nConfidence metrics...")
    conf_rdp_df = compute_confidence_distributions(merged_df, 'RDP')
    save_csv(conf_rdp_df, os.path.join(RESULTS_DIR_PATH, 'confidences_rdp.csv'))
    
    conf_m16s_df = compute_confidence_distributions(merged_df, 'M16S')
    save_csv(conf_m16s_df, os.path.join(RESULTS_DIR_PATH, 'confidences_m16s.csv'))
    
    calibration_df = compute_confidence_calibration(merged_df)
    save_csv(calibration_df, os.path.join(RESULTS_DIR_PATH, 'confidence_calibration.csv'))
    
    correlation_df = compute_confidence_correlation(merged_df)
    save_csv(correlation_df, os.path.join(RESULTS_DIR_PATH, 'confidence_correlation.csv'))
    
    # ----- Misclassification Tables -----
    print("\nMisclassification metrics...")
    misclass_df = compute_top_misclassifications(merged_df)
    save_csv(misclass_df, os.path.join(RESULTS_DIR_PATH, 'top_misclassifications.csv'))
    
    taxon_accuracy_df = compute_taxon_accuracy_top(merged_df)
    save_csv(taxon_accuracy_df, os.path.join(RESULTS_DIR_PATH, 'taxon_accuracy_top.csv'))
    
    # ----- Taxon Size Metrics (Recall, Precision, F1) -----
    print("\nTaxon size metrics (recall, precision, F1)...")
    taxon_metrics_by_rank = {}
    for rank in TAXONOMY_RANKS:
        taxon_metrics_df = compute_taxon_metrics_by_size(merged_df, train_taxa, rank)
        taxon_metrics_by_rank[rank] = taxon_metrics_df
        
        # Save recall (also known as sensitivity)
        recall_df = taxon_metrics_df[['Taxon', 'Train_Count', 'Test_Count', 'Recall_RDP', 'Recall_M16S']].copy()
        save_csv(recall_df, os.path.join(RESULTS_DIR_PATH, f'recall_by_{rank.lower()}_size.csv'))
        
        # Save precision
        precision_df = taxon_metrics_df[['Taxon', 'Train_Count', 'Test_Count', 'Precision_RDP', 'Precision_M16S']].copy()
        save_csv(precision_df, os.path.join(RESULTS_DIR_PATH, f'precision_by_{rank.lower()}_size.csv'))
        
        # Save F1
        f1_df = taxon_metrics_df[['Taxon', 'Train_Count', 'Test_Count', 'F1_RDP', 'F1_M16S']].copy()
        save_csv(f1_df, os.path.join(RESULTS_DIR_PATH, f'f1_by_{rank.lower()}_size.csv'))
    
    # ----- Absent Taxa Tables -----
    print("\nAbsent taxa metrics...")
    absence_freq_df = compute_ref_seq_absence_frequency(merged_df, train_taxa)
    save_csv(absence_freq_df, os.path.join(RESULTS_DIR_PATH, 'ref_seq_absence_frequency.csv'))
    
    absent_taxa_df = compute_all_absent_ref_seqs(merged_df, train_taxa)
    save_csv(absent_taxa_df, os.path.join(RESULTS_DIR_PATH, 'all_absent_ref_seqs.csv'))
    
    print("\n" + "=" * 60)
    print("Generating Plots...")
    print("=" * 60 + "\n")
    
    # ----- Accuracy Plots -----
    print("Accuracy plots...")
    plot_accuracy_bars(accuracy_df, os.path.join(RESULTS_DIR_PATH, 'accuracy_plot.png'))
    plot_macro_accuracy_bars(macro_df, os.path.join(RESULTS_DIR_PATH, 'macro_accuracy_plot.png'))
    plot_accuracy_delta(accuracy_df, os.path.join(RESULTS_DIR_PATH, 'accuracy_delta_plot.png'))
    plot_deepest_correct_rank(deepest_df, os.path.join(RESULTS_DIR_PATH, 'deepest_correct_rank.png'))

    # ----- Region Heatmaps -----
    print("\nRegion heatmap (combined)...")
    plot_accuracy_by_region_heatmap_combined(
        accuracy_by_region_df,
        os.path.join(RESULTS_DIR_PATH, 'accuracy_by_region_heatmap_per_rank.png')
    )
    
    # ----- Confidence Plots -----
    print("\nConfidence plots...")
    plot_confidence_violin(merged_df, os.path.join(RESULTS_DIR_PATH, 'confidence_violin.png'))
    plot_confidence_calibration(calibration_df, os.path.join(RESULTS_DIR_PATH, 'confidence_calibration_plot.png'))
    
    # ----- Taxon Size Plots (Recall, Precision, F1) -----
    print("\nTaxon size plots (recall, precision, F1)...")
    
    # Recall combined plot
    plot_metric_by_taxon_size_combined(
        taxon_metrics_by_rank, 'Recall',
        os.path.join(RESULTS_DIR_PATH, 'recall_by_taxon_size.png')
    )
    
    # Precision combined plot
    plot_metric_by_taxon_size_combined(
        taxon_metrics_by_rank, 'Precision',
        os.path.join(RESULTS_DIR_PATH, 'precision_by_taxon_size.png')
    )
    
    # F1 combined plot
    plot_metric_by_taxon_size_combined(
        taxon_metrics_by_rank, 'F1',
        os.path.join(RESULTS_DIR_PATH, 'f1_by_taxon_size.png')
    )
    
    # ----- Absent Taxa Plot -----
    print("\nAbsent taxa plot...")
    plot_absent_taxa_sequences(absence_freq_df, os.path.join(RESULTS_DIR_PATH, 'absent_taxa_sequences_plot.png'))
    
    print("\n" + "=" * 60)
    print("Writing Summary...")
    print("=" * 60 + "\n")
    
    write_summary(RESULTS_DIR_PATH, merged_df, accuracy_df, macro_df, train_taxa)
    
    print("\n" + "=" * 60)
    print("Analysis Complete!")
    print("=" * 60)
    print(f"\nResults saved to: {RESULTS_DIR_PATH}")
