"""
Classification Benchmarking Script for 16S rRNA Sequences for Micro16S

This is script 2 out of 3:
1. classify_bench_01_rdp.py
2. classify_bench_02_m16s.py
3. classify_bench_03_analyse.py

Important: You must have already run classify_bench_01_rdp.py to have these files in the classification_eval directory:
 - test_seqs.fna
 - train_seqs_rdp.fna
 - train_seqs_m16s.fna
 - true_classes.csv

Inputs:
 - A Micro16S model checkpoint (e.g. model_001.pth)
 - A set of test sequences (test_seqs.fna)
 - A set of train sequences for the Micro16S model (train_seqs_m16s.fna)

Outputs:
 - predicted_classes_m16s.csv
 - about_m16s_classification.txt

Classification Algorithms:
This script supports two classification algorithms, selected by CLASSIFIER_TYPE:

1. Weighted k-NN ("knn"):
   - Finds K nearest neighbors per rank (K varies by rank to handle sparse lower ranks)
   - Optional taxon size weighting to handle class imbalance (plain, sqrt, or log)
   - Optional distance weighting so closer neighbors have more influence
   - Confidence = weighted votes for winner / total weighted votes, compounded rank-by-rank

2. Prototypical Networks ("pnet"):
   - Computes centroid for each taxon (mean of L2-normalised embeddings, re-normalised)
   - Tracks resultant length R to measure cluster tightness
   - Optional taxon size factor to penalise taxa with few examples
   - Optional centroid quality factor to penalise scattered taxa
   - Confidence via softmax over similarities, compounded rank-by-rank

Both algorithms classify rank-by-rank (domain first, then phylum within that domain, etc.)
to ensure taxonomic consistency. Confidence scores are compounded so that lower ranks
inherit uncertainty from higher ranks.

Output Table Columns:
 - ASV_ID, Seq_Index, Region_Train, Region_Test
 - Domain, Phylum, Class, Order, Family, Genus, Species
 - Domain_Conf, Phylum_Conf, Class_Conf, Order_Conf, Family_Conf, Genus_Conf, Species_Conf

Output Table Example Snippet:
```csv
ASV_ID,Seq_Index,Region_Train,Region_Test,Domain,Phylum,Class,Order,Family,Genus,Species,Domain_Conf,Phylum_Conf,Class_Conf,Order_Conf,Family_Conf,Genus_Conf,Species_Conf
3_V3-V5-001,3,V3-001,V3-V5-001,d__Bacteria,p__Pseudomonadota,c__Gammaproteobacteria,o__Enterobacterales_A,f__Enterobacteriaceae_A,g__Wigglesworthia,s__Wigglesworthia glossinidia_A,0.935,0.935,0.935,0.935,0.935,0.935,0.935
3_V4-001,3,V3-001,V4-001,d__Bacteria,p__Pseudomonadota,c__Gammaproteobacteria,o__Enterobacterales_A,f__Enterobacteriaceae_A,g__Wigglesworthia,s__Wigglesworthia glossinidia_A,0.821,0.821,0.821,0.821,0.821,0.821,0.821
3_V4-002,3,V1-V3-002,V4-002,d__Bacteria,p__Pseudomonadota,c__Gammaproteobacteria,o__Enterobacterales_A,f__Enterobacteriaceae_A,g__Wigglesworthia,s__Wigglesworthia glossinidia_A,0.999,0.999,0.999,0.999,0.999,0.999,0.346
```
"""

import os
import csv
import numpy as np
import torch
import time
from collections import Counter
from model import load_micro16s_model

# =============================================================================
# Configuration
# =============================================================================

# Input/Output paths
# MODEL_PATH = "/home/haig/Repos/micro16s/models/m16s_001/ckpts/m16s_001_16000_batches.pth" # Validation model
MODEL_PATH = "/home/haig/Repos/micro16s/models/m16s_002/ckpts/m16s_002_16000_batches.pth" # Application model
DATABASE_DIR_PATH = "/mnt/secondary/micro16s_dbs/16s_databases/m16s_db_004"
DATASET_SPLIT_DIR_PATH = "/mnt/secondary/micro16s_dbs/16s_databases/m16s_db_004/seqs/split_001"
# EVAL_DIR_PATH = DATASET_SPLIT_DIR_PATH + "/classification_eval_test_validation"
# EVAL_DIR_PATH = DATASET_SPLIT_DIR_PATH + "/classification_eval_test_application"
EVAL_DIR_PATH = DATASET_SPLIT_DIR_PATH + "/classification_eval_excluded_validation"
EVAL_DIR_PATH = DATASET_SPLIT_DIR_PATH + "/classification_eval_excluded_application"
TRAIN_SEQS_PATH = EVAL_DIR_PATH + "/train_seqs_m16s.fna"
TEST_SEQS_PATH = EVAL_DIR_PATH + "/test_seqs.fna"
TRUE_CLASSES_PATH = EVAL_DIR_PATH + "/true_classes.csv"
PREDICTED_CLASSES_M16S_PATH = EVAL_DIR_PATH + "/predicted_classes_m16s.csv"
ABOUT_FILE_PATH = EVAL_DIR_PATH + "/about_m16s_classification.txt"

# General options
BATCH_SIZE = 100

# Classifier type: "knn" or "pnet"
# "knn" = Weighted k-NN: votes from K nearest neighbors, weighted by distance and taxon size
# "pnet" = Prototypical Networks: classify by similarity to taxon centroids
CLASSIFIER_TYPE = "knn"

# -----------------------------------------------------------------------------
# k-NN Configuration
# -----------------------------------------------------------------------------

# Number of nearest neighbors to consider per rank [domain, phylum, class, order, family, genus, species]
# Higher K at domain/phylum where there are many examples, lower K at genus/species where taxa are sparse
K_PER_RANK = [75, 50, 10, 10, 7, 5, 3]

# Taxon size weighting: weight votes by inverse of taxon frequency to handle class imbalance
# Without this, common taxa dominate; with it, rare taxa get a fair chance
USE_TAXON_SIZE_WEIGHTING = True

# How to compute taxon size weight: "plain" = 1/n, "sqrt" = 1/sqrt(n), "log" = 1/log(n+1)
# "sqrt" is a good middle ground - penalises large taxa but not as aggressively as "plain"
TAXON_SIZE_WEIGHTING_TYPE = "sqrt"

# Distance weighting: closer neighbors have more influence than distant ones
# Without this, all K neighbors vote equally regardless of distance
USE_DISTANCE_WEIGHTING = True

# Exponent for distance weighting: weight = 1/(dist^exp + epsilon)
# Higher values make "closeness" drop off faster; 1 is a reasonable starting point
DISTANCE_WEIGHTING_EXPONENT = 2

# Small constant to prevent division by zero for exact matches
DISTANCE_WEIGHTING_EPSILON = 1e-6

# -----------------------------------------------------------------------------
# Prototypical Networks Configuration
# -----------------------------------------------------------------------------

# Temperature parameter for softmax: lower = sharper (more confident), higher = softer
# Controls how much difference in similarity translates to difference in probability
# No effect on prediction, only on confidence scores.
KAPPA_TEMPERATURE = 0.02

# Taxon size factor: penalise taxa with few examples (risky to trust a centroid from 1 sample)
# Factor = n/(n+alpha), so small n gives low factor, large n approaches 1
USE_TAXON_SIZE_FACTOR = False
TAXON_SIZE_FACTOR_ALPHA = 1

# Centroid quality factor: penalise taxa whose embeddings are scattered (low resultant length R)
# Factor = R/(n+alpha), where R is the resultant length and n is taxon size
# Tight clusters (R close to n) get factor ~1, scattered clusters get lower factor
USE_CENTROID_QUALITY_FACTOR = False
CENTROID_QUALITY_FACTOR_ALPHA = 1


# =============================================================================
# Helper Functions
# =============================================================================

def parse_fasta_file(fasta_path, allow_ambiguous=False):
    """
    Parse a FASTA file and return a list of (header, sequence) tuples.
    
    Args:
        fasta_path: Path to the FASTA file
        allow_ambiguous: Whether to allow ambiguous bases (non-ATCG)
        
    Returns:
        List of (header, sequence) tuples, where header includes the '>'
    """
    sequences = []
    current_header = None
    current_seq_lines = []

    def validate_and_add_sequence(header, seq_lines):
        sequence = "".join(seq_lines).upper()
        if len(sequence) == 0:
            raise ValueError(f"Empty sequence in file: {fasta_path}\nHeader: {header}")
        if not allow_ambiguous:
            invalid_chars = set(sequence) - set("ATCG")
            if invalid_chars:
                raise ValueError(f"Sequence contains invalid characters {invalid_chars} in file: {fasta_path}\nHeader: {header}")
        sequences.append((header, sequence))
    
    with open(fasta_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if current_header is not None:
                    validate_and_add_sequence(current_header, current_seq_lines)
                current_header = line
                current_seq_lines = []
            else:
                current_seq_lines.append(line)
        
        if current_header is not None:
            validate_and_add_sequence(current_header, current_seq_lines)
    
    return sequences


def parse_asv_id_from_header(header):
    """
    Extract ASV_ID from a FASTA header.
    
    Expected format: >{ASV_ID} ...
    
    Returns:
        ASV_ID string (without braces)
    """
    header = header.lstrip('>').strip()
    if header.startswith("{") and "}" in header:
        return header[1:header.index("}")]
    # Fallback: take first token and strip braces
    return header.split()[0].strip("{}")


def parse_seq_index_and_region(asv_id):
    """
    Parse Seq_Index and Region from an ASV_ID.
    
    ASV_ID format: "{index}_{region}"
    """
    if "_" not in asv_id:
        return None, None
    seq_index_str, region = asv_id.split("_", 1)
    try:
        seq_index = int(seq_index_str)
    except ValueError:
        seq_index = None
    return seq_index, region


def parse_taxonomy_from_header(header):
    """
    Parse taxonomy string from a FASTA header.
    
    Args:
        header: FASTA header line (with or without '>')
        
    Returns:
        List of 7 taxonomy levels [Domain, Phylum, Class, Order, Family, Genus, Species]
    """
    try:
        header = header.lstrip('>')
        if ' ' not in header:
            return [''] * 7
        tax_str = header.split(' ', 1)[1].split('[')[0].strip()
        levels = tax_str.split(';') if tax_str else []
        levels += [''] * (7 - len(levels))
        return levels[:7]
    except Exception:
        return [''] * 7


def load_fasta_with_metadata(fasta_path, allow_ambiguous=False):
    """
    Load sequences from a FASTA file and parse metadata from headers.
    
    Returns:
        sequences: List of DNA strings
        metadata: List of dicts with ASV_ID, Seq_Index, Region, Taxonomy
    """
    records = parse_fasta_file(fasta_path, allow_ambiguous=allow_ambiguous)
    sequences = []
    metadata = []
    
    for header, sequence in records:
        asv_id = parse_asv_id_from_header(header)
        seq_index, region = parse_seq_index_and_region(asv_id)
        taxonomy = parse_taxonomy_from_header(header)
        
        sequences.append(sequence)
        metadata.append({
            'ASV_ID': asv_id,
            'Seq_Index': seq_index,
            'Region': region,
            'Taxonomy': taxonomy
        })
    
    return sequences, metadata


def run_inference_on_sequences(model, sequences, batch_size):
    """
    Run inference on a list of DNA strings using Micro16S.forward().
    
    Returns:
        Numpy array of embeddings with shape (n_sequences, embed_dims)
    """
    if not sequences:
        return np.empty((0, model.embed_dims), dtype=np.float32)
    
    embeddings = []
    model_device = next(model.parameters()).device
    model.eval()
    inference_context = torch.inference_mode() if hasattr(torch, 'inference_mode') else torch.no_grad()
    
    with inference_context:
        for start in range(0, len(sequences), batch_size):
            end = min(start + batch_size, len(sequences))
            batch = sequences[start:end]
            
            # Explicitly encode DNA strings first so we can move tensors onto the model device.
            # This keeps inference on CUDA when a GPU model is used.
            batch_encoded = model.encode_sequences_3bit(batch)
            if model_device.type == 'cuda':
                batch_encoded = batch_encoded.pin_memory().to(model_device, non_blocking=True)
            else:
                batch_encoded = batch_encoded.to(model_device)
            
            out = model(batch_encoded)
            out = out.detach().cpu()
            embeddings.append(out)
    
    return torch.cat(embeddings, dim=0).numpy()


def compute_taxon_size_weight(taxon_count, weighting_type):
    """
    Compute weight based on taxon size to handle class imbalance.
    Rarer taxa get higher weights.
    """
    if weighting_type == "plain":
        return 1.0 / taxon_count
    elif weighting_type == "sqrt":
        return 1.0 / np.sqrt(taxon_count)
    elif weighting_type == "log":
        return 1.0 / np.log(taxon_count + 1)
    else:
        raise ValueError(f"Unknown taxon size weighting type: {weighting_type}")


def get_active_indices_for_prefix(train_taxa, prefix, prefix_indices_cache):
    """
    Get training indices that match a taxonomy prefix, with caching.
    
    Example:
        prefix = ("d__Bacteria", "p__Bacillota")
        returns indices where Domain and Phylum match that prefix
    """
    if prefix in prefix_indices_cache:
        return prefix_indices_cache[prefix]
    
    if len(prefix) == 0:
        indices = np.arange(len(train_taxa), dtype=np.int64)
    else:
        parent_indices = get_active_indices_for_prefix(train_taxa, prefix[:-1], prefix_indices_cache)
        rank_idx = len(prefix) - 1
        indices = parent_indices[train_taxa[parent_indices, rank_idx] == prefix[-1]]
    
    prefix_indices_cache[prefix] = indices
    return indices


def classify_knn(test_embedding, train_embeddings, train_taxa, train_regions, prefix_indices_cache, taxon_counts_cache, full_sims=None):
    """
    Classify a single test sequence using weighted k-NN.
    
    Classifies rank-by-rank (domain first, then phylum within that domain, etc.)
    to ensure taxonomic consistency. Confidence is compounded across ranks.
    
    Returns:
        tuple (predicted_taxonomy, confidence_scores, best_train_region)
    """
    pred_taxonomy = []
    conf_scores = []

    # Compute similarities once for this test sequence and reuse across all ranks.
    if full_sims is None:
        full_sims = np.dot(train_embeddings, test_embedding)
    
    # Start with no prefix (all training sequences)
    prefix = ()
    compounded_conf = 1.0
    best_train_idx = None
    
    for rank_idx in range(7):
        # Get active training data for this rank
        active_indices = get_active_indices_for_prefix(train_taxa, prefix, prefix_indices_cache)
        active_taxa_at_rank = train_taxa[active_indices, rank_idx]
        
        # Reuse precomputed similarities (embeddings are L2-normalised, so dot product = cosine sim)
        dists = 1.0 - full_sims[active_indices]
        
        # Get up to K nearest neighbors
        k = K_PER_RANK[rank_idx] if rank_idx < len(K_PER_RANK) else K_PER_RANK[-1]
        k = min(k, len(active_indices))
        if k == len(active_indices):
            nn_indices_local = np.argsort(dists)
        else:
            # Partial sort is much faster than full sort when K is small.
            nn_indices_local = np.argpartition(dists, k - 1)[:k]
            nn_indices_local = nn_indices_local[np.argsort(dists[nn_indices_local])]
        
        # Count taxon frequencies in training set for this rank (for taxon size weighting)
        counts_key = (rank_idx, prefix)
        if counts_key in taxon_counts_cache:
            taxon_counts = taxon_counts_cache[counts_key]
        else:
            taxon_counts = Counter(active_taxa_at_rank)
            taxon_counts_cache[counts_key] = taxon_counts
        
        # Collect weighted votes
        votes = {}  # taxon -> total weight
        for local_idx in nn_indices_local:
            taxon = active_taxa_at_rank[local_idx]
            dist = dists[local_idx]
            
            weight = 1.0
            
            # Apply taxon size weighting
            if USE_TAXON_SIZE_WEIGHTING:
                weight *= compute_taxon_size_weight(taxon_counts[taxon], TAXON_SIZE_WEIGHTING_TYPE)
            
            # Apply distance weighting
            if USE_DISTANCE_WEIGHTING:
                weight *= 1.0 / (dist ** DISTANCE_WEIGHTING_EXPONENT + DISTANCE_WEIGHTING_EPSILON)
            
            votes[taxon] = votes.get(taxon, 0.0) + weight
        
        # Find winner (tie-break by taxon count, then alphabetical)
        winner = max(votes.keys(), key=lambda t: (votes[t], taxon_counts[t], t))
        winner_weight = votes[winner]
        total_weight = sum(votes.values())
        
        # Raw confidence is fraction of weighted votes for winner
        raw_conf = winner_weight / total_weight if total_weight > 0 else 0.0
        
        # Compound confidence with previous ranks
        compounded_conf *= raw_conf
        
        pred_taxonomy.append(winner)
        conf_scores.append(compounded_conf)
        
        # Track best training sequence (nearest neighbor with winning taxon at first rank)
        if rank_idx == 0:
            for local_idx in nn_indices_local:
                if active_taxa_at_rank[local_idx] == winner:
                    best_train_idx = active_indices[local_idx]
                    break
        
        # Extend prefix for next rank
        prefix = prefix + (winner,)
    
    # Get region of best training sequence
    best_region = train_regions[best_train_idx] if best_train_idx is not None else ""
    
    return pred_taxonomy, conf_scores, best_region


def build_pnet_centroids(train_embeddings, train_taxa, rank_idx, active_indices):
    """
    Build centroids for prototypical network classification at a given rank.
    
    For each taxon, computes:
    - centroid: mean of embeddings, re-normalised to unit length
    - n: number of examples
    - R: resultant length (magnitude of sum before normalisation)
    
    Returns:
        dict mapping taxon -> (centroid, n, R)
    """
    centroids = {}
    active_taxa = train_taxa[active_indices, rank_idx]
    active_embeddings = train_embeddings[active_indices]
    
    unique_taxa = np.unique(active_taxa)
    for taxon in unique_taxa:
        mask = active_taxa == taxon
        taxon_embeddings = active_embeddings[mask]
        n = len(taxon_embeddings)
        
        # Sum embeddings
        vec_sum = taxon_embeddings.sum(axis=0)
        
        # Resultant length (magnitude of sum before normalisation)
        R = np.linalg.norm(vec_sum)
        
        # Re-normalise to unit length
        if R > 0:
            centroid = vec_sum / R
        else:
            centroid = vec_sum
        
        centroids[taxon] = (centroid, n, R)
    
    return centroids


def classify_pnet(test_embedding, train_embeddings, train_taxa, train_regions, prefix_indices_cache, centroid_cache):
    """
    Classify a single test sequence using prototypical networks.
    
    Classifies rank-by-rank using taxon centroids. Confidence via softmax
    over similarities, compounded across ranks.
    
    Returns:
        tuple (predicted_taxonomy, confidence_scores, best_train_region)
    """
    pred_taxonomy = []
    conf_scores = []
    
    # Start with no prefix (all training sequences)
    prefix = ()
    compounded_conf = 1.0
    
    for rank_idx in range(7):
        # Build centroids for this rank (cached by rank + taxonomy prefix)
        centroid_key = (rank_idx, prefix)
        if centroid_key in centroid_cache:
            centroids = centroid_cache[centroid_key]
        else:
            active_indices = get_active_indices_for_prefix(train_taxa, prefix, prefix_indices_cache)
            centroids = build_pnet_centroids(train_embeddings, train_taxa, rank_idx, active_indices)
            centroid_cache[centroid_key] = centroids
        
        if len(centroids) == 0:
            # No training data at this rank (shouldn't happen)
            pred_taxonomy.append("")
            conf_scores.append(0.0)
            continue
        
        if len(centroids) == 1:
            # Only one taxon, assign with full confidence
            winner = list(centroids.keys())[0]
            pred_taxonomy.append(winner)
            conf_scores.append(compounded_conf)
            prefix = prefix + (winner,)
            continue
        
        # Compute similarities to each centroid
        taxa = list(centroids.keys())
        sims = []
        for taxon in taxa:
            centroid, n, R = centroids[taxon]
            sim = np.dot(test_embedding, centroid)
            
            # Apply taxon size factor
            if USE_TAXON_SIZE_FACTOR:
                size_factor = n / (n + TAXON_SIZE_FACTOR_ALPHA)
                sim *= size_factor
            
            # Apply centroid quality factor
            if USE_CENTROID_QUALITY_FACTOR:
                quality_factor = R / (n + CENTROID_QUALITY_FACTOR_ALPHA)
                sim *= quality_factor
            
            sims.append(sim)
        
        sims = np.array(sims)
        
        # Find winner (highest similarity)
        winner_idx = np.argmax(sims)
        winner = taxa[winner_idx]
        
        # Compute softmax probabilities for confidence
        # Apply temperature scaling
        scaled_sims = sims / KAPPA_TEMPERATURE
        # Subtract max for numerical stability
        scaled_sims = scaled_sims - np.max(scaled_sims)
        exp_sims = np.exp(scaled_sims)
        probs = exp_sims / np.sum(exp_sims)
        
        raw_conf = probs[winner_idx]
        
        # Compound confidence with previous ranks
        compounded_conf *= raw_conf
        
        pred_taxonomy.append(winner)
        conf_scores.append(compounded_conf)
        
        # Extend prefix for next rank
        prefix = prefix + (winner,)
    
    # Find best training sequence (nearest to test among those matching predicted taxonomy)
    final_indices = get_active_indices_for_prefix(train_taxa, tuple(pred_taxonomy), prefix_indices_cache)
    
    if len(final_indices) > 0:
        final_embeddings = train_embeddings[final_indices]
        sims = np.dot(final_embeddings, test_embedding)
        best_local_idx = np.argmax(sims)
        best_train_idx = final_indices[best_local_idx]
        best_region = train_regions[best_train_idx]
    else:
        best_region = ""
    
    return pred_taxonomy, conf_scores, best_region


def write_about_file(about_path, train_count, test_count, classification_time):
    """
    Write configuration and run information to an about file.
    """
    with open(about_path, 'w') as f:
        f.write("Micro16S Classification Benchmark - Run Information\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("Paths:\n")
        f.write(f"  Model: {MODEL_PATH}\n")
        f.write(f"  Train sequences: {TRAIN_SEQS_PATH}\n")
        f.write(f"  Test sequences: {TEST_SEQS_PATH}\n")
        f.write(f"  Output: {PREDICTED_CLASSES_M16S_PATH}\n\n")
        
        f.write("Dataset:\n")
        f.write(f"  Training sequences: {train_count}\n")
        f.write(f"  Test sequences: {test_count}\n\n")
        
        f.write("General Settings:\n")
        f.write(f"  Batch size: {BATCH_SIZE}\n")
        f.write(f"  Classifier type: {CLASSIFIER_TYPE}\n\n")
        
        if CLASSIFIER_TYPE == "knn":
            f.write("k-NN Settings:\n")
            f.write(f"  K per rank: {K_PER_RANK}\n")
            f.write(f"  Taxon size weighting: {USE_TAXON_SIZE_WEIGHTING}\n")
            if USE_TAXON_SIZE_WEIGHTING:
                f.write(f"    Type: {TAXON_SIZE_WEIGHTING_TYPE}\n")
            f.write(f"  Distance weighting: {USE_DISTANCE_WEIGHTING}\n")
            if USE_DISTANCE_WEIGHTING:
                f.write(f"    Exponent: {DISTANCE_WEIGHTING_EXPONENT}\n")
                f.write(f"    Epsilon: {DISTANCE_WEIGHTING_EPSILON}\n")
        else:
            f.write("Prototypical Network Settings:\n")
            f.write(f"  Kappa temperature: {KAPPA_TEMPERATURE}\n")
            f.write(f"  Taxon size factor: {USE_TAXON_SIZE_FACTOR}\n")
            if USE_TAXON_SIZE_FACTOR:
                f.write(f"    Alpha: {TAXON_SIZE_FACTOR_ALPHA}\n")
            f.write(f"  Centroid quality factor: {USE_CENTROID_QUALITY_FACTOR}\n")
            if USE_CENTROID_QUALITY_FACTOR:
                f.write(f"    Alpha: {CENTROID_QUALITY_FACTOR_ALPHA}\n")
        
        f.write(f"\nClassification time: {classification_time:.2f} seconds\n")


# =============================================================================
# Main Execution
# =============================================================================

if __name__ == "__main__":
    
    # Check if GPU is detected
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cpu':
        input("Warning. No GPU detected. Press any key to continue...")
    else:
        # Enable cuDNN autotuner for fixed-shape inference batches on CUDA.
        torch.backends.cudnn.benchmark = True


    print("Starting Micro16S classification benchmarking...")
    print(f"Classifier type: {CLASSIFIER_TYPE}")

    # Ensure output directory exists
    os.makedirs(EVAL_DIR_PATH, exist_ok=True)
    
    # Step 1: Load sequences and metadata
    print("Loading train sequences and metadata...")
    train_sequences, train_metadata = load_fasta_with_metadata(TRAIN_SEQS_PATH, allow_ambiguous=False)
    print(f"Train sequences: {len(train_sequences)}")
    
    print("Loading test sequences and metadata...")
    test_sequences, test_metadata = load_fasta_with_metadata(TEST_SEQS_PATH, allow_ambiguous=False)
    print(f"Test sequences: {len(test_sequences)}")
    
    # Step 2: Load model
    print("Loading Micro16S model...")
    model = load_micro16s_model(MODEL_PATH)
    model = model.to(device)
    
    # Step 3: Run inference on train sequences
    print("Running inference on train sequences...")
    classification_start_time = time.time()
    train_embeddings = run_inference_on_sequences(model, train_sequences, BATCH_SIZE)
    train_taxa = np.array([m['Taxonomy'] for m in train_metadata], dtype=object)
    train_regions = np.array([m['Region'] for m in train_metadata], dtype=object)
    
    # Cache structures shared across all test sequences
    prefix_indices_cache = {}
    taxon_counts_cache = {}
    centroid_cache = {}
    
    # Step 4: Classify test sequences
    print("Classifying test sequences...")
    output_headers = [
        "ASV_ID", "Seq_Index", "Region_Train", "Region_Test",
        "Domain", "Phylum", "Class", "Order", "Family", "Genus", "Species",
        "Domain_Conf", "Phylum_Conf", "Class_Conf", "Order_Conf", "Family_Conf", "Genus_Conf", "Species_Conf"
    ]
    
    with open(PREDICTED_CLASSES_M16S_PATH, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(output_headers)
        
        for start in range(0, len(test_sequences), BATCH_SIZE):
            end = min(start + BATCH_SIZE, len(test_sequences))
            batch_seqs = test_sequences[start:end]
            batch_meta = test_metadata[start:end]
            
            # Run inference for this batch
            batch_embeddings = run_inference_on_sequences(model, batch_seqs, BATCH_SIZE)
            
            # Classify each test sequence in batch
            for i in range(len(batch_seqs)):
                test_embedding = batch_embeddings[i]
                
                # Use selected classifier
                if CLASSIFIER_TYPE == "knn":
                    full_sims = np.dot(train_embeddings, test_embedding)
                    pred_taxonomy, conf_scores, region_train = classify_knn(
                        test_embedding, train_embeddings, train_taxa, train_regions,
                        prefix_indices_cache, taxon_counts_cache, full_sims=full_sims
                    )
                else:
                    pred_taxonomy, conf_scores, region_train = classify_pnet(
                        test_embedding, train_embeddings, train_taxa, train_regions,
                        prefix_indices_cache, centroid_cache
                    )
                
                meta = batch_meta[i]
                row = [
                    meta['ASV_ID'],
                    meta['Seq_Index'],
                    region_train,      # Region_Train
                    meta['Region']     # Region_Test
                ] + list(pred_taxonomy) + [f"{c:.3f}" for c in conf_scores]
                
                writer.writerow(row)
            
            # Progress update
            print(f"  Processed {end}/{len(test_sequences)} sequences...")
    
    classification_time = time.time() - classification_start_time
    
    # Step 5: Write about file
    print("Writing about file...")
    write_about_file(ABOUT_FILE_PATH, len(train_sequences), len(test_sequences), classification_time)
    
    print("\nClassification completed successfully!")
    print(f"Classification time: {classification_time:.2f} seconds")
    print(f"Predicted classes written to: {PREDICTED_CLASSES_M16S_PATH}")
    print(f"About file written to: {ABOUT_FILE_PATH}")
