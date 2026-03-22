"""
Quick Test Module

This module provides functionality to evaluate a Micro16S model.
It measures classification accuracy and clustering quality across different taxonomic 
ranks and on training and test data. Also gets results of kmer distributions for a baseline.

Key Functions:
- quick_test: Evaluates model performance in a multitude of ways and returns the results as three dictionaries
- print_quick_test: Prints a summary of the quick test results from a single quick test (three dictionaries)
- add_results_df: Adds quick test results into a dataframe so performance can be tracked across training
- plot_quick_test: Visualizes test results from a dataframe of quick test results across training

Excluded set:   These are all sequences from a selection of taxa which where entirely excluded from the test and train sets.
Train set:      The random proportion of sequences put into the train set.
Test set:       The random proportion of sequences put into the test set.

"""



# IMPORTS ================================================
from datetime import datetime
import os
import time
import csv
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from sklearn.decomposition import KernelPCA
import umap
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import v_measure_score
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr
import re
import glob
import gc


# Local Imports
from generate_seq_variants import gen_seq_variants
import globals_config as gb
from model import run_inference

# Taxonomic ranks
ALL_RANKS = ["domain", "phylum", "class", "order", "family", "genus", "species"]
ALL_RANKS_PLURAL = ["domains", "phyla", "classes", "orders", "families", "genera", "species"]

# Sequence selection strategies
QT_SEQ_SELECTION_STRATEGIES = ['full', 'subseqs', 'both']


def quick_test(model, model_dir, batch_num=None, verbose=True):
    """
    Evaluates a trained model's performance on test data.

    Args:
        model: The trained Micro16S model to evaluate
        model_dir (str): Directory for saving model outputs
        batch_num (int, optional): Current training batch number for logging
        c (int, optional): Number of sequences to process in each batch
        get_kmer_results (bool): Whether to compute k-mer baseline results
        kmer_k_values (tuple, optional): K-values to use for k-mer baseline results

    Returns:
        clustering_scores_dict: Dictionary containing clustering scores
            - Keys are specific scores. e.g. "6-mers_train-set_clust_rank-0_subseqs"
        classification_scores_dict: Dictionary containing classification scores
            - Keys are specific scores. e.g. "embedding_excluded-set_class_rank-2_full"
        macro_classification_scores_dict: Dictionary containing macro classification scores
            - Keys are specific scores. e.g. "macro_embedding_train-set_class_rank-3_both"
        ssc_scores_dict: Dictionary containing subsequence congruency scores.
            - Keys are specific scores. e.g. "embedding_train-set_ssc_mean"
        rank_level_clustering_dict: Dictionary containing per-taxon clustering scores at each rank.
            - Keys are rank indices (int). Values are dicts with keys like:
              "embedding_train-set_subseqs_Bacteria" -> silhouette score (float)
              "Bacteria_size" -> count of sequences (int)

    Dataset files:
        - dataset_dir (contents from construct_dataset.py)
            - excluded_taxa_indices.txt (indices mapping to the N_SEQS dimension of the 3bit_seq_reps.npy array)
            - testing_indices.txt (indices mapping to the N_SEQS dimension of the 3bit_seq_reps.npy array)
            - training_indices.txt (indices mapping to the N_SEQS dimension of the 3bit_seq_reps.npy array)
            - tax_objs/
                - train/
                    - full_tax_label_from_seq_id_dict.pkl (one of the "taxonomic datastructures" from construct_dataset.py)
                    - list_of_seq_indices_in_taxon_at_rank_dict.pkl (one of the "taxonomic datastructures" from construct_dataset.py)
                    - list_of_taxon_labels_in_taxon_at_rank_dict.pkl (one of the "taxonomic datastructures" from construct_dataset.py)
                    - list_of_taxon_labels_at_rank_dict.pkl (one of the "taxonomic datastructures" from construct_dataset.py)
                    - nested_list_of_seq_indices.pkl (one of the "taxonomic datastructures" from construct_dataset.py)
                    - nested_dicts_of_taxa.pkl (one of the "taxonomic datastructures" from construct_dataset.py)
                    - taxon_label_to_taxon_id.pkl (maps taxon labels to integer IDs)
                    - taxon_id_to_taxon_label.pkl (maps integer IDs to taxon labels)
                - test/
                    - Same as train above
                - excluded/
                    - Same as train above
            - labels/
                - train/
                    - seq_taxon_ids.npy (sequence taxon IDs matrix with shape (n_seqs, 7) and dtype int32)
                    - pairwise_ranks.npy (pairwise taxonomic ranks matrix with shape (n_seqs, n_seqs) and dtype int8)
                    - pairwise_pos_masks.npy (positive pair masks with shape (7, n_seqs, n_seqs) and dtype bool)
                    - pairwise_neg_masks.npy (negative pair masks with shape (7, n_seqs, n_seqs) and dtype bool)
                    - pairwise_mrca_taxon_ids.npy (MRCA taxon IDs matrix with shape (n_seqs, n_seqs) and dtype int32)
                    - pairwise_distances.npy (pairwise phylogenetic distances matrix with shape (n_seqs, n_seqs) and dtype float32)
                    - distances_lookup_array.npy (1D lookup array for taxon ID to distance mapping)
                    - distance_between_domains.npy (distance between Bacteria and Archaea)
                - test/
                    - seq_taxon_ids.npy (sequence taxon IDs matrix with shape (n_seqs, 7) and dtype int32)
                    - pairwise_ranks.npy (pairwise taxonomic ranks matrix with shape (n_seqs, n_seqs) and dtype int8)
                    - pairwise_pos_masks.npy (positive pair masks with shape (7, n_seqs, n_seqs) and dtype bool)
                    - pairwise_neg_masks.npy (negative pair masks with shape (7, n_seqs, n_seqs) and dtype bool)
                    - pairwise_mrca_taxon_ids.npy (MRCA taxon IDs matrix with shape (n_seqs, n_seqs) and dtype int32)
                    - pairwise_distances.npy (pairwise phylogenetic distances matrix with shape (n_seqs, n_seqs) and dtype float32)
                    - distances_lookup_array.npy (1D lookup array for taxon ID to distance mapping)
                    - distance_between_domains.npy (distance between Bacteria and Archaea)
                - excluded/
                    - seq_taxon_ids.npy (sequence taxon IDs matrix with shape (n_seqs, 7) and dtype int32)
                    - pairwise_ranks.npy (pairwise taxonomic ranks matrix with shape (n_seqs, n_seqs) and dtype int8)
                    - pairwise_pos_masks.npy (positive pair masks with shape (7, n_seqs, n_seqs) and dtype bool)
                    - pairwise_neg_masks.npy (negative pair masks with shape (7, n_seqs, n_seqs) and dtype bool)
                    - pairwise_mrca_taxon_ids.npy (MRCA taxon IDs matrix with shape (n_seqs, n_seqs) and dtype int32)
                    - pairwise_distances.npy (pairwise phylogenetic distances matrix with shape (n_seqs, n_seqs) and dtype float32)
                    - distances_lookup_array.npy (1D lookup array for taxon ID to distance mapping)
                    - distance_between_domains.npy (distance between Bacteria and Archaea)
        -  os.path.dirname(dataset_dir) (contents from extract_regions.py and encode_sequences_3bit.py)
            - 3bit_seq_reps.npy OR 3bit_seq_reps_packed.npy
                - This is an important file, as it contains the 3-bit representations of all the sequences
                - The shape of the array is [N_REGIONS, N_SEQS, MAX_IMPORTED_SEQ_LEN, 3]
            - {K}-mer_seq_reps.npy
                - These files contain the k-mer representations of all the sequences
                - The are not assumed to exist, and are created if they are not found
                - The shape of the arrays are [N_REGIONS, N_SEQS, 4**K]
    """
    # Record start time
    qt_start_time = time.time()
    if verbose:
        print(f"Quick test starting...")
    
    # Ensure the dataset has been loaded via load_micro16s_dataset first
    if not gb.DATASET_IS_LOADED:
        raise RuntimeError("Dataset has not been loaded. Call load_micro16s_dataset first.")

    # Move model to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Initialise dictionary for new k-mer quick test results that should be cached
    kmer_qt_to_be_cached = {}
    # Load k-mer quick test cache
    gb.KMER_QT_CACHE = load_kmer_qt_cache(gb.DATASET_SPLIT_DIR)
    

    # Ensure we have then required k-mer representations ------------------------------------------
    if gb.QT_GET_KMER_RESULTS:
        # Determine which K-mers are needed
        needed_k_values = gb.QT_KMER_K_VALUES
        # If not all Ks are in gb.ALL_KMER_SEQ_REPS, raise an error
        if not all(k in gb.ALL_KMER_SEQ_REPS for k in needed_k_values):
            missing_k_values = [k for k in needed_k_values if k not in gb.ALL_KMER_SEQ_REPS]
            raise ValueError(f"Not all Ks are loaded in gb.ALL_KMER_SEQ_REPS: {missing_k_values}")
        

    # Compute embeddings ------------------------------------------
    if verbose:
        print("Computing embeddings...")
    part_start_time = time.time()
    
    # Compute the embeddings
    training_3bit_seq_reps_ = None
    testing_3bit_seq_reps_ = None
    excluded_3bit_seq_reps_ = None
    if gb.SEQLESS_MODE:
        # Sequenceless mode: build embeddings directly from indices (no sequence inference)
        n_train = len(gb.TRAINING_INDICES)
        indices = torch.arange(n_train, device=device, dtype=torch.long)
        with torch.no_grad():
            train_seq_embeddings = model(indices).detach().cpu().numpy()
        # Add region axis so downstream logic still works (N_REGIONS=1)
        train_seq_embeddings = train_seq_embeddings[None, :, :]
        test_seq_embeddings = None
        excluded_seq_embeddings = None
    else:
        # Ensure imported sequences are at least as long as the model expects
        if gb.TRAINING_3BIT_SEQ_REPS.shape[-2] < model.max_seq_len:
            raise ValueError(f"Training sequences length ({gb.TRAINING_3BIT_SEQ_REPS.shape[-2]}) is shorter than model.max_seq_len ({model.max_seq_len}). Increase MAX_IMPORTED_SEQ_LEN.")

        # Shift sequences if required by QT_PROP_SHIFT_SEQS
        training_3bit_seq_reps_ = gen_seq_variants(gb.TRAINING_3BIT_SEQ_REPS.copy(), mutation_rate=gb.QT_MUTATION_RATE, min_trunc_start=gb.QT_MIN_TRUNC_START, max_trunc_start=gb.QT_MAX_TRUNC_START, min_trunc_end=gb.QT_MIN_TRUNC_END, max_trunc_end=gb.QT_MAX_TRUNC_END, trunc_prop=gb.QT_PROP_TRUNC, shift_prop=gb.QT_PROP_SHIFT_SEQS, target_seq_len=gb.MAX_MODEL_SEQ_LEN)
        testing_3bit_seq_reps_ = gen_seq_variants(gb.TESTING_3BIT_SEQ_REPS.copy(), mutation_rate=gb.QT_MUTATION_RATE, min_trunc_start=gb.QT_MIN_TRUNC_START, max_trunc_start=gb.QT_MAX_TRUNC_START, min_trunc_end=gb.QT_MIN_TRUNC_END, max_trunc_end=gb.QT_MAX_TRUNC_END, trunc_prop=gb.QT_PROP_TRUNC, shift_prop=gb.QT_PROP_SHIFT_SEQS, target_seq_len=gb.MAX_MODEL_SEQ_LEN)
        excluded_3bit_seq_reps_ = None
        if gb.HAS_EXCLUDED_SET and gb.EXCLUDED_3BIT_SEQ_REPS is not None:
            excluded_3bit_seq_reps_ = gen_seq_variants(gb.EXCLUDED_3BIT_SEQ_REPS.copy(), mutation_rate=gb.QT_MUTATION_RATE, min_trunc_start=gb.QT_MIN_TRUNC_START, max_trunc_start=gb.QT_MAX_TRUNC_START, min_trunc_end=gb.QT_MIN_TRUNC_END, max_trunc_end=gb.QT_MAX_TRUNC_END, trunc_prop=gb.QT_PROP_TRUNC, shift_prop=gb.QT_PROP_SHIFT_SEQS, target_seq_len=gb.MAX_MODEL_SEQ_LEN)

        # Cap number of regions for quick test ------------------------------------------
        if gb.QT_MAX_N_REGIONS is not None:
            n_regions_total = training_3bit_seq_reps_.shape[0]
            if n_regions_total > 1:
                n_subseqs_available = n_regions_total - 1
                n_subseqs_to_use = min(gb.QT_MAX_N_REGIONS, n_subseqs_available)
                if n_subseqs_to_use < n_subseqs_available:
                    subseq_indices = np.random.choice(np.arange(1, n_regions_total), size=n_subseqs_to_use, replace=False)
                    region_indices = np.concatenate(([0], np.sort(subseq_indices)))
                    training_3bit_seq_reps_ = training_3bit_seq_reps_[region_indices]
                    testing_3bit_seq_reps_ = testing_3bit_seq_reps_[region_indices]
                    if excluded_3bit_seq_reps_ is not None:
                        excluded_3bit_seq_reps_ = excluded_3bit_seq_reps_[region_indices]

        # Run inference
        train_seq_embeddings = run_inference(model, training_3bit_seq_reps_, device, batch_size=gb.QT_BATCH_SIZE)
        test_seq_embeddings = run_inference(model, testing_3bit_seq_reps_, device, batch_size=gb.QT_BATCH_SIZE)
        excluded_seq_embeddings = None
        if excluded_3bit_seq_reps_ is not None:
            excluded_seq_embeddings = run_inference(model, excluded_3bit_seq_reps_, device, batch_size=gb.QT_BATCH_SIZE)


    part_end_time = time.time()
    if verbose:
        print(f"  Took: {(part_end_time - part_start_time) * 1000:.0f}ms")


    # Move model back to CPU and clear GPU memory
    model = model.cpu()
    torch.cuda.empty_cache()

    
    

    # Embeddings --------------------------------
        # train_seq_embeddings.shape = (N_REGIONS, N_SEQS, D)
        # test_seq_embeddings.shape = (N_REGIONS, N_SEQS, D)
        # excluded_seq_embeddings.shape = (N_REGIONS, N_SEQS, D)
        # (numpy arrays)
    # K-mer representations ----------------------------
        # gb.TRAINING_KMER_SEQ_REPS[k].shape = (N_REGIONS, N_SEQS, 4**k)
        # gb.TESTING_KMER_SEQ_REPS[k].shape = (N_REGIONS, N_SEQS, 4**k)
        # gb.EXCLUDED_KMER_SEQ_REPS[k].shape = (N_REGIONS, N_SEQS, 4**k)
        # (dictionaries of numpy arrays)
    # Taxonomic labels --------------------------------
        # train_full_tax_labels.shape = (N_SEQS,)
        # test_full_tax_labels.shape = (N_SEQS,)
        # excluded_full_tax_labels.shape = (N_SEQS,)
        # (each is a list of N_SEQS lists of 7 strings)
    # Taxonomic datastructures ----------------------------
        # Train:
            # train_full_tax_label_from_seq_id_dict
            # train_list_of_seq_indices_in_taxon_at_rank_dict
            # train_list_of_taxon_labels_in_taxon_at_rank_dict
            # train_list_of_taxon_labels_at_rank_dict
            # train_nested_list_of_seq_indices
            # train_nested_dicts_of_taxa
        # Test:
            # test_full_tax_label_from_seq_id_dict
            # test_list_of_seq_indices_in_taxon_at_rank_dict
            # test_list_of_taxon_labels_in_taxon_at_rank_dict
            # test_list_of_taxon_labels_at_rank_dict
            # test_nested_list_of_seq_indices
            # test_nested_dicts_of_taxa
        # Excluded:
            # excluded_full_tax_label_from_seq_id_dict
            # excluded_list_of_seq_indices_in_taxon_at_rank_dict
            # excluded_list_of_taxon_labels_in_taxon_at_rank_dict
            # excluded_list_of_taxon_labels_at_rank_dict
            # excluded_nested_list_of_seq_indices
            # excluded_nested_dicts_of_taxa

    

    # Select regions ------------------------------------------
    if verbose:
        print("Selecting sequence regions for evaluation...")
    part_start_time = time.time()

    # Determine the number of regions
    n_regions = train_seq_embeddings.shape[0]

    # Create dictionaries to hold the selected embeddings and k-mer representations
    train_embeddings, test_embeddings, excluded_embeddings = {}, {}, {}
    train_kmer_reps, test_kmer_reps, excluded_kmer_reps = {}, {}, {}
    
    # Loop over the selection strategies ('full', 'subseqs', 'both')
    for strategy in QT_SEQ_SELECTION_STRATEGIES:
        # Check if any test requires this strategy for embeddings
        if any(f'-{strategy}-embed' in t for t in gb.QT_TESTS_TODO):
            # --- Training Set ---
            if 'full' in strategy:  # Handles 'full' and 'both'
                train_indices = np.random.randint(0, 1 if strategy == 'full' else n_regions, size=train_seq_embeddings.shape[1])
            elif n_regions > 1: # 'subseqs'
                train_indices = np.random.randint(1, n_regions, size=train_seq_embeddings.shape[1])
            else: # 'subseqs' with no subsequences
                train_indices = None

            if train_indices is not None:
                train_embeddings[strategy] = train_seq_embeddings[train_indices, np.arange(train_seq_embeddings.shape[1]), :]

            # --- Testing Set ---
            test_indices = None
            if test_seq_embeddings is not None:
                if 'full' in strategy:
                    test_indices = np.random.randint(0, 1 if strategy == 'full' else n_regions, size=test_seq_embeddings.shape[1])
                elif n_regions > 1:
                    test_indices = np.random.randint(1, n_regions, size=test_seq_embeddings.shape[1])
            
            if test_indices is not None:
                test_embeddings[strategy] = test_seq_embeddings[test_indices, np.arange(test_seq_embeddings.shape[1]), :]

            # --- Excluded Set ---
            excluded_indices = None
            if gb.HAS_EXCLUDED_SET and excluded_seq_embeddings is not None:
                if 'full' in strategy:
                    excluded_indices = np.random.randint(0, 1 if strategy == 'full' else n_regions, size=excluded_seq_embeddings.shape[1])
                elif n_regions > 1:
                    excluded_indices = np.random.randint(1, n_regions, size=excluded_seq_embeddings.shape[1])
                
            if excluded_indices is not None:
                excluded_embeddings[strategy] = excluded_seq_embeddings[excluded_indices, np.arange(excluded_seq_embeddings.shape[1]), :]

        # Check if any test requires this strategy for k-mers
        if gb.QT_GET_KMER_RESULTS and any(f'-{strategy}-kmer' in t for t in gb.QT_TESTS_TODO):
            train_kmer_reps[strategy], test_kmer_reps[strategy] = {}, {}
            if gb.HAS_EXCLUDED_SET:
                excluded_kmer_reps[strategy] = {}
            for k in gb.QT_KMER_K_VALUES:
                # --- Training Set ---
                if 'full' in strategy:
                    train_indices_kmer = np.random.randint(0, 1 if strategy == 'full' else n_regions, size=gb.TRAINING_KMER_SEQ_REPS[k].shape[1])
                elif n_regions > 1:
                    train_indices_kmer = np.random.randint(1, n_regions, size=gb.TRAINING_KMER_SEQ_REPS[k].shape[1])
                else:
                    train_indices_kmer = None
                
                if train_indices_kmer is not None:
                    train_kmer_reps[strategy][k] = gb.TRAINING_KMER_SEQ_REPS[k][train_indices_kmer, np.arange(gb.TRAINING_KMER_SEQ_REPS[k].shape[1]), :]

                # --- Testing Set ---
                if 'full' in strategy:
                    test_indices_kmer = np.random.randint(0, 1 if strategy == 'full' else n_regions, size=gb.TESTING_KMER_SEQ_REPS[k].shape[1])
                elif n_regions > 1:
                    test_indices_kmer = np.random.randint(1, n_regions, size=gb.TESTING_KMER_SEQ_REPS[k].shape[1])
                else:
                    test_indices_kmer = None
                
                if test_indices_kmer is not None:
                    test_kmer_reps[strategy][k] = gb.TESTING_KMER_SEQ_REPS[k][test_indices_kmer, np.arange(gb.TESTING_KMER_SEQ_REPS[k].shape[1]), :]

                # --- Excluded Set ---
                if gb.HAS_EXCLUDED_SET:
                    if 'full' in strategy:
                        excluded_indices_kmer = np.random.randint(0, 1 if strategy == 'full' else n_regions, size=gb.EXCLUDED_KMER_SEQ_REPS[k].shape[1])
                    elif n_regions > 1:
                        excluded_indices_kmer = np.random.randint(1, n_regions, size=gb.EXCLUDED_KMER_SEQ_REPS[k].shape[1])
                    else:
                        excluded_indices_kmer = None

                    if excluded_indices_kmer is not None:
                        excluded_kmer_reps[strategy][k] = gb.EXCLUDED_KMER_SEQ_REPS[k][excluded_indices_kmer, np.arange(gb.EXCLUDED_KMER_SEQ_REPS[k].shape[1]), :]

    part_end_time = time.time()
    if verbose:
        print(f"  Took: {(part_end_time - part_start_time) * 1000:.0f}ms")




    # Embeddings --------------------------------
        # train_embeddings[strategy].shape = (N_SEQS, D)
        # test_embeddings[strategy].shape = (N_SEQS, D)
        # excluded_embeddings[strategy].shape = (N_SEQS, D)
        # (dictionary of numpy arrays)
    # K-mer representations ----------------------------
        # train_kmer_reps[strategy][k].shape = (N_SEQS, 4**k,)
        # test_kmer_reps[strategy][k].shape = (N_SEQS, 4**k,)
        # excluded_kmer_reps[strategy][k].shape = (N_SEQS, 4**k,)
        # (dictionary of dictionaries of numpy arrays)




    # Initialise results dictionaries --------------------------------
    # Key format:
    # Clustering: f"{rep_name}_{set}-set_clust_rank-{rank}_{strategy}"
    #   e.g. "6-mers_train-set_clust_rank-0_subseqs"
    #   e.g. "embedding_excluded-set_clust_rank-2_full"
    # Classification: f"{rep_name}_{set}-set_class_rank-{rank}_{strategy}"
    #   e.g. "6-mers_test-set_class_rank-3_subseqs"
    #   e.g. "embedding_test-set_class_rank-1_full"
    # Macro Classification: f"macro_{rep_name}_{set}-set_class_rank-{rank}_{strategy}"
    #   e.g. "macro_6-mers_test-set_class_rank-3_subseqs"
    # SSC: f"{rep_name}_{set}-set_ssc_{metric}"
    #   e.g. "6-mers_train-set_ssc_mean"
    clustering_scores_dict = {}
    classification_scores_dict = {}
    macro_classification_scores_dict = {}
    ssc_scores_dict = {}
    rank_level_clustering_dict = {}


    # Perform PCoAs ------------------------------------------
    if batch_num % gb.QT_PCOA_EVERY_N_BATCHES == 0:
        if verbose:
            print("Performing PCoA analysis...")
        part_start_time = time.time()

        train_embeddings_pcoa, test_embeddings_pcoa, excluded_embeddings_pcoa = {}, {}, {}
        for strategy in QT_SEQ_SELECTION_STRATEGIES:
            if f'train-{strategy}-embed-pcoa' in gb.QT_TESTS_TODO and strategy in train_embeddings:
                train_embeddings_pcoa[strategy] = KernelPCA(n_components=2, kernel='cosine', n_jobs=12).fit_transform(train_embeddings[strategy])
            if f'test-{strategy}-embed-pcoa' in gb.QT_TESTS_TODO and strategy in test_embeddings:
                test_embeddings_pcoa[strategy] = KernelPCA(n_components=2, kernel='cosine', n_jobs=12).fit_transform(test_embeddings[strategy])
            if f'excl-{strategy}-embed-pcoa' in gb.QT_TESTS_TODO and strategy in excluded_embeddings:
                excluded_embeddings_pcoa[strategy] = KernelPCA(n_components=2, kernel='cosine', n_jobs=12).fit_transform(excluded_embeddings[strategy])
        
        # PCoA on k-mer representations
        if gb.QT_GET_KMER_RESULTS:
            train_kmer_reps_pcoa, test_kmer_reps_pcoa, excluded_kmer_reps_pcoa = {}, {}, {}
            for strategy in QT_SEQ_SELECTION_STRATEGIES:
                train_kmer_reps_pcoa[strategy], test_kmer_reps_pcoa[strategy], excluded_kmer_reps_pcoa[strategy] = {}, {}, {}
                for k in gb.QT_KMER_K_VALUES:
                    # Check for existing plots for k-mer PCoAs
                    skip_train = f'train-{strategy}-kmer-pcoa' not in gb.QT_TESTS_TODO or \
                                 all(glob.glob(os.path.join(gb.DATASET_SPLIT_DIR, f"{k}-mer_pcoa_plots", f"train-set_{strategy}_K{k}_pcoa_{rank}.*")) for rank in gb.QT_PCOA_RANKS)
                    skip_test = f'test-{strategy}-kmer-pcoa' not in gb.QT_TESTS_TODO or \
                                all(glob.glob(os.path.join(gb.DATASET_SPLIT_DIR, f"{k}-mer_pcoa_plots", f"test-set_{strategy}_K{k}_pcoa_{rank}.*")) for rank in gb.QT_PCOA_RANKS)
                    skip_excl = f'excl-{strategy}-kmer-pcoa' not in gb.QT_TESTS_TODO or \
                                all(glob.glob(os.path.join(gb.DATASET_SPLIT_DIR, f"{k}-mer_pcoa_plots", f"excluded-set_{strategy}_K{k}_pcoa_{rank}.*")) for rank in gb.QT_PCOA_RANKS)

                    # Perform PCoA for k-mers if plots don't exist
                    if not skip_train and strategy in train_kmer_reps and k in train_kmer_reps[strategy]:
                        train_kmer_reps_pcoa[strategy][k] = KernelPCA(n_components=2, kernel='cosine', n_jobs=12).fit_transform(train_kmer_reps[strategy][k])
                    if not skip_test and strategy in test_kmer_reps and k in test_kmer_reps[strategy]:
                        test_kmer_reps_pcoa[strategy][k] = KernelPCA(n_components=2, kernel='cosine', n_jobs=12).fit_transform(test_kmer_reps[strategy][k])
                    if not skip_excl and strategy in excluded_kmer_reps and k in excluded_kmer_reps[strategy]:
                        excluded_kmer_reps_pcoa[strategy][k] = KernelPCA(n_components=2, kernel='cosine', n_jobs=12).fit_transform(excluded_kmer_reps[strategy][k])

        part_end_time = time.time()
        if verbose:
            print(f"  Took: {(part_end_time - part_start_time) * 1000:.0f}ms")
        
        # Embedding PCoAs --------------------------------
            # train_embeddings_pcoa[strategy].shape = (N_SEQS, 2)
            # test_embeddings_pcoa[strategy].shape = (N_SEQS, 2)
            # excluded_embeddings_pcoa[strategy].shape = (N_SEQS, 2)
            # (dictionary of numpy arrays, keyed by strategy)
        # K-mer representation PCoAs --------------------------------
            # train_kmer_reps_pcoa[strategy][k].shape = (N_SEQS, 2)
            # test_kmer_reps_pcoa[strategy][k].shape = (N_SEQS, 2)
            # excluded_kmer_reps_pcoa[strategy][k].shape = (N_SEQS, 2)
            # (dictionary of dictionaries of numpy arrays, keyed by strategy and k)


    # Perform UMAPs ------------------------------------------
    if batch_num % gb.QT_UMAP_EVERY_N_BATCHES == 0:
        if verbose:
            print("Performing UMAP analysis...")
        part_start_time = time.time()

        train_embeddings_umap, test_embeddings_umap, excluded_embeddings_umap = {}, {}, {}
        for strategy in QT_SEQ_SELECTION_STRATEGIES:
            if f'train-{strategy}-embed-umap' in gb.QT_TESTS_TODO and strategy in train_embeddings:
                train_embeddings_umap[strategy] = umap.UMAP(n_components=2, n_neighbors=36, min_dist=0.05, metric='cosine', init='spectral', n_jobs=12).fit_transform(train_embeddings[strategy])
            if f'test-{strategy}-embed-umap' in gb.QT_TESTS_TODO and strategy in test_embeddings:
                test_embeddings_umap[strategy] = umap.UMAP(n_components=2, n_neighbors=36, min_dist=0.05, metric='cosine', init='spectral', n_jobs=12).fit_transform(test_embeddings[strategy])
            if f'excl-{strategy}-embed-umap' in gb.QT_TESTS_TODO and strategy in excluded_embeddings:
                excluded_embeddings_umap[strategy] = umap.UMAP(n_components=2, n_neighbors=36, min_dist=0.05, metric='cosine', init='spectral', n_jobs=12).fit_transform(excluded_embeddings[strategy])
        
        # UMAP on k-mer representations
        if gb.QT_GET_KMER_RESULTS:
            train_kmer_reps_umap, test_kmer_reps_umap, excluded_kmer_reps_umap = {}, {}, {}
            for strategy in QT_SEQ_SELECTION_STRATEGIES:
                train_kmer_reps_umap[strategy], test_kmer_reps_umap[strategy], excluded_kmer_reps_umap[strategy] = {}, {}, {}
                for k in gb.QT_KMER_K_VALUES:
                    # Check for existing plots for k-mer UMAPs
                    skip_train = f'train-{strategy}-kmer-umap' not in gb.QT_TESTS_TODO or \
                                 all(glob.glob(os.path.join(gb.DATASET_SPLIT_DIR, f"{k}-mer_umap_plots", f"train-set_{strategy}_K{k}_umap_{rank}.*")) for rank in gb.QT_UMAP_RANKS)
                    skip_test = f'test-{strategy}-kmer-umap' not in gb.QT_TESTS_TODO or \
                                all(glob.glob(os.path.join(gb.DATASET_SPLIT_DIR, f"{k}-mer_umap_plots", f"test-set_{strategy}_K{k}_umap_{rank}.*")) for rank in gb.QT_UMAP_RANKS)
                    skip_excl = f'excl-{strategy}-kmer-umap' not in gb.QT_TESTS_TODO or \
                                all(glob.glob(os.path.join(gb.DATASET_SPLIT_DIR, f"{k}-mer_umap_plots", f"excluded-set_{strategy}_K{k}_umap_{rank}.*")) for rank in gb.QT_UMAP_RANKS)

                    # Perform UMAP for k-mers if plots don't exist
                    if not skip_train and strategy in train_kmer_reps and k in train_kmer_reps[strategy]:
                        train_kmer_reps_umap[strategy][k] = umap.UMAP(n_components=2, n_neighbors=36, min_dist=0.05, metric='cosine', init='spectral', n_jobs=12).fit_transform(train_kmer_reps[strategy][k])
                    if not skip_test and strategy in test_kmer_reps and k in test_kmer_reps[strategy]:
                        test_kmer_reps_umap[strategy][k] = umap.UMAP(n_components=2, n_neighbors=36, min_dist=0.05, metric='cosine', init='spectral', n_jobs=12).fit_transform(test_kmer_reps[strategy][k])
                    if not skip_excl and strategy in excluded_kmer_reps and k in excluded_kmer_reps[strategy]:
                        excluded_kmer_reps_umap[strategy][k] = umap.UMAP(n_components=2, n_neighbors=36, min_dist=0.05, metric='cosine', init='spectral', n_jobs=12).fit_transform(excluded_kmer_reps[strategy][k])

        part_end_time = time.time()
        if verbose:
            print(f"  Took: {(part_end_time - part_start_time) * 1000:.0f}ms")
        
        # Embedding UMAPs --------------------------------
            # train_embeddings_umap[strategy].shape = (N_SEQS, 2)
            # test_embeddings_umap[strategy].shape = (N_SEQS, 2)
            # excluded_embeddings_umap[strategy].shape = (N_SEQS, 2)
            # (dictionary of numpy arrays, keyed by strategy)
        # K-mer representation UMAPs --------------------------------
            # train_kmer_reps_umap[strategy][k].shape = (N_SEQS, 2)
            # test_kmer_reps_umap[strategy][k].shape = (N_SEQS, 2)
            # excluded_kmer_reps_umap[strategy][k].shape = (N_SEQS, 2)
            # (dictionary of dictionaries of numpy arrays, keyed by strategy and k)


    # Plot PCoAs -----------------------------------------
    if batch_num % gb.QT_PCOA_EVERY_N_BATCHES == 0:
        if verbose:
            print("Plotting PCoA plots...")
        part_start_time = time.time()
        # Produce PCoA plots of k-mer representations
        if gb.QT_GET_KMER_RESULTS:
            for k in gb.QT_KMER_K_VALUES:
                for rank in gb.QT_PCOA_RANKS:
                    for strategy in QT_SEQ_SELECTION_STRATEGIES:
                        if strategy in train_kmer_reps_pcoa and k in train_kmer_reps_pcoa[strategy]:
                            visualise_ordination(train_kmer_reps_pcoa[strategy][k], gb.TRAIN_FULL_TAX_LABELS, rank, gb.DATASET_SPLIT_DIR, 'PCoA', batch_num=None, rep_name=f"{k}-mer", set_name=f"train-set_{strategy}_K{k}", skip_batch_num=True)
                        if strategy in test_kmer_reps_pcoa and k in test_kmer_reps_pcoa[strategy]:
                            visualise_ordination(test_kmer_reps_pcoa[strategy][k], gb.TEST_FULL_TAX_LABELS, rank, gb.DATASET_SPLIT_DIR, 'PCoA', batch_num=None, rep_name=f"{k}-mer", set_name=f"test-set_{strategy}_K{k}", skip_batch_num=True)
                        if strategy in excluded_kmer_reps_pcoa and k in excluded_kmer_reps_pcoa[strategy]:
                            visualise_ordination(excluded_kmer_reps_pcoa[strategy][k], gb.EXCLUDED_FULL_TAX_LABELS, rank, gb.DATASET_SPLIT_DIR, 'PCoA', batch_num=None, rep_name=f"{k}-mer", set_name=f"excluded-set_{strategy}_K{k}", skip_batch_num=True)

        # Produce PCoA plots of embeddings
        for rank in gb.QT_PCOA_RANKS:
            for strategy in QT_SEQ_SELECTION_STRATEGIES:
                if f'train-{strategy}-embed-pcoa' in gb.QT_TESTS_TODO and strategy in train_embeddings_pcoa:
                    visualise_ordination(train_embeddings_pcoa[strategy], gb.TRAIN_FULL_TAX_LABELS, rank, model_dir, 'PCoA', batch_num=batch_num, rep_name="embeddings", set_name=f"train-set_{strategy}")
                if f'test-{strategy}-embed-pcoa' in gb.QT_TESTS_TODO and strategy in test_embeddings_pcoa:
                    visualise_ordination(test_embeddings_pcoa[strategy], gb.TEST_FULL_TAX_LABELS, rank, model_dir, 'PCoA', batch_num=batch_num, rep_name="embeddings", set_name=f"test-set_{strategy}")
                if f'excl-{strategy}-embed-pcoa' in gb.QT_TESTS_TODO and strategy in excluded_embeddings_pcoa:
                    visualise_ordination(excluded_embeddings_pcoa[strategy], gb.EXCLUDED_FULL_TAX_LABELS, rank, model_dir, 'PCoA', batch_num=batch_num, rep_name="embeddings", set_name=f"excluded-set_{strategy}")

        part_end_time = time.time()
        if verbose:
            print(f"  Took: {(part_end_time - part_start_time) * 1000:.0f}ms")


    # Plot UMAPs -----------------------------------------
    if batch_num % gb.QT_UMAP_EVERY_N_BATCHES == 0:
        if verbose:
            print("Plotting UMAP plots...")
        part_start_time = time.time()
        # Produce UMAP plots of k-mer representations
        if gb.QT_GET_KMER_RESULTS:
            for k in gb.QT_KMER_K_VALUES:
                for rank in gb.QT_UMAP_RANKS:
                    for strategy in QT_SEQ_SELECTION_STRATEGIES:
                        if strategy in train_kmer_reps_umap and k in train_kmer_reps_umap[strategy]:
                            visualise_ordination(train_kmer_reps_umap[strategy][k], gb.TRAIN_FULL_TAX_LABELS, rank, gb.DATASET_SPLIT_DIR, 'UMAP', batch_num=None, rep_name=f"{k}-mer", set_name=f"train-set_{strategy}_K{k}", skip_batch_num=True)
                        if strategy in test_kmer_reps_umap and k in test_kmer_reps_umap[strategy]:
                            visualise_ordination(test_kmer_reps_umap[strategy][k], gb.TEST_FULL_TAX_LABELS, rank, gb.DATASET_SPLIT_DIR, 'UMAP', batch_num=None, rep_name=f"{k}-mer", set_name=f"test-set_{strategy}_K{k}", skip_batch_num=True)
                        if strategy in excluded_kmer_reps_umap and k in excluded_kmer_reps_umap[strategy]:
                            visualise_ordination(excluded_kmer_reps_umap[strategy][k], gb.EXCLUDED_FULL_TAX_LABELS, rank, gb.DATASET_SPLIT_DIR, 'UMAP', batch_num=None, rep_name=f"{k}-mer", set_name=f"excluded-set_{strategy}_K{k}", skip_batch_num=True)

        # Produce UMAP plots of embeddings
        for rank in gb.QT_UMAP_RANKS:
            for strategy in QT_SEQ_SELECTION_STRATEGIES:
                if f'train-{strategy}-embed-umap' in gb.QT_TESTS_TODO and strategy in train_embeddings_umap:
                    visualise_ordination(train_embeddings_umap[strategy], gb.TRAIN_FULL_TAX_LABELS, rank, model_dir, 'UMAP', batch_num=batch_num, rep_name="embeddings", set_name=f"train-set_{strategy}")
                if f'test-{strategy}-embed-umap' in gb.QT_TESTS_TODO and strategy in test_embeddings_umap:
                    visualise_ordination(test_embeddings_umap[strategy], gb.TEST_FULL_TAX_LABELS, rank, model_dir, 'UMAP', batch_num=batch_num, rep_name="embeddings", set_name=f"test-set_{strategy}")
                if f'excl-{strategy}-embed-umap' in gb.QT_TESTS_TODO and strategy in excluded_embeddings_umap:
                    visualise_ordination(excluded_embeddings_umap[strategy], gb.EXCLUDED_FULL_TAX_LABELS, rank, model_dir, 'UMAP', batch_num=batch_num, rep_name="embeddings", set_name=f"excluded-set_{strategy}")

        part_end_time = time.time()
        if verbose:
            print(f"  Took: {(part_end_time - part_start_time) * 1000:.0f}ms")







    # Perform Subsequence Congruency Tests -----------------------------------------
    if verbose:
        print("Performing Subsequence Congruency tests...")
    part_start_time = time.time()

    # SSC for embeddings
    ssc_train_mean, ssc_train_std = test_subsequence_congruency(train_seq_embeddings) if 'train-embed-ssc' in gb.QT_TESTS_TODO else (None, None)
    ssc_scores_dict['embedding_train-set_ssc_mean'] = ssc_train_mean
    ssc_scores_dict['embedding_train-set_ssc_std'] = ssc_train_std
    
    ssc_test_mean, ssc_test_std = test_subsequence_congruency(test_seq_embeddings) if 'test-embed-ssc' in gb.QT_TESTS_TODO else (None, None)
    ssc_scores_dict['embedding_test-set_ssc_mean'] = ssc_test_mean
    ssc_scores_dict['embedding_test-set_ssc_std'] = ssc_test_std

    if gb.HAS_EXCLUDED_SET and excluded_seq_embeddings is not None and 'excl-embed-ssc' in gb.QT_TESTS_TODO:
        ssc_excl_mean, ssc_excl_std = test_subsequence_congruency(excluded_seq_embeddings)
    else:
        ssc_excl_mean, ssc_excl_std = (None, None)
    ssc_scores_dict['embedding_excluded-set_ssc_mean'] = ssc_excl_mean
    ssc_scores_dict['embedding_excluded-set_ssc_std'] = ssc_excl_std

    # SSC for k-mers
    if gb.QT_GET_KMER_RESULTS:
        for k in gb.QT_KMER_K_VALUES:
            # Train set
            key_mean_train, key_std_train = f'{k}-mers_train-set_ssc_mean', f'{k}-mers_train-set_ssc_std'
            if 'train-kmer-ssc' in gb.QT_TESTS_TODO:
                if key_mean_train in gb.KMER_QT_CACHE and key_std_train in gb.KMER_QT_CACHE:
                    ssc_scores_dict[key_mean_train], ssc_scores_dict[key_std_train] = gb.KMER_QT_CACHE[key_mean_train], gb.KMER_QT_CACHE[key_std_train]
                else:
                    ssc_mean, ssc_std = test_subsequence_congruency(gb.TRAINING_KMER_SEQ_REPS[k])
                    ssc_scores_dict[key_mean_train], ssc_scores_dict[key_std_train] = ssc_mean, ssc_std
                    kmer_qt_to_be_cached[key_mean_train], kmer_qt_to_be_cached[key_std_train] = ssc_mean, ssc_std
            else:
                ssc_scores_dict[key_mean_train], ssc_scores_dict[key_std_train] = None, None

            # Test set
            key_mean_test, key_std_test = f'{k}-mers_test-set_ssc_mean', f'{k}-mers_test-set_ssc_std'
            if 'test-kmer-ssc' in gb.QT_TESTS_TODO:
                if key_mean_test in gb.KMER_QT_CACHE and key_std_test in gb.KMER_QT_CACHE:
                    ssc_scores_dict[key_mean_test], ssc_scores_dict[key_std_test] = gb.KMER_QT_CACHE[key_mean_test], gb.KMER_QT_CACHE[key_std_test]
                else:
                    ssc_mean, ssc_std = test_subsequence_congruency(gb.TESTING_KMER_SEQ_REPS[k])
                    ssc_scores_dict[key_mean_test], ssc_scores_dict[key_std_test] = ssc_mean, ssc_std
                    kmer_qt_to_be_cached[key_mean_test], kmer_qt_to_be_cached[key_std_test] = ssc_mean, ssc_std
            else:
                ssc_scores_dict[key_mean_test], ssc_scores_dict[key_std_test] = None, None

            # Excluded set
            key_mean_excl, key_std_excl = f'{k}-mers_excluded-set_ssc_mean', f'{k}-mers_excluded-set_ssc_std'
            if gb.HAS_EXCLUDED_SET and 'excl-kmer-ssc' in gb.QT_TESTS_TODO:
                if key_mean_excl in gb.KMER_QT_CACHE and key_std_excl in gb.KMER_QT_CACHE:
                    ssc_scores_dict[key_mean_excl], ssc_scores_dict[key_std_excl] = gb.KMER_QT_CACHE[key_mean_excl], gb.KMER_QT_CACHE[key_std_excl]
                else:
                    ssc_mean, ssc_std = test_subsequence_congruency(gb.EXCLUDED_KMER_SEQ_REPS[k])
                    ssc_scores_dict[key_mean_excl], ssc_scores_dict[key_std_excl] = ssc_mean, ssc_std
                    kmer_qt_to_be_cached[key_mean_excl], kmer_qt_to_be_cached[key_std_excl] = ssc_mean, ssc_std
            else:
                ssc_scores_dict[key_mean_excl], ssc_scores_dict[key_std_excl] = None, None
    
    part_end_time = time.time()
    if verbose:
        print(f"  Took: {(part_end_time - part_start_time) * 1000:.0f}ms")


    # Perform Clustering -----------------------------------------
    if verbose:
        print("Evaluating clustering performance...")
    part_start_time = time.time()

    # Whether to save contingency matrices this batch
    save_contingency = (batch_num is not None and
                        batch_num % gb.RANK_LEVEL_CLUSTERING_PLOTS_EVERY_N_BATCHES == 0)

    for strategy in QT_SEQ_SELECTION_STRATEGIES:
        # Perform Clustering of k-mer representations
        if gb.QT_GET_KMER_RESULTS and strategy in train_kmer_reps:
            for k in gb.QT_KMER_K_VALUES:
                for rank in gb.QT_CLUSTERING_RANKS:
                    # Train set
                    key_train = f"{k}-mers_train-set_clust_rank-{rank}_{strategy}"
                    if f'train-{strategy}-kmer-clust' in gb.QT_TESTS_TODO:
                        if key_train in gb.KMER_QT_CACHE:
                            clustering_scores_dict[key_train] = gb.KMER_QT_CACHE[key_train]
                        else:
                            score = cluster_reps(train_kmer_reps[strategy][k], gb.TRAIN_FULL_TAX_LABELS, rank, dist_measure='euclidean')
                            clustering_scores_dict[key_train] = score
                            kmer_qt_to_be_cached[key_train] = score
                    else:
                        clustering_scores_dict[key_train] = None

                    # Test set
                    key_test = f"{k}-mers_test-set_clust_rank-{rank}_{strategy}"
                    if f'test-{strategy}-kmer-clust' in gb.QT_TESTS_TODO:
                        if key_test in gb.KMER_QT_CACHE:
                            clustering_scores_dict[key_test] = gb.KMER_QT_CACHE[key_test]
                        else:
                            score = cluster_reps(test_kmer_reps[strategy][k], gb.TEST_FULL_TAX_LABELS, rank, dist_measure='euclidean')
                            clustering_scores_dict[key_test] = score
                            kmer_qt_to_be_cached[key_test] = score
                    else:
                        clustering_scores_dict[key_test] = None

                    # Excluded set
                    key_excl = f"{k}-mers_excluded-set_clust_rank-{rank}_{strategy}"
                    if (gb.HAS_EXCLUDED_SET and f'excl-{strategy}-kmer-clust' in gb.QT_TESTS_TODO 
                            and strategy in excluded_kmer_reps and k in excluded_kmer_reps[strategy]):
                        if key_excl in gb.KMER_QT_CACHE:
                            clustering_scores_dict[key_excl] = gb.KMER_QT_CACHE[key_excl]
                        else:
                            score = cluster_reps(excluded_kmer_reps[strategy][k], gb.EXCLUDED_FULL_TAX_LABELS, rank, dist_measure='euclidean')
                            clustering_scores_dict[key_excl] = score
                            kmer_qt_to_be_cached[key_excl] = score
                    else:
                        clustering_scores_dict[key_excl] = None

        # Perform Clustering of embeddings
        # When rank is in QT_RANK_LEVEL_CLUSTERING_METRICS_AT_RANKS, also extract per-taxon scores
        if strategy in train_embeddings:
            for rank in gb.QT_CLUSTERING_RANKS:
                need_per_taxon = rank in gb.QT_RANK_LEVEL_CLUSTERING_METRICS_AT_RANKS

                # Train set
                if f'train-{strategy}-embed-clust' in gb.QT_TESTS_TODO:
                    want_assignments = need_per_taxon and save_contingency
                    result = cluster_reps(train_embeddings[strategy], gb.TRAIN_FULL_TAX_LABELS, rank, compute_per_taxon=need_per_taxon, return_assignments=want_assignments, max_taxa=gb.QT_MAX_TAXA_PER_RANK_LEVEL_CLUSTERING, dist_measure='cosine')
                    if need_per_taxon:
                        if want_assignments:
                            score_train, taxon_scores, taxon_counts, pred_clusters, true_labels = result
                            save_contingency_matrix(pred_clusters, true_labels, rank, 'train_set', strategy,
                                                    'embedding', batch_num, model_dir, max_taxa=gb.QT_MAX_TAXA_PER_RANK_LEVEL_CLUSTERING)
                        else:
                            score_train, taxon_scores, taxon_counts = result
                        # Store per-taxon scores
                        if rank not in rank_level_clustering_dict:
                            rank_level_clustering_dict[rank] = {}
                        for taxon, score in taxon_scores.items():
                            rank_level_clustering_dict[rank][f"embedding_train-set_{strategy}_{taxon}"] = score
                        for taxon, count in taxon_counts.items():
                            if f"{taxon}_size" not in rank_level_clustering_dict[rank]:
                                rank_level_clustering_dict[rank][f"{taxon}_size"] = count
                    else:
                        score_train = result
                else:
                    score_train = None
                clustering_scores_dict[f"embedding_train-set_clust_rank-{rank}_{strategy}"] = score_train

                # Test set
                if f'test-{strategy}-embed-clust' in gb.QT_TESTS_TODO:
                    want_assignments = need_per_taxon and save_contingency
                    result = cluster_reps(test_embeddings[strategy], gb.TEST_FULL_TAX_LABELS, rank, compute_per_taxon=need_per_taxon, return_assignments=want_assignments, max_taxa=gb.QT_MAX_TAXA_PER_RANK_LEVEL_CLUSTERING, dist_measure='cosine')
                    if need_per_taxon:
                        if want_assignments:
                            score_test, taxon_scores, _, pred_clusters, true_labels = result
                            save_contingency_matrix(pred_clusters, true_labels, rank, 'test_set', strategy,
                                                    'embedding', batch_num, model_dir, max_taxa=gb.QT_MAX_TAXA_PER_RANK_LEVEL_CLUSTERING)
                        else:
                            score_test, taxon_scores, _ = result
                        if rank not in rank_level_clustering_dict:
                            rank_level_clustering_dict[rank] = {}
                        for taxon, score in taxon_scores.items():
                            rank_level_clustering_dict[rank][f"embedding_test-set_{strategy}_{taxon}"] = score
                    else:
                        score_test = result
                else:
                    score_test = None
                clustering_scores_dict[f"embedding_test-set_clust_rank-{rank}_{strategy}"] = score_test

                # Excluded set
                if gb.HAS_EXCLUDED_SET and f'excl-{strategy}-embed-clust' in gb.QT_TESTS_TODO and strategy in excluded_embeddings:
                    score_excl = cluster_reps(excluded_embeddings[strategy], gb.EXCLUDED_FULL_TAX_LABELS, rank, dist_measure='cosine')
                else:
                    score_excl = None
                clustering_scores_dict[f"embedding_excluded-set_clust_rank-{rank}_{strategy}"] = score_excl

    part_end_time = time.time()
    if verbose:
        print(f"  Took: {(part_end_time - part_start_time) * 1000:.0f}ms")


    # Perform Classification -----------------------------------------
    if verbose:
        print("Evaluating classification performance...")
    part_start_time = time.time()

    # Prepare dictionary of sets of taxon labels at each rank
    train_set_of_taxon_labels_at_rank_dict = {rank: set(gb.TRAIN_LIST_OF_TAXON_LABELS_AT_RANK_DICT[rank]) for rank in gb.QT_CLASSIFICATION_RANKS}
    # Missclassification recording is embedding-only (not k-mer classification).
    ranks_to_record_missclassifications = get_missclassification_ranks_to_record(gb.QT_CLASSIFICATION_RANKS)

    for strategy in QT_SEQ_SELECTION_STRATEGIES:
        # Perform Classification of k-mer representations
        if gb.QT_GET_KMER_RESULTS and strategy in train_kmer_reps:
            # Missclassification CSV recording is intentionally disabled for k-mer classification.
            for k in gb.QT_KMER_K_VALUES:
                # --- Train set (self-match blocked) ---
                key_base_train = f"{k}-mers_train-set_class_rank-{{}}_{strategy}"
                macro_key_base_train = f"macro_{k}-mers_train-set_class_rank-{{}}_{strategy}"
                # Cache-key version for train-set classification with self-match exclusion.
                # Keeps old cache entries (self-classification) from being reused accidentally.
                cache_key_base_train = f"{k}-mers_train-set_class_no-self_rank-{{}}_{strategy}"
                macro_cache_key_base_train = f"macro_{k}-mers_train-set_class_no-self_rank-{{}}_{strategy}"
                if f'train-{strategy}-kmer-class' in gb.QT_TESTS_TODO:
                    use_cache = all(
                        macro_cache_key_base_train.format(r) in gb.KMER_QT_CACHE
                        and cache_key_base_train.format(r) in gb.KMER_QT_CACHE
                        for r in gb.QT_CLASSIFICATION_RANKS
                    )
                    if use_cache:
                        classify_scores = {r: gb.KMER_QT_CACHE[cache_key_base_train.format(r)] for r in gb.QT_CLASSIFICATION_RANKS}
                        macro_scores = {r: gb.KMER_QT_CACHE[macro_cache_key_base_train.format(r)] for r in gb.QT_CLASSIFICATION_RANKS}
                    else:
                        classify_scores, macro_scores, _ = classify_reps(train_kmer_reps[strategy][k], train_kmer_reps[strategy][k], gb.TRAIN_FULL_TAX_LABELS, gb.TRAIN_FULL_TAX_LABELS, train_set_of_taxon_labels_at_rank_dict, gb.QT_CLASSIFICATION_RANKS, dist_measure='euclidean', exclude_self_matches=True)
                        for r in gb.QT_CLASSIFICATION_RANKS:
                            kmer_qt_to_be_cached[cache_key_base_train.format(r)] = classify_scores[r]
                            kmer_qt_to_be_cached[macro_cache_key_base_train.format(r)] = macro_scores[r]
                    for r in gb.QT_CLASSIFICATION_RANKS:
                        classification_scores_dict[key_base_train.format(r)] = classify_scores[r]
                        macro_classification_scores_dict[macro_key_base_train.format(r)] = macro_scores[r]

                # --- Test set ---
                key_base_test = f"{k}-mers_test-set_class_rank-{{}}_{strategy}"
                macro_key_base_test = f"macro_{k}-mers_test-set_class_rank-{{}}_{strategy}"
                if f'test-{strategy}-kmer-class' in gb.QT_TESTS_TODO:
                    use_cache = all(
                        macro_key_base_test.format(r) in gb.KMER_QT_CACHE
                        and key_base_test.format(r) in gb.KMER_QT_CACHE
                        for r in gb.QT_CLASSIFICATION_RANKS
                    )
                    if use_cache:
                        classify_scores = {r: gb.KMER_QT_CACHE[key_base_test.format(r)] for r in gb.QT_CLASSIFICATION_RANKS}
                        macro_scores = {r: gb.KMER_QT_CACHE[macro_key_base_test.format(r)] for r in gb.QT_CLASSIFICATION_RANKS}
                    else:
                        classify_scores, macro_scores, _ = classify_reps(test_kmer_reps[strategy][k], train_kmer_reps[strategy][k], gb.TEST_FULL_TAX_LABELS, gb.TRAIN_FULL_TAX_LABELS, train_set_of_taxon_labels_at_rank_dict, gb.QT_CLASSIFICATION_RANKS, dist_measure='euclidean')
                        for r in gb.QT_CLASSIFICATION_RANKS:
                            kmer_qt_to_be_cached[key_base_test.format(r)] = classify_scores[r]
                            kmer_qt_to_be_cached[macro_key_base_test.format(r)] = macro_scores[r]
                    for r in gb.QT_CLASSIFICATION_RANKS:
                        classification_scores_dict[key_base_test.format(r)] = classify_scores[r]
                        macro_classification_scores_dict[macro_key_base_test.format(r)] = macro_scores[r]

                # --- Excluded set ---
                key_base_excl = f"{k}-mers_excluded-set_class_rank-{{}}_{strategy}"
                macro_key_base_excl = f"macro_{k}-mers_excluded-set_class_rank-{{}}_{strategy}"
                if (gb.HAS_EXCLUDED_SET and f'excl-{strategy}-kmer-class' in gb.QT_TESTS_TODO 
                        and strategy in excluded_kmer_reps and k in excluded_kmer_reps[strategy]):
                    use_cache = all(
                        macro_key_base_excl.format(r) in gb.KMER_QT_CACHE
                        and key_base_excl.format(r) in gb.KMER_QT_CACHE
                        for r in gb.QT_CLASSIFICATION_RANKS
                    )
                    if use_cache:
                        classify_scores = {r: gb.KMER_QT_CACHE[key_base_excl.format(r)] for r in gb.QT_CLASSIFICATION_RANKS}
                        macro_scores = {r: gb.KMER_QT_CACHE[macro_key_base_excl.format(r)] for r in gb.QT_CLASSIFICATION_RANKS}
                    else:
                        classify_scores, macro_scores, _ = classify_reps(excluded_kmer_reps[strategy][k], train_kmer_reps[strategy][k], gb.EXCLUDED_FULL_TAX_LABELS, gb.TRAIN_FULL_TAX_LABELS, train_set_of_taxon_labels_at_rank_dict, gb.QT_CLASSIFICATION_RANKS, dist_measure='euclidean')
                        for r in gb.QT_CLASSIFICATION_RANKS:
                            kmer_qt_to_be_cached[key_base_excl.format(r)] = classify_scores[r]
                            kmer_qt_to_be_cached[macro_key_base_excl.format(r)] = macro_scores[r]
                    for r in gb.QT_CLASSIFICATION_RANKS:
                        classification_scores_dict[key_base_excl.format(r)] = classify_scores[r]
                        macro_classification_scores_dict[macro_key_base_excl.format(r)] = macro_scores[r]
                else:
                    for r in gb.QT_CLASSIFICATION_RANKS:
                        classification_scores_dict[key_base_excl.format(r)] = None
                        macro_classification_scores_dict[macro_key_base_excl.format(r)] = None

        # Perform Classification of embeddings
        if strategy in train_embeddings:
            # Train set
            if f'train-{strategy}-embed-class' in gb.QT_TESTS_TODO:
                classify_scores, macro_scores, missclassifications_by_rank = classify_reps(train_embeddings[strategy], train_embeddings[strategy], gb.TRAIN_FULL_TAX_LABELS, gb.TRAIN_FULL_TAX_LABELS, train_set_of_taxon_labels_at_rank_dict, gb.QT_CLASSIFICATION_RANKS, dist_measure='cosine', exclude_self_matches=True, test_seq_ids=gb.TRAINING_INDICES, ranks_to_record_missclassifications=ranks_to_record_missclassifications)
                for r in gb.QT_CLASSIFICATION_RANKS:
                    classification_scores_dict[f"embedding_train-set_class_rank-{r}_{strategy}"] = classify_scores[r]
                    macro_classification_scores_dict[f"macro_embedding_train-set_class_rank-{r}_{strategy}"] = macro_scores[r]
                if missclassifications_by_rank:
                    save_classification_missclassifications(missclassifications_by_rank=missclassifications_by_rank, set_name='train_set', strategy=strategy, rep_name='embedding', batch_num=batch_num, model_dir=model_dir)
            
            # Test set
            if f'test-{strategy}-embed-class' in gb.QT_TESTS_TODO:
                classify_scores, macro_scores, missclassifications_by_rank = classify_reps(test_embeddings[strategy], train_embeddings[strategy], gb.TEST_FULL_TAX_LABELS, gb.TRAIN_FULL_TAX_LABELS, train_set_of_taxon_labels_at_rank_dict, gb.QT_CLASSIFICATION_RANKS, dist_measure='cosine', test_seq_ids=gb.TESTING_INDICES, ranks_to_record_missclassifications=ranks_to_record_missclassifications)
                for r in gb.QT_CLASSIFICATION_RANKS:
                    classification_scores_dict[f"embedding_test-set_class_rank-{r}_{strategy}"] = classify_scores[r]
                    macro_classification_scores_dict[f"macro_embedding_test-set_class_rank-{r}_{strategy}"] = macro_scores[r]
                if missclassifications_by_rank:
                    save_classification_missclassifications(missclassifications_by_rank=missclassifications_by_rank, set_name='test_set', strategy=strategy, rep_name='embedding', batch_num=batch_num, model_dir=model_dir)

            # Excluded set
            if gb.HAS_EXCLUDED_SET and f'excl-{strategy}-embed-class' in gb.QT_TESTS_TODO and strategy in excluded_embeddings:
                classify_scores, macro_scores, missclassifications_by_rank = classify_reps(excluded_embeddings[strategy], train_embeddings[strategy], gb.EXCLUDED_FULL_TAX_LABELS, gb.TRAIN_FULL_TAX_LABELS, train_set_of_taxon_labels_at_rank_dict, gb.QT_CLASSIFICATION_RANKS, dist_measure='cosine', test_seq_ids=gb.EXCLUDED_TAXA_INDICES, ranks_to_record_missclassifications=ranks_to_record_missclassifications)
                for r in gb.QT_CLASSIFICATION_RANKS:
                    classification_scores_dict[f"embedding_excluded-set_class_rank-{r}_{strategy}"] = classify_scores[r]
                    macro_classification_scores_dict[f"macro_embedding_excluded-set_class_rank-{r}_{strategy}"] = macro_scores[r]
                if missclassifications_by_rank:
                    save_classification_missclassifications(missclassifications_by_rank=missclassifications_by_rank, set_name='excluded_set', strategy=strategy, rep_name='embedding', batch_num=batch_num, model_dir=model_dir)
            else:
                for r in gb.QT_CLASSIFICATION_RANKS:
                    classification_scores_dict[f"embedding_excluded-set_class_rank-{r}_{strategy}"] = None
                    macro_classification_scores_dict[f"macro_embedding_excluded-set_class_rank-{r}_{strategy}"] = None

    part_end_time = time.time()
    if verbose:
        print(f"  Took: {(part_end_time - part_start_time) * 1000:.0f}ms")

    qt_end_time = time.time()
    if verbose:
        print(f"Quick test done. Total time taken: {(qt_end_time - qt_start_time):.2f}s")


    # Help free memory
    del training_3bit_seq_reps_, testing_3bit_seq_reps_, excluded_3bit_seq_reps_
    del train_embeddings, test_embeddings, excluded_embeddings
    del train_kmer_reps, test_kmer_reps, excluded_kmer_reps
    del train_seq_embeddings, test_seq_embeddings, excluded_seq_embeddings
    torch.cuda.empty_cache()
    gc.collect()


    # Save new k-mer results to cache
    if kmer_qt_to_be_cached:
        save_kmer_qt_cache(kmer_qt_to_be_cached, gb.DATASET_SPLIT_DIR)

    # Clean results dictionaries -----------------------------------------
    # Remove None values from results dictionaries
    clustering_scores_dict = {k: v for k, v in clustering_scores_dict.items() if v is not None}
    classification_scores_dict = {k: v for k, v in classification_scores_dict.items() if v is not None}
    macro_classification_scores_dict = {k: v for k, v in macro_classification_scores_dict.items() if v is not None}
    ssc_scores_dict = {k: v for k, v in ssc_scores_dict.items() if v is not None}
    
    # Return results dictionaries -----------------------------------------
    return clustering_scores_dict, classification_scores_dict, macro_classification_scores_dict, ssc_scores_dict, rank_level_clustering_dict

def plot_rank_level_clustering(rank_level_clustering_dfs, out_dir, batch_num):
    """
    Generates two plots based on rank-level clustering metrics for each rank:
    1) qt_plots/clustering_per_rank/scatter/{rank_name}_{batch_num}.png
       Scatter plot of Taxon Size (log2) vs Clustering Metric (F1) for the current batch.
    2) qt_plots/clustering_per_rank/r/{rank_name}.png
       Time series of Pearson Correlation Coefficient (r) between log2(Size) and Clustering Metric.
    """
    
    # Check frequency
    if batch_num % gb.RANK_LEVEL_CLUSTERING_PLOTS_EVERY_N_BATCHES != 0:
        return

    for rank, df in rank_level_clustering_dfs.items():
        rank_name = ALL_RANKS_PLURAL[rank]

        # 1. Setup directories
        scatter_dir = os.path.join(out_dir, "clustering_per_rank", "scatter")
        r_dir = os.path.join(out_dir, "clustering_per_rank", "r")
        os.makedirs(scatter_dir, exist_ok=True)
        os.makedirs(r_dir, exist_ok=True)

        # 2. Extract Data Structure
        # We need to group columns by (set, strategy) to plot them as distinct series
        # Pattern: embedding_{set}_{strategy}_{taxon}
        # Taxon size pattern: {taxon}_size

        # Find all score columns
        score_cols = [c for c in df.columns if c.startswith('embedding_')]
        
        # Parse columns to organize data
        # series_map maps (set_name, strategy_name) -> list of (taxon, score_col, size_col)
        series_map = {}
        
        for col in score_cols:
            parts = col.split('_')
            # Expecting: embedding, set, strategy, taxon...
            if len(parts) < 4:
                continue
                
            set_name = parts[1]      # e.g. train-set
            strategy_name = parts[2] # e.g. subseqs
            taxon = "_".join(parts[3:]) # e.g. Bacteria (or taxon_with_underscore)
            
            size_col = f"{taxon}_size"
            if size_col not in df.columns:
                continue
                
            key = (set_name, strategy_name)
            if key not in series_map:
                series_map[key] = []
            series_map[key].append((taxon, col, size_col))

        if not series_map:
            continue

        # Helper to calculate r for a given row (batch) and series
        def get_r_for_series(row, series_data):
            sizes = []
            scores = []
            for _, score_col, size_col in series_data:
                s = row[score_col]
                sz = row[size_col]
                if pd.notna(s) and pd.notna(sz) and sz > 0:
                    sizes.append(np.log2(sz))
                    scores.append(s)
            
            if len(sizes) > 2:
                # Check for variance
                if np.std(sizes) > 0 and np.std(scores) > 0:
                    r, _ = pearsonr(sizes, scores)
                    return r
            return np.nan

        # 3. Plot 1: Scatter Plot (Current Batch)
        current_batch_df = df[df['Batch'] == batch_num]
        if not current_batch_df.empty:
            row = current_batch_df.iloc[0]
            
            plt.figure(figsize=(10, 8))
            
            # Plot each series
            for (set_name, strategy_name), series_data in series_map.items():
                sizes = []
                scores = []
                taxa = []
                
                for taxon, score_col, size_col in series_data:
                    s = row[score_col]
                    sz = row[size_col]
                    if pd.notna(s) and pd.notna(sz) and sz > 0:
                        sizes.append(np.log2(sz))
                        scores.append(s)
                        taxa.append(taxon)
                
                if sizes:
                    label = f"{set_name} {strategy_name}".title().replace('-', ' ')
                    plt.scatter(sizes, scores, alpha=0.6, label=label, edgecolors='w', s=60)
                    
            plt.title(f"Clustering Metrics vs Taxon Size ({rank_name.capitalize()}) - Batch {batch_num}")
            plt.xlabel("Log2(Taxon Size)")
            plt.ylabel("Clustering Metric (F1)")
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.legend()
            plt.tight_layout()
            
            plt.savefig(os.path.join(scatter_dir, f"{rank_name}_{batch_num}.png"), dpi=100)
            plt.close()

        # 4. Plot 2: Correlation Time Series (All Batches)
        plt.figure(figsize=(12, 6))
        
        # Calculate r for each series across all batches
        has_data = False
        for (set_name, strategy_name), series_data in series_map.items():
            rs = []
            batch_indices = []
            
            for idx, row in df.iterrows():
                r = get_r_for_series(row, series_data)
                rs.append(r)
                batch_indices.append(row['Batch'])
                
            # Filter out NaNs for plotting lines
            valid_points = [(b, r) for b, r in zip(batch_indices, rs) if pd.notna(r)]
            if valid_points:
                has_data = True
                bs, rs_clean = zip(*valid_points)
                label = f"{set_name} {strategy_name}".title().replace('-', ' ')
                plt.plot(bs, rs_clean, marker='o', label=label, markersize=3, alpha=0.8)

        if has_data:
            plt.title(f"Correlation (Size vs Clustering) over Time ({rank_name.capitalize()})")
            plt.xlabel("Batch")
            plt.ylabel("Pearson r (log2(Size) vs Metric)")
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(r_dir, f"{rank_name}.png"), dpi=150)
        
        plt.close()

def save_contingency_matrix(predicted_clusters, labels, rank, set_name, strategy,
                            rep_name, batch_num, model_dir, max_taxa=None):
    """
    Build, sort, and save a contingency matrix (cluster assignments vs true labels).
    
    True labels (columns) are sorted by count descending.
    Cluster rows (Cluster_1, Cluster_2, ...) are sorted to approximate a diagonal
    by ordering each cluster according to the position of its dominant true label.
    
    Saves as both CSV and PNG heatmap under:
        qt_results/clustering_per_rank/contingency_matrices/
        qt_plots/clustering_per_rank/contingency_matrices/
    """
    rank_name = ALL_RANKS_PLURAL[rank]

    # Count true labels and sort by count descending
    unique_labels, label_counts = np.unique(labels, return_counts=True)
    col_order = np.argsort(-label_counts)
    if max_taxa is not None and len(col_order) > max_taxa:
        col_order = col_order[:max_taxa]
    sorted_labels = unique_labels[col_order]
    n_cols = len(sorted_labels)
    if n_cols == 0:
        return

    # Build contingency matrix: rows = clusters, columns = sorted true labels
    label_to_col = {lab: ci for ci, lab in enumerate(sorted_labels)}
    unique_clusters = np.unique(predicted_clusters)
    n_clusters = len(unique_clusters)

    matrix = np.zeros((n_clusters, n_cols), dtype=int)
    for ri, cid in enumerate(unique_clusters):
        cluster_labels = labels[predicted_clusters == cid]
        for lab in cluster_labels:
            if lab in label_to_col:
                matrix[ri, label_to_col[lab]] += 1

    # Remove empty rows (clusters with no members in the shown taxa)
    non_empty = matrix.sum(axis=1) > 0
    if not non_empty.any():
        return
    matrix = matrix[non_empty]
    n_clusters = matrix.shape[0]

    # Sort rows to approximate diagonal:
    # Each cluster is placed according to the column of its dominant true label,
    # ties broken by count in that column (descending).
    dominant_col = np.argmax(matrix, axis=1)
    dominant_count = np.array([matrix[i, dominant_col[i]] for i in range(n_clusters)])
    row_order = np.lexsort((-dominant_count, dominant_col))
    matrix = matrix[row_order]

    # Create labels and DataFrame
    row_labels = [f"Cluster_{i+1}" for i in range(n_clusters)]
    df = pd.DataFrame(matrix, index=row_labels, columns=sorted_labels)

    # File naming: clust_cont_matrix_(train_set,phyla,embedding,subseqs)_5000
    base_name = f"clust_cont_matrix_({set_name},{rank_name},{rep_name},{strategy})_{batch_num}"

    # Save CSV
    csv_dir = os.path.join(model_dir, "qt_results", "clustering_per_rank", "contingency_matrices")
    os.makedirs(csv_dir, exist_ok=True)
    df.to_csv(os.path.join(csv_dir, f"{base_name}.csv"))

    # Save PNG heatmap
    plot_dir = os.path.join(model_dir, "qt_plots", "clustering_per_rank", "contingency_matrices")
    os.makedirs(plot_dir, exist_ok=True)

    fig_w = max(8, n_cols * 0.4 + 2)
    fig_h = max(5, n_clusters * 0.3 + 2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # Use log1p for colour scale (handles zeros and large count differences gracefully)
    display_data = np.log1p(matrix.astype(float))
    im = ax.imshow(display_data, aspect='auto', cmap='Blues', interpolation='nearest')

    # Annotate cells with counts if matrix is small enough
    if n_clusters <= 30 and n_cols <= 30:
        threshold = display_data.max() * 0.6
        for ri in range(n_clusters):
            for ci in range(n_cols):
                val = matrix[ri, ci]
                if val > 0:
                    colour = 'white' if display_data[ri, ci] > threshold else 'black'
                    ax.text(ci, ri, str(val), ha='center', va='center', fontsize=6, color=colour)

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(sorted_labels, rotation=90, fontsize=7)
    ax.set_yticks(range(n_clusters))
    ax.set_yticklabels(row_labels, fontsize=7)
    ax.set_xlabel("True Label")
    ax.set_ylabel("Cluster Assignment")
    title_set = set_name.replace('_', ' ').title()
    ax.set_title(f"Contingency Matrix ({title_set}, {rank_name.capitalize()}, {strategy}) - Batch {batch_num}")
    fig.colorbar(im, ax=ax, label='log(count + 1)')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"{base_name}.png"), dpi=150, bbox_inches='tight')
    plt.close('all')

def visualise_ordination(points, full_tax_labels, rank, out_dir, ord_type, batch_num=None, rep_name="unnamed", set_name="unnamed", max_points=10000, skip_batch_num=False):
    """
    Visualize ordination results (PCoA or UMAP).
    """
    
    # Extract labels at the specified rank
    labels = [tax_label[rank] for tax_label in full_tax_labels]
    
    # Subsample points if we have more than max_points
    n_pts = points.shape[0]
    if max_points is not None and n_pts > max_points:
        idxs = np.random.choice(n_pts, max_points, replace=False)
        points = points[idxs]
        labels = [labels[i] for i in idxs]

    # Count label frequencies in current data to prioritize biggest classes
    label_counts = pd.Series(labels).value_counts()
    sorted_labels = label_counts.index.tolist()

    def generate_colors(n_colors):
        """Generate a list of distinct colors. """
        distinct_colors = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231',  '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe',  '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000',  '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080',  '#ffffff', '#000000']
        if n_colors <= len(distinct_colors):
            return distinct_colors[:n_colors]
        # If more colors needed, start with the distinct ones
        colors = list(distinct_colors)
        # Then generate the rest from a spectral map
        remaining = n_colors - len(distinct_colors)
        spectral_colors = plt.cm.nipy_spectral(np.linspace(0.05, 0.95, remaining))
        colors.extend(spectral_colors)
        return colors

    # Retrieve existing cache for this rank, if any
    cached = gb.ORDINATION_LABEL_COLOR_CACHE.get(rank)
    if cached:
        unique_labels, label_to_idx, cmap = cached
        
        # See if we've encountered new labels this time around
        new_labels = [lab for lab in sorted_labels if lab not in label_to_idx]
        if new_labels:
            # Append without mutating cached list in-place
            unique_labels = unique_labels + new_labels
            
            # Re-create label_to_idx
            label_to_idx = {lab: i for i, lab in enumerate(unique_labels)}
            
            # Re-generate colormap to accommodate new labels
            # We want to try and keep the biggest classes distinct
            colors = generate_colors(len(unique_labels))
            cmap = ListedColormap(colors)
            
            gb.ORDINATION_LABEL_COLOR_CACHE[rank] = (unique_labels, label_to_idx, cmap)
    else:
        # First time seeing this rank -> build and cache
        unique_labels = sorted_labels # Use sorted labels so most frequent get first colors
        label_to_idx = {lab: i for i, lab in enumerate(unique_labels)}
        
        colors = generate_colors(len(unique_labels))
        cmap = ListedColormap(colors)
        
        gb.ORDINATION_LABEL_COLOR_CACHE[rank] = (unique_labels, label_to_idx, cmap)

    # Map each label to its numeric index
    class_indices = np.array([label_to_idx[lab] for lab in labels])

    # Plot
    plt.figure(figsize=(10, 8))
    plt.scatter(points[:, 0], points[:, 1], c=class_indices, cmap=cmap, alpha=0.6)
    plt.title(f'{ord_type} of {set_name.title()} Set Embeddings at {ALL_RANKS[rank].capitalize()} Level')
    plt.xlabel('First Component')
    plt.ylabel('Second Component')

    # Legend for the top N classes in this specific plot (not all cached classes)
    # We use label_counts from this plot's data
    unique_labels_in_plot = label_counts.index.tolist()
    
    # Only show legend if <= 15 classes in this plot
    if len(unique_labels_in_plot) <= 15:
        # Get colors from the global cmap using the global indices
        plot_colors = [cmap.colors[label_to_idx[lab]] for lab in unique_labels_in_plot]
        handles = [
            plt.Line2D([], [], marker='o', color=plot_colors[i], linestyle='None',
                       label=lab, alpha=0.6)
            for i, lab in enumerate(unique_labels_in_plot)
        ]
        plt.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    # Save
    time_or_batch = batch_num if batch_num is not None else datetime.now().strftime("%H%M%S")
    time_or_batch = '' if skip_batch_num else '_' + str(time_or_batch)
    os.makedirs(f'{out_dir}/{rep_name}_{ord_type.lower()}_plots/', exist_ok=True)
    plt.savefig(f'{out_dir}/{rep_name}_{ord_type.lower()}_plots/{set_name}_{ord_type.lower()}_{rank}{time_or_batch}.jpg',
            bbox_inches='tight', dpi=75)
    # Better quality:
    # plt.savefig(f'{out_dir}/{rep_name}_{ord_type.lower()}_plots/{set_name}_{ord_type.lower()}_{rank}_{time_or_batch}.png',
    #         bbox_inches='tight', dpi=250)
    plt.close('all')


def cluster_reps(reps, full_tax_labels, rank, compute_per_taxon=False, return_assignments=False, max_taxa=100, dist_measure='cosine'):
    """
    Cluster representations and compute clustering quality scores.
    
    Performs K-Means clustering once and extracts both the overall V-measure score
    and optionally per-taxon scores from the same clustering result.
    
    For per-taxon scores, we use a Precision/Recall approach based on the dominant 
    cluster assigned to each taxon. This is O(N) and much faster than Silhouette scores.

    Args:
        reps: numpy array of shape (N_SEQS, DIM)
        full_tax_labels: list of full taxonomic labels shape: (N_SEQS,)
            Each element is a list of 7 strings
        rank: taxonomic rank (integer) 0=domain, 1=phylum, etc.
        compute_per_taxon: if True, also compute per-taxon scores
        return_assignments: if True (requires compute_per_taxon=True), also return
            predicted_clusters and labels arrays for downstream analysis
        max_taxa: maximum number of taxa to include in results (sorted by count, descending)
        dist_measure: distance measure for clustering ('cosine' or 'euclidean')

    Returns:
        If compute_per_taxon=False:
            v_measure: float, the V-measure clustering score
        If compute_per_taxon=True and return_assignments=False:
            v_measure: float, the V-measure clustering score
            taxon_scores: dict mapping taxon_name -> F1 clustering score (float)
            taxon_counts: dict mapping taxon_name -> number of sequences (int)
        If compute_per_taxon=True and return_assignments=True:
            (as above, plus)
            predicted_clusters: numpy int32 array of cluster IDs per sequence
            labels: numpy string array of true labels at the given rank
    """
    # Row-wise L2 normalisation helper.
    # Using float32 keeps memory/compute low while being numerically stable enough for eval.
    def l2_normalize_rows(x, eps=1e-8):
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        return x / np.clip(norms, eps, None)

    if dist_measure not in ['cosine', 'euclidean']:
        raise ValueError(f"dist_measure must be 'cosine' or 'euclidean'. Got: {dist_measure}")

    # Extract labels at the specified rank from full taxonomic labels
    labels = np.array([tax_label[rank] for tax_label in full_tax_labels])
    reps = reps.astype(np.float32, copy=False)

    # Find centroids to start K-Means.
    # We initialise from per-label means to mirror prior behaviour.
    unique_labels, inverse, counts = np.unique(labels, return_inverse=True, return_counts=True)
    n_unique = len(unique_labels)
    centroids = np.zeros((n_unique, reps.shape[1]), dtype=np.float32)
    np.add.at(centroids, inverse, reps)
    centroids /= counts[:, None].astype(np.float32)

    if dist_measure == 'cosine':
        # Spherical K-Means:
        # assignment -> argmax cosine similarity; update -> mean then renormalise.
        reps_norm = l2_normalize_rows(reps)
        centroids = l2_normalize_rows(centroids)
        predicted_clusters = np.full(reps_norm.shape[0], -1, dtype=np.int32)
        for _ in range(75):
            sims = reps_norm @ centroids.T
            new_clusters = np.argmax(sims, axis=1).astype(np.int32)
            if np.array_equal(new_clusters, predicted_clusters):
                break
            predicted_clusters = new_clusters

            new_centroids = np.zeros_like(centroids)
            cluster_sizes = np.bincount(predicted_clusters, minlength=n_unique)
            np.add.at(new_centroids, predicted_clusters, reps_norm)

            nonempty = cluster_sizes > 0
            if np.any(nonempty):
                new_centroids[nonempty] /= cluster_sizes[nonempty, None].astype(np.float32)
                new_centroids[nonempty] = l2_normalize_rows(new_centroids[nonempty])

            # Keep old centroids for empty clusters (rare, but robust).
            if np.any(~nonempty):
                new_centroids[~nonempty] = centroids[~nonempty]

            centroids = new_centroids
    else:
        # Euclidean K-Means (Lloyd), seeded from per-label means.
        kmeans = KMeans(n_clusters=n_unique, n_init=1, init=centroids, random_state=24, max_iter=75, tol=1e-3, algorithm='lloyd')
        predicted_clusters = kmeans.fit_predict(reps)

    # Calculate V-measure score comparing true labels and predicted clusters
    v_measure = v_measure_score(labels, predicted_clusters)

    # Return early if per-taxon scores not needed
    if not compute_per_taxon:
        return v_measure

    # Compute per-taxon metrics using Precision/Recall on predicted clusters
    # This captures the "V-measure" philosophy (harmonic mean of purity and completeness)
    # but is extremely efficient to compute.
    taxon_scores = {}
    taxon_counts = {}
    
    # Precompute cluster sizes
    cluster_sizes = np.bincount(predicted_clusters, minlength=n_unique)
    
    # Sort taxa by count (descending) and limit to max_taxa
    sorted_indices = np.argsort(-counts)
    if max_taxa is not None and len(sorted_indices) > max_taxa:
        sorted_indices = sorted_indices[:max_taxa]

    for idx in sorted_indices:
        taxon_name = unique_labels[idx]
        taxon_count = counts[idx]
        
        # Get clusters for this taxon
        cls_in_taxon = predicted_clusters[labels == taxon_name]
        
        # Find the dominant cluster for this taxon
        # (The cluster where the majority of this taxon's sequences ended up)
        cluster_counts = np.bincount(cls_in_taxon, minlength=n_unique)
        dominant_cluster = np.argmax(cluster_counts)
        
        # Recall (Completeness): Proportion of the taxon in its dominant cluster
        recall = cluster_counts[dominant_cluster] / taxon_count
        
        # Precision (Homogeneity): Proportion of that cluster belonging to this taxon
        precision = cluster_counts[dominant_cluster] / cluster_sizes[dominant_cluster]
        
        # F1 score: Harmonic mean of Precision and Recall
        # This provides a taxon-specific clustering quality score [0, 1]
        f1 = 2 * (precision * recall) / (precision + recall + 1e-9)
        
        taxon_scores[taxon_name] = float(f1)
        taxon_counts[taxon_name] = int(taxon_count)

    if return_assignments:
        return v_measure, taxon_scores, taxon_counts, predicted_clusters, labels
    return v_measure, taxon_scores, taxon_counts


def classify_reps(test_reps, train_reps, test_full_tax_labels, train_full_tax_labels, train_set_of_taxon_labels_at_rank_dict, ranks, dist_measure='cosine', exclude_self_matches=False, test_seq_ids=None, ranks_to_record_missclassifications=None):
    """
    Classify representations using nearest neighbor classification across all given taxonomic ranks.

    Test = Query
    Train = Target

    For each test representation, the nearest training representation is found.
    The predicted label at the given rank is taken from the training representation's taxonomic labels.
    Only test samples whose true label at a given rank is in the training label set (train_set_of_taxon_labels_at_rank_dict)
    are counted when computing accuracies.
    
    The function returns two dictionaries:
      - The first contains the overall accuracy for each rank (the proportion of correct predictions over valid test samples).
            e.g. {0: 0.95, 1: 0.90, 2: 0.85}
      - The second contains the macro accuracy for each rank (the average of per-label accuracies over the labels present in the training set).
            e.g. {0: 0.95, 1: 0.90, 2: 0.85}
      
    Args:
        test_reps (np.ndarray): Array of test representations with shape (N_test, DIM).
        train_reps (np.ndarray): Array of train representations with shape (N_train, DIM).
        test_full_tax_labels (list): List of full taxonomic labels for test samples; each element is a list of taxonomic labels.
        train_full_tax_labels (list): List of full taxonomic labels for train samples; each element is a list of taxonomic labels.
        train_set_of_taxon_labels_at_rank_dict (dict): Dictionary mapping rank (int) to a set of valid training labels.
        ranks (list): List of taxonomic ranks (integers) to assess (e.g. [0, 1, 2, 3, 4, 5]).
        dist_measure (str): Distance measure for nearest neighbour lookup ('cosine' or 'euclidean').
        exclude_self_matches (bool): If True, blocks query i from matching target i.
            This is for train-set leave-one-out style evaluation where test and train are aligned.
        test_seq_ids (list or np.ndarray, optional): Sequence IDs aligned with test_reps.
            If None, uses positional indices [0..N_test-1].
        ranks_to_record_missclassifications (list/tuple/set, optional): Subset of ranks to
            record missclassified sequence rows for.
        
    Returns:
        tuple: (classify_score_dict, macro_classify_score_dict, missclassifications_by_rank)
            classify_score_dict: dict mapping each rank to overall accuracy (float) or None if no valid test samples.
            macro_classify_score_dict: dict mapping each rank to macro accuracy (float) or None if no valid test samples.
            missclassifications_by_rank: dict mapping rank -> {
                "total_num_seqs": int,
                "rows": list of [sequence_index, true_taxon, predicted_taxon]
            }
    """
    if dist_measure not in ['cosine', 'euclidean']:
        raise ValueError(f"dist_measure must be 'cosine' or 'euclidean'. Got: {dist_measure}")
    if exclude_self_matches and test_reps.shape[0] != train_reps.shape[0]:
        raise ValueError("exclude_self_matches=True requires aligned test/train sets with the same length.")
    n_test = len(test_full_tax_labels)
    if test_seq_ids is not None and len(test_seq_ids) != n_test:
        raise ValueError("test_seq_ids must have the same length as test_full_tax_labels.")

    if ranks_to_record_missclassifications is None:
        ranks_to_record_missclassifications = []
    ranks_to_record_missclassifications = set(ranks_to_record_missclassifications)
    missclassifications_by_rank = {
        rank: {"total_num_seqs": 0, "rows": []}
        for rank in ranks_to_record_missclassifications
        if rank in ranks
    }

    if exclude_self_matches and train_reps.shape[0] < 2:
        classify_score_dict = {rank: None for rank in ranks}
        macro_classify_score_dict = {rank: None for rank in ranks}
        return classify_score_dict, macro_classify_score_dict, missclassifications_by_rank

    # Compute nearest neighbours using the chosen distance measure.
    test_reps = test_reps.astype(np.float32, copy=False)
    train_reps = train_reps.astype(np.float32, copy=False)
    if dist_measure == 'cosine':
        # Minimising cosine distance is equivalent to maximising cosine similarity.
        test_reps = test_reps / np.clip(np.linalg.norm(test_reps, axis=1, keepdims=True), 1e-8, None)
        train_reps = train_reps / np.clip(np.linalg.norm(train_reps, axis=1, keepdims=True), 1e-8, None)
        sims = test_reps @ train_reps.T
        if exclude_self_matches:
            np.fill_diagonal(sims, -np.inf)
        nearest_indices = np.argmax(sims, axis=1)  # Shape: (N_test,)
    else:
        dists = cdist(test_reps, train_reps, metric='sqeuclidean')
        if exclude_self_matches:
            np.fill_diagonal(dists, np.inf)
        nearest_indices = np.argmin(dists, axis=1)  # Shape: (N_test,)
    
    # Get predicted labels for each test sample based on its nearest training neighbor.
    predicted_labels = [train_full_tax_labels[idx] for idx in nearest_indices]
    
    classify_score_dict = {}
    macro_classify_score_dict = {}
    
    # Process each rank separately.
    for rank in ranks:
        correct_count = 0
        valid_count = 0
        per_label_total = {}    # Count of test samples per label.
        per_label_correct = {}  # Count of correct predictions for each label.
        
        # Iterate over all test samples.
        for i in range(n_test):
            true_label = test_full_tax_labels[i][rank]
            # Skip test samples whose true label is not present in the training set at this rank.
            if true_label not in train_set_of_taxon_labels_at_rank_dict[rank]:
                continue  
            valid_count += 1
            per_label_total[true_label] = per_label_total.get(true_label, 0) + 1
            pred_label = predicted_labels[i][rank]
            if true_label == pred_label:
                correct_count += 1
                per_label_correct[true_label] = per_label_correct.get(true_label, 0) + 1
            else:
                per_label_correct[true_label] = per_label_correct.get(true_label, 0)
                if rank in missclassifications_by_rank:
                    seq_id = i if test_seq_ids is None else test_seq_ids[i]
                    missclassifications_by_rank[rank]["rows"].append([int(seq_id), true_label, pred_label])
        
        # Calculate overall accuracy for this rank.
        if valid_count > 0:
            overall_accuracy = correct_count / valid_count
        else:
            overall_accuracy = None

        classify_score_dict[rank] = overall_accuracy
        if rank in missclassifications_by_rank:
            missclassifications_by_rank[rank]["total_num_seqs"] = int(valid_count)
        
        # Calculate macro accuracy (average accuracy over all labels present in the training set that appear in the test set).
        per_label_accuracies = []
        for label in train_set_of_taxon_labels_at_rank_dict[rank]:
            if label in per_label_total:
                label_acc = per_label_correct[label] / per_label_total[label]
                per_label_accuracies.append(label_acc)
        if per_label_accuracies:
            macro_accuracy = sum(per_label_accuracies) / len(per_label_accuracies)
        else:
            macro_accuracy = None
        macro_classify_score_dict[rank] = macro_accuracy
    
    # # Print some comparisons of the true and predicted labels
    # for i in range(10):
    #     print(f"True label: {test_full_tax_labels[i]}")
    #     print(f"Predicted label: {predicted_labels[i]}")
    #     usr_input = input("Press Enter to continue. Or s to skip: ")
    #     if usr_input == 's':
    #         break
    # # Print accuracy for the ranks
    # for rank in ranks:
    #     print(f"Accuracy for rank {rank}: {classify_score_dict[rank]}")
    #     print(f"Macro accuracy for rank {rank}: {macro_classify_score_dict[rank]}")
    # input()

    return classify_score_dict, macro_classify_score_dict, missclassifications_by_rank

def test_subsequence_congruency(reps, n_samples=200, n_replicates=20, exclude_full=True):
    """
    Calculate the Subsequence Congruency (SSC) score.

    This test measures how robust sequence representations are to the region of the 
    full-length 16S gene they originate from. It quantifies how close embeddings of 
    the same sequence across different regions are, relative to embeddings of different sequences.

    Args:
        reps (np.ndarray): Representations of shape (N_REGIONS, N_SEQS, DIM).
        n_samples (int): Number of sequences to sample in each replicate.
        n_replicates (int): Number of sampling replicates to perform.
        exclude_full (bool): Whether to exclude the full sequence (at index 0) from regions.

    Returns:
        tuple: A tuple containing the mean and standard deviation of the SSC ratio.
    """
    if reps.ndim != 3:
        raise ValueError("`reps` must be a 3D array of shape (N_REGIONS, N_SEQS, DIM).")

    n_regions, n_seqs, dim = reps.shape

    subsequence_reps = reps[1:, :, :] if exclude_full else reps
    n_regions_to_use = subsequence_reps.shape[0]

    if n_regions_to_use < 2:
        # Not enough regions to compare for the 'same' sequence part.
        return (np.nan, np.nan)
    
    if n_seqs < 2 * n_samples:
        print(f"Warning: Not enough sequences for SSC sampling (need {2*n_samples}, have {n_seqs}). Skipping test.")
        return (np.nan, np.nan)

    subseq_congruencies = []
    all_seq_indices = np.arange(n_seqs)

    for _ in range(n_replicates):
        np.random.shuffle(all_seq_indices)
        
        # --- "Same" sequence distances ---
        # Average distance between different region representations of the *same* sequence.
        sample_indices_same = all_seq_indices[:n_samples]
        
        mean_dists_same_list = []
        for seq_idx in sample_indices_same:
            reps_for_one_seq = subsequence_reps[:, seq_idx, :]
            dists = cdist(reps_for_one_seq, reps_for_one_seq, metric='sqeuclidean')
            # Get mean of upper triangle, ignoring diagonal
            mean_dists_same_list.append(np.mean(dists[np.triu_indices(n_regions_to_use, k=1)]))
        
        mean_distance_same_sequence = np.mean(mean_dists_same_list) if mean_dists_same_list else 0

        # --- "Other" sequences distances ---
        # Average distance between representations of *different* sequences.
        sample_indices_1 = all_seq_indices[:n_samples]
        sample_indices_2 = all_seq_indices[n_samples:2*n_samples]

        # Get all region representations for the two sets of sequences
        reps1 = subsequence_reps[:, sample_indices_1, :]
        reps2 = subsequence_reps[:, sample_indices_2, :]

        # Flatten to get lists of vectors
        reps1_flat = reps1.reshape(-1, dim)
        reps2_flat = reps2.reshape(-1, dim)
        
        other_dists = cdist(reps1_flat, reps2_flat, metric='sqeuclidean')
        mean_distance_other_sequences = np.mean(other_dists)

        # --- Ratio & SSC ---
        if mean_distance_other_sequences > 0:
            # Calculate the subsequence congruency
            # SSC = (other - same) / other
            # High values indicate high congruency. 1.0 is perfect congruency.
            epsilon = 0.00001
            ssc = (mean_distance_other_sequences - mean_distance_same_sequence) / (mean_distance_other_sequences + epsilon)
            subseq_congruencies.append(ssc)

    if not subseq_congruencies:
        return (np.nan, np.nan)

    return (np.mean(subseq_congruencies), np.std(subseq_congruencies))

def add_results_df(scores_dict, batch_num, existing_df=None):
    """
    Add quick test results into a dataframe to track performance across training batches.

    This function combines the clustering, classification, and macro classification scores
    from a quick test into a single row, using the provided batch number as an identifier.
    If an existing dataframe is passed in, the new row is appended; otherwise, a new dataframe is created.

    Args:
        scores_dict (dict): Dictionary of scores.
            Keys are strings (e.g. "6-mers_train-set_clust_rank-0_subseqs" or "embedding_test-set_class_rank-3_full") 
            and values are floats, "ditto", or None.
        batch_num (int or str): The batch number or identifier for the current quick test results.
        existing_df (pd.DataFrame, optional): An existing dataframe to which the new row should be appended.
            If None, a new dataframe is created.

    Returns:
        pd.DataFrame: The updated dataframe containing the new quick test results row.
    """

    # Create a dictionary for the new row with the batch number.
    row_dict = {"Batch": batch_num}
    # Merge all three result dictionaries into one.
    row_dict.update(scores_dict)
    
    for key, value in row_dict.items():
        if value == "ditto":
            if existing_df is None or existing_df.empty:
                raise ValueError(f"Received 'ditto' for column '{key}' but no existing dataframe to copy from.")
            
            last_row = existing_df.iloc[-1]
            
            if key not in last_row or pd.isna(last_row.get(key)):
                raise ValueError(f"Cannot use 'ditto' for column '{key}': previous row has no value.")
            
            row_dict[key] = last_row[key]

    # Create a new dataframe row from the combined dictionary.
    new_row_df = pd.DataFrame([row_dict])
    
    # If an existing dataframe is provided, append the new row.
    if existing_df is not None:
        updated_df = pd.concat([existing_df, new_row_df], ignore_index=True)
    else:
        updated_df = new_row_df
    
    return updated_df


def print_quick_test(clustering_scores_dict, classification_scores_dict, macro_classification_scores_dict, ssc_scores_dict):
    """
    Print a concise, clear summary of the quick test results.

    This function prints out the clustering scores, classification scores, and macro
    classification scores from the quick test. Each section is printed with keys and
    their corresponding scores. If a score is absent (None), it is shown as an empty space.
    The function handles missing results gracefully by skipping sections with no entries.

    Args:
        clustering_scores_dict (dict): Dictionary containing clustering scores.
            Keys are strings (e.g. "6-mers_train-set_clust_rank-0_subseqs")
            and values are floats or None.
        classification_scores_dict (dict): Dictionary containing classification scores.
            Keys are strings (e.g. "embedding_excluded-set_class_rank-2_full")
            and values are floats or None.
        macro_classification_scores_dict (dict): Dictionary containing macro classification scores.
            Keys are strings (e.g. "macro_embedding_test-set_class_rank-1_subseqs") and values are floats or None.
        ssc_scores_dict (dict): Dictionary containing subsequence congruency scores.
            Keys are strings (e.g. "embedding_train-set_ssc_mean") and values are floats or None.
    """
    if gb.QT_PRINT_RESULTS:
        # Header for the summary
        print("\nQuick Test Results Summary --------------------------")
        
        # Define a helper function to print a section of results
        def print_section(title, results_dict):
            print(f"----- {title} -----")
            if results_dict:
                # Determine the maximum key length for nice alignment
                max_key_length = max(len(key) for key in results_dict.keys())
                # Sort keys to have a consistent order in the printout
                for key in sorted(results_dict.keys()):
                    resultt = results_dict[key]

                    # If the result is a score, print it
                    if isinstance(resultt, (int, float)): # ("ditto" and None are ignored)
                        # Format score if available, else leave empty
                        score_str = f"{resultt:.3f}" if resultt is not None else ""
                        print(f"{key.ljust(max_key_length)} : {score_str}")

                    # If the result is a dictionary (per rank scores)
                    elif isinstance(resultt, dict):
                        # If the result is a dictionary, print each key-value pair
                        for sub_key, sub_result in resultt.items():
                            if isinstance(sub_result, (int, float)): # ("ditto" and None are ignored)
                                # Format score if available, else leave empty
                                sub_score_str = f"{sub_result:.3f}" if sub_result is not None else ""
                                print_key = f"{key}-{ALL_RANKS[sub_key]}"
                                print(f"{print_key.ljust(max_key_length)} : {sub_score_str}")
            else:
                print("No results available.")

        # Print each section using the helper function
        print_section("Clustering Scores", clustering_scores_dict)
        print_section("Classification Scores", classification_scores_dict)
        print_section("Macro Classification Scores", macro_classification_scores_dict)
        print_section("Subsequence Congruency Scores", ssc_scores_dict)


def plot_quick_test(df, model_dir, column_name, name="", show_kmer_equivalents=False, show=False):
    """
    Plot quick test results with batch number on x-axis and scores on y-axis.
    Plots lines for all columns matching the `column_name` pattern (which can include '*').
    The legend label for each line is derived from the part(s) of the column name
    matching the wildcard(s).

    Example column_name patterns:
       - embedding_excluded-set_clust_rank-0_full  (no wildcard, clustering)
       - embedding_train-set_class_rank-*_full     (wildcard for rank, classification)
       - *_train-set_clust_rank-0_both      (wildcard for rep_name, clustering)
       - embedding_*-set_class_rank-0_subseqs      (wildcard for set, classification)
       - 6-mers_excluded-set_clust_rank-0_*        (wildcard for strategy, clustering)
       - macro_embedding_test-set_class_rank-*_subseqs  (macro classification)

    """
    if df is None or len(df) == 0:
        return

    # Convert the fnmatch-style pattern to a regex pattern
    # Escape regex special characters in the original pattern, then replace '*' with '(.*?)'
    regex_pattern = '^' + re.escape(column_name).replace('\\*', '(.*?)') + '$'

    matching_columns = []
    for col in df.columns:
        if col == 'Batch':
            continue
        match = re.match(regex_pattern, col)
        if match:
            # Extract the part(s) corresponding to the wildcard(s)
            wildcard_parts = match.groups()
            # Create a label from the wildcard parts
            label_part = " ".join(wildcard_parts) if wildcard_parts else '' # Use empty string if no wildcard
            matching_columns.append((col, label_part))

    # No matching columns means this plot is not relevant for the current run
    # (e.g. no test/excluded set in the loaded dataset split).
    if not matching_columns:
        return

    plt.figure(figsize=(12, 6))

    # Determine the representation name from the column name
    processed_column_name = column_name
    if processed_column_name.startswith('macro_'):
        processed_column_name = processed_column_name[len('macro_'):]
    representation_name = processed_column_name.split('_')[0] + 's'
    representation_name = 'Representations' if representation_name == '*' else representation_name
    # Plot each matching column
    for column, label_part in matching_columns:
        
        # Calculate best score for this column
        # Check if column exists and has non-NA values before calling max() or min()
        if column in df and not df[column].isnull().all():
            # What is "best" maximum or minimum?
            max_or_min = 'max' # "Best" is always the highest score
            if max_or_min == 'max':
                best_score = df[column].max()
                label = f"{label_part} {representation_name} (max={best_score:.3f})"
            elif max_or_min == 'min':
                best_score = df[column].min()
                label = f"{label_part} {representation_name} (min={best_score:.3f})"
        else:
            label = f"{label_part} {representation_name} (no data)" # Handle empty or all-NA columns
        label = label.title() # Title case
        label = label.replace('-', ' ') # Replace dashes with spaces

        # Only show markers if less than 20 data points
        if len(df) <= 20:
            line, = plt.plot(df['Batch'].astype(int), df[column], marker='o', label=label, linewidth=2, markersize=4)
        else:
            line, = plt.plot(df['Batch'].astype(int), df[column], label=label, linewidth=2)

        if show_kmer_equivalents and 'embedding' in column:
            # Handle both regular and macro columns
            if column.startswith('macro_embedding'):
                suffix = column[len('macro_embedding'):]
                kmer_equivalents = [c for c in df.columns if c.startswith('macro_') and c.endswith(suffix) and '-mers' in c]
            elif column.startswith('embedding'):
                suffix = column[len('embedding'):]
                kmer_equivalents = [c for c in df.columns if not c.startswith('macro_') and c.endswith(suffix) and '-mers' in c]
            else:
                suffix = None
                kmer_equivalents = []

            if kmer_equivalents:
                best_kmer_col = None
                best_kmer_score = -1

                max_or_min = 'max' # "Best" is always the highest score

                for kmer_col in kmer_equivalents:
                    if kmer_col in df and not df[kmer_col].isnull().all():
                        score = df[kmer_col].dropna().iloc[0]
                        if best_kmer_col is None:
                            best_kmer_score = score
                            best_kmer_col = kmer_col
                        elif max_or_min == 'max' and score > best_kmer_score:
                            best_kmer_score = score
                            best_kmer_col = kmer_col
                        elif max_or_min == 'min' and score < best_kmer_score:
                            best_kmer_score = score
                            best_kmer_col = kmer_col
                
                if best_kmer_col:
                    parts = best_kmer_col.split('_')
                    kmer_name = parts[0]
                    dataset_name = parts[1].split('-')[0].title()

                    kmer_label = f"{dataset_name} {kmer_name} (Best K-mer): {best_kmer_score:.3f}"
                    plt.axhline(y=best_kmer_score, color=line.get_color(), linestyle='--', label=kmer_label)


    plt.xlabel('Batch Number')
    plt.ylabel('Score')
    # Use the original pattern in the title for clarity
    title_name = name if name else column_name.replace('*', 'All').replace('_', ' ').title()
    plt.title(f'Quick Test Results: {title_name}')
    plt.grid(True, linestyle='--', alpha=0.7)

    # Adjust y-ticks if scores are reasonably expected to be between 0 and 1
    # Check if all plotted values are within [0, 1] range (ignoring NaNs)
    all_scores = pd.concat([df[col] for col, _ in matching_columns if col in df]).dropna()
    if all(0 <= score <= 1 for score in all_scores) and len(all_scores) > 0:
        plt.yticks(np.arange(0, 1.1, 0.1))
        plt.ylim(bottom=0) # Ensure y-axis starts at 0

    # For SSC only 
    if 'ssc' in column_name:
        # Ensure y-axis reaches 1.0 (max congruency)
        plt.ylim(top=1.0)

        # If at least 2 data points are > 0.4, set y-axis min to 0.0
        if (all_scores > 0.4).sum() >= 2:
            plt.ylim(bottom=0.0)

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    # Sanitize the column_name pattern for use in a filename
    safe_column_name = re.sub(r'[^a-zA-Z0-9_-]', '_', column_name) # Replace unsafe chars with _
    # Save the plot using the sanitized pattern name
    file_name = f"qt_{name.replace(' ', '_')}_{safe_column_name}.png".replace('*', 'all').replace('___', '_').replace('__', '_').lower()
    path = os.path.join(model_dir, file_name)
    plt.savefig(path, bbox_inches='tight', dpi=300)
    if show:
        plt.show()
    plt.close('all')


def load_kmer_qt_cache(dataset_split_dir):
    """
    Loads the k-mer quick test cache from a TSV file.

    Args:
        dataset_split_dir (str): The directory of the dataset split.

    Returns:
        dict: A dictionary containing the cached k-mer quick test results.
              Returns an empty dictionary if the cache file does not exist.
    """
    cache_path = os.path.join(dataset_split_dir, "kmer_qt_results.tsv")
    if not os.path.exists(cache_path):
        return {}
    
    try:
        df = pd.read_csv(cache_path, sep='\t')
        # Check if required columns exist
        if "K-mer Quick Test" not in df.columns or "Score" not in df.columns:
            print("Warning: kmer_qt_results.tsv is malformed. Missing required columns. Ignoring cache.")
            return {}
        cache_dict = pd.Series(df.Score.values, index=df['K-mer Quick Test']).to_dict()
        # The scores might be loaded as strings, so convert them to floats
        for key, value in cache_dict.items():
            try:
                cache_dict[key] = float(value)
            except (ValueError, TypeError):
                # Keep as is if conversion fails (e.g. "ditto" or other strings)
                pass
        return cache_dict
    except Exception as e:
        print(f"Warning: Could not load k-mer quick test cache. Error: {e}")
        return {}


def save_kmer_qt_cache(kmer_qt_to_be_cached, dataset_split_dir):
    """
    Saves new k-mer quick test results to the cache file.

    Args:
        kmer_qt_to_be_cached (dict): A dictionary of new results to cache.
        dataset_split_dir (str): The directory of the dataset split.
    """
    cache_path = os.path.join(dataset_split_dir, "kmer_qt_results.tsv")
    try:
        # Load existing cache if it exists
        if os.path.exists(cache_path):
            existing_df = pd.read_csv(cache_path, sep='\t')
        else:
            existing_df = pd.DataFrame(columns=["K-mer Quick Test", "Score"])

        # Create a DataFrame from the new results
        new_results_df = pd.DataFrame(list(kmer_qt_to_be_cached.items()), columns=["K-mer Quick Test", "Score"])
        
        # Combine and remove duplicates, keeping the new results
        combined_df = pd.concat([existing_df, new_results_df], ignore_index=True).drop_duplicates(subset=["K-mer Quick Test"], keep='last')
        
        # Save back to the TSV
        combined_df.to_csv(cache_path, sep='\t', index=False)
    except Exception as e:
        print(f"Warning: Could not save k-mer quick test cache. Error: {e}")



def get_missclassification_ranks_to_record(ranks):
    """
    Return the subset of ranks where missclassification sequence IDs should be recorded.
    """
    cfg = gb.QT_RECORD_MISS_CLASSIFICATION_SEQUENCES_AT_RANKS
    if cfg is None:
        return []
    ranks_to_record = []
    for rank in ranks:
        if 0 <= rank < len(cfg) and cfg[rank]:
            ranks_to_record.append(rank)
    return ranks_to_record


def save_missclassifications(missclass_rows, total_num_seqs, rank, set_name, strategy, rep_name, batch_num, model_dir):
    """
    Save missclassified sequence rows to CSV.

    CSV format:
      - first row: total_num_seqs=<N>,,
      - second row: column headers
      - remaining rows: missclassified sequences
    """
    rank_name = ALL_RANKS_PLURAL[rank]
    batch_num_str = "NA" if batch_num is None else str(batch_num)
    base_name = f"missclassifications_({set_name},{rank_name},{rep_name},{strategy})_{batch_num_str}.csv"
    out_dir = os.path.join(model_dir, "qt_results", "missclassifications")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, base_name)

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([f"total_num_seqs={int(total_num_seqs)}", "", ""])
        writer.writerow(["sequence_index", "true_taxon", "predicted_taxon"])
        if missclass_rows:
            writer.writerows(missclass_rows)


def save_classification_missclassifications(missclassifications_by_rank, set_name, strategy, rep_name, batch_num, model_dir):
    """
    Save per-rank missclassification CSVs for one classification run.
    """
    for rank, rank_data in missclassifications_by_rank.items():
        save_missclassifications(
            missclass_rows=rank_data["rows"],
            total_num_seqs=rank_data["total_num_seqs"],
            rank=rank,
            set_name=set_name,
            strategy=strategy,
            rep_name=rep_name,
            batch_num=batch_num,
            model_dir=model_dir,
        )
