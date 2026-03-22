"""
construct_dataset.py - A script to split a dataset and preprocess taxonomic labels.
 - Splits sequences from a dataset into a training set, a testing set, etc. 
 - Also preprocesses taxonomic labels for faster triplet and pair selection. 

 Splitting of datasets
  - Excluded taxa
    - Whole taxa are randomly selected as a testing set
    - At a specified taxonomic rank (e.g. family)
    - At a specified number of taxa (e.g. 12 families)
    - At a specified minimum size of taxa (e.g. minimum of 64 sequences)
    - At a specified maximum size of taxa (e.g. maximum of 128 sequences)
  - Test set
    - A random subset of remaining sequences
  - Training set
    - All remaining sequences

Input: Directory with a "FULL_seqs.fasta" file (resulting from the extract_regions.py script)
 - This file is assumed to include indices in the header (e.g. ">{1345}seq_1_1234567890")

Output: 
 - excluded_taxa_indices.txt: Indices of sequences to exclude for testing (one per line)
 - testing_set_indices.txt: Indices of sequences to include in the testing set (one per line)
 - training_set_indices.txt: Indices of sequences to include in the training set (one per line)
 - about_dataset_split.txt: Information about the dataset split
 - Data Structures for taxonomic information (for train, test, and excluded sets)
    - train_list_of_seq_indices_in_taxon_at_rank_dict.pkl
      - {rank_i: {taxon_label: [seq_indices]}}
      - useful for getting all sequences in a particular taxon
      - constructed by construct_seqs_in_taxon_dict(tax_labels, seq_indices)
    - train_list_of_taxon_labels_in_taxon_at_rank_dict.pkl
      - {rank_i: {taxon_label: [taxon_labels]}}
      - useful for getting all taxa in a particular taxon
      - constructed by construct_labels_in_taxon_dict(tax_labels, seq_indices)
    - train_list_of_taxon_labels_at_rank_dict.pkl
      - {rank_i: [taxon_labels]}
      - useful for getting all taxa at a particular rank (within any taxon)
      - constructed by construct_labels_at_rank_dict(train_list_of_taxon_labels_in_taxon_at_rank_dict)
    - train_nested_list_of_seq_indices.pkl
      - [[[seq_indices], [seq_indices]], [[seq_indices], [seq_indices]]]
      - useful for selecting sequences based on taxonomy
      - constructed by construct_nested_seq_list(train_list_of_seq_indices_in_taxon_at_rank_dict, train_list_of_taxon_labels_in_taxon_at_rank_dict)
    - train_full_tax_label_from_seq_id_dict.pkl
      - {seq_id: [full_taxon_labels]}
      - useful for quickly accessing taxonomies of particular sequences
      - constructed by construct_seq_id_to_full_label_dict(tax_labels, seq_indices)
    - train_nested_dicts_of_taxa.pkl
      - {"Prokaryotes": {"Bacteria": {...}, "Archaea": {...}}}
      - At the species-level, dictionaries point to lists of sequence indices of that species
      - a well rounded datastructure useful for many things
      - constructed by construct_nested_dicts_of_taxa(train_list_of_seq_indices_in_taxon_at_rank_dict, train_list_of_taxon_labels_in_taxon_at_rank_dict)
    - train_taxon_label_to_taxon_id.pkl
      - {taxon_label: taxon_id}
      - Maps rank-qualified taxon labels (str) to integer IDs (starting from 0)
      - Example keys: d__Bacteria, p__Bacillota, c__Bacilli
      - Rank qualification prevents collisions when the same plain label appears at multiple ranks
      - Useful for creating integer-indexed arrays of taxonomic information
      - constructed by construct_taxon_id_mappings(train_list_of_taxon_labels_at_rank_dict)
    - train_taxon_id_to_taxon_label.pkl
      - {taxon_id: taxon_label}
      - Maps integer IDs back to taxon labels (str)
      - Useful for decoding integer-indexed arrays back to taxon names
      - constructed by construct_taxon_id_mappings(train_list_of_taxon_labels_at_rank_dict)
 - Label Arrays (for train, test, and excluded sets)
    - seq_taxon_ids.npy
      - shape: (n_seqs, 7), dtype: int32
      - The taxon ID of every sequence at every rank
      - For each sequence i and rank j, stores the integer taxon ID
      - These IDs map to taxon labels via taxon_id_to_taxon_label.pkl
      - constructed by construct_seq_taxon_ids(full_tax_labels, seq_indices, taxon_label_to_taxon_id)
    - pairwise_ranks.npy
      - shape: (n_seqs, n_seqs), dtype: int8
      - The taxonomic rank at which each pair of sequences shares the same classification
      - For each pair (i, j), stores the deepest rank where they share the same taxon
      - Range: -1 (different domains) to 6 (same species)
      - constructed by construct_pairwise_ranks(full_tax_labels, seq_indices)
    - pairwise_pos_masks.npy
      - shape: (7, n_seqs, n_seqs), dtype: bool
      - For each rank, indicates which pairs of sequences share the same taxon at that rank
      - pairwise_pos_masks[rank, i, j] = True if sequences i and j share the same taxon at rank
      - Diagonal is False (i == j) since sequences should not be compared with themselves
      - constructed by construct_pairwise_masks(seq_taxon_ids)
    - pairwise_neg_masks.npy
      - shape: (7, n_seqs, n_seqs), dtype: bool
      - For each rank, indicates which pairs of sequences have different taxa at that rank
      - pairwise_neg_masks[rank, i, j] = True if sequences i and j have different taxa at rank
      - Diagonal is False (i == j) since sequences should not be compared with themselves
      - constructed by construct_pairwise_masks(seq_taxon_ids)
    - pairwise_mrca_taxon_ids.npy
      - shape: (n_seqs, n_seqs), dtype: int32
      - The taxon ID of the Most Recent Common Ancestor (MRCA) for each pair of sequences
      - For each pair (i, j), stores the taxon ID at the highest rank they share in common
      - Value is -1 if sequences differ at domain level (no common ancestor in taxonomy)
      - constructed by construct_pairwise_mrca_taxon_ids(seq_taxon_ids, pairwise_ranks)
    - pairwise_distances.npy
      - shape: (n_seqs, n_seqs), dtype: float32
      - The phylogenetic distance between each pair of sequences
      - For each pair (i, j), stores the RED distance based on their MRCA
      - Calculated exactly from leaf-to-leaf MRCAs in the decorated GTDB trees
      - Equivalent to direct RedTree.dist_between_nodes(leaf_i, leaf_j) for all pairs
      - constructed by construct_pairwise_distances(gtdb_ids_for_split, redvals_id_from_gtdb_id, red_trees, distance_between_domains)
    - distances_lookup_array.npy
      - shape: (max_taxon_id + 1,), dtype: float32
      - Lookup array mapping taxon IDs to phylogenetic distances
      - distances_lookup_array[taxon_id] = RED distance within that taxon
      - Used as an auxiliary lookup based on taxonomic labels (legacy/debug compatibility)
      - constructed by construct_distances_from_mrca_taxon_id_dict(taxon_id_to_taxon_label, taxon_id_to_rank, red_trees)
    - distance_between_domains.npy
      - shape: (1,), dtype: float32
      - Single value: RED distance between Bacteria and Archaea domains
      - Used for pairs with taxon_id = -1 (different domains)
      - Set by RED_DISTANCE_BETWEEN_DOMAINS
      - constructed by construct_distances_from_mrca_taxon_id_dict(taxon_id_to_taxon_label, taxon_id_to_rank, red_trees)

Output Directory Structure:
/mnt/secondary/micro16s_dbs/16s_databases/m16s_db_004/seqs/split_002/
├── about_dataset_split.txt
├── excluded_taxa_indices.txt
├── testing_indices.txt
├── training_indices.txt
├── tax_objs/
│   ├── train/
│   │   ├── list_of_seq_indices_in_taxon_at_rank_dict.pkl
│   │   ├── list_of_taxon_labels_in_taxon_at_rank_dict.pkl
│   │   ├── list_of_taxon_labels_at_rank_dict.pkl
│   │   ├── nested_list_of_seq_indices.pkl
│   │   ├── full_tax_label_from_seq_id_dict.pkl
│   │   ├── nested_dicts_of_taxa.pkl
│   │   ├── taxon_label_to_taxon_id.pkl
│   │   └── taxon_id_to_taxon_label.pkl
│   ├── test/
│   │   ├── list_of_seq_indices_in_taxon_at_rank_dict.pkl
│   │   ├── list_of_taxon_labels_in_taxon_at_rank_dict.pkl
│   │   ├── list_of_taxon_labels_at_rank_dict.pkl
│   │   ├── nested_list_of_seq_indices.pkl
│   │   ├── full_tax_label_from_seq_id_dict.pkl
│   │   ├── nested_dicts_of_taxa.pkl
│   │   ├── taxon_label_to_taxon_id.pkl
│   │   └── taxon_id_to_taxon_label.pkl
│   └── excluded/
│       ├── list_of_seq_indices_in_taxon_at_rank_dict.pkl
│       ├── list_of_taxon_labels_in_taxon_at_rank_dict.pkl
│       ├── list_of_taxon_labels_at_rank_dict.pkl
│       ├── nested_list_of_seq_indices.pkl
│       ├── full_tax_label_from_seq_id_dict.pkl
│       ├── nested_dicts_of_taxa.pkl
│       ├── taxon_label_to_taxon_id.pkl
│       └── taxon_id_to_taxon_label.pkl
└── labels/
    ├── train/
    │   ├── seq_taxon_ids.npy
    │   ├── pairwise_ranks.npy
    │   ├── pairwise_pos_masks.npy
    │   ├── pairwise_neg_masks.npy
    │   ├── pairwise_mrca_taxon_ids.npy
    │   ├── pairwise_distances.npy
    │   ├── distances_lookup_array.npy
    │   └── distance_between_domains.npy
    ├── test/
    │   ├── seq_taxon_ids.npy
    │   ├── pairwise_ranks.npy
    │   ├── pairwise_pos_masks.npy
    │   ├── pairwise_neg_masks.npy
    │   ├── pairwise_mrca_taxon_ids.npy
    │   ├── pairwise_distances.npy
    │   ├── distances_lookup_array.npy
    │   └── distance_between_domains.npy
    └── excluded/
        ├── seq_taxon_ids.npy
        ├── pairwise_ranks.npy
        ├── pairwise_pos_masks.npy
        ├── pairwise_neg_masks.npy
        ├── pairwise_mrca_taxon_ids.npy
        ├── pairwise_distances.npy
        ├── distances_lookup_array.npy
        └── distance_between_domains.npy

This script can take 10-30 minutes to run. Most of the compute is for exact pairwise RED distance construction for the training set.
"""

import os
import random
import datetime
import pickle
import shutil
import numpy as np
import time

# Local Imports
from redvals.redvals import RedTree

# Input directory containing the "FULL_seqs.fasta" file 
INPUT_DIR = "/mnt/secondary/micro16s_dbs/16s_databases/m16s_db_004/seqs/"
INPUT_FILE = INPUT_DIR + "FULL_seqs.fasta"

# Output files
OUTPUT_DATASET_SPLIT_DIR = INPUT_DIR + "split_001/"
EXCLUDED_TAXA_FILE = OUTPUT_DATASET_SPLIT_DIR + "excluded_taxa_indices.txt"
TESTING_FILE = OUTPUT_DATASET_SPLIT_DIR + "testing_indices.txt"
TRAINING_FILE = OUTPUT_DATASET_SPLIT_DIR + "training_indices.txt"

# Parameters for excluded taxa
EXCLUDED_TAXA_RANK = 4  # 0=domain, 1=phylum, 2=class, 3=order, 4=family, 5=genus, 6=species
N_EXCLUDED_TAXA = 12    # Number of taxa to exclude for testing
MIN_TAXA_SIZE = 64
MAX_TAXA_SIZE = 128


# Test/train ratio (ratio to divide the data after excluding taxa)
TEST_TRAIN_RATIO = 0.2

RANKS = {0: "domain", 1: "phylum", 2: "class", 3: "order", 4: "family", 5: "genus", 6: "species"}
RANKS_PLURAL = {0: "domains", 1: "phyla", 2: "classes", 3: "orders", 4: "families", 5: "genera", 6: "species"}

# Dictionary mapping rank numbers to their prefixes
RANK_PREFIXES = {
    0: "d__",  # domain
    1: "p__",  # phylum
    2: "c__",  # class
    3: "o__",  # order
    4: "f__",  # family
    5: "g__",  # genus
    6: "s__"   # species
}

# Constants for distance calculations
RED_DISTANCE_BETWEEN_DOMAINS = 1.65
# Print ETA updates this often during exact pairwise distance construction
ETA_REPORT_INTERVAL_SECONDS = 20


def make_ranked_taxon_key(rank, taxon_label):
    """
    Build a rank-qualified taxon key (e.g., rank=2, label='Bacilli' -> 'c__Bacilli').
    """
    return f"{RANK_PREFIXES[rank]}{taxon_label}"


def read_input_file():
    """
    Reads the input file:
     - Checking indices are in order
     - Collecting taxonomic labels
     - Collecting GTDB IDs in sequence order
    """
    tax_labels = []
    gtdb_ids = []
    i = 1
    with open(INPUT_FILE, "r") as f:
        for line in f:
            if line.startswith(">"):
                # If the header line does not contain curly braces, raise an error
                if "{" not in line and "}" not in line:
                    raise Exception(f"Error: Sequence header line does not contain curly braces: \n{line}\n...\n")
                # Try to get the index of the sequence
                try:
                    index = int(line.split("{")[1].split("}")[0])
                    # Assert that the index is a positive integer
                    assert index >= 0, f"Error: Sequence index is not a positive integer: \n{line}\n...\n"
                except Exception as e:
                    print(f"Error: Could not get index for sequence \n{line}\n...\n")
                    raise e

                # Ensure it is in order starting from 0
                if index != i:
                    print(f"Error: Sequence index is not in order. Expected {i}, got {index} for sequence \n{line}\n...\n")
                    raise Exception("Sequence index is not in order")
                else:
                    # Get taxonomic label
                    tax_label = parse_taxonomy(line)
                    tax_labels.append(tax_label)
                    # Get GTDB ID
                    gtdb_id = parse_gtdb_id(line)
                    gtdb_ids.append(gtdb_id)
                    # Increment index
                    i += 1

    return tax_labels, gtdb_ids

def parse_taxonomy(description):
    """
    Parses the taxonomy from the fasta header.

    Args:
        description (str): The description line from the fasta file.
            Example: ">{1235}GB_GCA_000008085.1~AE017199.1 d__Archaea;p__Nanoarchaeota;c__Nanoarchaeia;o__Nanoarchaeales;f__Nanoarchaeaceae;g__Nanoarchaeum;s__Nanoarchaeum equitans [location=432327..433825] [ssu_len=1499] [contig_len=490885]"
        
    Returns:
        list: A list of taxonomic classifications from domain to species.
        EXAMPLES: 
            ['Bacteria', 'Pseudomonadota', 'Gammaproteobacteria', 'Enterobacterales', 'Enterobacteriaceae', 'Escherichia', 'Escherichia coli']
            ['Bacteria', 'Bacillota_A', 'Clostridia', 'Clostridiales', 'Clostridiaceae', 'Clostridium', 'Clostridium beijerinckii']
            ['Archaea', 'Thermoproteota', 'Thermoprotei_A', 'Sulfolobales', 'Ignisphaeraceae', 'Ignisphaera', 'Ignisphaera sp023269755']
    """
    # The taxonomy string is between the first space and the first '['
    tax_string = description.split(' ', 1)[1].split('[', 1)[0].strip()

    # Split ranks by semicolons
    tax_levels = tax_string.strip().split(';')

    # Remove rank prefixes (e.g., 'd__', 'p__', etc.)
    tax_levels = [level.split('__')[1] if '__' in level else level for level in tax_levels]

    # Ensure there are 7 ranks
    assert len(tax_levels) == 7, f"Expected 7 taxonomic ranks, got {len(tax_levels)} for sequence: \n{description}"
        
    return tax_levels

def parse_gtdb_id(description):
    """
    Parses the GTDB genome ID from the fasta header.

    Args:
        description (str): The description line from the fasta file.
            Example: ">{1235}GB_GCA_000008085.1~AE017199.1 d__Archaea;..."

    Returns:
        str: The GTDB genome ID.
            EXAMPLES:
                "GB_GCA_000008085.1"
                "RS_GCF_018344175.1"
    """
    # The token immediately after the closing "}" is expected to be:
    #   "{GTDB_ID}~{contig_id}"
    # Example:
    #   "GB_GCA_000008085.1~AE017199.1"
    try:
        token_after_index = description.split("}", 1)[1].split(" ", 1)[0]
    except Exception as e:
        raise ValueError(f"Could not parse header token from sequence:\n{description}") from e

    if "~" not in token_after_index:
        raise ValueError(f"Expected '~' separator in header token, got:\n{description}")

    gtdb_id = token_after_index.split("~", 1)[0].strip()
    if not gtdb_id:
        raise ValueError(f"Parsed empty GTDB ID from sequence:\n{description}")

    return gtdb_id

def construct_seqs_in_taxon_dict(full_tax_labels, seq_indices):
    """
    Collects pointers to all sequences in all taxa at all ranks.

    Example output:
    list_of_seq_indices_in_taxon_at_rank_dict = {
        0: {  # rank 0 (domain)
            'Bacteria': [0, 1, 2],  # sequence indices with Bacteria at domain level
            'Archaea': [3, 4]       # sequence indices with Archaea at domain level
        },
        1: {  # rank 1 (phylum) 
            'Proteobacteria': [0, 1],     # sequence indices with Proteobacteria at phylum level
            'Firmicutes': [2],            # sequence indices with Firmicutes at phylum level
            'Euryarchaeota': [3, 4]       # sequence indices with Euryarchaeota at phylum level
        }
    }
    """

    list_of_seq_indices_in_taxon_at_rank_dict = {}
    for sequence_index, tax_label in zip(seq_indices, full_tax_labels):
        for rank_i, rank in RANKS.items():

            # Extract the taxonomic label at this rank
            tax_label_at_rank = tax_label[rank_i]
            # Add an empty dictionary for this rank if it didn't exist yet
            if rank_i not in list_of_seq_indices_in_taxon_at_rank_dict:
                list_of_seq_indices_in_taxon_at_rank_dict[rank_i] = {}
            # Add an empty list for this taxonomic label if it didn't exist yet
            if tax_label_at_rank not in list_of_seq_indices_in_taxon_at_rank_dict[rank_i]:
                list_of_seq_indices_in_taxon_at_rank_dict[rank_i][tax_label_at_rank] = []
            # Add the sequence index to the list
            list_of_seq_indices_in_taxon_at_rank_dict[rank_i][tax_label_at_rank].append(sequence_index)
    return list_of_seq_indices_in_taxon_at_rank_dict

def construct_labels_in_taxon_dict(full_tax_labels):
    """
    Collects all taxonomic labels within each taxon at each rank.

    Example output:
    list_of_taxon_labels_in_taxon_at_rank_dict = {
        0: {  # rank 0 (domain)
            'Bacteria': ['Proteobacteria', 'Firmicutes'],  # phyla within Bacteria
            'Archaea': ['Euryarchaeota']                   # phyla within Archaea
        },
        1: {  # rank 1 (phylum)
            'Proteobacteria': ['Alphaproteobacteria', 'Gammaproteobacteria'],
            'Firmicutes': ['Bacilli', 'Clostridia'],
            'Euryarchaeota': ['Methanobacteria']
        }
    }

    Args:
        full_tax_labels (list): List of full taxonomic classifications of sequences from domain to species

    Returns:
        dict: Dictionary of taxonomic labels within each taxon at each rank
    """
    list_of_taxon_labels_in_taxon_at_rank_dict = {}
    
    # for each taxonomy path (e.g. ['Bacteria', 'Proteobacteria', 'Alphaproteobacteria', ...])
    for tax in full_tax_labels:
        # note: we only go to the penultimate rank because the last level (species) has no child
        for i in range(len(tax) - 1):
            parent = tax[i]
            child = tax[i + 1]
            if i not in list_of_taxon_labels_in_taxon_at_rank_dict:
                list_of_taxon_labels_in_taxon_at_rank_dict[i] = {}
            if parent not in list_of_taxon_labels_in_taxon_at_rank_dict[i]:
                list_of_taxon_labels_in_taxon_at_rank_dict[i][parent] = []
            # append the child if it is not already in the list (to avoid duplicates)
            if child not in list_of_taxon_labels_in_taxon_at_rank_dict[i][parent]:
                list_of_taxon_labels_in_taxon_at_rank_dict[i][parent].append(child)
    
    return list_of_taxon_labels_in_taxon_at_rank_dict

def construct_labels_at_rank_dict(list_of_taxon_labels_in_taxon_at_rank_dict):
    """
    Creates a dictionary of all unique taxonomic labels at each rank.

    Example output:
    list_of_taxon_labels_at_rank_dict = {
        0: ['Bacteria', 'Archaea'],  # all domains
        1: ['Proteobacteria', 'Firmicutes', 'Euryarchaeota'],  # all phyla
        2: ['Alphaproteobacteria', 'Gammaproteobacteria', 'Bacilli', 'Clostridia', 'Methanobacteria']  # all classes
    }

    Args:
        list_of_taxon_labels_in_taxon_at_rank_dict (dict): Dictionary of taxonomic labels within each taxon

    Returns:
        dict: Dictionary of all unique taxonomic labels at each rank
    """
    list_of_taxon_labels_at_rank_dict = {}
    
    # For ranks 0-5, collect all parent taxa at each rank
    for rank in list_of_taxon_labels_in_taxon_at_rank_dict:
        # The keys at this rank are the taxa that exist at this rank
        unique_labels = set(list_of_taxon_labels_in_taxon_at_rank_dict[rank].keys())
        list_of_taxon_labels_at_rank_dict[rank] = sorted(list(unique_labels))
    
    # For rank 6 (species), collect all children from rank 5 (genus)
    # Species don't appear as keys in list_of_taxon_labels_in_taxon_at_rank_dict
    # because they have no children
    if 5 in list_of_taxon_labels_in_taxon_at_rank_dict:
        species_labels = set()
        for children_list in list_of_taxon_labels_in_taxon_at_rank_dict[5].values():
            species_labels.update(children_list)
        list_of_taxon_labels_at_rank_dict[6] = sorted(list(species_labels))
    
    return list_of_taxon_labels_at_rank_dict

def construct_nested_seq_list(list_of_seq_indices_in_taxon_at_rank_dict, list_of_taxon_labels_in_taxon_at_rank_dict, list_of_taxon_labels_at_rank_dict):
    """
    Creates a nested list of sequence indices organized by taxonomy.

    Example output:
    nested_list_of_seq_indices = [
        # Bacteria
        [
            # Proteobacteria
            [
                [0, 1],  # Alphaproteobacteria sequences
                [2, 3]   # Gammaproteobacteria sequences
            ],
            # Firmicutes
            [
                [4, 5],  # Bacilli sequences
                [6, 7]   # Clostridia sequences
            ]
        ],
        # Archaea
        [
            # Euryarchaeota
            [
                [8, 9]   # Methanobacteria sequences
            ]
        ]
    ]
    This example is flawed in that it should ALWAYS go to the species level:
      - So for example a family with only 1 sequence will look like: [[[245]]]
      - Or for example a species with 2 sequences will look like: [246, 247]

    Args:
        list_of_seq_indices_in_taxon_at_rank_dict (dict): Dictionary of sequence indices in each taxon
        list_of_taxon_labels_in_taxon_at_rank_dict (dict): Dictionary of taxonomic labels within each taxon

    Returns:
        list: Nested list of sequence indices organized by taxonomy
    """
    def build_nested_list(taxon, current_rank):
        # Base case: if we're at the species level (rank 6), return sequence indices
        if current_rank == 6:
            return list_of_seq_indices_in_taxon_at_rank_dict[current_rank].get(taxon, [])
        
        # Get child taxa for the current taxon
        child_taxa = list_of_taxon_labels_in_taxon_at_rank_dict[current_rank].get(taxon, [])

        # Recursively build nested lists for each child taxon
        return [build_nested_list(child, current_rank + 1) for child in child_taxa]
    
    # Start with domains (rank 0)
    domains = list_of_taxon_labels_at_rank_dict.get(0, [])
    nested_list_of_seq_indices = [build_nested_list(domain, 0) for domain in domains]
    return nested_list_of_seq_indices

def construct_seq_id_to_full_label_dict(full_tax_labels, seq_indices):
    """
    Creates a dictionary mapping sequence IDs to their full taxonomic classification.

    Example output:
    full_tax_label_from_seq_id_dict = {
        0: ['Bacteria', 'Proteobacteria', 'Alphaproteobacteria', ...],
        1: ['Bacteria', 'Proteobacteria', 'Alphaproteobacteria', ...],
        2: ['Archaea', 'Euryarchaeota', 'Methanobacteria', ...]
    }

    Args:
        full_tax_labels (list): List of taxonomic classifications for every sequence from domain to species
        seq_indices (list): List of sequence indices to include

    Returns:
        dict: Dictionary mapping sequence IDs to full taxonomic classifications
    """
    full_tax_label_from_seq_id_dict = {}
    for sequence_index, full_tax_label in zip(seq_indices, full_tax_labels):
        full_tax_label_from_seq_id_dict[sequence_index] = full_tax_label
    return full_tax_label_from_seq_id_dict

def construct_taxon_id_mappings(list_of_taxon_labels_at_rank_dict):
    """
    Creates bidirectional mappings between taxon labels and integer IDs.
    
    Taxon IDs are assigned sequentially starting from 0, iterating through all ranks
    in order (domain, phylum, class, order, family, genus, species). Within each rank,
    taxa are processed in sorted order to ensure consistency.
    
    IMPORTANT:
    Taxon labels are rank-qualified in taxon_label_to_taxon_id (e.g., "c__Bacilli")
    to prevent collisions when the same plain label appears at multiple ranks.
    This avoids ID overwrites that can corrupt MRCA distance lookup.
    
    Args:
        list_of_taxon_labels_at_rank_dict (dict): Dictionary mapping rank (int) to list of taxon labels at that rank
        
    Returns:
        tuple: (taxon_label_to_taxon_id, taxon_id_to_taxon_label, taxon_id_to_rank)
            - taxon_label_to_taxon_id (dict): Maps rank-qualified taxon label (str) -> taxon ID (int)
            - taxon_id_to_taxon_label (dict): Maps taxon ID (int) -> taxon label (str)
            - taxon_id_to_rank (dict): Maps taxon ID (int) -> the taxonomic rank (int)
    
    Example:
        Input: {0: ['Archaea', 'Bacteria'], 1: ['Firmicutes', 'Proteobacteria']}
        Output: 
            taxon_label_to_taxon_id = {'d__Archaea': 0, 'd__Bacteria': 1, 'p__Firmicutes': 2, 'p__Proteobacteria': 3}
            taxon_id_to_taxon_label = {0: 'Archaea', 1: 'Bacteria', 2: 'Firmicutes', 3: 'Proteobacteria'}
    """
    taxon_label_to_taxon_id = {}
    taxon_id_to_taxon_label = {}
    taxon_id_to_rank = {}
    current_id = 0
    
    # Iterate through all ranks in order (0 to 6)
    for rank in sorted(list_of_taxon_labels_at_rank_dict.keys()):
        # Get all taxa at this rank (already sorted in list_of_taxon_labels_at_rank_dict)
        taxa_at_rank = list_of_taxon_labels_at_rank_dict[rank]
        
        # Assign IDs to each taxon at this rank
        for taxon_label in taxa_at_rank:
            ranked_taxon_key = make_ranked_taxon_key(rank, taxon_label)
            taxon_label_to_taxon_id[ranked_taxon_key] = current_id
            taxon_id_to_taxon_label[current_id] = taxon_label
            taxon_id_to_rank[current_id] = rank
            current_id += 1

    # Backward-compatible aliases for domain labels only (unambiguous and used in runtime code).
    for domain_label in list_of_taxon_labels_at_rank_dict.get(0, []):
        ranked_domain_key = make_ranked_taxon_key(0, domain_label)
        if ranked_domain_key in taxon_label_to_taxon_id and domain_label not in taxon_label_to_taxon_id:
            taxon_label_to_taxon_id[domain_label] = taxon_label_to_taxon_id[ranked_domain_key]
    
    return taxon_label_to_taxon_id, taxon_id_to_taxon_label, taxon_id_to_rank

def construct_nested_dicts_of_taxa(list_of_seq_indices_in_taxon_at_rank_dict, list_of_taxon_labels_in_taxon_at_rank_dict):
    """
    Creates a nested dictionary structure of taxonomic labels.
    
    For taxonomic ranks 0 through 5 (domain to genus), each node is represented
    as a dictionary mapping a taxon label (in lowercase) to its child taxa.
    At the species level (rank 6), the node is a dictionary mapping a species label 
    (in lowercase) to a list of sequence indices belonging to that species.
    
    The entire structure is wrapped under the key "Prokaryotes".
    
    Args:
        list_of_seq_indices_in_taxon_at_rank_dict (dict): Dictionary mapping each taxonomic rank (0-6) to another dictionary
            that maps a taxon label to a list of sequence indices that have that label at that rank.
        list_of_taxon_labels_in_taxon_at_rank_dict (dict): Dictionary mapping each taxonomic rank (0-5) to another dictionary that maps
            a parent taxon label to a list of its child taxon labels.
    
    Returns:
        dict: A nested dictionary structure of taxonomic labels.
              Example output:
              {
                  "Prokaryotes": {
                      "Bacteria": {
                          "Proteobacteria": {
                              ...,
                              "Escherichia": [0, 1, 2]
                          },
                          "Firmicutes": {
                              ...
                          }
                      },
                      "Archaea": {
                          "Euryarchaeota": {
                              ...
                          }
                      }
                  }
              }
    """
    # Recursive helper function to build the nested dictionary for a given taxon at a particular rank.
    def build_node(rank, taxon):
        # Base case: if at species level (rank 6), return the species and its sequence indices.
        if rank == 6:
            # Retrieve the list of sequence indices for this species; default to empty if not found.
            seq_indices = list_of_seq_indices_in_taxon_at_rank_dict.get(6, {}).get(taxon, [])
            return {taxon: seq_indices}
        
        # For intermediate ranks, obtain the child taxa of the current taxon.
        child_taxa = list_of_taxon_labels_in_taxon_at_rank_dict.get(rank, {}).get(taxon, [])
        children_dict = {}
        # Recursively build the nested dictionary for each child taxon at the next rank.
        for child in child_taxa:
            # The recursive call returns a dictionary that we merge into the current children dictionary.
            children_dict.update(build_node(rank + 1, child))
        return {taxon: children_dict}
    
    # Start building the structure from the top-level taxa (domains).
    # The top-level taxa are obtained from the keys in list_of_taxon_labels_in_taxon_at_rank_dict at rank 0.
    top_level_taxa = list_of_taxon_labels_in_taxon_at_rank_dict.get(0, {}).keys()
    nested_structure = {}
    # Build the nested branch for each top-level taxon, sorted for consistent ordering.
    for taxon in sorted(top_level_taxa, key=lambda x: x):
        nested_structure.update(build_node(0, taxon))
    
    # Wrap the resulting structure under the "Prokaryotes" key and return it.
    train_nested_dicts_of_taxa = {"Prokaryotes": nested_structure}
    return train_nested_dicts_of_taxa


def select_excluded_taxa(list_of_seq_indices_in_taxon_at_rank_dict):
    """
    Selects taxa to exclude for testing based on the parameters:
    - EXCLUDED_TAXA_RANK: taxonomic rank to select from
    - N_EXCLUDED_TAXA: number of taxa to select
    - MIN_TAXA_SIZE: minimum number of sequences in a taxon
    - MAX_TAXA_SIZE: maximum number of sequences in a taxon

    Args:
        list_of_seq_indices_in_taxon_at_rank_dict (dict): Dictionary of all taxa and their sequence indices at each rank

    Returns:
        set: Set of sequence indices to exclude for testing
    """

    # Get the dict of all taxa for the specified rank
    taxa_at_rank_dict = list_of_seq_indices_in_taxon_at_rank_dict[EXCLUDED_TAXA_RANK]
    
    # Filter taxa dict by size criteria
    eligible_taxa_dict = {
        taxon: indices 
        for taxon, indices in taxa_at_rank_dict.items()
        if MIN_TAXA_SIZE <= len(indices) <= MAX_TAXA_SIZE
    }

    # Check if we have enough eligible taxa
    n_eligible_taxa = len(eligible_taxa_dict)
    if n_eligible_taxa < N_EXCLUDED_TAXA:
        raise ValueError(
            f"Not enough taxa of size {MIN_TAXA_SIZE}-{MAX_TAXA_SIZE} sequences "
            f"at rank {RANKS[EXCLUDED_TAXA_RANK]} ({n_eligible_taxa} found, "
            f"need {N_EXCLUDED_TAXA})"
        )

    # Randomly select N_EXCLUDED_TAXA taxa
    selected_taxa_labels = random.sample(list(eligible_taxa_dict.keys()), N_EXCLUDED_TAXA)

    # Collect all sequence indices from selected taxa
    excluded_indices = set()
    for taxon in selected_taxa_labels:
        excluded_indices.update(eligible_taxa_dict[taxon])

    return selected_taxa_labels, excluded_indices, n_eligible_taxa

def write_seq_indices(seq_indices, file_path):
    """
    Writes sequence seq_indices to a file, one per line.
    
    Args:
        seq_indices (list): List of sequence seq_indices to write
        file_path (str): Path to the output file
    """
    # Check that the file doesn't exist
    if os.path.exists(file_path):
        raise FileExistsError(f"File {file_path} already exists")

    # Write the seq_indices to the file
    with open(file_path, "w") as f:
        for index in seq_indices:
            f.write(f"{index}\n")

def write_about_file(n_seqs, selected_taxa_labels, list_of_seq_indices_in_taxon_at_rank_dict, 
                    excluded_taxa_seq_indices, testing_seq_indices, training_seq_indices,
                    distances_between_taxa_results):
    """
    Writes information about the dataset split to a file.
    """
    about_file = OUTPUT_DATASET_SPLIT_DIR + "about_dataset_split.txt"
    
    # Check that the file doesn't exist
    if os.path.exists(about_file):
        raise FileExistsError(f"File {about_file} already exists")
    
    with open(about_file, "w") as f:
        # Write time and input file
        f.write(f"Created: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Input file: {INPUT_FILE}\n\n")
        
        # Write parameters
        f.write("Dataset Split Parameters:\n")
        f.write(f"Excluded taxa rank: {RANKS[EXCLUDED_TAXA_RANK]}\n")
        f.write(f"Number of excluded taxa: {N_EXCLUDED_TAXA}\n")
        f.write(f"Minimum taxa size: {MIN_TAXA_SIZE}\n")
        f.write(f"Maximum taxa size: {MAX_TAXA_SIZE}\n")
        f.write(f"Test/train ratio: {TEST_TRAIN_RATIO}\n\n")
        
        # Write results
        f.write(f"Number of sequences: {n_seqs}\n\n")
        
        f.write(f"There were {len(list_of_seq_indices_in_taxon_at_rank_dict[EXCLUDED_TAXA_RANK])} total {RANKS_PLURAL[EXCLUDED_TAXA_RANK]}\n")
        f.write(f"Selected {N_EXCLUDED_TAXA} {RANKS_PLURAL[EXCLUDED_TAXA_RANK]} for exclusion:\n")
        for taxon_label in selected_taxa_labels:
            n_seqs_in_taxon = len(list_of_seq_indices_in_taxon_at_rank_dict[EXCLUDED_TAXA_RANK][taxon_label])
            f.write(f"  - {taxon_label}: {n_seqs_in_taxon} sequences\n")
        f.write("\n")
        
        f.write(f"Excluded taxa: {len(excluded_taxa_seq_indices)} sequences ({len(excluded_taxa_seq_indices)/n_seqs*100:.1f}%)\n")
        f.write(f"Testing set: {len(testing_seq_indices)} sequences ({len(testing_seq_indices)/n_seqs*100:.1f}%)\n")
        f.write(f"Training set: {len(training_seq_indices)} sequences ({len(training_seq_indices)/n_seqs*100:.1f}%)\n")

        # Calculate species statistics
        species_dict = list_of_seq_indices_in_taxon_at_rank_dict.get(6, {}) # Rank 6 is species
        total_species = len(species_dict)
        species_gt1_seq = sum(1 for indices in species_dict.values() if len(indices) > 1)
        percentage_gt1_seq = (species_gt1_seq / total_species * 100) if total_species > 0 else 0

        # Write species statistics
        f.write("\nOther Taxonomic Statistics:\n")
        f.write(f"Number of species with > 1 sequence: {species_gt1_seq} ({percentage_gt1_seq:.4f}%)\n")


def print_unique_taxa_counts(list_of_taxon_labels_at_rank_dict, prefix=""):
    """Prints the number of unique taxa at each rank."""
    print(f"{prefix}Number of Unique Taxa:")
    for rank in sorted(list_of_taxon_labels_at_rank_dict.keys()):
        count = len(list_of_taxon_labels_at_rank_dict[rank])
        print(f"  - Rank {rank} ({RANKS[rank].capitalize()}): {count}")

def print_label_array_sanity_checks(prefix, seq_taxon_ids, pairwise_ranks, pairwise_pos_masks, pairwise_neg_masks, pairwise_mrca_taxon_ids, pairwise_distances):
    """Prints simple sanity checks for the label arrays."""
    n_seqs = seq_taxon_ids.shape[0]
    print(f"{prefix}Label Array Sanity Checks:")
    print(f"  - Number of sequences: {n_seqs}")
    print(f"  - seq_taxon_ids shape: {seq_taxon_ids.shape} (expected: ({n_seqs}, 7))")
    print(f"  - pairwise_ranks shape: {pairwise_ranks.shape} (expected: ({n_seqs}, {n_seqs}))")
    
    if pairwise_pos_masks is not None:
        print(f"  - pairwise_pos_masks shape: {pairwise_pos_masks.shape} (expected: (7, {n_seqs}, {n_seqs}))")
    else:
        print(f"  - pairwise_pos_masks: None (Skipped to save memory)")
        
    if pairwise_neg_masks is not None:
        print(f"  - pairwise_neg_masks shape: {pairwise_neg_masks.shape} (expected: (7, {n_seqs}, {n_seqs}))")
    else:
        print(f"  - pairwise_neg_masks: None (Skipped to save memory)")

    print(f"  - pairwise_mrca_taxon_ids shape: {pairwise_mrca_taxon_ids.shape} (expected: ({n_seqs}, {n_seqs}))")
    print(f"  - pairwise_distances shape: {pairwise_distances.shape} (expected: ({n_seqs}, {n_seqs}))")
    
    # Quick distribution check of pairwise ranks
    unique_ranks, counts = np.unique(pairwise_ranks, return_counts=True)
    if len(unique_ranks) > 0:
        print(f"  - Pairwise ranks range: [{unique_ranks.min()}, {unique_ranks.max()}] (expected: [-1, 6])")
    else:
        print(f"  - Pairwise ranks range: [N/A, N/A] (expected: [-1, 6])")
    
    # Quick distance stats (excluding diagonal)
    distances_off_diag = pairwise_distances[~np.eye(n_seqs, dtype=bool)]
    if len(distances_off_diag) > 0:
        print(f"  - Distance range: [{distances_off_diag.min():.4f}, {distances_off_diag.max():.4f}]")

def save_pickles(input_dir, list_of_seq_indices_in_taxon_at_rank_dict, list_of_taxon_labels_in_taxon_at_rank_dict, 
                 list_of_taxon_labels_at_rank_dict, nested_list_of_seq_indices, 
                 full_tax_label_from_seq_id_dict, nested_dicts_of_taxa, 
                 taxon_label_to_taxon_id, taxon_id_to_taxon_label):
    """
    Saves the taxonomic data structures as pickle files.
    
    Args:
        input_dir (str): Directory to save the pickle files
        list_of_seq_indices_in_taxon_at_rank_dict (dict): Sequence indices in each taxon at each rank
        list_of_taxon_labels_in_taxon_at_rank_dict (dict): Taxon labels within each taxon at each rank
        list_of_taxon_labels_at_rank_dict (dict): All taxon labels at each rank
        nested_list_of_seq_indices (list): Nested list of sequence indices organized by taxonomy
        full_tax_label_from_seq_id_dict (dict): Full taxonomic labels for each sequence
        nested_dicts_of_taxa (dict): Nested dictionary structure of taxonomic labels
        taxon_label_to_taxon_id (dict): Mapping from taxon labels to integer IDs
        taxon_id_to_taxon_label (dict): Mapping from integer IDs to taxon labels
    """    
    with open(input_dir + "list_of_seq_indices_in_taxon_at_rank_dict.pkl", 'wb') as f:
        pickle.dump(list_of_seq_indices_in_taxon_at_rank_dict, f)
    with open(input_dir + "list_of_taxon_labels_in_taxon_at_rank_dict.pkl", 'wb') as f:
        pickle.dump(list_of_taxon_labels_in_taxon_at_rank_dict, f)
    with open(input_dir + "list_of_taxon_labels_at_rank_dict.pkl", 'wb') as f:
        pickle.dump(list_of_taxon_labels_at_rank_dict, f)
    with open(input_dir + "nested_list_of_seq_indices.pkl", 'wb') as f:
        pickle.dump(nested_list_of_seq_indices, f)
    with open(input_dir + "full_tax_label_from_seq_id_dict.pkl", 'wb') as f:
        pickle.dump(full_tax_label_from_seq_id_dict, f)
    with open(input_dir + "nested_dicts_of_taxa.pkl", 'wb') as f:
        pickle.dump(nested_dicts_of_taxa, f)
    with open(input_dir + "taxon_label_to_taxon_id.pkl", 'wb') as f:
        pickle.dump(taxon_label_to_taxon_id, f)
    with open(input_dir + "taxon_id_to_taxon_label.pkl", 'wb') as f:
        pickle.dump(taxon_id_to_taxon_label, f)

def construct_seq_taxon_ids(full_tax_labels, seq_indices, taxon_label_to_taxon_id):
    """
    Constructs an array of taxon IDs for each sequence at each taxonomic rank.
    
    For each sequence, maps its taxonomic labels at all ranks (domain through species)
    to their corresponding integer taxon IDs using rank-qualified keys.
    
    Args:
        full_tax_labels (list): List of taxonomic classifications for sequences from domain to species
                               e.g. [['Bacteria', 'Firmicutes', ...], ['Archaea', 'Euryarchaeota', ...]]
        seq_indices (list): List of sequence indices to include
        taxon_label_to_taxon_id (dict): Maps rank-qualified taxon labels (str) -> taxon ID (int)
    
    Returns:
        np.ndarray: Matrix of shape (n_seqs, 7) with dtype int32 containing taxon IDs
                   seq_taxon_ids[i, j] is the taxon ID for sequence i at rank j
    
    Example:
        If sequence 0 has taxonomy ['Bacteria', 'Firmicutes', 'Bacilli', ...]
        and the taxon_label_to_taxon_id maps these to [1, 15, 42, ...]
        then seq_taxon_ids[0, :] = [1, 15, 42, ...]
    """
    n_seqs = len(seq_indices)
    seq_taxon_ids = np.zeros((n_seqs, 7), dtype=np.int32)
    
    # For each sequence, map its taxon labels to taxon IDs at all ranks
    for i in range(n_seqs):
        full_tax_label = full_tax_labels[i]
        # For each rank (0=domain, 1=phylum, ..., 6=species)
        for rank in range(7):
            taxon_label = full_tax_label[rank]
            ranked_taxon_key = make_ranked_taxon_key(rank, taxon_label)
            if ranked_taxon_key in taxon_label_to_taxon_id:
                taxon_id = taxon_label_to_taxon_id[ranked_taxon_key]
            elif taxon_label in taxon_label_to_taxon_id:
                # Legacy fallback for older mapping files that used plain labels only.
                taxon_id = taxon_label_to_taxon_id[taxon_label]
            else:
                raise KeyError(
                    f"Could not find taxon ID for rank={rank} taxon='{taxon_label}'. "
                    f"Tried keys '{ranked_taxon_key}' and '{taxon_label}'."
                )
            seq_taxon_ids[i, rank] = taxon_id
    
    return seq_taxon_ids

def construct_pairwise_ranks(full_tax_labels, seq_indices):
    """
    Constructs a pairwise rank matrix for sequences.
    
    For each pair of sequences, computes the taxonomic rank at which both sequences
    share the same classification, while they belong to different classifications at 
    the rank immediately below (rank + 1).
    
    For example:
        - If two sequences differ at domain level, their pairwise rank is -1 (no shared classification)
        - If two sequences differ at phylum level, their pairwise rank is 0 (shared domain, different phylum)
        - If two sequences share the same genus but different species, their pairwise rank is 5 (genus)
        - If two sequences are identical at all ranks, their pairwise rank is 6 (species)
    
    Args:
        full_tax_labels (list): List of taxonomic classifications for sequences from domain to species
                               e.g. [['Bacteria', 'Firmicutes', ...], ['Archaea', 'Euryarchaeota', ...]]
        seq_indices (list): List of sequence indices to include
    
    Returns:
        np.ndarray: Matrix of shape (n_seqs, n_seqs) with dtype int8 containing pairwise ranks
    """
    n_seqs = len(seq_indices)
    pairwise_ranks = np.zeros((n_seqs, n_seqs), dtype=np.int8)
    
    # Iterate through all pairs of sequences
    for i in range(n_seqs):
        for j in range(n_seqs):
            if i == j:
                # Same sequence: set to species level (rank 6)
                pairwise_ranks[i, j] = 6
            else:
                # Find the deepest rank where they share the same classification
                tax_i = full_tax_labels[i]
                tax_j = full_tax_labels[j]
                
                # Start from rank -1 (no shared classification)
                shared_rank = -1
                
                # Check each rank from domain (0) to species (6)
                for rank in range(7):
                    if tax_i[rank] == tax_j[rank]:
                        shared_rank = rank
                    else:
                        # They differ at this rank, so stop
                        break
                
                pairwise_ranks[i, j] = shared_rank
    
    return pairwise_ranks

def construct_pairwise_masks(seq_taxon_ids):
    """
    Constructs pairwise positive and negative masks for all taxonomic ranks.
    
    For each rank, creates boolean masks indicating which pairs of sequences share
    the same taxon (positive mask) or have different taxa (negative mask).
    The diagonal is set to False for both masks since sequences should not be 
    compared with themselves.
    
    Args:
        seq_taxon_ids (np.ndarray): Matrix of shape (n_seqs, 7) with dtype int32
                                    containing taxon IDs for each sequence at each rank
    
    Returns:
        tuple: (pairwise_pos_masks, pairwise_neg_masks)
            - pairwise_pos_masks (np.ndarray): Shape (7, n_seqs, n_seqs), dtype bool
                                               True where pairs share the same taxon at each rank
            - pairwise_neg_masks (np.ndarray): Shape (7, n_seqs, n_seqs), dtype bool
                                               True where pairs have different taxa at each rank
    
    Example:
        For rank 0 (domain), if sequences i and j are both Bacteria:
            pairwise_pos_masks[0, i, j] = True
            pairwise_neg_masks[0, i, j] = False
        For rank 0 (domain), if sequence i is Bacteria and j is Archaea:
            pairwise_pos_masks[0, i, j] = False
            pairwise_neg_masks[0, i, j] = True
    """
    n_seqs = seq_taxon_ids.shape[0]
    n_ranks = 7
    
    # Initialize the masks
    # pairwise_pos_masks = np.zeros((n_ranks, n_seqs, n_seqs), dtype=bool)
    # pairwise_neg_masks = np.zeros((n_ranks, n_seqs, n_seqs), dtype=bool)
    
    # For each rank, create the masks
    # for rank in range(n_ranks):
    #     # Get taxon IDs at this rank for all sequences
    #     taxon_ids_at_rank = seq_taxon_ids[:, rank]  # Shape: (n_seqs,)
        
    #     # Create pairwise comparison: taxon_ids_at_rank[i] == taxon_ids_at_rank[j]
    #     # Broadcasting: (n_seqs, 1) == (1, n_seqs) -> (n_seqs, n_seqs)
    #     pos_mask = taxon_ids_at_rank[:, np.newaxis] == taxon_ids_at_rank[np.newaxis, :]
        
    #     # Negative mask is the inverse of positive mask
    #     neg_mask = ~pos_mask
        
    #     # Set diagonal to False for both masks (sequences should not be compared with themselves)
    #     np.fill_diagonal(pos_mask, False)
    #     np.fill_diagonal(neg_mask, False)
        
    #     # Store the masks
    #     pairwise_pos_masks[rank] = pos_mask
    #     pairwise_neg_masks[rank] = neg_mask
    
    print("Skipping pairwise_pos_masks and pairwise_neg_masks construction to save memory.")
    return None, None

def construct_pairwise_mrca_taxon_ids(seq_taxon_ids, pairwise_ranks):
    """
    Constructs a pairwise MRCA (Most Recent Common Ancestor) taxon ID matrix.
    
    For each pair of sequences, determines the taxon ID of their most recent common ancestor,
    which is the taxon at the highest (most specific) rank they share in common.
    
    For example:
        - If two sequences share the same species, their MRCA taxon ID is their species taxon ID
        - If they share the same genus but different species, their MRCA taxon ID is their genus taxon ID
        - If they differ at domain level, their MRCA taxon ID is -1 (no common ancestor in the taxonomy)
    
    Args:
        seq_taxon_ids (np.ndarray): Matrix of shape (n_seqs, 7) with dtype int32
                                    containing taxon IDs for each sequence at each rank
        pairwise_ranks (np.ndarray): Matrix of shape (n_seqs, n_seqs) with dtype int8
                                     containing the shared taxonomic rank for each pair
                                     (-1 means different domains, 0-6 means shared at that rank)
    
    Returns:
        np.ndarray: Matrix of shape (n_seqs, n_seqs) with dtype int32 containing MRCA taxon IDs
                   pairwise_mrca_taxon_ids[i, j] is the taxon ID of the MRCA for sequences i and j
                   Value is -1 if sequences differ at domain level
    
    Example:
        If sequences i and j both have genus taxon ID 42 but different species,
        and pairwise_ranks[i, j] = 5 (genus level),
        then pairwise_mrca_taxon_ids[i, j] = 42
    """
    n_seqs = seq_taxon_ids.shape[0]
    pairwise_mrca_taxon_ids = np.zeros((n_seqs, n_seqs), dtype=np.int32)
    
    # For each pair of sequences
    for i in range(n_seqs):
        for j in range(n_seqs):
            shared_rank = pairwise_ranks[i, j]
            
            if shared_rank == -1:
                # No common ancestor at domain level
                pairwise_mrca_taxon_ids[i, j] = -1
            else:
                # MRCA is at the shared rank
                pairwise_mrca_taxon_ids[i, j] = seq_taxon_ids[i, shared_rank]
    
    return pairwise_mrca_taxon_ids

def load_red_trees(arc_decorated_tree_path="/home/haig/Repos/micro16s/redvals/decorated_trees/ar53_r226_decorated.pkl", 
                   bac_decorated_tree_path="/home/haig/Repos/micro16s/redvals/decorated_trees/bac120_r226_decorated.pkl", 
                   precomputed_mapping_path="/home/haig/Repos/micro16s/redvals/taxon_mappings/taxon_to_node_mapping_r226.pkl"):
    """
    Load the RedTrees object for calculating phylogenetic distances.
    
    Args:
        arc_decorated_tree_path (str): Path to the Archaea decorated tree pickle file
        bac_decorated_tree_path (str): Path to the Bacteria decorated tree pickle file
        precomputed_mapping_path (str): Path to the taxon to node mapping pickle file
    
    Returns:
        RedTree: Initialized RedTree object with decorated trees and taxon mappings
    """
    # Initialise (already decorated) RedTree object
    red_trees = RedTree(bac_decorated_tree_path, arc_decorated_tree_path, verbose=False)
    
    # The trees must already be decorated with RED values
    assert red_trees.is_decorated()
    
    # Load the taxon to node mappings
    red_trees.load_taxa_to_node_mapping(precomputed_mapping_path)
    
    return red_trees

def print_eta_progress(prefix, done, total, start_time):
    """
    Prints progress with elapsed time and estimated time remaining.
    """
    if total <= 0:
        return
    elapsed_seconds = max(time.time() - start_time, 1e-9)
    completion_fraction = done / total
    rate = done / elapsed_seconds
    remaining_seconds = ((total - done) / rate) if rate > 0 else float("inf")
    elapsed_minutes = elapsed_seconds / 60.0
    remaining_minutes = remaining_seconds / 60.0 if np.isfinite(remaining_seconds) else float("inf")
    eta_text = f"{remaining_minutes:.2f}m" if np.isfinite(remaining_minutes) else "unknown"
    print(
        f"{prefix}{done:,}/{total:,} ({completion_fraction * 100:.1f}%) "
        f"| elapsed={elapsed_minutes:.2f}m | eta={eta_text}"
    )

def construct_redvals_id_lookup(gtdb_ids, red_trees):
    """
    Builds a dictionary mapping GTDB IDs to redvals IDs.

    This is used by exact pairwise distance construction to avoid repeatedly
    querying the RedTree object for every split and every sequence.

    Hard fails if any GTDB IDs from the dataset are missing in the decorated trees.
    """
    print("Building GTDB ID -> redvals ID lookup...")
    unique_gtdb_ids = sorted(set(gtdb_ids))
    n_unique = len(unique_gtdb_ids)

    redvals_id_from_gtdb_id = {}
    missing_gtdb_ids = []

    lookup_start_time = time.time()
    last_eta_print_time = lookup_start_time

    for i, gtdb_id in enumerate(unique_gtdb_ids, 1):
        try:
            redvals_id_from_gtdb_id[gtdb_id] = red_trees.get_redvals_id(gtdb_id)
        except Exception:
            missing_gtdb_ids.append(gtdb_id)

        now = time.time()
        if now - last_eta_print_time >= ETA_REPORT_INTERVAL_SECONDS:
            print_eta_progress("[GTDB->redvals] ", i, n_unique, lookup_start_time)
            last_eta_print_time = now

    # Final progress print
    print_eta_progress("[GTDB->redvals] ", n_unique, n_unique, lookup_start_time)

    if missing_gtdb_ids:
        preview = missing_gtdb_ids[:20]
        raise KeyError(
            f"Found {len(missing_gtdb_ids)} GTDB IDs in FULL_seqs.fasta that were not found in the decorated GTDB trees. "
            f"First {len(preview)} missing IDs: {preview}"
        )

    return redvals_id_from_gtdb_id

def _fill_pairwise_distances_for_domain_tree(pairwise_distances, tree, redvals_id_to_local_indices, domain_label, split_prefix):
    """
    Fills same-domain entries of pairwise_distances using exact MRCA RED distances.

    The algorithm is equivalent to computing dist_between_nodes for all leaf pairs,
    but is much faster by writing block assignments at each internal node:
      - each internal node defines pairs whose MRCA is exactly that node
      - those pairs are assigned node.red_distance in one vectorized operation
    """
    # Build permutation of local indices in tree leaf order.
    # This allows each subtree to correspond to a contiguous index interval.
    permutation_local_indices = []
    selected_leaf_count_from_redvals_id = {}

    for leaf_node in tree.get_terminals():
        local_indices = redvals_id_to_local_indices.get(leaf_node.redvals_id, None)
        if local_indices:
            permutation_local_indices.extend(local_indices)
            selected_leaf_count_from_redvals_id[leaf_node.redvals_id] = len(local_indices)

    if len(permutation_local_indices) == 0:
        print(f"{split_prefix}[{domain_label}] No sequences in this domain. Skipping.")
        return

    permutation_local_indices = np.asarray(permutation_local_indices, dtype=np.int64)
    n_selected_domain_seqs = len(permutation_local_indices)
    print(f"{split_prefix}[{domain_label}] Exact MRCA fill for {n_selected_domain_seqs:,} sequences...")

    # First pass (postorder): count selected descendants for every node.
    # selected_count_from_redvals_id[node_id] = number of selected leaves under that node.
    postorder_nodes = list(tree.find_clades(order="postorder"))
    selected_count_from_redvals_id = {}

    count_pass_start_time = time.time()
    last_eta_print_time = count_pass_start_time

    for i, node in enumerate(postorder_nodes, 1):
        if node.is_terminal():
            selected_count_from_redvals_id[node.redvals_id] = selected_leaf_count_from_redvals_id.get(node.redvals_id, 0)
        else:
            selected_count = 0
            for child in node.clades:
                selected_count += selected_count_from_redvals_id[child.redvals_id]
            selected_count_from_redvals_id[node.redvals_id] = selected_count

        now = time.time()
        if now - last_eta_print_time >= ETA_REPORT_INTERVAL_SECONDS:
            print_eta_progress(f"{split_prefix}[{domain_label}] Count pass: ", i, len(postorder_nodes), count_pass_start_time)
            last_eta_print_time = now

    print_eta_progress(f"{split_prefix}[{domain_label}] Count pass: ", len(postorder_nodes), len(postorder_nodes), count_pass_start_time)

    # Second pass (preorder): assign exact RED distance blocks for pairs whose MRCA is each internal node.
    n_internal_with_selected = 0
    for node in postorder_nodes:
        if not node.is_terminal() and selected_count_from_redvals_id[node.redvals_id] > 0:
            n_internal_with_selected += 1

    fill_pass_start_time = time.time()
    last_eta_print_time = fill_pass_start_time
    processed_internal = 0

    # Stack holds tuples: (node, start_offset_in_permutation)
    # The selected descendants of each node are guaranteed to occupy a contiguous
    # interval in permutation_local_indices.
    stack = [(tree.root, 0)]
    while stack:
        node, node_start = stack.pop()
        node_selected_count = selected_count_from_redvals_id[node.redvals_id]

        if node_selected_count == 0:
            continue

        if node.is_terminal():
            continue

        processed_internal += 1

        # Build selected child spans in permutation space.
        child_spans = []
        child_start = node_start
        for child in node.clades:
            child_selected_count = selected_count_from_redvals_id[child.redvals_id]
            child_end = child_start + child_selected_count
            if child_selected_count > 0:
                child_spans.append((child, child_start, child_end))
            child_start = child_end

        # Assign this node's RED distance to cross-child blocks.
        # Those are exactly the pairs whose MRCA is this node.
        if len(child_spans) >= 2:
            red_distance = np.float32(node.red_distance)
            for i in range(len(child_spans)):
                left_start, left_end = child_spans[i][1], child_spans[i][2]
                left_indices = permutation_local_indices[left_start:left_end]
                for j in range(i + 1, len(child_spans)):
                    right_start, right_end = child_spans[j][1], child_spans[j][2]
                    right_indices = permutation_local_indices[right_start:right_end]
                    pairwise_distances[np.ix_(left_indices, right_indices)] = red_distance
                    pairwise_distances[np.ix_(right_indices, left_indices)] = red_distance

        # Traverse preorder (parent before descendants) so descendant assignments
        # can override parent-level values within each child subtree.
        for child, child_start, child_end in reversed(child_spans):
            stack.append((child, child_start))

        now = time.time()
        if now - last_eta_print_time >= ETA_REPORT_INTERVAL_SECONDS:
            print_eta_progress(
                f"{split_prefix}[{domain_label}] Fill pass: ",
                processed_internal,
                n_internal_with_selected,
                fill_pass_start_time,
            )
            last_eta_print_time = now

    print_eta_progress(
        f"{split_prefix}[{domain_label}] Fill pass: ",
        processed_internal,
        n_internal_with_selected,
        fill_pass_start_time,
    )

def construct_pairwise_distances(gtdb_ids_for_split, redvals_id_from_gtdb_id, red_trees, distance_between_domains, split_name=""):
    """
    Constructs exact pairwise RED distances for all sequences in a split.

    This method is equivalent to computing RedTree.dist_between_nodes for every
    pair of leaves, but is much faster by assigning subtree blocks using each
    internal node's precomputed red_distance.

    Args:
        gtdb_ids_for_split (list): GTDB IDs for the split, in local matrix order
        redvals_id_from_gtdb_id (dict): Mapping GTDB ID -> redvals ID
        red_trees (RedTree): Decorated RedTree object
        distance_between_domains (float): Distance assigned to cross-domain pairs
        split_name (str): Optional split name used for informative logging

    Returns:
        np.ndarray: Exact pairwise RED distance matrix of shape (n_seqs, n_seqs), dtype float32
    """
    n_seqs = len(gtdb_ids_for_split)
    split_prefix = f"[{split_name}] " if split_name else ""

    # Default all off-diagonal values to cross-domain distance. Same-domain values
    # will be overwritten by exact MRCA distances in the domain-specific passes.
    pairwise_distances = np.full((n_seqs, n_seqs), np.float32(distance_between_domains), dtype=np.float32)
    np.fill_diagonal(pairwise_distances, 0.0)

    # Build mapping from redvals leaf ID to local sequence indices.
    redvals_id_to_local_indices = {}
    n_bac = 0
    n_arc = 0
    for local_i, gtdb_id in enumerate(gtdb_ids_for_split):
        if gtdb_id not in redvals_id_from_gtdb_id:
            raise KeyError(f"GTDB ID '{gtdb_id}' was not found in the GTDB->redvals lookup dictionary.")

        redvals_id = redvals_id_from_gtdb_id[gtdb_id]
        redvals_id_to_local_indices.setdefault(redvals_id, []).append(local_i)

        if redvals_id.startswith("bac"):
            n_bac += 1
        elif redvals_id.startswith("arc"):
            n_arc += 1
        else:
            raise ValueError(f"Unexpected redvals ID prefix for '{redvals_id}' from GTDB ID '{gtdb_id}'.")

    print(f"{split_prefix}Exact pairwise RED distance construction started for {n_seqs:,} sequences ({n_bac:,} bac, {n_arc:,} arc).")
    total_start_time = time.time()

    # Handle duplicates (same GTDB leaf appearing multiple times in FASTA).
    # Distances between duplicate leaf entries are exactly zero.
    for local_indices in redvals_id_to_local_indices.values():
        if len(local_indices) > 1:
            duplicate_idx = np.asarray(local_indices, dtype=np.int64)
            pairwise_distances[np.ix_(duplicate_idx, duplicate_idx)] = 0.0

    # Fill bacterial and archaeal blocks exactly.
    # We keep this sequential for robustness because both passes write into the
    # same large matrix and the bottleneck is memory bandwidth rather than CPU.
    _fill_pairwise_distances_for_domain_tree(
        pairwise_distances=pairwise_distances,
        tree=red_trees.bac_tree,
        redvals_id_to_local_indices=redvals_id_to_local_indices,
        domain_label="bac",
        split_prefix=split_prefix,
    )
    _fill_pairwise_distances_for_domain_tree(
        pairwise_distances=pairwise_distances,
        tree=red_trees.arc_tree,
        redvals_id_to_local_indices=redvals_id_to_local_indices,
        domain_label="arc",
        split_prefix=split_prefix,
    )

    elapsed_minutes = (time.time() - total_start_time) / 60.0
    print(f"{split_prefix}Exact pairwise RED distance construction complete ({elapsed_minutes:.2f} minutes).")

    return pairwise_distances

def get_distance_from_taxon_label(taxon_label, rank, red_trees):
    """
    Calculate the phylogenetic distance for a given taxon at a specified rank.
    
    This function is only used for constructing the auxiliary taxon lookup array
    (distances_lookup_array.npy). Exact pairwise distances are now computed from
    leaf MRCAs directly in construct_pairwise_distances().
    
    Args:
        taxon_label (str): The taxonomic label (e.g. "Bacteria", "Proteobacteria", etc.)
        rank (int): The taxonomic rank (0=domain, 1=phylum, ..., 6=species)
        red_trees (RedTree): RedTree object for calculating distances
    
    Returns:
        float: The phylogenetic distance within that taxon
    """
    # Special case: if taxon is "Prokaryota", it represents both domains
    if taxon_label == "Prokaryota":
        return RED_DISTANCE_BETWEEN_DOMAINS
    
    # Format the taxon label with the rank prefix
    # e.g. "Bacteria" -> "d__Bacteria"
    formatted_taxon_label = RANK_PREFIXES[rank] + taxon_label
    
    # Use the RedTrees object to get the distance
    distance = red_trees.get_distance_in_taxon(formatted_taxon_label)
    
    return distance

def construct_distances_from_mrca_taxon_id_dict(taxon_id_to_taxon_label, taxon_id_to_rank, red_trees):
    """
    Constructs a lookup array mapping MRCA taxon IDs to their phylogenetic distances.
    
    Uses the cached rank computed when taxon IDs were assigned, eliminating the
    expensive scan over pairwise matrices. The returned array uses taxon IDs
    as indices, allowing O(1) lookup time. Taxon ID -1 (different domains) is
    handled separately and always maps to RED_DISTANCE_BETWEEN_DOMAINS.

    Note:
        This lookup array is retained for compatibility/debugging.
        pairwise_distances.npy is now generated via exact leaf-MRCA calculations.
    
    Args:
        taxon_id_to_taxon_label (dict): Maps integer taxon IDs to taxon label strings
        taxon_id_to_rank (dict): Maps integer taxon IDs to their taxonomic rank
        red_trees (RedTree): RedTree object for calculating distances
    
    Returns:
        tuple: (distances_lookup_array, distance_between_domains)
            - distances_lookup_array (np.ndarray): 1D array of shape (max_taxon_id + 1,) with dtype float32
                                                   where distances_lookup_array[taxon_id] = distance
            - distance_between_domains (float): Distance for taxon_id = -1 (different domains)
    """
    if not taxon_id_to_taxon_label:
        return np.zeros(0, dtype=np.float32), RED_DISTANCE_BETWEEN_DOMAINS
    
    max_taxon_id = max(taxon_id_to_taxon_label.keys())
    distances_lookup_array = np.zeros(max_taxon_id + 1, dtype=np.float32)
    
    for i, (taxon_id, taxon_label) in enumerate(taxon_id_to_taxon_label.items(), 1):
        rank = taxon_id_to_rank[taxon_id]
        distances_lookup_array[taxon_id] = get_distance_from_taxon_label(taxon_label, rank, red_trees)
    
    distance_between_domains = RED_DISTANCE_BETWEEN_DOMAINS
    
    return distances_lookup_array, distance_between_domains

def save_labels_arrays(labels_dir, pairwise_ranks, seq_taxon_ids, pairwise_pos_masks, pairwise_neg_masks, 
                       pairwise_mrca_taxon_ids, pairwise_distances, distances_lookup_array, distance_between_domains):
    """
    Saves the label arrays as .npy files.
    
    Args:
        labels_dir (str): Directory to save the label arrays
        pairwise_ranks (np.ndarray): Pairwise rank matrix to save
        seq_taxon_ids (np.ndarray): Sequence taxon IDs matrix to save
        pairwise_pos_masks (np.ndarray): Pairwise positive masks to save
        pairwise_neg_masks (np.ndarray): Pairwise negative masks to save
        pairwise_mrca_taxon_ids (np.ndarray): Pairwise MRCA taxon IDs matrix to save
        pairwise_distances (np.ndarray): Pairwise phylogenetic distances matrix to save
        distances_lookup_array (np.ndarray): 1D lookup array mapping taxon IDs to distances
        distance_between_domains (float): Distance value for different domains (taxon_id = -1)
    """
    np.save(labels_dir + "pairwise_ranks.npy", pairwise_ranks)
    np.save(labels_dir + "seq_taxon_ids.npy", seq_taxon_ids)
    # np.save(labels_dir + "pairwise_pos_masks.npy", pairwise_pos_masks)  UNUSED AND LARGE FILE
    # np.save(labels_dir + "pairwise_neg_masks.npy", pairwise_neg_masks)  UNUSED AND LARGE FILE
    np.save(labels_dir + "pairwise_mrca_taxon_ids.npy", pairwise_mrca_taxon_ids)
    np.save(labels_dir + "pairwise_distances.npy", pairwise_distances)
    np.save(labels_dir + "distances_lookup_array.npy", distances_lookup_array)
    np.save(labels_dir + "distance_between_domains.npy", np.array([distance_between_domains], dtype=np.float32))


if __name__ == "__main__":

  start_time = time.time()
  # If output directory exists and is not empty, warn the user
  if os.path.exists(OUTPUT_DATASET_SPLIT_DIR) and os.listdir(OUTPUT_DATASET_SPLIT_DIR):
    print(f"Warning: {OUTPUT_DATASET_SPLIT_DIR} exists and is not empty. This script will overwrite the contents of this directory.")
    input("Press Enter to DELETE CONTENTS and continue...")
    # Delete the contents of the output directory
    for file in os.listdir(OUTPUT_DATASET_SPLIT_DIR):
      file_path = os.path.join(OUTPUT_DATASET_SPLIT_DIR, file)
      if os.path.isdir(file_path):
        shutil.rmtree(file_path)
      else:
        os.remove(file_path)

  # If the output directory doesn't exist, create it
  if not os.path.exists(OUTPUT_DATASET_SPLIT_DIR):
    os.makedirs(OUTPUT_DATASET_SPLIT_DIR)
  if not os.path.exists(OUTPUT_DATASET_SPLIT_DIR + "tax_objs/"):
    os.makedirs(OUTPUT_DATASET_SPLIT_DIR + "tax_objs/")
  if not os.path.exists(OUTPUT_DATASET_SPLIT_DIR + "tax_objs/train/"):
    os.makedirs(OUTPUT_DATASET_SPLIT_DIR + "tax_objs/train/")
  if not os.path.exists(OUTPUT_DATASET_SPLIT_DIR + "tax_objs/test/"):
    os.makedirs(OUTPUT_DATASET_SPLIT_DIR + "tax_objs/test/")
  if not os.path.exists(OUTPUT_DATASET_SPLIT_DIR + "tax_objs/excluded/"):
    os.makedirs(OUTPUT_DATASET_SPLIT_DIR + "tax_objs/excluded/")
  if not os.path.exists(OUTPUT_DATASET_SPLIT_DIR + "labels/"):
    os.makedirs(OUTPUT_DATASET_SPLIT_DIR + "labels/")
  if not os.path.exists(OUTPUT_DATASET_SPLIT_DIR + "labels/train/"):
    os.makedirs(OUTPUT_DATASET_SPLIT_DIR + "labels/train/")
  if not os.path.exists(OUTPUT_DATASET_SPLIT_DIR + "labels/test/"):
    os.makedirs(OUTPUT_DATASET_SPLIT_DIR + "labels/test/")
  if not os.path.exists(OUTPUT_DATASET_SPLIT_DIR + "labels/excluded/"):
    os.makedirs(OUTPUT_DATASET_SPLIT_DIR + "labels/excluded/")

  # Read the input file to get the number of sequences, taxonomic labels, and GTDB IDs
  all_tax_labels, all_gtdb_ids = read_input_file()
  n_seqs = len(all_tax_labels)
  if len(all_gtdb_ids) != n_seqs:
    raise ValueError(f"Mismatch between number of taxonomies ({n_seqs}) and GTDB IDs ({len(all_gtdb_ids)}).")
  all_seq_indices = set(range(n_seqs))

  print(f"Number of sequences: {n_seqs}\n")

  # Collect pointers to all sequences in all taxa at all ranks
  list_of_seq_indices_in_taxon_at_rank_dict = construct_seqs_in_taxon_dict(all_tax_labels, list(range(n_seqs)))

  # Select excluded taxa
  selected_taxa_labels, excluded_taxa_seq_indices, n_eligible_taxa = select_excluded_taxa(list_of_seq_indices_in_taxon_at_rank_dict)

  print(f"There were {n_eligible_taxa} eligible {RANKS_PLURAL[EXCLUDED_TAXA_RANK]}")
  print(f"Selected {N_EXCLUDED_TAXA} {RANKS_PLURAL[EXCLUDED_TAXA_RANK]} for exclusion:")
  for taxon_label in selected_taxa_labels:
      print(f"  - {taxon_label}: {len(list_of_seq_indices_in_taxon_at_rank_dict[EXCLUDED_TAXA_RANK][taxon_label])} sequences")
  print()

  # Select testing training sets
  remaining_seq_indices = all_seq_indices - excluded_taxa_seq_indices
  n_remaining_seqs = len(remaining_seq_indices)
  n_testing_seqs = int(n_remaining_seqs * TEST_TRAIN_RATIO)
  testing_seq_indices = set(random.sample(list(remaining_seq_indices), n_testing_seqs))
  training_seq_indices = remaining_seq_indices - testing_seq_indices

  # Convert all to sorted lists
  excluded_taxa_seq_indices = sorted(list(excluded_taxa_seq_indices))
  testing_seq_indices = sorted(list(testing_seq_indices))
  training_seq_indices = sorted(list(training_seq_indices))

  print(f"Excluded taxa: {len(excluded_taxa_seq_indices)} sequences ({len(excluded_taxa_seq_indices)/n_seqs*100:.1f}%)")
  print(f"Testing set: {len(testing_seq_indices)} sequences ({len(testing_seq_indices)/n_seqs*100:.1f}%)")
  print(f"Training set: {len(training_seq_indices)} sequences ({len(training_seq_indices)/n_seqs*100:.1f}%)\n")

  # Write the indices to files
  write_seq_indices(excluded_taxa_seq_indices, EXCLUDED_TAXA_FILE)
  write_seq_indices(testing_seq_indices, TESTING_FILE)
  write_seq_indices(training_seq_indices, TRAINING_FILE)
  
  # Load RED trees for distance calculations
  red_trees = load_red_trees()
  # Build GTDB -> redvals ID lookup once (hard fail on missing GTDB IDs)
  redvals_id_from_gtdb_id = construct_redvals_id_lookup(all_gtdb_ids, red_trees)
  
  # Calculate distances between taxa
  distances_between_taxa_results = None # TODO: Implement this

  # Write the about file
  write_about_file(n_seqs, selected_taxa_labels, list_of_seq_indices_in_taxon_at_rank_dict,
                  excluded_taxa_seq_indices, testing_seq_indices, training_seq_indices,
                  distances_between_taxa_results)
  
  # Create taxonomic data structures and label arrays
  
  # Excluded taxa ---
  print("\nExcluded taxa ---")
  # Taxonomic data structures
  print("[Excluded] Starting seqs in taxon dict construction...")
  excluded_list_of_seq_indices_in_taxon_at_rank_dict = construct_seqs_in_taxon_dict(
      [all_tax_labels[i] for i in excluded_taxa_seq_indices], 
      excluded_taxa_seq_indices
  )
  print("[Excluded] Starting labels in taxon dict construction...")
  excluded_list_of_taxon_labels_in_taxon_at_rank_dict = construct_labels_in_taxon_dict(
      [all_tax_labels[i] for i in excluded_taxa_seq_indices]
  )
  print("[Excluded] Starting labels at rank dict construction...")
  excluded_list_of_taxon_labels_at_rank_dict = construct_labels_at_rank_dict(
      excluded_list_of_taxon_labels_in_taxon_at_rank_dict
  )
  print_unique_taxa_counts(excluded_list_of_taxon_labels_at_rank_dict, prefix="[Excluded] ")
  print("[Excluded] Starting nested list of seq indices construction...")
  excluded_nested_list_of_seq_indices = construct_nested_seq_list(
      excluded_list_of_seq_indices_in_taxon_at_rank_dict,
      excluded_list_of_taxon_labels_in_taxon_at_rank_dict,
      excluded_list_of_taxon_labels_at_rank_dict
  )
  print("[Excluded] Starting full tax label from seq id dict construction...")
  excluded_full_tax_label_from_seq_id_dict = construct_seq_id_to_full_label_dict(
      [all_tax_labels[i] for i in excluded_taxa_seq_indices],
      excluded_taxa_seq_indices
  )
  print("[Excluded] Starting nested dicts of taxa construction...")
  excluded_nested_dicts_of_taxa = construct_nested_dicts_of_taxa(
      excluded_list_of_seq_indices_in_taxon_at_rank_dict, 
      excluded_list_of_taxon_labels_in_taxon_at_rank_dict
  )
  print("[Excluded] Starting taxon label to taxon id dict construction...")
  excluded_taxon_label_to_taxon_id, excluded_taxon_id_to_taxon_label, excluded_taxon_id_to_rank = construct_taxon_id_mappings(
      excluded_list_of_taxon_labels_at_rank_dict
  )
  # Label arrays
  print("[Excluded] Starting pairwise ranks construction...")
  excluded_pairwise_ranks = construct_pairwise_ranks(
      [all_tax_labels[i] for i in excluded_taxa_seq_indices],
      excluded_taxa_seq_indices
  )
  print("[Excluded] Starting seq taxon ids construction...")
  excluded_seq_taxon_ids = construct_seq_taxon_ids(
      [all_tax_labels[i] for i in excluded_taxa_seq_indices],
      excluded_taxa_seq_indices,
      excluded_taxon_label_to_taxon_id
  )
  print("[Excluded] Starting pairwise pos masks construction...")
  excluded_pairwise_pos_masks, excluded_pairwise_neg_masks = construct_pairwise_masks(
    excluded_seq_taxon_ids
  )
  print("[Excluded] Starting pairwise mrca taxon ids construction...")
  excluded_pairwise_mrca_taxon_ids = construct_pairwise_mrca_taxon_ids(
    excluded_seq_taxon_ids, excluded_pairwise_ranks
  )
  print("[Excluded] Starting distances from mrca taxon id construction...")
  excluded_distances_lookup_array, excluded_distance_between_domains = construct_distances_from_mrca_taxon_id_dict(
      excluded_taxon_id_to_taxon_label, excluded_taxon_id_to_rank, red_trees
  )
  print("[Excluded] Starting exact pairwise distances construction...")
  excluded_gtdb_ids = [all_gtdb_ids[i] for i in excluded_taxa_seq_indices]
  excluded_pairwise_distances = construct_pairwise_distances(
      excluded_gtdb_ids, redvals_id_from_gtdb_id, red_trees, excluded_distance_between_domains, split_name="Excluded"
  )
  print_label_array_sanity_checks("[Excluded] ", excluded_seq_taxon_ids, excluded_pairwise_ranks,
                                   excluded_pairwise_pos_masks, excluded_pairwise_neg_masks,
                                   excluded_pairwise_mrca_taxon_ids, excluded_pairwise_distances)
  print("[Excluded] Saving taxonomic data structures and label arrays...")
  # Save taxonomic data structures and label arrays
  save_pickles(OUTPUT_DATASET_SPLIT_DIR + "tax_objs/excluded/", excluded_list_of_seq_indices_in_taxon_at_rank_dict, excluded_list_of_taxon_labels_in_taxon_at_rank_dict,
      excluded_list_of_taxon_labels_at_rank_dict, excluded_nested_list_of_seq_indices,
      excluded_full_tax_label_from_seq_id_dict, excluded_nested_dicts_of_taxa,
      excluded_taxon_label_to_taxon_id, excluded_taxon_id_to_taxon_label)
  save_labels_arrays(OUTPUT_DATASET_SPLIT_DIR + "labels/excluded/", excluded_pairwise_ranks, excluded_seq_taxon_ids,
                     excluded_pairwise_pos_masks, excluded_pairwise_neg_masks, excluded_pairwise_mrca_taxon_ids, excluded_pairwise_distances,
                     excluded_distances_lookup_array, excluded_distance_between_domains)

  # Testing set ---
  print("\nTesting set ---")
  # Taxonomic data structures
  print("[Test] Starting seqs in taxon dict construction...")
  test_list_of_seq_indices_in_taxon_at_rank_dict = construct_seqs_in_taxon_dict(
      [all_tax_labels[i] for i in testing_seq_indices], 
      testing_seq_indices
  )
  print("[Test] Starting labels in taxon dict construction...")
  test_list_of_taxon_labels_in_taxon_at_rank_dict = construct_labels_in_taxon_dict(
      [all_tax_labels[i] for i in testing_seq_indices]
  )
  print("[Test] Starting labels at rank dict construction...")
  test_list_of_taxon_labels_at_rank_dict = construct_labels_at_rank_dict(
      test_list_of_taxon_labels_in_taxon_at_rank_dict
  )
  print_unique_taxa_counts(test_list_of_taxon_labels_at_rank_dict, prefix="[Test] ")
  print("[Test] Starting nested list of seq indices construction...")
  test_nested_list_of_seq_indices = construct_nested_seq_list(
      test_list_of_seq_indices_in_taxon_at_rank_dict,
      test_list_of_taxon_labels_in_taxon_at_rank_dict,
      test_list_of_taxon_labels_at_rank_dict
  )
  print("[Test] Starting full tax label from seq id dict construction...")
  test_full_tax_label_from_seq_id_dict = construct_seq_id_to_full_label_dict(
      [all_tax_labels[i] for i in testing_seq_indices],
      testing_seq_indices
  )
  print("[Test] Starting nested dicts of taxa construction...")
  test_nested_dicts_of_taxa = construct_nested_dicts_of_taxa(
      test_list_of_seq_indices_in_taxon_at_rank_dict, 
      test_list_of_taxon_labels_in_taxon_at_rank_dict
  )
  print("[Test] Starting taxon label to taxon id dict construction...")
  test_taxon_label_to_taxon_id, test_taxon_id_to_taxon_label, test_taxon_id_to_rank = construct_taxon_id_mappings(
      test_list_of_taxon_labels_at_rank_dict
  )
  # Label arrays
  print("[Test] Starting pairwise ranks construction...")
  test_pairwise_ranks = construct_pairwise_ranks(
      [all_tax_labels[i] for i in testing_seq_indices],
      testing_seq_indices
  )
  print("[Test] Starting seq taxon ids construction...")
  test_seq_taxon_ids = construct_seq_taxon_ids(
      [all_tax_labels[i] for i in testing_seq_indices],
      testing_seq_indices,
      test_taxon_label_to_taxon_id
  )
  print("[Test] Starting pairwise pos masks construction...")
  test_pairwise_pos_masks, test_pairwise_neg_masks = construct_pairwise_masks(
    test_seq_taxon_ids
  )
  print("[Test] Starting pairwise mrca taxon ids construction...")
  test_pairwise_mrca_taxon_ids = construct_pairwise_mrca_taxon_ids(
    test_seq_taxon_ids, test_pairwise_ranks
  )
  print("[Test] Starting distances from mrca taxon id construction...")
  test_distances_lookup_array, test_distance_between_domains = construct_distances_from_mrca_taxon_id_dict(
      test_taxon_id_to_taxon_label, test_taxon_id_to_rank, red_trees
  )
  print("[Test] Starting exact pairwise distances construction...")
  test_gtdb_ids = [all_gtdb_ids[i] for i in testing_seq_indices]
  test_pairwise_distances = construct_pairwise_distances(
      test_gtdb_ids, redvals_id_from_gtdb_id, red_trees, test_distance_between_domains, split_name="Test"
  )
  print_label_array_sanity_checks("[Test] ", test_seq_taxon_ids, test_pairwise_ranks,
                                   test_pairwise_pos_masks, test_pairwise_neg_masks,
                                   test_pairwise_mrca_taxon_ids, test_pairwise_distances)
  print("[Test] Saving taxonomic data structures and label arrays...")
  # Save taxonomic data structures and label arrays
  save_pickles(OUTPUT_DATASET_SPLIT_DIR + "tax_objs/test/", test_list_of_seq_indices_in_taxon_at_rank_dict, test_list_of_taxon_labels_in_taxon_at_rank_dict,
      test_list_of_taxon_labels_at_rank_dict, test_nested_list_of_seq_indices,
      test_full_tax_label_from_seq_id_dict, test_nested_dicts_of_taxa,
      test_taxon_label_to_taxon_id, test_taxon_id_to_taxon_label)
  save_labels_arrays(OUTPUT_DATASET_SPLIT_DIR + "labels/test/", test_pairwise_ranks, test_seq_taxon_ids,
                     test_pairwise_pos_masks, test_pairwise_neg_masks, test_pairwise_mrca_taxon_ids, test_pairwise_distances,
                     test_distances_lookup_array, test_distance_between_domains)
  
  # Training set ---
  print("\nTraining set ---")
  # Taxonomic data structures
  print("[Train] Starting seqs in taxon dict construction...")
  train_list_of_seq_indices_in_taxon_at_rank_dict = construct_seqs_in_taxon_dict(
      [all_tax_labels[i] for i in training_seq_indices], 
      training_seq_indices
  )
  print("[Train] Starting labels in taxon dict construction...")
  train_list_of_taxon_labels_in_taxon_at_rank_dict = construct_labels_in_taxon_dict(
      [all_tax_labels[i] for i in training_seq_indices]
  )
  print("[Train] Starting labels at rank dict construction...")
  train_list_of_taxon_labels_at_rank_dict = construct_labels_at_rank_dict(
      train_list_of_taxon_labels_in_taxon_at_rank_dict
  )
  print_unique_taxa_counts(train_list_of_taxon_labels_at_rank_dict, prefix="[Train] ")
  print("[Train] Starting nested list of seq indices construction...")
  train_nested_list_of_seq_indices = construct_nested_seq_list(
      train_list_of_seq_indices_in_taxon_at_rank_dict,
      train_list_of_taxon_labels_in_taxon_at_rank_dict,
      train_list_of_taxon_labels_at_rank_dict
  )
  print("[Train] Starting full tax label from seq id dict construction...")
  train_full_tax_label_from_seq_id_dict = construct_seq_id_to_full_label_dict(
      [all_tax_labels[i] for i in training_seq_indices],
      training_seq_indices
  )
  print("[Train] Starting nested dicts of taxa construction...")
  train_nested_dicts_of_taxa = construct_nested_dicts_of_taxa(
      train_list_of_seq_indices_in_taxon_at_rank_dict, 
      train_list_of_taxon_labels_in_taxon_at_rank_dict
  )
  print("[Train] Starting taxon label to taxon id dict construction...")
  train_taxon_label_to_taxon_id, train_taxon_id_to_taxon_label, train_taxon_id_to_rank = construct_taxon_id_mappings(
      train_list_of_taxon_labels_at_rank_dict
  )
  print("[Train] Starting pairwise ranks construction...")
  # Label arrays
  train_pairwise_ranks = construct_pairwise_ranks(
      [all_tax_labels[i] for i in training_seq_indices],
      training_seq_indices
  )
  print("[Train] Starting seq taxon ids construction...")
  train_seq_taxon_ids = construct_seq_taxon_ids(
      [all_tax_labels[i] for i in training_seq_indices],
      training_seq_indices,
      train_taxon_label_to_taxon_id
  )
  print("[Train] Starting pairwise pos masks construction...")
  train_pairwise_pos_masks, train_pairwise_neg_masks = construct_pairwise_masks(
    train_seq_taxon_ids
  )
  print("[Train] Starting pairwise mrca taxon ids construction...")
  train_pairwise_mrca_taxon_ids = construct_pairwise_mrca_taxon_ids(
    train_seq_taxon_ids, train_pairwise_ranks
  )
  print("[Train] Starting distances from mrca taxon id construction...")
  train_distances_lookup_array, train_distance_between_domains = construct_distances_from_mrca_taxon_id_dict(
      train_taxon_id_to_taxon_label, train_taxon_id_to_rank, red_trees
  )
  print("[Train] Starting exact pairwise distances construction...")
  train_gtdb_ids = [all_gtdb_ids[i] for i in training_seq_indices]
  train_pairwise_distances = construct_pairwise_distances(
      train_gtdb_ids, redvals_id_from_gtdb_id, red_trees, train_distance_between_domains, split_name="Train"
  )
  print_label_array_sanity_checks("[Train] ", train_seq_taxon_ids, train_pairwise_ranks,
                                   train_pairwise_pos_masks, train_pairwise_neg_masks,
                                   train_pairwise_mrca_taxon_ids, train_pairwise_distances)
  print("[Train] Saving taxonomic data structures and label arrays...")
  # Save taxonomic data structures and label arrays
  save_pickles(OUTPUT_DATASET_SPLIT_DIR + "tax_objs/train/", train_list_of_seq_indices_in_taxon_at_rank_dict, train_list_of_taxon_labels_in_taxon_at_rank_dict,
      train_list_of_taxon_labels_at_rank_dict, train_nested_list_of_seq_indices,
      train_full_tax_label_from_seq_id_dict, train_nested_dicts_of_taxa,
      train_taxon_label_to_taxon_id, train_taxon_id_to_taxon_label)
  save_labels_arrays(OUTPUT_DATASET_SPLIT_DIR + "labels/train/", train_pairwise_ranks, train_seq_taxon_ids,
                     train_pairwise_pos_masks, train_pairwise_neg_masks, train_pairwise_mrca_taxon_ids, train_pairwise_distances,
                     train_distances_lookup_array, train_distance_between_domains)
  
  print(f"\nDataset construction complete! (took {(time.time() - start_time)/60:.2f} minutes)\n")
