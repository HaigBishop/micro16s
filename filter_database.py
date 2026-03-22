"""
filter_database.py - A script for filtering sequences from a GTDB 16S rRNA database.
 - The output of this script should be used as input for the extract_regions.py script.

1. Reads the database file (e.g. ssu_all_r226.fna)
2. Filters sequences (also prints statistics for each filter)
    a) Presence in trees (sequence must be in either bacterial or archaeal tree)
    b) Taxonomic resolution (sequence must have at least MIN_TAXONOMIC_RESOLUTION taxonomic levels)
    c) Duplicate sequences (no duplicate DNA sequences - when duplicates are found, keeps only alphabetically first occurrences)
    d) Sequence length (sequence must be between MIN_LENGTH and MAX_LENGTH bp)
    e) Max taxon sizes (no taxon can be larger than MAX_SEQS_PER_TAXON[rank] - when more than the maximum, that number are randomly sampled)
    f) Min taxon sizes (no taxon can be smaller than MIN_SEQS_PER_TAXON[rank] - when less than the minimum, they are all removed)
    g) Max genes per genome (configurable cap on the number of 16S entries that survive per genome)
3. Writes the filtered sequences to a new file (e.g. ssu_all_r226_filtered.fna) with exact same FASTA record header format as input (DNA sequences are on one line each)
4. Calculates info and prints and writes the info
    - Number of sequences removed at each step
    - Number of sequences remaining at each step
    - Percentage of sequences removed at each step
    - Percentage of sequences remaining at each step
    - Number of sequences in the final database
    - Lengths of the sequences in the final database (min, max, 1p, 10p, 90p, 99p, mean, median, std)
    - All of the above reported for the combined dataset as well as separately for Bacteria and Archaea

Example GTDB FASTA entries:
>RS_GCF_028472785.1~NZ_CP047694.1-#4 d__Bacteria;p__Pseudomonadota;c__Gammaproteobacteria;o__Pseudomonadales;f__Marinomonadaceae;g__Marinomonas;s__Marinomonas mediterranea [location=3261577..3263111] [ssu_len=1535] [contig_len=4700249]
TACAGAGGGTGCAAGCGTTAATCGGAA...
>RS_GCF_008416195.1~NZ_VVXS01000056.1 d__Bacteria;p__Bacteroidota;c__Bacteroidia;o__Bacteroidales;f__Bacteroidaceae;g__Bacteroides;s__Bacteroides caccae [location=12..1537] [ssu_len=1526] [contig_len=5109]
TACGGAGGATCCGAGCGTTATCCGGAT...
>RS_GCF_015238635.1~NZ_CP049958.1-#7 d__Bacteria;p__Bacillota_A;c__Clostridia;o__Peptostreptococcales;f__Peptostreptococcaceae;g__Clostridioides;s__Clostridioides difficile [location=154321..155821] [ssu_len=1501] [contig_len=4313277]
TACGTAGGGGGCTAGCGTTATCCGGAT...

"""


# Imports
import random
import os
import statistics
import math
from collections import defaultdict
from Bio import SeqIO
import matplotlib.pyplot as plt
import datetime

# Local Imports
from redvals.redvals import RedTree
from utils import parent_dir



# Input -------------------------
INPUT_FILE = "/mnt/secondary/micro16s_dbs/16s_databases/ssu_all_r226.fna"
# Paths to the phylogenetic trees
BACTERIAL_TREE_PATH = "/home/haig/Repos/micro16s/redvals/decorated_trees/bac120_r226_decorated.pkl"
ARCHAEAL_TREE_PATH = "/home/haig/Repos/micro16s/redvals/decorated_trees/ar53_r226_decorated.pkl"

# Outputs -------------------------
# Filtered database
OUTPUT_FILE = "/mnt/secondary/micro16s_dbs/16s_databases/m16s_db_004/filtered/ssu_all_r226_filtered.fna"
# About file
ABOUT_FILE = "/mnt/secondary/micro16s_dbs/16s_databases/m16s_db_004/filtered/ssu_all_r226_filtered_about.txt"
# Sequence lengths plot
SEQ_LENGTHS_PLOT = "/mnt/secondary/micro16s_dbs/16s_databases/m16s_db_004/filtered/ssu_all_r226_filtered_seqs_lengths.png"


# Minimum taxonomic resolution to keep
MIN_TAXONOMIC_RESOLUTION = 6 # (0=Domain, 1=Phylum, 2=Class, 3=Order, 4=Family, 5=Genus, 6=Species)

# Min num sequences per taxa at each rank filter 
# (Sequences from taxa which have less than these number of sequence will be removed)
MIN_SEQS_PER_TAXON = [1, 1, 1, 1, 1, 1, 1] # [domain, phylum, class, order, family, genus, species]

# Max num sequences per taxa at each rank filter
MAX_SEQS_PER_TAXON = [1e14, 1e14, 1e14, 1e14, 1e14, 1e14, 1e14] # [domain, phylum, class, order, family, genus, species]

# Add priority sequences here (GTDB IDs like 'RS_GCF_028472785.1')
# These sequences bypass all filters but are only added once.
PRIORITY_SEQ_IDS = [
    # "RS_GCF_000016525.1~NC_009515.1", # Methanocatella smithii
    # "RS_GCF_003697165.2~NZ_CP033092.2" # Escherichia coli
    ]

# Length filter parameters
MIN_LENGTH = 0  # Minimum sequence length in bp
MAX_LENGTH = 1e6  # Maximum sequence length in bp

# Max number of 16S genes retained per genome
# NOTE: When limiting, we keep the first sequences encountered during the taxon iteration loops (earlier records survive).
MAX_GENES_PER_GENOME = None  # Set to None for no limit



# Load the phylogenetic trees
red_trees = RedTree(BACTERIAL_TREE_PATH, ARCHAEAL_TREE_PATH)

# Get all GTDB IDs of sequences in the trees
ALL_LEAF_NODE_GTDB_IDS = set(red_trees.get_node_ids(domain='both', node_type='leaf', id_type='gtdb'))
LEAF_NODE_IDS_WITHOUT_SEQS = set(ALL_LEAF_NODE_GTDB_IDS)

# Print first 10 GTDB IDs in the trees
print(f"---\nFirst 10 GTDB IDs in the trees: {list(ALL_LEAF_NODE_GTDB_IDS)[:10]}\n---")
# This prints: 
# ['GB_GCA_013214575.1', 'RS_GCF_001707885.1', 'GB_GCA_036265755.1', 'GB_GCA_963575695.1', 'GB_GCA_024408745.1', 'RS_GCF_021300655.1', 'GB_GCA_030524715.1', 'GB_GCA_019244295.1', 'GB_GCA_002711995.1', 'GB_GCA_028409275.1']


DOMAINS = ["Bacteria", "Archaea"]
STATS_GROUPS = [
    ('all', 'All'),
    ('Bacteria', 'Bacteria'),
    ('Archaea', 'Archaea')
]
FILTERING_SUMMARY_ORDER = [
    ('tree_filtered', 'Tree Filtered', -1),
    ('resolution_filtered', 'Resolution Filtered', -1),
    ('duplicate_filtered', 'Duplicate Filtered', -1),
    ('length_filtered', 'Length Filtered', -1),
    ('max_size_filtered', 'Max Size Filtered', -1),
    ('min_size_filtered', 'Min Size Filtered', -1),
    ('max_genes_filtered', 'Max Genes Per Genome Filtered', -1),
    ('priority_added_back', 'Priority Added Back', 1)
]
TAXONOMIC_RANKS = ["Domain", "Phylum", "Class", "Order", "Family", "Genus", "Species"]
NO_SHARED_TAXONOMY_LABEL = "No Shared Taxonomy"
DUPLICATE_TAXONOMY_RANKS = TAXONOMIC_RANKS + ["Genome", NO_SHARED_TAXONOMY_LABEL]


def get_domain_key(taxonomy):
    """Return the domain label for the provided taxonomy."""
    if taxonomy:
        domain = taxonomy[0]
        if domain in DOMAINS:
            return domain
    return None


def init_stats_bucket():
    """Initialize a stats bucket for a domain."""
    return {
        'initial': 0,
        'tree_filtered': 0,
        'resolution_filtered': 0,
        'duplicate_filtered': 0,
        'length_filtered': 0,
        'max_size_filtered': 0,
        'min_size_filtered': 0,
        'max_genes_filtered': 0,
        'priority_added_back': 0
    }


def update_stats(stats, key, domain_key, amount=1):
    """Update stats for all sequences and per-domain counts."""
    stats['all'][key] += amount
    if domain_key in DOMAINS:
        stats[domain_key][key] += amount



def is_seq_in_tree(gtdb_id):
    """
    Checks if a sequence is present in the phylogenetic tree.
        gtdb_id is the ID of the sequence (e.g. RS_GCF_028472785.1)
    """
    in_tree = gtdb_id in ALL_LEAF_NODE_GTDB_IDS
    if in_tree:
        LEAF_NODE_IDS_WITHOUT_SEQS.discard(gtdb_id)
    return in_tree

def parse_header(header):
    """Parse FASTA header to get sequence ID and taxonomy"""
    parts = header.split(' ', 1)
    gtdb_id = parts[0].strip('>').split('~')[0]
    taxonomy = parts[1].split(' [')[0].replace('d__', '').replace('p__', '').replace('c__', '').replace('o__', '').replace('f__', '').replace('g__', '').replace('s__', '').split(';')
    return gtdb_id, taxonomy


def plot_sequence_lengths(lengths, output_path):
    """Create a histogram plot of sequence lengths.
    
    Args:
        lengths: List of sequence lengths
        output_path: Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    plt.hist(lengths, bins=50, edgecolor='black')
    plt.title('Sequence Length Distribution')
    plt.xlabel('Sequence Length (bp)')
    plt.ylabel('Count')
    
    # Add vertical lines for min/max length filters
    no_lines = True
    if MIN_LENGTH > 150:
        plt.axvline(x=MIN_LENGTH, color='r', linestyle='--', label=f'Min Length ({MIN_LENGTH} bp)')
        no_lines = False
    if MAX_LENGTH < 3600:
        plt.axvline(x=MAX_LENGTH, color='r', linestyle='--', label=f'Max Length ({MAX_LENGTH} bp)')
        no_lines = False
    if not no_lines:
        plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def compute_percentile(sorted_vals, fraction):
    """Compute a percentile using simple linear interpolation."""
    if not sorted_vals:
        return 0
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    position = fraction * (len(sorted_vals) - 1)
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return sorted_vals[lower_index]
    weight = position - lower_index
    return sorted_vals[lower_index] + (sorted_vals[upper_index] - sorted_vals[lower_index]) * weight


def compute_length_stats(lengths):
    """Compute descriptive statistics for sequence lengths."""
    if not lengths:
        return {k: 0 for k in ['min', 'max', 'mean', 'median', 'std', 'p1', 'p10', 'p90', 'p99']}
    sorted_lengths = sorted(lengths)
    return {
        'min': sorted_lengths[0],
        'max': sorted_lengths[-1],
        'mean': statistics.mean(sorted_lengths),
        'median': statistics.median(sorted_lengths),
        'std': statistics.stdev(sorted_lengths) if len(sorted_lengths) > 1 else 0,
        'p1': compute_percentile(sorted_lengths, 0.01),
        'p10': compute_percentile(sorted_lengths, 0.10),
        'p90': compute_percentile(sorted_lengths, 0.90),
        'p99': compute_percentile(sorted_lengths, 0.99)
    }


def determine_shared_taxonomy_rank(taxonomies, genomes):
    """Determine the most specific taxonomy rank shared by all provided taxonomies."""
    if len(taxonomies) < 2:
        return None
    if len(genomes) == 1:
        return "Genome"
    # Traverse from most specific taxonomy rank (Species) up to Domain.
    for idx in reversed(range(len(TAXONOMIC_RANKS))):
        unique_labels = set()
        level_supported = True
        for taxonomy in taxonomies:
            if len(taxonomy) <= idx:
                level_supported = False
                break
            unique_labels.add(taxonomy[idx])
            if len(unique_labels) > 1:
                break
        if level_supported and len(unique_labels) == 1:
            return TAXONOMIC_RANKS[idx]
    return None


def summarize_duplicate_taxonomy(seq_tracker):
    """Summarize how closely duplicate sequences match taxonomically."""
    summary_stats = {
        rank: {'count': 0, 'max_len': 0}
        for rank in DUPLICATE_TAXONOMY_RANKS
    }
    total_duplicates = 0
    for metadata in seq_tracker.values():
        taxonomies = metadata['taxonomies']
        if len(taxonomies) < 2:
            continue
        duplicates_here = len(taxonomies) - 1
        total_duplicates += duplicates_here
        shared_rank = determine_shared_taxonomy_rank(taxonomies, metadata['genomes'])
        if shared_rank is None:
            shared_rank = NO_SHARED_TAXONOMY_LABEL
        entry = summary_stats[shared_rank]
        entry['count'] += duplicates_here
        entry['max_len'] = max(entry['max_len'], metadata.get('length', 0))
    return total_duplicates, summary_stats



if __name__ == "__main__":
    
    parent_directory = parent_dir(OUTPUT_FILE)
    # If the output directory exists and is not empty, ask the user if they want to continue
    if os.path.exists(parent_directory) and os.listdir(parent_directory):
        print(f"WARNING: Output directory {parent_directory} exists and is not empty.")
        input("Press Enter to continue...")
    else:
        # Create it
        os.makedirs(parent_directory, exist_ok=True)
    
    # Statistics tracking
    stats = {group: init_stats_bucket() for group, _ in STATS_GROUPS}
    
    # Read all sequences
    seen_sequences = set()
    sequence_taxonomy_tracker = defaultdict(lambda: {'taxonomies': [], 'genomes': set(), 'length': 0})
    taxa_counts = [defaultdict(list) for _ in range(7)]  # One dict per taxonomic level


    print(f"---\nStarting filtering sequences from {INPUT_FILE}.")
    print("Applying independent filters...")

    # Read all sequences
    # Filters 1-4 (all independent filters)
    for record in SeqIO.parse(INPUT_FILE, "fasta"):
        gtdb_id, taxonomy = parse_header(record.description)
        domain_key = get_domain_key(taxonomy)
        update_stats(stats, 'initial', domain_key)

        # Filter 1: Must be in tree
        if not is_seq_in_tree(gtdb_id):
            update_stats(stats, 'tree_filtered', domain_key)
            continue
            
        # Filter 2: Must have minimum taxonomic resolution
        if len(taxonomy) < MIN_TAXONOMIC_RESOLUTION:
            update_stats(stats, 'resolution_filtered', domain_key)
            continue
            
        # Filter 3: Remove duplicates (keep the first alphabetically)
        seq_str = str(record.seq)
        seq_metadata = sequence_taxonomy_tracker[seq_str]
        seq_length = len(seq_str)
        if seq_metadata['length'] == 0:
            seq_metadata['length'] = seq_length
        if seq_str in seen_sequences:
            seq_metadata['taxonomies'].append(taxonomy)
            seq_metadata['genomes'].add(gtdb_id)
            update_stats(stats, 'duplicate_filtered', domain_key)
            continue
        seen_sequences.add(seq_str)
        seq_metadata['taxonomies'].append(taxonomy)
        seq_metadata['genomes'].add(gtdb_id)

        # Filter 4: Check sequence length
        if seq_length < MIN_LENGTH or seq_length > MAX_LENGTH:
            update_stats(stats, 'length_filtered', domain_key)
            continue
        
        # Store sequence and update taxa counts
        seq_info = (record, gtdb_id, taxonomy, domain_key)
        for level in range(len(taxonomy)):
            taxa_counts[level][';'.join(taxonomy[:level+1])].append(seq_info)
    
    total_duplicate_sequences, duplicate_taxonomy_stats = summarize_duplicate_taxonomy(sequence_taxonomy_tracker)
    unresolved_duplicate_taxonomies = duplicate_taxonomy_stats[NO_SHARED_TAXONOMY_LABEL]['count']
    if unresolved_duplicate_taxonomies:
        print(
            f"Warning: {unresolved_duplicate_taxonomies} duplicate sequences lacked a shared taxonomy rank "
            f"and are reported under '{NO_SHARED_TAXONOMY_LABEL}'."
        )
    
    print("Applying max taxa size filters...")
    # Filter 5: Apply max taxa size filters first
    max_filtered_sequences = []
    processed_sequences = set()
    genome_seq_counts = defaultdict(int)
    for level in range(len(MAX_SEQS_PER_TAXON)):
        for taxon, seqs in taxa_counts[level].items():
            # Randomly sample if above max size
            keep_indices = None
            max_allowed = MAX_SEQS_PER_TAXON[level]
            if max_allowed is not None and len(seqs) > max_allowed:
                sample_size = int(max_allowed)
                keep_indices = set(random.sample(range(len(seqs)), sample_size))
            for idx, seq in enumerate(seqs):
                record, gtdb_id, taxonomy, domain_key = seq
                seq_key = record.description
                if seq_key in processed_sequences:
                    continue
                if keep_indices is not None and idx not in keep_indices:
                    processed_sequences.add(seq_key)
                    update_stats(stats, 'max_size_filtered', domain_key)
                    continue
                processed_sequences.add(seq_key)
                if MAX_GENES_PER_GENOME is not None:
                    if genome_seq_counts[gtdb_id] >= MAX_GENES_PER_GENOME:
                        update_stats(stats, 'max_genes_filtered', domain_key)
                        continue
                    genome_seq_counts[gtdb_id] += 1
                max_filtered_sequences.append(seq)

    # Update taxa counts after max size filtering
    taxa_counts = [defaultdict(list) for _ in range(7)]
    for seq in max_filtered_sequences:
        _, _, taxonomy, _ = seq
        for level in range(len(taxonomy)):
            taxa_counts[level][';'.join(taxonomy[:level+1])].append(seq)
    

    print("Applying min taxa size filters...")
    # Filter 6: Apply min taxa size filters
    filtered_sequences = []
    processed_sequences = set()
    for level in range(len(MIN_SEQS_PER_TAXON)):
        for taxon, seqs in taxa_counts[level].items():
            if len(seqs) < MIN_SEQS_PER_TAXON[level]:
                for seq in seqs:
                    record, _, _, domain_key = seq
                    seq_key = record.description
                    if seq_key in processed_sequences:
                        continue
                    processed_sequences.add(seq_key)
                    update_stats(stats, 'min_size_filtered', domain_key)
                continue
            
            for seq in seqs:
                record, _, _, _ = seq
                seq_key = record.description
                if seq_key in processed_sequences:
                    continue
                processed_sequences.add(seq_key)
                filtered_sequences.append(seq)
    print("Done filtering sequences.")

    # Ensure priority sequences are present after filtering steps 5 & 6
    final_seqs = {seq[0].description for seq in filtered_sequences}
    for record in SeqIO.parse(INPUT_FILE, "fasta"):
        if any(p_id in record.description for p_id in PRIORITY_SEQ_IDS):
            if record.description not in final_seqs:
                gtdb_id, taxonomy = parse_header(record.description)
                domain_key = get_domain_key(taxonomy)
                filtered_sequences.append((record, gtdb_id, taxonomy, domain_key))
                final_seqs.add(record.description)
                update_stats(stats, 'priority_added_back', domain_key)
                print(f"Added priority sequence containing ID: {gtdb_id}")

    # Calculate sequence length statistics
    lengths_by_domain = {group: [] for group, _ in STATS_GROUPS}
    if filtered_sequences:
        for record, _, taxonomy, domain_key in filtered_sequences:
            length = len(record.seq)
            lengths_by_domain['all'].append(length)
            if domain_key in DOMAINS:
                lengths_by_domain[domain_key].append(length)
        length_stats = {group: compute_length_stats(lengths_by_domain[group]) for group, _ in STATS_GROUPS}
        # Create length distribution plot for combined sequences
        plot_sequence_lengths(lengths_by_domain['all'], SEQ_LENGTHS_PLOT)
    else:
        length_stats = {group: compute_length_stats([]) for group, _ in STATS_GROUPS}
    

    # Write filtered sequences
    print(f"---\nWriting filtered sequences to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w') as f:
        for record, _, _, _ in sorted(filtered_sequences, key=lambda x: x[1]):
            f.write(f">{record.description}\n{str(record.seq)}\n")
    
    # Write about file
    print(f"Writing filtering information to {ABOUT_FILE}...")
    with open(ABOUT_FILE, 'w') as f:
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"Report generated on: {now}\n\n")
        f.write("GTDB 16S rRNA Database Filtering Summary\n")
        f.write("=====================================\n\n")
        f.write(f"Input file: {INPUT_FILE}\n")
        f.write(f"Output file: {OUTPUT_FILE}\n\n")
        
        for group, label in STATS_GROUPS:
            domain_stats = stats[group]
            f.write(f"Filtering Statistics ({label}):\n")
            f.write(f"Initial sequences: {domain_stats['initial']}\n")
            remaining = domain_stats['initial']
            for key, title, effect in FILTERING_SUMMARY_ORDER:
                count = domain_stats[key]
                if effect < 0:
                    remaining -= count
                    action = "removed"
                else:
                    remaining += count
                    action = "added"
                percent = (count / domain_stats['initial'] * 100) if domain_stats['initial'] else 0
                f.write(f"{title}: {count} {action} ({percent:.3f}%), {remaining} remaining\n")
            f.write("\n")
        
        for group, label in STATS_GROUPS:
            stats_block = length_stats[group]
            f.write(f"Sequence Length Statistics ({label}):\n")
            f.write(f"Minimum: {stats_block['min']} bp\n")
            f.write(f"1st percentile: {stats_block['p1']} bp\n")
            f.write(f"10th percentile: {stats_block['p10']} bp\n")
            f.write(f"Mean: {stats_block['mean']:.3f} bp\n")
            f.write(f"Median: {stats_block['median']} bp\n")
            f.write(f"Standard deviation: {stats_block['std']:.3f} bp\n")
            f.write(f"90th percentile: {stats_block['p90']} bp\n")
            f.write(f"99th percentile: {stats_block['p99']} bp\n")
            f.write(f"Maximum: {stats_block['max']} bp\n\n")

        for group, label in STATS_GROUPS:
            final_count = len(lengths_by_domain[group])
            initial_sequences = stats[group]['initial']
            percentage = (final_count / initial_sequences * 100) if initial_sequences else 0
            f.write(f"Final Results ({label}):\n")
            f.write(f"Final sequences: {final_count}\n")
            f.write(f"Percentage of initial: {percentage:.3f}%\n\n")

        total_leaf_nodes = len(ALL_LEAF_NODE_GTDB_IDS)
        unmatched_leaf_nodes = len(LEAF_NODE_IDS_WITHOUT_SEQS)
        f.write("Tree Leaf Node Coverage:\n")
        if unmatched_leaf_nodes == 0:
            f.write(f"All {total_leaf_nodes} leaf nodes from the reference trees were matched by at least one sequence in the input FASTA.\n\n")
        else:
            f.write(f"Leaf nodes without matching sequences: {unmatched_leaf_nodes} / {total_leaf_nodes}\n")
            sample_count = min(10, unmatched_leaf_nodes)
            if sample_count > 0:
                sample_ids = random.sample(list(LEAF_NODE_IDS_WITHOUT_SEQS), sample_count)
                f.write("Example unmatched leaf nodes:\n")
                for node_id in sample_ids:
                    f.write(f"- {node_id}\n")
            f.write("\n")
        f.write("Shared Taxonomy of Duplicate Sequences:\n")
        f.write(f"Total Num Duplicate Sequences: {total_duplicate_sequences}\n")
        for rank in DUPLICATE_TAXONOMY_RANKS:
            rank_stats = duplicate_taxonomy_stats[rank]
            count = rank_stats['count']
            max_len = rank_stats['max_len']
            if count > 0 and max_len > 0:
                f.write(f"{rank}: {count} (Max Seq Length = {max_len})\n")
            else:
                f.write(f"{rank}: {count}\n")
        f.write("\n")

    # Print done message
    print(f"Done filtering! \n{len(filtered_sequences)} / {stats['all']['initial']} sequences remaining.")
