"""
Globals Configuration Module

This module serves as a centralised access to global constants shared across the Micro16S training system.
It initializes all dataset-related variables to `None`, acting as a placeholder until the dataset is loaded.

The primary function `load_micro16s_dataset` in the `micro16s_dataset_loader` module
is responsible for populating these variables with the actual dataset contents (sequence
representations, taxonomic information, index splits, etc.). 

Using this shared module allows different parts of the codebase during training to conveniently 
access the same, consistently loaded dataset information without needing to pass several 
arguments between functions and modules.

The `DATASET_IS_LOADED` flag provides a safeguard, ensuring that other modules attempt
to access these constant variables only after the dataset has been successfully loaded.
"""

# Flag to indicate if the dataset has been loaded
# This must be checked when any globals are accessed
DATASET_IS_LOADED = False

# --- Dataset and File Paths ---
MODELS_DIR = None
REDVALS_DIR = None
ENCODED_SEQS_DIR = None # Root directory of the database
DATASET_SPLIT_DIR = None  # Directory containing the specific dataset split being used
ALL_3BIT_SEQ_REPS_PATH = None # Path to the .npy file containing all 3-bit sequence representations
HAS_EXCLUDED_SET = False  # Indicates whether the current dataset split has an excluded partition

# --- Sequence Indices ---
# Numpy arrays containing the integer indices for sequences in each dataset split
EXCLUDED_TAXA_INDICES = None
TESTING_INDICES = None
TRAINING_INDICES = None

# --- Sequence Representations ---
# Numpy arrays holding the 3-bit encoded sequence representations
ALL_3BIT_SEQ_REPS = None      # All sequences [N_REGIONS, N_SEQS, MAX_IMPORTED_SEQ_LEN, 3]
EXCLUDED_3BIT_SEQ_REPS = None # Slice for excluded sequences
TESTING_3BIT_SEQ_REPS = None  # Slice for testing sequences
TRAINING_3BIT_SEQ_REPS = None # Slice for training sequences
TRAINING_3BIT_SEQ_REPS_TRANSPOSED = None # Slice for training sequences, transposed to [N_SEQS, N_REGIONS, MAX_IMPORTED_SEQ_LEN, 3]

# --- K-mer Sequence Representations ---
# Dictionaries mapping k (int) to numpy arrays holding the k-mer encoded sequence representations
ALL_KMER_SEQ_REPS = None      # All sequences for each k
EXCLUDED_KMER_SEQ_REPS = None # Slice for excluded sequences for each k
TESTING_KMER_SEQ_REPS = None  # Slice for testing sequences for each k
TRAINING_KMER_SEQ_REPS = None # Slice for training sequences for each k

# --- Training Set Taxonomy ---
# Data structures describing the taxonomy for the training set
TRAIN_FULL_TAX_LABEL_FROM_SEQ_ID_DICT = None        # Maps sequence ID -> full taxonomic label list
TRAIN_LIST_OF_SEQ_INDICES_IN_TAXON_AT_RANK_DICT = None # Maps (rank, taxon) -> list of sequence indices
TRAIN_LIST_OF_TAXON_LABELS_IN_TAXON_AT_RANK_DICT = None# Maps (rank, taxon) -> list of child taxa
TRAIN_LIST_OF_TAXON_LABELS_AT_RANK_DICT = None       # Maps rank -> list of all taxa at that rank
TRAIN_NESTED_LIST_OF_SEQ_INDICES = None           # Nested list representing taxonomic hierarchy with seq indices
TRAIN_NESTED_DICTS_OF_TAXA = None                 # Nested dict representing taxonomic hierarchy
TRAIN_TAXON_LABEL_TO_TAXON_ID = None              # Maps taxon label (str) -> taxon ID (int) ... these are the integers used in the Labels Arrays
TRAIN_TAXON_ID_TO_TAXON_LABEL = None              # Maps taxon ID (int) -> taxon label (str)

# --- Testing Set Taxonomy ---
# Data structures describing the taxonomy for the testing set
TEST_FULL_TAX_LABEL_FROM_SEQ_ID_DICT = None
TEST_LIST_OF_SEQ_INDICES_IN_TAXON_AT_RANK_DICT = None
TEST_LIST_OF_TAXON_LABELS_IN_TAXON_AT_RANK_DICT = None
TEST_LIST_OF_TAXON_LABELS_AT_RANK_DICT = None
TEST_NESTED_LIST_OF_SEQ_INDICES = None
TEST_NESTED_DICTS_OF_TAXA = None
TEST_TAXON_LABEL_TO_TAXON_ID = None
TEST_TAXON_ID_TO_TAXON_LABEL = None

# --- Excluded Set Taxonomy ---
# Data structures describing the taxonomy for the excluded set
EXCLUDED_FULL_TAX_LABEL_FROM_SEQ_ID_DICT = None
EXCLUDED_LIST_OF_SEQ_INDICES_IN_TAXON_AT_RANK_DICT = None
EXCLUDED_LIST_OF_TAXON_LABELS_IN_TAXON_AT_RANK_DICT = None
EXCLUDED_LIST_OF_TAXON_LABELS_AT_RANK_DICT = None
EXCLUDED_NESTED_LIST_OF_SEQ_INDICES = None
EXCLUDED_NESTED_DICTS_OF_TAXA = None
EXCLUDED_TAXON_LABEL_TO_TAXON_ID = None
EXCLUDED_TAXON_ID_TO_TAXON_LABEL = None

# --- Combined Full Taxonomic Labels ---
# Lists containing the full taxonomic label (list of 7 strings) for each sequence in the respective set
TRAIN_FULL_TAX_LABELS = None
TEST_FULL_TAX_LABELS = None
EXCLUDED_FULL_TAX_LABELS = None

# --- Training Set Label Arrays ---
# NumPy arrays containing precomputed label information for the training set
TRAIN_SEQ_TAXON_IDS = None         # Sequence taxon IDs matrix (n_seqs, 7) with dtype int32
TRAIN_PAIRWISE_RANKS = None        # Pairwise taxonomic ranks matrix (n_seqs, n_seqs) with dtype int8
TRAIN_TAXON_COUNTS_PER_SEQ = None  # Per-seq taxon sizes (n_seqs, 7): number of seqs in same taxon at each rank
TRAIN_TAXON_BASELINE_COUNT_PER_RANK = None  # Per-rank baseline taxon size (7,), computed per-taxon (not per-seq)
TRAIN_BACTERIA_INDICES = None      # Precomputed indices of bacteria sequences (1D array)
TRAIN_ARCHAEA_INDICES = None       # Precomputed indices of archaea sequences (1D array)
TRAIN_PAIRWISE_POS_MASKS = None    # Pairwise positive masks (7, n_seqs, n_seqs) with dtype bool
TRAIN_PAIRWISE_NEG_MASKS = None    # Pairwise negative masks (7, n_seqs, n_seqs) with dtype bool
TRAIN_PAIRWISE_MRCA_TAXON_IDS = None # Pairwise MRCA taxon IDs matrix (n_seqs, n_seqs) with dtype int32
TRAIN_PAIRWISE_DISTANCES = None    # Pairwise phylogenetic distances matrix (n_seqs, n_seqs) with dtype float32
ADJUSTED_TRAIN_PAIRWISE_DISTANCES = None # Domain-adjusted pairwise distances matrix (n_seqs, n_seqs) with dtype float32
TRAIN_DISTANCES_LOOKUP_ARRAY = None # 1D lookup array mapping taxon IDs to distances (max_taxon_id + 1,) with dtype float32
TRAIN_DISTANCE_BETWEEN_DOMAINS = None # Distance value for different domains (scalar float32)

# --- Testing Set Label Arrays ---
# NumPy arrays containing precomputed label information for the testing set
TEST_SEQ_TAXON_IDS = None
TEST_PAIRWISE_RANKS = None
TEST_PAIRWISE_POS_MASKS = None
TEST_PAIRWISE_NEG_MASKS = None
TEST_PAIRWISE_MRCA_TAXON_IDS = None
TEST_PAIRWISE_DISTANCES = None
TEST_DISTANCES_LOOKUP_ARRAY = None
TEST_DISTANCE_BETWEEN_DOMAINS = None

# --- Excluded Set Label Arrays ---
# NumPy arrays containing precomputed label information for the excluded set
EXCLUDED_SEQ_TAXON_IDS = None
EXCLUDED_PAIRWISE_RANKS = None
EXCLUDED_PAIRWISE_POS_MASKS = None
EXCLUDED_PAIRWISE_NEG_MASKS = None
EXCLUDED_PAIRWISE_MRCA_TAXON_IDS = None
EXCLUDED_PAIRWISE_DISTANCES = None
EXCLUDED_DISTANCES_LOOKUP_ARRAY = None
EXCLUDED_DISTANCE_BETWEEN_DOMAINS = None

# --- Phylogenetic Distances ---
# Object handling Relative Evolutionary Divergence (RED) values based on phylogenetic trees
RED_TREES = None

# --- Ordination (PCoA/UMAP) ---
ORDINATION_LABEL_COLOR_CACHE = {}  # Cache for Ordination visualisation: maps rank -> (unique_labels, label_to_idx, ListedColormap)

# --- K-mer Representations ---
MEAN_PER_KMERS = None
STD_PER_KMERS = None

# --- Quick Test ---
# Cache for k-mer quick test results
KMER_QT_CACHE = None 

# --- Per-Rank Mining EMA State ---
# These are runtime state variables (NOT config) - initialized during trainer startup.
# EMA buffers for per-rank hardness metrics, used to allocate batch budgets across ranks.
# Seeded with 100 fabricated 1.0 observations during trainer initialization.
# Shape: (9,) for pairs (domain->subsequence), (6,) for triplets (domain->genus)
PAIR_MINING_EMA_HARDNESS = None    # np.ndarray of shape (9,) or None if not initialized
TRIPLET_MINING_EMA_HARDNESS = None # np.ndarray of shape (6,) or None if not initialized
MINED_CACHE = None # Cache for mined pairs/triplets



def init_mining_ema_buffers():
    """
    Initialize the per-rank EMA hardness buffers for pair and triplet mining.
    
    This should be called once during trainer initialization, after set_train_config_as_globals().
    
    The buffers are seeded to 1.0 (equivalent to 100 fabricated observations of 1.0).
    This ensures equal initial budget allocation across all enabled ranks.
    
    Disabled ranks (from PAIR_RANKS / TRIPLET_RANKS) are set to 0.0 so they receive zero allocation.
    """
    import numpy as np
    global PAIR_MINING_EMA_HARDNESS, TRIPLET_MINING_EMA_HARDNESS
    
    # Initialize pair EMA buffer (9 ranks: domain->subsequence)
    # Seed to 1.0 (equivalent to 100 fabricated observations of 1.0)
    PAIR_MINING_EMA_HARDNESS = np.ones(9, dtype=np.float64)
    
    # Zero out disabled pair ranks
    if PAIR_RANKS is not None:
        for rank_idx, enabled in enumerate(PAIR_RANKS):
            if not enabled:
                PAIR_MINING_EMA_HARDNESS[rank_idx] = 0.0
    
    # Initialize triplet EMA buffer (6 ranks: domain->genus)
    TRIPLET_MINING_EMA_HARDNESS = np.ones(6, dtype=np.float64)
    
    # Zero out disabled triplet ranks
    if TRIPLET_RANKS is not None:
        for rank_idx, enabled in enumerate(TRIPLET_RANKS):
            if not enabled:
                TRIPLET_MINING_EMA_HARDNESS[rank_idx] = 0.0


# --- Training Configuration ---
DESCRIPTION = None
MODELS_DIR = None
DATASET_SPLIT_DIR = None
EMBED_DIMS = None
MAX_MODEL_SEQ_LEN = None
MAX_IMPORTED_SEQ_LEN = None
POOLING_TYPE = None
USE_CONV_STEM = None
CONV_STEM_KERNEL_SIZE = None
CONV_STEM_RESIDUAL = None
CONV_STEM_INIT_SCALE = None
USE_CONVFORMER = None
CONFORMER_KERNEL_SIZE = None
DROPOUT_PROP = None
ATT_DROPOUT_PROP = None
D_MODEL = None
N_LAYERS = None
N_HEAD = None
D_FF = None
PAIR_LOSS_WEIGHT = None
TRIPLET_LOSS_WEIGHT = None
USE_UNCERTAINTY_WEIGHTING = None
UNCERTAINTY_LEARNING_RATE = None
IS_USING_UNCERTAINTY_WEIGHTING = None  # Runtime flag: True if uncertainty weighting is actually active
BACTERIA_DISTANCE_FACTOR = None
ARCHEA_DISTANCE_FACTOR = None
DISTANCE_GAMMA_CORRECTION_GAMMA = None
TRIPLET_MARGIN_EPSILON = None
MANUAL_TRIPLET_MARGINS_PER_RANK = None
MANUAL_TRIPLET_MARGINS = None
NUM_BATCHES_PER_MINING_RUN = None
SHUFFLE_TRIPLETS_AND_PAIRS_AFTER_MINING = None
MINING_TRAIN_SEQUENCES_SUB_SAMPLE_BACTERIA = None
MINING_TRAIN_SEQUENCES_SUB_SAMPLE_ARCHAEA = None
TAXON_SIZE_MINING_BIAS_BASELINE_STAT = None
TAXON_SIZE_MINING_BIAS_BASELINE_PERCENTILE = None
USE_REPRESENTATIVE_TAXON_SIZE_BALANCING = None
PAIR_REPRESENTATIVE_TAXON_SIZE_BALANCE_LAMBDA = None
TRIPLET_REPRESENTATIVE_TAXON_SIZE_BALANCE_LAMBDA = None
REPRESENTATIVE_TAXON_SIZE_BALANCE_EPS = None
REPRESENTATIVE_TAXON_SIZE_BALANCE_WEIGHT_CLIP = None
USE_TAXON_SIZE_MINING_BIAS = None
TAXON_SIZE_MINING_BIAS_ALPHAS_PER_RANK = None
TAXON_SIZE_MINING_BIAS_MIN_KEEP_PER_RANK = None
RELATIVE_ERROR_EPSILONS_PAIR_LOSS = None
RELATIVE_ERROR_EPSILONS_PAIR_MINING = None
PAIR_RELATIVE_LOSS_CAP = None
RELATIVE_ERROR_EPSILONS_TRIPLET_LOSS = None
TRIPLET_RELATIVE_LOSS_CAP = None
PAIR_RANKS = None
INTRODUCE_RANK_AT_BATCHES = None
DOWNSAMPLE_PAIRS_AT_RANK = None
PAIR_MINING_BUCKETS = None
# Per-rank sign bias for pair mining hardness (length 9, domain->subseq)
PAIR_SIGN_BIAS_BETA_PER_RANK = None
# Per-rank representative set sizes (length 9 for pairs, 6 for triplets)
PAIR_MINING_REPRESENTATIVE_SET_SIZES = None
TRIPLET_MINING_REPRESENTATIVE_SET_SIZES = None
# Per-rank EMA parameters for hardness-based batch allocation
PAIR_MINING_EMA_ALPHA = None
TRIPLET_MINING_EMA_ALPHA = None
PAIR_MINING_EMA_WEIGHT_EXPONENT = None
TRIPLET_MINING_EMA_WEIGHT_EXPONENT = None
TRIPLET_EMA_HARD_WEIGHT = None
TRIPLET_EMA_MODERATE_WEIGHT = None
PAIR_EMA_MEAN_WEIGHT = None
PAIR_EMA_QUARTILES_WEIGHT = None
PAIR_PER_RANK_BATCH_PROPORTION_MAX = None
TRIPLET_PER_RANK_BATCH_PROPORTION_MAX = None
PAIR_PER_RANK_BATCH_PROPORTION_MIN = None
TRIPLET_PER_RANK_BATCH_PROPORTION_MIN = None
PROPORTION_IDENTICAL_SEQUENCES_TO_GENERATE = None
SUBSEQUENCES_ALWAYS_CROSS_REGION = None
SUB_SEQUENCE_TRUE_DISTANCE = None
TRIPLET_RANKS = None
TRIPLET_MINING_BUCKETS = None
FILTER_ZERO_LOSS_TRIPLETS = None
MIN_TRIPLETS_AFTER_FILTER_PER_RANK_BATCH_SIZE_FACTOR = None
N_PAIRS_PER_BATCH = None
N_TRIPLETS_PER_BATCH = None
USE_FULL_SEQS = None
USE_SUB_SEQS = None
MAX_NUM_SUBSEQS = None
MUTATION_RATE = None
MIN_TRUNC_START = None
MAX_TRUNC_START = None
MIN_TRUNC_END = None
MAX_TRUNC_END = None
PROP_SHIFT_SEQS = None
PROP_TRUNC = None
LEARNING_RATE = None
WEIGHT_DECAY = None
NUM_BATCHES = None
START_FROM_CKPT = None
NUM_MICRO_BATCHES_PER_BATCH = None
PIN_MEMORY_FOR_MINING = None
USE_FLOAT_16_FOR_MINING_COSINE_DISTANCES = None
LR_SCHEDULER_TYPE = None
LR_SCHEDULER_KWARGS = None
SAVE_EVERY_N_BATCHES = None
RECORD_LOSS_EVERY_N_BATCHES = None
PLOT_LOSS_EVERY_N_BATCHES = None
PLOT_TOTAL_LOSS = None
PLOT_TRIPLET_LOSS = None
PLOT_PAIR_LOSS = None
PLOT_UNCERTAINTY_WEIGHTING = None
SECOND_LOSS_PLOT_AFTER_N_BATCHES = None
FREE_LARGE_INTERMEDIATES_WHEN_MINING = None
SECOND_TRIPLET_SATISFACTION_PLOT_AFTER_N_BATCHES = None
SECOND_PAIR_ERROR_METRICS_PLOT_AFTER_N_BATCHES = None
SECOND_TRIPLET_ERROR_METRICS_PLOT_AFTER_N_BATCHES = None
DO_QT = None
QT_BATCH_SIZE = None
MINING_BATCH_SIZE = None
PAIR_MINING_WARMUP_UNIFORM_DURATION = None
PAIR_MINING_WARMUP_TRANSITION_DURATION = None
TRIPLET_MINING_WARMUP_UNIFORM_DURATION = None
TRIPLET_MINING_WARMUP_TRANSITION_DURATION = None
USE_TRIPLET_MINING_POSITIVE_SELECTION_BIAS = None
TRIPLET_MINING_POSITIVE_SELECTION_BIAS_BETA = None
TRIPLET_MINING_POSITIVE_SELECTION_BIAS_EPS = None
USE_TRIPLET_MINING_NEGATIVE_SELECTION_BIAS = None
TRIPLET_MINING_NEGATIVE_SELECTION_BIAS_BETA = None
TRIPLET_MINING_NEGATIVE_SELECTION_BIAS_EPS = None
TRIPLET_NEGATIVES_PER_AP = None
QT_EVERY_N_BATCHES = None
QT_BEFORE_FIRST_BATCH = None
QT_KMER_K_VALUES = None
QT_GET_KMER_RESULTS = None
QT_PCOA_RANKS = None
QT_PCOA_EVERY_N_BATCHES = None
QT_UMAP_RANKS = None
QT_UMAP_EVERY_N_BATCHES = None
QT_CLUSTERING_RANKS = None
QT_RANK_LEVEL_CLUSTERING_METRICS_AT_RANKS = None
QT_MAX_TAXA_PER_RANK_LEVEL_CLUSTERING = None
RANK_LEVEL_CLUSTERING_PLOTS_EVERY_N_BATCHES = None
QT_CLASSIFICATION_RANKS = None
QT_RECORD_MISS_CLASSIFICATION_SEQUENCES_AT_RANKS = None
QT_MAX_N_REGIONS = None
QT_PROP_SHIFT_SEQS = None
QT_MUTATION_RATE = None
QT_PROP_TRUNC = None
QT_MIN_TRUNC_START = None
QT_MAX_TRUNC_START = None
QT_MIN_TRUNC_END = None
QT_MAX_TRUNC_END = None
VERBOSE_TRAINING_TIMING = None
VERBOSE_TRIPLET_LOSS = None
VERBOSE_PAIR_LOSS = None
VERBOSE_PAIR_MINING = None
VERBOSE_TRIPLET_MINING = None
VERBOSE_EVERY_N_BATCHES = None
VERBOSE_MINING_TIMING = None
VERBOSE_UNCERTAINTY_WEIGHTING = None
LOG_TRIPLET_LOSS = None
LOG_PAIR_LOSS = None
LOG_TRIPLET_MINING = None
LOG_PAIR_MINING = None
LOG_MINING = None
LOG_MINING_BUCKET_THRESHOLDS = None
CROSS_REGION_LOGGING_MAX_LINES_PER_TABLE = None
LOG_MINING_PHYLA_COUNTS = None
MINING_PHYLA_LOG_MAX_LINES_PER_TABLE = None
LOG_UNCERTAINTY_WEIGHTING = None
LOG_MINING_ARC_BAC_COUNTS = None
LOG_EVERY_N_BATCHES = None
PLOT_TRIPLET_SATISFACTION_EVERY_N_BATCHES = None
PLOT_PAIR_DISTANCES_EVERY_N_BATCHES = None
PLOT_PAIR_ERROR_METRICS_EVERY_N_BATCHES = None
PLOT_TRIPLET_ERROR_METRICS_EVERY_N_BATCHES = None
PRINT_BATCH_NUM_EVERY_N_BATCHES = None
SEQ_3BIT_REPRESENTATION_DICT = None
QT_VERBOSE = None
QT_PRINT_RESULTS = None
QT_TESTS_TODO = None
RED_DISTANCE_BETWEEN_DOMAINS = None
SEQLESS_MODE = None
TAXA_CAP_MAX_NUM_TAXA = None
TAXA_CAP_RANK = None
TAXA_CAP_VERBOSE = None
