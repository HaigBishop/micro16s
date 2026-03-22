import globals_config as gb
from datetime import datetime
import json as json_lib
import numpy as np


# ================================================
# Training Configuration 
# ================================================

class TrainingConfig:

    # Description ------------------
    # DESCRIPTION = "Final Run: Validation Model"
    DESCRIPTION = "Final Run: Application Model"

    # Paths ------------------
    # Local machine
    MODELS_DIR = "/home/haig/Repos/micro16s/models/"
    # DATASET_SPLIT_DIR = "/mnt/secondary/micro16s_dbs/16s_databases/m16s_db_004/seqs/split_001/"
    DATASET_SPLIT_DIR = "/mnt/secondary/micro16s_dbs/16s_databases/m16s_db_004/seqs/split_002/"
    REDVALS_DIR = "/home/haig/Repos/micro16s/redvals/"

    # Runpod instance
    # MODELS_DIR = "/workspace/micro16s/models/"
    # DATASET_SPLIT_DIR = "/workspace/micro16s_dbs/16s_databases/m16s_db_004/seqs/split_001/"
    # REDVALS_DIR = "/workspace/micro16s/redvals/"

    # Taxa Cap ------------------
    # Optionally limit the number of taxa included in the dataset at a given rank.
    # Sequences belonging to taxa outside the top N (by training-set count) are filtered
    # out from the training, testing, and excluded sets at data-loading time.
    TAXA_CAP_MAX_NUM_TAXA = None  # None = no cap, or a positive integer
    TAXA_CAP_RANK = 1             # 0=domain, 1=phylum, 2=class, 3=order, 4=family, 5=genus
    TAXA_CAP_VERBOSE = True

    # Sequenceless Mode ------------------
    # When True, the model learns a lookup table of embeddings for the training set
    # rather than learning how to embed DNA sequences. This removes sequence processing
    # challenges and focuses purely on mining algorithm, loss functions, and embedding space.
    # Note: Should increase batch size by a factor of over ~100, increase training sequence 
    #       subsampling proportion and increase learning rate.
    SEQLESS_MODE = False


    # Start From Checkpoint ------------------
    # False OR .pth file path
    START_FROM_CKPT = False #"/home/haig/Repos/micro16s/models/m16s_042/ckpts/m16s_042_6000_batches.pth"


    # Model parameters ------------------
    # Higher sequence length cap applied when importing sequences before augmentation
    MAX_IMPORTED_SEQ_LEN = 600
    # Sequence length cap applied after truncation every batch
    # This is the sequence length that will be used for the model input
    MAX_MODEL_SEQ_LEN = 600
    # Conv stem (optional local mixing before first attention layer)
    USE_CONV_STEM = True
    CONV_STEM_KERNEL_SIZE = 7
    CONV_STEM_RESIDUAL = True
    CONV_STEM_INIT_SCALE = 0.1
    # Transformer model shape
    D_MODEL = 96 # Must be divisible by N_HEAD*2
    N_LAYERS = 4
    N_HEAD = 4
    D_FF = 512
    # Convformer (optional depthwise conv sublayer in each transformer layer)
    USE_CONVFORMER = True
    CONFORMER_KERNEL_SIZE = 15
    # Sequence pooling
    POOLING_TYPE = 'attention' # 'mean' or 'attention'
    # Embedding size
    EMBED_DIMS = 256
    # Dropout
    DROPOUT_PROP = 0.0       # Global dropout probability for residual/FFN/conv paths
    ATT_DROPOUT_PROP = 0.0   # Dropout probability for attention weights only. Attention dropout can hurt small models sometimes, so treat it separately

    # Model training ---
    LEARNING_RATE = 0.00025
    WEIGHT_DECAY = 1e-3
    NUM_BATCHES = 30000
    
    # Batch size ---
    N_PAIRS_PER_BATCH = 60*64
    N_TRIPLETS_PER_BATCH = 120*64
    NUM_MICRO_BATCHES_PER_BATCH = 64

    # Uncertainty Weighting ---
    # When training with both pair loss (regression) and triplet loss (ranking), their magnitudes
    # can diverge during training. Hard-mined triplet loss often stays numerically large while
    # pair loss converges, causing triplet gradients to dominate updates.
    # Uncertainty weighting learns task-specific log-variance parameters (log_vars) to dynamically
    # scale each loss, balancing their contributions throughout training.
    # Note: IS_USING_UNCERTAINTY_WEIGHTING (global) is True only when this is True AND both
    # pair and triplet losses are actively used (weights > 0, batch sizes > 0).
    USE_UNCERTAINTY_WEIGHTING = False
    UNCERTAINTY_LEARNING_RATE = 0.002

    # Loss parameters ---
    PAIR_LOSS_WEIGHT = 1.0
    TRIPLET_LOSS_WEIGHT = 1.0

    # LR Scheduler ---
    # 0. No LR Scheduler
    LR_SCHEDULER_TYPE = None
    LR_SCHEDULER_KWARGS = None

    # 1. CosineAnnealingWarmRestarts
    # Good for escaping local minima by periodically resetting LR.
    # LR_SCHEDULER_TYPE = 'CosineAnnealingWarmRestarts'
    # LR_SCHEDULER_KWARGS = {
    #     'T_0': 1000,      # Batches until the first restart.
    #     'T_mult': 2,      # Multiplies T_0 after every restart (2 = cycle length doubles).
    #     'eta_min': 1e-6,  # The floor; LR won't go below this.
    # }

    # 2. OneCycleLR
    # Aggressive warmup then decay. often yields faster convergence ('Super-convergence').
    # LR_SCHEDULER_TYPE = 'OneCycleLR'
    # LR_SCHEDULER_KWARGS = {
    #     'max_lr': 0.004,   # The peak LR (usually 10x your starting LR).
    #     'pct_start': 0.3,  # % of training spent increasing LR (warmup).
    #     'div_factor': 25,  # initial_lr = max_lr / div_factor.
    # }

    # 3. StepLR
    # Classic approach. Drops LR by a factor at fixed intervals.
    # LR_SCHEDULER_TYPE = 'StepLR'
    # LR_SCHEDULER_KWARGS = {
    #     'step_size': 2500, # Drop LR every N batches.
    #     'gamma': 0.1,      # Multiplicative factor (e.g., 0.1 drops it by 10x).
    # }

    # 4. LinearWarmupToLearningRate
    # Start at start_lr, then linearly ramp to LEARNING_RATE over warmup_batches.
    # LR_SCHEDULER_TYPE = 'LinearWarmupToLearningRate'
    # LR_SCHEDULER_KWARGS = {
    #     'start_lr': 1e-6,       # Batch 1 learning rate.
    #     'warmup_batches': 150,  # Number of scheduler steps to reach LEARNING_RATE.
    # }


    # Pairs and triplets ------------------

    # Distances & Margins---
    # The RED distance between any pair of sequences from different domains (Archaea and Bacteria)
    RED_DISTANCE_BETWEEN_DOMAINS = 1.6
    # Distance factors for bacteria and archaea
    BACTERIA_DISTANCE_FACTOR = 0.75
    ARCHEA_DISTANCE_FACTOR = 0.75
    # Distance gamma correction factor
    #  - Tuning parameter for non-linear compression of the embedding space [0,2]
    #  - Values 0.1-1.0 are recommended
    #  - Applies this function to all distances:   `γ * distance ** log2(2 / γ)`
    #  - 1.0 has no effect
    DISTANCE_GAMMA_CORRECTION_GAMMA = 0.6
    # Triplet margin epsilon
    TRIPLET_MARGIN_EPSILON = 1.0
    # Use manual triplet margins?
    MANUAL_TRIPLET_MARGINS = False
    # Margins for the triplet loss
    MANUAL_TRIPLET_MARGINS_PER_RANK = {
        0: 0.5, # domain
        1: 0.5, # phylum
        2: 0.5, # class
        3: 0.5, # order
        4: 0.5, # family
        5: 0.5, # genus
        6: 0.5, # species
    }


    # Error Metrics ---
    # Pair mining sign-bias beta per rank (used in mining hardness only, not in pair loss).
    # Hardness transform:
    #   hardness = rel_sq_error * exp(beta_r * sign)
    # where sign = +1 (too far), -1 (too close), 0 (exact target).
    #   beta_r < 0 -> favours too-close errors
    #   beta_r > 0 -> favours too-far errors
    #   beta_r = 0 -> legacy behaviour (no sign bias)
    #                                     [0,    1,    2,    3,    4,    5,    6,    7,    8   ]
    #                                     [doma, phyl, clas, orde, fami, genu, spec, sequ, subs]
    PAIR_SIGN_BIAS_BETA_PER_RANK = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    # ---
    # Think of this as our best expectations for precision of pair distances. At the level of V4 or V3 for example, 
    # we can't expect perfect precision. Similarly, at higher ranks, we don't expect perfect precision.
    # See relative_error_epsilon.png for visualisation of the maths behind this.
    #                                     [0,    1,    2,    3,    4,    5,    6,    7,    8   ]
    #                                     [doma, phyl, clas, orde, fami, genu, spec, sequ, subs]
    RELATIVE_ERROR_EPSILONS_PAIR_LOSS   = [0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08]      # Recommended value is roughly 0.01-0.2
    RELATIVE_ERROR_EPSILONS_PAIR_MINING = [0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08]      # Recommended value is roughly 0.01-0.2
    PAIR_RELATIVE_LOSS_CAP = 2.0
    # ---
    # Triplet relative weighting mirrors the pair loss behaviour:
    # scale denominator uses (margin + eps_per_rank), not max(margin, eps)
    #                                     [0,    1,    2,    3,    4,    5,  ]
    #                                     [doma, phyl, clas, orde, fami, genu]
    RELATIVE_ERROR_EPSILONS_TRIPLET_LOSS = [0.02, 0.02, 0.02, 0.02, 0.02, 0.02]
    TRIPLET_RELATIVE_LOSS_CAP = 4.0


    # Mining ---
    # Boolean list for which ranks to mine pairs
    # e.g. 0 = domain, which means mine pairs that are **different** domains (i.e. different domains, both prokaryotes)
    # e.g. 5 = genus, which means mine pairs that are **different** genus (i.e. different genus, same family)
    # e.g. 7 = sequence, which means mine pairs that are **different** sequence (i.e. different sequence, same species)
        # Note: There are very few examples of multiple sequences per species, so we don't typically mine pairs at rank 7.
    # e.g. 8 = subsequence, which means mine duplicate pairs of the same sequence (regions may match depending on SUBSEQUENCES_ALWAYS_CROSS_REGION)
    #            [0,    1,    2,    3,    4,    5,    6,    7,    8   ]
    #            [doma, phyl, clas, orde, fami, genu, spec, sequ, subs]
    PAIR_RANKS = [True, True, True, True, True, True, True, False, True]
    # Boolean list for which ranks to mine triplets at
    #  - Anchor and Positive share the same taxon at rank r (may also share deeper taxonomy)
    #  - Anchor and Negative share the same classification at rank r-1, but differ at rank r
    # e.g. 0 = domain, which means:
    #    - Anchor and Positive are from the same domain
    #    - Anchor and Negative are both prokaryotes, but different domains
    # e.g. 5 = genus, which means:
    #    - Anchor and Positive are from the same genus
    #    - Anchor and Negative are same family, but different genera
    # Note: We have not implemented mining triplets at ranks 6 (species), 7 (sequence) or 8 (subsequence) yet. 
    #       This is partially due to the fact that there are very few examples of multiple sequences per species. 
    #       But, also in future, triplets at subsequence/species level could be useful. For now, pair mining works 
    #       well at this level.
    #               [0,    1,    2,    3,    4,    5,  ]
    #               [doma, phyl, clas, orde, fami, genu]
    TRIPLET_RANKS = [True, True, True, True, True, True]
    # Batch number when each rank is introduced for pair/triplet mining.
    # Until batch_num reaches this threshold, the rank is treated as disabled for mining.
    # Length 9: [domain, phylum, class, order, family, genus, species, sequence, subseq]
    INTRODUCE_RANK_AT_BATCHES = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    # Number of batches worth of pairs/triplets to mine at once
    # When > 1, mining runs less frequently but collects more data per run
    # This reduces overhead by amortizing inference/distance computation across batches
    # Must be >= 1. Value of 1 means mine every batch (original behavior).
    NUM_BATCHES_PER_MINING_RUN = 1
    # Shuffle mined pairs/triplets before returning from mine_pairs_and_triplets().
    # This affects only training sample order, not candidate selection.
    SHUFFLE_TRIPLETS_AND_PAIRS_AFTER_MINING = True
    # Train sequence subsampling proportion for all mining
    MINING_TRAIN_SEQUENCES_SUB_SAMPLE_BACTERIA = 0.2
    MINING_TRAIN_SEQUENCES_SUB_SAMPLE_ARCHAEA = 0.2
    # Representative-set taxon-size balancing (weighted sampling at representative-set stage)
    # When enabled, representative-set subsampling uses weighted sampling without replacement
    # to counteract combinatorial taxon-size effects. Weight formula per candidate pair/AP (i,j)
    # at rank r:  w = ((b_r^2) / (c_i * c_j + eps)) ^ lambda
    #   lambda=0 -> uniform (current behaviour), lambda=1 -> full combinatorial counteraction.
    # We split lambda by mining objective because pair and triplet representative pools
    # scale differently with taxon size and usually need different balancing strengths.
    # Note: Taxon counts (c_i, c_j) are corrected for domain-based downsampling at mining time
    # by scaling each sequence's counts with its domain's subsample fraction
    # (MINING_TRAIN_SEQUENCES_SUB_SAMPLE_BACTERIA / _ARCHAEA), so the weights reflect
    # actual mining pool sizes rather than full training set counts.
    USE_REPRESENTATIVE_TAXON_SIZE_BALANCING = True
    PAIR_REPRESENTATIVE_TAXON_SIZE_BALANCE_LAMBDA = 0.35    # Float in [0, 1]
    TRIPLET_REPRESENTATIVE_TAXON_SIZE_BALANCE_LAMBDA = 0.8 # Float in [0, 1]
    REPRESENTATIVE_TAXON_SIZE_BALANCE_EPS = 1e-12     # Numerical safety for denominator
    REPRESENTATIVE_TAXON_SIZE_BALANCE_WEIGHT_CLIP = 50.0  # Optional float cap for extreme weights (e.g. 50.0)
    # Taxon-size bias (pairwise down-weighting for large taxa)
    # Note: If using this feature in future, consider if it appropriately takes into account the domain-based downsampling of the training set (i.e. should it use taxon_counts_for_rep_balancing and other numbers)
    USE_TAXON_SIZE_MINING_BIAS = False
    # Taxon-size bias baseline stats (used to compute per-rank baseline counts at load time)
    TAXON_SIZE_MINING_BIAS_BASELINE_STAT = "median"  # "median", "mean", or "percentile"
    TAXON_SIZE_MINING_BIAS_BASELINE_PERCENTILE = 50.0
    # Length 7: [domain, phylum, class, order, family, genus, species]
    # Alphas control strength (0 = off, 0.5 = gentle, 1 = strong)
    TAXON_SIZE_MINING_BIAS_ALPHAS_PER_RANK = [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]
    # Optional keep floor per rank (None disables the floor)
    TAXON_SIZE_MINING_BIAS_MIN_KEEP_PER_RANK = None
    # Representative set sizes PER RANK for pair mining
    # Used to estimate error distribution and also the subset which pairs are sampled from
    # Length 9:                          [domain, phylum, class, order, family, genus, species, seq, subseq]
    PAIR_MINING_REPRESENTATIVE_SET_SIZES = [40000, 40000, 40000, 40000, 40000, 40000, 40000, 0, 8000]
    # Representative set sizes PER RANK for triplet mining
    # Used to estimate error distribution and also the subset which triplets are sampled from
    # Length 6:                             [domain, phylum, class, order, family, genus]
    TRIPLET_MINING_REPRESENTATIVE_SET_SIZES = [40000, 40000, 40000, 40000, 40000, 40000]
    # Per-rank hardness EMA (Exponential Moving Average) parameters
    # These control how quickly the per-rank batch allocation adapts to changing hardness metrics
    # Alpha in (0, 1]: higher = more weight on recent observations, lower = smoother
    # α = 0.5    →   ~3 batch moving average
    # α = 0.3    →   ~6 batch moving average
    # α = 0.25   →   ~7 batch moving average
    # α = 0.2    →   ~9 batch moving average
    # α = 0.15   →   ~12 batch moving average
    # α = 0.1    →   ~19 batch moving average
    # α = 0.05   →   ~39 batch moving average
    # α = 0.02   →   ~100 batch moving average
    # α = 0.01   →   ~200 batch moving average
    PAIR_MINING_EMA_ALPHA = 0.05
    TRIPLET_MINING_EMA_ALPHA = 0.05
    # Exponent applied to EMA hardness before normalising into proportions.
    # 1.0 keeps the existing linear behaviour, >1 exaggerates differences, <1 flattens them.
    # A practical range is roughly 0.5-2.0
    PAIR_MINING_EMA_WEIGHT_EXPONENT = 1.0
    TRIPLET_MINING_EMA_WEIGHT_EXPONENT = 1.0
    # EMA Metric weights for triplets
    # The metric is calculated as:
    # metric = hard_triplet_prop * TRIPLET_EMA_HARD_WEIGHT + moderate_triplet_prop * TRIPLET_EMA_MODERATE_WEIGHT
    # where:
    #   - hard_triplet_prop: proportion of triplets where AP >= AN
    #   - moderate_triplet_prop: proportion of triplets where AP < AN < AP + M (i.e. violation > 0 but AP < AN)
    TRIPLET_EMA_HARD_WEIGHT = 3.0
    TRIPLET_EMA_MODERATE_WEIGHT = 1.0
    # EMA Metric weights for pairs
    # The metric is calculated as:
    # metric = mean_error * PAIR_EMA_MEAN_WEIGHT + (p25_error + p75_error) * PAIR_EMA_QUARTILES_WEIGHT
    # where:
    #   - mean_error: mean relative squared error
    #   - p25_error: 25th percentile of relative squared error
    #   - p75_error: 75th percentile of relative squared error
    PAIR_EMA_MEAN_WEIGHT = 2.0
    PAIR_EMA_QUARTILES_WEIGHT = 1.0
    # Minimum proportion of the batch that any single rank must receive
    PAIR_PER_RANK_BATCH_PROPORTION_MIN = 0.05
    TRIPLET_PER_RANK_BATCH_PROPORTION_MIN = 0.05
    # Cap on how much of the batch any single rank can receive (prevents one rank from dominating)
    PAIR_PER_RANK_BATCH_PROPORTION_MAX = 0.5
    TRIPLET_PER_RANK_BATCH_PROPORTION_MAX = 0.5
    # Uniform mining warmup to avoid early collapse
    # Stage 1: Uniform mining (MINING_WARMUP_UNIFORM_DURATION)
    # Stage 2: Linear transition to bucketed mining and EMA-weighted rank budgets (MINING_WARMUP_TRANSITION_DURATION)
    # Stage 3: Full EMA-driven and bucketed mining (remainder of training)
    PAIR_MINING_WARMUP_UNIFORM_DURATION = 0
    PAIR_MINING_WARMUP_TRANSITION_DURATION = 0
    TRIPLET_MINING_WARMUP_UNIFORM_DURATION = 0
    TRIPLET_MINING_WARMUP_TRANSITION_DURATION = 0
    # Triplet Positive Selection Bias ---
    # When enabled, AP candidates are re-sampled within each (anchor, triplet_rank)
    # group using softmax-biased probabilities over predicted anchor-positive distance.
    #   p(p) = softmax(+beta_eff * d_pred(anchor, p))  over in-group positives
    # This upweights farther (harder) positives while preserving AP pool size.
    # beta_eff ramps using the existing triplet warmup:
    #   beta_eff = BETA * (1 - triplet_warmup_phase)
    # beta_eff = 0 -> no bias/original AP pool; larger beta_eff -> stronger hard-positive preference
    USE_TRIPLET_MINING_POSITIVE_SELECTION_BIAS = True
    TRIPLET_MINING_POSITIVE_SELECTION_BIAS_BETA = 1.0
    TRIPLET_MINING_POSITIVE_SELECTION_BIAS_EPS = 1e-12  # Denominator guard for degenerate softmax
    # Triplet Negative Selection Bias ---
    # When enabled, negatives are sampled with softmax-biased probabilities over predicted
    # embedding distance, so closer (harder) negatives are more likely to be selected.
    #   p(n) = softmax(-beta_eff * d_pred(anchor, n))  over valid negatives
    # beta_eff ramps using the existing triplet warmup:
    #   beta_eff = BETA * (1 - triplet_warmup_phase)
    # This keeps behaviour near-uniform during warmup and turns bias on gradually.
    # beta_eff = 0 -> uniform (no bias), larger beta_eff -> stronger preference for close/hard negatives
    USE_TRIPLET_MINING_NEGATIVE_SELECTION_BIAS = True
    TRIPLET_MINING_NEGATIVE_SELECTION_BIAS_BETA = 2.0
    TRIPLET_MINING_NEGATIVE_SELECTION_BIAS_EPS = 1e-12  # Denominator guard for degenerate softmax
    # Number of negatives to sample per (anchor, positive) candidate during triplet mining.
    # 1 = current behaviour. K > 1 expands the candidate triplet pool by Kx before
    # filtering/bucketing, then the final batch still follows N_TRIPLETS_PER_BATCH.
    TRIPLET_NEGATIVES_PER_AP = 1
    # Proportion of sequences to duplicate for subsequence pair mining (rank 8)
    # These duplicates can be forced onto different regions (see SUBSEQUENCES_ALWAYS_CROSS_REGION) and always get fresh augmentations
    PROPORTION_IDENTICAL_SEQUENCES_TO_GENERATE = 1.0
    # If True, duplicates used for subsequence pairs must sample different regions than their originals
    SUBSEQUENCES_ALWAYS_CROSS_REGION = True
    # Target distance applied to subsequence pairs during pair loss/mining
    SUB_SEQUENCE_TRUE_DISTANCE = 0.0
    # Downsample pairs at ranks (note: this affects both pair and triplet mining)
    DOWNSAMPLE_PAIRS_AT_RANK = [
        # (rank, proportion)
        (0, 1.0), # domain
        (1, 1.0), # phylum
        (2, 1.0), # class
        (3, 1.0), # order
        (4, 1.0), # family
        (5, 1.0), # genus
        (6, 1.0), # species
        (7, 1.0), # sequence
        # (8, 1.0), # subsequence   <- Use PROPORTION_IDENTICAL_SEQUENCES_TO_GENERATE instead for this
    ]
    # Pin CPU tensors that are shipped to the GPU multiple times during mining/inference
    PIN_MEMORY_FOR_MINING = True
    # If True, explicitly delete large intermediate matrices at the end of the mining step to free memory
    FREE_LARGE_INTERMEDIATES_WHEN_MINING = False
    # Batch size
    MINING_BATCH_SIZE = 300
    # Use float16 for the mined cosine distance matrix to halve transfer cost 
    # Note: This is somewhat unreliable and sometimes only saves <20ms per batch
    USE_FLOAT_16_FOR_MINING_COSINE_DISTANCES = False


    # Pair Mining ---
    # Pair mining buckets
    # List of tuples: (percentile_gap, sampling_proportion), 
    # where the first tuple is the first bucket with the lowest errors
    # The 'any' bucket is specified with percentile_gap=None
    PAIR_MINING_BUCKETS = [
        # (percentile_gap, sampling_proportion)
        (0.25, 0.10),
        (0.25, 0.15),
        (0.25, 0.20),
        (0.15, 0.25),
        (0.08, 0.15),
        (0.02, 0.15),
        (None, 0.00),
    ]

    # Triplet Mining ---
    # Triplet mining buckets
    # List of tuples: (percentile_gap, sampling_proportion), 
    # where the first tuple is the first bucket with the lowest errors
    # The 'any' bucket is specified with percentile_gap=None
    TRIPLET_MINING_BUCKETS = [
        # (percentile_gap, sampling_proportion)
        (0.35, 0.25),
        (0.30, 0.25),
        (0.20, 0.25),
        (0.15, 0.25),
        (None, 0.00),
    ]


    # Filter out triplets that would produce zero loss (violation <= 0)
    # This focuses mining on triplets that actually provide a training signal
    FILTER_ZERO_LOSS_TRIPLETS = True
    # Controls the minimum volume of hard triplets to keep per rank after filtering.
    # The effective minimum = MIN_TRIPLETS_AFTER_FILTER_PER_RANK_BATCH_SIZE_FACTOR * per_rank_batch_size.
    MIN_TRIPLETS_AFTER_FILTER_PER_RANK_BATCH_SIZE_FACTOR = 1.0


    # Sequence variants (regions & augmentation) ------------------
    # Sequence regions
    USE_FULL_SEQS = False
    USE_SUB_SEQS = True
    MAX_NUM_SUBSEQS = None           # Max subsequence regions (None = keep all)
    # Training augmentation
    PROP_SHIFT_SEQS = 0.5
    PROP_TRUNC = 1.0
    MIN_TRUNC_START = 0
    MAX_TRUNC_START = 60
    MIN_TRUNC_END = 0
    MAX_TRUNC_END = 60
    MUTATION_RATE = 0.01        # Does not seem to improve performance  |  >0.05 has negative impact on performance
    # Quick test augmentation
    QT_PROP_SHIFT_SEQS = 0.0
    QT_PROP_TRUNC = 1.0
    QT_MIN_TRUNC_START = 0
    QT_MAX_TRUNC_START = 60
    QT_MIN_TRUNC_END = 0
    QT_MAX_TRUNC_END = 60
    QT_MUTATION_RATE = 0.01


    # Validation ------------------
    # Saving ---
    SAVE_EVERY_N_BATCHES = 500
    # Loss recording ---
    RECORD_LOSS_EVERY_N_BATCHES = 5
    PLOT_TOTAL_LOSS = True
    PLOT_TRIPLET_LOSS = True
    PLOT_PAIR_LOSS = True
    PLOT_UNCERTAINTY_WEIGHTING = True
    PLOT_LOSS_EVERY_N_BATCHES = 25
    SECOND_LOSS_PLOT_AFTER_N_BATCHES = 3000


    # Verbose printing and logging ---
    # Note: These operations can be computationally intensive
    # Training loop (timing)
    VERBOSE_TRAINING_TIMING = False # (always prints every batch)
    # Mining (timing) 
    VERBOSE_MINING_TIMING = False  # (always prints every batch)
    # Mining (triplets)
    VERBOSE_TRIPLET_MINING = False
    LOG_TRIPLET_MINING = True
    # Mining (pairs)
    VERBOSE_PAIR_MINING = False
    LOG_PAIR_MINING = True
    # Mining (combined)
    LOG_MINING = True
    # Mining (triplet/pair buckets)
    LOG_MINING_BUCKET_THRESHOLDS = False
    # Mining (phylum diagnostics)
    LOG_MINING_PHYLA_COUNTS = False 
    MINING_PHYLA_LOG_MAX_LINES_PER_TABLE = 200
    # Mining (counts of Archaeal and Bacterial sequences)
    LOG_MINING_ARC_BAC_COUNTS = True
    # Mining (cross-region logging)
    CROSS_REGION_LOGGING_MAX_LINES_PER_TABLE = 6 # With 3 regions, there are 3 combinations. but with 30 regions, 435 combinations. This caps logging to N lines per table.
    # Loss (triplets)
    VERBOSE_TRIPLET_LOSS = False
    LOG_TRIPLET_LOSS = True
    # Loss (pairs)
    VERBOSE_PAIR_LOSS = False
    LOG_PAIR_LOSS = True
    # Uncertainty weighting
    VERBOSE_UNCERTAINTY_WEIGHTING = False
    LOG_UNCERTAINTY_WEIGHTING = True
    # Conv stem scale
    VERBOSE_CONV_STEM_SCALE = False
    LOG_CONV_STEM_SCALE = True
    # Frequency of verbose printing and logging
    VERBOSE_EVERY_N_BATCHES = 100
    LOG_EVERY_N_BATCHES = 100
    # Training loop
    PRINT_BATCH_NUM_EVERY_N_BATCHES = 25
    # Quick test
    QT_VERBOSE = False
    QT_PRINT_RESULTS = False

    # Logging Plotting ---
    # Plotting frequency
    PLOT_TRIPLET_SATISFACTION_EVERY_N_BATCHES = 50   # Should be divisible by LOG_EVERY_N_BATCHES and LOG_TRIPLET_MINING must be True
    PLOT_PAIR_DISTANCES_EVERY_N_BATCHES = 50         # Should be divisible by LOG_EVERY_N_BATCHES and LOG_PAIR_MINING must be True
    PLOT_PAIR_ERROR_METRICS_EVERY_N_BATCHES = 50     # Should be divisible by LOG_EVERY_N_BATCHES and LOG_PAIR_MINING must be True
    PLOT_TRIPLET_ERROR_METRICS_EVERY_N_BATCHES = 50  # Should be divisible by LOG_EVERY_N_BATCHES and LOG_TRIPLET_MINING must be True
    PLOT_CONV_STEM_SCALE_EVERY_N_BATCHES = 50       # Should be divisible by LOG_EVERY_N_BATCHES and LOG_CONV_STEM_SCALE must be True
    # Second (cropped) plot after n batches
    SECOND_TRIPLET_SATISFACTION_PLOT_AFTER_N_BATCHES = 3000
    SECOND_PAIR_ERROR_METRICS_PLOT_AFTER_N_BATCHES = 3000
    SECOND_TRIPLET_ERROR_METRICS_PLOT_AFTER_N_BATCHES = 3000
    SECOND_CONV_STEM_SCALE_PLOT_AFTER_N_BATCHES = 3000

    # Quick test ---
    # Do quick test? and how frequently?
    DO_QT = True
    QT_BEFORE_FIRST_BATCH = True
    QT_EVERY_N_BATCHES = 200
    # Batch size
    QT_BATCH_SIZE = 300
    # K-mer results
    QT_KMER_K_VALUES = ()
    QT_GET_KMER_RESULTS = False
    # Ordination
    QT_PCOA_RANKS = (0,1)
    QT_PCOA_EVERY_N_BATCHES = 100000   # SHOULD BE DIVISIBLE BY QT_EVERY_N_BATCHES
    QT_UMAP_RANKS = (0,1)
    QT_UMAP_EVERY_N_BATCHES = 1000   # SHOULD BE DIVISIBLE BY QT_EVERY_N_BATCHES
    # Clustering
    QT_CLUSTERING_RANKS = (0,1,2,3,4) # 0-6 (domain to species) is valid
    # Rank-level clustering (per-taxon clustering scores at specific ranks)
    # e.g. (1,2,3) creates phyla.csv, classes.csv, orders.csv in qt_results/clustering_per_rank/
    QT_RANK_LEVEL_CLUSTERING_METRICS_AT_RANKS = (0,1,2)
    QT_MAX_TAXA_PER_RANK_LEVEL_CLUSTERING = 1000  # Max taxa per CSV file to limit disk usage
    RANK_LEVEL_CLUSTERING_PLOTS_EVERY_N_BATCHES = 1000
    # Classification
    QT_CLASSIFICATION_RANKS = (0,1,2,3,4)
    # Record missclassified sequence IDs per rank (domain -> species)
    # Note: this only applies to embedding classification (not k-mer classification).
    QT_RECORD_MISS_CLASSIFICATION_SEQUENCES_AT_RANKS = [False, True, False, False, False, False, False]
    # Max subsequence regions to use in quick test (full sequence always included)
    QT_MAX_N_REGIONS = 8

    # Quick test tests to perform
    QT_TESTS_TODO = {
    # PCoAs -------------------
    # Embeddings
        # 'train-full-embed-pcoa',
        # 'test-full-embed-pcoa',
        'excl-full-embed-pcoa',
        'train-subseqs-embed-pcoa',
        'test-subseqs-embed-pcoa',
        # 'excl-subseqs-embed-pcoa',
        # 'train-both-embed-pcoa',
        # 'test-both-embed-pcoa',
        # 'excl-both-embed-pcoa',
    # K-mers
        # 'train-full-kmer-pcoa',
        # 'test-full-kmer-pcoa',
        # 'excl-full-kmer-pcoa',
        # 'train-subseqs-kmer-pcoa',
        # 'test-subseqs-kmer-pcoa',
        # 'excl-subseqs-kmer-pcoa',
        # 'train-both-kmer-pcoa',
        # 'test-both-kmer-pcoa',
        # 'excl-both-kmer-pcoa',

    # UMAPs -------------------
    # Embeddings
        # 'train-full-embed-umap',
        # 'test-full-embed-umap',
        'excl-full-embed-umap',
        'train-subseqs-embed-umap',
        'test-subseqs-embed-umap',
        # 'excl-subseqs-embed-umap',
        # 'train-both-embed-umap',
        # 'test-both-embed-umap',
        # 'excl-both-embed-umap',
    # K-mers
        # 'train-full-kmer-umap',
        # 'test-full-kmer-umap',
        # 'excl-full-kmer-umap',
        # 'train-subseqs-kmer-umap',
        # 'test-subseqs-kmer-umap',
        # 'excl-subseqs-kmer-umap',
        # 'train-both-kmer-umap',
        # 'test-both-kmer-umap',
        # 'excl-both-kmer-umap',

    # Clustering -----------------
    # Embeddings
        # 'train-full-embed-clust',
        # 'test-full-embed-clust',
        # 'excl-full-embed-clust',
        'train-subseqs-embed-clust',
        'test-subseqs-embed-clust',
        'excl-subseqs-embed-clust',
        # 'train-both-embed-clust',
        # 'test-both-embed-clust',
        # 'excl-both-embed-clust',
    # K-mers
        # 'train-full-kmer-clust',
        # 'test-full-kmer-clust',
        # 'excl-full-kmer-clust',
        # 'train-subseqs-kmer-clust',
        # 'test-subseqs-kmer-clust',
        # 'excl-subseqs-kmer-clust',
        # 'train-both-kmer-clust',
        # 'test-both-kmer-clust',
        # 'excl-both-kmer-clust',

    # Classification -------------
    # Note: Train-set classification is leave-one-out style (self-match blocked).
    # Embeddings
        # 'train-full-embed-class',
        # 'test-full-embed-class',
        # 'excl-full-embed-class',
        'train-subseqs-embed-class',
        'test-subseqs-embed-class',
        'excl-subseqs-embed-class',
        # 'train-both-embed-class',
        # 'test-both-embed-class',
        # 'excl-both-embed-class',
    # K-mers
        # 'train-full-kmer-class',
        # 'test-full-kmer-class',
        # 'excl-full-kmer-class',
        # 'train-subseqs-kmer-class',
        # 'test-subseqs-kmer-class',
        # 'excl-subseqs-kmer-class',
        # 'train-both-kmer-class',
        # 'test-both-kmer-class',
        # 'excl-both-kmer-class',

    # Subsequence Congruency (SSC) -------------
    # Embeddings
        'train-embed-ssc',
        'test-embed-ssc',
        'excl-embed-ssc',
    # K-mers
        # 'train-kmer-ssc',
        # 'test-kmer-ssc',
        # 'excl-kmer-ssc',
    }

    # General Constants ------------------
    # 3-bit representation of the sequences
    # (the 0th bit is for masking, these 1st and 2nd bits are for the sequence)
    SEQ_3BIT_REPRESENTATION_DICT = {
        'A': (0,0), 'C': (0,1), 
        'G': (1,0), 'T': (1,1)
        }
    


    def __init__(self):
        """Enforce some things upon initialization"""

        if not isinstance(self.NUM_BATCHES_PER_MINING_RUN, int) or self.NUM_BATCHES_PER_MINING_RUN < 1:
            raise ValueError("NUM_BATCHES_PER_MINING_RUN must be an integer >= 1")

        if not isinstance(self.NUM_MICRO_BATCHES_PER_BATCH, int) or self.NUM_MICRO_BATCHES_PER_BATCH < 1:
            raise ValueError("NUM_MICRO_BATCHES_PER_BATCH must be an integer >= 1")

        # Validate rank config structure early so rank indexing is always safe below.
        self._validate_rank_introduction_config(check_batch_zero_enabled=False)

        self._validate_start_from_ckpt()
        
        # Validate batch sizes when using microbatches
        if self.NUM_MICRO_BATCHES_PER_BATCH > 1:
            # At least one loss type must be active
            if self.N_TRIPLETS_PER_BATCH == 0 and self.N_PAIRS_PER_BATCH == 0:
                raise ValueError("At least one of N_TRIPLETS_PER_BATCH or N_PAIRS_PER_BATCH must be > 0")
            
            # Only check microbatch requirement for active loss types
            if self.N_TRIPLETS_PER_BATCH > 0 and self.N_TRIPLETS_PER_BATCH < self.NUM_MICRO_BATCHES_PER_BATCH:
                raise ValueError(f"N_TRIPLETS_PER_BATCH ({self.N_TRIPLETS_PER_BATCH}) must be >= NUM_MICRO_BATCHES_PER_BATCH ({self.NUM_MICRO_BATCHES_PER_BATCH}) when using triplets")
            if self.N_PAIRS_PER_BATCH > 0 and self.N_PAIRS_PER_BATCH < self.NUM_MICRO_BATCHES_PER_BATCH:
                raise ValueError(f"N_PAIRS_PER_BATCH ({self.N_PAIRS_PER_BATCH}) must be >= NUM_MICRO_BATCHES_PER_BATCH ({self.NUM_MICRO_BATCHES_PER_BATCH}) when using pairs")

        if self.PAIR_RANKS[8] and not self.USE_SUB_SEQS:
            raise ValueError("PAIR_RANKS[8]=True and USE_SUB_SEQS=False is not supported")
        
        if self.PAIR_RANKS[8] and self.PROPORTION_IDENTICAL_SEQUENCES_TO_GENERATE <= 0:
            raise ValueError("PAIR_RANKS[8]=True requires PROPORTION_IDENTICAL_SEQUENCES_TO_GENERATE > 0")
    
        # Sequenceless mode validations
        if self.SEQLESS_MODE:
            # Warn if subsequence pairs are enabled in sequenceless mode
            if self.PAIR_RANKS[8]:
                print("WARNING: SEQLESS_MODE is True but PAIR_RANKS[8] (subsequence pairs) is enabled.")
                print("  Subsequence pairs are automatically disabled in sequenceless mode (distance would be 0).")
                print("  Setting PAIR_RANKS[8]=False.")
                self.PAIR_RANKS = list(self.PAIR_RANKS)  # Convert to list if tuple
                self.PAIR_RANKS[8] = False
                self.PAIR_RANKS = tuple(self.PAIR_RANKS)  # Convert back to tuple
            
            # Quick test adjustments for sequenceless mode
            if self.QT_GET_KMER_RESULTS:
                print("NOTICE: SEQLESS_MODE is True -> setting QT_GET_KMER_RESULTS=False.")
            self.QT_GET_KMER_RESULTS = False
            # Only one region exists in sequenceless mode
            self.QT_MAX_N_REGIONS = 1
            # Ensure QT_TESTS_TODO is a set for edits
            if not isinstance(self.QT_TESTS_TODO, set):
                self.QT_TESTS_TODO = set(self.QT_TESTS_TODO)
            # Remove unsupported tests
            remove_terms = ('ssc', 'test', 'excl', 'class', 'kmer', 'subseqs', 'both')
            removed = {t for t in self.QT_TESTS_TODO if any(term in t for term in remove_terms)}
            if removed:
                print(f"NOTICE: SEQLESS_MODE is True -> removing {len(removed)} quick test items not supported in sequenceless mode.")
                self.QT_TESTS_TODO -= removed
            # Ensure core train-full embedding tests are present
            required_tests = (
                'train-full-embed-pcoa',
                'train-full-embed-umap',
                'train-full-embed-clust',
            )
            for test in required_tests:
                if test not in self.QT_TESTS_TODO:
                    print(f"NOTICE: SEQLESS_MODE is True -> adding '{test}' to QT_TESTS_TODO.")
                    self.QT_TESTS_TODO.add(test)

        # Validate rank-introduction config again after any runtime rank edits (e.g. SEQLESS_MODE).
        self._validate_rank_introduction_config(check_batch_zero_enabled=True)
        
        if self.PROPORTION_IDENTICAL_SEQUENCES_TO_GENERATE < 0 or self.PROPORTION_IDENTICAL_SEQUENCES_TO_GENERATE > 1:
            raise ValueError("PROPORTION_IDENTICAL_SEQUENCES_TO_GENERATE must be between 0 and 1")
        if not (0.0 <= self.SUB_SEQUENCE_TRUE_DISTANCE <= 2.0):
            raise ValueError("SUB_SEQUENCE_TRUE_DISTANCE must be between 0.0 and 2.0 (cosine distance range)")

        if not self.USE_FULL_SEQS and not self.USE_SUB_SEQS:
            raise ValueError("At least one of USE_FULL_SEQS or USE_SUB_SEQS must be True")
        self._validate_max_num_subseqs()
        if (self.MAX_IMPORTED_SEQ_LEN is not None and self.MAX_MODEL_SEQ_LEN is not None 
                and self.MAX_IMPORTED_SEQ_LEN < self.MAX_MODEL_SEQ_LEN):
            raise ValueError("MAX_IMPORTED_SEQ_LEN must be >= MAX_MODEL_SEQ_LEN to preserve model input length.")

        for attr_name in ("PROP_SHIFT_SEQS", "QT_PROP_SHIFT_SEQS", "DISTANCE_GAMMA_CORRECTION_GAMMA"):
            val = getattr(self, attr_name)
            if attr_name == "DISTANCE_GAMMA_CORRECTION_GAMMA":
                if not (0.0 < val <= 2.0): # 2.0 is the upper bound where log2(2/gamma) = 0
                    raise ValueError(f"{attr_name} must be between 0.0 (exclusive) and 2.0")
            elif not (0.0 <= val <= 1.0):
                raise ValueError(f"{attr_name} must be between 0.0 and 1.0")

        self._validate_pooling_type()

        self._validate_logging_params()
        self._validate_quick_test_params()
        
        # Validate PAIR_MINING_BUCKETS
        total_sampling = 0.0
        total_gap = 0.0
        has_any_bucket = False

        for gap, prop in self.PAIR_MINING_BUCKETS:
            total_sampling += prop
            if gap is not None:
                total_gap += gap
            else:
                if has_any_bucket:
                    raise ValueError("PAIR_MINING_BUCKETS can only have one 'any' bucket (gap=None)")
                has_any_bucket = True
        
        if not np.isclose(total_sampling, 1.0):
            raise ValueError(f"PAIR_MINING_BUCKETS sampling proportions must sum to 1.0. Got {total_sampling}")
        
        if not np.isclose(total_gap, 1.0):
            raise ValueError(f"PAIR_MINING_BUCKETS percentile gaps must sum to 1.0. Got {total_gap}")

        # Validate TRIPLET_MINING_BUCKETS
        total_sampling = 0.0
        total_gap = 0.0
        has_any_bucket = False

        for gap, prop in self.TRIPLET_MINING_BUCKETS:
            total_sampling += prop
            if gap is not None:
                total_gap += gap
            else:
                if has_any_bucket:
                    raise ValueError("TRIPLET_MINING_BUCKETS can only have one 'any' bucket (gap=None)")
                has_any_bucket = True
        
        if not np.isclose(total_sampling, 1.0):
            raise ValueError(f"TRIPLET_MINING_BUCKETS sampling proportions must sum to 1.0. Got {total_sampling}")
        
        if not np.isclose(total_gap, 1.0):
            raise ValueError(f"TRIPLET_MINING_BUCKETS percentile gaps must sum to 1.0. Got {total_gap}")

        # Validate DOWNSAMPLE_PAIRS_AT_RANK
        if self.DOWNSAMPLE_PAIRS_AT_RANK:
            seen_ranks = set()
            for rank, prop in self.DOWNSAMPLE_PAIRS_AT_RANK:
                if not isinstance(rank, int) or rank < 0 or rank > 8:
                    raise ValueError(f"DOWNSAMPLE_PAIRS_AT_RANK: rank must be integer 0-8, got {rank}")
                if rank in seen_ranks:
                    raise ValueError(f"DOWNSAMPLE_PAIRS_AT_RANK: duplicate rank {rank}")
                seen_ranks.add(rank)
                if not (0.0 <= prop <= 1.0):
                    raise ValueError(f"DOWNSAMPLE_PAIRS_AT_RANK: proportion must be 0-1, got {prop}")

        self._validate_lr_scheduler_config()

        # Validate per-rank mining parameters
        self._validate_per_rank_mining_params()

        # Validate taxa cap config
        self._validate_taxa_cap()

    def write_training_config(self, model_name, model_dir, txt=True, json=True):
        if txt:
            with open(f"{model_dir}/training_config.txt", "w") as f:
                f.write(f"----------------------------------------\n")
                f.write(f"Micro16S Model Training Configuration for: {model_name}\n")
                f.write(f"----------------------------------------\n\n")
                f.write(f"DATE_AND_TIME: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

                f.write("# Description ------------------\n")
                f.write(f"DESCRIPTION: {self.DESCRIPTION}\n\n")

                f.write("# Paths ------------------\n")
                f.write(f"MODELS_DIR: {self.MODELS_DIR}\n")
                f.write(f"DATASET_SPLIT_DIR: {self.DATASET_SPLIT_DIR}\n\n")
                f.write(f"REDVALS_DIR: {self.REDVALS_DIR}\n\n")

                f.write("# Dataset Filtering ------------------\n")
                f.write(f"TAXA_CAP_MAX_NUM_TAXA: {self.TAXA_CAP_MAX_NUM_TAXA}\n")
                f.write(f"TAXA_CAP_RANK: {self.TAXA_CAP_RANK}\n")
                f.write(f"TAXA_CAP_VERBOSE: {self.TAXA_CAP_VERBOSE}\n\n")

                f.write("# Model parameters ------------------\n")
                f.write(f"EMBED_DIMS: {self.EMBED_DIMS}\n")
                f.write(f"MAX_MODEL_SEQ_LEN: {self.MAX_MODEL_SEQ_LEN}\n")
                f.write(f"MAX_IMPORTED_SEQ_LEN: {self.MAX_IMPORTED_SEQ_LEN}\n")
                f.write(f"D_MODEL: {self.D_MODEL}\n")
                f.write(f"N_LAYERS: {self.N_LAYERS}\n")
                f.write(f"N_HEAD: {self.N_HEAD}\n")
                f.write(f"D_FF: {self.D_FF}\n")
                f.write(f"POOLING_TYPE: {self.POOLING_TYPE}\n")
                f.write(f"USE_CONV_STEM: {self.USE_CONV_STEM}\n")
                f.write(f"CONV_STEM_KERNEL_SIZE: {self.CONV_STEM_KERNEL_SIZE}\n")
                f.write(f"CONV_STEM_RESIDUAL: {self.CONV_STEM_RESIDUAL}\n")
                f.write(f"CONV_STEM_INIT_SCALE: {self.CONV_STEM_INIT_SCALE}\n")
                f.write(f"USE_CONVFORMER: {self.USE_CONVFORMER}\n")
                f.write(f"CONFORMER_KERNEL_SIZE: {self.CONFORMER_KERNEL_SIZE}\n")
                f.write(f"DROPOUT_PROP: {self.DROPOUT_PROP}\n")
                f.write(f"ATT_DROPOUT_PROP: {self.ATT_DROPOUT_PROP}\n")
                f.write("\n")

                f.write("# Training parameters ------------------\n")
                f.write(f"LEARNING_RATE: {self.LEARNING_RATE}\n")
                f.write(f"WEIGHT_DECAY: {self.WEIGHT_DECAY}\n")
                f.write(f"NUM_BATCHES: {self.NUM_BATCHES}\n")
                f.write(f"START_FROM_CKPT: {self.START_FROM_CKPT}\n")
                f.write(f"NUM_MICRO_BATCHES_PER_BATCH: {self.NUM_MICRO_BATCHES_PER_BATCH}\n")
                f.write(f"LR_SCHEDULER_TYPE: {self.LR_SCHEDULER_TYPE}\n")
                f.write(f"LR_SCHEDULER_KWARGS: {self.LR_SCHEDULER_KWARGS}\n")
                f.write(f"PAIR_LOSS_WEIGHT: {self.PAIR_LOSS_WEIGHT}\n")
                f.write(f"TRIPLET_LOSS_WEIGHT: {self.TRIPLET_LOSS_WEIGHT}\n")
                f.write(f"USE_UNCERTAINTY_WEIGHTING: {self.USE_UNCERTAINTY_WEIGHTING}\n")
                f.write(f"UNCERTAINTY_LEARNING_RATE: {self.UNCERTAINTY_LEARNING_RATE}\n\n")

                f.write("# Pairs and triplets ------------------\n")
                f.write(f"N_PAIRS_PER_BATCH: {self.N_PAIRS_PER_BATCH}\n")
                f.write(f"N_TRIPLETS_PER_BATCH: {self.N_TRIPLETS_PER_BATCH}\n")
                f.write(f"MINING_BATCH_SIZE: {self.MINING_BATCH_SIZE}\n")
                f.write(f"PIN_MEMORY_FOR_MINING: {self.PIN_MEMORY_FOR_MINING}\n")
                f.write(f"FREE_LARGE_INTERMEDIATES_WHEN_MINING: {self.FREE_LARGE_INTERMEDIATES_WHEN_MINING}\n")
                f.write(f"USE_FLOAT_16_FOR_MINING_COSINE_DISTANCES: {self.USE_FLOAT_16_FOR_MINING_COSINE_DISTANCES}\n")
                f.write(f"NUM_BATCHES_PER_MINING_RUN: {self.NUM_BATCHES_PER_MINING_RUN}\n")
                f.write(f"SHUFFLE_TRIPLETS_AND_PAIRS_AFTER_MINING: {self.SHUFFLE_TRIPLETS_AND_PAIRS_AFTER_MINING}\n")
                f.write(f"MINING_TRAIN_SEQUENCES_SUB_SAMPLE_BACTERIA: {self.MINING_TRAIN_SEQUENCES_SUB_SAMPLE_BACTERIA}\n")
                f.write(f"MINING_TRAIN_SEQUENCES_SUB_SAMPLE_ARCHAEA: {self.MINING_TRAIN_SEQUENCES_SUB_SAMPLE_ARCHAEA}\n")
                f.write(f"TAXON_SIZE_MINING_BIAS_BASELINE_STAT: {self.TAXON_SIZE_MINING_BIAS_BASELINE_STAT}\n")
                f.write(f"TAXON_SIZE_MINING_BIAS_BASELINE_PERCENTILE: {self.TAXON_SIZE_MINING_BIAS_BASELINE_PERCENTILE}\n")
                f.write(f"USE_REPRESENTATIVE_TAXON_SIZE_BALANCING: {self.USE_REPRESENTATIVE_TAXON_SIZE_BALANCING}\n")
                f.write(f"PAIR_REPRESENTATIVE_TAXON_SIZE_BALANCE_LAMBDA: {self.PAIR_REPRESENTATIVE_TAXON_SIZE_BALANCE_LAMBDA}\n")
                f.write(f"TRIPLET_REPRESENTATIVE_TAXON_SIZE_BALANCE_LAMBDA: {self.TRIPLET_REPRESENTATIVE_TAXON_SIZE_BALANCE_LAMBDA}\n")
                f.write(f"REPRESENTATIVE_TAXON_SIZE_BALANCE_EPS: {self.REPRESENTATIVE_TAXON_SIZE_BALANCE_EPS}\n")
                f.write(f"REPRESENTATIVE_TAXON_SIZE_BALANCE_WEIGHT_CLIP: {self.REPRESENTATIVE_TAXON_SIZE_BALANCE_WEIGHT_CLIP}\n")
                f.write(f"USE_TAXON_SIZE_MINING_BIAS: {self.USE_TAXON_SIZE_MINING_BIAS}\n")
                f.write(f"TAXON_SIZE_MINING_BIAS_ALPHAS_PER_RANK: {self.TAXON_SIZE_MINING_BIAS_ALPHAS_PER_RANK}\n")
                f.write(f"TAXON_SIZE_MINING_BIAS_MIN_KEEP_PER_RANK: {self.TAXON_SIZE_MINING_BIAS_MIN_KEEP_PER_RANK}\n")
                f.write("\n")

                f.write("# Pairs (online mining) ------------------\n")
                f.write(f"PAIR_RANKS: {self.PAIR_RANKS}\n")
                f.write(f"INTRODUCE_RANK_AT_BATCHES: {self.INTRODUCE_RANK_AT_BATCHES}\n")
                f.write(f"DOWNSAMPLE_PAIRS_AT_RANK: {self.DOWNSAMPLE_PAIRS_AT_RANK}\n")
                f.write(f"PAIR_MINING_BUCKETS: {self.PAIR_MINING_BUCKETS}\n")
                f.write(f"PAIR_SIGN_BIAS_BETA_PER_RANK: {self.PAIR_SIGN_BIAS_BETA_PER_RANK}\n")
                f.write(f"PAIR_MINING_REPRESENTATIVE_SET_SIZES: {self.PAIR_MINING_REPRESENTATIVE_SET_SIZES}\n")
                f.write(f"TRIPLET_MINING_REPRESENTATIVE_SET_SIZES: {self.TRIPLET_MINING_REPRESENTATIVE_SET_SIZES}\n")
                f.write(f"PAIR_MINING_EMA_ALPHA: {self.PAIR_MINING_EMA_ALPHA}\n")
                f.write(f"TRIPLET_MINING_EMA_ALPHA: {self.TRIPLET_MINING_EMA_ALPHA}\n")
                f.write(f"PAIR_MINING_EMA_WEIGHT_EXPONENT: {self.PAIR_MINING_EMA_WEIGHT_EXPONENT}\n")
                f.write(f"TRIPLET_MINING_EMA_WEIGHT_EXPONENT: {self.TRIPLET_MINING_EMA_WEIGHT_EXPONENT}\n")
                f.write(f"TRIPLET_EMA_HARD_WEIGHT: {self.TRIPLET_EMA_HARD_WEIGHT}\n")
                f.write(f"TRIPLET_EMA_MODERATE_WEIGHT: {self.TRIPLET_EMA_MODERATE_WEIGHT}\n")
                f.write(f"PAIR_PER_RANK_BATCH_PROPORTION_MAX: {self.PAIR_PER_RANK_BATCH_PROPORTION_MAX}\n")
                f.write(f"TRIPLET_PER_RANK_BATCH_PROPORTION_MAX: {self.TRIPLET_PER_RANK_BATCH_PROPORTION_MAX}\n")
                f.write(f"PAIR_PER_RANK_BATCH_PROPORTION_MIN: {self.PAIR_PER_RANK_BATCH_PROPORTION_MIN}\n")
                f.write(f"TRIPLET_PER_RANK_BATCH_PROPORTION_MIN: {self.TRIPLET_PER_RANK_BATCH_PROPORTION_MIN}\n")
                f.write(f"PROPORTION_IDENTICAL_SEQUENCES_TO_GENERATE: {self.PROPORTION_IDENTICAL_SEQUENCES_TO_GENERATE}\n")
                f.write(f"SUB_SEQUENCE_TRUE_DISTANCE: {self.SUB_SEQUENCE_TRUE_DISTANCE}\n")
                f.write("# Pairs (offline selection) ------------------\n")
                f.write(f"RED_DISTANCE_BETWEEN_DOMAINS: {self.RED_DISTANCE_BETWEEN_DOMAINS}\n")
                f.write(f"BACTERIA_DISTANCE_FACTOR: {self.BACTERIA_DISTANCE_FACTOR}\n")
                f.write(f"ARCHEA_DISTANCE_FACTOR: {self.ARCHEA_DISTANCE_FACTOR}\n")
                f.write("\n")

                f.write("# Triplet selection biases ------------------\n")
                f.write(f"USE_TRIPLET_MINING_POSITIVE_SELECTION_BIAS: {self.USE_TRIPLET_MINING_POSITIVE_SELECTION_BIAS}\n")
                f.write(f"TRIPLET_MINING_POSITIVE_SELECTION_BIAS_BETA: {self.TRIPLET_MINING_POSITIVE_SELECTION_BIAS_BETA}\n")
                f.write(f"TRIPLET_MINING_POSITIVE_SELECTION_BIAS_EPS: {self.TRIPLET_MINING_POSITIVE_SELECTION_BIAS_EPS}\n")
                f.write(f"USE_TRIPLET_MINING_NEGATIVE_SELECTION_BIAS: {self.USE_TRIPLET_MINING_NEGATIVE_SELECTION_BIAS}\n")
                f.write(f"TRIPLET_MINING_NEGATIVE_SELECTION_BIAS_BETA: {self.TRIPLET_MINING_NEGATIVE_SELECTION_BIAS_BETA}\n")
                f.write(f"TRIPLET_MINING_NEGATIVE_SELECTION_BIAS_EPS: {self.TRIPLET_MINING_NEGATIVE_SELECTION_BIAS_EPS}\n")
                f.write(f"TRIPLET_NEGATIVES_PER_AP: {self.TRIPLET_NEGATIVES_PER_AP}\n\n")

                f.write("# Triplets ------------------\n")
                f.write(f"MANUAL_TRIPLET_MARGINS: {self.MANUAL_TRIPLET_MARGINS}\n")
                f.write(f"MANUAL_TRIPLET_MARGINS_PER_RANK: {self.MANUAL_TRIPLET_MARGINS_PER_RANK}\n")
                f.write(f"TRIPLET_MARGIN_EPSILON: {self.TRIPLET_MARGIN_EPSILON}\n")
                f.write("\n")

                f.write("# Relative squared error ------------------\n")
                f.write(f"RELATIVE_ERROR_EPSILONS_PAIR_LOSS: {self.RELATIVE_ERROR_EPSILONS_PAIR_LOSS}\n")
                f.write(f"RELATIVE_ERROR_EPSILONS_PAIR_MINING: {self.RELATIVE_ERROR_EPSILONS_PAIR_MINING}\n")
                f.write(f"PAIR_RELATIVE_LOSS_CAP: {self.PAIR_RELATIVE_LOSS_CAP}\n")
                f.write(f"RELATIVE_ERROR_EPSILONS_TRIPLET_LOSS: {self.RELATIVE_ERROR_EPSILONS_TRIPLET_LOSS}\n")
                f.write(f"TRIPLET_RELATIVE_LOSS_CAP: {self.TRIPLET_RELATIVE_LOSS_CAP}\n\n")

                f.write("# Sequence variants ------------------\n")
                f.write(f"USE_FULL_SEQS: {self.USE_FULL_SEQS}\n")
                f.write(f"USE_SUB_SEQS: {self.USE_SUB_SEQS}\n")
                f.write(f"MUTATION_RATE: {self.MUTATION_RATE}\n")
                f.write(f"PROP_TRUNC: {self.PROP_TRUNC}\n")
                f.write(f"MIN_TRUNC_START: {self.MIN_TRUNC_START}\n")
                f.write(f"MAX_TRUNC_START: {self.MAX_TRUNC_START}\n")
                f.write(f"MIN_TRUNC_END: {self.MIN_TRUNC_END}\n")
                f.write(f"MAX_TRUNC_END: {self.MAX_TRUNC_END}\n")
                f.write(f"PROP_SHIFT_SEQS: {self.PROP_SHIFT_SEQS}\n\n")

                f.write("# Validation ------------------\n")
                f.write(f"SAVE_EVERY_N_BATCHES: {self.SAVE_EVERY_N_BATCHES}\n")

                f.write("# Loss recording ------------------\n")
                f.write(f"RECORD_LOSS_EVERY_N_BATCHES: {self.RECORD_LOSS_EVERY_N_BATCHES}\n")
                f.write(f"PLOT_LOSS_EVERY_N_BATCHES: {self.PLOT_LOSS_EVERY_N_BATCHES}\n")
                f.write(f"PLOT_TRIPLET_LOSS: {self.PLOT_TRIPLET_LOSS}\n")
                f.write(f"PLOT_PAIR_LOSS: {self.PLOT_PAIR_LOSS}\n")
                f.write(f"PLOT_UNCERTAINTY_WEIGHTING: {self.PLOT_UNCERTAINTY_WEIGHTING}\n")
                f.write(f"SECOND_LOSS_PLOT_AFTER_N_BATCHES: {self.SECOND_LOSS_PLOT_AFTER_N_BATCHES}\n")
                f.write(f"SECOND_TRIPLET_SATISFACTION_PLOT_AFTER_N_BATCHES: {self.SECOND_TRIPLET_SATISFACTION_PLOT_AFTER_N_BATCHES}\n")
                f.write(f"SECOND_PAIR_ERROR_METRICS_PLOT_AFTER_N_BATCHES: {self.SECOND_PAIR_ERROR_METRICS_PLOT_AFTER_N_BATCHES}\n")
                f.write(f"SECOND_TRIPLET_ERROR_METRICS_PLOT_AFTER_N_BATCHES: {self.SECOND_TRIPLET_ERROR_METRICS_PLOT_AFTER_N_BATCHES}\n")
                f.write(f"PLOT_CONV_STEM_SCALE_EVERY_N_BATCHES: {self.PLOT_CONV_STEM_SCALE_EVERY_N_BATCHES}\n")
                f.write(f"SECOND_CONV_STEM_SCALE_PLOT_AFTER_N_BATCHES: {self.SECOND_CONV_STEM_SCALE_PLOT_AFTER_N_BATCHES}\n\n")

                f.write("# Quick test ------------------\n")
                f.write(f"DO_QT: {self.DO_QT}\n")
                f.write(f"QT_BATCH_SIZE: {self.QT_BATCH_SIZE}\n")
                f.write(f"QT_EVERY_N_BATCHES: {self.QT_EVERY_N_BATCHES}\n")
                f.write(f"QT_BEFORE_FIRST_BATCH: {self.QT_BEFORE_FIRST_BATCH}\n")
                f.write(f"QT_KMER_K_VALUES: {self.QT_KMER_K_VALUES}\n")
                f.write(f"QT_GET_KMER_RESULTS: {self.QT_GET_KMER_RESULTS}\n")
                f.write(f"QT_PCOA_RANKS: {self.QT_PCOA_RANKS}\n")
                f.write(f"QT_PCOA_EVERY_N_BATCHES: {self.QT_PCOA_EVERY_N_BATCHES}\n")
                f.write(f"QT_UMAP_RANKS: {self.QT_UMAP_RANKS}\n")
                f.write(f"QT_UMAP_EVERY_N_BATCHES: {self.QT_UMAP_EVERY_N_BATCHES}\n")
                f.write(f"QT_CLUSTERING_RANKS: {self.QT_CLUSTERING_RANKS}\n")
                f.write(f"QT_RANK_LEVEL_CLUSTERING_METRICS_AT_RANKS: {self.QT_RANK_LEVEL_CLUSTERING_METRICS_AT_RANKS}\n")
                f.write(f"QT_MAX_TAXA_PER_RANK_LEVEL_CLUSTERING: {self.QT_MAX_TAXA_PER_RANK_LEVEL_CLUSTERING}\n")
                f.write(f"RANK_LEVEL_CLUSTERING_PLOTS_EVERY_N_BATCHES: {self.RANK_LEVEL_CLUSTERING_PLOTS_EVERY_N_BATCHES}\n")
                f.write(f"QT_CLASSIFICATION_RANKS: {self.QT_CLASSIFICATION_RANKS}\n")
                f.write(f"QT_RECORD_MISS_CLASSIFICATION_SEQUENCES_AT_RANKS: {self.QT_RECORD_MISS_CLASSIFICATION_SEQUENCES_AT_RANKS}\n")
                f.write(f"QT_MAX_N_REGIONS: {self.QT_MAX_N_REGIONS}\n")
                f.write(f"QT_PROP_SHIFT_SEQS: {self.QT_PROP_SHIFT_SEQS}\n")
                f.write(f"QT_MUTATION_RATE: {self.QT_MUTATION_RATE}\n")
                f.write(f"QT_PROP_TRUNC: {self.QT_PROP_TRUNC}\n")
                f.write(f"QT_MIN_TRUNC_START: {self.QT_MIN_TRUNC_START}\n")
                f.write(f"QT_MAX_TRUNC_START: {self.QT_MAX_TRUNC_START}\n")
                f.write(f"QT_MIN_TRUNC_END: {self.QT_MIN_TRUNC_END}\n")
                f.write(f"QT_MAX_TRUNC_END: {self.QT_MAX_TRUNC_END}\n")
                f.write(f"QT_VERBOSE: {self.QT_VERBOSE}\n")
                f.write(f"QT_PRINT_RESULTS: {self.QT_PRINT_RESULTS}\n")
                f.write("# Verbose printing and logging ---\n")
                f.write(f"VERBOSE_TRAINING_TIMING: {self.VERBOSE_TRAINING_TIMING}\n")
                f.write(f"VERBOSE_TRIPLET_LOSS: {self.VERBOSE_TRIPLET_LOSS}\n")
                f.write(f"VERBOSE_PAIR_LOSS: {self.VERBOSE_PAIR_LOSS}\n")
                f.write(f"VERBOSE_PAIR_MINING: {self.VERBOSE_PAIR_MINING}\n")
                f.write(f"VERBOSE_TRIPLET_MINING: {self.VERBOSE_TRIPLET_MINING}\n")
                f.write(f"VERBOSE_EVERY_N_BATCHES: {self.VERBOSE_EVERY_N_BATCHES}\n")
                f.write(f"VERBOSE_MINING_TIMING: {self.VERBOSE_MINING_TIMING}\n")
                f.write(f"VERBOSE_UNCERTAINTY_WEIGHTING: {self.VERBOSE_UNCERTAINTY_WEIGHTING}\n")
                f.write(f"VERBOSE_CONV_STEM_SCALE: {self.VERBOSE_CONV_STEM_SCALE}\n")
                f.write(f"LOG_TRIPLET_LOSS: {self.LOG_TRIPLET_LOSS}\n")
                f.write(f"LOG_PAIR_LOSS: {self.LOG_PAIR_LOSS}\n")
                f.write(f"LOG_PAIR_MINING: {self.LOG_PAIR_MINING}\n")
                f.write(f"LOG_TRIPLET_MINING: {self.LOG_TRIPLET_MINING}\n")
                f.write(f"LOG_MINING: {self.LOG_MINING}\n")
                f.write(f"LOG_MINING_BUCKET_THRESHOLDS: {self.LOG_MINING_BUCKET_THRESHOLDS}\n")
                f.write(f"LOG_MINING_PHYLA_COUNTS: {self.LOG_MINING_PHYLA_COUNTS}\n")
                f.write(f"MINING_PHYLA_LOG_MAX_LINES_PER_TABLE: {self.MINING_PHYLA_LOG_MAX_LINES_PER_TABLE}\n")
                f.write(f"LOG_MINING_ARC_BAC_COUNTS: {self.LOG_MINING_ARC_BAC_COUNTS}\n")
                f.write(f"LOG_UNCERTAINTY_WEIGHTING: {self.LOG_UNCERTAINTY_WEIGHTING}\n")
                f.write(f"LOG_CONV_STEM_SCALE: {self.LOG_CONV_STEM_SCALE}\n")
                f.write(f"LOG_EVERY_N_BATCHES: {self.LOG_EVERY_N_BATCHES}\n")
                f.write(f"PRINT_BATCH_NUM_EVERY_N_BATCHES: {self.PRINT_BATCH_NUM_EVERY_N_BATCHES}\n")
                f.write(f"QT_TESTS_TODO:\n")
                for test in sorted(list(self.QT_TESTS_TODO)):
                    f.write(f"  - {test}\n")
                f.write("\n")

                f.write("# General Constants ------------------\n")
                f.write(f"SEQ_3BIT_REPRESENTATION_DICT: {self.SEQ_3BIT_REPRESENTATION_DICT}\n")

        if json:
            config_dict = {attr: getattr(self, attr) for attr in dir(self) if not callable(getattr(self, attr)) and not attr.startswith("__") and attr.isupper()}
            # Convert sets to lists for JSON serialization
            for key, value in config_dict.items():
                if isinstance(value, set):
                    config_dict[key] = list(value)
                # Convert dicts with int keys to string keys for JSON
                elif isinstance(value, dict):
                    if any(isinstance(k, int) for k in value.keys()):
                        config_dict[key] = {str(k): v for k, v in value.items()}

            with open(f"{model_dir}/training_config.json", "w") as f:
                json_lib.dump(config_dict, f, indent=4)

    def load_training_config(self, json_path):
        with open(json_path, 'r') as f:
            config_dict = json_lib.load(f)
        if 'SHIFT_SEQS' in config_dict and 'PROP_SHIFT_SEQS' not in config_dict:
            legacy_val = config_dict.pop('SHIFT_SEQS')
            config_dict['PROP_SHIFT_SEQS'] = 1.0 if legacy_val else 0.0
        if 'QT_SHIFT_SEQS' in config_dict and 'QT_PROP_SHIFT_SEQS' not in config_dict:
            legacy_val = config_dict.pop('QT_SHIFT_SEQS')
            config_dict['QT_PROP_SHIFT_SEQS'] = 1.0 if legacy_val else 0.0
        if 'REPRESENTATIVE_TAXON_SIZE_BALANCE_LAMBDA' in config_dict:
            legacy_lambda = config_dict.pop('REPRESENTATIVE_TAXON_SIZE_BALANCE_LAMBDA')
            if 'PAIR_REPRESENTATIVE_TAXON_SIZE_BALANCE_LAMBDA' not in config_dict:
                config_dict['PAIR_REPRESENTATIVE_TAXON_SIZE_BALANCE_LAMBDA'] = legacy_lambda
            if 'TRIPLET_REPRESENTATIVE_TAXON_SIZE_BALANCE_LAMBDA' not in config_dict:
                config_dict['TRIPLET_REPRESENTATIVE_TAXON_SIZE_BALANCE_LAMBDA'] = legacy_lambda
        for key, value in config_dict.items():
            # Handle potential tuple conversion for lists read from JSON
            if isinstance(getattr(self, key, None), tuple) and isinstance(value, list):
                 setattr(self, key, tuple(value))
            # Handle potential set conversion for lists read from JSON (e.g., QT_TESTS_TODO)
            elif isinstance(getattr(self, key, None), set) and isinstance(value, list):
                 setattr(self, key, set(value))
            # Handle dicts with integer keys (like MARGINS_PER_RANK)
            elif isinstance(getattr(self, key, None), dict) and isinstance(value, dict):
                original_dict = getattr(self, key)
                new_dict = {int(k) if k.isdigit() else k: v for k, v in value.items()}
                setattr(self, key, new_dict)
            # Default case
            else:
                 setattr(self, key, value)

        self._validate_pooling_type()
        self._validate_start_from_ckpt()
        self._validate_max_num_subseqs()
        self._validate_per_rank_mining_params()
        self._validate_logging_params()
        self._validate_quick_test_params()
        self._validate_lr_scheduler_config()
        self._validate_taxa_cap()

    def _validate_start_from_ckpt(self):
        """Ensure START_FROM_CKPT is False or a .pth path string."""
        if self.START_FROM_CKPT is False:
            return
        if not isinstance(self.START_FROM_CKPT, str):
            raise ValueError("START_FROM_CKPT must be False or a string path to a .pth file.")
        if not self.START_FROM_CKPT.endswith(".pth"):
            raise ValueError(f"START_FROM_CKPT must point to a .pth file, got: {self.START_FROM_CKPT}")

    def _validate_lr_scheduler_config(self):
        """Validate LR scheduler type and kwargs."""
        allowed_scheduler_types = {
            'CosineAnnealingWarmRestarts',
            'OneCycleLR',
            'StepLR',
            'LinearWarmupToLearningRate'
        }

        if self.LR_SCHEDULER_TYPE is None:
            if self.LR_SCHEDULER_KWARGS is not None:
                raise ValueError("LR_SCHEDULER_KWARGS must be None when LR_SCHEDULER_TYPE is None.")
            return

        if self.LR_SCHEDULER_TYPE not in allowed_scheduler_types:
            raise ValueError(
                f"Unknown LR_SCHEDULER_TYPE: {self.LR_SCHEDULER_TYPE}. "
                f"Expected one of: {sorted(allowed_scheduler_types)}"
            )

        if not isinstance(self.LR_SCHEDULER_KWARGS, dict):
            raise ValueError("LR_SCHEDULER_KWARGS must be a dict when LR_SCHEDULER_TYPE is set.")

        if self.LR_SCHEDULER_TYPE == 'CosineAnnealingWarmRestarts':
            if 'T_0' not in self.LR_SCHEDULER_KWARGS:
                raise ValueError("CosineAnnealingWarmRestarts requires LR_SCHEDULER_KWARGS['T_0'].")
            if not isinstance(self.LR_SCHEDULER_KWARGS['T_0'], int) or self.LR_SCHEDULER_KWARGS['T_0'] < 1:
                raise ValueError(f"Expected integer T_0 >= 1, but got {self.LR_SCHEDULER_KWARGS['T_0']}")
            if 'T_mult' in self.LR_SCHEDULER_KWARGS:
                if not isinstance(self.LR_SCHEDULER_KWARGS['T_mult'], int) or self.LR_SCHEDULER_KWARGS['T_mult'] < 1:
                    raise ValueError(f"Expected integer T_mult >= 1, but got {self.LR_SCHEDULER_KWARGS['T_mult']}")
            if 'eta_min' in self.LR_SCHEDULER_KWARGS:
                eta_min = self.LR_SCHEDULER_KWARGS['eta_min']
                if not isinstance(eta_min, (int, float)) or eta_min < 0:
                    raise ValueError(f"Expected numeric eta_min >= 0, but got {eta_min}")

        elif self.LR_SCHEDULER_TYPE == 'OneCycleLR':
            if 'max_lr' not in self.LR_SCHEDULER_KWARGS:
                raise ValueError("OneCycleLR requires LR_SCHEDULER_KWARGS['max_lr'].")
            max_lr = self.LR_SCHEDULER_KWARGS['max_lr']
            if isinstance(max_lr, (list, tuple)):
                if len(max_lr) == 0 or any((not isinstance(v, (int, float)) or v <= 0) for v in max_lr):
                    raise ValueError("OneCycleLR max_lr list/tuple must contain only positive numbers.")
            elif not isinstance(max_lr, (int, float)) or max_lr <= 0:
                raise ValueError(f"OneCycleLR max_lr must be a positive number, got {max_lr}.")
            if 'pct_start' in self.LR_SCHEDULER_KWARGS:
                pct_start = self.LR_SCHEDULER_KWARGS['pct_start']
                if not isinstance(pct_start, (int, float)) or not (0 < pct_start <= 1):
                    raise ValueError(f"OneCycleLR pct_start must be in (0,1], got {pct_start}.")
            if 'div_factor' in self.LR_SCHEDULER_KWARGS:
                div_factor = self.LR_SCHEDULER_KWARGS['div_factor']
                if not isinstance(div_factor, (int, float)) or div_factor <= 0:
                    raise ValueError(f"OneCycleLR div_factor must be > 0, got {div_factor}.")

        elif self.LR_SCHEDULER_TYPE == 'StepLR':
            if 'step_size' not in self.LR_SCHEDULER_KWARGS:
                raise ValueError("StepLR requires LR_SCHEDULER_KWARGS['step_size'].")
            if not isinstance(self.LR_SCHEDULER_KWARGS['step_size'], int) or self.LR_SCHEDULER_KWARGS['step_size'] < 1:
                raise ValueError(f"StepLR step_size must be an integer >= 1, got {self.LR_SCHEDULER_KWARGS['step_size']}.")
            if 'gamma' in self.LR_SCHEDULER_KWARGS:
                gamma = self.LR_SCHEDULER_KWARGS['gamma']
                if not isinstance(gamma, (int, float)) or gamma <= 0:
                    raise ValueError(f"StepLR gamma must be > 0, got {gamma}.")

        elif self.LR_SCHEDULER_TYPE == 'LinearWarmupToLearningRate':
            required_keys = {'start_lr', 'warmup_batches'}
            missing = required_keys - set(self.LR_SCHEDULER_KWARGS.keys())
            if missing:
                raise ValueError(f"LinearWarmupToLearningRate missing required LR_SCHEDULER_KWARGS keys: {sorted(missing)}.")
            start_lr = self.LR_SCHEDULER_KWARGS['start_lr']
            warmup_batches = self.LR_SCHEDULER_KWARGS['warmup_batches']
            if not isinstance(start_lr, (int, float)) or start_lr < 0:
                raise ValueError(f"LinearWarmupToLearningRate start_lr must be a number >= 0, got {start_lr}.")
            if not isinstance(warmup_batches, int) or warmup_batches < 1:
                raise ValueError(f"LinearWarmupToLearningRate warmup_batches must be an integer >= 1, got {warmup_batches}.")

    def _validate_pooling_type(self):
        """Ensure POOLING_TYPE is recognised and lowercase."""
        if not isinstance(self.POOLING_TYPE, str):
            raise ValueError("POOLING_TYPE must be a string.")
        pooling_type = self.POOLING_TYPE.lower()
        if pooling_type not in ('mean', 'attention'):
            raise ValueError(f"POOLING_TYPE must be 'mean' or 'attention', got {self.POOLING_TYPE}")
        self.POOLING_TYPE = pooling_type

    def _validate_max_num_subseqs(self):
        """Ensure MAX_NUM_SUBSEQS is None or a positive integer."""
        if self.MAX_NUM_SUBSEQS is None:
            return
        if not isinstance(self.MAX_NUM_SUBSEQS, int):
            raise ValueError("MAX_NUM_SUBSEQS must be an integer or None.")
        if self.MAX_NUM_SUBSEQS < 1:
            raise ValueError("MAX_NUM_SUBSEQS must be >= 1 when provided.")

    def _validate_logging_params(self):
        """Validate logging parameters."""
        if not isinstance(self.LOG_MINING_PHYLA_COUNTS, bool):
            raise ValueError("LOG_MINING_PHYLA_COUNTS must be a bool.")
        if not isinstance(self.MINING_PHYLA_LOG_MAX_LINES_PER_TABLE, int) or self.MINING_PHYLA_LOG_MAX_LINES_PER_TABLE < 1:
            raise ValueError("MINING_PHYLA_LOG_MAX_LINES_PER_TABLE must be an integer >= 1.")

    def _validate_quick_test_params(self):
        """Validate quick-test configuration parameters."""
        if not isinstance(self.QT_RECORD_MISS_CLASSIFICATION_SEQUENCES_AT_RANKS, (list, tuple)):
            raise ValueError("QT_RECORD_MISS_CLASSIFICATION_SEQUENCES_AT_RANKS must be a list or tuple.")
        if len(self.QT_RECORD_MISS_CLASSIFICATION_SEQUENCES_AT_RANKS) != 7:
            raise ValueError("QT_RECORD_MISS_CLASSIFICATION_SEQUENCES_AT_RANKS must have length 7 (domain-species).")
        for idx, val in enumerate(self.QT_RECORD_MISS_CLASSIFICATION_SEQUENCES_AT_RANKS):
            if not isinstance(val, bool):
                raise ValueError(
                    f"QT_RECORD_MISS_CLASSIFICATION_SEQUENCES_AT_RANKS[{idx}] must be a bool, got {type(val).__name__}."
                )

    def _validate_rank_introduction_config(self, check_batch_zero_enabled=True):
        """Validate PAIR_RANKS/TRIPLET_RANKS and INTRODUCE_RANK_AT_BATCHES."""
        if not isinstance(self.PAIR_RANKS, (list, tuple)):
            raise ValueError("PAIR_RANKS must be a list or tuple.")
        if len(self.PAIR_RANKS) != 9:
            raise ValueError(f"PAIR_RANKS must have length 9 (domain-subsequence), got {len(self.PAIR_RANKS)}.")
        for idx, enabled in enumerate(self.PAIR_RANKS):
            if not isinstance(enabled, bool):
                raise ValueError(f"PAIR_RANKS[{idx}] must be a bool, got {type(enabled).__name__}.")

        if not isinstance(self.TRIPLET_RANKS, (list, tuple)):
            raise ValueError("TRIPLET_RANKS must be a list or tuple.")
        if len(self.TRIPLET_RANKS) != 6:
            raise ValueError(f"TRIPLET_RANKS must have length 6 (domain-genus), got {len(self.TRIPLET_RANKS)}.")
        for idx, enabled in enumerate(self.TRIPLET_RANKS):
            if not isinstance(enabled, bool):
                raise ValueError(f"TRIPLET_RANKS[{idx}] must be a bool, got {type(enabled).__name__}.")

        if not isinstance(self.INTRODUCE_RANK_AT_BATCHES, (list, tuple)):
            raise ValueError("INTRODUCE_RANK_AT_BATCHES must be a list or tuple.")
        if len(self.INTRODUCE_RANK_AT_BATCHES) != 9:
            raise ValueError(
                "INTRODUCE_RANK_AT_BATCHES must have length 9 (domain-subsequence), "
                f"got {len(self.INTRODUCE_RANK_AT_BATCHES)}."
            )
        for idx, batch_num in enumerate(self.INTRODUCE_RANK_AT_BATCHES):
            if isinstance(batch_num, bool) or not isinstance(batch_num, int):
                raise ValueError(
                    f"INTRODUCE_RANK_AT_BATCHES[{idx}] must be an integer batch number, got {type(batch_num).__name__}."
                )
            if batch_num < 0:
                raise ValueError(
                    f"INTRODUCE_RANK_AT_BATCHES[{idx}] must be >= 0, got {batch_num}."
                )

        if not check_batch_zero_enabled:
            return

        # Keep mining viable from the first batch: among objectives that are configured
        # to contribute to each batch, at least one enabled rank must be introduced at batch 0.
        has_pair_rank_at_zero = (
            self.N_PAIRS_PER_BATCH > 0 and any(
                self.PAIR_RANKS[idx] and self.INTRODUCE_RANK_AT_BATCHES[idx] == 0
                for idx in range(9)
            )
        )
        has_triplet_rank_at_zero = (
            self.N_TRIPLETS_PER_BATCH > 0 and any(
                self.TRIPLET_RANKS[idx] and self.INTRODUCE_RANK_AT_BATCHES[idx] == 0
                for idx in range(6)
            )
        )
        if not (has_pair_rank_at_zero or has_triplet_rank_at_zero):
            raise ValueError(
                "When pair/triplet mining is enabled, at least one enabled pair or triplet rank "
                "must be introduced at batch 0 (INTRODUCE_RANK_AT_BATCHES[idx] == 0) so mining "
                "has an active objective from the first batch."
            )

    def _validate_per_rank_mining_params(self):
        """Validate per-rank mining configuration parameters."""
        if not isinstance(self.SHUFFLE_TRIPLETS_AND_PAIRS_AFTER_MINING, bool):
            raise ValueError("SHUFFLE_TRIPLETS_AND_PAIRS_AFTER_MINING must be a bool.")

        # Validate taxon-size baseline stats
        if not isinstance(self.TAXON_SIZE_MINING_BIAS_BASELINE_STAT, str):
            raise ValueError("TAXON_SIZE_MINING_BIAS_BASELINE_STAT must be a string.")
        baseline_stat = self.TAXON_SIZE_MINING_BIAS_BASELINE_STAT.lower()
        if baseline_stat not in ("median", "mean", "percentile"):
            raise ValueError(f"TAXON_SIZE_MINING_BIAS_BASELINE_STAT must be 'median', 'mean', or 'percentile', got {self.TAXON_SIZE_MINING_BIAS_BASELINE_STAT}.")
        self.TAXON_SIZE_MINING_BIAS_BASELINE_STAT = baseline_stat
        if baseline_stat == "percentile":
            if not isinstance(self.TAXON_SIZE_MINING_BIAS_BASELINE_PERCENTILE, (int, float)):
                raise ValueError("TAXON_SIZE_MINING_BIAS_BASELINE_PERCENTILE must be a number.")
            if not (0.0 <= float(self.TAXON_SIZE_MINING_BIAS_BASELINE_PERCENTILE) <= 100.0):
                raise ValueError("TAXON_SIZE_MINING_BIAS_BASELINE_PERCENTILE must be in [0, 100].")
        
        # Validate representative-set taxon-size balancing config
        if not isinstance(self.USE_REPRESENTATIVE_TAXON_SIZE_BALANCING, bool):
            raise ValueError("USE_REPRESENTATIVE_TAXON_SIZE_BALANCING must be a bool.")
        if not isinstance(self.PAIR_REPRESENTATIVE_TAXON_SIZE_BALANCE_LAMBDA, (int, float)):
            raise ValueError("PAIR_REPRESENTATIVE_TAXON_SIZE_BALANCE_LAMBDA must be a number.")
        if not (0.0 <= float(self.PAIR_REPRESENTATIVE_TAXON_SIZE_BALANCE_LAMBDA) <= 1.0):
            raise ValueError(f"PAIR_REPRESENTATIVE_TAXON_SIZE_BALANCE_LAMBDA must be in [0, 1], got {self.PAIR_REPRESENTATIVE_TAXON_SIZE_BALANCE_LAMBDA}.")
        if not isinstance(self.TRIPLET_REPRESENTATIVE_TAXON_SIZE_BALANCE_LAMBDA, (int, float)):
            raise ValueError("TRIPLET_REPRESENTATIVE_TAXON_SIZE_BALANCE_LAMBDA must be a number.")
        if not (0.0 <= float(self.TRIPLET_REPRESENTATIVE_TAXON_SIZE_BALANCE_LAMBDA) <= 1.0):
            raise ValueError(f"TRIPLET_REPRESENTATIVE_TAXON_SIZE_BALANCE_LAMBDA must be in [0, 1], got {self.TRIPLET_REPRESENTATIVE_TAXON_SIZE_BALANCE_LAMBDA}.")
        if not isinstance(self.REPRESENTATIVE_TAXON_SIZE_BALANCE_EPS, (int, float)):
            raise ValueError("REPRESENTATIVE_TAXON_SIZE_BALANCE_EPS must be a number.")
        if self.REPRESENTATIVE_TAXON_SIZE_BALANCE_EPS <= 0.0:
            raise ValueError(f"REPRESENTATIVE_TAXON_SIZE_BALANCE_EPS must be > 0, got {self.REPRESENTATIVE_TAXON_SIZE_BALANCE_EPS}.")
        if self.REPRESENTATIVE_TAXON_SIZE_BALANCE_WEIGHT_CLIP is not None:
            if not isinstance(self.REPRESENTATIVE_TAXON_SIZE_BALANCE_WEIGHT_CLIP, (int, float)):
                raise ValueError("REPRESENTATIVE_TAXON_SIZE_BALANCE_WEIGHT_CLIP must be a number or None.")
            if self.REPRESENTATIVE_TAXON_SIZE_BALANCE_WEIGHT_CLIP <= 0.0:
                raise ValueError(f"REPRESENTATIVE_TAXON_SIZE_BALANCE_WEIGHT_CLIP must be > 0, got {self.REPRESENTATIVE_TAXON_SIZE_BALANCE_WEIGHT_CLIP}.")

        # Validate taxon-size mining bias config
        if not isinstance(self.USE_TAXON_SIZE_MINING_BIAS, bool):
            raise ValueError("USE_TAXON_SIZE_MINING_BIAS must be a bool.")
        if not isinstance(self.TAXON_SIZE_MINING_BIAS_ALPHAS_PER_RANK, (list, tuple)):
            raise ValueError("TAXON_SIZE_MINING_BIAS_ALPHAS_PER_RANK must be a list or tuple.")
        if len(self.TAXON_SIZE_MINING_BIAS_ALPHAS_PER_RANK) != 7:
            raise ValueError(f"TAXON_SIZE_MINING_BIAS_ALPHAS_PER_RANK must have length 7 (domain-species), got {len(self.TAXON_SIZE_MINING_BIAS_ALPHAS_PER_RANK)}.")
        for idx, val in enumerate(self.TAXON_SIZE_MINING_BIAS_ALPHAS_PER_RANK):
            if not isinstance(val, (int, float)) or val < 0.0:
                raise ValueError(f"TAXON_SIZE_MINING_BIAS_ALPHAS_PER_RANK[{idx}] must be >= 0, got {val}.")
        if self.TAXON_SIZE_MINING_BIAS_MIN_KEEP_PER_RANK is not None:
            if not isinstance(self.TAXON_SIZE_MINING_BIAS_MIN_KEEP_PER_RANK, (list, tuple)):
                raise ValueError("TAXON_SIZE_MINING_BIAS_MIN_KEEP_PER_RANK must be a list, tuple, or None.")
            if len(self.TAXON_SIZE_MINING_BIAS_MIN_KEEP_PER_RANK) != 7:
                raise ValueError(f"TAXON_SIZE_MINING_BIAS_MIN_KEEP_PER_RANK must have length 7 (domain-species), got {len(self.TAXON_SIZE_MINING_BIAS_MIN_KEEP_PER_RANK)}.")
            for idx, val in enumerate(self.TAXON_SIZE_MINING_BIAS_MIN_KEEP_PER_RANK):
                if not isinstance(val, (int, float)) or not (0.0 <= val <= 1.0):
                    raise ValueError(f"TAXON_SIZE_MINING_BIAS_MIN_KEEP_PER_RANK[{idx}] must be in [0, 1], got {val}.")

        # Validate triplet positive selection bias
        if not isinstance(self.USE_TRIPLET_MINING_POSITIVE_SELECTION_BIAS, bool):
            raise ValueError("USE_TRIPLET_MINING_POSITIVE_SELECTION_BIAS must be a bool.")
        if not isinstance(self.TRIPLET_MINING_POSITIVE_SELECTION_BIAS_BETA, (int, float)):
            raise ValueError("TRIPLET_MINING_POSITIVE_SELECTION_BIAS_BETA must be a number.")
        if self.TRIPLET_MINING_POSITIVE_SELECTION_BIAS_BETA < 0.0:
            raise ValueError(f"TRIPLET_MINING_POSITIVE_SELECTION_BIAS_BETA must be >= 0, got {self.TRIPLET_MINING_POSITIVE_SELECTION_BIAS_BETA}.")
        if not isinstance(self.TRIPLET_MINING_POSITIVE_SELECTION_BIAS_EPS, (int, float)):
            raise ValueError("TRIPLET_MINING_POSITIVE_SELECTION_BIAS_EPS must be a number.")
        if self.TRIPLET_MINING_POSITIVE_SELECTION_BIAS_EPS <= 0.0:
            raise ValueError(f"TRIPLET_MINING_POSITIVE_SELECTION_BIAS_EPS must be > 0, got {self.TRIPLET_MINING_POSITIVE_SELECTION_BIAS_EPS}.")

        # Validate triplet negative selection bias
        if not isinstance(self.USE_TRIPLET_MINING_NEGATIVE_SELECTION_BIAS, bool):
            raise ValueError("USE_TRIPLET_MINING_NEGATIVE_SELECTION_BIAS must be a bool.")
        if not isinstance(self.TRIPLET_MINING_NEGATIVE_SELECTION_BIAS_BETA, (int, float)):
            raise ValueError("TRIPLET_MINING_NEGATIVE_SELECTION_BIAS_BETA must be a number.")
        if self.TRIPLET_MINING_NEGATIVE_SELECTION_BIAS_BETA < 0.0:
            raise ValueError(f"TRIPLET_MINING_NEGATIVE_SELECTION_BIAS_BETA must be >= 0, got {self.TRIPLET_MINING_NEGATIVE_SELECTION_BIAS_BETA}.")
        if not isinstance(self.TRIPLET_MINING_NEGATIVE_SELECTION_BIAS_EPS, (int, float)):
            raise ValueError("TRIPLET_MINING_NEGATIVE_SELECTION_BIAS_EPS must be a number.")
        if self.TRIPLET_MINING_NEGATIVE_SELECTION_BIAS_EPS <= 0.0:
            raise ValueError(f"TRIPLET_MINING_NEGATIVE_SELECTION_BIAS_EPS must be > 0, got {self.TRIPLET_MINING_NEGATIVE_SELECTION_BIAS_EPS}.")
        if not isinstance(self.TRIPLET_NEGATIVES_PER_AP, int):
            raise ValueError("TRIPLET_NEGATIVES_PER_AP must be an integer.")
        if self.TRIPLET_NEGATIVES_PER_AP < 1:
            raise ValueError(f"TRIPLET_NEGATIVES_PER_AP must be >= 1, got {self.TRIPLET_NEGATIVES_PER_AP}.")

        # Validate PAIR_MINING_REPRESENTATIVE_SET_SIZES (length 9)
        if not isinstance(self.PAIR_MINING_REPRESENTATIVE_SET_SIZES, (list, tuple)):
            raise ValueError("PAIR_MINING_REPRESENTATIVE_SET_SIZES must be a list or tuple.")
        if len(self.PAIR_MINING_REPRESENTATIVE_SET_SIZES) != 9:
            raise ValueError(f"PAIR_MINING_REPRESENTATIVE_SET_SIZES must have length 9 (domain-subsequence), got {len(self.PAIR_MINING_REPRESENTATIVE_SET_SIZES)}.")
        for idx, val in enumerate(self.PAIR_MINING_REPRESENTATIVE_SET_SIZES):
            if not isinstance(val, int) or val < 0:
                raise ValueError(f"PAIR_MINING_REPRESENTATIVE_SET_SIZES[{idx}] must be a non-negative integer, got {val}.")

        # Validate PAIR_SIGN_BIAS_BETA_PER_RANK (length 9)
        if not isinstance(self.PAIR_SIGN_BIAS_BETA_PER_RANK, (list, tuple)):
            raise ValueError("PAIR_SIGN_BIAS_BETA_PER_RANK must be a list or tuple.")
        if len(self.PAIR_SIGN_BIAS_BETA_PER_RANK) != 9:
            raise ValueError(
                f"PAIR_SIGN_BIAS_BETA_PER_RANK must have length 9 (domain-subsequence), got {len(self.PAIR_SIGN_BIAS_BETA_PER_RANK)}."
            )
        for idx, val in enumerate(self.PAIR_SIGN_BIAS_BETA_PER_RANK):
            if not isinstance(val, (int, float)) or not np.isfinite(val):
                raise ValueError(f"PAIR_SIGN_BIAS_BETA_PER_RANK[{idx}] must be a finite number, got {val}.")
        
        # Validate TRIPLET_MINING_REPRESENTATIVE_SET_SIZES (length 6)
        if not isinstance(self.TRIPLET_MINING_REPRESENTATIVE_SET_SIZES, (list, tuple)):
            raise ValueError("TRIPLET_MINING_REPRESENTATIVE_SET_SIZES must be a list or tuple.")
        if len(self.TRIPLET_MINING_REPRESENTATIVE_SET_SIZES) != 6:
            raise ValueError(f"TRIPLET_MINING_REPRESENTATIVE_SET_SIZES must have length 6 (domain-genus), got {len(self.TRIPLET_MINING_REPRESENTATIVE_SET_SIZES)}.")
        for idx, val in enumerate(self.TRIPLET_MINING_REPRESENTATIVE_SET_SIZES):
            if not isinstance(val, int) or val < 0:
                raise ValueError(f"TRIPLET_MINING_REPRESENTATIVE_SET_SIZES[{idx}] must be a non-negative integer, got {val}.")
        
        # Validate PAIR_MINING_EMA_ALPHA (must be in (0, 1])
        if not isinstance(self.PAIR_MINING_EMA_ALPHA, (int, float)):
            raise ValueError("PAIR_MINING_EMA_ALPHA must be a number.")
        if not (0.0 < self.PAIR_MINING_EMA_ALPHA <= 1.0):
            raise ValueError(f"PAIR_MINING_EMA_ALPHA must be in (0, 1], got {self.PAIR_MINING_EMA_ALPHA}.")
        # Validate PAIR_MINING_EMA_WEIGHT_EXPONENT (must be >= 0)
        if not isinstance(self.PAIR_MINING_EMA_WEIGHT_EXPONENT, (int, float)):
            raise ValueError("PAIR_MINING_EMA_WEIGHT_EXPONENT must be a number.")
        if self.PAIR_MINING_EMA_WEIGHT_EXPONENT < 0.0:
            raise ValueError(f"PAIR_MINING_EMA_WEIGHT_EXPONENT must be >= 0, got {self.PAIR_MINING_EMA_WEIGHT_EXPONENT}.")
        
        # Validate TRIPLET_MINING_EMA_ALPHA (must be in (0, 1])
        if not isinstance(self.TRIPLET_MINING_EMA_ALPHA, (int, float)):
            raise ValueError("TRIPLET_MINING_EMA_ALPHA must be a number.")
        if not (0.0 < self.TRIPLET_MINING_EMA_ALPHA <= 1.0):
            raise ValueError(f"TRIPLET_MINING_EMA_ALPHA must be in (0, 1], got {self.TRIPLET_MINING_EMA_ALPHA}.")
        # Validate TRIPLET_MINING_EMA_WEIGHT_EXPONENT (must be >= 0)
        if not isinstance(self.TRIPLET_MINING_EMA_WEIGHT_EXPONENT, (int, float)):
            raise ValueError("TRIPLET_MINING_EMA_WEIGHT_EXPONENT must be a number.")
        if self.TRIPLET_MINING_EMA_WEIGHT_EXPONENT < 0.0:
            raise ValueError(f"TRIPLET_MINING_EMA_WEIGHT_EXPONENT must be >= 0, got {self.TRIPLET_MINING_EMA_WEIGHT_EXPONENT}.")
        
        # Validate PAIR_PER_RANK_BATCH_PROPORTION_MAX (must be in (0, 1])
        if not isinstance(self.PAIR_PER_RANK_BATCH_PROPORTION_MAX, (int, float)):
            raise ValueError("PAIR_PER_RANK_BATCH_PROPORTION_MAX must be a number.")
        if not (0.0 < self.PAIR_PER_RANK_BATCH_PROPORTION_MAX <= 1.0):
            raise ValueError(f"PAIR_PER_RANK_BATCH_PROPORTION_MAX must be in (0, 1], got {self.PAIR_PER_RANK_BATCH_PROPORTION_MAX}.")
        
        # Validate TRIPLET_PER_RANK_BATCH_PROPORTION_MAX (must be in (0, 1])
        if not isinstance(self.TRIPLET_PER_RANK_BATCH_PROPORTION_MAX, (int, float)):
            raise ValueError("TRIPLET_PER_RANK_BATCH_PROPORTION_MAX must be a number.")
        if not (0.0 < self.TRIPLET_PER_RANK_BATCH_PROPORTION_MAX <= 1.0):
            raise ValueError(f"TRIPLET_PER_RANK_BATCH_PROPORTION_MAX must be in (0, 1], got {self.TRIPLET_PER_RANK_BATCH_PROPORTION_MAX}.")

        # Validate PAIR_PER_RANK_BATCH_PROPORTION_MIN (must be in [0, 1])
        if not isinstance(self.PAIR_PER_RANK_BATCH_PROPORTION_MIN, (int, float)):
            raise ValueError("PAIR_PER_RANK_BATCH_PROPORTION_MIN must be a number.")
        if not (0.0 <= self.PAIR_PER_RANK_BATCH_PROPORTION_MIN <= 1.0):
            raise ValueError(f"PAIR_PER_RANK_BATCH_PROPORTION_MIN must be in [0, 1], got {self.PAIR_PER_RANK_BATCH_PROPORTION_MIN}.")
        
        # Validate TRIPLET_PER_RANK_BATCH_PROPORTION_MIN (must be in [0, 1])
        if not isinstance(self.TRIPLET_PER_RANK_BATCH_PROPORTION_MIN, (int, float)):
            raise ValueError("TRIPLET_PER_RANK_BATCH_PROPORTION_MIN must be a number.")
        if not (0.0 <= self.TRIPLET_PER_RANK_BATCH_PROPORTION_MIN <= 1.0):
            raise ValueError(f"TRIPLET_PER_RANK_BATCH_PROPORTION_MIN must be in [0, 1], got {self.TRIPLET_PER_RANK_BATCH_PROPORTION_MIN}.")
        
        # Validate MIN <= MAX for both pair and triplet mining
        if self.PAIR_PER_RANK_BATCH_PROPORTION_MIN > self.PAIR_PER_RANK_BATCH_PROPORTION_MAX:
            raise ValueError(f"PAIR_PER_RANK_BATCH_PROPORTION_MIN ({self.PAIR_PER_RANK_BATCH_PROPORTION_MIN}) must be <= "
                             f"PAIR_PER_RANK_BATCH_PROPORTION_MAX ({self.PAIR_PER_RANK_BATCH_PROPORTION_MAX}).")
        if self.TRIPLET_PER_RANK_BATCH_PROPORTION_MIN > self.TRIPLET_PER_RANK_BATCH_PROPORTION_MAX:
            raise ValueError(f"TRIPLET_PER_RANK_BATCH_PROPORTION_MIN ({self.TRIPLET_PER_RANK_BATCH_PROPORTION_MIN}) must be <= "
                             f"TRIPLET_PER_RANK_BATCH_PROPORTION_MAX ({self.TRIPLET_PER_RANK_BATCH_PROPORTION_MAX}).")
        
        # Warn if MIN + 0.1 <= MAX for both pair and triplet mining
        if self.PAIR_PER_RANK_BATCH_PROPORTION_MIN + 0.1 > self.PAIR_PER_RANK_BATCH_PROPORTION_MAX:
            raise ValueError(f"PAIR_PER_RANK_BATCH_PROPORTION_MIN ({self.PAIR_PER_RANK_BATCH_PROPORTION_MIN}) + 0.1 is recommended be <= "
                             f"PAIR_PER_RANK_BATCH_PROPORTION_MAX ({self.PAIR_PER_RANK_BATCH_PROPORTION_MAX}).")
        if self.TRIPLET_PER_RANK_BATCH_PROPORTION_MIN + 0.1 > self.TRIPLET_PER_RANK_BATCH_PROPORTION_MAX:
            raise ValueError(f"TRIPLET_PER_RANK_BATCH_PROPORTION_MIN ({self.TRIPLET_PER_RANK_BATCH_PROPORTION_MIN}) + 0.1 is recommended be <= "
                             f"TRIPLET_PER_RANK_BATCH_PROPORTION_MAX ({self.TRIPLET_PER_RANK_BATCH_PROPORTION_MAX}).")

    def _validate_taxa_cap(self):
        """Validate TAXA_CAP_* configuration parameters."""
        if self.TAXA_CAP_MAX_NUM_TAXA is not None:
            if not isinstance(self.TAXA_CAP_MAX_NUM_TAXA, int) or self.TAXA_CAP_MAX_NUM_TAXA < 1:
                raise ValueError("TAXA_CAP_MAX_NUM_TAXA must be a positive integer or None.")
        if not isinstance(self.TAXA_CAP_RANK, int) or not (0 <= self.TAXA_CAP_RANK <= 5):
            raise ValueError("TAXA_CAP_RANK must be an integer from 0 to 5 (domain to genus).")
        if not isinstance(self.TAXA_CAP_VERBOSE, bool):
            raise ValueError("TAXA_CAP_VERBOSE must be a boolean.")


# ================================================
# Set Training Config as Globals
# ================================================

def set_train_config_as_globals(tc: TrainingConfig):
    """
    This sets all training configurations as global variables
    
    The idea behind TrainingConfig is that running tc = TrainingConfig() uses the default values,
    then we can easily alter particular parameters as we please before giving the TrainingConfig object
    to the set_train_config_as_globals function.
    """

    gb.DESCRIPTION = tc.DESCRIPTION
    gb.MODELS_DIR = tc.MODELS_DIR
    gb.DATASET_SPLIT_DIR = tc.DATASET_SPLIT_DIR
    gb.REDVALS_DIR = tc.REDVALS_DIR
    gb.EMBED_DIMS = tc.EMBED_DIMS
    gb.MAX_MODEL_SEQ_LEN = tc.MAX_MODEL_SEQ_LEN
    gb.MAX_IMPORTED_SEQ_LEN = tc.MAX_IMPORTED_SEQ_LEN
    if (gb.MAX_IMPORTED_SEQ_LEN is not None and gb.MAX_MODEL_SEQ_LEN is not None
            and gb.MAX_IMPORTED_SEQ_LEN < gb.MAX_MODEL_SEQ_LEN):
        raise ValueError("MAX_IMPORTED_SEQ_LEN cannot be smaller than MAX_MODEL_SEQ_LEN")
    gb.D_MODEL = tc.D_MODEL
    gb.N_LAYERS = tc.N_LAYERS
    gb.N_HEAD = tc.N_HEAD
    gb.D_FF = tc.D_FF
    gb.POOLING_TYPE = tc.POOLING_TYPE
    gb.USE_CONV_STEM = tc.USE_CONV_STEM
    gb.CONV_STEM_KERNEL_SIZE = tc.CONV_STEM_KERNEL_SIZE
    gb.CONV_STEM_RESIDUAL = tc.CONV_STEM_RESIDUAL
    gb.CONV_STEM_INIT_SCALE = tc.CONV_STEM_INIT_SCALE
    gb.USE_CONVFORMER = tc.USE_CONVFORMER
    gb.CONFORMER_KERNEL_SIZE = tc.CONFORMER_KERNEL_SIZE
    gb.DROPOUT_PROP = tc.DROPOUT_PROP
    gb.ATT_DROPOUT_PROP = tc.ATT_DROPOUT_PROP
    gb.PAIR_LOSS_WEIGHT = tc.PAIR_LOSS_WEIGHT
    gb.TRIPLET_LOSS_WEIGHT = tc.TRIPLET_LOSS_WEIGHT
    gb.USE_UNCERTAINTY_WEIGHTING = tc.USE_UNCERTAINTY_WEIGHTING
    gb.UNCERTAINTY_LEARNING_RATE = tc.UNCERTAINTY_LEARNING_RATE
    gb.BACTERIA_DISTANCE_FACTOR = tc.BACTERIA_DISTANCE_FACTOR
    gb.ARCHEA_DISTANCE_FACTOR = tc.ARCHEA_DISTANCE_FACTOR
    gb.DISTANCE_GAMMA_CORRECTION_GAMMA = tc.DISTANCE_GAMMA_CORRECTION_GAMMA
    gb.TRIPLET_MARGIN_EPSILON = tc.TRIPLET_MARGIN_EPSILON
    gb.MANUAL_TRIPLET_MARGINS_PER_RANK = tc.MANUAL_TRIPLET_MARGINS_PER_RANK
    gb.MANUAL_TRIPLET_MARGINS = tc.MANUAL_TRIPLET_MARGINS
    gb.PAIR_RANKS = tc.PAIR_RANKS
    gb.INTRODUCE_RANK_AT_BATCHES = tc.INTRODUCE_RANK_AT_BATCHES
    gb.PAIR_MINING_BUCKETS = tc.PAIR_MINING_BUCKETS
    gb.PAIR_SIGN_BIAS_BETA_PER_RANK = tc.PAIR_SIGN_BIAS_BETA_PER_RANK
    gb.DOWNSAMPLE_PAIRS_AT_RANK = tc.DOWNSAMPLE_PAIRS_AT_RANK
    gb.PAIR_MINING_REPRESENTATIVE_SET_SIZES = tc.PAIR_MINING_REPRESENTATIVE_SET_SIZES
    gb.TRIPLET_MINING_REPRESENTATIVE_SET_SIZES = tc.TRIPLET_MINING_REPRESENTATIVE_SET_SIZES
    gb.PAIR_MINING_EMA_ALPHA = tc.PAIR_MINING_EMA_ALPHA
    gb.TRIPLET_MINING_EMA_ALPHA = tc.TRIPLET_MINING_EMA_ALPHA
    gb.PAIR_MINING_EMA_WEIGHT_EXPONENT = tc.PAIR_MINING_EMA_WEIGHT_EXPONENT
    gb.TRIPLET_MINING_EMA_WEIGHT_EXPONENT = tc.TRIPLET_MINING_EMA_WEIGHT_EXPONENT
    gb.TRIPLET_EMA_HARD_WEIGHT = tc.TRIPLET_EMA_HARD_WEIGHT
    gb.TRIPLET_EMA_MODERATE_WEIGHT = tc.TRIPLET_EMA_MODERATE_WEIGHT
    gb.PAIR_EMA_MEAN_WEIGHT = tc.PAIR_EMA_MEAN_WEIGHT
    gb.PAIR_EMA_QUARTILES_WEIGHT = tc.PAIR_EMA_QUARTILES_WEIGHT
    gb.PAIR_PER_RANK_BATCH_PROPORTION_MAX = tc.PAIR_PER_RANK_BATCH_PROPORTION_MAX
    gb.TRIPLET_PER_RANK_BATCH_PROPORTION_MAX = tc.TRIPLET_PER_RANK_BATCH_PROPORTION_MAX
    gb.PAIR_PER_RANK_BATCH_PROPORTION_MIN = tc.PAIR_PER_RANK_BATCH_PROPORTION_MIN
    gb.TRIPLET_PER_RANK_BATCH_PROPORTION_MIN = tc.TRIPLET_PER_RANK_BATCH_PROPORTION_MIN
    gb.PROPORTION_IDENTICAL_SEQUENCES_TO_GENERATE = tc.PROPORTION_IDENTICAL_SEQUENCES_TO_GENERATE
    gb.SUBSEQUENCES_ALWAYS_CROSS_REGION = tc.SUBSEQUENCES_ALWAYS_CROSS_REGION
    gb.SUB_SEQUENCE_TRUE_DISTANCE = tc.SUB_SEQUENCE_TRUE_DISTANCE
    gb.TRIPLET_RANKS = tc.TRIPLET_RANKS
    gb.TRIPLET_MINING_BUCKETS = tc.TRIPLET_MINING_BUCKETS
    gb.FILTER_ZERO_LOSS_TRIPLETS = tc.FILTER_ZERO_LOSS_TRIPLETS
    gb.MIN_TRIPLETS_AFTER_FILTER_PER_RANK_BATCH_SIZE_FACTOR = tc.MIN_TRIPLETS_AFTER_FILTER_PER_RANK_BATCH_SIZE_FACTOR
    gb.N_PAIRS_PER_BATCH = tc.N_PAIRS_PER_BATCH
    gb.N_TRIPLETS_PER_BATCH = tc.N_TRIPLETS_PER_BATCH
    gb.USE_FULL_SEQS = tc.USE_FULL_SEQS
    gb.USE_SUB_SEQS = tc.USE_SUB_SEQS
    gb.MAX_NUM_SUBSEQS = tc.MAX_NUM_SUBSEQS
    gb.MUTATION_RATE = tc.MUTATION_RATE
    gb.MIN_TRUNC_START = tc.MIN_TRUNC_START
    gb.MAX_TRUNC_START = tc.MAX_TRUNC_START
    gb.MIN_TRUNC_END = tc.MIN_TRUNC_END
    gb.MAX_TRUNC_END = tc.MAX_TRUNC_END
    gb.PROP_TRUNC = tc.PROP_TRUNC
    gb.PROP_SHIFT_SEQS = tc.PROP_SHIFT_SEQS
    gb.LEARNING_RATE = tc.LEARNING_RATE
    gb.WEIGHT_DECAY = tc.WEIGHT_DECAY
    gb.NUM_BATCHES = tc.NUM_BATCHES
    gb.START_FROM_CKPT = tc.START_FROM_CKPT
    gb.NUM_MICRO_BATCHES_PER_BATCH = tc.NUM_MICRO_BATCHES_PER_BATCH
    gb.LR_SCHEDULER_TYPE = tc.LR_SCHEDULER_TYPE
    gb.LR_SCHEDULER_KWARGS = tc.LR_SCHEDULER_KWARGS
    gb.SAVE_EVERY_N_BATCHES = tc.SAVE_EVERY_N_BATCHES
    gb.RECORD_LOSS_EVERY_N_BATCHES = tc.RECORD_LOSS_EVERY_N_BATCHES
    gb.PLOT_LOSS_EVERY_N_BATCHES = tc.PLOT_LOSS_EVERY_N_BATCHES
    gb.PLOT_TOTAL_LOSS = tc.PLOT_TOTAL_LOSS
    gb.PLOT_TRIPLET_LOSS = tc.PLOT_TRIPLET_LOSS
    gb.PLOT_PAIR_LOSS = tc.PLOT_PAIR_LOSS
    gb.PLOT_UNCERTAINTY_WEIGHTING = tc.PLOT_UNCERTAINTY_WEIGHTING
    gb.SECOND_LOSS_PLOT_AFTER_N_BATCHES = tc.SECOND_LOSS_PLOT_AFTER_N_BATCHES
    gb.SECOND_TRIPLET_SATISFACTION_PLOT_AFTER_N_BATCHES = tc.SECOND_TRIPLET_SATISFACTION_PLOT_AFTER_N_BATCHES
    gb.SECOND_PAIR_ERROR_METRICS_PLOT_AFTER_N_BATCHES = tc.SECOND_PAIR_ERROR_METRICS_PLOT_AFTER_N_BATCHES
    gb.SECOND_TRIPLET_ERROR_METRICS_PLOT_AFTER_N_BATCHES = tc.SECOND_TRIPLET_ERROR_METRICS_PLOT_AFTER_N_BATCHES
    gb.DO_QT = tc.DO_QT
    gb.QT_BATCH_SIZE = tc.QT_BATCH_SIZE
    gb.MINING_BATCH_SIZE = tc.MINING_BATCH_SIZE
    gb.FREE_LARGE_INTERMEDIATES_WHEN_MINING = tc.FREE_LARGE_INTERMEDIATES_WHEN_MINING
    gb.PAIR_MINING_WARMUP_UNIFORM_DURATION = tc.PAIR_MINING_WARMUP_UNIFORM_DURATION
    gb.PAIR_MINING_WARMUP_TRANSITION_DURATION = tc.PAIR_MINING_WARMUP_TRANSITION_DURATION
    gb.TRIPLET_MINING_WARMUP_UNIFORM_DURATION = tc.TRIPLET_MINING_WARMUP_UNIFORM_DURATION
    gb.TRIPLET_MINING_WARMUP_TRANSITION_DURATION = tc.TRIPLET_MINING_WARMUP_TRANSITION_DURATION
    gb.USE_TRIPLET_MINING_POSITIVE_SELECTION_BIAS = tc.USE_TRIPLET_MINING_POSITIVE_SELECTION_BIAS
    gb.TRIPLET_MINING_POSITIVE_SELECTION_BIAS_BETA = tc.TRIPLET_MINING_POSITIVE_SELECTION_BIAS_BETA
    gb.TRIPLET_MINING_POSITIVE_SELECTION_BIAS_EPS = tc.TRIPLET_MINING_POSITIVE_SELECTION_BIAS_EPS
    gb.USE_TRIPLET_MINING_NEGATIVE_SELECTION_BIAS = tc.USE_TRIPLET_MINING_NEGATIVE_SELECTION_BIAS
    gb.TRIPLET_MINING_NEGATIVE_SELECTION_BIAS_BETA = tc.TRIPLET_MINING_NEGATIVE_SELECTION_BIAS_BETA
    gb.TRIPLET_MINING_NEGATIVE_SELECTION_BIAS_EPS = tc.TRIPLET_MINING_NEGATIVE_SELECTION_BIAS_EPS
    gb.TRIPLET_NEGATIVES_PER_AP = tc.TRIPLET_NEGATIVES_PER_AP
    gb.QT_EVERY_N_BATCHES = tc.QT_EVERY_N_BATCHES
    gb.QT_BEFORE_FIRST_BATCH = tc.QT_BEFORE_FIRST_BATCH
    gb.QT_KMER_K_VALUES = tc.QT_KMER_K_VALUES
    gb.QT_GET_KMER_RESULTS = tc.QT_GET_KMER_RESULTS
    gb.QT_PCOA_RANKS = tc.QT_PCOA_RANKS
    gb.QT_PCOA_EVERY_N_BATCHES = tc.QT_PCOA_EVERY_N_BATCHES
    gb.QT_UMAP_RANKS = tc.QT_UMAP_RANKS
    gb.QT_UMAP_EVERY_N_BATCHES = tc.QT_UMAP_EVERY_N_BATCHES
    gb.QT_CLUSTERING_RANKS = tc.QT_CLUSTERING_RANKS
    gb.QT_RANK_LEVEL_CLUSTERING_METRICS_AT_RANKS = tc.QT_RANK_LEVEL_CLUSTERING_METRICS_AT_RANKS
    gb.QT_MAX_TAXA_PER_RANK_LEVEL_CLUSTERING = tc.QT_MAX_TAXA_PER_RANK_LEVEL_CLUSTERING
    gb.RANK_LEVEL_CLUSTERING_PLOTS_EVERY_N_BATCHES = tc.RANK_LEVEL_CLUSTERING_PLOTS_EVERY_N_BATCHES
    gb.QT_CLASSIFICATION_RANKS = tc.QT_CLASSIFICATION_RANKS
    gb.QT_RECORD_MISS_CLASSIFICATION_SEQUENCES_AT_RANKS = tc.QT_RECORD_MISS_CLASSIFICATION_SEQUENCES_AT_RANKS
    gb.QT_MAX_N_REGIONS = tc.QT_MAX_N_REGIONS
    gb.QT_PROP_SHIFT_SEQS = tc.QT_PROP_SHIFT_SEQS
    gb.QT_MUTATION_RATE = tc.QT_MUTATION_RATE
    gb.QT_PROP_TRUNC = tc.QT_PROP_TRUNC
    gb.QT_MIN_TRUNC_START = tc.QT_MIN_TRUNC_START
    gb.QT_MAX_TRUNC_START = tc.QT_MAX_TRUNC_START
    gb.QT_MIN_TRUNC_END = tc.QT_MIN_TRUNC_END
    gb.QT_MAX_TRUNC_END = tc.QT_MAX_TRUNC_END
    gb.PIN_MEMORY_FOR_MINING = tc.PIN_MEMORY_FOR_MINING
    gb.USE_FLOAT_16_FOR_MINING_COSINE_DISTANCES = tc.USE_FLOAT_16_FOR_MINING_COSINE_DISTANCES
    gb.VERBOSE_TRAINING_TIMING = tc.VERBOSE_TRAINING_TIMING
    gb.VERBOSE_TRIPLET_LOSS = tc.VERBOSE_TRIPLET_LOSS
    gb.VERBOSE_PAIR_LOSS = tc.VERBOSE_PAIR_LOSS
    gb.VERBOSE_UNCERTAINTY_WEIGHTING = tc.VERBOSE_UNCERTAINTY_WEIGHTING
    gb.VERBOSE_PAIR_MINING = tc.VERBOSE_PAIR_MINING
    gb.VERBOSE_TRIPLET_MINING = tc.VERBOSE_TRIPLET_MINING
    gb.VERBOSE_EVERY_N_BATCHES = tc.VERBOSE_EVERY_N_BATCHES
    gb.VERBOSE_MINING_TIMING = tc.VERBOSE_MINING_TIMING
    gb.LOG_TRIPLET_LOSS = tc.LOG_TRIPLET_LOSS
    gb.LOG_PAIR_LOSS = tc.LOG_PAIR_LOSS
    gb.LOG_UNCERTAINTY_WEIGHTING = tc.LOG_UNCERTAINTY_WEIGHTING
    gb.VERBOSE_CONV_STEM_SCALE = tc.VERBOSE_CONV_STEM_SCALE
    gb.LOG_CONV_STEM_SCALE = tc.LOG_CONV_STEM_SCALE
    gb.LOG_TRIPLET_MINING = tc.LOG_TRIPLET_MINING
    gb.LOG_PAIR_MINING = tc.LOG_PAIR_MINING
    gb.LOG_MINING = tc.LOG_MINING
    gb.LOG_MINING_BUCKET_THRESHOLDS = tc.LOG_MINING_BUCKET_THRESHOLDS
    gb.LOG_MINING_PHYLA_COUNTS = tc.LOG_MINING_PHYLA_COUNTS
    gb.MINING_PHYLA_LOG_MAX_LINES_PER_TABLE = tc.MINING_PHYLA_LOG_MAX_LINES_PER_TABLE
    gb.LOG_MINING_ARC_BAC_COUNTS = tc.LOG_MINING_ARC_BAC_COUNTS
    gb.CROSS_REGION_LOGGING_MAX_LINES_PER_TABLE = tc.CROSS_REGION_LOGGING_MAX_LINES_PER_TABLE
    gb.LOG_EVERY_N_BATCHES = tc.LOG_EVERY_N_BATCHES
    gb.PLOT_TRIPLET_SATISFACTION_EVERY_N_BATCHES = tc.PLOT_TRIPLET_SATISFACTION_EVERY_N_BATCHES
    gb.PLOT_PAIR_DISTANCES_EVERY_N_BATCHES = tc.PLOT_PAIR_DISTANCES_EVERY_N_BATCHES
    gb.PLOT_PAIR_ERROR_METRICS_EVERY_N_BATCHES = tc.PLOT_PAIR_ERROR_METRICS_EVERY_N_BATCHES
    gb.PLOT_TRIPLET_ERROR_METRICS_EVERY_N_BATCHES = tc.PLOT_TRIPLET_ERROR_METRICS_EVERY_N_BATCHES
    gb.PLOT_CONV_STEM_SCALE_EVERY_N_BATCHES = tc.PLOT_CONV_STEM_SCALE_EVERY_N_BATCHES
    gb.SECOND_CONV_STEM_SCALE_PLOT_AFTER_N_BATCHES = tc.SECOND_CONV_STEM_SCALE_PLOT_AFTER_N_BATCHES
    gb.PRINT_BATCH_NUM_EVERY_N_BATCHES = tc.PRINT_BATCH_NUM_EVERY_N_BATCHES
    gb.SEQ_3BIT_REPRESENTATION_DICT = tc.SEQ_3BIT_REPRESENTATION_DICT
    gb.QT_VERBOSE = tc.QT_VERBOSE
    gb.QT_PRINT_RESULTS = tc.QT_PRINT_RESULTS
    gb.QT_TESTS_TODO = tc.QT_TESTS_TODO
    gb.RED_DISTANCE_BETWEEN_DOMAINS = tc.RED_DISTANCE_BETWEEN_DOMAINS
    gb.NUM_BATCHES_PER_MINING_RUN = tc.NUM_BATCHES_PER_MINING_RUN
    gb.SHUFFLE_TRIPLETS_AND_PAIRS_AFTER_MINING = tc.SHUFFLE_TRIPLETS_AND_PAIRS_AFTER_MINING
    gb.MINING_TRAIN_SEQUENCES_SUB_SAMPLE_BACTERIA = tc.MINING_TRAIN_SEQUENCES_SUB_SAMPLE_BACTERIA
    gb.MINING_TRAIN_SEQUENCES_SUB_SAMPLE_ARCHAEA = tc.MINING_TRAIN_SEQUENCES_SUB_SAMPLE_ARCHAEA
    gb.TAXON_SIZE_MINING_BIAS_BASELINE_STAT = tc.TAXON_SIZE_MINING_BIAS_BASELINE_STAT
    gb.TAXON_SIZE_MINING_BIAS_BASELINE_PERCENTILE = tc.TAXON_SIZE_MINING_BIAS_BASELINE_PERCENTILE
    gb.USE_REPRESENTATIVE_TAXON_SIZE_BALANCING = tc.USE_REPRESENTATIVE_TAXON_SIZE_BALANCING
    gb.PAIR_REPRESENTATIVE_TAXON_SIZE_BALANCE_LAMBDA = tc.PAIR_REPRESENTATIVE_TAXON_SIZE_BALANCE_LAMBDA
    gb.TRIPLET_REPRESENTATIVE_TAXON_SIZE_BALANCE_LAMBDA = tc.TRIPLET_REPRESENTATIVE_TAXON_SIZE_BALANCE_LAMBDA
    gb.REPRESENTATIVE_TAXON_SIZE_BALANCE_EPS = tc.REPRESENTATIVE_TAXON_SIZE_BALANCE_EPS
    gb.REPRESENTATIVE_TAXON_SIZE_BALANCE_WEIGHT_CLIP = tc.REPRESENTATIVE_TAXON_SIZE_BALANCE_WEIGHT_CLIP
    gb.USE_TAXON_SIZE_MINING_BIAS = tc.USE_TAXON_SIZE_MINING_BIAS
    gb.TAXON_SIZE_MINING_BIAS_ALPHAS_PER_RANK = tc.TAXON_SIZE_MINING_BIAS_ALPHAS_PER_RANK
    gb.TAXON_SIZE_MINING_BIAS_MIN_KEEP_PER_RANK = tc.TAXON_SIZE_MINING_BIAS_MIN_KEEP_PER_RANK
    gb.RELATIVE_ERROR_EPSILONS_PAIR_LOSS = tc.RELATIVE_ERROR_EPSILONS_PAIR_LOSS
    gb.RELATIVE_ERROR_EPSILONS_PAIR_MINING = tc.RELATIVE_ERROR_EPSILONS_PAIR_MINING
    gb.PAIR_RELATIVE_LOSS_CAP = tc.PAIR_RELATIVE_LOSS_CAP
    gb.RELATIVE_ERROR_EPSILONS_TRIPLET_LOSS = tc.RELATIVE_ERROR_EPSILONS_TRIPLET_LOSS
    gb.TRIPLET_RELATIVE_LOSS_CAP = tc.TRIPLET_RELATIVE_LOSS_CAP
    gb.SEQLESS_MODE = tc.SEQLESS_MODE
    gb.TAXA_CAP_MAX_NUM_TAXA = tc.TAXA_CAP_MAX_NUM_TAXA
    gb.TAXA_CAP_RANK = tc.TAXA_CAP_RANK
    gb.TAXA_CAP_VERBOSE = tc.TAXA_CAP_VERBOSE
