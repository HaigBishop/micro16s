"""
This script is used to train Micro16S models. 
See the micro16s README.md for more details on the whole pipeline.
"""


# IMPORTS ================================================
# Packages
import pandas as pd
import torch
import time
import os
import gc
import importlib
import matplotlib
from contextlib import nullcontext

# Configure Matplotlib backend to avoid GUI-related memory leaks
matplotlib.use('Agg')

# Local Imports
import globals_config as gb
from training_config import set_train_config_as_globals, TrainingConfig
from micro16s_dataset_loader import load_micro16s_dataset, unload_micro16s_dataset
from model import Micro16S, embedding_triplet_loss, embedding_pair_loss, UncertaintyLoss, LinearWarmupToLearningRate
from seqless_model import SequencelessMicro16S
from triplet_pair_mining import mine
from quick_test import quick_test, plot_quick_test, print_quick_test, add_results_df, plot_rank_level_clustering
from utils import parent_dir, print_micro16s, get_next_model_name, write_about_training, get_vram_usage_str
from loss_results import add_loss_results_df, plot_loss_results, plot_uncertainty_weights
from logging_utils import (print_log_uncertainty_weighting, print_log_conv_stem_scale,
                          print_triplet_loss_stats, write_triplet_loss_log,
                          print_pair_loss_stats, write_pair_loss_log)
from log_plotting import (init_triplet_satisfaction_df, plot_triplet_satisfaction, save_triplet_satisfaction_csv,
                          init_pair_distances_df, plot_pair_distances, plot_pair_distances_log2, save_pair_distances_csv,
                          init_pair_error_metrics_df, plot_pair_error_metrics, save_pair_error_metrics_csv,
                          init_triplet_error_metrics_df, plot_triplet_error_metrics, save_triplet_error_metrics_csv,
                          plot_conv_stem_scale)



# Start Script
if __name__ == "__main__":


    # MICRO16S ================================================
    print("\nWelcome to Micro16S --------------------------")
    print_micro16s()



    # CHECK GPU ================================================
    # Check if GPU is detected
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cpu':
        input("Warning. No GPU detected. Press any key to continue...")


    # PERFORMANCE OPTIMIZATIONS ================================================
    # These settings improve inference/training throughput on CUDA GPUs.
    if device.type == 'cuda':
        # cuDNN benchmark: auto-tunes convolution algorithms for the current hardware.
        # First forward pass is slower (profiling), subsequent passes are faster.
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # We only autocast on CUDA. CPU stays full precision.
    use_amp = (device.type == 'cuda')
    amp_ctx = torch.autocast(device_type='cuda', dtype=torch.bfloat16) if use_amp else nullcontext()


    # TRAINING REGIMINE ================================================
    # Training regimines:
    # - SINGLE: Just train one model using the default TrainingConfig
    # - SPECIFIED: Train specific model(s), explicitly defined by config paths
    # - EXPERIMENT: Train a model for every config found in a particular experiment directory
    TRAINING_REGIMINE = "SINGLE" # "SINGLE" or "SPECIFIED" or "EXPERIMENT"

    if TRAINING_REGIMINE == "SINGLE":
        # Just use the default TrainingConfig
        # This just uses the config in training_config.py
        TRAINING_RUNS = ['DEFAULT']

    elif TRAINING_REGIMINE == "SPECIFIED":
        # Define the TrainingConfig for each training run
        # (training runs are just paths to json files - see training_config.py)
        TRAINING_RUNS = [
            '/home/haig/Repos/micro16s/training_configs/a.json',
            '/home/haig/Repos/micro16s/training_configs/b.json',
            '/home/haig/Repos/micro16s/training_configs/c.json',
        ]

    elif TRAINING_REGIMINE == "EXPERIMENT":
        # Get the list of config files in the experiment directory
        EXPERIMENT_DIR = '/home/haig/Repos/micro16s/experiments/exp_006/cfgs/'
        TRAINING_RUNS = sorted([os.path.join(EXPERIMENT_DIR, f) for f in os.listdir(EXPERIMENT_DIR) if f.endswith('.json')])



    # TRAINING LOOP ================================================
    # Loop through each training run
    for i, TRAINING_RUN in enumerate(TRAINING_RUNS):

        # Skip the first n training runs
        skip_first_n_runs = 0
        if i + 1 <= skip_first_n_runs:
            continue

        # If this is not the first training run
        if i > skip_first_n_runs:
            # Drop dataset variables from the previous run
            unload_micro16s_dataset()
            # Re-import globals_config for the next training run
            importlib.reload(gb)
            # Delete the results dataframes from the previous run
            del loss_results_df, clustering_scores_df, classification_scores_df, macro_classification_scores_df, ssc_scores_df, rank_level_clustering_dfs, triplet_satisfaction_df, pair_distances_df, pair_error_metrics_df, triplet_error_metrics_df
        
        # Print start of training run
        TRAINING_RUN_FILENAME = TRAINING_RUN.split('/')[-1]
        if len(TRAINING_RUNS) > 1:
            print(f"\n\n{'='*100}\nStarting training run: {TRAINING_RUN_FILENAME} --------------------------")



        # SET TRAINING CONFIGURATION ================================================
        # TrainingConfig object
        tc = TrainingConfig()
        # Load the training config (unless DEFAULT)
        if TRAINING_RUN != 'DEFAULT':
            tc.load_training_config(TRAINING_RUN)
        # This sets all training configurations as global variables
        # Configurations are in training_config.py
        set_train_config_as_globals(tc)
        # Initialise EMA buffers used for per-rank mining budgets
        gb.init_mining_ema_buffers()

        # Determine if uncertainty weighting is actually active (requires: USE_UNCERTAINTY_WEIGHTING=True AND both losses actively used)
        gb.IS_USING_UNCERTAINTY_WEIGHTING = gb.USE_UNCERTAINTY_WEIGHTING and gb.TRIPLET_LOSS_WEIGHT > 0 and gb.PAIR_LOSS_WEIGHT > 0 and gb.N_TRIPLETS_PER_BATCH > 0 and gb.N_PAIRS_PER_BATCH > 0


        # OUTPUT FILES ================================================
        AUTO_MODEL_NAME = True # Model name will be automatically generated according to existing model directories
        MODEL_NAME = "m16s_001" if not AUTO_MODEL_NAME else get_next_model_name(gb.MODELS_DIR)
        MODEL_DIR = gb.MODELS_DIR + MODEL_NAME + "/"
        LATEST_MODEL_PATH = gb.MODELS_DIR + MODEL_NAME + "/" + MODEL_NAME + "_latest.pth"
        def EVERY_N_BATCHES_MODEL_PATH(n_batch):
            return gb.MODELS_DIR + MODEL_NAME + "/ckpts/" + MODEL_NAME + "_" + str(n_batch) + "_batches.pth"
        QT_RESULTS_CSV_DIR = gb.MODELS_DIR + MODEL_NAME + "/qt_results/"
        QT_PLOTS_DIR = gb.MODELS_DIR + MODEL_NAME + "/qt_plots/"
        LOG_PLOTS_DIR = gb.MODELS_DIR + MODEL_NAME + "/log_plots/"
        CFG_DIR = gb.MODELS_DIR + MODEL_NAME + "/cfg/"
        LOSS_DIR = gb.MODELS_DIR + MODEL_NAME + "/loss/"
        ABOUT_TRAINING_PATH = gb.MODELS_DIR + MODEL_NAME + "/about_training.txt"
        LOGS_DIR = gb.MODELS_DIR + MODEL_NAME + "/logs/"

        # Write dirs if they don't exist
        os.makedirs(parent_dir(LATEST_MODEL_PATH), exist_ok=True)
        os.makedirs(parent_dir(ABOUT_TRAINING_PATH), exist_ok=True)
        os.makedirs(parent_dir(EVERY_N_BATCHES_MODEL_PATH(0)), exist_ok=True)
        os.makedirs(QT_RESULTS_CSV_DIR, exist_ok=True)
        os.makedirs(QT_PLOTS_DIR, exist_ok=True)
        os.makedirs(LOG_PLOTS_DIR, exist_ok=True)
        os.makedirs(CFG_DIR, exist_ok=True)
        os.makedirs(LOSS_DIR, exist_ok=True)
        os.makedirs(LOGS_DIR, exist_ok=True)


        # WRITE TRAINING CONFIGURATION ================================================
        # Write training config to model dir
        tc.write_training_config(MODEL_NAME, CFG_DIR)

        # Write placeholder about_training info so we have breadcrumbs even if training aborts early
        placeholder_time = time.time()
        write_about_training(output_path=ABOUT_TRAINING_PATH, model_name=MODEL_NAME, model_dir=MODEL_DIR, training_run=TRAINING_RUN, training_regimine=TRAINING_REGIMINE, description=tc.DESCRIPTION, start_time=placeholder_time, end_time=placeholder_time, batches_completed=0, device=device, logs_dir=LOGS_DIR, qt_results_dir=QT_RESULTS_CSV_DIR, qt_plots_dir=QT_PLOTS_DIR, latest_model_path=LATEST_MODEL_PATH, is_placeholder_about_file=True)


        # LOAD DATASET ================================================
        # Load the dataset directory
        load_micro16s_dataset(gb.DATASET_SPLIT_DIR, gb.QT_KMER_K_VALUES)


        
        HAS_TEST_SET = len(gb.TESTING_INDICES) > 0
        HAS_EXCLUDED_SET = gb.HAS_EXCLUDED_SET

        # If there is no test set, remove test-set quick tests.
        if not HAS_TEST_SET and gb.DO_QT:
            if gb.QT_TESTS_TODO is None:
                gb.QT_TESTS_TODO = set()
            if not isinstance(gb.QT_TESTS_TODO, set):
                gb.QT_TESTS_TODO = set(gb.QT_TESTS_TODO)
            n_tests_before = len(gb.QT_TESTS_TODO)
            gb.QT_TESTS_TODO = {t for t in gb.QT_TESTS_TODO if not t.startswith('test-')}
            n_removed = n_tests_before - len(gb.QT_TESTS_TODO)
            if n_removed > 0:
                print(f"NOTICE: No testing sequences -> removing {n_removed} test-set quick test items.")
            if len(gb.QT_TESTS_TODO) == 0:
                print("NOTICE: No quick test items remain after removing test-set items -> setting DO_QT=False.")
                gb.DO_QT = False
        # If there is no excluded set, remove excluded-set quick tests.
        if not HAS_EXCLUDED_SET and gb.DO_QT:
            if gb.QT_TESTS_TODO is None:
                gb.QT_TESTS_TODO = set()
            if not isinstance(gb.QT_TESTS_TODO, set):
                gb.QT_TESTS_TODO = set(gb.QT_TESTS_TODO)
            n_tests_before = len(gb.QT_TESTS_TODO)
            gb.QT_TESTS_TODO = {t for t in gb.QT_TESTS_TODO if not t.startswith('excl-')}
            n_removed = n_tests_before - len(gb.QT_TESTS_TODO)
            if n_removed > 0:
                print(f"NOTICE: No excluded sequences -> removing {n_removed} excluded-set quick test items.")
            if len(gb.QT_TESTS_TODO) == 0:
                print("NOTICE: No quick test items remain after removing excluded-set items -> setting DO_QT=False.")
                gb.DO_QT = False


        # TRAINING ================================================
        # Initialise model and optimizer
        if gb.SEQLESS_MODE:
            # Sequenceless mode: model is a lookup table of embeddings (no sequence processing)
            n_train_sequences = len(gb.TRAINING_INDICES)
            model = SequencelessMicro16S(n_train_sequences=n_train_sequences, embed_dims=gb.EMBED_DIMS, name=MODEL_NAME)
        else:
            # Standard mode: full Micro16S model with sequence processing
            model = Micro16S(embed_dims=gb.EMBED_DIMS, max_seq_len=gb.MAX_MODEL_SEQ_LEN, d_model=gb.D_MODEL, 
                                n_layers=gb.N_LAYERS, n_head=gb.N_HEAD, d_ff=gb.D_FF,
                                seq_3bit_representation_dict=gb.SEQ_3BIT_REPRESENTATION_DICT,
                                name=MODEL_NAME, pooling_type=gb.POOLING_TYPE,
                                use_convformer=gb.USE_CONVFORMER, conformer_kernel_size=gb.CONFORMER_KERNEL_SIZE,
                                use_conv_stem=gb.USE_CONV_STEM, conv_stem_kernel_size=gb.CONV_STEM_KERNEL_SIZE,
                                conv_stem_residual=gb.CONV_STEM_RESIDUAL, conv_stem_init_scale=gb.CONV_STEM_INIT_SCALE,
                                dropout=gb.DROPOUT_PROP, attn_dropout=gb.ATT_DROPOUT_PROP)
        # Move model to device and create optimizer after
        model = model.to(device)

        # Optionally initialize model weights from a checkpoint
        if gb.START_FROM_CKPT is not False:
            if not isinstance(gb.START_FROM_CKPT, str):
                raise ValueError("START_FROM_CKPT must be False or a string path to a .pth file.")
            if not gb.START_FROM_CKPT.endswith(".pth"):
                raise ValueError(f"START_FROM_CKPT must point to a .pth file, got: {gb.START_FROM_CKPT}")
            if not os.path.isfile(gb.START_FROM_CKPT):
                raise FileNotFoundError(f"START_FROM_CKPT file not found: {gb.START_FROM_CKPT}")

            checkpoint = torch.load(gb.START_FROM_CKPT, map_location=device)
            checkpoint_state_dict = checkpoint['model_state_dict'] if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint else checkpoint
            try:
                model.load_state_dict(checkpoint_state_dict)
            except RuntimeError as e:
                raise RuntimeError(
                    f"Failed to load START_FROM_CKPT due to model incompatibility: {gb.START_FROM_CKPT}\n"
                    "Check that model hyperparameters match the checkpoint (e.g. D_MODEL, N_LAYERS, N_HEAD, D_FF, EMBED_DIMS, MAX_MODEL_SEQ_LEN).\n"
                    f"Original error: {e}"
                ) from e
            print(f"Loaded model weights from START_FROM_CKPT: {gb.START_FROM_CKPT}")

        # Create uncertainty loss module if enabled
        uncertainty_loss = None
        if gb.IS_USING_UNCERTAINTY_WEIGHTING:
            uncertainty_loss = UncertaintyLoss()
            uncertainty_loss = uncertainty_loss.to(device)

        # Create optimizer with decoupled weight decay (AdamW)
        # Weight decay is applied selectively via parameter groups: embedding/linear/conv
        # weights are decayed, while biases, LayerNorm params, and gating scalars are not.
        # Uncertainty loss parameters (if enabled) use a separate learning rate and no decay.
        if hasattr(model, 'get_parameter_groups'):
            model_param_groups = model.get_parameter_groups(gb.WEIGHT_DECAY)
        else:
            # Fallback for models without parameter group support (e.g. sequenceless mode)
            model_param_groups = [{'params': model.parameters(), 'weight_decay': gb.WEIGHT_DECAY}]

        if gb.IS_USING_UNCERTAINTY_WEIGHTING:
            # We use a separate learning rate for the uncertainty log-variance parameters
            # as they often need to adapt at a different speed than the model weights.
            optimizer = torch.optim.AdamW(
                model_param_groups + [
                    {'params': uncertainty_loss.parameters(), 'lr': gb.UNCERTAINTY_LEARNING_RATE, 'weight_decay': 0.0}
                ],
                lr=gb.LEARNING_RATE
            )
        else:
            optimizer = torch.optim.AdamW(model_param_groups, lr=gb.LEARNING_RATE)

        # Initialize empty loss dataframe for results
        loss_results_df = pd.DataFrame()

        # LR Scheduler ------------------
        scheduler = None
        if gb.LR_SCHEDULER_TYPE == 'CosineAnnealingWarmRestarts':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, **gb.LR_SCHEDULER_KWARGS)
        elif gb.LR_SCHEDULER_TYPE == 'OneCycleLR':
            scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, total_steps=gb.NUM_BATCHES, **gb.LR_SCHEDULER_KWARGS)
        elif gb.LR_SCHEDULER_TYPE == 'StepLR':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **gb.LR_SCHEDULER_KWARGS)
        elif gb.LR_SCHEDULER_TYPE == 'LinearWarmupToLearningRate':
            scheduler = LinearWarmupToLearningRate(
                optimizer,
                start_lr=gb.LR_SCHEDULER_KWARGS['start_lr'],
                warmup_batches=gb.LR_SCHEDULER_KWARGS['warmup_batches']
            )

        # Initialize tracking variables for performance monitoring
        previous_classification_accuracy = None
        previous_clustering_v_measure = None

        # Test before first batch if enabled (skip in sequenceless mode)
        if gb.DO_QT and gb.QT_BEFORE_FIRST_BATCH:
            print("\nPre-Training Test --------------------------")
            qt_start_time = time.time()
            model.eval()
            clustering_scores_dict, classification_scores_dict, macro_classification_scores_dict, ssc_scores_dict, rank_level_clustering_dict = quick_test(model=model, model_dir=MODEL_DIR, batch_num=0, verbose=gb.QT_VERBOSE)
            # Return model to device
            model = model.to(device)
            print_quick_test(clustering_scores_dict, classification_scores_dict, macro_classification_scores_dict, ssc_scores_dict)
            clustering_scores_df = add_results_df(clustering_scores_dict, batch_num=0)
            classification_scores_df = add_results_df(classification_scores_dict, batch_num=0)
            macro_classification_scores_df = add_results_df(macro_classification_scores_dict, batch_num=0)
            ssc_scores_df = add_results_df(ssc_scores_dict, batch_num=0)
            # Initialize per-rank clustering DataFrames
            rank_level_clustering_dfs = {}
            for rank, rank_data in rank_level_clustering_dict.items():
                rank_level_clustering_dfs[rank] = add_results_df(rank_data, batch_num=0)
            # Run garbage collection
            gc.collect()
            if gb.VERBOSE_TRAINING_TIMING:
                print(f"> Time taken for pre-training quick test: {time.time() - qt_start_time:.4f} seconds")
        else:
            clustering_scores_df = None
            classification_scores_df = None
            macro_classification_scores_df = None
            ssc_scores_df = None
            rank_level_clustering_dfs = {}

        # Initialise logging plotting dataframes if enabled
        triplet_satisfaction_df = None
        if gb.PLOT_TRIPLET_SATISFACTION_EVERY_N_BATCHES is not None and gb.LOG_TRIPLET_MINING:
            triplet_satisfaction_df = init_triplet_satisfaction_df()
        
        pair_distances_df = None
        if gb.PLOT_PAIR_DISTANCES_EVERY_N_BATCHES is not None and gb.LOG_PAIR_MINING:
            pair_distances_df = init_pair_distances_df()
        
        pair_error_metrics_df = None
        if gb.PLOT_PAIR_ERROR_METRICS_EVERY_N_BATCHES is not None and gb.LOG_PAIR_MINING:
            pair_error_metrics_df = init_pair_error_metrics_df()
        
        triplet_error_metrics_df = None
        if gb.PLOT_TRIPLET_ERROR_METRICS_EVERY_N_BATCHES is not None and gb.LOG_TRIPLET_MINING:
            triplet_error_metrics_df = init_triplet_error_metrics_df()


        # Start
        print("\nStarting training --------------------------")

        start_time = time.time()
        batch_num = 0
        while batch_num < gb.NUM_BATCHES:
            batch_num += 1
            batch_start_time = time.time()

            # Print batch number
            should_log_batch_info = batch_num in (1,) or batch_num % gb.PRINT_BATCH_NUM_EVERY_N_BATCHES == 0
            if should_log_batch_info:
                percentage_complete = ((batch_num - 1) / gb.NUM_BATCHES) * 100
                elapsed_time = time.time() - start_time
                time_str = "" if batch_num == 1 else f"  -  {elapsed_time/60:.0f}/{elapsed_time / batch_num * gb.NUM_BATCHES/60:.0f} min"
                # Get current LR
                current_lr = optimizer.param_groups[0]['lr']
                vram_str = get_vram_usage_str(device)
                # print("\n---")
                print(f"Batch {batch_num}/{gb.NUM_BATCHES}  ({percentage_complete:.1f}%{time_str}) - LR: {current_lr:.6f}{vram_str}")

            # Print and/or log uncertainty weighting parameters (if enabled)
            print_log_uncertainty_weighting(uncertainty_loss, batch_num=batch_num, logs_dir=LOSS_DIR)

            # Print and/or log conv stem scale (if enabled)
            print_log_conv_stem_scale(model, batch_num=batch_num, logs_dir=LOG_PLOTS_DIR)


            # Train -------------------

            # Mine pairs and triplets for batch
            triplets, pairs, triplet_margins, pair_distances, triplet_ranks, pair_ranks, pair_region_pairs, pair_buckets, triplet_buckets, triplet_indices, pair_indices = mine(batch_num, model, LOGS_DIR, triplet_satisfaction_df, pair_distances_df, triplet_error_metrics_df, pair_error_metrics_df)
            has_triplets_this_batch = triplets.shape[0] > 0
            has_pairs_this_batch = pairs.shape[0] > 0
          
            # Set model to training mode
            model.train()

            # Move tensors to the correct device
            transfer_start_time = time.time()
            # In sequenceless mode, use indices instead of sequences
            if gb.SEQLESS_MODE:
                # Use indices for sequenceless mode (sequences are ignored)
                triplets = triplet_indices.to(device, non_blocking=gb.PIN_MEMORY_FOR_MINING)
                pairs = pair_indices.to(device, non_blocking=gb.PIN_MEMORY_FOR_MINING)
            else:
                # Use sequences for standard mode (indices are ignored)
                triplets = triplets.to(device, non_blocking=gb.PIN_MEMORY_FOR_MINING)
                pairs = pairs.to(device, non_blocking=gb.PIN_MEMORY_FOR_MINING)
            triplet_margins = triplet_margins.to(device, non_blocking=gb.PIN_MEMORY_FOR_MINING)
            pair_distances = pair_distances.to(device, non_blocking=gb.PIN_MEMORY_FOR_MINING)
            # Keep these in fp32, they directly control loss magnitudes
            triplet_margins = triplet_margins.float()
            pair_distances = pair_distances.float()
            if pair_buckets is not None:
                pair_buckets = pair_buckets.to(device, non_blocking=gb.PIN_MEMORY_FOR_MINING)
            if triplet_buckets is not None:
                triplet_buckets = triplet_buckets.to(device, non_blocking=gb.PIN_MEMORY_FOR_MINING)
            if gb.VERBOSE_TRAINING_TIMING:
                device_desc = 'cpu' if device.type != 'cuda' else f"cuda:{device.index if device.index is not None else torch.cuda.current_device()}"
                print(f"> Time taken to transfer mined data to {device_desc}: {time.time() - transfer_start_time:.4f} seconds")

            # Forward + loss + backprop (with microbatching) ---
            # Inference/loss/backprop for each batch (which uses one mining run) is split into N microbatches
            optimizer.zero_grad(set_to_none=True)
            forward_start_time = time.time()

            # Verbose/log flags for loss functions (aggregated across all microbatches after loop)
            do_verbose_triplet, do_log_triplet = has_triplets_this_batch and gb.VERBOSE_TRIPLET_LOSS and batch_num % gb.VERBOSE_EVERY_N_BATCHES == 0, has_triplets_this_batch and gb.LOG_TRIPLET_LOSS and batch_num % gb.LOG_EVERY_N_BATCHES == 0
            do_verbose_pair, do_log_pair = has_pairs_this_batch and gb.VERBOSE_PAIR_LOSS and batch_num % gb.VERBOSE_EVERY_N_BATCHES == 0, has_pairs_this_batch and gb.LOG_PAIR_LOSS and batch_num % gb.LOG_EVERY_N_BATCHES == 0
            need_triplet_stats = do_verbose_triplet or do_log_triplet
            need_pair_stats = do_verbose_pair or do_log_pair
            use_uncertainty_this_batch = gb.IS_USING_UNCERTAINTY_WEIGHTING and has_triplets_this_batch and has_pairs_this_batch

            # Split mined data into microbatches along the batch dimension
            triplet_chunks = torch.chunk(triplets, gb.NUM_MICRO_BATCHES_PER_BATCH)
            pair_chunks = torch.chunk(pairs, gb.NUM_MICRO_BATCHES_PER_BATCH)
            triplet_margin_chunks = torch.chunk(triplet_margins, gb.NUM_MICRO_BATCHES_PER_BATCH)
            pair_distance_chunks = torch.chunk(pair_distances, gb.NUM_MICRO_BATCHES_PER_BATCH)
            triplet_rank_chunks = torch.chunk(triplet_ranks, gb.NUM_MICRO_BATCHES_PER_BATCH)
            pair_rank_chunks = torch.chunk(pair_ranks, gb.NUM_MICRO_BATCHES_PER_BATCH)
            pair_region_pair_chunks = torch.chunk(pair_region_pairs, gb.NUM_MICRO_BATCHES_PER_BATCH) if pair_region_pairs is not None else (None,) * gb.NUM_MICRO_BATCHES_PER_BATCH
            triplet_bucket_chunks = torch.chunk(triplet_buckets, gb.NUM_MICRO_BATCHES_PER_BATCH) if triplet_buckets is not None else (None,) * gb.NUM_MICRO_BATCHES_PER_BATCH
            pair_bucket_chunks = torch.chunk(pair_buckets, gb.NUM_MICRO_BATCHES_PER_BATCH) if pair_buckets is not None else (None,) * gb.NUM_MICRO_BATCHES_PER_BATCH

            all_triplet_losses = []
            all_pair_losses = []
            all_a_p_dists, all_a_n_dists, all_pred_dists = [], [], []
            for micro_idx in range(gb.NUM_MICRO_BATCHES_PER_BATCH):
                # Forward pass (autocast for speed and VRAM savings)
                with amp_ctx:
                    micro_triplet_emb = model(triplet_chunks[micro_idx])
                    micro_pair_emb = model(pair_chunks[micro_idx])

                # Losses in float32 for stability
                if micro_triplet_emb.dtype != torch.float32:
                    micro_triplet_emb = micro_triplet_emb.float()
                if micro_pair_emb.dtype != torch.float32:
                    micro_pair_emb = micro_pair_emb.float()

                # Compute per-element losses (verbose/log disabled here; stats aggregated across all microbatches after loop)
                micro_triplet_losses = embedding_triplet_loss(micro_triplet_emb, triplet_margin_chunks[micro_idx], triplet_rank_chunks[micro_idx], buckets=triplet_bucket_chunks[micro_idx], normalize_embeddings=False, return_dists=need_triplet_stats)
                micro_pair_losses = embedding_pair_loss(micro_pair_emb, pair_distance_chunks[micro_idx], pair_rank_chunks[micro_idx], buckets=pair_bucket_chunks[micro_idx], region_pairs=pair_region_pair_chunks[micro_idx], normalize_embeddings=False, return_dists=need_pair_stats)

                # Unpack and collect intermediate distances for full-batch aggregated logging
                if need_triplet_stats:
                    micro_triplet_losses, micro_ap, micro_an = micro_triplet_losses
                    all_a_p_dists.append(micro_ap); all_a_n_dists.append(micro_an)
                if need_pair_stats:
                    micro_pair_losses, micro_pd = micro_pair_losses
                    all_pred_dists.append(micro_pd)

                # Apply loss weights
                micro_triplet_losses = micro_triplet_losses * gb.TRIPLET_LOSS_WEIGHT
                micro_pair_losses = micro_pair_losses * gb.PAIR_LOSS_WEIGHT

                # Mean losses for this microbatch
                micro_mean_triplet = micro_triplet_losses.mean() if micro_triplet_losses.numel() > 0 else torch.zeros((), device=device)
                micro_mean_pair = micro_pair_losses.mean() if micro_pair_losses.numel() > 0 else torch.zeros((), device=device)

                # Combine losses (with or without uncertainty weighting)
                if not use_uncertainty_this_batch:
                    micro_total = micro_mean_triplet + micro_mean_pair
                else:
                    micro_wt, micro_wp = uncertainty_loss(micro_mean_triplet, micro_mean_pair)
                    micro_total = micro_wt + micro_wp

                # Scale for gradient accumulation and backward
                (micro_total / gb.NUM_MICRO_BATCHES_PER_BATCH).backward()

                # Collect per-element losses for logging (detached from computation graph)
                all_triplet_losses.append(micro_triplet_losses.detach())
                all_pair_losses.append(micro_pair_losses.detach())

            if gb.VERBOSE_TRAINING_TIMING:
                print(f"> Time taken for forward + loss + backward ({gb.NUM_MICRO_BATCHES_PER_BATCH} microbatch{'es' if gb.NUM_MICRO_BATCHES_PER_BATCH > 1 else ''}): {time.time() - forward_start_time:.4f} seconds")

            # Optimizer step
            optimizer_step_start_time = time.time()
            optimizer.step()
            if gb.VERBOSE_TRAINING_TIMING:
                print(f"> Time taken for optimizer step: {time.time() - optimizer_step_start_time:.4f} seconds")

            # Aggregate per-element losses across microbatches for logging
            triplet_losses = torch.cat(all_triplet_losses) if len(all_triplet_losses) > 1 else all_triplet_losses[0]
            pair_losses = torch.cat(all_pair_losses) if len(all_pair_losses) > 1 else all_pair_losses[0]
            mean_triplet_loss = triplet_losses.mean() if triplet_losses.numel() > 0 else torch.zeros((), device=device)
            mean_pair_loss = pair_losses.mean() if pair_losses.numel() > 0 else torch.zeros((), device=device)
            weighted_triplet_loss = None
            weighted_pair_loss = None
            if not use_uncertainty_this_batch:
                total_loss = mean_triplet_loss + mean_pair_loss
            else:
                with torch.no_grad():
                    weighted_triplet_loss, weighted_pair_loss = uncertainty_loss(mean_triplet_loss, mean_pair_loss)
                total_loss = weighted_triplet_loss + weighted_pair_loss

            # Full-batch aggregated verbose printing and logging (across all microbatches)
            if need_triplet_stats:
                full_a_p = torch.cat(all_a_p_dists) if len(all_a_p_dists) > 1 else all_a_p_dists[0]
                full_a_n = torch.cat(all_a_n_dists) if len(all_a_n_dists) > 1 else all_a_n_dists[0]
                raw_triplet = triplet_losses / gb.TRIPLET_LOSS_WEIGHT if gb.TRIPLET_LOSS_WEIGHT > 0 else triplet_losses
                if do_verbose_triplet:
                    print_triplet_loss_stats(full_a_p, full_a_n, triplet_margins, raw_triplet, triplet_ranks, triplet_buckets)
                if do_log_triplet:
                    write_triplet_loss_log(batch_num, LOGS_DIR, full_a_p, full_a_n, triplet_margins, raw_triplet, triplet_ranks, triplet_buckets)
            if need_pair_stats:
                full_pred = torch.cat(all_pred_dists) if len(all_pred_dists) > 1 else all_pred_dists[0]
                raw_pair = pair_losses / gb.PAIR_LOSS_WEIGHT if gb.PAIR_LOSS_WEIGHT > 0 else pair_losses
                if do_verbose_pair:
                    print_pair_loss_stats(raw_pair, full_pred, pair_distances, pair_ranks, pair_buckets, region_pairs=pair_region_pairs)
                if do_log_pair:
                    write_pair_loss_log(batch_num, LOGS_DIR, raw_pair, full_pred, pair_distances, pair_ranks, pair_buckets, region_pairs=pair_region_pairs)

            # Update LR Scheduler
            if scheduler is not None:
                scheduler_start_time = time.time()
                scheduler.step()
                if gb.VERBOSE_TRAINING_TIMING:
                    print(f"> Time taken for scheduler step: {time.time() - scheduler_start_time:.4f} seconds")

            # Save model (skip in sequenceless mode)
            if batch_num % gb.SAVE_EVERY_N_BATCHES == 0 and not gb.SEQLESS_MODE:
                checkpoint_save_start_time = time.time()
                checkpoint_path = EVERY_N_BATCHES_MODEL_PATH(batch_num)
                model.save_model(checkpoint_path)
                print(f"Saved checkpoint (batch {batch_num}) ------------------------")
                if gb.VERBOSE_TRAINING_TIMING:
                    print(f"> Time taken to save checkpoint: {time.time() - checkpoint_save_start_time:.4f} seconds")


            # Quick test (skip in sequenceless mode)
            if gb.DO_QT and batch_num % gb.QT_EVERY_N_BATCHES == 0:
                print(f"\nQuick test (batch {batch_num}) -----------------------")
                qt_start_time = time.time()
                model.eval()
                clustering_scores_dict, classification_scores_dict, macro_classification_scores_dict, ssc_scores_dict, rank_level_clustering_dict = quick_test(model=model, model_dir=MODEL_DIR, batch_num=batch_num, verbose=gb.QT_VERBOSE)
                # Return model to device
                model = model.to(device)
                print_quick_test(clustering_scores_dict, classification_scores_dict, macro_classification_scores_dict, ssc_scores_dict)
                clustering_scores_df = add_results_df(clustering_scores_dict, batch_num=batch_num, existing_df=clustering_scores_df)
                classification_scores_df = add_results_df(classification_scores_dict, batch_num=batch_num, existing_df=classification_scores_df)
                macro_classification_scores_df = add_results_df(macro_classification_scores_dict, batch_num=batch_num, existing_df=macro_classification_scores_df)
                ssc_scores_df = add_results_df(ssc_scores_dict, batch_num=batch_num, existing_df=ssc_scores_df)
                # Update per-rank clustering DataFrames
                for rank, rank_data in rank_level_clustering_dict.items():
                    existing_rank_df = rank_level_clustering_dfs.get(rank, None)
                    rank_level_clustering_dfs[rank] = add_results_df(rank_data, batch_num=batch_num, existing_df=existing_rank_df)
                        
                # Plot the desired quick test results (customisable)
                if not gb.SEQLESS_MODE:
                    # --- Classification ---
                    plot_quick_test(classification_scores_df, QT_PLOTS_DIR, column_name='embedding_train-set_class_rank-*_subseqs', show_kmer_equivalents=True, name="Classification (Train Set, Subseqs)")
                    if HAS_TEST_SET:
                        plot_quick_test(classification_scores_df, QT_PLOTS_DIR, column_name='embedding_test-set_class_rank-*_subseqs', show_kmer_equivalents=True, name="Classification (Test Set, Subseqs)")
                    if HAS_EXCLUDED_SET:
                        plot_quick_test(classification_scores_df, QT_PLOTS_DIR, column_name='embedding_excluded-set_class_rank-*_subseqs', show_kmer_equivalents=True, name="Classification (Excluded Set, Subseqs)")
                    # --- Macro Classification ---
                    plot_quick_test(macro_classification_scores_df, QT_PLOTS_DIR, column_name='macro_embedding_train-set_class_rank-*_subseqs', show_kmer_equivalents=True, name="Macro Classification (Train Set, Subseqs)")
                    if HAS_TEST_SET:
                        plot_quick_test(macro_classification_scores_df, QT_PLOTS_DIR, column_name='macro_embedding_test-set_class_rank-*_subseqs', show_kmer_equivalents=True, name="Macro Classification (Test Set, Subseqs)")
                    if HAS_EXCLUDED_SET:
                        plot_quick_test(macro_classification_scores_df, QT_PLOTS_DIR, column_name='macro_embedding_excluded-set_class_rank-*_subseqs', show_kmer_equivalents=True, name="Macro Classification (Excluded Set, Subseqs)")
                    # --- Clustering ---
                    plot_quick_test(clustering_scores_df, QT_PLOTS_DIR, column_name='embedding_train-set_clust_rank-*_subseqs', show_kmer_equivalents=True, name="Clustering (Train Set, Subseqs)")
                    if HAS_TEST_SET:
                        plot_quick_test(clustering_scores_df, QT_PLOTS_DIR, column_name='embedding_test-set_clust_rank-*_subseqs', show_kmer_equivalents=True, name="Clustering (Test Set, Subseqs)")
                    if HAS_EXCLUDED_SET:
                        plot_quick_test(clustering_scores_df, QT_PLOTS_DIR, column_name='embedding_excluded-set_clust_rank-*_subseqs', show_kmer_equivalents=True, name="Clustering (Excluded Set, Subseqs)")
                    # --- Subsequence Congruency ---
                    plot_quick_test(ssc_scores_df, QT_PLOTS_DIR, column_name='embedding_*-set_ssc_mean', show_kmer_equivalents=True, name="Subsequence Congruency")

                    # --- Rank-level Clustering ---
                    plot_rank_level_clustering(rank_level_clustering_dfs, QT_PLOTS_DIR, batch_num)
                else:
                    # Sequenceless mode: train-full embeddings only
                    plot_quick_test(clustering_scores_df, QT_PLOTS_DIR, column_name='embedding_train-set_clust_rank-*_full', show_kmer_equivalents=False, name="Clustering (Train Set, Full)")
                    plot_rank_level_clustering(rank_level_clustering_dfs, QT_PLOTS_DIR, batch_num)

                # Write quick test results to CSVs
                macro_classification_scores_df.to_csv(QT_RESULTS_CSV_DIR + "macro_classification.csv", index=False)
                classification_scores_df.to_csv(QT_RESULTS_CSV_DIR + "classification.csv", index=False)
                clustering_scores_df.to_csv(QT_RESULTS_CSV_DIR + "clustering.csv", index=False)
                ssc_scores_df.to_csv(QT_RESULTS_CSV_DIR + "ssc.csv", index=False)
                # Write per-rank clustering CSVs
                if rank_level_clustering_dfs:
                    from quick_test import ALL_RANKS_PLURAL
                    rank_clust_dir = QT_RESULTS_CSV_DIR + "clustering_per_rank/"
                    os.makedirs(rank_clust_dir, exist_ok=True)
                    for rank, rank_df in rank_level_clustering_dfs.items():
                        rank_name = ALL_RANKS_PLURAL[rank]
                        rank_df.to_csv(rank_clust_dir + f"{rank_name}.csv", index=False)
                
                # Run garbage collection
                gc.collect()

                if gb.VERBOSE_TRAINING_TIMING:
                    print(f"> Time taken for quick test: {time.time() - qt_start_time:.4f} seconds")
                print("\nResuming training --------------------------")

            # Record losses
            if batch_num % gb.RECORD_LOSS_EVERY_N_BATCHES == 0:
                loss_logging_start_time = time.time()
                # Add loss to dataframe
                loss_results_df = add_loss_results_df(mean_triplet_loss, mean_pair_loss, triplet_losses, triplet_ranks,
                                                      pair_losses, pair_ranks, batch_num=batch_num, total_loss=total_loss,
                                                      weighted_triplet_loss=weighted_triplet_loss, weighted_pair_loss=weighted_pair_loss,
                                                      existing_df=loss_results_df)
                # Save loss to CSV
                loss_results_df.to_csv(LOSS_DIR + "losses.csv", index=False)
                # Plot loss
                if batch_num % gb.PLOT_LOSS_EVERY_N_BATCHES == 0:
                    if gb.PLOT_TOTAL_LOSS and loss_results_df['Total_Loss'].any():
                        plot_loss_results(loss_results_df, LOSS_DIR + "losses_total.png", plot_total=True, plot_triplets=False, plot_pairs=False)
                        if gb.SECOND_LOSS_PLOT_AFTER_N_BATCHES is not None and batch_num >= gb.SECOND_LOSS_PLOT_AFTER_N_BATCHES:
                            plot_loss_results(loss_results_df, LOSS_DIR + "losses_total_2.png", truncate_to_after_n_batches=gb.SECOND_LOSS_PLOT_AFTER_N_BATCHES, plot_total=True, plot_triplets=False, plot_pairs=False)

                    if gb.PLOT_TRIPLET_LOSS and loss_results_df['Triplet_Loss'].any():
                        plot_loss_results(loss_results_df, LOSS_DIR + "losses_triplets.png", plot_triplets=True, plot_pairs=False)
                        if gb.SECOND_LOSS_PLOT_AFTER_N_BATCHES is not None and batch_num >= gb.SECOND_LOSS_PLOT_AFTER_N_BATCHES:
                            plot_loss_results(loss_results_df, LOSS_DIR + "losses_triplets_2.png", truncate_to_after_n_batches=gb.SECOND_LOSS_PLOT_AFTER_N_BATCHES, plot_triplets=True, plot_pairs=False)
                    
                    if gb.PLOT_PAIR_LOSS and loss_results_df['Pair_Loss'].any():
                        plot_loss_results(loss_results_df, LOSS_DIR + "losses_pairs.png", plot_triplets=False, plot_pairs=True)
                        if gb.SECOND_LOSS_PLOT_AFTER_N_BATCHES is not None and batch_num >= gb.SECOND_LOSS_PLOT_AFTER_N_BATCHES:
                            plot_loss_results(loss_results_df, LOSS_DIR + "losses_pairs_2.png", truncate_to_after_n_batches=gb.SECOND_LOSS_PLOT_AFTER_N_BATCHES, plot_triplets=False, plot_pairs=True)

                    if gb.PLOT_UNCERTAINTY_WEIGHTING and gb.IS_USING_UNCERTAINTY_WEIGHTING:
                        plot_uncertainty_weights(LOSS_DIR + "uncertainty_weighting.csv", LOSS_DIR + "uncertainty_weights.png", log_var_output_path=LOSS_DIR + "log_vars.png", y_min_zero=False)
                        if gb.SECOND_LOSS_PLOT_AFTER_N_BATCHES is not None and batch_num >= gb.SECOND_LOSS_PLOT_AFTER_N_BATCHES:
                            plot_uncertainty_weights(LOSS_DIR + "uncertainty_weighting.csv", LOSS_DIR + "uncertainty_weights_2.png", log_var_output_path=LOSS_DIR + "log_vars_2.png", truncate_to_after_n_batches=gb.SECOND_LOSS_PLOT_AFTER_N_BATCHES, y_min_zero=False)
                if gb.VERBOSE_TRAINING_TIMING:
                    print(f"> Time taken to record/log losses: {time.time() - loss_logging_start_time:.4f} seconds")
                    
            # Logging plotting (conv stem scale)
            if gb.USE_CONV_STEM and gb.LOG_CONV_STEM_SCALE and gb.PLOT_CONV_STEM_SCALE_EVERY_N_BATCHES is not None:
                if batch_num % gb.PLOT_CONV_STEM_SCALE_EVERY_N_BATCHES == 0:
                    conv_stem_scale_logging_start_time = time.time()
                    conv_stem_csv_path = LOG_PLOTS_DIR + "conv_stem_scale.csv"
                    if os.path.exists(conv_stem_csv_path):
                        plot_conv_stem_scale(conv_stem_csv_path, LOG_PLOTS_DIR + "conv_stem_scale.png")
                        if gb.SECOND_CONV_STEM_SCALE_PLOT_AFTER_N_BATCHES is not None and batch_num >= gb.SECOND_CONV_STEM_SCALE_PLOT_AFTER_N_BATCHES:
                            plot_conv_stem_scale(conv_stem_csv_path, LOG_PLOTS_DIR + "conv_stem_scale_2.png", truncate_to_after_n_batches=gb.SECOND_CONV_STEM_SCALE_PLOT_AFTER_N_BATCHES)
                    if gb.VERBOSE_TRAINING_TIMING:
                        print(f"> Time taken to update conv stem scale plots: {time.time() - conv_stem_scale_logging_start_time:.4f} seconds")

            # Logging plotting (triplet satisfaction)
            if triplet_satisfaction_df is not None and gb.PLOT_TRIPLET_SATISFACTION_EVERY_N_BATCHES is not None:
                if batch_num % gb.PLOT_TRIPLET_SATISFACTION_EVERY_N_BATCHES == 0 and len(triplet_satisfaction_df) > 0:
                    triplet_satisfaction_logging_start_time = time.time()
                    save_triplet_satisfaction_csv(triplet_satisfaction_df, LOG_PLOTS_DIR + "triplet_satisfaction.csv")
                    plot_triplet_satisfaction(triplet_satisfaction_df, LOG_PLOTS_DIR + "triplet_satisfaction.png")
                    if gb.SECOND_TRIPLET_SATISFACTION_PLOT_AFTER_N_BATCHES is not None and batch_num >= gb.SECOND_TRIPLET_SATISFACTION_PLOT_AFTER_N_BATCHES:
                        plot_triplet_satisfaction(triplet_satisfaction_df, LOG_PLOTS_DIR + "triplet_satisfaction_2.png", truncate_to_after_n_batches=gb.SECOND_TRIPLET_SATISFACTION_PLOT_AFTER_N_BATCHES, y_min_zero=False)
                    if gb.VERBOSE_TRAINING_TIMING:
                        print(f"> Time taken to update triplet satisfaction plots: {time.time() - triplet_satisfaction_logging_start_time:.4f} seconds")
            
            # Logging plotting (pair distances)
            if pair_distances_df is not None and gb.PLOT_PAIR_DISTANCES_EVERY_N_BATCHES is not None:
                if batch_num % gb.PLOT_PAIR_DISTANCES_EVERY_N_BATCHES == 0 and len(pair_distances_df) > 0:
                    pair_distances_logging_start_time = time.time()
                    save_pair_distances_csv(pair_distances_df, LOG_PLOTS_DIR + "pair_distances.csv")
                    plot_pair_distances(pair_distances_df, LOG_PLOTS_DIR + "pair_distances.png")
                    plot_pair_distances_log2(pair_distances_df, LOG_PLOTS_DIR + "pair_distances_log2.png")
                    if gb.VERBOSE_TRAINING_TIMING:
                        print(f"> Time taken to update pair distance plots: {time.time() - pair_distances_logging_start_time:.4f} seconds")
            
            # Logging plotting (pair per-rank error metrics)
            if pair_error_metrics_df is not None and gb.PLOT_PAIR_ERROR_METRICS_EVERY_N_BATCHES is not None:
                if batch_num % gb.PLOT_PAIR_ERROR_METRICS_EVERY_N_BATCHES == 0 and len(pair_error_metrics_df) > 0:
                    pair_error_logging_start_time = time.time()
                    save_pair_error_metrics_csv(pair_error_metrics_df, LOG_PLOTS_DIR + "pair_per_rank_error_metrics.csv")
                    plot_pair_error_metrics(pair_error_metrics_df, LOG_PLOTS_DIR + "pair_per_rank_error_metrics.png")
                    if gb.SECOND_PAIR_ERROR_METRICS_PLOT_AFTER_N_BATCHES is not None and batch_num >= gb.SECOND_PAIR_ERROR_METRICS_PLOT_AFTER_N_BATCHES:
                        plot_pair_error_metrics(pair_error_metrics_df, LOG_PLOTS_DIR + "pair_per_rank_error_metrics_2.png", truncate_to_after_n_batches=gb.SECOND_PAIR_ERROR_METRICS_PLOT_AFTER_N_BATCHES)
                    if gb.VERBOSE_TRAINING_TIMING:
                        print(f"> Time taken to update pair error metric plots: {time.time() - pair_error_logging_start_time:.4f} seconds")
            
            # Logging plotting (triplet per-rank error metrics)
            if triplet_error_metrics_df is not None and gb.PLOT_TRIPLET_ERROR_METRICS_EVERY_N_BATCHES is not None:
                if batch_num % gb.PLOT_TRIPLET_ERROR_METRICS_EVERY_N_BATCHES == 0 and len(triplet_error_metrics_df) > 0:
                    triplet_error_logging_start_time = time.time()
                    save_triplet_error_metrics_csv(triplet_error_metrics_df, LOG_PLOTS_DIR + "triplet_per_rank_error_metrics.csv")
                    plot_triplet_error_metrics(triplet_error_metrics_df, LOG_PLOTS_DIR + "triplet_per_rank_error_metrics.png")
                    if gb.SECOND_TRIPLET_ERROR_METRICS_PLOT_AFTER_N_BATCHES is not None and batch_num >= gb.SECOND_TRIPLET_ERROR_METRICS_PLOT_AFTER_N_BATCHES:
                        plot_triplet_error_metrics(triplet_error_metrics_df, LOG_PLOTS_DIR + "triplet_per_rank_error_metrics_2.png", truncate_to_after_n_batches=gb.SECOND_TRIPLET_ERROR_METRICS_PLOT_AFTER_N_BATCHES)
                    if gb.VERBOSE_TRAINING_TIMING:
                        print(f"> Time taken to update triplet error metric plots: {time.time() - triplet_error_logging_start_time:.4f} seconds")

            # Free memory every 100 batches
            if batch_num % 100 == 0:
                del triplets, pairs, triplet_margins, pair_distances, triplet_ranks, pair_ranks
                del pair_region_pairs, pair_buckets, triplet_buckets, triplet_indices, pair_indices
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                gc.collect()

            if gb.VERBOSE_TRAINING_TIMING:
                print(f"> Time taken for training batch: {time.time() - batch_start_time:.4f} seconds\n---")

        # Done (skip model save in sequenceless mode)
        if not gb.SEQLESS_MODE:
            model.save_model(LATEST_MODEL_PATH)
            print(f"Training complete. Final model saved at: {LATEST_MODEL_PATH}")
        else:
            print(f"Training complete. (Model not saved in sequenceless mode)")
        end_time = time.time()
        elapsed = end_time - start_time
        print(f"Training time: {elapsed/60:.2f} minutes")
        write_about_training(output_path=ABOUT_TRAINING_PATH, model_name=MODEL_NAME, model_dir=MODEL_DIR, training_run=TRAINING_RUN, training_regimine=TRAINING_REGIMINE, description=tc.DESCRIPTION, start_time=start_time, end_time=end_time, batches_completed=batch_num, device=device, logs_dir=LOGS_DIR, qt_results_dir=QT_RESULTS_CSV_DIR, qt_plots_dir=QT_PLOTS_DIR, latest_model_path=LATEST_MODEL_PATH, loss_results_df=loss_results_df, clustering_scores_df=clustering_scores_df, classification_scores_df=classification_scores_df, macro_classification_scores_df=macro_classification_scores_df, ssc_scores_df=ssc_scores_df)
        print(f"Training summary written to: {ABOUT_TRAINING_PATH}")
