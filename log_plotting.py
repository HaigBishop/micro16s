"""
Log Plotting Module

This module provides functionality to track and visualize training metrics from mining logs
during Micro16S model training. It allows for recording metric values across training batches 
and creating plots to monitor training progress.

Current Features:
- Triplet satisfaction plotting: Shows % of triplets where AP+M < AN (margin satisfied) over time
- Pair distances plotting: Box plot showing true vs predicted cosine distances by rank over time

Future additions will include additional mining and loss metrics plots.

Key Functions:
- init_triplet_satisfaction_df: Initialize empty dataframe for triplet satisfaction tracking
- add_triplet_satisfaction_row: Add a row of triplet satisfaction data to the dataframe
- plot_triplet_satisfaction: Create a line plot showing triplet satisfaction by rank over time
- init_pair_distances_df: Initialize empty dataframe for pair distances tracking
- add_pair_distances_row_inplace: Add a row of pair distances data to the dataframe
- plot_pair_distances: Create a box plot showing true vs predicted distances by rank
"""


# IMPORTS ================================================
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os



# CONSTANTS ================================================
# Triplet rank map: triplet_rank -> display name
TRIPLET_RANK_MAP = {0: 'Domain', 1: 'Phylum', 2: 'Class', 3: 'Order', 4: 'Family', 5: 'Genus'}

# Pair rank map: pair_rank -> display name (pair_rank = shared_rank + 1, or 7 for subsequence)
# Note: -1 = Domain (different domains), 0-6 = standard taxonomy ranks, 7 = Subseq
PAIR_RANK_MAP = {-1: 'Domain', 0: 'Phylum', 1: 'Class', 2: 'Order', 3: 'Family', 
                 4: 'Genus', 5: 'Species', 6: 'Sequence', 7: 'Subseq'}



# TRIPLET SATISFACTION ================================================

def init_triplet_satisfaction_df():
    """
    Initialize an empty dataframe for tracking triplet satisfaction metrics over training.
    
    Pre-initializes all rank and overall columns to avoid DataFrame fragmentation warnings.
    The dataframe will be populated row by row during training via add_triplet_satisfaction_row().
    
    Returns:
        pd.DataFrame: DataFrame with 'Batch', 'Overall', and rank columns initialized.
    """
    columns = ['Batch', 'Overall']
    for rank_name in TRIPLET_RANK_MAP.values():
        columns.append(rank_name)
    return pd.DataFrame(columns=columns)


def add_triplet_satisfaction_row(df, batch_num, satisfaction_by_rank, overall_satisfaction):
    """
    Add a row of triplet satisfaction data to the dataframe.
    
    Args:
        df (pd.DataFrame): Existing triplet satisfaction dataframe.
        batch_num (int): The current batch number.
        satisfaction_by_rank (dict): Dictionary mapping rank (int) to satisfaction % (float).
                                     e.g. {0: 85.2, 1: 72.1, 2: 68.5, ...}
        overall_satisfaction (float): Overall satisfaction % across all ranks.
    
    Returns:
        pd.DataFrame: Updated dataframe with the new row added.
    """
    row_dict = {'Batch': batch_num, 'Overall': overall_satisfaction}
    
    # Add per-rank satisfaction values
    for rank, satisfaction in satisfaction_by_rank.items():
        rank_name = TRIPLET_RANK_MAP.get(rank, f'Rank-{rank}')
        row_dict[rank_name] = satisfaction
    
    # Create new row and append
    new_row_df = pd.DataFrame([row_dict])
    
    if df is None or len(df) == 0:
        updated_df = new_row_df
    else:
        updated_df = pd.concat([df, new_row_df], ignore_index=True)
    
    return updated_df


def add_triplet_satisfaction_row_inplace(df, batch_num, satisfaction_by_rank, overall_satisfaction):
    """
    Add a row of triplet satisfaction data to the dataframe in-place.
    
    This function modifies the dataframe directly, suitable for passing dataframe
    references through function call chains.
    
    Args:
        df (pd.DataFrame): Existing triplet satisfaction dataframe (modified in-place).
        batch_num (int): The current batch number.
        satisfaction_by_rank (dict): Dictionary mapping rank (int) to satisfaction % (float).
                                     e.g. {0: 85.2, 1: 72.1, 2: 68.5, ...}
        overall_satisfaction (float): Overall satisfaction % across all ranks.
    """
    if df is None:
        return
    
    row_dict = {'Batch': batch_num, 'Overall': overall_satisfaction}
    
    # Add per-rank satisfaction values
    for rank, satisfaction in satisfaction_by_rank.items():
        rank_name = TRIPLET_RANK_MAP.get(rank, f'Rank-{rank}')
        row_dict[rank_name] = satisfaction
    
    # Ensure all columns exist in the dataframe
    for col in row_dict.keys():
        if col not in df.columns:
            df[col] = pd.NA
    
    # Add the new row using loc
    df.loc[len(df)] = row_dict


def plot_triplet_satisfaction(df, output_path, show=False, truncate_to_after_n_batches=None, y_min_zero=True):
    """
    Create a line plot showing triplet satisfaction (% AP+M < AN) by rank over time.
    
    This follows the same style as quick test plots (plot_quick_test) with:
    - Lines for each rank present in the data
    - Legend showing rank names with max values like "Domain (Max=1.000)"
    - Y-axis ranging 0-100% (unless y_min_zero=False)
    - Grid lines and clean formatting
    
    Args:
        df (pd.DataFrame): DataFrame containing triplet satisfaction data with columns:
            - 'Batch': Batch numbers
            - 'Overall': Overall satisfaction %
            - Rank columns (e.g., 'Domain', 'Phylum', etc.): Per-rank satisfaction %
        output_path (str): File path where the plot should be saved (should include .png extension).
        show (bool, optional): Whether to display the plot interactively. Defaults to False.
        truncate_to_after_n_batches (int, optional): If provided, the plot will be truncated to batches after this number. Defaults to None.
        y_min_zero (bool, optional): Whether to force the y-axis to start at zero. Defaults to True.
    """
    # Truncate the dataframe if requested
    if truncate_to_after_n_batches is not None:
        df = df[df['Batch'] >= truncate_to_after_n_batches].copy()

    # Check if there's any data to plot
    if df is None or len(df) == 0:
        print("Warning: No triplet satisfaction data available for plotting.")
        return
    
    # Get rank columns and sort them correctly according to the rank hierarchy
    # (Domain, Phylum, Class, Order, Family, Genus)
    rank_names = list(TRIPLET_RANK_MAP.values())
    present_rank_columns = [col for col in rank_names if col in df.columns]
    
    # Check if 'Overall' is present
    has_overall = 'Overall' in df.columns
    
    if not present_rank_columns and not has_overall:
        print("Warning: No triplet satisfaction columns found for plotting.")
        return
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    
    # Define a color cycle for consistent rank coloring
    # Using a qualitative colormap that looks good
    colors = plt.cm.tab10.colors
    
    # 1. Plot each rank column first (following color cycle)
    for i, column in enumerate(present_rank_columns):
        # Get non-null values for this column
        valid_data = df[['Batch', column]].dropna()
        if len(valid_data) == 0:
            continue
        
        # Calculate max score for legend
        max_score = valid_data[column].max()
        label = f"{column} (Max={max_score:.1f}%)"
        
        # Use color cycle for ranks
        color = colors[i % len(colors)]
        
        # Only show markers if less than 20 data points
        if len(valid_data) <= 20:
            plt.plot(valid_data['Batch'].astype(int), valid_data[column], 
                     marker='o', label=label, linewidth=2, markersize=4,
                     color=color)
        else:
            plt.plot(valid_data['Batch'].astype(int), valid_data[column], 
                     label=label, linewidth=2, color=color)

    # 2. Plot 'Overall' last (dark grey)
    if has_overall:
        column = 'Overall'
        valid_data = df[['Batch', column]].dropna()
        if len(valid_data) > 0:
            max_score = valid_data[column].max()
            label = f"{column} (Max={max_score:.1f}%)"
            color = '#333333' # Dark grey
            
            if len(valid_data) <= 20:
                plt.plot(valid_data['Batch'].astype(int), valid_data[column], 
                         marker='o', label=label, linewidth=2.5, markersize=4,
                         color=color)
            else:
                plt.plot(valid_data['Batch'].astype(int), valid_data[column], 
                         label=label, linewidth=2.5, color=color)

    
    # Customize the plot
    plt.xlabel('Batch Number', fontsize=12)
    plt.ylabel('Satisfied Triplets (% AP+M<AN)', fontsize=12)
    plt.title('Triplet Satisfaction Over Training', fontsize=14, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Set y-axis to 0-100 range
    if y_min_zero:
        plt.ylim(0, 100)
        plt.yticks(np.arange(0, 110, 10))
    else:
        plt.ylim(top=100)
    
    # Position legend outside plot area (same as quick test plots)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.tight_layout()
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Save the plot
    try:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    except Exception as e:
        print(f"Error saving triplet satisfaction plot to {output_path}: {e}")
    
    # Show the plot if requested
    if show:
        plt.show()
    
    # Close the figure to free memory
    plt.close('all')


def save_triplet_satisfaction_csv(df, output_path):
    """
    Save triplet satisfaction dataframe to CSV.
    
    Args:
        df (pd.DataFrame): Triplet satisfaction dataframe.
        output_path (str): File path for the CSV file.
    """
    if df is None or len(df) == 0:
        return
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    try:
        df.to_csv(output_path, index=False)
    except Exception as e:
        print(f"Error saving triplet satisfaction CSV to {output_path}: {e}")



# PAIR DISTANCES ================================================
# Box plot showing true vs predicted cosine distances by rank over training.
# The plot shows distributions of distances for each rank, allowing comparison of 
# model predictions against true (phylogenetic) distances.

def init_pair_distances_df():
    """
    Initialize an empty dataframe for tracking pair distances metrics over training.
    
    The dataframe stores box plot statistics (min, q1, median, q3, max) for both true 
    and predicted distances at each rank, along with running minimum relative error.
    
    Pre-initializes all columns to avoid DataFrame fragmentation warnings.
    
    Returns:
        pd.DataFrame: Initialized dataframe with all expected columns.
    """
    columns = ['Batch']
    
    # Define stats we track for each rank
    stats = ['true_min', 'true_q1', 'true_median', 'true_q3', 'true_max',
             'pred_min', 'pred_q1', 'pred_median', 'pred_q3', 'pred_max',
             'mean_rel_error']
    
    # Add per-rank columns
    for rank_name in PAIR_RANK_MAP.values():
        for stat in stats:
            columns.append(f'{rank_name}_{stat}')
            
    # Add overall columns
    for stat in stats:
        columns.append(f'Overall_{stat}')
        
    return pd.DataFrame(columns=columns)


def add_pair_distances_row_inplace(df, batch_num, distances_by_rank, overall_distances):
    """
    Add a row of pair distances data to the dataframe in-place.
    
    This function modifies the dataframe directly, suitable for passing dataframe
    references through function call chains.
    
    Args:
        df (pd.DataFrame): Existing pair distances dataframe (modified in-place).
        batch_num (int): The current batch number.
        distances_by_rank (dict): Dictionary mapping rank (int) to dict with keys:
                                  'true_min', 'true_q1', 'true_median', 'true_q3', 'true_max',
                                  'pred_min', 'pred_q1', 'pred_median', 'pred_q3', 'pred_max',
                                  'mean_rel_error'
        overall_distances (dict): Same structure as distances_by_rank for overall stats.
    """
    if df is None:
        return
    
    row_dict = {'Batch': batch_num}
    
    # Add per-rank distance statistics
    for rank, stats in distances_by_rank.items():
        rank_name = PAIR_RANK_MAP.get(rank, f'Rank-{rank}')
        for stat_key, stat_val in stats.items():
            col_name = f'{rank_name}_{stat_key}'
            row_dict[col_name] = stat_val
    
    # Add overall distance statistics
    if overall_distances:
        for stat_key, stat_val in overall_distances.items():
            col_name = f'Overall_{stat_key}'
            row_dict[col_name] = stat_val
    
    # Ensure all columns exist in the dataframe
    for col in row_dict.keys():
        if col not in df.columns:
            df[col] = pd.NA
    
    # Add the new row using loc
    df.loc[len(df)] = row_dict


def plot_pair_distances(df, output_path, show=False):
    """
    Create a box plot showing true vs predicted cosine distances by rank.
    
    This follows a similar style to the triplet satisfaction plot with:
    - Box plots for each rank showing true and predicted distance distributions
    - X-axis: Domain (True), Domain (Pred), ..., Overall (True), Overall (Pred)
    - Y-axis: Cosine Distance (0.0-2.0)
    - Legend showing rank names with min relative error like "Domain (Min Rel Err=0.102)"
    
    Args:
        df (pd.DataFrame): DataFrame containing pair distances data with columns:
            - 'Batch': Batch numbers
            - Per-rank columns: '{Rank}_true_min', '{Rank}_true_q1', '{Rank}_true_median', 
              '{Rank}_true_q3', '{Rank}_true_max', '{Rank}_pred_*', '{Rank}_mean_rel_error',
              '{Rank}_min_rel_error'
        output_path (str): File path where the plot should be saved (should include .png extension).
        show (bool, optional): Whether to display the plot interactively. Defaults to False.
    """
    _plot_pair_distances_base(df, output_path, show=show, log_scale=False)


def plot_pair_distances_log2(df, output_path, show=False):
    """
    Create a box plot showing true vs predicted cosine distances by rank with log2 Y-axis.
    
    This is identical to plot_pair_distances but uses a symlog scale (base 2) for the Y-axis.
    
    Args:
        df (pd.DataFrame): DataFrame containing pair distances data.
        output_path (str): File path where the plot should be saved (should include .png extension).
        show (bool, optional): Whether to display the plot interactively. Defaults to False.
    """
    _plot_pair_distances_base(df, output_path, show=show, log_scale=True)


def _plot_pair_distances_base(df, output_path, show=False, log_scale=False):
    """
    Implementation of pair distances plotting.
    """
    # Check if there's any data to plot
    if df is None or len(df) == 0:
        print("Warning: No pair distances data available for plotting.")
        return
    
    # Get the most recent row for plotting (box plot shows current state)
    latest_row = df.iloc[-1]
    
    # Define the rank order for consistent display
    # Note: -1 = Domain in pair mining, but is displayed as 'Domain'
    rank_order = [-1, 0, 1, 2, 3, 4, 5, 6, 7]  # Domain through Subseq
    rank_names = [PAIR_RANK_MAP.get(r, f'Rank-{r}') for r in rank_order]
    
    # Determine which ranks are present in the data
    present_ranks = []
    for rank_val, rank_name in zip(rank_order, rank_names):
        true_median_col = f'{rank_name}_true_median'
        if true_median_col in df.columns and pd.notna(latest_row.get(true_median_col)):
            present_ranks.append((rank_val, rank_name))
    
    # Add 'Overall' if present
    has_overall = 'Overall_true_median' in df.columns and pd.notna(latest_row.get('Overall_true_median'))
    
    if not present_ranks and not has_overall:
        print("Warning: No pair distances columns found for plotting.")
        return
    
    # Prepare box plot data
    # We'll create box plot statistics for matplotlib's bxp() function
    box_data = []
    x_labels = []
    colors = []
    legend_items = []
    
    # Define a color cycle for consistent rank coloring (matching triplet satisfaction)
    color_cycle = plt.cm.tab10.colors
    
    # Build box plot data for each rank
    for idx, (rank_val, rank_name) in enumerate(present_ranks):
        color = color_cycle[idx % len(color_cycle)]
        
        # Get min relative error for legend
        min_rel_error_col = f'{rank_name}_min_rel_error'
        min_rel_error = latest_row.get(min_rel_error_col, pd.NA)
        if pd.isna(min_rel_error):
            # Fallback to mean_rel_error if min not available
            mean_rel_error_col = f'{rank_name}_mean_rel_error'
            min_rel_error = latest_row.get(mean_rel_error_col, 0.0)
        
        # Legend entry
        legend_items.append((rank_name, color, min_rel_error))
        
        # True distances box stats
        true_stats = _get_box_stats(latest_row, rank_name, 'true')
        if true_stats:
            box_data.append(true_stats)
            x_labels.append(f'{rank_name}\n(True)')
            colors.append(color)
        
        # Predicted distances box stats
        pred_stats = _get_box_stats(latest_row, rank_name, 'pred')
        if pred_stats:
            box_data.append(pred_stats)
            x_labels.append(f'{rank_name}\n(Pred)')
            colors.append(color)
    
    # Add Overall if present
    if has_overall:
        color = '#333333'  # Dark grey for Overall
        
        # Get min relative error for legend
        min_rel_error = latest_row.get('Overall_min_rel_error', pd.NA)
        if pd.isna(min_rel_error):
            min_rel_error = latest_row.get('Overall_mean_rel_error', 0.0)
        
        legend_items.append(('Overall', color, min_rel_error))
        
        # True distances box stats
        true_stats = _get_box_stats(latest_row, 'Overall', 'true')
        if true_stats:
            box_data.append(true_stats)
            x_labels.append('Overall\n(True)')
            colors.append(color)
        
        # Predicted distances box stats
        pred_stats = _get_box_stats(latest_row, 'Overall', 'pred')
        if pred_stats:
            box_data.append(pred_stats)
            x_labels.append('Overall\n(Pred)')
            colors.append(color)
    
    if not box_data:
        print("Warning: No valid box plot data found.")
        return
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Draw box plots using bxp with pre-computed statistics
    bp = ax.bxp(box_data, positions=range(len(box_data)), patch_artist=True, showfliers=False)
    
    # Color the boxes
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    # Color the medians
    for median in bp['medians']:
        median.set_color('black')
        median.set_linewidth(1.5)
    
    # Customize the plot
    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(x_labels, fontsize=9, rotation=45, ha='right')
    ax.set_xlabel('Rank and Distance Type', fontsize=12)
    ax.set_ylabel('Cosine Distance', fontsize=12)
    title_suffix = ' (Log2 Scale)' if log_scale else ''
    ax.set_title(f'Pair Distances by Rank (Batch {int(latest_row["Batch"])}){title_suffix}', fontsize=14, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.7, axis='y')
    
    # Set y-axis to 0-2 range (cosine distance range)
    if log_scale:
        ax.set_yscale('symlog', base=2, linthresh=0.01)
        
    ax.set_ylim(0, 2.0)
    ax.set_yticks(np.arange(0, 2.1, 0.2))
    
    # Create legend with diff of medians
    legend_handles = []
    for rank_name, color, _ in legend_items: # 3rd item was min_rel_err, now unused
        # Get medians for difference
        true_med_col = f'{rank_name}_true_median'
        pred_med_col = f'{rank_name}_pred_median'
        
        # Calculate diff of medians (Pred - True)
        if true_med_col in latest_row and pred_med_col in latest_row and pd.notna(latest_row[true_med_col]) and pd.notna(latest_row[pred_med_col]):
            diff_medians = float(latest_row[pred_med_col]) - float(latest_row[true_med_col])
            label = f"{rank_name} (Diff of Medians={diff_medians:.3f})"
        else:
            label = rank_name
            
        patch = plt.Rectangle((0, 0), 1, 1, facecolor=color, alpha=0.6, edgecolor='black')
        legend_handles.append((patch, label))
    
    ax.legend([h[0] for h in legend_handles], [h[1] for h in legend_handles],
              bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    
    plt.tight_layout()
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Save the plot
    try:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    except Exception as e:
        print(f"Error saving pair distances plot to {output_path}: {e}")
    
    # Show the plot if requested
    if show:
        plt.show()
    
    # Close the figure to free memory
    plt.close('all')


def _get_box_stats(row, rank_name, dist_type):
    """
    Extract box plot statistics from a dataframe row for a given rank and distance type.
    
    Args:
        row (pd.Series): A row from the pair distances dataframe.
        rank_name (str): The rank name (e.g., 'Domain', 'Phylum', 'Overall').
        dist_type (str): Either 'true' or 'pred'.
    
    Returns:
        dict: Box plot statistics dict for matplotlib's bxp(), or None if data is missing.
              Keys: 'whislo', 'q1', 'med', 'q3', 'whishi'
    """
    prefix = f'{rank_name}_{dist_type}'
    
    # Get statistics
    min_val = row.get(f'{prefix}_min', pd.NA)
    q1_val = row.get(f'{prefix}_q1', pd.NA)
    median_val = row.get(f'{prefix}_median', pd.NA)
    q3_val = row.get(f'{prefix}_q3', pd.NA)
    max_val = row.get(f'{prefix}_max', pd.NA)
    
    # Check if we have valid data
    if any(pd.isna(v) for v in [min_val, q1_val, median_val, q3_val, max_val]):
        return None
    
    return {
        'whislo': float(min_val),
        'q1': float(q1_val),
        'med': float(median_val),
        'q3': float(q3_val),
        'whishi': float(max_val),
    }


def save_pair_distances_csv(df, output_path):
    """
    Save pair distances dataframe to CSV.
    
    Args:
        df (pd.DataFrame): Pair distances dataframe.
        output_path (str): File path for the CSV file.
    """
    if df is None or len(df) == 0:
        return
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    try:
        df.to_csv(output_path, index=False)
    except Exception as e:
        print(f"Error saving pair distances CSV to {output_path}: {e}")



# PAIR PER-RANK ERROR METRICS ================================================
# Line plot showing the composite error metric per rank over training.
# This metric feeds the EMA for per-rank batch allocation:
#   metric = mean_error * PAIR_EMA_MEAN_WEIGHT + (p25_error + p75_error) * PAIR_EMA_QUARTILES_WEIGHT

def init_pair_error_metrics_df():
    """
    Initialize an empty dataframe for tracking pair per-rank error metrics over training.
    
    The error metric is the value fed to the EMA for per-rank batch allocation:
        metric = mean_error * PAIR_EMA_MEAN_WEIGHT + (p25_error + p75_error) * PAIR_EMA_QUARTILES_WEIGHT
    
    Pre-initializes all rank columns to avoid DataFrame fragmentation warnings.
    
    Returns:
        pd.DataFrame: DataFrame with 'Batch' and rank columns initialized.
    """
    columns = ['Batch']
    for rank_name in PAIR_RANK_MAP.values():
        columns.append(rank_name)
    return pd.DataFrame(columns=columns)


def add_pair_error_metrics_row_inplace(df, batch_num, metrics_by_rank):
    """
    Add a row of pair error metrics data to the dataframe in-place.
    
    This function modifies the dataframe directly, suitable for passing dataframe
    references through function call chains.
    
    Args:
        df (pd.DataFrame): Existing pair error metrics dataframe (modified in-place).
        batch_num (int): The current batch number.
        metrics_by_rank (dict): Dictionary mapping rank (int) to metric value (float).
                                e.g. {-1: 0.52, 0: 0.48, 1: 0.35, ...}
                                Ranks are internal pair rank values (-1 to 7).
    """
    if df is None:
        return
    
    row_dict = {'Batch': batch_num}
    
    # Add per-rank metric values
    for rank, metric_val in metrics_by_rank.items():
        rank_name = PAIR_RANK_MAP.get(rank, f'Rank-{rank}')
        row_dict[rank_name] = metric_val
    
    # Ensure all columns exist in the dataframe
    for col in row_dict.keys():
        if col not in df.columns:
            df[col] = pd.NA
    
    # Add the new row using loc
    df.loc[len(df)] = row_dict


def plot_pair_error_metrics(df, output_path, show=False, truncate_to_after_n_batches=None):
    """
    Create a line plot showing pair per-rank error metrics over time.
    
    The error metric is the composite value fed to the EMA:
        metric = mean_error * PAIR_EMA_MEAN_WEIGHT + (p25_error + p75_error) * PAIR_EMA_QUARTILES_WEIGHT
    
    This follows the same style as triplet satisfaction plots with:
    - Lines for each rank present in the data
    - Legend showing rank names with latest values like "Domain (Latest=0.523)"
    - Y-axis auto-scaled based on data range
    - Grid lines and clean formatting
    
    Args:
        df (pd.DataFrame): DataFrame containing pair error metrics data with columns:
            - 'Batch': Batch numbers
            - Rank columns (e.g., 'Domain', 'Phylum', etc.): Per-rank metric values
        output_path (str): File path where the plot should be saved (should include .png extension).
        show (bool, optional): Whether to display the plot interactively. Defaults to False.
        truncate_to_after_n_batches (int, optional): If provided, the plot will be truncated to batches after this number. Defaults to None.
    """
    # Truncate the dataframe if requested
    if truncate_to_after_n_batches is not None:
        df = df[df['Batch'] >= truncate_to_after_n_batches].copy()

    # Check if there's any data to plot
    if df is None or len(df) == 0:
        print("Warning: No pair error metrics data available for plotting.")
        return
    
    # Get rank columns and sort them correctly according to the rank hierarchy
    # (Domain, Phylum, Class, Order, Family, Genus, Species, Sequence, Subseq)
    rank_names = list(PAIR_RANK_MAP.values())
    present_rank_columns = [col for col in rank_names if col in df.columns]
    
    if not present_rank_columns:
        print("Warning: No pair error metrics columns found for plotting.")
        return
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    
    # Define a color cycle for consistent rank coloring
    colors = plt.cm.tab10.colors
    
    # Plot each rank column
    for i, column in enumerate(present_rank_columns):
        # Get non-null values for this column
        valid_data = df[['Batch', column]].dropna()
        if len(valid_data) == 0:
            continue
        
        # Get latest score for legend
        latest_score = valid_data[column].iloc[-1]
        label = f"{column} (Latest={latest_score:.3f})"
        
        # Use color cycle for ranks
        color = colors[i % len(colors)]
        
        # Only show markers if less than 20 data points
        if len(valid_data) <= 20:
            plt.plot(valid_data['Batch'].astype(int), valid_data[column], 
                     marker='o', label=label, linewidth=2, markersize=4,
                     color=color)
        else:
            plt.plot(valid_data['Batch'].astype(int), valid_data[column], 
                     label=label, linewidth=2, color=color)
    
    # Customize the plot
    plt.xlabel('Batch Number', fontsize=12)
    plt.ylabel('Error Metric (EMA Input)', fontsize=12)
    plt.title('Pair Per-Rank Error Metrics Over Training', fontsize=14, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Auto-scale y-axis with some padding
    all_values = []
    for col in present_rank_columns:
        valid_vals = df[col].dropna().values
        all_values.extend(valid_vals)
    if all_values:
        y_min = min(0, min(all_values) * 0.95)
        y_max = max(all_values) * 1.05
        plt.ylim(y_min, y_max)
    
    # Position legend outside plot area
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.tight_layout()
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Save the plot
    try:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    except Exception as e:
        print(f"Error saving pair error metrics plot to {output_path}: {e}")
    
    # Show the plot if requested
    if show:
        plt.show()
    
    # Close the figure to free memory
    plt.close('all')


def save_pair_error_metrics_csv(df, output_path):
    """
    Save pair error metrics dataframe to CSV.
    
    Args:
        df (pd.DataFrame): Pair error metrics dataframe.
        output_path (str): File path for the CSV file.
    """
    if df is None or len(df) == 0:
        return
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    try:
        df.to_csv(output_path, index=False)
    except Exception as e:
        print(f"Error saving pair error metrics CSV to {output_path}: {e}")



# TRIPLET PER-RANK ERROR METRICS ================================================
# Line plot showing the composite hardness metric per rank over training.
# This metric feeds the EMA for per-rank batch allocation:
#   metric = hard_triplet_prop * TRIPLET_EMA_HARD_WEIGHT + moderate_triplet_prop * TRIPLET_EMA_MODERATE_WEIGHT

def init_triplet_error_metrics_df():
    """
    Initialize an empty dataframe for tracking triplet per-rank error metrics over training.
    
    The error metric is the value fed to the EMA for per-rank batch allocation:
        metric = hard_triplet_prop * TRIPLET_EMA_HARD_WEIGHT + moderate_triplet_prop * TRIPLET_EMA_MODERATE_WEIGHT
    
    Pre-initializes all rank columns to avoid DataFrame fragmentation warnings.
    
    Returns:
        pd.DataFrame: DataFrame with 'Batch' and rank columns initialized.
    """
    columns = ['Batch']
    for rank_name in TRIPLET_RANK_MAP.values():
        columns.append(rank_name)
    return pd.DataFrame(columns=columns)


def add_triplet_error_metrics_row_inplace(df, batch_num, metrics_by_rank):
    """
    Add a row of triplet error metrics data to the dataframe in-place.
    
    This function modifies the dataframe directly, suitable for passing dataframe
    references through function call chains.
    
    Args:
        df (pd.DataFrame): Existing triplet error metrics dataframe (modified in-place).
        batch_num (int): The current batch number.
        metrics_by_rank (dict): Dictionary mapping rank (int) to metric value (float).
                                e.g. {0: 1.52, 1: 0.98, 2: 0.75, ...}
                                Ranks are 0-5 (Domain through Genus).
    """
    if df is None:
        return
    
    row_dict = {'Batch': batch_num}
    
    # Add per-rank metric values
    for rank, metric_val in metrics_by_rank.items():
        rank_name = TRIPLET_RANK_MAP.get(rank, f'Rank-{rank}')
        row_dict[rank_name] = metric_val
    
    # Ensure all columns exist in the dataframe
    for col in row_dict.keys():
        if col not in df.columns:
            df[col] = pd.NA
    
    # Add the new row using loc
    df.loc[len(df)] = row_dict


def plot_triplet_error_metrics(df, output_path, show=False, truncate_to_after_n_batches=None):
    """
    Create a line plot showing triplet per-rank error metrics over time.
    
    The error metric is the composite value fed to the EMA:
        metric = hard_triplet_prop * TRIPLET_EMA_HARD_WEIGHT + moderate_triplet_prop * TRIPLET_EMA_MODERATE_WEIGHT
    
    This follows the same style as triplet satisfaction plots with:
    - Lines for each rank present in the data
    - Legend showing rank names with latest values like "Domain (Latest=1.234)"
    - Y-axis auto-scaled based on data range
    - Grid lines and clean formatting
    
    Args:
        df (pd.DataFrame): DataFrame containing triplet error metrics data with columns:
            - 'Batch': Batch numbers
            - Rank columns (e.g., 'Domain', 'Phylum', etc.): Per-rank metric values
        output_path (str): File path where the plot should be saved (should include .png extension).
        show (bool, optional): Whether to display the plot interactively. Defaults to False.
        truncate_to_after_n_batches (int, optional): If provided, the plot will be truncated to batches after this number. Defaults to None.
    """
    # Truncate the dataframe if requested
    if truncate_to_after_n_batches is not None:
        df = df[df['Batch'] >= truncate_to_after_n_batches].copy()

    # Check if there's any data to plot
    if df is None or len(df) == 0:
        print("Warning: No triplet error metrics data available for plotting.")
        return
    
    # Get rank columns and sort them correctly according to the rank hierarchy
    # (Domain, Phylum, Class, Order, Family, Genus)
    rank_names = list(TRIPLET_RANK_MAP.values())
    present_rank_columns = [col for col in rank_names if col in df.columns]
    
    if not present_rank_columns:
        print("Warning: No triplet error metrics columns found for plotting.")
        return
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    
    # Define a color cycle for consistent rank coloring
    colors = plt.cm.tab10.colors
    
    # Plot each rank column
    for i, column in enumerate(present_rank_columns):
        # Get non-null values for this column
        valid_data = df[['Batch', column]].dropna()
        if len(valid_data) == 0:
            continue
        
        # Get latest score for legend
        latest_score = valid_data[column].iloc[-1]
        label = f"{column} (Latest={latest_score:.3f})"
        
        # Use color cycle for ranks
        color = colors[i % len(colors)]
        
        # Only show markers if less than 20 data points
        if len(valid_data) <= 20:
            plt.plot(valid_data['Batch'].astype(int), valid_data[column], 
                     marker='o', label=label, linewidth=2, markersize=4,
                     color=color)
        else:
            plt.plot(valid_data['Batch'].astype(int), valid_data[column], 
                     label=label, linewidth=2, color=color)
    
    # Customize the plot
    plt.xlabel('Batch Number', fontsize=12)
    plt.ylabel('Hardness Metric (EMA Input)', fontsize=12)
    plt.title('Triplet Per-Rank Hardness Metrics Over Training', fontsize=14, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Auto-scale y-axis with some padding
    all_values = []
    for col in present_rank_columns:
        valid_vals = df[col].dropna().values
        all_values.extend(valid_vals)
    if all_values:
        y_min = min(0, min(all_values) * 0.95)
        y_max = max(all_values) * 1.05
        plt.ylim(y_min, y_max)
    
    # Position legend outside plot area
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.tight_layout()
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Save the plot
    try:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    except Exception as e:
        print(f"Error saving triplet error metrics plot to {output_path}: {e}")
    
    # Show the plot if requested
    if show:
        plt.show()
    
    # Close the figure to free memory
    plt.close('all')


def save_triplet_error_metrics_csv(df, output_path):
    """
    Save triplet error metrics dataframe to CSV.
    
    Args:
        df (pd.DataFrame): Triplet error metrics dataframe.
        output_path (str): File path for the CSV file.
    """
    if df is None or len(df) == 0:
        return
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    try:
        df.to_csv(output_path, index=False)
    except Exception as e:
        print(f"Error saving triplet error metrics CSV to {output_path}: {e}")



# CONV STEM SCALE ================================================

def plot_conv_stem_scale(csv_path, output_path, show=False, truncate_to_after_n_batches=None):
    """
    Create a line plot showing the progression of conv stem scale over batches.

    This function reads conv stem scale values from a CSV file and plots them
    over training time. The scale parameter controls the magnitude of the 
    convolutional stem's residual contribution.

    Args:
        csv_path (str): Path to the conv_stem_scale.csv file.
        output_path (str): File path where the plot should be saved.
        show (bool, optional): Whether to display the plot interactively. Defaults to False.
        truncate_to_after_n_batches (int, optional): If provided, the plot will be truncated 
                                                     to batches after this number. Defaults to None.
    """
    if not os.path.exists(csv_path):
        print(f"Warning: Conv stem scale CSV not found at {csv_path}")
        return

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading conv stem scale from {csv_path}: {e}")
        return

    if df.empty:
        print("Warning: Conv stem scale dataframe is empty.")
        return

    # Truncate the dataframe if requested
    if truncate_to_after_n_batches is not None:
        df = df[df['batch'] >= truncate_to_after_n_batches].copy()

    if df.empty:
        print(f"Warning: No data available for conv stem scale after batch {truncate_to_after_n_batches}.")
        return

    plt.figure(figsize=(12, 6))

    # Plot conv stem scale
    if len(df) <= 20:
        plt.plot(df['batch'], df['conv_stem_scale'],
                 label=f"Conv Stem Scale (Latest={df['conv_stem_scale'].iloc[-1]:.6f})",
                 linewidth=2, color='#2ca02c', marker='o', markersize=4)
    else:
        plt.plot(df['batch'], df['conv_stem_scale'],
                 label=f"Conv Stem Scale (Latest={df['conv_stem_scale'].iloc[-1]:.6f})",
                 linewidth=2, color='#2ca02c')

    # Customize the plot
    plt.xlabel('Batch Number', fontsize=12)
    plt.ylabel('Scale Value', fontsize=12)
    plt.title('Conv Stem Scale Over Training', fontsize=14, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.tight_layout()

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Save the plot
    try:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    except Exception as e:
        print(f"Error saving conv stem scale plot to {output_path}: {e}")

    # Show the plot if requested
    if show:
        plt.show()

    # Close the figure to free memory
    plt.close('all')
