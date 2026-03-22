"""
Loss Results Module

This module provides functionality to track and visualize training losses during Micro16S model training.
It allows for recording loss values across training batches and creating plots to monitor training progress.

Key Functions:
- add_loss_results_df: Adds loss values to a dataframe for tracking across training batches
- plot_loss_results: Creates a line plot showing loss progression during training

Used in conjunction with train.py to monitor triplet loss, pair loss, and total loss during training.
"""


# IMPORTS ================================================
import pandas as pd
import matplotlib.pyplot as plt
import os
import torch
import globals_config as gb



# FUNCTIONS ================================================

def add_loss_results_df(mean_triplet_loss, mean_pair_loss,
                        triplet_losses, triplet_ranks,
                        pair_losses, pair_ranks,
                        batch_num, total_loss,
                        weighted_triplet_loss=None,
                        weighted_pair_loss=None,
                        existing_df=None):
    """
    Add training loss results into a dataframe to track performance across training batches.

    This function calculates and records total loss, triplet loss, pair loss, and per-rank losses.

    Args:
        mean_triplet_loss (torch.Tensor or float): The triplet loss component for the current batch.
        mean_pair_loss (torch.Tensor or float): The pair loss component for the current batch.
        triplet_losses (torch.Tensor): Tensor of triplet losses for the batch.
        triplet_ranks (torch.Tensor): Tensor of ranks for each triplet.
        pair_losses (torch.Tensor): Tensor of pair losses for the batch.
        pair_ranks (torch.Tensor): Tensor of ranks for each pair.
        batch_num (int): The batch number identifier.
        total_loss (torch.Tensor or float): The final objective value used for backpropagation.
        weighted_triplet_loss (torch.Tensor or float, optional): Triplet loss after uncertainty weighting.
        weighted_pair_loss (torch.Tensor or float, optional): Pair loss after uncertainty weighting.
        existing_df (pd.DataFrame, optional): Existing dataframe to append to. Defaults to None.

    Returns:
        pd.DataFrame: The updated dataframe containing the new loss results row.
    """

    # Convert tensor values to float if necessary
    total_loss_val = total_loss.item() if hasattr(total_loss, 'item') else float(total_loss)
    triplet_loss_val = mean_triplet_loss.item() if hasattr(mean_triplet_loss, 'item') else float(mean_triplet_loss)
    pair_loss_val = mean_pair_loss.item() if hasattr(mean_pair_loss, 'item') else float(mean_pair_loss)
    
    # Create a dictionary for the new row with the batch number and loss values
    row_dict = {
        "Batch": batch_num,
        "Total_Loss": total_loss_val,
        "Triplet_Loss": triplet_loss_val,
        "Pair_Loss": pair_loss_val
    }

    using_uncertainty = (
        gb.IS_USING_UNCERTAINTY_WEIGHTING and
        weighted_triplet_loss is not None and
        weighted_pair_loss is not None
    )

    if using_uncertainty:
        weighted_triplet_val = (weighted_triplet_loss.item()
                                if hasattr(weighted_triplet_loss, 'item')
                                else float(weighted_triplet_loss))
        weighted_pair_val = (weighted_pair_loss.item()
                             if hasattr(weighted_pair_loss, 'item')
                             else float(weighted_pair_loss))
        row_dict["Triplet_Loss_Weighted"] = weighted_triplet_val
        row_dict["Pair_Loss_Weighted"] = weighted_pair_val
        row_dict["Total_Loss_Unweighted"] = triplet_loss_val + pair_loss_val

    # Calculate per-rank triplet losses
    if triplet_losses.numel() > 0:
        unique_triplet_ranks = torch.unique(triplet_ranks)
        for rank in unique_triplet_ranks:
            rank_int = rank.item()
            rank_mask = triplet_ranks == rank
            mean_loss_for_rank = triplet_losses[rank_mask].mean().item()
            row_dict[f"Rank-{rank_int}_Triplet_Loss"] = mean_loss_for_rank

    # Calculate per-rank pair losses
    if pair_losses.numel() > 0:
        unique_pair_ranks = torch.unique(pair_ranks)
        for rank in unique_pair_ranks:
            rank_int = rank.item()
            rank_mask = pair_ranks == rank
            mean_loss_for_rank = pair_losses[rank_mask].mean().item()
            row_dict[f"Rank-{rank_int}_Pair_Loss"] = mean_loss_for_rank
    
    # Create a new dataframe row from the dictionary
    new_row_df = pd.DataFrame([row_dict])
    
    # If an existing dataframe is provided, append the new row
    if existing_df is not None:
        updated_df = pd.concat([existing_df, new_row_df], ignore_index=True)
    else:
        updated_df = new_row_df

    # Ensure the primary loss columns remain in the desired order
    column_priority = [
        "Batch",
        "Total_Loss",
        "Triplet_Loss_Weighted",
        "Pair_Loss_Weighted",
        "Total_Loss_Unweighted",
        "Triplet_Loss",
        "Pair_Loss",
    ]
    ordered_columns = [col for col in column_priority if col in updated_df.columns]
    remaining_columns = [col for col in updated_df.columns if col not in ordered_columns]
    updated_df = updated_df[ordered_columns + remaining_columns]

    return updated_df


def plot_loss_results(loss_df, output_path, show=False, plot_loss_per_rank=True, truncate_to_after_n_batches=None, 
                      plot_triplets=True, plot_pairs=True, plot_total=None):
    """
    Create a line plot showing the progression of training losses over batches.

    This function generates a comprehensive plot displaying total loss, triplet loss, and pair loss
    across training batches. The plot helps visualize training progress and convergence behavior.

    Args:
        loss_df (pd.DataFrame): DataFrame containing loss results with columns:
            - 'Batch': Batch numbers
            - 'Total_Loss': Total loss values
            - 'Triplet_Loss': Triplet loss values  
            - 'Pair_Loss': Pair loss values
            - 'Rank-X_Triplet_Loss': (Optional) Per-rank triplet losses
            - 'Rank-X_Pair_Loss': (Optional) Per-rank pair losses
        output_path (str): File path where the plot should be saved (should include .png extension).
        show (bool, optional): Whether to display the plot interactively. Defaults to False.
        plot_loss_per_rank (bool, optional): Whether to plot per-rank losses. Defaults to True.
        truncate_to_after_n_batches (int, optional): If provided, the plot will be truncated to batches after this number. Defaults to None.
        plot_triplets (bool, optional): Whether to plot triplet losses. Defaults to True.
        plot_pairs (bool, optional): Whether to plot pair losses. Defaults to True.
        plot_total (bool, optional): Whether to plot the total loss. If None, it is only plotted 
                                     if BOTH triplet and pair losses are requested; when plotting only
                                     the total loss, the triplet and pair contributions are overlaid.
    """
    
    # Truncate the dataframe if requested
    if truncate_to_after_n_batches is not None:
        loss_df = loss_df[loss_df['Batch'] >= truncate_to_after_n_batches].copy()

    # Check if there's any data to plot
    if len(loss_df) == 0:
        print("Warning: No data available for plotting losses.")
        return

    def _has_signal(column_name):
        """
        Check whether a column exists and contains any non-zero values.
        """
        if column_name not in loss_df.columns:
            return False
        return not (loss_df[column_name].fillna(0) == 0).all()

    using_uncertainty_cols = (
        gb.IS_USING_UNCERTAINTY_WEIGHTING and
        'Triplet_Loss_Weighted' in loss_df.columns and
        'Pair_Loss_Weighted' in loss_df.columns
    )
    triplet_component_col = 'Triplet_Loss_Weighted' if using_uncertainty_cols else 'Triplet_Loss'
    pair_component_col = 'Pair_Loss_Weighted' if using_uncertainty_cols else 'Pair_Loss'
    plot_total_components = plot_total and not plot_triplets and not plot_pairs
    use_triplet_component = plot_total_components and _has_signal(triplet_component_col)
    use_pair_component = plot_total_components and _has_signal(pair_component_col)

    # Detect which losses are being used
    use_triplet_loss = plot_triplets and _has_signal('Triplet_Loss')
    use_pair_loss = plot_pairs and _has_signal('Pair_Loss')

    # Determine if total loss should be plotted
    # If plot_total is not explicitly set, we only plot it when BOTH triplet and pair losses are requested
    if plot_total is None:
        use_total_loss = use_triplet_loss and use_pair_loss
    else:
        use_total_loss = plot_total and 'Total_Loss' in loss_df.columns

    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Plot Total Loss
    if use_total_loss:
        plt.plot(loss_df['Batch'], loss_df['Total_Loss'], 
                 label='Total Loss', linewidth=2.5, color='black', marker='o' if len(loss_df) <= 50 else None, markersize=3, zorder=10)

    # Plot Triplet Loss and its per-rank components
    if use_triplet_loss:
        # Triplet losses are logged after TRIPLET_LOSS_WEIGHT scaling but before uncertainty weighting.
        plt.plot(loss_df['Batch'], loss_df['Triplet_Loss'], 
                 label='Triplet Loss', linewidth=2, color='#FF6347', marker='s' if len(loss_df) <= 50 else None, markersize=3)
        if plot_loss_per_rank:
            triplet_rank_cols = sorted([col for col in loss_df.columns if 'Triplet_Loss' in col and 'Rank-' in col])
            for i, col in enumerate(triplet_rank_cols):
                rank_num = col.split('_')[0].split('-')[1]
                plt.plot(loss_df['Batch'], loss_df[col].fillna(0), 
                         label=f'Triplet Loss (Rank {rank_num})', linestyle='--', linewidth=1.5, color=plt.cm.Reds(0.4 + 0.6 * i / len(triplet_rank_cols)))

    # Plot Pair Loss and its per-rank components
    if use_pair_loss:
        # Pair losses are logged after PAIR_LOSS_WEIGHT scaling but before uncertainty weighting.
        plt.plot(loss_df['Batch'], loss_df['Pair_Loss'], 
                 label='Pair Loss', linewidth=2, color='#4169E1', marker='^' if len(loss_df) <= 50 else None, markersize=3)
        if plot_loss_per_rank:
            pair_rank_cols = sorted([col for col in loss_df.columns if 'Pair_Loss' in col and 'Rank-' in col])
            for i, col in enumerate(pair_rank_cols):
                rank_num = col.split('_')[0].split('-')[1]
                plt.plot(loss_df['Batch'], loss_df[col].fillna(0), 
                         label=f'Pair Loss (Rank {rank_num})', linestyle='--', linewidth=1.5, color=plt.cm.Blues(0.4 + 0.6 * i / len(pair_rank_cols)))

    if use_triplet_component:
        # When plotting total loss by itself, also show the triplet contribution (post uncertainty weighting if enabled).
        plt.plot(loss_df['Batch'], loss_df[triplet_component_col],
                 label='Triplet Loss', linewidth=2, color='#FF6347', marker='s' if len(loss_df) <= 50 else None, markersize=3)

    if use_pair_component:
        # Same as above for the pair contribution.
        plt.plot(loss_df['Batch'], loss_df[pair_component_col],
                 label='Pair Loss', linewidth=2, color='#4169E1', marker='^' if len(loss_df) <= 50 else None, markersize=3)

    # Customize the plot
    plt.xlabel('Batch Number', fontsize=12)
    plt.ylabel('Loss Value', fontsize=12)

    # Title
    title = 'Training Loss Progression'
    if use_triplet_loss and not use_pair_loss:
        title = 'Training Triplet Loss Progression'
    elif use_pair_loss and not use_triplet_loss:
        title = 'Training Pair Loss Progression'
    elif use_total_loss and not use_triplet_loss and not use_pair_loss:
        title = 'Training Total Loss Progression'
    
    plt.title(title, fontsize=14, fontweight='bold')
    
    # Legend - only show if more than one thing is plotted (triplet, pair, or both)
    # OR if we are explicitly plotting a mixed plot.
    # If ONLY plotting total loss, no legend.
    if use_triplet_loss or use_pair_loss or use_triplet_component or use_pair_component:
        plt.legend(loc='upper right', fontsize=10)
    
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Set y-axis to start from 0 if all losses are positive
    plot_cols = []
    if use_total_loss: plot_cols.append('Total_Loss')
    if use_triplet_loss: plot_cols.append('Triplet_Loss')
    if use_pair_loss: plot_cols.append('Pair_Loss')
    if use_triplet_component: plot_cols.append(triplet_component_col)
    if use_pair_component: plot_cols.append(pair_component_col)
    
    if plot_cols and loss_df[plot_cols].min().min() >= 0:
        plt.ylim(bottom=0)
    
    # Adjust layout to prevent any cutoff
    plt.tight_layout()
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Save the plot
    try:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    except Exception as e:
        print(f"Error saving loss plot to {output_path}: {e}")
    
    # Show the plot if requested
    if show:
        plt.show()
    
    # Close the figure to free memory
    plt.close('all')


def plot_uncertainty_weights(csv_path, output_path, log_var_output_path=None,
                             show=False, truncate_to_after_n_batches=None, y_min_zero=True):
    """
    Create a line plot showing the progression of uncertainty weights over batches.

    This function reads uncertainty weights from a CSV file and plots the weights for
    triplets and pairs. Consistent with other plots, triplets are shown in red and
    pairs in blue.

    Args:
        csv_path (str): Path to the uncertainty_weighting.csv file.
        output_path (str): File path where the weight plot should be saved.
        log_var_output_path (str, optional): File path where the log-var plot should be saved.
                                             Defaults to the output_path filename with "log_vars".
        show (bool, optional): Whether to display the plot interactively. Defaults to False.
        truncate_to_after_n_batches (int, optional): If provided, the plot will be truncated 
                                                     to batches after this number. Defaults to None.
        y_min_zero (bool, optional): Whether to force the y-axis to start at zero. Defaults to True.
    """
    if not os.path.exists(csv_path):
        print(f"Warning: Uncertainty weights CSV not found at {csv_path}")
        return

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading uncertainty weights from {csv_path}: {e}")
        return

    if df.empty:
        print("Warning: Uncertainty weights dataframe is empty.")
        return

    # Truncate the dataframe if requested
    if truncate_to_after_n_batches is not None:
        df = df[df['batch'] >= truncate_to_after_n_batches].copy()

    if df.empty:
        print(f"Warning: No data available for uncertainty weights after batch {truncate_to_after_n_batches}.")
        return

    if log_var_output_path is None:
        base_dir = os.path.dirname(output_path)
        filename = os.path.basename(output_path)
        name, ext = os.path.splitext(filename)
        if 'uncertainty_weight' in name:
            name = name.replace('uncertainty_weight', 'log_vars')
        else:
            name = f"{name}_log_vars"
        log_var_output_path = os.path.join(base_dir, name + ext)

    plt.figure(figsize=(12, 8))

    # Plot Triplet Weight (Red)
    if 'triplet_weight' in df.columns:
        plt.plot(df['batch'], df['triplet_weight'], 
                 label='Triplet Weight', linewidth=2, color='#FF6347', marker='s' if len(df) <= 50 else None, markersize=3)

    # Plot Pair Weight (Blue)
    if 'pair_weight' in df.columns:
        plt.plot(df['batch'], df['pair_weight'], 
                 label='Pair Weight', linewidth=2, color='#4169E1', marker='^' if len(df) <= 50 else None, markersize=3)

    plt.xlabel('Batch Number', fontsize=12)
    plt.ylabel('Weight Value', fontsize=12)
    plt.title('Uncertainty Weight Progression', fontsize=14, fontweight='bold')
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)

    # Set y-axis to start from 0 if all weights are positive
    if y_min_zero and df[['triplet_weight', 'pair_weight']].min().min() >= 0:
        plt.ylim(bottom=0)

    plt.tight_layout()

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Save the plot
    try:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    except Exception as e:
        print(f"Error saving uncertainty weights plot to {output_path}: {e}")

    if show:
        plt.show()

    plt.close('all')

    has_log_var_cols = 'triplet_log_var' in df.columns and 'pair_log_var' in df.columns
    if not has_log_var_cols:
        print("Warning: Log variance columns not found in uncertainty weighting CSV; skipping log_vars plot.")
        return

    plt.figure(figsize=(12, 8))

    # Plot Triplet Log-Variance (Red)
    plt.plot(df['batch'], df['triplet_log_var'],
             label='Triplet Log-Var', linewidth=2, color='#FF6347', marker='s' if len(df) <= 50 else None, markersize=3)

    # Plot Pair Log-Variance (Blue)
    plt.plot(df['batch'], df['pair_log_var'],
             label='Pair Log-Var', linewidth=2, color='#4169E1', marker='^' if len(df) <= 50 else None, markersize=3)

    plt.xlabel('Batch Number', fontsize=12)
    plt.ylabel('Log-Variance Value', fontsize=12)
    plt.title('Uncertainty Log-Variance Progression', fontsize=14, fontweight='bold')
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    log_var_dir = os.path.dirname(log_var_output_path)
    if log_var_dir and not os.path.exists(log_var_dir):
        os.makedirs(log_var_dir, exist_ok=True)

    try:
        plt.savefig(log_var_output_path, dpi=300, bbox_inches='tight')
    except Exception as e:
        print(f"Error saving log-vars plot to {log_var_output_path}: {e}")

    if show:
        plt.show()

    plt.close('all')
