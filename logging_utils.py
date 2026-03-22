"""Just some logging/printing utility functions."""

# Imports --------------------------
import os
import torch
import numpy as np
import globals_config as gb
from log_plotting import (add_triplet_satisfaction_row_inplace, add_pair_distances_row_inplace,
                          add_pair_error_metrics_row_inplace, add_triplet_error_metrics_row_inplace)


def _get_bucket_labels(count):
    if count <= 0:
        return []
    labels = [f"B{i}" for i in range(max(count - 1, 0))]
    labels.append("Any")
    return labels


def _format_bucket_vector(vector):
    if vector is None or len(vector) == 0:
        return ""
    labels = _get_bucket_labels(len(vector))
    formatted = []
    for label, value in zip(labels, vector):
        formatted.append(f"{label}={float(value):.3f}")
    return ", ".join(formatted)


def _bucket_vector_for_csv(vector):
    if vector is None or len(vector) == 0:
        return ""
    return "|".join(f"{float(value):.6f}" for value in vector)


def _format_bucket_diag_entries(diag):
    if not diag:
        return None
    targets = diag.get('targets') or []
    if not targets:
        return None
    labels = _get_bucket_labels(len(targets))
    sampled = diag.get('sampled') or [0] * len(labels)
    deficits = diag.get('residual_deficit') or [0] * len(labels)
    borrowed = diag.get('borrowed_from') or [{}] * len(labels)
    
    segments = []
    for idx, label in enumerate(labels):
        target_val = targets[idx] if idx < len(targets) else 0
        sampled_val = sampled[idx] if idx < len(sampled) else 0
        deficit_val = deficits[idx] if idx < len(deficits) else 0
        borrow_map = borrowed[idx] if idx < len(borrowed) else {}
        segment = f"{label} {sampled_val}/{target_val}"
        if deficit_val > 0:
            segment += f" (-{deficit_val})"
        if borrow_map:
            borrow_bits = []
            for src_idx in sorted(borrow_map.keys()):
                src_label = labels[src_idx] if src_idx < len(labels) else f"B{src_idx}"
                borrow_bits.append(f"{src_label}:{borrow_map[src_idx]}")
            if borrow_bits:
                segment += f" <- {','.join(borrow_bits)}"
        segments.append(segment)
    return " | ".join(segments)


def _bucket_diag_strings_for_csv(diag):
    if not diag:
        return ("", "", "", "")
    targets = diag.get('targets')
    if not targets:
        return ("", "", "", "")
    labels = _get_bucket_labels(len(targets))
    
    def _join(values):
        if not values:
            return ""
        return "|".join(str(int(v)) for v in values)
    
    borrowed_entries = []
    borrowed_list = diag.get('borrowed_from') or []
    for idx, borrow_map in enumerate(borrowed_list):
        if not borrow_map:
            continue
        target_label = labels[idx] if idx < len(labels) else f"B{idx}"
        for src_idx in sorted(borrow_map.keys()):
            src_label = labels[src_idx] if src_idx < len(labels) else f"B{src_idx}"
            borrowed_entries.append(f"{target_label}<-{src_label}:{int(borrow_map[src_idx])}")
    
    return (
        _join(targets),
        _join(diag.get('sampled')),
        _join(diag.get('residual_deficit')),
        "|".join(borrowed_entries)
    )


def _format_table_float_with_tiny_indicator(value, width=8, decimals=4):
    """
    Format a float for fixed-width tables, showing tiny non-zero values as <threshold.
    Example for decimals=4: 0.00003 -> "<0.0001"
    """
    value = float(value)
    threshold = 10 ** (-decimals)
    if 0.0 < abs(value) < threshold:
        tiny_str = f"{'-' if value < 0 else ''}<{threshold:.{decimals}f}"
        return f"{tiny_str:<{width}}"
    return f"{value:<{width}.{decimals}f}"




# Triplet Loss --------------------------

def print_triplet_loss_stats(a_p_dist, a_n_dist, margins, losses, ranks, buckets=None):
    """Helper to print stats for embedding_triplet_loss."""
    # Early return if no triplets
    if len(ranks) == 0:
        return
    
    # Move tensors to CPU
    a_p_dist = a_p_dist.detach().cpu()
    a_n_dist = a_n_dist.detach().cpu()
    margins = margins.detach().cpu()
    losses = losses.detach().cpu()
    ranks = ranks.detach().cpu()
    if buckets is not None:
        buckets = buckets.detach().cpu()

    # Triplet rank map: triplet_rank -> name
    triplet_rank_map = {0: 'Domain', 1: 'Phylum', 2: 'Class', 3: 'Order', 4: 'Family', 5: 'Genus'}

    n_triplets = len(ranks)
    # Three mutually exclusive categories that sum to 100%:
    # 1. hard_violations: AP >= AN (ordering violated)
    # 2. ordered_no_margin: AP < AN but AP + margin >= AN (correctly ordered but margin not satisfied)
    # 3. margin_satisfied: AP + margin < AN (margin fully satisfied)
    hard_violations = a_p_dist >= a_n_dist
    correctly_ordered = a_p_dist < a_n_dist
    margin_satisfied = a_p_dist + margins < a_n_dist
    ordered_no_margin = correctly_ordered & ~margin_satisfied

    # Compute scaling factors for display
    # For triplets: scale = TRIPLET_MARGIN_EPSILON / safe_margins (or 1/approx_delta_true for dynamic)
    epsilons_tensor = torch.tensor(gb.RELATIVE_ERROR_EPSILONS_TRIPLET_LOSS, device=margins.device, dtype=margins.dtype)
    triplet_epsilons = epsilons_tensor[ranks.long()]
    safe_margins = torch.maximum(margins, triplet_epsilons)
    if gb.MANUAL_TRIPLET_MARGINS:
        scale = gb.TRIPLET_MARGIN_EPSILON / safe_margins
    else:
        approx_delta_true = torch.maximum(safe_margins / gb.TRIPLET_MARGIN_EPSILON, triplet_epsilons)
        scale = 1.0 / approx_delta_true

    print("\nTriplet Loss Verbose  -------------------------------------")
    print("Triplet Loss Statistics:")
    print(f"  {'Rank':<8} | {'N':<6} | {'Min Loss':<9} | {'Mean Loss':<9} | {'Max Loss':<9} | {'% AP>=AN':<10} | {'% AP<AN<AP+M':<14} | {'% AP+M<AN':<10}")
    
    def _p(name, mask):
        if not mask.any(): return
        n = mask.sum().item()
        pct_hard = hard_violations[mask].float().mean().item() * 100
        pct_ordered_no_margin = ordered_no_margin[mask].float().mean().item() * 100
        pct_margin = margin_satisfied[mask].float().mean().item() * 100
        
        subset_losses = losses[mask]
        mean_loss = subset_losses.mean().item()
        min_loss = subset_losses.min().item()
        max_loss = subset_losses.max().item()
        
        print(f"  {name:<8} | {n:<6} | {min_loss:<9.4f} | {mean_loss:<9.4f} | {max_loss:<9.4f} | {pct_hard:>9.2f}% | {pct_ordered_no_margin:>13.2f}% | {pct_margin:>9.2f}%")

    _p("Overall", torch.ones_like(losses, dtype=torch.bool))
    for rank in sorted(torch.unique(ranks).tolist()):
        rank_name = triplet_rank_map.get(rank, str(rank))
        _p(rank_name, ranks == rank)

    if buckets is not None:
        print("Triplet Loss by Bucket:")
        print(f"  {'Bucket':<8} | {'N':<6} | {'Min Loss':<9} | {'Mean Loss':<9} | {'Max Loss':<9} | {'% AP>=AN':<10} | {'% AP<AN<AP+M':<14} | {'% AP+M<AN':<10}")
        for b in sorted(torch.unique(buckets).tolist()):
            _p(str(b), buckets == b)


def write_triplet_loss_log(batch_num, logs_dir, a_p_dist, a_n_dist, margins, losses, ranks, buckets=None):
    """
    Write triplet loss statistics to log and CSV files.
    
    Log files:
        triplet_loss.log: Text dump of printed output with batch headers
    
    CSV files:
        triplet_loss.csv: Overall stats per batch
            Columns: batch, n_triplets, ap_dist_min, ap_dist_max, ap_dist_mean, ap_dist_std, 
                     ap_dist_5pct, ap_dist_95pct, an_dist_min, an_dist_max, an_dist_mean, 
                     an_dist_std, an_dist_5pct, an_dist_95pct, margin_min, margin_max, 
                     margin_mean, margin_std, margin_5pct, margin_95pct, loss_min, loss_max, 
                     loss_mean, loss_std, loss_5pct, loss_95pct, pct_ap_gte_an, pct_ap_lt_an,
                     pct_margin_satisfied
        
        triplet_loss_by_rank.csv: Stats per rank per batch
            Columns: batch, rank, n_triplets, pct_ap_gte_an, pct_ap_lt_an, pct_margin_satisfied,
                     ap_dist_mean, an_dist_mean, margin_mean, loss_mean, scale_mean
        
        triplet_loss_by_bucket.csv: Stats per bucket per batch
            Columns: batch, bucket, n_triplets, pct_ap_gte_an, pct_ap_lt_an, pct_margin_satisfied,
                     ap_dist_mean, an_dist_mean, margin_mean, loss_mean, scale_mean
    """
    # Early return if no triplets
    if len(ranks) == 0:
        return
    
    # Move tensors to CPU for file I/O operations
    a_p_dist = a_p_dist.detach().cpu()
    a_n_dist = a_n_dist.detach().cpu()
    margins = margins.detach().cpu()
    losses = losses.detach().cpu()
    ranks = ranks.detach().cpu()
    if buckets is not None:
        buckets = buckets.detach().cpu()
    log_path = os.path.join(logs_dir, "triplet_loss.log")
    csv_path = os.path.join(logs_dir, "triplet_loss.csv")
    csv_by_rank_path = os.path.join(logs_dir, "triplet_loss_by_rank.csv")
    csv_by_bucket_path = os.path.join(logs_dir, "triplet_loss_by_bucket.csv")
    n_triplets = len(ranks)
    
    # Triplet rank map: triplet_rank -> name
    triplet_rank_map = {0: 'Domain', 1: 'Phylum', 2: 'Class', 3: 'Order', 4: 'Family', 5: 'Genus'}
    
    # Three mutually exclusive categories that sum to 100%:
    # 1. hard_violations: AP >= AN (ordering violated)
    # 2. ordered_no_margin: AP < AN but AP + margin >= AN (correctly ordered but margin not satisfied)
    # 3. margin_satisfied: AP + margin < AN (margin fully satisfied)
    hard_violations = a_p_dist >= a_n_dist
    correctly_ordered = a_p_dist < a_n_dist
    margin_satisfied = a_p_dist + margins < a_n_dist
    ordered_no_margin = correctly_ordered & ~margin_satisfied
    
    pct_ap_gte_an = hard_violations.float().mean().item() * 100
    pct_ap_lt_an = ordered_no_margin.float().mean().item() * 100
    pct_margin_sat = margin_satisfied.float().mean().item() * 100
    
    # Compute scaling factors
    epsilons_tensor = torch.tensor(gb.RELATIVE_ERROR_EPSILONS_TRIPLET_LOSS, device=margins.device, dtype=margins.dtype)
    triplet_epsilons = epsilons_tensor[ranks.long()]
    safe_margins = torch.maximum(margins, triplet_epsilons)
    if gb.MANUAL_TRIPLET_MARGINS:
        scale = gb.TRIPLET_MARGIN_EPSILON / safe_margins
    else:
        approx_delta_true = torch.maximum(safe_margins / gb.TRIPLET_MARGIN_EPSILON, triplet_epsilons)
        scale = 1.0 / approx_delta_true
    
    # Write .log file
    with open(log_path, "a") as f:
        f.write(f"\n---\nBatch {batch_num}\n")
        f.write(f"\nTriplet Loss Logging -------------------------------------\n")
        f.write(f"Anchor-Positive:  min: {a_p_dist.min():.4f}, max: {a_p_dist.max():.4f}, mean: {a_p_dist.mean():.4f}, std: {a_p_dist.std():.4f}, 5th %: {torch.quantile(a_p_dist, 0.05):.4f}, 95th %: {torch.quantile(a_p_dist, 0.95):.4f}\n")
        f.write(f"Anchor-Negative:  min: {a_n_dist.min():.4f}, max: {a_n_dist.max():.4f}, mean: {a_n_dist.mean():.4f}, std: {a_n_dist.std():.4f}, 5th %: {torch.quantile(a_n_dist, 0.05):.4f}, 95th %: {torch.quantile(a_n_dist, 0.95):.4f}\n")
        f.write(f"Margins:          min: {margins.min():.4f}, max: {margins.max():.4f}, mean: {margins.mean():.4f}, std: {margins.std():.4f}, 5th %: {torch.quantile(margins, 0.05):.4f}, 95th %: {torch.quantile(margins, 0.95):.4f}\n")
        f.write(f"Triplet losses:   min: {losses.min():.4f}, max: {losses.max():.4f}, mean: {losses.mean():.4f}, std: {losses.std():.4f}, 5th %: {torch.quantile(losses, 0.05):.4f}, 95th %: {torch.quantile(losses, 0.95):.4f}\n")
        f.write(f"\nTriplet Loss Statistics:\n")
        f.write(f"  {'Rank':<8} | {'N':<6} | {'Min Loss':<9} | {'Mean Loss':<9} | {'Max Loss':<9} | {'% AP>=AN':<10} | {'% AP<AN<AP+M':<14} | {'% AP+M<AN':<10}\n")
        
        def _w(name, mask):
            if not mask.any(): return
            n = mask.sum().item()
            pct_hard = hard_violations[mask].float().mean().item() * 100
            pct_ordered_no_margin = ordered_no_margin[mask].float().mean().item() * 100
            pct_margin = margin_satisfied[mask].float().mean().item() * 100
            
            subset_losses = losses[mask]
            mean_loss = subset_losses.mean().item()
            min_loss = subset_losses.min().item()
            max_loss = subset_losses.max().item()
            
            f.write(f"  {name:<8} | {n:<6} | {min_loss:<9.4f} | {mean_loss:<9.4f} | {max_loss:<9.4f} | {pct_hard:>9.2f}% | {pct_ordered_no_margin:>13.2f}% | {pct_margin:>9.2f}%\n")

        _w("Overall", torch.ones_like(losses, dtype=torch.bool))
        for rank in sorted(torch.unique(ranks).tolist()):
            rank_name = triplet_rank_map.get(rank, str(rank))
            _w(rank_name, ranks == rank)

        # Write bucket stats to log if buckets provided
        if buckets is not None:
            f.write(f"\nTriplet Loss by Bucket:\n")
            f.write(f"  {'Bucket':<8} | {'N':<6} | {'Min Loss':<9} | {'Mean Loss':<9} | {'Max Loss':<9} | {'% AP>=AN':<10} | {'% AP<AN<AP+M':<14} | {'% AP+M<AN':<10}\n")
            for b in sorted(torch.unique(buckets).tolist()):
                _w(str(b), buckets == b)
    # Write main CSV
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a") as f:
        if write_header:
            f.write("batch,n_triplets,ap_dist_min,ap_dist_max,ap_dist_mean,ap_dist_std,ap_dist_5pct,ap_dist_95pct,an_dist_min,an_dist_max,an_dist_mean,an_dist_std,an_dist_5pct,an_dist_95pct,margin_min,margin_max,margin_mean,margin_std,margin_5pct,margin_95pct,loss_min,loss_max,loss_mean,loss_std,loss_5pct,loss_95pct,pct_ap_gte_an,pct_ap_lt_an,pct_margin_satisfied,scale_mean\n")
        f.write(f"{batch_num},{n_triplets},{a_p_dist.min():.6f},{a_p_dist.max():.6f},{a_p_dist.mean():.6f},{a_p_dist.std():.6f},{torch.quantile(a_p_dist, 0.05):.6f},{torch.quantile(a_p_dist, 0.95):.6f},{a_n_dist.min():.6f},{a_n_dist.max():.6f},{a_n_dist.mean():.6f},{a_n_dist.std():.6f},{torch.quantile(a_n_dist, 0.05):.6f},{torch.quantile(a_n_dist, 0.95):.6f},{margins.min():.6f},{margins.max():.6f},{margins.mean():.6f},{margins.std():.6f},{torch.quantile(margins, 0.05):.6f},{torch.quantile(margins, 0.95):.6f},{losses.min():.6f},{losses.max():.6f},{losses.mean():.6f},{losses.std():.6f},{torch.quantile(losses, 0.05):.6f},{torch.quantile(losses, 0.95):.6f},{pct_ap_gte_an:.4f},{pct_ap_lt_an:.4f},{pct_margin_sat:.4f},{scale.mean().item():.6f}\n")
    # Write by-rank CSV
    write_header = not os.path.exists(csv_by_rank_path)
    with open(csv_by_rank_path, "a") as f:
        if write_header:
            f.write("batch,rank,n_triplets,pct_ap_gte_an,pct_ap_lt_an,pct_margin_satisfied,ap_dist_mean,an_dist_mean,margin_mean,loss_mean,scale_mean\n")
        for rank in sorted(torch.unique(ranks).tolist()):
            mask = (ranks == rank)
            n = mask.sum().item()
            if n > 0:
                pct_hard = hard_violations[mask].float().mean().item() * 100
                pct_ordered_no_margin = ordered_no_margin[mask].float().mean().item() * 100
                pct_margin = margin_satisfied[mask].float().mean().item() * 100
                ap_m = a_p_dist[mask].mean().item()
                an_m = a_n_dist[mask].mean().item()
                mg_m = margins[mask].mean().item()
                ls_m = losses[mask].mean().item()
                sc_m = scale[mask].mean().item()
                rank_name = triplet_rank_map.get(rank, str(rank))
                f.write(f"{batch_num},{rank_name},{n},{pct_hard:.4f},{pct_ordered_no_margin:.4f},{pct_margin:.4f},{ap_m:.6f},{an_m:.6f},{mg_m:.6f},{ls_m:.6f},{sc_m:.6f}\n")
    # Write by-bucket CSV
    if buckets is not None:
        write_header = not os.path.exists(csv_by_bucket_path)
        with open(csv_by_bucket_path, "a") as f:
            if write_header:
                f.write("batch,bucket,n_triplets,pct_ap_gte_an,pct_ap_lt_an,pct_margin_satisfied,ap_dist_mean,an_dist_mean,margin_mean,loss_mean,scale_mean\n")
            for b in sorted(torch.unique(buckets).tolist()):
                mask = (buckets == b)
                n = mask.sum().item()
                if n > 0:
                    pct_hard = hard_violations[mask].float().mean().item() * 100
                    pct_ordered_no_margin = ordered_no_margin[mask].float().mean().item() * 100
                    pct_margin = margin_satisfied[mask].float().mean().item() * 100
                    ap_m = a_p_dist[mask].mean().item()
                    an_m = a_n_dist[mask].mean().item()
                    mg_m = margins[mask].mean().item()
                    ls_m = losses[mask].mean().item()
                    sc_m = scale[mask].mean().item()
                    f.write(f"{batch_num},{b},{n},{pct_hard:.4f},{pct_ordered_no_margin:.4f},{pct_margin:.4f},{ap_m:.6f},{an_m:.6f},{mg_m:.6f},{ls_m:.6f},{sc_m:.6f}\n")




# Pair Loss --------------------------

def _format_region_label(region_idx):
    """Convert a numeric region index into a readable label."""
    return "Full" if region_idx == 0 else f"R{region_idx}"


def _build_region_pair_rank_summary(region_pairs, ranks, losses, pair_rank_map):
    """
    Prepare summary stats for region-pair/rank tables.
    Returns None when inputs are missing or empty.
    """
    if region_pairs is None or losses.numel() == 0 or ranks.numel() == 0:
        return None
    if isinstance(region_pairs, torch.Tensor):
        if region_pairs.numel() == 0:
            return None
        region_pairs = region_pairs.detach().cpu().long()
    else:
        region_pairs = torch.as_tensor(region_pairs, dtype=torch.long)
    if region_pairs.ndim != 2 or region_pairs.shape[1] != 2:
        return None
    # Canonicalise region combos so R1-R2 == R2-R1
    sorted_pairs, _ = torch.sort(region_pairs, dim=1)
    unique_pairs_tensor = torch.unique(sorted_pairs, dim=0)
    if unique_pairs_tensor.shape[0] == 0:
        return None
    # Ensure deterministic ordering of region pair rows
    unique_pairs = [tuple(map(int, pair)) for pair in unique_pairs_tensor.tolist()]
    unique_pairs.sort()
    rank_ids = torch.unique(ranks).cpu().tolist()
    rank_ids.sort()
    if not rank_ids:
        return None
    rows = []
    for pair in unique_pairs:
        pair_mask = (sorted_pairs[:, 0] == pair[0]) & (sorted_pairs[:, 1] == pair[1])
        if not torch.any(pair_mask):
            continue
        mean_values = []
        counts = []
        for rank_id in rank_ids:
            rank_mask = pair_mask & (ranks == rank_id)
            count = int(rank_mask.sum().item())
            counts.append(count)
            if count > 0:
                mean_values.append(losses[rank_mask].mean().item())
            else:
                mean_values.append(None)
        overall_count = int(pair_mask.sum().item())
        overall_mean = losses[pair_mask].mean().item() if overall_count > 0 else None
        label = f"{_format_region_label(pair[0])}-{_format_region_label(pair[1])}"
        rows.append({
            "label": label,
            "mean_values": mean_values,
            "counts": counts,
            "overall_mean": overall_mean,
            "overall_count": overall_count,
        })
    return {
        "rank_ids": rank_ids,
        "rank_labels": [pair_rank_map.get(r, str(r)) for r in rank_ids],
        "rows": rows,
    }

def print_pair_loss_stats(relative_losses, pred_dists, true_distances, ranks, buckets=None, region_pairs=None):
    """Helper to print stats for embedding_pair_loss."""
    # Early return if no pairs
    if len(ranks) == 0:
        return
    
    # Move tensors to CPU if needed
    relative_losses = relative_losses.detach().cpu() if relative_losses.is_cuda else relative_losses.detach()
    pred_dists = pred_dists.detach().cpu() if pred_dists.is_cuda else pred_dists.detach()
    true_distances = true_distances.detach().cpu() if true_distances.is_cuda else true_distances.detach()
    ranks = ranks.detach().cpu() if ranks.is_cuda else ranks.detach()
    if buckets is not None:
        buckets = buckets.detach().cpu() if buckets.is_cuda else buckets.detach()
    
    # Pair rank map: pair_rank -> name (pair_rank = shared_rank + 1)
    pair_rank_map = {0: 'Domain', 1: 'Phylum', 2: 'Class', 3: 'Order', 4: 'Family', 
                     5: 'Genus', 6: 'Species', 7: 'Sequence', 8: 'Subseq'}
    
    abs_errors = torch.abs(pred_dists - true_distances)
    epsilons_tensor = torch.tensor(gb.RELATIVE_ERROR_EPSILONS_PAIR_LOSS, device=true_distances.device, dtype=true_distances.dtype)
    pair_epsilons = epsilons_tensor[ranks.long()]
    safe_true_dists = torch.maximum(true_distances, pair_epsilons)
    abs_rel_errors = abs_errors / safe_true_dists
    signed_errors = pred_dists - true_distances
    is_pos = signed_errors > 0
    
    # Compute scaling factors for display (1/true_distance)
    scale = 1.0 / safe_true_dists
    
    print("\nPair Loss Verbose  -------------------------------------")
    print("Pair Loss Statistics:")
    print(f"  {'Rank':<8} | {'Num Pairs':<9} | {'Mean Loss':<9} | {'Mean Abs Err':<12} | {'Mean Abs Rel Err':<16} | {'% Pos':<8} | {'% Neg':<8} | {'Scaling':<10}")
    def _p(name, mask):
        if not mask.any(): return
        n = mask.sum().item()
        pct_pos = (is_pos[mask].float().mean().item() * 100) if mask.any() else 0.0
        pct_neg = 100.0 - pct_pos
        mean_scale = scale[mask].mean().item()
        print(f"  {name:<8} | {n:<9} | {relative_losses[mask].mean():.4f}    | {abs_errors[mask].mean():.4f}       | {abs_rel_errors[mask].mean():.4f}           | {pct_pos:>6.1f}%  | {pct_neg:>6.1f}%  | {mean_scale:.4f}")
    _p("Overall", torch.ones_like(relative_losses, dtype=torch.bool))
    for rank in sorted(torch.unique(ranks).tolist()):
        rank_name = pair_rank_map.get(rank, str(rank))
        _p(rank_name, ranks == rank)
    
    print("Pair Loss by Bucket & Sign:")
    print(f"  {'Bucket':<8} | {'Num Pairs':<9} | {'Num Pos':<9} | {'Num Neg':<9} | {'All':<9} | {'Pos':<9} | {'Neg':<9} | {'Scaling':<10}")
    def _p_b(name, mask):
        if not mask.any(): return
        mask_pos = mask & is_pos
        mask_neg = mask & (~is_pos)
        l_all = relative_losses[mask].mean()
        l_pos = relative_losses[mask_pos].mean() if mask_pos.any() else 0.0
        l_neg = relative_losses[mask_neg].mean() if mask_neg.any() else 0.0
        mean_scale = scale[mask].mean().item()
        print(f"  {name:<8} | {mask.sum():<9} | {mask_pos.sum():<9} | {mask_neg.sum():<9} | {l_all:.4f}    | {l_pos:.4f}    | {l_neg:.4f}    | {mean_scale:.4f}")
    if buckets is not None:
        for b in sorted(torch.unique(buckets).tolist()):
            _p_b(str(b), buckets == b)
    _p_b("Overall", torch.ones_like(relative_losses, dtype=torch.bool))

    # Region + rank breakdowns
    summary = _build_region_pair_rank_summary(region_pairs, ranks, relative_losses, pair_rank_map)
    if summary and summary["rows"]:
        rank_headers = " ".join([f"{label:<10}" for label in summary["rank_labels"]])
        print("\nPair Loss Mean Loss by Region Pair & Rank:")
        print(f"  {'Region Pair':<16} {rank_headers} {'Overall':<10}")
        
        max_lines = gb.CROSS_REGION_LOGGING_MAX_LINES_PER_TABLE
        rows_printed = 0
        
        for row in summary["rows"]:
            if max_lines is not None and rows_printed >= max_lines:
                print(f"  {'...':<16}")
                break
                
            row_str = f"  {row['label']:<16} "
            for mean_val in row["mean_values"]:
                if mean_val is not None:
                    row_str += f"{mean_val:<10.4f}"
                else:
                    row_str += f"{'-':<10}"
                row_str += " "
            overall = row["overall_mean"]
            row_str += f"{overall:<10.4f}" if overall is not None else f"{'-':<10}"
            print(row_str)
            rows_printed += 1

        print("Pair Counts by Region Pair & Rank:")
        print(f"  {'Region Pair':<16} {rank_headers} {'Overall':<10}")
        
        rows_printed = 0
        for row in summary["rows"]:
            if max_lines is not None and rows_printed >= max_lines:
                print(f"  {'...':<16}")
                break
                
            row_str = f"  {row['label']:<16} "
            for count in row["counts"]:
                row_str += f"{count:<10} "
            row_str += f"{row['overall_count']:<10}"
            print(row_str)
            rows_printed += 1

def write_pair_loss_log(batch_num, logs_dir, relative_losses, pred_dists, true_distances, ranks, buckets=None, region_pairs=None):
    """
    Write pair loss statistics to log and CSV files.
    
    Log files:
        pair_loss.log: Text dump of printed output with batch headers (includes region pair tables when available)
    
    CSV files:
        pair_loss.csv: Overall stats per batch
            Columns: batch, n_pairs, mean_loss, mean_abs_err, mean_abs_rel_err,
                     pred_dist_min, pred_dist_max, pred_dist_mean, pred_dist_std,
                     true_dist_min, true_dist_max, true_dist_mean, true_dist_std,
                     n_pos_err, n_neg_err, pct_pos_err, pct_neg_err, scale_mean
        
        pair_loss_by_rank.csv: Stats per rank per batch
            Columns: batch, rank, n_pairs, mean_loss, mean_abs_err, mean_abs_rel_err,
                     pred_dist_mean, true_dist_mean, n_pos_err, n_neg_err, pct_pos_err, pct_neg_err, scale_mean
        
        pair_loss_by_bucket.csv: Stats per bucket per batch
            Columns: batch, bucket, n_pairs, n_pos, n_neg, loss_all, loss_pos, loss_neg, scale_mean
    """
    # Early return if no pairs
    if len(ranks) == 0:
        return
    
    # Move tensors to CPU for file I/O operations
    relative_losses = relative_losses.detach().cpu()
    pred_dists = pred_dists.detach().cpu()
    true_distances = true_distances.detach().cpu()
    ranks = ranks.detach().cpu()
    if buckets is not None:
        buckets = buckets.detach().cpu()
    log_path = os.path.join(logs_dir, "pair_loss.log")
    csv_path = os.path.join(logs_dir, "pair_loss.csv")
    csv_by_rank_path = os.path.join(logs_dir, "pair_loss_by_rank.csv")
    csv_by_bucket_path = os.path.join(logs_dir, "pair_loss_by_bucket.csv")
    
    # Pair rank map: pair_rank -> name (pair_rank = shared_rank + 1)
    pair_rank_map = {0: 'Domain', 1: 'Phylum', 2: 'Class', 3: 'Order', 4: 'Family', 
                     5: 'Genus', 6: 'Species', 7: 'Sequence', 8: 'Subseq'}
    
    abs_errors = torch.abs(pred_dists - true_distances)
    epsilons_tensor = torch.tensor(gb.RELATIVE_ERROR_EPSILONS_PAIR_LOSS, device=true_distances.device, dtype=true_distances.dtype)
    pair_epsilons = epsilons_tensor[ranks.long()]
    safe_true_dists = torch.maximum(true_distances, pair_epsilons)
    abs_rel_errors = abs_errors / safe_true_dists
    signed_errors = pred_dists - true_distances
    is_pos = signed_errors > 0
    n_pairs = len(ranks)
    n_pos = is_pos.sum().item()
    n_neg = n_pairs - n_pos
    
    # Compute scaling factors (1/true_distance)
    scale = 1.0 / safe_true_dists
    
    # Write .log file
    with open(log_path, "a") as f:
        f.write(f"\n---\nBatch {batch_num}\n")
        f.write(f"\nPair Loss Logging -------------------------------------\n")
        f.write("Pair Loss Statistics:\n")
        f.write(f"  {'Rank':<8} | {'Num Pairs':<9} | {'Mean Loss':<9} | {'Mean Abs Err':<12} | {'Mean Abs Rel Err':<16} | {'% Pos':<8} | {'% Neg':<8} | {'Scaling':<10}\n")
        def _w(name, mask):
            if not mask.any(): return
            n = mask.sum().item()
            pct_pos = (is_pos[mask].float().mean().item() * 100) if mask.any() else 0.0
            pct_neg = 100.0 - pct_pos
            mean_scale = scale[mask].mean().item()
            f.write(f"  {name:<8} | {n:<9} | {relative_losses[mask].mean():.4f}    | {abs_errors[mask].mean():.4f}       | {abs_rel_errors[mask].mean():.4f}           | {pct_pos:>6.1f}%  | {pct_neg:>6.1f}%  | {mean_scale:.4f}\n")
        _w("Overall", torch.ones_like(relative_losses, dtype=torch.bool))
        for rank in sorted(torch.unique(ranks).tolist()):
            rank_name = pair_rank_map.get(rank, str(rank))
            _w(rank_name, ranks == rank)
        f.write("Pair Loss by Bucket & Sign:\n")
        f.write(f"  {'Bucket':<8} | {'Num Pairs':<9} | {'Num Pos':<9} | {'Num Neg':<9} | {'All':<9} | {'Pos':<9} | {'Neg':<9} | {'Scaling':<10}\n")
        def _w_b(name, mask):
            if not mask.any(): return
            mask_pos = mask & is_pos
            mask_neg = mask & (~is_pos)
            l_all = relative_losses[mask].mean().item()
            l_pos = relative_losses[mask_pos].mean().item() if mask_pos.any() else 0.0
            l_neg = relative_losses[mask_neg].mean().item() if mask_neg.any() else 0.0
            mean_scale = scale[mask].mean().item()
            f.write(f"  {name:<8} | {mask.sum():<9} | {mask_pos.sum():<9} | {mask_neg.sum():<9} | {l_all:.4f}    | {l_pos:.4f}    | {l_neg:.4f}    | {mean_scale:.4f}\n")
        if buckets is not None:
            for b in sorted(torch.unique(buckets).tolist()):
                _w_b(str(b), buckets == b)
        _w_b("Overall", torch.ones_like(relative_losses, dtype=torch.bool))
        summary = _build_region_pair_rank_summary(region_pairs, ranks, relative_losses, pair_rank_map)
        if summary and summary["rows"]:
            rank_headers = " ".join([f"{label:<10}" for label in summary["rank_labels"]])
            f.write("Pair Loss Mean Loss by Region Pair & Rank:\n")
            f.write(f"  {'Region Pair':<16} {rank_headers} {'Overall':<10}\n")
            
            max_lines = gb.CROSS_REGION_LOGGING_MAX_LINES_PER_TABLE
            rows_printed = 0
            
            for row in summary["rows"]:
                if max_lines is not None and rows_printed >= max_lines:
                    f.write(f"  {'...':<16}\n")
                    break

                row_str = f"  {row['label']:<16} "
                for mean_val in row["mean_values"]:
                    if mean_val is not None:
                        row_str += f"{mean_val:<10.4f}"
                    else:
                        row_str += f"{'-':<10}"
                    row_str += " "
                overall = row["overall_mean"]
                row_str += f"{overall:<10.4f}" if overall is not None else f"{'-':<10}"
                f.write(row_str + "\n")
                rows_printed += 1

            f.write("Pair Counts by Region Pair & Rank:\n")
            f.write(f"  {'Region Pair':<16} {rank_headers} {'Overall':<10}\n")
            
            rows_printed = 0
            for row in summary["rows"]:
                if max_lines is not None and rows_printed >= max_lines:
                    f.write(f"  {'...':<16}\n")
                    break

                row_str = f"  {row['label']:<16} "
                for count in row["counts"]:
                    row_str += f"{count:<10} "
                row_str += f"{row['overall_count']:<10}"
                f.write(row_str + "\n")
                rows_printed += 1
    # Write main CSV
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a") as f:
        if write_header:
            f.write("batch,n_pairs,mean_loss,mean_abs_err,mean_abs_rel_err,pred_dist_min,pred_dist_max,pred_dist_mean,pred_dist_std,true_dist_min,true_dist_max,true_dist_mean,true_dist_std,n_pos_err,n_neg_err,pct_pos_err,pct_neg_err,scale_mean\n")
        mean_loss = relative_losses.mean().item()
        mean_abs_err = abs_errors.mean().item()
        mean_abs_rel_err = abs_rel_errors.mean().item()
        pct_pos = (n_pos / n_pairs * 100) if n_pairs > 0 else 0.0
        pct_neg = (n_neg / n_pairs * 100) if n_pairs > 0 else 0.0
        mean_scale = scale.mean().item()
        f.write(f"{batch_num},{n_pairs},{mean_loss:.6f},{mean_abs_err:.6f},{mean_abs_rel_err:.6f},{pred_dists.min():.6f},{pred_dists.max():.6f},{pred_dists.mean():.6f},{pred_dists.std():.6f},{true_distances.min():.6f},{true_distances.max():.6f},{true_distances.mean():.6f},{true_distances.std():.6f},{n_pos},{n_neg},{pct_pos:.4f},{pct_neg:.4f},{mean_scale:.6f}\n")
    # Write by-rank CSV
    write_header = not os.path.exists(csv_by_rank_path)
    with open(csv_by_rank_path, "a") as f:
        if write_header:
            f.write("batch,rank,n_pairs,mean_loss,mean_abs_err,mean_abs_rel_err,pred_dist_mean,true_dist_mean,n_pos_err,n_neg_err,pct_pos_err,pct_neg_err,scale_mean\n")
        for rank in sorted(torch.unique(ranks).tolist()):
            mask = (ranks == rank)
            n = mask.sum().item()
            if n > 0:
                ml = relative_losses[mask].mean().item()
                mae = abs_errors[mask].mean().item()
                mare = abs_rel_errors[mask].mean().item()
                pdm = pred_dists[mask].mean().item()
                tdm = true_distances[mask].mean().item()
                np_r = (mask & is_pos).sum().item()
                nn_r = (mask & (~is_pos)).sum().item()
                pct_pos_r = (np_r / n * 100) if n > 0 else 0.0
                pct_neg_r = (nn_r / n * 100) if n > 0 else 0.0
                sc_m = scale[mask].mean().item()
                rank_name = pair_rank_map.get(rank, str(rank))
                f.write(f"{batch_num},{rank_name},{n},{ml:.6f},{mae:.6f},{mare:.6f},{pdm:.6f},{tdm:.6f},{np_r},{nn_r},{pct_pos_r:.4f},{pct_neg_r:.4f},{sc_m:.6f}\n")
    # Write by-bucket CSV
    if buckets is not None:
        write_header = not os.path.exists(csv_by_bucket_path)
        with open(csv_by_bucket_path, "a") as f:
            if write_header:
                f.write("batch,bucket,n_pairs,n_pos,n_neg,loss_all,loss_pos,loss_neg,scale_mean\n")
            for b in sorted(torch.unique(buckets).tolist()):
                mask = (buckets == b)
                n = mask.sum().item()
                if n > 0:
                    mask_pos = mask & is_pos
                    mask_neg = mask & (~is_pos)
                    l_all = relative_losses[mask].mean().item()
                    l_pos = relative_losses[mask_pos].mean().item() if mask_pos.any() else 0.0
                    l_neg = relative_losses[mask_neg].mean().item() if mask_neg.any() else 0.0
                    sc_m = scale[mask].mean().item()
                    f.write(f"{batch_num},{b},{n},{mask_pos.sum().item()},{mask_neg.sum().item()},{l_all:.6f},{l_pos:.6f},{l_neg:.6f},{sc_m:.6f}\n")





# Pair Mining --------------------------

def print_pair_mining_stats(relative_errors, signed_errors, bucket_assignments, sampled_local_indices, valid_flat_indices, percentile_thresholds, flat_true_distances, flat_pred_distances, train_regions, n_train_seqs, pairwise_ranks, per_rank_stats=None):
    """Helper to print useful statistics during pair mining."""
    try:        
        n_buckets = len(percentile_thresholds) + 1
        flat_ranks_full = pairwise_ranks.ravel()
        pool_ranks = flat_ranks_full[valid_flat_indices]
        rank_map = {-2: 'Ignore', -1: 'Domain', 0: 'Phylum', 1: 'Class', 2: 'Order', 3: 'Family', 4: 'Genus', 5: 'Species', 6: 'Sequence', 7: 'Subsequence'}
        # Map pair rank index (0-8) to internal rank (-1 to 7)
        pair_rank_idx_to_internal = {i: i - 1 for i in range(9)}  # 0->-1, 1->0, ..., 8->7
        pair_rank_idx_to_name = {0: 'Domain', 1: 'Phylum', 2: 'Class', 3: 'Order', 4: 'Family', 5: 'Genus', 6: 'Species', 7: 'Sequence', 8: 'Subseq'}
        unique_ranks = np.unique(pool_ranks)
        unique_ranks.sort()
        global_unique_ranks, global_counts = np.unique(flat_ranks_full, return_counts=True)
        global_counts_map = dict(zip(global_unique_ranks, global_counts))
        print("\nPair Mining Verbose  -------------------------------------")
        
        # Print per-rank mining summary (new section for per-rank mining)
        if per_rank_stats is not None:
            warmup = per_rank_stats.get('warmup_phase', 0.0)
            print(f"\nPer-Rank Mining Summary (warmup={warmup:.2%}):")
            print(f"  {'Rank':<10} | {'EMA Pre':<8} | {'EMA Warm':<8} | {'EMA Post':<8} | {'BatchHard':<10} | {'Budget':<8} | {'Prop %':<8} | {'Pool':<8} | {'Sampled':<8} | {'Deficit':<8}")
            print(f"  {'-'*10}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*10}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")
            
            ema_pre = per_rank_stats.get('ema_hardness_pre')
            ema_budget = per_rank_stats.get('ema_hardness_budget')
            ema_post = per_rank_stats.get('ema_hardness_post')
            hardness_metrics = per_rank_stats.get('hardness_metrics')
            budgets = per_rank_stats.get('budgets')
            proportions = per_rank_stats.get('proportions')
            pool_counts = per_rank_stats.get('pool_counts')
            sampled_counts = per_rank_stats.get('sampled_counts')
            deficits = per_rank_stats.get('deficits')
            
            for rank_idx in range(9):
                rank_name = pair_rank_idx_to_name.get(rank_idx, str(rank_idx))
                ema_pre_val = f"{ema_pre[rank_idx]:.4f}" if ema_pre is not None else "-"
                ema_budget_val = f"{ema_budget[rank_idx]:.4f}" if ema_budget is not None else "-"
                ema_post_val = f"{ema_post[rank_idx]:.4f}" if ema_post is not None else "-"
                batch_hard_val = "-"
                if hardness_metrics is not None and not np.isnan(hardness_metrics[rank_idx]):
                    batch_hard_val = f"{hardness_metrics[rank_idx]:.4f}"
                budget_val = f"{budgets[rank_idx]}" if budgets is not None else "-"
                prop_val = f"{proportions[rank_idx]*100:.1f}%" if proportions is not None else "-"
                pool_val = f"{pool_counts[rank_idx]}" if pool_counts is not None else "-"
                sampled_val = f"{sampled_counts[rank_idx]}" if sampled_counts is not None else "-"
                deficit_val = f"{deficits[rank_idx]}" if deficits is not None else "-"
                print(f"  {rank_name:<10} | {ema_pre_val:<8} | {ema_budget_val:<8} | {ema_post_val:<8} | {batch_hard_val:<10} | {budget_val:<8} | {prop_val:<8} | {pool_val:<8} | {sampled_val:<8} | {deficit_val:<8}")
            
            # Print totals
            total_budget = budgets.sum() if budgets is not None else 0
            total_pool = pool_counts.sum() if pool_counts is not None else 0
            total_sampled = sampled_counts.sum() if sampled_counts is not None else 0
            total_deficit = deficits.sum() if deficits is not None else 0
            print(f"  {'-'*10}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*10}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")
            print(f"  {'TOTAL':<10} | {'-':<8} | {'-':<8} | {'-':<8} | {'-':<10} | {total_budget:<8} | {'100%':<8} | {total_pool:<8} | {total_sampled:<8} | {total_deficit:<8}")
            
            bucket_base = per_rank_stats.get('bucket_proportions_base')
            bucket_target = per_rank_stats.get('bucket_proportions_target')
            if bucket_base is not None and bucket_target is not None:
                print("\nBucket Target Proportions:")
                base_str = _format_bucket_vector(bucket_base)
                target_str = _format_bucket_vector(bucket_target)
                if base_str:
                    print(f"  Base:   {base_str}")
                if target_str:
                    print(f"  Target: {target_str}")
            
            # Print per-rank bucket thresholds
            per_rank_thresholds = per_rank_stats.get('per_rank_thresholds', {})
            if per_rank_thresholds:
                print("\nPer-Rank Bucket Thresholds (Relative Error):")
                for rank_idx in range(9):
                    internal_rank = pair_rank_idx_to_internal[rank_idx]
                    rank_name = pair_rank_idx_to_name.get(rank_idx, str(rank_idx))
                    thresholds = per_rank_thresholds.get(internal_rank, [])
                    if thresholds:
                        thresh_str = ", ".join([f"{t:.4f}" for t in thresholds])
                        print(f"  {rank_name:<10}: [{thresh_str}]")
                    elif budgets is not None and budgets[rank_idx] > 0:
                        print(f"  {rank_name:<10}: (uniform/warmup)")
            
            bucket_diag_all = per_rank_stats.get('bucket_diagnostics')
            if bucket_diag_all and any(entry is not None for entry in bucket_diag_all):
                print("\nPer-Rank Bucket Fulfillment:")
                for rank_idx in range(9):
                    diag_entry = bucket_diag_all[rank_idx] if bucket_diag_all is not None else None
                    if not diag_entry:
                        continue
                    diag_str = _format_bucket_diag_entries(diag_entry)
                    if not diag_str:
                        continue
                    rank_name = pair_rank_idx_to_name.get(rank_idx, str(rank_idx))
                    print(f"  {rank_name:<10}: {diag_str}")
            
            shortages = per_rank_stats.get('representative_shortages') if per_rank_stats else None
            if shortages:
                print("\nRepresentative Set Shortages:")
                for shortage in shortages:
                    deficit = shortage['requested'] - shortage['available']
                    print(f"  - {shortage['rank_name']}: requested {shortage['requested']} but found "
                          f"{shortage['available']} (short by {deficit}).")

        print("Bucket Relative Error Ranges:")
        for b in range(n_buckets):
            lower_pct = 0.0 if b == 0 else percentile_thresholds[b-1] * 100
            upper_pct = 100.0 if b == len(percentile_thresholds) else percentile_thresholds[b] * 100
            bucket_label = f"Bucket {b} ({lower_pct:.0f}-{upper_pct:.0f}%):"
            mask = (bucket_assignments == b)
            if np.any(mask):
                errs = relative_errors[mask]
                n_in_bucket = len(errs)
                n_pool_total = len(relative_errors)
                n_sampled_in_bucket = np.sum(bucket_assignments[sampled_local_indices] == b)
                n_sampled_total = len(sampled_local_indices)
                pct_pool = (n_in_bucket / n_pool_total * 100) if n_pool_total > 0 else 0.0
                pct_sampled = (n_sampled_in_bucket / n_sampled_total * 100) if n_sampled_total > 0 else 0.0
                print(f"  {bucket_label:<22} Range=[{errs.min():.4f}, {errs.max():.4f}] N={n_in_bucket:<6} Sampled={n_sampled_in_bucket:<6} ({pct_pool:3.0f}% -> {pct_sampled:3.0f}%)")
            else:
                print(f"  {bucket_label:<22} N=0")
        print("Relative Error Stats by Rank (Pool):")
        print(f"  {'Rank':<12} {'N Total':<10} {'N':<6} {'Min':<8} {'Mean':<8} {'Max':<8} {'Std':<8} {'5%':<8} {'95%':<8} {'% Pos':<8} {'% Neg':<8}")
        total_n_global = 0
        for r in unique_ranks:
            mask = (pool_ranks == r)
            errs = relative_errors[mask]
            signs = signed_errors[mask]
            r_name = rank_map.get(r, str(r))
            n_total = len(errs)
            n_global = global_counts_map.get(r, 0)
            total_n_global += n_global
            n_pos = np.sum(signs > 0)
            n_neg = np.sum(signs < 0)
            pct_pos = (n_pos / n_total * 100) if n_total > 0 else 0.0
            pct_neg = (n_neg / n_total * 100) if n_total > 0 else 0.0
            print(f"  {r_name:<12} {n_global:<10} {n_total:<6} {errs.min():.4f}   {errs.mean():.4f}   {errs.max():.4f}   {errs.std():.4f}   {np.percentile(errs, 5):.4f}   {np.percentile(errs, 95):.4f}   {pct_pos:6.1f}%   {pct_neg:6.1f}%")
        if len(relative_errors) > 0:
            n_total_all = len(relative_errors)
            n_pos_all = np.sum(signed_errors > 0)
            n_neg_all = np.sum(signed_errors < 0)
            pct_pos_all = (n_pos_all / n_total_all * 100)
            pct_neg_all = (n_neg_all / n_total_all * 100)
            print(f"  {'All Ranks':<12} {total_n_global:<10} {n_total_all:<6} {relative_errors.min():.4f}   {relative_errors.mean():.4f}   {relative_errors.max():.4f}   {relative_errors.std():.4f}   {np.percentile(relative_errors, 5):.4f}   {np.percentile(relative_errors, 95):.4f}   {pct_pos_all:6.1f}%   {pct_neg_all:6.1f}%")
        print("Rank Proportions and True Distances (Pool vs Sampled):")
        sampled_ranks = pool_ranks[sampled_local_indices]
        n_pool = len(pool_ranks)
        n_sampled = len(sampled_ranks)
        print(f"  {'Rank':<12} {'Pool %':<8} {'Sampled %':<8} {'Count (T)':<10} {'Count (P)':<10} {'Count (S)':<10} {'Min':<8} {'5%':<8} {'Mean':<8} {'95%':<8} {'Max':<8}")
        for r in unique_ranks:
            n_g = global_counts_map.get(r, 0)
            n_p = np.sum(pool_ranks == r)
            n_s = np.sum(sampled_ranks == r)
            p_pool = n_p / n_pool * 100
            p_sampled = n_s / n_sampled * 100 if n_sampled > 0 else 0
            r_name = rank_map.get(r, str(r))
            mask = (pool_ranks == r)
            dists = flat_true_distances[mask]
            d_min, d_5, d_mean, d_95, d_max = dists.min(), np.percentile(dists, 5), dists.mean(), np.percentile(dists, 95), dists.max()
            print(f"  {r_name:<12} {p_pool:6.2f}%   {p_sampled:6.2f}%   {n_g:<10} {n_p:<10} {n_s:<10} {d_min:.4f}   {d_5:.4f}   {d_mean:.4f}   {d_95:.4f}   {d_max:.4f}")
        print("Bucket Composition by Rank (Pool):")
        rank_names = [rank_map.get(r, str(r)) for r in unique_ranks]
        print(f"  {'Bucket':<22} {'N':<6} " + " ".join([f"{name:<8}" for name in rank_names]))
        for b in range(n_buckets):
            lower_pct = 0.0 if b == 0 else percentile_thresholds[b-1] * 100
            upper_pct = 100.0 if b == len(percentile_thresholds) else percentile_thresholds[b] * 100
            bucket_label = f"{b} ({lower_pct:.0f}-{upper_pct:.0f}%)"
            mask = (bucket_assignments == b)
            n_in_bucket = np.sum(mask)
            if n_in_bucket > 0:
                ranks_in_bucket = pool_ranks[mask]
                unique_in_bucket, counts_in_bucket = np.unique(ranks_in_bucket, return_counts=True)
                counts_map = dict(zip(unique_in_bucket, counts_in_bucket))
                row_str = f"  {bucket_label:<22} {n_in_bucket:<6}"
                for r in unique_ranks:
                    count = counts_map.get(r, 0)
                    pct = count / n_in_bucket * 100
                    row_str += f" {pct:<8.1f}"
                print(row_str)
            else:
                print(f"  {bucket_label:<22} {0:<6}")
        pool_seq_i = valid_flat_indices // n_train_seqs
        pool_seq_j = valid_flat_indices % n_train_seqs
        regions_i = train_regions[pool_seq_i]
        regions_j = train_regions[pool_seq_j]
        stacked_regions = np.stack([regions_i, regions_j], axis=1)
        stacked_regions.sort(axis=1)
        unique_region_pairs = np.unique(stacked_regions, axis=0)
        def get_region_label(r_idx):
            return "Full" if r_idx == 0 else f"R{r_idx}"
        rank_cols = [rank_map.get(r, str(r)) for r in unique_ranks]
        abs_errors = np.abs(flat_pred_distances - flat_true_distances)
        def _print_region_pair_metric_table(title, values):
            print(title)
            header = f"  {'Region Pair':<16} " + " ".join([f"{rc:<10}" for rc in rank_cols]) + f" {'Overall':<10}"
            print(header)
            
            max_lines = gb.CROSS_REGION_LOGGING_MAX_LINES_PER_TABLE
            rows_printed = 0
            
            for r_pair in unique_region_pairs:
                if max_lines is not None and rows_printed >= max_lines:
                    print(f"  {'...':<16}")
                    break
                    
                r1, r2 = r_pair
                label = f"{get_region_label(r1)}-{get_region_label(r2)}"
                pair_mask = (stacked_regions[:, 0] == r1) & (stacked_regions[:, 1] == r2)
                row_str = f"  {label:<16} "
                for r in unique_ranks:
                    rank_mask = (pool_ranks == r)
                    combined_mask = pair_mask & rank_mask
                    if np.any(combined_mask):
                        vals = values[combined_mask]
                        mean_val = vals.mean()
                        row_str += f"{mean_val:<10.4f} "
                    else:
                        row_str += f"{'-':<10} "
                if np.any(pair_mask):
                    overall_vals = values[pair_mask]
                    overall_mean = overall_vals.mean()
                    row_str += f"{overall_mean:<10.4f}"
                else:
                    row_str += f"{'-':<10}"
                print(row_str)
                rows_printed += 1
        def _print_region_pair_count_table(title):
            print(title)
            header = f"  {'Region Pair':<16} " + " ".join([f"{rc:<10}" for rc in rank_cols]) + f" {'Overall':<10}"
            print(header)
            
            max_lines = gb.CROSS_REGION_LOGGING_MAX_LINES_PER_TABLE
            rows_printed = 0
            
            for r_pair in unique_region_pairs:
                if max_lines is not None and rows_printed >= max_lines:
                    print(f"  {'...':<16}")
                    break
                    
                r1, r2 = r_pair
                label = f"{get_region_label(r1)}-{get_region_label(r2)}"
                pair_mask = (stacked_regions[:, 0] == r1) & (stacked_regions[:, 1] == r2)
                row_str = f"  {label:<16} "
                for r in unique_ranks:
                    rank_mask = (pool_ranks == r)
                    combined_mask = pair_mask & rank_mask
                    count = int(np.sum(combined_mask))
                    row_str += f"{count:<10} "
                overall_count = int(np.sum(pair_mask))
                row_str += f"{overall_count:<10}"
                print(row_str)
                rows_printed += 1
        _print_region_pair_metric_table("Predicted Distances Stats by Region Pair (Pool):", flat_pred_distances)
        _print_region_pair_metric_table("Absolute Error Stats by Region Pair (Pool):", abs_errors)
        _print_region_pair_metric_table("Relative Error Stats by Region Pair (Pool):", relative_errors)
        _print_region_pair_count_table("Pair Counts by Region Pair & Rank (Pool):")
        print("Predicted Distances & True Distances (Pool):")
        print(f"  {'Rank':<12} {'Count (P)':<10} {'Min (T)':<8} {'Min (P)':<8} {'5% (T)':<8} {'5% (P)':<8} {'Mean (T)':<8} {'Mean (P)':<8} {'95% (T)':<8} {'95% (P)':<8} {'Max (T)':<8} {'Max (P)':<8} {'Mean AE':<8}")
        for r in unique_ranks:
            mask = (pool_ranks == r)
            if np.any(mask):
                t_dists = flat_true_distances[mask]
                p_dists = flat_pred_distances[mask]      
                cnt = len(t_dists)
                t_min, t_5, t_mean, t_95, t_max = t_dists.min(), np.percentile(t_dists, 5), t_dists.mean(), np.percentile(t_dists, 95), t_dists.max()
                p_min, p_5, p_mean, p_95, p_max = p_dists.min(), np.percentile(p_dists, 5), p_dists.mean(), np.percentile(p_dists, 95), p_dists.max()
                mean_ae = np.mean(np.abs(t_dists - p_dists))
                r_name = rank_map.get(r, str(r))
                print(
                    f"  {r_name:<12} {cnt:<10} "
                    f"{_format_table_float_with_tiny_indicator(t_min)} "
                    f"{_format_table_float_with_tiny_indicator(p_min)} "
                    f"{_format_table_float_with_tiny_indicator(t_5)} "
                    f"{_format_table_float_with_tiny_indicator(p_5)} "
                    f"{_format_table_float_with_tiny_indicator(t_mean)} "
                    f"{_format_table_float_with_tiny_indicator(p_mean)} "
                    f"{_format_table_float_with_tiny_indicator(t_95)} "
                    f"{_format_table_float_with_tiny_indicator(p_95)} "
                    f"{_format_table_float_with_tiny_indicator(t_max)} "
                    f"{_format_table_float_with_tiny_indicator(p_max)} "
                    f"{_format_table_float_with_tiny_indicator(mean_ae)}"
                )
        if len(flat_true_distances) > 0:
            t_dists = flat_true_distances
            p_dists = flat_pred_distances
            cnt = len(t_dists)
            t_min, t_5, t_mean, t_95, t_max = t_dists.min(), np.percentile(t_dists, 5), t_dists.mean(), np.percentile(t_dists, 95), t_dists.max()
            p_min, p_5, p_mean, p_95, p_max = p_dists.min(), np.percentile(p_dists, 5), p_dists.mean(), np.percentile(p_dists, 95), p_dists.max()
            mean_ae = np.mean(np.abs(t_dists - p_dists))
            print(
                f"  {'All Ranks':<12} {cnt:<10} "
                f"{_format_table_float_with_tiny_indicator(t_min)} "
                f"{_format_table_float_with_tiny_indicator(p_min)} "
                f"{_format_table_float_with_tiny_indicator(t_5)} "
                f"{_format_table_float_with_tiny_indicator(p_5)} "
                f"{_format_table_float_with_tiny_indicator(t_mean)} "
                f"{_format_table_float_with_tiny_indicator(p_mean)} "
                f"{_format_table_float_with_tiny_indicator(t_95)} "
                f"{_format_table_float_with_tiny_indicator(p_95)} "
                f"{_format_table_float_with_tiny_indicator(t_max)} "
                f"{_format_table_float_with_tiny_indicator(p_max)} "
                f"{_format_table_float_with_tiny_indicator(mean_ae)}"
            )
    except Exception as e:
        print(f"Warning: Failed to print mining stats: {e}")


def write_pair_mining_log(batch_num, logs_dir, relative_errors, signed_errors, bucket_assignments, sampled_local_indices, valid_flat_indices, percentile_thresholds, flat_true_distances, flat_pred_distances, train_regions, n_train_seqs, pairwise_ranks, pair_distances_df=None, per_rank_stats=None, pair_error_metrics_df=None):
    """
    Write pair mining statistics to log and CSV files.
    
    Log files:
        pair_mining.log: Text dump of printed output with batch headers
    
    CSV files:
        pair_mining.csv: Overall stats per batch
            Columns: batch, n_pool, n_sampled, rel_err_min, rel_err_max, rel_err_mean, rel_err_std,
                     rel_err_5pct, rel_err_95pct, pct_pos_err, pct_neg_err, true_dist_mean, 
                     pred_dist_mean, mean_abs_err, warmup_phase, total_deficit,
                     bucket_base_proportions, bucket_target_proportions
        
        pair_mining_by_rank.csv: Stats per rank per batch (updated with per-rank mining data)
            Columns: batch, rank, rank_idx, ema_hardness_pre, ema_hardness_budget, ema_hardness_post,
                     batch_hardness, budget, proportion, n_pool, n_sampled, deficit, rel_err_mean,
                     rel_err_std, true_dist_mean, pred_dist_mean, mean_abs_err, bucket_thresholds,
                     bucket_targets, bucket_sampled, bucket_residual_deficit, bucket_borrowed_from
        
        pair_mining_by_bucket.csv: Stats per bucket per batch
            Columns: batch, bucket, pct_low, pct_high, n_in_bucket, n_sampled, rel_err_min, rel_err_max,
                     pct_pool, pct_sampled
    
    Args:
        pair_distances_df: Optional pd.DataFrame for tracking pair distances over time.
                          If provided, will be populated with distance statistics per rank.
        per_rank_stats: Optional dict containing per-rank mining statistics from mine_pairs()
        pair_error_metrics_df: Optional pd.DataFrame for tracking per-rank error metrics over time.
                               If provided, will be populated with the metric fed to the EMA.
    """
    try:
        log_path = os.path.join(logs_dir, "pair_mining.log")
        csv_path = os.path.join(logs_dir, "pair_mining.csv")
        csv_by_rank_path = os.path.join(logs_dir, "pair_mining_by_rank.csv")
        csv_by_bucket_path = os.path.join(logs_dir, "pair_mining_by_bucket.csv")
        n_buckets = len(percentile_thresholds) + 1
        flat_ranks_full = pairwise_ranks.ravel()
        pool_ranks = flat_ranks_full[valid_flat_indices]
        rank_map = {-2: 'Ignore', -1: 'Domain', 0: 'Phylum', 1: 'Class', 2: 'Order', 3: 'Family', 4: 'Genus', 5: 'Species', 6: 'Sequence', 7: 'Subsequence'}
        unique_ranks = np.unique(pool_ranks)
        unique_ranks.sort()
        global_unique_ranks, global_counts = np.unique(flat_ranks_full, return_counts=True)
        global_counts_map = dict(zip(global_unique_ranks, global_counts))
        sampled_ranks = pool_ranks[sampled_local_indices]
        n_pool = len(pool_ranks)
        n_sampled = len(sampled_ranks)
        # Map pair rank index (0-8) to internal rank (-1 to 7) and names
        pair_rank_idx_to_internal = {i: i - 1 for i in range(9)}
        pair_rank_idx_to_name = {0: 'Domain', 1: 'Phylum', 2: 'Class', 3: 'Order', 4: 'Family', 5: 'Genus', 6: 'Species', 7: 'Sequence', 8: 'Subseq'}
        
        # Write .log file (text dump)
        with open(log_path, "a") as f:
            f.write(f"\n---\nBatch {batch_num}\n")
            f.write(f"\nPair Mining Logging -------------------------------------\n")
            
            # Write per-rank mining summary (new section for per-rank mining)
            if per_rank_stats is not None:
                warmup = per_rank_stats.get('warmup_phase', 0.0)
                f.write(f"\nPer-Rank Mining Summary (warmup={warmup:.2%}):\n")
                f.write(f"  {'Rank':<10} | {'EMA Pre':<8} | {'EMA Warm':<8} | {'EMA Post':<8} | {'BatchHard':<10} | {'Budget':<8} | {'Prop %':<8} | {'Pool':<8} | {'Sampled':<8} | {'Deficit':<8}\n")
                f.write(f"  {'-'*10}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*10}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}\n")
                
                ema_pre = per_rank_stats.get('ema_hardness_pre')
                ema_budget = per_rank_stats.get('ema_hardness_budget')
                ema_post = per_rank_stats.get('ema_hardness_post')
                hardness_metrics = per_rank_stats.get('hardness_metrics')
                budgets = per_rank_stats.get('budgets')
                proportions = per_rank_stats.get('proportions')
                pool_counts = per_rank_stats.get('pool_counts')
                sampled_counts = per_rank_stats.get('sampled_counts')
                deficits = per_rank_stats.get('deficits')
                
                for rank_idx in range(9):
                    rank_name = pair_rank_idx_to_name.get(rank_idx, str(rank_idx))
                    ema_pre_val = f"{ema_pre[rank_idx]:.4f}" if ema_pre is not None else "-"
                    ema_budget_val = f"{ema_budget[rank_idx]:.4f}" if ema_budget is not None else "-"
                    ema_post_val = f"{ema_post[rank_idx]:.4f}" if ema_post is not None else "-"
                    batch_hard_val = "-"
                    if hardness_metrics is not None and not np.isnan(hardness_metrics[rank_idx]):
                        batch_hard_val = f"{hardness_metrics[rank_idx]:.4f}"
                    budget_val = f"{budgets[rank_idx]}" if budgets is not None else "-"
                    prop_val = f"{proportions[rank_idx]*100:.1f}%" if proportions is not None else "-"
                    pool_val = f"{pool_counts[rank_idx]}" if pool_counts is not None else "-"
                    sampled_val = f"{sampled_counts[rank_idx]}" if sampled_counts is not None else "-"
                    deficit_val = f"{deficits[rank_idx]}" if deficits is not None else "-"
                    f.write(f"  {rank_name:<10} | {ema_pre_val:<8} | {ema_budget_val:<8} | {ema_post_val:<8} | {batch_hard_val:<10} | {budget_val:<8} | {prop_val:<8} | {pool_val:<8} | {sampled_val:<8} | {deficit_val:<8}\n")
                
                total_budget = budgets.sum() if budgets is not None else 0
                total_pool = pool_counts.sum() if pool_counts is not None else 0
                total_sampled = sampled_counts.sum() if sampled_counts is not None else 0
                total_deficit = deficits.sum() if deficits is not None else 0
                f.write(f"  {'-'*10}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*10}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}\n")
                f.write(f"  {'TOTAL':<10} | {'-':<8} | {'-':<8} | {'-':<8} | {'-':<10} | {total_budget:<8} | {'100%':<8} | {total_pool:<8} | {total_sampled:<8} | {total_deficit:<8}\n")
                
                bucket_base = per_rank_stats.get('bucket_proportions_base')
                bucket_target = per_rank_stats.get('bucket_proportions_target')
                if bucket_base is not None and bucket_target is not None:
                    base_str = _format_bucket_vector(bucket_base)
                    target_str = _format_bucket_vector(bucket_target)
                    if base_str or target_str:
                        f.write("\nBucket Target Proportions:\n")
                        if base_str:
                            f.write(f"  Base:   {base_str}\n")
                        if target_str:
                            f.write(f"  Target: {target_str}\n")
                
                per_rank_thresholds = per_rank_stats.get('per_rank_thresholds', {})
                if per_rank_thresholds:
                    f.write("\nPer-Rank Bucket Thresholds (Relative Error):\n")
                    for rank_idx in range(9):
                        internal_rank = pair_rank_idx_to_internal[rank_idx]
                        rank_name = pair_rank_idx_to_name.get(rank_idx, str(rank_idx))
                        thresholds = per_rank_thresholds.get(internal_rank, [])
                        if thresholds:
                            thresh_str = ", ".join([f"{t:.4f}" for t in thresholds])
                            f.write(f"  {rank_name:<10}: [{thresh_str}]\n")
                        elif budgets is not None and budgets[rank_idx] > 0:
                            f.write(f"  {rank_name:<10}: (uniform/warmup)\n")
                bucket_diag_all = per_rank_stats.get('bucket_diagnostics')
                if bucket_diag_all and any(entry is not None for entry in bucket_diag_all):
                    f.write("\nPer-Rank Bucket Fulfillment:\n")
                    for rank_idx in range(9):
                        diag_entry = bucket_diag_all[rank_idx] if bucket_diag_all is not None else None
                        if not diag_entry:
                            continue
                        diag_str = _format_bucket_diag_entries(diag_entry)
                        if not diag_str:
                            continue
                        rank_name = pair_rank_idx_to_name.get(rank_idx, str(rank_idx))
                        f.write(f"  {rank_name:<10}: {diag_str}\n")
                shortages = per_rank_stats.get('representative_shortages') if per_rank_stats else None
                if shortages:
                    f.write("\nRepresentative Set Shortages:\n")
                    for shortage in shortages:
                        deficit = shortage['requested'] - shortage['available']
                        f.write(f"  - {shortage['rank_name']}: requested {shortage['requested']} but found "
                                f"{shortage['available']} (short by {deficit}).\n")
                f.write("\n")
            
            f.write("Bucket Relative Error Ranges:\n")
            for b in range(n_buckets):
                lower_pct = 0.0 if b == 0 else percentile_thresholds[b-1] * 100
                upper_pct = 100.0 if b == len(percentile_thresholds) else percentile_thresholds[b] * 100
                bucket_label = f"Bucket {b} ({lower_pct:.0f}-{upper_pct:.0f}%):"
                mask = (bucket_assignments == b)
                if np.any(mask):
                    errs = relative_errors[mask]
                    n_in_bucket = len(errs)
                    n_pool_total = len(relative_errors)
                    n_sampled_in_bucket = np.sum(bucket_assignments[sampled_local_indices] == b)
                    n_sampled_total = len(sampled_local_indices)
                    pct_pool = (n_in_bucket / n_pool_total * 100) if n_pool_total > 0 else 0.0
                    pct_sampled = (n_sampled_in_bucket / n_sampled_total * 100) if n_sampled_total > 0 else 0.0
                    f.write(f"  {bucket_label:<22} Range=[{errs.min():.4f}, {errs.max():.4f}] N={n_in_bucket:<6} Sampled={n_sampled_in_bucket:<6} ({pct_pool:3.0f}% -> {pct_sampled:3.0f}%)\n")
                else:
                    f.write(f"  {bucket_label:<22} N=0\n")
            f.write("Relative Error Stats by Rank (Pool):\n")
            f.write(f"  {'Rank':<12} {'N Total':<10} {'N':<6} {'Min':<8} {'Mean':<8} {'Max':<8} {'Std':<8} {'5%':<8} {'95%':<8} {'% Pos':<8} {'% Neg':<8}\n")
            total_n_global = 0
            for r in unique_ranks:
                mask = (pool_ranks == r)
                errs = relative_errors[mask]
                signs = signed_errors[mask]
                r_name = rank_map.get(r, str(r))
                n_total = len(errs)
                n_global = global_counts_map.get(r, 0)
                total_n_global += n_global
                n_pos = np.sum(signs > 0)
                n_neg = np.sum(signs < 0)
                pct_pos = (n_pos / n_total * 100) if n_total > 0 else 0.0
                pct_neg = (n_neg / n_total * 100) if n_total > 0 else 0.0
                f.write(f"  {r_name:<12} {n_global:<10} {n_total:<6} {errs.min():.4f}   {errs.mean():.4f}   {errs.max():.4f}   {errs.std():.4f}   {np.percentile(errs, 5):.4f}   {np.percentile(errs, 95):.4f}   {pct_pos:6.1f}%   {pct_neg:6.1f}%\n")
            if len(relative_errors) > 0:
                n_total_all = len(relative_errors)
                n_pos_all = np.sum(signed_errors > 0)
                n_neg_all = np.sum(signed_errors < 0)
                pct_pos_all = (n_pos_all / n_total_all * 100)
                pct_neg_all = (n_neg_all / n_total_all * 100)
                f.write(f"  {'All Ranks':<12} {total_n_global:<10} {n_total_all:<6} {relative_errors.min():.4f}   {relative_errors.mean():.4f}   {relative_errors.max():.4f}   {relative_errors.std():.4f}   {np.percentile(relative_errors, 5):.4f}   {np.percentile(relative_errors, 95):.4f}   {pct_pos_all:6.1f}%   {pct_neg_all:6.1f}%\n")
            f.write("Rank Proportions and True Distances (Pool vs Sampled):\n")
            f.write(f"  {'Rank':<12} {'Pool %':<8} {'Sampled %':<8} {'Count (T)':<10} {'Count (P)':<10} {'Count (S)':<10} {'Min':<8} {'5%':<8} {'Mean':<8} {'95%':<8} {'Max':<8}\n")
            for r in unique_ranks:
                n_g = global_counts_map.get(r, 0)
                n_p = np.sum(pool_ranks == r)
                n_s = np.sum(sampled_ranks == r)
                p_pool = n_p / n_pool * 100 if n_pool > 0 else 0.0
                p_sampled = n_s / n_sampled * 100 if n_sampled > 0 else 0.0
                r_name = rank_map.get(r, str(r))
                mask = (pool_ranks == r)
                dists = flat_true_distances[mask]
                if len(dists) > 0:
                    d_min = dists.min()
                    d_5 = np.percentile(dists, 5)
                    d_mean = dists.mean()
                    d_95 = np.percentile(dists, 95)
                    d_max = dists.max()
                else:
                    d_min = d_5 = d_mean = d_95 = d_max = 0.0
                f.write(f"  {r_name:<12} {p_pool:6.2f}%   {p_sampled:6.2f}%   {n_g:<10} {n_p:<10} {n_s:<10} {d_min:.4f}   {d_5:.4f}   {d_mean:.4f}   {d_95:.4f}   {d_max:.4f}\n")
            f.write("Bucket Composition by Rank (Pool):\n")
            rank_names = [rank_map.get(r, str(r)) for r in unique_ranks]
            f.write(f"  {'Bucket':<22} {'N':<6} " + " ".join([f"{name:<8}" for name in rank_names]) + "\n")
            for b in range(n_buckets):
                lower_pct = 0.0 if b == 0 else percentile_thresholds[b-1] * 100
                upper_pct = 100.0 if b == len(percentile_thresholds) else percentile_thresholds[b] * 100
                bucket_label = f"{b} ({lower_pct:.0f}-{upper_pct:.0f}%)"
                mask = (bucket_assignments == b)
                n_in_bucket = np.sum(mask)
                if n_in_bucket > 0:
                    ranks_in_bucket = pool_ranks[mask]
                    unique_in_bucket, counts_in_bucket = np.unique(ranks_in_bucket, return_counts=True)
                    counts_map = dict(zip(unique_in_bucket, counts_in_bucket))
                    row_str = f"  {bucket_label:<22} {n_in_bucket:<6}"
                    for r in unique_ranks:
                        count = counts_map.get(r, 0)
                        pct = count / n_in_bucket * 100
                        row_str += f" {pct:<8.1f}"
                    f.write(row_str + "\n")
                else:
                    f.write(f"  {bucket_label:<22} {0:<6}\n")
            pool_seq_i = valid_flat_indices // n_train_seqs
            pool_seq_j = valid_flat_indices % n_train_seqs
            regions_i = train_regions[pool_seq_i]
            regions_j = train_regions[pool_seq_j]
            stacked_regions = np.stack([regions_i, regions_j], axis=1)
            stacked_regions.sort(axis=1)
            unique_region_pairs = np.unique(stacked_regions, axis=0)
            def get_region_label(r_idx):
                return "Full" if r_idx == 0 else f"R{r_idx}"
            rank_cols = [rank_map.get(r, str(r)) for r in unique_ranks]
            abs_errors = np.abs(flat_pred_distances - flat_true_distances)
            def _write_region_pair_metric_table(title, values):
                f.write(f"{title}\n")
                header = f"  {'Region Pair':<16} " + " ".join([f"{rc:<10}" for rc in rank_cols]) + f" {'Overall':<10}\n"
                f.write(header)
                
                max_lines = gb.CROSS_REGION_LOGGING_MAX_LINES_PER_TABLE
                rows_printed = 0
                
                for r_pair in unique_region_pairs:
                    if max_lines is not None and rows_printed >= max_lines:
                        f.write(f"  {'...':<16}\n")
                        break
                        
                    r1, r2 = r_pair
                    label = f"{get_region_label(r1)}-{get_region_label(r2)}"
                    pair_mask = (stacked_regions[:, 0] == r1) & (stacked_regions[:, 1] == r2)
                    row_str = f"  {label:<16} "
                    for r in unique_ranks:
                        rank_mask = (pool_ranks == r)
                        combined_mask = pair_mask & rank_mask
                        if np.any(combined_mask):
                            vals = values[combined_mask]
                            mean_val = vals.mean()
                            row_str += f"{mean_val:<10.4f} "
                        else:
                            row_str += f"{'-':<10} "
                    if np.any(pair_mask):
                        overall_vals = values[pair_mask]
                        overall_mean = overall_vals.mean()
                        row_str += f"{overall_mean:<10.4f}"
                    else:
                        row_str += f"{'-':<10}"
                    f.write(row_str + "\n")
                    rows_printed += 1
            def _write_region_pair_count_table(title):
                f.write(f"{title}\n")
                header = f"  {'Region Pair':<16} " + " ".join([f"{rc:<10}" for rc in rank_cols]) + f" {'Overall':<10}\n"
                f.write(header)
                
                max_lines = gb.CROSS_REGION_LOGGING_MAX_LINES_PER_TABLE
                rows_printed = 0
                
                for r_pair in unique_region_pairs:
                    if max_lines is not None and rows_printed >= max_lines:
                        f.write(f"  {'...':<16}\n")
                        break
                        
                    r1, r2 = r_pair
                    label = f"{get_region_label(r1)}-{get_region_label(r2)}"
                    pair_mask = (stacked_regions[:, 0] == r1) & (stacked_regions[:, 1] == r2)
                    row_str = f"  {label:<16} "
                    for r in unique_ranks:
                        rank_mask = (pool_ranks == r)
                        combined_mask = pair_mask & rank_mask
                        count = int(np.sum(combined_mask))
                        row_str += f"{count:<10} "
                    overall_count = int(np.sum(pair_mask))
                    row_str += f"{overall_count:<10}"
                    f.write(row_str + "\n")
                    rows_printed += 1
            def _format_distribution_row(label, values):
                row = f"  {label:<56}"
                if values is None or len(values) == 0:
                    row += f"{0:>10}" + "".join([f"{'-':>11}"] * 10)
                    return row
                arr = np.asarray(values, dtype=np.float64)
                stats = [
                    arr.min(),
                    np.percentile(arr, 1),
                    np.percentile(arr, 5),
                    np.percentile(arr, 25),
                    np.median(arr),
                    np.percentile(arr, 75),
                    np.percentile(arr, 95),
                    np.percentile(arr, 99),
                    arr.max(),
                    arr.std(ddof=0),
                ]
                row += f"{len(arr):>10}" + "".join([f"{val:>11.4f}" for val in stats])
                return row
            def _write_cross_region_predicted_distance_table():
                # Focus on Cross-Region region pairs (any pair where regions differ)
                cross_region_mask = (stacked_regions[:, 0] != stacked_regions[:, 1])
                species_mask = cross_region_mask & (pool_ranks == 5)
                subseq_mask = cross_region_mask & (pool_ranks == 7)
                species_values = flat_pred_distances[species_mask]
                subseq_values = flat_pred_distances[subseq_mask]
                # Map each anchor sequence to its cross-region subsequence pair distance
                subseq_anchor_map = {}
                if np.any(subseq_mask):
                    subseq_indices = np.nonzero(subseq_mask)[0]
                    for idx in subseq_indices:
                        anchor_seq = int(pool_seq_i[idx])
                        subseq_anchor_map[anchor_seq] = subseq_anchor_map.get(anchor_seq, [])
                        subseq_anchor_map[anchor_seq].append(float(flat_pred_distances[idx]))
                subset_values = []
                missing_anchor_count = 0
                if np.any(species_mask):
                    species_indices = np.nonzero(species_mask)[0]
                    for idx in species_indices:
                        anchor_seq = int(pool_seq_i[idx])
                        anchor_vals = subseq_anchor_map.get(anchor_seq)
                        if anchor_vals:
                            subset_values.append(anchor_vals[0])
                        else:
                            missing_anchor_count += 1
                f.write("Cross-Region Predicted Distance Distributions (Pool):\n")
                f.write(f"  {'Category':<56}{'Count':>10} {'Min':>8} {'P01':>10} {'P05':>10} {'P25':>10} {'Median':>10} {'P75':>10} {'P95':>10} {'P99':>10} {'Max':>10} {'SD':>10}\n")
                f.write(_format_distribution_row("Cross-Region Species Pairs", species_values) + "\n")
                f.write(_format_distribution_row("Cross-Region Subsequence Pairs", subseq_values) + "\n")
                f.write(_format_distribution_row("Cross-Region Subsequence Pairs (Species Anchors Only)", subset_values) + "\n")
                if missing_anchor_count > 0:
                    total_species = int(np.sum(species_mask))
                    f.write(f"  Note: Missing Cross-Region subsequence anchors for {missing_anchor_count} of {total_species} species pairs.\n")
            _write_region_pair_metric_table("Predicted Distances Stats by Region Pair (Pool):", flat_pred_distances)
            _write_region_pair_metric_table("Absolute Error Stats by Region Pair (Pool):", abs_errors)
            _write_region_pair_metric_table("Relative Error Stats by Region Pair (Pool):", relative_errors)
            _write_region_pair_count_table("Pair Counts by Region Pair & Rank (Pool):")
            _write_cross_region_predicted_distance_table()
            f.write("Predicted Distances & True Distances (Pool):\n")
            f.write(f"  {'Rank':<12} {'Count (P)':<10} {'Min (T)':<8} {'Min (P)':<8} {'5% (T)':<8} {'5% (P)':<8} {'Mean (T)':<8} {'Mean (P)':<8} {'95% (T)':<8} {'95% (P)':<8} {'Max (T)':<8} {'Max (P)':<8} {'Mean AE':<8}\n")
            for r in unique_ranks:
                mask = (pool_ranks == r)
                if np.any(mask):
                    t_dists = flat_true_distances[mask]
                    p_dists = flat_pred_distances[mask]
                    cnt = len(t_dists)
                    t_min = t_dists.min()
                    t_5 = np.percentile(t_dists, 5)
                    t_mean = t_dists.mean()
                    t_95 = np.percentile(t_dists, 95)
                    t_max = t_dists.max()
                    p_min = p_dists.min()
                    p_5 = np.percentile(p_dists, 5)
                    p_mean = p_dists.mean()
                    p_95 = np.percentile(p_dists, 95)
                    p_max = p_dists.max()
                    mean_ae = np.mean(np.abs(t_dists - p_dists))
                    r_name = rank_map.get(r, str(r))
                    f.write(
                        f"  {r_name:<12} {cnt:<10} "
                        f"{_format_table_float_with_tiny_indicator(t_min)} "
                        f"{_format_table_float_with_tiny_indicator(p_min)} "
                        f"{_format_table_float_with_tiny_indicator(t_5)} "
                        f"{_format_table_float_with_tiny_indicator(p_5)} "
                        f"{_format_table_float_with_tiny_indicator(t_mean)} "
                        f"{_format_table_float_with_tiny_indicator(p_mean)} "
                        f"{_format_table_float_with_tiny_indicator(t_95)} "
                        f"{_format_table_float_with_tiny_indicator(p_95)} "
                        f"{_format_table_float_with_tiny_indicator(t_max)} "
                        f"{_format_table_float_with_tiny_indicator(p_max)} "
                        f"{_format_table_float_with_tiny_indicator(mean_ae)}\n"
                    )
            if len(flat_true_distances) > 0:
                t_dists = flat_true_distances
                p_dists = flat_pred_distances
                cnt = len(t_dists)
                t_min = t_dists.min()
                t_5 = np.percentile(t_dists, 5)
                t_mean = t_dists.mean()
                t_95 = np.percentile(t_dists, 95)
                t_max = t_dists.max()
                p_min = p_dists.min()
                p_5 = np.percentile(p_dists, 5)
                p_mean = p_dists.mean()
                p_95 = np.percentile(p_dists, 95)
                p_max = p_dists.max()
                mean_ae = np.mean(np.abs(t_dists - p_dists))
                f.write(
                    f"  {'All Ranks':<12} {cnt:<10} "
                    f"{_format_table_float_with_tiny_indicator(t_min)} "
                    f"{_format_table_float_with_tiny_indicator(p_min)} "
                    f"{_format_table_float_with_tiny_indicator(t_5)} "
                    f"{_format_table_float_with_tiny_indicator(p_5)} "
                    f"{_format_table_float_with_tiny_indicator(t_mean)} "
                    f"{_format_table_float_with_tiny_indicator(p_mean)} "
                    f"{_format_table_float_with_tiny_indicator(t_95)} "
                    f"{_format_table_float_with_tiny_indicator(p_95)} "
                    f"{_format_table_float_with_tiny_indicator(t_max)} "
                    f"{_format_table_float_with_tiny_indicator(p_max)} "
                    f"{_format_table_float_with_tiny_indicator(mean_ae)}\n"
                )
        # Write main CSV
        write_header = not os.path.exists(csv_path)
        with open(csv_path, "a") as f:
            if write_header:
                f.write("batch,n_pool,n_sampled,rel_err_min,rel_err_max,rel_err_mean,rel_err_std,rel_err_5pct,rel_err_95pct,pct_pos_err,pct_neg_err,true_dist_mean,pred_dist_mean,mean_abs_err,warmup_phase,total_deficit,bucket_base_proportions,bucket_target_proportions\n")
            n_pos = np.sum(signed_errors > 0)
            n_neg = np.sum(signed_errors < 0)
            pct_pos = (n_pos / n_pool * 100) if n_pool > 0 else 0.0
            pct_neg = (n_neg / n_pool * 100) if n_pool > 0 else 0.0
            mean_ae = np.mean(np.abs(flat_true_distances - flat_pred_distances))
            warmup = per_rank_stats.get('warmup_phase', 0.0) if per_rank_stats is not None else 0.0
            total_deficit = per_rank_stats['deficits'].sum() if per_rank_stats is not None and per_rank_stats.get('deficits') is not None else 0
            bucket_base = _bucket_vector_for_csv(per_rank_stats.get('bucket_proportions_base')) if per_rank_stats is not None else ""
            bucket_target = _bucket_vector_for_csv(per_rank_stats.get('bucket_proportions_target')) if per_rank_stats is not None else ""
            f.write(f"{batch_num},{n_pool},{n_sampled},{relative_errors.min():.6f},{relative_errors.max():.6f},{relative_errors.mean():.6f},{relative_errors.std():.6f},{np.percentile(relative_errors, 5):.6f},{np.percentile(relative_errors, 95):.6f},{pct_pos:.4f},{pct_neg:.4f},{flat_true_distances.mean():.6f},{flat_pred_distances.mean():.6f},{mean_ae:.6f},{warmup:.4f},{total_deficit},{bucket_base},{bucket_target}\n")
        # Write by-rank CSV (updated with per-rank mining data)
        write_header = not os.path.exists(csv_by_rank_path)
        with open(csv_by_rank_path, "a") as f:
            if write_header:
                f.write("batch,rank,rank_idx,ema_hardness_pre,ema_hardness_budget,ema_hardness_post,batch_hardness,budget,proportion,n_pool,n_sampled,deficit,rel_err_mean,rel_err_std,true_dist_mean,pred_dist_mean,mean_abs_err,bucket_thresholds,bucket_targets,bucket_sampled,bucket_residual_deficit,bucket_borrowed_from\n")
            
            # Use per_rank_stats if available for comprehensive per-rank data
            if per_rank_stats is not None:
                ema_pre = per_rank_stats.get('ema_hardness_pre')
                ema_budget = per_rank_stats.get('ema_hardness_budget')
                ema_post = per_rank_stats.get('ema_hardness_post')
                hardness_metrics = per_rank_stats.get('hardness_metrics')
                budgets = per_rank_stats.get('budgets')
                proportions = per_rank_stats.get('proportions')
                pool_counts = per_rank_stats.get('pool_counts')
                sampled_counts = per_rank_stats.get('sampled_counts')
                deficits = per_rank_stats.get('deficits')
                per_rank_thresholds = per_rank_stats.get('per_rank_thresholds', {})
                bucket_diag_all = per_rank_stats.get('bucket_diagnostics') or []
                
                for rank_idx in range(9):
                    internal_rank = pair_rank_idx_to_internal[rank_idx]
                    rank_name = pair_rank_idx_to_name.get(rank_idx, str(rank_idx))
                    
                    ema_pre_val = ema_pre[rank_idx] if ema_pre is not None else 0.0
                    ema_budget_val = ema_budget[rank_idx] if ema_budget is not None else 0.0
                    ema_post_val = ema_post[rank_idx] if ema_post is not None else 0.0
                    batch_hard_val = hardness_metrics[rank_idx] if hardness_metrics is not None else float('nan')
                    budget_val = budgets[rank_idx] if budgets is not None else 0
                    prop_val = proportions[rank_idx] if proportions is not None else 0.0
                    pool_val = pool_counts[rank_idx] if pool_counts is not None else 0
                    sampled_val = sampled_counts[rank_idx] if sampled_counts is not None else 0
                    deficit_val = deficits[rank_idx] if deficits is not None else 0
                    
                    # Compute error stats for this rank from the pool
                    rank_mask = (pool_ranks == internal_rank)
                    if np.any(rank_mask):
                        errs = relative_errors[rank_mask]
                        t_dists = flat_true_distances[rank_mask]
                        p_dists = flat_pred_distances[rank_mask]
                        rel_err_mean = errs.mean()
                        rel_err_std = errs.std()
                        true_dist_mean = t_dists.mean()
                        pred_dist_mean = p_dists.mean()
                        mean_ae = np.mean(np.abs(t_dists - p_dists))
                    else:
                        rel_err_mean = rel_err_std = true_dist_mean = pred_dist_mean = mean_ae = 0.0
                    
                    thresholds = per_rank_thresholds.get(internal_rank, [])
                    thresh_str = "|".join([f"{t:.6f}" for t in thresholds]) if thresholds else ""
                    
                    bucket_diag = bucket_diag_all[rank_idx] if rank_idx < len(bucket_diag_all) else None
                    bucket_targets_str, bucket_sampled_str, bucket_residual_str, bucket_borrowed_str = _bucket_diag_strings_for_csv(bucket_diag)
                    
                    f.write(f"{batch_num},{rank_name},{rank_idx},{ema_pre_val:.6f},{ema_budget_val:.6f},{ema_post_val:.6f},{batch_hard_val:.6f},{budget_val},{prop_val:.6f},{pool_val},{sampled_val},{deficit_val},{rel_err_mean:.6f},{rel_err_std:.6f},{true_dist_mean:.6f},{pred_dist_mean:.6f},{mean_ae:.6f},{thresh_str},{bucket_targets_str},{bucket_sampled_str},{bucket_residual_str},{bucket_borrowed_str}\n")
            else:
                # Fallback to old behavior if per_rank_stats not provided
                for r in unique_ranks:
                    mask = (pool_ranks == r)
                    errs = relative_errors[mask]
                    t_dists = flat_true_distances[mask]
                    p_dists = flat_pred_distances[mask]
                    n_p = np.sum(mask)
                    n_s = np.sum(sampled_ranks == r)
                    mean_ae = np.mean(np.abs(t_dists - p_dists))
                    rank_idx = r + 1  # internal rank to rank_idx
                    rank_name = rank_map.get(r, str(r))
                    f.write(f"{batch_num},{rank_name},{rank_idx},0.0,0.0,0.0,0.0,0,0.0,{n_p},{n_s},0,{errs.mean():.6f},{errs.std():.6f},{t_dists.mean():.6f},{p_dists.mean():.6f},{mean_ae:.6f},,,,,\n")
        # Write by-bucket CSV
        write_header = not os.path.exists(csv_by_bucket_path)
        with open(csv_by_bucket_path, "a") as f:
            if write_header:
                f.write("batch,bucket,pct_low,pct_high,n_in_bucket,n_sampled,rel_err_min,rel_err_max,pct_pool,pct_sampled\n")
            for b in range(n_buckets):
                lower_pct = 0.0 if b == 0 else percentile_thresholds[b-1] * 100
                upper_pct = 100.0 if b == len(percentile_thresholds) else percentile_thresholds[b] * 100
                mask = (bucket_assignments == b)
                if np.any(mask):
                    errs = relative_errors[mask]
                    n_in_bucket = len(errs)
                    n_pool_total = len(relative_errors)
                    n_sampled_in_bucket = np.sum(bucket_assignments[sampled_local_indices] == b)
                    n_sampled_total = len(sampled_local_indices)
                    pct_pool = (n_in_bucket / n_pool_total * 100) if n_pool_total > 0 else 0.0
                    pct_sampled = (n_sampled_in_bucket / n_sampled_total * 100) if n_sampled_total > 0 else 0.0
                    f.write(f"{batch_num},{b},{lower_pct:.2f},{upper_pct:.2f},{n_in_bucket},{n_sampled_in_bucket},{errs.min():.6f},{errs.max():.6f},{pct_pool:.4f},{pct_sampled:.4f}\n")
                else:
                    f.write(f"{batch_num},{b},{lower_pct:.2f},{upper_pct:.2f},0,0,0.0,0.0,0.0,0.0\n")

        # Populate pair distances dataframe if provided
        # Stores box plot statistics (min, q1, median, q3, max) for true and predicted distances by rank
        if pair_distances_df is not None:
            distances_by_rank = {}
            for r in unique_ranks:
                mask = (pool_ranks == r)
                if np.sum(mask) > 0:
                    t_dists = flat_true_distances[mask]
                    p_dists = flat_pred_distances[mask]
                    errs = relative_errors[mask]
                    distances_by_rank[r] = {
                        'true_min': float(np.min(t_dists)),
                        'true_q1': float(np.percentile(t_dists, 25)),
                        'true_median': float(np.median(t_dists)),
                        'true_q3': float(np.percentile(t_dists, 75)),
                        'true_max': float(np.max(t_dists)),
                        'pred_min': float(np.min(p_dists)),
                        'pred_q1': float(np.percentile(p_dists, 25)),
                        'pred_median': float(np.median(p_dists)),
                        'pred_q3': float(np.percentile(p_dists, 75)),
                        'pred_max': float(np.max(p_dists)),
                        'mean_rel_error': float(np.mean(errs)),
                    }
            
            # Compute overall statistics
            overall_distances = None
            if len(flat_true_distances) > 0:
                overall_distances = {
                    'true_min': float(np.min(flat_true_distances)),
                    'true_q1': float(np.percentile(flat_true_distances, 25)),
                    'true_median': float(np.median(flat_true_distances)),
                    'true_q3': float(np.percentile(flat_true_distances, 75)),
                    'true_max': float(np.max(flat_true_distances)),
                    'pred_min': float(np.min(flat_pred_distances)),
                    'pred_q1': float(np.percentile(flat_pred_distances, 25)),
                    'pred_median': float(np.median(flat_pred_distances)),
                    'pred_q3': float(np.percentile(flat_pred_distances, 75)),
                    'pred_max': float(np.max(flat_pred_distances)),
                    'mean_rel_error': float(np.mean(relative_errors)),
                }
            
            # Add row to the dataframe in-place
            add_pair_distances_row_inplace(pair_distances_df, batch_num, distances_by_rank, overall_distances)
        
        # Populate pair error metrics dataframe if provided
        # The metric is: mean_error * PAIR_EMA_MEAN_WEIGHT + (p25_error + p75_error) * PAIR_EMA_QUARTILES_WEIGHT
        # This is stored in per_rank_stats['hardness_metrics'], indexed by rank_idx (0-8)
        if pair_error_metrics_df is not None and per_rank_stats is not None:
            hardness_metrics = per_rank_stats.get('hardness_metrics')
            if hardness_metrics is not None:
                # Convert from rank_idx (0-8) to internal rank (-1 to 7) for the dataframe
                metrics_by_rank = {}
                for rank_idx in range(9):
                    internal_rank = rank_idx - 1  # 0->-1, 1->0, ..., 8->7
                    metric_val = hardness_metrics[rank_idx]
                    if not np.isnan(metric_val):
                        metrics_by_rank[internal_rank] = float(metric_val)
                
                # Add row to the dataframe in-place
                add_pair_error_metrics_row_inplace(pair_error_metrics_df, batch_num, metrics_by_rank)
            
    except Exception as e:
        print(f"Warning: Failed to write pair mining log: {e}")



# Triplet Mining --------------------------

def _build_triplet_mining_tables(normalised_violations, bucket_assignments, sampled_local_indices, 
                                 percentile_thresholds, triplet_ranks_arr, true_ap, true_an, 
                                 pred_ap, pred_an, margins,
                                 pool_normalised_violations, pool_triplet_ranks_arr, pool_true_ap,
                                 pool_true_an, pool_pred_ap, pool_pred_an, pool_margins):
    """Create reusable text tables + stats for triplet mining verbose/log output."""
    n_buckets = len(percentile_thresholds) + 1
    rank_map = {0: 'Domain', 1: 'Phylum', 2: 'Class', 3: 'Order', 4: 'Family', 5: 'Genus'}

    pool_unique_ranks = np.unique(pool_triplet_ranks_arr)
    pool_unique_ranks.sort()
    filtered_unique_ranks = np.unique(triplet_ranks_arr)
    filtered_unique_ranks.sort()
    sampled_ranks = triplet_ranks_arr[sampled_local_indices]
    sampled_buckets = bucket_assignments[sampled_local_indices] if len(sampled_local_indices) > 0 else np.empty(0, dtype=np.int32)

    n_pool = len(pool_triplet_ranks_arr)
    n_filtered = len(triplet_ranks_arr)
    n_sampled = len(sampled_ranks)

    pool_hard_violations = pool_pred_ap >= pool_pred_an
    pool_correctly_ordered = pool_pred_ap < pool_pred_an
    pool_margin_satisfied = pool_pred_ap + pool_margins < pool_pred_an
    pool_ordered_no_margin = pool_correctly_ordered & ~pool_margin_satisfied

    hard_violations = pred_ap >= pred_an
    correctly_ordered = pred_ap < pred_an
    margin_satisfied = pred_ap + margins < pred_an
    ordered_no_margin = correctly_ordered & ~margin_satisfied

    lines = []

    # 1. Normalised Violation Stats by Rank (Pool)
    lines.append("Normalised Violation Stats by Rank (Pool):")
    lines.append(f"  {'Rank':<12} {'N':<6} {'Min':<10} {'Mean':<10} {'Max':<10} {'Std':<10} {'5%':<10} {'95%':<10} {'% AP>=AN':<10} {'% AP<AN<AP+M':<10} {'% AP+M<AN':<10}")
    for r in pool_unique_ranks:
        mask = (pool_triplet_ranks_arr == r)
        viols = pool_normalised_violations[mask]
        r_name = rank_map.get(r, str(r))
        n_total = len(viols)
        pct_hard = pool_hard_violations[mask].mean() * 100
        pct_ordered_no_margin = pool_ordered_no_margin[mask].mean() * 100
        pct_margin = pool_margin_satisfied[mask].mean() * 100
        lines.append(f"  {r_name:<12} {n_total:<6} {viols.min():>10.4f} {viols.mean():>10.4f} {viols.max():>10.4f} {viols.std():>10.4f} {np.percentile(viols, 5):>10.4f} {np.percentile(viols, 95):>10.4f} {pct_hard:>9.1f}% {pct_ordered_no_margin:>9.1f}% {pct_margin:>9.1f}%")
    if len(pool_normalised_violations) > 0:
        n_total_all = len(pool_normalised_violations)
        pct_hard_all = pool_hard_violations.mean() * 100
        pct_ordered_no_margin_all = pool_ordered_no_margin.mean() * 100
        pct_margin_all = pool_margin_satisfied.mean() * 100
        lines.append(f"  {'All Ranks':<12} {n_total_all:<6} {pool_normalised_violations.min():>10.4f} {pool_normalised_violations.mean():>10.4f} {pool_normalised_violations.max():>10.4f} {pool_normalised_violations.std():>10.4f} {np.percentile(pool_normalised_violations, 5):>10.4f} {np.percentile(pool_normalised_violations, 95):>10.4f} {pct_hard_all:>9.1f}% {pct_ordered_no_margin_all:>9.1f}% {pct_margin_all:>9.1f}%")

    # 2. True & Predicted Distances (Pool)
    lines.append("True & Predicted Distances (Pool):")
    lines.append(f"  {'Rank':<12} {'Count':<8} {'True AP':<8} {'Pred AP':<8} {'True AN':<8} {'Pred AN':<8} {'AN-AP True':<10} {'AN-AP Pred':<10} {'Margin':<8}")
    for r in pool_unique_ranks:
        mask = (pool_triplet_ranks_arr == r)
        if np.any(mask):
            r_name = rank_map.get(r, str(r))
            cnt = np.sum(mask)
            t_ap = pool_true_ap[mask].mean()
            p_ap = pool_pred_ap[mask].mean()
            t_an = pool_true_an[mask].mean()
            p_an = pool_pred_an[mask].mean()
            an_ap_t = (pool_true_an[mask] - pool_true_ap[mask]).mean()
            an_ap_p = (pool_pred_an[mask] - pool_pred_ap[mask]).mean()
            margin = pool_margins[mask].mean()
            lines.append(f"  {r_name:<12} {cnt:<8} {t_ap:<8.4f} {p_ap:<8.4f} {t_an:<8.4f} {p_an:<8.4f} {an_ap_t:<10.4f} {an_ap_p:<10.4f} {margin:<8.4f}")
    if len(pool_true_ap) > 0:
        cnt = len(pool_true_ap)
        t_ap = pool_true_ap.mean()
        p_ap = pool_pred_ap.mean()
        t_an = pool_true_an.mean()
        p_an = pool_pred_an.mean()
        an_ap_t = (pool_true_an - pool_true_ap).mean()
        an_ap_p = (pool_pred_an - pool_pred_ap).mean()
        margin = pool_margins.mean()
        lines.append(f"  {'All Ranks':<12} {cnt:<8} {t_ap:<8.4f} {p_ap:<8.4f} {t_an:<8.4f} {p_an:<8.4f} {an_ap_t:<10.4f} {an_ap_p:<10.4f} {margin:<8.4f}")

    # 3. Bucket Normalised Violation Ranges (Filtered)
    lines.append("Bucket Normalised Violation Ranges:")
    for b in range(n_buckets):
        lower_pct = 0.0 if b == 0 else percentile_thresholds[b-1] * 100
        upper_pct = 100.0 if b == len(percentile_thresholds) else percentile_thresholds[b] * 100
        bucket_label = f"Bucket {b} ({lower_pct:.0f}-{upper_pct:.0f}%):"
        mask = (bucket_assignments == b)
        if np.any(mask):
            viols = normalised_violations[mask]
            n_in_bucket = len(viols)
            n_filtered_total = len(normalised_violations)
            n_sampled_in_bucket = np.sum(bucket_assignments[sampled_local_indices] == b)
            n_sampled_total = len(sampled_local_indices)
            pct_filtered = (n_in_bucket / n_filtered_total * 100) if n_filtered_total > 0 else 0.0
            pct_sampled = (n_sampled_in_bucket / n_sampled_total * 100) if n_sampled_total > 0 else 0.0
            lines.append(f"  {bucket_label:<22} Range=[{viols.min():.4f}, {viols.max():.4f}] N={n_in_bucket:<6} Sampled={n_sampled_in_bucket:<6} ({pct_filtered:3.0f}% -> {pct_sampled:3.0f}%)")
        else:
            lines.append(f"  {bucket_label:<22} N=0")

    # 4. Normalised Violation Stats by Rank (Filtered)
    lines.append("Normalised Violation Stats by Rank (Filtered):")
    lines.append(f"  {'Rank':<12} {'N':<6} {'Min':<10} {'Mean':<10} {'Max':<10} {'Std':<10} {'5%':<10} {'95%':<10} {'% AP>=AN':<10} {'% AP<AN<AP+M':<10} {'% AP+M<AN':<10}")
    for r in filtered_unique_ranks:
        mask = (triplet_ranks_arr == r)
        viols = normalised_violations[mask]
        r_name = rank_map.get(r, str(r))
        n_total = len(viols)
        pct_hard = hard_violations[mask].mean() * 100
        pct_ordered_no_margin = ordered_no_margin[mask].mean() * 100
        pct_margin = margin_satisfied[mask].mean() * 100
        lines.append(f"  {r_name:<12} {n_total:<6} {viols.min():>10.4f} {viols.mean():>10.4f} {viols.max():>10.4f} {viols.std():>10.4f} {np.percentile(viols, 5):>10.4f} {np.percentile(viols, 95):>10.4f} {pct_hard:>9.1f}% {pct_ordered_no_margin:>9.1f}% {pct_margin:>9.1f}%")
    if len(normalised_violations) > 0:
        n_total_all = len(normalised_violations)
        pct_hard_all = hard_violations.mean() * 100
        pct_ordered_no_margin_all = ordered_no_margin.mean() * 100
        pct_margin_all = margin_satisfied.mean() * 100
        lines.append(f"  {'All Ranks':<12} {n_total_all:<6} {normalised_violations.min():>10.4f} {normalised_violations.mean():>10.4f} {normalised_violations.max():>10.4f} {normalised_violations.std():>10.4f} {np.percentile(normalised_violations, 5):>10.4f} {np.percentile(normalised_violations, 95):>10.4f} {pct_hard_all:>9.1f}% {pct_ordered_no_margin_all:>9.1f}% {pct_margin_all:>9.1f}%")

    # 5. Rank Proportions (Pool vs Filtered vs Sampled)
    lines.append("Rank Proportions (Pool vs Filtered vs Sampled):")
    lines.append(f"  {'Rank':<12} {'Pool %':<8} {'Filt %':<8} {'Samp %':<8} {'Count (P)':<10} {'Count (F)':<10} {'Count (S)':<10}")
    for r in pool_unique_ranks:
        n_p = np.sum(pool_triplet_ranks_arr == r)
        n_f = np.sum(triplet_ranks_arr == r)
        n_s = np.sum(sampled_ranks == r)
        p_pool = n_p / n_pool * 100 if n_pool > 0 else 0
        p_filtered = n_f / n_filtered * 100 if n_filtered > 0 else 0
        p_sampled = n_s / n_sampled * 100 if n_sampled > 0 else 0
        r_name = rank_map.get(r, str(r))
        lines.append(f"  {r_name:<12} {p_pool:6.2f}%   {p_filtered:6.2f}%   {p_sampled:6.2f}%   {n_p:<10} {n_f:<10} {n_s:<10}")

    # 6. Bucket Composition per Rank (Filtered Pool vs Sampled)
    lines.append("Bucket Composition per Rank (Filtered Pool vs Sampled %):")
    header_labels = [f"B{b} P%/S%" for b in range(n_buckets)]
    header_cols = " ".join([f"{label:<13}" for label in header_labels])
    lines.append(f"  {'Rank':<12} {header_cols}")
    for r in filtered_unique_ranks:
        r_name = rank_map.get(r, str(r))
        rank_mask = (triplet_ranks_arr == r)
        n_rank_pool = np.sum(rank_mask)
        rank_bucket_counts = (
            np.bincount(bucket_assignments[rank_mask], minlength=n_buckets)
            if n_rank_pool > 0 else np.zeros(n_buckets, dtype=np.int64)
        )
        sampled_rank_mask = (sampled_ranks == r)
        n_rank_sampled = np.sum(sampled_rank_mask)
        sampled_rank_counts = (
            np.bincount(sampled_buckets[sampled_rank_mask], minlength=n_buckets)
            if n_rank_sampled > 0 else np.zeros(n_buckets, dtype=np.int64)
        )
        row_str = f"  {r_name:<12} "
        for b in range(n_buckets):
            pool_pct = (rank_bucket_counts[b] / n_rank_pool * 100) if n_rank_pool > 0 else 0.0
            sampled_pct = (sampled_rank_counts[b] / n_rank_sampled * 100) if n_rank_sampled > 0 else 0.0
            row_str += f"{pool_pct:5.1f}/{sampled_pct:5.1f} "
        lines.append(row_str.rstrip())

    # 7. True & Predicted Distances (Filtered)
    lines.append("True & Predicted Distances (Filtered):")
    lines.append(f"  {'Rank':<12} {'Count':<8} {'True AP':<8} {'Pred AP':<8} {'True AN':<8} {'Pred AN':<8} {'AN-AP True':<10} {'AN-AP Pred':<10} {'Margin':<8}")
    for r in filtered_unique_ranks:
        mask = (triplet_ranks_arr == r)
        if np.any(mask):
            r_name = rank_map.get(r, str(r))
            cnt = np.sum(mask)
            t_ap = true_ap[mask].mean()
            p_ap = pred_ap[mask].mean()
            t_an = true_an[mask].mean()
            p_an = pred_an[mask].mean()
            an_ap_t = (true_an[mask] - true_ap[mask]).mean()
            an_ap_p = (pred_an[mask] - pred_ap[mask]).mean()
            margin = margins[mask].mean()
            lines.append(f"  {r_name:<12} {cnt:<8} {t_ap:<8.4f} {p_ap:<8.4f} {t_an:<8.4f} {p_an:<8.4f} {an_ap_t:<10.4f} {an_ap_p:<10.4f} {margin:<8.4f}")
    if len(true_ap) > 0:
        cnt = len(true_ap)
        t_ap = true_ap.mean()
        p_ap = pred_ap.mean()
        t_an = true_an.mean()
        p_an = pred_an.mean()
        an_ap_t = (true_an - true_ap).mean()
        an_ap_p = (pred_an - pred_ap).mean()
        margin = margins.mean()
        lines.append(f"  {'All Ranks':<12} {cnt:<8} {t_ap:<8.4f} {p_ap:<8.4f} {t_an:<8.4f} {p_an:<8.4f} {an_ap_t:<10.4f} {an_ap_p:<10.4f} {margin:<8.4f}")

    stats = {
        "pool_unique_ranks": pool_unique_ranks,
        "filtered_unique_ranks": filtered_unique_ranks,
        "sampled_ranks": sampled_ranks,
        "n_pool": n_pool,
        "n_filtered": n_filtered,
        "n_sampled": n_sampled,
        "pool_hard_violations": pool_hard_violations,
        "pool_ordered_no_margin": pool_ordered_no_margin,
        "pool_margin_satisfied": pool_margin_satisfied,
        "hard_violations": hard_violations,
        "ordered_no_margin": ordered_no_margin,
        "margin_satisfied": margin_satisfied,
    }
    return lines, stats


def print_triplet_mining_stats(normalised_violations, bucket_assignments, sampled_local_indices, 
                               percentile_thresholds, triplet_ranks_arr, true_ap, true_an, 
                               pred_ap, pred_an, margins,
                               pool_normalised_violations, pool_triplet_ranks_arr, pool_true_ap,
                               pool_true_an, pool_pred_ap, pool_pred_an, pool_margins,
                               per_rank_stats=None):
    """Helper to print useful statistics during triplet mining."""
    try:
        triplet_rank_to_name = {0: 'Domain', 1: 'Phylum', 2: 'Class', 3: 'Order', 4: 'Family', 5: 'Genus'}
        
        print("\nTriplet Mining Verbose  -------------------------------------")
        
        # Print per-rank mining summary (new section for per-rank mining)
        if per_rank_stats is not None:
            warmup = per_rank_stats.get('warmup_phase', 0.0)
            print(f"\nPer-Rank Mining Summary (warmup={warmup:.2%}):")
            print(f"  {'Rank':<8} | {'EMA Pre':<8} | {'EMA Warm':<8} | {'EMA Post':<8} | {'BatchHard':<10} | {'Budget':<8} | {'Prop %':<8} | {'PreFilt':<8} | {'Hard':<8} | {'TopUp':<8} | {'Min':<8} | {'Pool':<8} | {'Sampled':<8} | {'Deficit':<8}")
            print(f"  {'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*10}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")
            
            ema_pre = per_rank_stats.get('ema_hardness_pre')
            ema_budget = per_rank_stats.get('ema_hardness_budget')
            ema_post = per_rank_stats.get('ema_hardness_post')
            hardness_metrics = per_rank_stats.get('hardness_metrics')
            budgets = per_rank_stats.get('budgets')
            proportions = per_rank_stats.get('proportions')
            pool_counts = per_rank_stats.get('pool_counts')
            pool_counts_pre_filter = per_rank_stats.get('pool_counts_pre_filter')
            sampled_counts = per_rank_stats.get('sampled_counts')
            deficits = per_rank_stats.get('deficits')
            hard_counts = per_rank_stats.get('filter_hard_counts')
            topup_counts = per_rank_stats.get('filter_topup_counts')
            min_counts = per_rank_stats.get('filter_minimums')
            
            for rank_idx in range(6):
                rank_name = triplet_rank_to_name.get(rank_idx, str(rank_idx))
                ema_pre_val = f"{ema_pre[rank_idx]:.4f}" if ema_pre is not None else "-"
                ema_budget_val = f"{ema_budget[rank_idx]:.4f}" if ema_budget is not None else "-"
                ema_post_val = f"{ema_post[rank_idx]:.4f}" if ema_post is not None else "-"
                batch_hard_val = "-"
                if hardness_metrics is not None and not np.isnan(hardness_metrics[rank_idx]):
                    batch_hard_val = f"{hardness_metrics[rank_idx]:.4f}"
                budget_val = f"{budgets[rank_idx]}" if budgets is not None else "-"
                prop_val = f"{proportions[rank_idx]*100:.1f}%" if proportions is not None else "-"
                pre_filt_val = f"{pool_counts_pre_filter[rank_idx]}" if pool_counts_pre_filter is not None else "-"
                pool_val = f"{pool_counts[rank_idx]}" if pool_counts is not None else "-"
                sampled_val = f"{sampled_counts[rank_idx]}" if sampled_counts is not None else "-"
                deficit_val = f"{deficits[rank_idx]}" if deficits is not None else "-"
                hard_val = f"{hard_counts[rank_idx]}" if hard_counts is not None else "-"
                topup_val = f"{topup_counts[rank_idx]}" if topup_counts is not None else "-"
                min_val = f"{min_counts[rank_idx]}" if min_counts is not None else "-"
                print(f"  {rank_name:<8} | {ema_pre_val:<8} | {ema_budget_val:<8} | {ema_post_val:<8} | {batch_hard_val:<10} | {budget_val:<8} | {prop_val:<8} | {pre_filt_val:<8} | {hard_val:<8} | {topup_val:<8} | {min_val:<8} | {pool_val:<8} | {sampled_val:<8} | {deficit_val:<8}")
            
            # Print totals
            total_budget = budgets.sum() if budgets is not None else 0
            total_pre_filt = pool_counts_pre_filter.sum() if pool_counts_pre_filter is not None else 0
            total_pool = pool_counts.sum() if pool_counts is not None else 0
            total_sampled = sampled_counts.sum() if sampled_counts is not None else 0
            total_deficit = deficits.sum() if deficits is not None else 0
            total_hard = hard_counts.sum() if hard_counts is not None else 0
            total_topup = topup_counts.sum() if topup_counts is not None else 0
            total_min = min_counts.sum() if min_counts is not None else 0
            print(f"  {'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*10}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")
            print(f"  {'TOTAL':<8} | {'-':<8} | {'-':<8} | {'-':<8} | {'-':<10} | {total_budget:<8} | {'100%':<8} | {total_pre_filt:<8} | {total_hard:<8} | {total_topup:<8} | {total_min:<8} | {total_pool:<8} | {total_sampled:<8} | {total_deficit:<8}")
            
            bucket_base = per_rank_stats.get('bucket_proportions_base')
            bucket_target = per_rank_stats.get('bucket_proportions_target')
            if bucket_base is not None and bucket_target is not None:
                print("\nBucket Target Proportions:")
                base_str = _format_bucket_vector(bucket_base)
                target_str = _format_bucket_vector(bucket_target)
                if base_str:
                    print(f"  Base:   {base_str}")
                if target_str:
                    print(f"  Target: {target_str}")
            
            # Print per-rank bucket thresholds
            per_rank_thresholds = per_rank_stats.get('per_rank_thresholds', {})
            if per_rank_thresholds:
                print("\nPer-Rank Bucket Thresholds (Normalised Violation):")
                for rank_idx in range(6):
                    rank_name = triplet_rank_to_name.get(rank_idx, str(rank_idx))
                    thresholds = per_rank_thresholds.get(rank_idx, [])
                    if thresholds:
                        thresh_str = ", ".join([f"{t:.4f}" for t in thresholds])
                        print(f"  {rank_name:<8}: [{thresh_str}]")
                    elif budgets is not None and budgets[rank_idx] > 0:
                        print(f"  {rank_name:<8}: (uniform/warmup)")
            
            bucket_diag_all = per_rank_stats.get('bucket_diagnostics')
            if bucket_diag_all and any(entry is not None for entry in bucket_diag_all):
                print("\nPer-Rank Bucket Fulfillment:")
                for rank_idx in range(6):
                    diag_entry = bucket_diag_all[rank_idx] if bucket_diag_all is not None else None
                    if not diag_entry:
                        continue
                    diag_str = _format_bucket_diag_entries(diag_entry)
                    if not diag_str:
                        continue
                    rank_name = triplet_rank_to_name.get(rank_idx, str(rank_idx))
                    print(f"  {rank_name:<8}: {diag_str}")
            
            shortages = per_rank_stats.get('representative_shortages') if per_rank_stats else None
            if shortages:
                print("\nRepresentative Set Shortages:")
                for shortage in shortages:
                    deficit = shortage['requested'] - shortage['available']
                    print(f"  - {shortage['rank_name']}: requested {shortage['requested']} but found "
                          f"{shortage['available']} (short by {deficit}).")
        
        # Print detailed tables from _build_triplet_mining_tables
        table_lines, _ = _build_triplet_mining_tables(
            normalised_violations, bucket_assignments, sampled_local_indices,
            percentile_thresholds, triplet_ranks_arr, true_ap, true_an,
            pred_ap, pred_an, margins,
            pool_normalised_violations, pool_triplet_ranks_arr, pool_true_ap,
            pool_true_an, pool_pred_ap, pool_pred_an, pool_margins)
        for line in table_lines:
            print(line)
    except Exception as e:
        print(f"Warning: Failed to print triplet mining stats: {e}")


def write_triplet_mining_log(batch_num, logs_dir, normalised_violations, bucket_assignments, 
                              sampled_local_indices, percentile_thresholds, triplet_ranks_arr,
                              true_ap, true_an, pred_ap, pred_an, margins,
                              pool_normalised_violations, pool_triplet_ranks_arr, pool_true_ap,
                              pool_true_an, pool_pred_ap, pool_pred_an, pool_margins,
                              triplet_satisfaction_df=None, per_rank_stats=None, triplet_error_metrics_df=None):
    """
    Write triplet mining statistics to log and CSV files.
    
    Log files:
        triplet_mining.log: Text dump of printed output with batch headers
    
    CSV files:
        triplet_mining.csv: Overall stats per batch (both pool and filtered)
            Columns: batch, n_pool, n_filtered, n_sampled, pool_norm_viol_min, pool_norm_viol_max,
                     pool_norm_viol_mean, pool_norm_viol_std, pool_pct_ap_gte_an, pool_pct_ap_lt_an,
                     pool_pct_margin_satisfied, filtered_norm_viol_min, filtered_norm_viol_max,
                     filtered_norm_viol_mean, filtered_norm_viol_std, filtered_pct_ap_gte_an,
                     filtered_pct_ap_lt_an, filtered_pct_margin_satisfied, pool_true_ap_mean,
                     pool_pred_ap_mean, pool_true_an_mean, pool_pred_an_mean, filtered_true_ap_mean,
                     filtered_pred_ap_mean, filtered_true_an_mean, filtered_pred_an_mean, margin_mean,
                     warmup_phase, total_deficit, bucket_base_proportions, bucket_target_proportions
        
        triplet_mining_by_rank.csv: Stats per rank per batch (updated with per-rank mining data)
            Columns: batch, rank, ema_hardness_pre, ema_hardness_budget, ema_hardness_post,
                     batch_hardness, budget, proportion, n_pool_pre_filter, n_pool, n_sampled, deficit,
                     n_hard_after_filter, n_topup, min_after_filter, pool_norm_viol_mean,
                     filtered_norm_viol_mean, pool_pct_ap_gte_an, filtered_pct_ap_gte_an, margin_mean,
                     bucket_thresholds, bucket_targets, bucket_sampled, bucket_residual_deficit,
                     bucket_borrowed_from
        
        triplet_mining_by_bucket.csv: Stats per bucket per batch (filtered pool)
            Columns: batch, bucket, pct_low, pct_high, n_in_bucket, n_sampled, rel_err_min, rel_err_max,
                     pct_filtered, pct_sampled
    
    Args:
        normalised_violations: Filtered pool normalised violations
        bucket_assignments: Bucket assignments for filtered pool
        sampled_local_indices: Indices of sampled triplets within filtered pool
        percentile_thresholds: Percentile thresholds for bucket boundaries
        triplet_ranks_arr: Filtered pool triplet ranks
        true_ap, true_an, pred_ap, pred_an, margins: Filtered pool distances/margins
        pool_*: Unfiltered pool versions of the above arrays
        triplet_satisfaction_df: Optional pd.DataFrame for tracking triplet satisfaction over time.
                                 If provided, will be populated with satisfaction metrics.
        per_rank_stats: Optional dict containing per-rank mining statistics from mine_triplets()
        triplet_error_metrics_df: Optional pd.DataFrame for tracking per-rank error metrics over time.
                                  If provided, will be populated with the metric fed to the EMA.
    """
    try:
        log_path = os.path.join(logs_dir, "triplet_mining.log")
        csv_path = os.path.join(logs_dir, "triplet_mining.csv")
        csv_by_rank_path = os.path.join(logs_dir, "triplet_mining_by_rank.csv")
        csv_by_bucket_path = os.path.join(logs_dir, "triplet_mining_by_bucket.csv")

        table_lines, table_stats = _build_triplet_mining_tables(
            normalised_violations, bucket_assignments, sampled_local_indices,
            percentile_thresholds, triplet_ranks_arr, true_ap, true_an,
            pred_ap, pred_an, margins,
            pool_normalised_violations, pool_triplet_ranks_arr, pool_true_ap,
            pool_true_an, pool_pred_ap, pool_pred_an, pool_margins)
        
        triplet_rank_to_name = {0: 'Domain', 1: 'Phylum', 2: 'Class', 3: 'Order', 4: 'Family', 5: 'Genus'}
        
        with open(log_path, "a") as f:
            f.write(f"\n---\nBatch {batch_num}\n")
            f.write(f"\nTriplet Mining Logging -------------------------------------\n")
            
            # Write per-rank mining summary (new section for per-rank mining)
            if per_rank_stats is not None:
                warmup = per_rank_stats.get('warmup_phase', 0.0)
                f.write(f"\nPer-Rank Mining Summary (warmup={warmup:.2%}):\n")
                f.write(f"  {'Rank':<8} | {'EMA Pre':<8} | {'EMA Warm':<8} | {'EMA Post':<8} | {'BatchHard':<10} | {'Budget':<8} | {'Prop %':<8} | {'PreFilt':<8} | {'Hard':<8} | {'TopUp':<8} | {'Min':<8} | {'Pool':<8} | {'Sampled':<8} | {'Deficit':<8}\n")
                f.write(f"  {'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*10}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}\n")
                
                ema_pre = per_rank_stats.get('ema_hardness_pre')
                ema_budget = per_rank_stats.get('ema_hardness_budget')
                ema_post = per_rank_stats.get('ema_hardness_post')
                hardness_metrics = per_rank_stats.get('hardness_metrics')
                budgets = per_rank_stats.get('budgets')
                proportions = per_rank_stats.get('proportions')
                pool_counts = per_rank_stats.get('pool_counts')
                pool_counts_pre_filter = per_rank_stats.get('pool_counts_pre_filter')
                sampled_counts = per_rank_stats.get('sampled_counts')
                deficits = per_rank_stats.get('deficits')
                hard_counts = per_rank_stats.get('filter_hard_counts')
                topup_counts = per_rank_stats.get('filter_topup_counts')
                min_counts = per_rank_stats.get('filter_minimums')
                
                for rank_idx in range(6):
                    rank_name = triplet_rank_to_name.get(rank_idx, str(rank_idx))
                    ema_pre_val = f"{ema_pre[rank_idx]:.4f}" if ema_pre is not None else "-"
                    ema_budget_val = f"{ema_budget[rank_idx]:.4f}" if ema_budget is not None else "-"
                    ema_post_val = f"{ema_post[rank_idx]:.4f}" if ema_post is not None else "-"
                    batch_hard_val = "-"
                    if hardness_metrics is not None and not np.isnan(hardness_metrics[rank_idx]):
                        batch_hard_val = f"{hardness_metrics[rank_idx]:.4f}"
                    budget_val = f"{budgets[rank_idx]}" if budgets is not None else "-"
                    prop_val = f"{proportions[rank_idx]*100:.1f}%" if proportions is not None else "-"
                    pre_filt_val = f"{pool_counts_pre_filter[rank_idx]}" if pool_counts_pre_filter is not None else "-"
                    pool_val = f"{pool_counts[rank_idx]}" if pool_counts is not None else "-"
                    sampled_val = f"{sampled_counts[rank_idx]}" if sampled_counts is not None else "-"
                    deficit_val = f"{deficits[rank_idx]}" if deficits is not None else "-"
                    hard_val = f"{hard_counts[rank_idx]}" if hard_counts is not None else "-"
                    topup_val = f"{topup_counts[rank_idx]}" if topup_counts is not None else "-"
                    min_val = f"{min_counts[rank_idx]}" if min_counts is not None else "-"
                    f.write(f"  {rank_name:<8} | {ema_pre_val:<8} | {ema_budget_val:<8} | {ema_post_val:<8} | {batch_hard_val:<10} | {budget_val:<8} | {prop_val:<8} | {pre_filt_val:<8} | {hard_val:<8} | {topup_val:<8} | {min_val:<8} | {pool_val:<8} | {sampled_val:<8} | {deficit_val:<8}\n")
                
                total_budget = budgets.sum() if budgets is not None else 0
                total_pre_filt = pool_counts_pre_filter.sum() if pool_counts_pre_filter is not None else 0
                total_pool = pool_counts.sum() if pool_counts is not None else 0
                total_sampled = sampled_counts.sum() if sampled_counts is not None else 0
                total_deficit = deficits.sum() if deficits is not None else 0
                total_hard = hard_counts.sum() if hard_counts is not None else 0
                total_topup = topup_counts.sum() if topup_counts is not None else 0
                total_min = min_counts.sum() if min_counts is not None else 0
                f.write(f"  {'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*10}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}\n")
                f.write(f"  {'TOTAL':<8} | {'-':<8} | {'-':<8} | {'-':<8} | {'-':<10} | {total_budget:<8} | {'100%':<8} | {total_pre_filt:<8} | {total_hard:<8} | {total_topup:<8} | {total_min:<8} | {total_pool:<8} | {total_sampled:<8} | {total_deficit:<8}\n")
                
                bucket_base = per_rank_stats.get('bucket_proportions_base')
                bucket_target = per_rank_stats.get('bucket_proportions_target')
                if bucket_base is not None and bucket_target is not None:
                    base_str = _format_bucket_vector(bucket_base)
                    target_str = _format_bucket_vector(bucket_target)
                    if base_str or target_str:
                        f.write("\nBucket Target Proportions:\n")
                        if base_str:
                            f.write(f"  Base:   {base_str}\n")
                        if target_str:
                            f.write(f"  Target: {target_str}\n")
                
                per_rank_thresholds = per_rank_stats.get('per_rank_thresholds', {})
                if per_rank_thresholds:
                    f.write("\nPer-Rank Bucket Thresholds (Normalised Violation):\n")
                    for rank_idx in range(6):
                        rank_name = triplet_rank_to_name.get(rank_idx, str(rank_idx))
                        thresholds = per_rank_thresholds.get(rank_idx, [])
                        if thresholds:
                            thresh_str = ", ".join([f"{t:.4f}" for t in thresholds])
                            f.write(f"  {rank_name:<8}: [{thresh_str}]\n")
                        elif budgets is not None and budgets[rank_idx] > 0:
                            f.write(f"  {rank_name:<8}: (uniform/warmup)\n")
                bucket_diag_all = per_rank_stats.get('bucket_diagnostics')
                if bucket_diag_all and any(entry is not None for entry in bucket_diag_all):
                    f.write("\nPer-Rank Bucket Fulfillment:\n")
                    for rank_idx in range(6):
                        diag_entry = bucket_diag_all[rank_idx] if bucket_diag_all is not None else None
                        if not diag_entry:
                            continue
                        diag_str = _format_bucket_diag_entries(diag_entry)
                        if not diag_str:
                            continue
                        rank_name = triplet_rank_to_name.get(rank_idx, str(rank_idx))
                        f.write(f"  {rank_name:<8}: {diag_str}\n")
                shortages = per_rank_stats.get('representative_shortages') if per_rank_stats else None
                if shortages:
                    f.write("\nRepresentative Set Shortages:\n")
                    for shortage in shortages:
                        deficit = shortage['requested'] - shortage['available']
                        f.write(f"  - {shortage['rank_name']}: requested {shortage['requested']} but found "
                                f"{shortage['available']} (short by {deficit}).\n")
                f.write("\n")
            
            for line in table_lines:
                f.write(line + "\n")
        
        pool_unique_ranks = table_stats["pool_unique_ranks"]
        filtered_unique_ranks = table_stats["filtered_unique_ranks"]
        sampled_ranks = table_stats["sampled_ranks"]
        n_pool = table_stats["n_pool"]
        n_filtered = table_stats["n_filtered"]
        n_sampled = table_stats["n_sampled"]
        pool_hard_violations = table_stats["pool_hard_violations"]
        pool_ordered_no_margin = table_stats["pool_ordered_no_margin"]
        pool_margin_satisfied = table_stats["pool_margin_satisfied"]
        hard_violations = table_stats["hard_violations"]
        ordered_no_margin = table_stats["ordered_no_margin"]
        margin_satisfied = table_stats["margin_satisfied"]
        n_buckets = len(percentile_thresholds) + 1

        # Write main CSV - now includes both pool and filtered stats plus per-rank mining data
        write_header = not os.path.exists(csv_path)
        with open(csv_path, "a") as f:
            if write_header:
                f.write("batch,n_pool,n_filtered,n_sampled,pool_norm_viol_min,pool_norm_viol_max,pool_norm_viol_mean,pool_norm_viol_std,pool_pct_ap_gte_an,pool_pct_ap_lt_an,pool_pct_margin_satisfied,filtered_norm_viol_min,filtered_norm_viol_max,filtered_norm_viol_mean,filtered_norm_viol_std,filtered_pct_ap_gte_an,filtered_pct_ap_lt_an,filtered_pct_margin_satisfied,pool_true_ap_mean,pool_pred_ap_mean,pool_true_an_mean,pool_pred_an_mean,filtered_true_ap_mean,filtered_pred_ap_mean,filtered_true_an_mean,filtered_pred_an_mean,margin_mean,warmup_phase,total_deficit,bucket_base_proportions,bucket_target_proportions\n")
            pool_pct_hard = pool_hard_violations.mean() * 100
            pool_pct_ordered_no_margin = pool_ordered_no_margin.mean() * 100
            pool_pct_margin = pool_margin_satisfied.mean() * 100
            filt_pct_hard = hard_violations.mean() * 100
            filt_pct_ordered_no_margin = ordered_no_margin.mean() * 100
            filt_pct_margin = margin_satisfied.mean() * 100
            warmup = per_rank_stats.get('warmup_phase', 0.0) if per_rank_stats is not None else 0.0
            total_deficit = per_rank_stats['deficits'].sum() if per_rank_stats is not None and per_rank_stats.get('deficits') is not None else 0
            bucket_base = _bucket_vector_for_csv(per_rank_stats.get('bucket_proportions_base')) if per_rank_stats is not None else ""
            bucket_target = _bucket_vector_for_csv(per_rank_stats.get('bucket_proportions_target')) if per_rank_stats is not None else ""
            f.write(f"{batch_num},{n_pool},{n_filtered},{n_sampled},{pool_normalised_violations.min():.6f},{pool_normalised_violations.max():.6f},{pool_normalised_violations.mean():.6f},{pool_normalised_violations.std():.6f},{pool_pct_hard:.4f},{pool_pct_ordered_no_margin:.4f},{pool_pct_margin:.4f},{normalised_violations.min():.6f},{normalised_violations.max():.6f},{normalised_violations.mean():.6f},{normalised_violations.std():.6f},{filt_pct_hard:.4f},{filt_pct_ordered_no_margin:.4f},{filt_pct_margin:.4f},{pool_true_ap.mean():.6f},{pool_pred_ap.mean():.6f},{pool_true_an.mean():.6f},{pool_pred_an.mean():.6f},{true_ap.mean():.6f},{pred_ap.mean():.6f},{true_an.mean():.6f},{pred_an.mean():.6f},{margins.mean():.6f},{warmup:.4f},{total_deficit},{bucket_base},{bucket_target}\n")
        
        # Write by-rank CSV - updated with per-rank mining data (EMA, budgets, deficits)
        write_header = not os.path.exists(csv_by_rank_path)
        with open(csv_by_rank_path, "a") as f:
            if write_header:
                f.write("batch,rank,ema_hardness_pre,ema_hardness_budget,ema_hardness_post,batch_hardness,budget,proportion,n_pool_pre_filter,n_pool,n_sampled,deficit,n_hard_after_filter,n_topup,min_after_filter,pool_norm_viol_mean,filtered_norm_viol_mean,pool_pct_ap_gte_an,filtered_pct_ap_gte_an,margin_mean,bucket_thresholds,bucket_targets,bucket_sampled,bucket_residual_deficit,bucket_borrowed_from\n")
            
            # Use per_rank_stats if available for comprehensive per-rank data
            if per_rank_stats is not None:
                ema_pre = per_rank_stats.get('ema_hardness_pre')
                ema_budget = per_rank_stats.get('ema_hardness_budget')
                ema_post = per_rank_stats.get('ema_hardness_post')
                hardness_metrics = per_rank_stats.get('hardness_metrics')
                budgets = per_rank_stats.get('budgets')
                proportions = per_rank_stats.get('proportions')
                pool_counts = per_rank_stats.get('pool_counts')
                pool_counts_pre_filter = per_rank_stats.get('pool_counts_pre_filter')
                sampled_counts_stats = per_rank_stats.get('sampled_counts')
                deficits = per_rank_stats.get('deficits')
                hard_counts = per_rank_stats.get('filter_hard_counts')
                topup_counts = per_rank_stats.get('filter_topup_counts')
                min_counts = per_rank_stats.get('filter_minimums')
                per_rank_thresholds = per_rank_stats.get('per_rank_thresholds', {})
                bucket_diag_all = per_rank_stats.get('bucket_diagnostics') or []
                
                for rank_idx in range(6):
                    rank_name = triplet_rank_to_name.get(rank_idx, str(rank_idx))
                    
                    ema_pre_val = ema_pre[rank_idx] if ema_pre is not None else 0.0
                    ema_budget_val = ema_budget[rank_idx] if ema_budget is not None else 0.0
                    ema_post_val = ema_post[rank_idx] if ema_post is not None else 0.0
                    batch_hard_val = hardness_metrics[rank_idx] if hardness_metrics is not None else float('nan')
                    budget_val = budgets[rank_idx] if budgets is not None else 0
                    prop_val = proportions[rank_idx] if proportions is not None else 0.0
                    pre_filt_val = pool_counts_pre_filter[rank_idx] if pool_counts_pre_filter is not None else 0
                    pool_val = pool_counts[rank_idx] if pool_counts is not None else 0
                    sampled_val = sampled_counts_stats[rank_idx] if sampled_counts_stats is not None else 0
                    deficit_val = deficits[rank_idx] if deficits is not None else 0
                    hard_val = hard_counts[rank_idx] if hard_counts is not None else 0
                    topup_val = topup_counts[rank_idx] if topup_counts is not None else 0
                    min_val = min_counts[rank_idx] if min_counts is not None else 0
                    
                    # Compute violation stats for this rank from the arrays
                    pool_mask = (pool_triplet_ranks_arr == rank_idx)
                    filt_mask = (triplet_ranks_arr == rank_idx)
                    n_p = np.sum(pool_mask)
                    n_f = np.sum(filt_mask)
                    
                    pool_viol_mean = pool_normalised_violations[pool_mask].mean() if n_p > 0 else 0.0
                    filt_viol_mean = normalised_violations[filt_mask].mean() if n_f > 0 else 0.0
                    pool_pct_hard_val = pool_hard_violations[pool_mask].mean() * 100 if n_p > 0 else 0.0
                    filt_pct_hard_val = hard_violations[filt_mask].mean() * 100 if n_f > 0 else 0.0
                    margin_mean_val = margins[filt_mask].mean() if n_f > 0 else 0.0
                    
                    thresholds = per_rank_thresholds.get(rank_idx, [])
                    thresh_str = "|".join([f"{t:.6f}" for t in thresholds]) if thresholds else ""
                    
                    bucket_diag = bucket_diag_all[rank_idx] if rank_idx < len(bucket_diag_all) else None
                    bucket_targets_str, bucket_sampled_str, bucket_residual_str, bucket_borrowed_str = _bucket_diag_strings_for_csv(bucket_diag)
                    
                    f.write(f"{batch_num},{rank_name},{ema_pre_val:.6f},{ema_budget_val:.6f},{ema_post_val:.6f},{batch_hard_val:.6f},{budget_val},{prop_val:.6f},{pre_filt_val},{pool_val},{sampled_val},{deficit_val},{hard_val},{topup_val},{min_val},{pool_viol_mean:.6f},{filt_viol_mean:.6f},{pool_pct_hard_val:.4f},{filt_pct_hard_val:.4f},{margin_mean_val:.6f},{thresh_str},{bucket_targets_str},{bucket_sampled_str},{bucket_residual_str},{bucket_borrowed_str}\n")
            else:
                # Fallback to old behavior if per_rank_stats not provided
                for r in pool_unique_ranks:
                    pool_mask = (pool_triplet_ranks_arr == r)
                    filt_mask = (triplet_ranks_arr == r)
                    n_p = np.sum(pool_mask)
                    n_f = np.sum(filt_mask)
                    n_s = np.sum(sampled_ranks == r)
                    pool_viol_mean = pool_normalised_violations[pool_mask].mean() if n_p > 0 else 0.0
                    filt_viol_mean = normalised_violations[filt_mask].mean() if n_f > 0 else 0.0
                    pool_pct_hard_val = pool_hard_violations[pool_mask].mean() * 100 if n_p > 0 else 0.0
                    filt_pct_hard_val = hard_violations[filt_mask].mean() * 100 if n_f > 0 else 0.0
                    margin_mean_val = margins[filt_mask].mean() if n_f > 0 else 0.0
                    rank_name = triplet_rank_to_name.get(r, str(r))
                    f.write(f"{batch_num},{rank_name},0.0,0.0,0.0,0.0,0,0.0,0,{n_f},{n_s},0,0,0,0,{pool_viol_mean:.6f},{filt_viol_mean:.6f},{pool_pct_hard_val:.4f},{filt_pct_hard_val:.4f},{margin_mean_val:.6f},,,,,\n")
        
        # Write by-bucket CSV (filtered pool only - buckets are based on filtered)
        write_header = not os.path.exists(csv_by_bucket_path)
        with open(csv_by_bucket_path, "a") as f:
            if write_header:
                f.write("batch,bucket,pct_low,pct_high,n_in_bucket,n_sampled,norm_viol_min,norm_viol_max,pct_filtered,pct_sampled\n")
            for b in range(n_buckets):
                lower_pct = 0.0 if b == 0 else percentile_thresholds[b-1] * 100
                upper_pct = 100.0 if b == len(percentile_thresholds) else percentile_thresholds[b] * 100
                mask = (bucket_assignments == b)
                if np.any(mask):
                    viols = normalised_violations[mask]
                    n_in_bucket = len(viols)
                    n_filtered_total = len(normalised_violations)
                    n_sampled_in_bucket = np.sum(bucket_assignments[sampled_local_indices] == b)
                    n_sampled_total = len(sampled_local_indices)
                    pct_filtered = (n_in_bucket / n_filtered_total * 100) if n_filtered_total > 0 else 0.0
                    pct_sampled = (n_sampled_in_bucket / n_sampled_total * 100) if n_sampled_total > 0 else 0.0
                    f.write(f"{batch_num},{b},{lower_pct:.2f},{upper_pct:.2f},{n_in_bucket},{n_sampled_in_bucket},{viols.min():.6f},{viols.max():.6f},{pct_filtered:.4f},{pct_sampled:.4f}\n")
                else:
                    f.write(f"{batch_num},{b},{lower_pct:.2f},{upper_pct:.2f},0,0,0.0,0.0,0.0,0.0\n")
        
        # Populate triplet satisfaction dataframe if provided
        # Uses pool (unfiltered) data for consistent metrics across all triplet candidates
        if triplet_satisfaction_df is not None:
            satisfaction_by_rank = {}
            for r in pool_unique_ranks:
                pool_mask = (pool_triplet_ranks_arr == r)
                if np.sum(pool_mask) > 0:
                    pct_satisfied = pool_margin_satisfied[pool_mask].mean() * 100
                    satisfaction_by_rank[r] = pct_satisfied
            overall_satisfaction = pool_margin_satisfied.mean() * 100 if len(pool_margin_satisfied) > 0 else 0.0
            
            # Add row to the dataframe in-place
            add_triplet_satisfaction_row_inplace(triplet_satisfaction_df, batch_num, satisfaction_by_rank, overall_satisfaction)
        
        # Populate triplet error metrics dataframe if provided
        # The metric is: hard_triplet_prop * TRIPLET_EMA_HARD_WEIGHT + moderate_triplet_prop * TRIPLET_EMA_MODERATE_WEIGHT
        # This is stored in per_rank_stats['hardness_metrics'], indexed by rank (0-5)
        if triplet_error_metrics_df is not None and per_rank_stats is not None:
            hardness_metrics = per_rank_stats.get('hardness_metrics')
            if hardness_metrics is not None:
                # Convert to dict mapping rank to metric value
                metrics_by_rank = {}
                for rank_idx in range(len(hardness_metrics)):
                    metric_val = hardness_metrics[rank_idx]
                    if not np.isnan(metric_val):
                        metrics_by_rank[rank_idx] = float(metric_val)
                
                # Add row to the dataframe in-place
                add_triplet_error_metrics_row_inplace(triplet_error_metrics_df, batch_num, metrics_by_rank)
            
    except Exception as e:
        print(f"Warning: Failed to write triplet mining log: {e}")


# Model Parameters --------------------------

def print_log_uncertainty_weighting(uncertainty_loss, batch_num, logs_dir):
    """
    Prints and/or logs the model's uncertainty weighting parameters if they exist.

    - Printing is controlled by VERBOSE_UNCERTAINTY_WEIGHTING and VERBOSE_EVERY_N_BATCHES.
    - Logging is controlled by LOG_UNCERTAINTY_WEIGHTING and LOG_EVERY_N_BATCHES.
    """
    if gb.IS_USING_UNCERTAINTY_WEIGHTING and uncertainty_loss is not None:
        # Access log_vars from the UncertaintyLoss module
        # log_vars[0] = triplet log_var, log_vars[1] = pair log_var
        log_vars = uncertainty_loss.log_vars.detach()
        s_triplet, s_pair = log_vars[0].item(), log_vars[1].item()
        
        # Convert log-variance to weighting factor: w = exp(-s)
        w_triplet, w_pair = torch.exp(-log_vars[0]).item(), torch.exp(-log_vars[1]).item()

        # Printing
        should_print = gb.VERBOSE_UNCERTAINTY_WEIGHTING and (batch_num % gb.VERBOSE_EVERY_N_BATCHES == 0)
        if should_print:
            print(f"Uncertainty Weights - Triplet: {w_triplet:.4f} (log_var: {s_triplet:.4f}), Pair: {w_pair:.4f} (log_var: {s_pair:.4f})")

        # Logging
        should_log = gb.LOG_UNCERTAINTY_WEIGHTING and (batch_num == 1 or batch_num % gb.LOG_EVERY_N_BATCHES == 0)
        if should_log:
            csv_path = os.path.join(logs_dir, "uncertainty_weighting.csv")
            write_header = not os.path.exists(csv_path)
            with open(csv_path, "a") as f:
                if write_header:
                    f.write("batch,triplet_log_var,pair_log_var,triplet_weight,pair_weight\n")
                f.write(f"{batch_num},{s_triplet:.6f},{s_pair:.6f},{w_triplet:.6f},{w_pair:.6f}\n")


def print_log_conv_stem_scale(model, batch_num, logs_dir):
    """
    Prints and/or logs the model's conv stem scale parameter if it exists.

    - Printing is controlled by VERBOSE_CONV_STEM_SCALE and VERBOSE_EVERY_N_BATCHES.
    - Logging is controlled by LOG_CONV_STEM_SCALE and LOG_EVERY_N_BATCHES.
    """
    if gb.USE_CONV_STEM and hasattr(model, 'conv_stem_scale'):
        scale_value = model.conv_stem_scale.detach().item()

        # Printing
        should_print = gb.VERBOSE_CONV_STEM_SCALE and (batch_num % gb.VERBOSE_EVERY_N_BATCHES == 0)
        if should_print:
            print(f"Conv Stem Scale: {scale_value:.6f}")

        # Logging
        should_log = gb.LOG_CONV_STEM_SCALE and (batch_num == 1 or batch_num % gb.LOG_EVERY_N_BATCHES == 0)
        if should_log:
            csv_path = os.path.join(logs_dir, "conv_stem_scale.csv")
            write_header = not os.path.exists(csv_path)
            with open(csv_path, "a") as f:
                if write_header:
                    f.write("batch,conv_stem_scale\n")
                f.write(f"{batch_num},{scale_value:.6f}\n")


def log_arc_bac_counts(batch_num, logs_dir, num_archaea, num_bacteria):
    """
    Append the per-batch archaea and bacteria counts to arc_bac_counts.csv.
    """
    if logs_dir is None or batch_num is None:
        return
    csv_path = os.path.join(logs_dir, "arc_bac_counts.csv")
    write_header = not os.path.exists(csv_path)
    num_archaea = int(num_archaea)
    num_bacteria = int(num_bacteria)
    total = num_archaea + num_bacteria
    with open(csv_path, "a") as f:
        if write_header:
            f.write("batch,num_arc,num_bac,total\n")
        f.write(f"{batch_num},{num_archaea},{num_bacteria},{total}\n")


def write_mining_phylum_log(batch_num, logs_dir, phylum_labels, step_stats,
                            final_pair_combo_counts, final_triplet_an_combo_counts,
                            max_lines_per_table=8):
    """
    Write mining phylum diagnostics to a dedicated logging directory.

    Outputs:
        logs/mining_phyla/phylum_step_totals.csv
        logs/mining_phyla/phylum_step_counts.csv
        logs/mining_phyla/final_pair_phylum_counts.csv
        logs/mining_phyla/final_triplet_an_phylum_counts.csv
        logs/mining_phyla/phylum_mining.log
    """
    if logs_dir is None or batch_num is None or phylum_labels is None:
        return

    def _csv_safe(text):
        return str(text).replace(",", ";")

    try:
        max_lines = max(1, int(max_lines_per_table))
    except Exception:
        max_lines = 8

    phylum_dir = os.path.join(logs_dir, "mining_phyla")
    os.makedirs(phylum_dir, exist_ok=True)

    n_phyla = len(phylum_labels)
    safe_labels = [_csv_safe(label) for label in phylum_labels]

    step_totals_path = os.path.join(phylum_dir, "phylum_step_totals.csv")
    step_counts_path = os.path.join(phylum_dir, "phylum_step_counts.csv")
    final_pairs_path = os.path.join(phylum_dir, "final_pair_phylum_counts.csv")
    final_triplets_path = os.path.join(phylum_dir, "final_triplet_an_phylum_counts.csv")
    text_log_path = os.path.join(phylum_dir, "phylum_mining.log")

    try:
        # Per-step totals
        write_header = not os.path.exists(step_totals_path)
        with open(step_totals_path, "a") as f:
            if write_header:
                f.write("batch,step,n_sequences,n_active_pairs,n_active_pair_endpoints\n")
            for entry in step_stats:
                if not entry:
                    continue
                step_name = _csv_safe(entry.get("step", "unknown"))
                seq_counts = np.asarray(entry.get("seq_counts", np.zeros(n_phyla, dtype=np.int64)), dtype=np.int64)
                endpoint_counts = np.asarray(entry.get("pair_endpoint_counts", np.zeros(n_phyla, dtype=np.int64)), dtype=np.int64)
                n_sequences = int(seq_counts.sum())
                n_active_pair_endpoints = int(endpoint_counts.sum())
                n_active_pairs = int(entry.get("n_active_pairs", n_active_pair_endpoints // 2))
                f.write(f"{batch_num},{step_name},{n_sequences},{n_active_pairs},{n_active_pair_endpoints}\n")

        # Per-step phylum rows
        write_header = not os.path.exists(step_counts_path)
        with open(step_counts_path, "a") as f:
            if write_header:
                f.write("batch,step,phylum,seq_count,active_pair_endpoints,active_pair_endpoints_prop\n")
            for entry in step_stats:
                if not entry:
                    continue
                step_name = _csv_safe(entry.get("step", "unknown"))
                seq_counts = np.asarray(entry.get("seq_counts", np.zeros(n_phyla, dtype=np.int64)), dtype=np.int64)
                endpoint_counts = np.asarray(entry.get("pair_endpoint_counts", np.zeros(n_phyla, dtype=np.int64)), dtype=np.int64)
                endpoint_total = float(endpoint_counts.sum())
                for phylum_idx, phylum_label in enumerate(safe_labels):
                    seq_count = int(seq_counts[phylum_idx]) if phylum_idx < len(seq_counts) else 0
                    endpoint_count = int(endpoint_counts[phylum_idx]) if phylum_idx < len(endpoint_counts) else 0
                    endpoint_prop = (endpoint_count / endpoint_total) if endpoint_total > 0 else 0.0
                    f.write(f"{batch_num},{step_name},{phylum_label},{seq_count},{endpoint_count},{endpoint_prop:.8f}\n")

        # Final mined pair phylum combinations (unordered)
        total_final_pairs = int(sum(final_pair_combo_counts.values())) if final_pair_combo_counts else 0
        write_header = not os.path.exists(final_pairs_path)
        with open(final_pairs_path, "a") as f:
            if write_header:
                f.write("batch,phylum_a,phylum_b,count,prop_of_mined_pairs\n")
            pair_items = sorted(
                final_pair_combo_counts.items(),
                key=lambda kv: (-int(kv[1]), safe_labels[kv[0][0]], safe_labels[kv[0][1]])
            ) if final_pair_combo_counts else []
            for (a_idx, b_idx), count in pair_items:
                prop = (int(count) / float(total_final_pairs)) if total_final_pairs > 0 else 0.0
                f.write(f"{batch_num},{safe_labels[a_idx]},{safe_labels[b_idx]},{int(count)},{prop:.8f}\n")

        # Final mined triplet A-N phylum combinations (ordered)
        total_final_triplets = int(sum(final_triplet_an_combo_counts.values())) if final_triplet_an_combo_counts else 0
        write_header = not os.path.exists(final_triplets_path)
        with open(final_triplets_path, "a") as f:
            if write_header:
                f.write("batch,anchor_phylum,negative_phylum,count,prop_of_mined_triplets\n")
            triplet_items = sorted(
                final_triplet_an_combo_counts.items(),
                key=lambda kv: (-int(kv[1]), safe_labels[kv[0][0]], safe_labels[kv[0][1]])
            ) if final_triplet_an_combo_counts else []
            for (a_idx, n_idx), count in triplet_items:
                prop = (int(count) / float(total_final_triplets)) if total_final_triplets > 0 else 0.0
                f.write(f"{batch_num},{safe_labels[a_idx]},{safe_labels[n_idx]},{int(count)},{prop:.8f}\n")

        # Human-readable text summary (top rows only)
        with open(text_log_path, "a") as f:
            f.write(f"\n---\nBatch {batch_num}\n")
            f.write("\nMining Phylum Logging -------------------------------------\n")

            for entry in step_stats:
                if not entry:
                    continue
                step_name = entry.get("step", "unknown")
                seq_counts = np.asarray(entry.get("seq_counts", np.zeros(n_phyla, dtype=np.int64)), dtype=np.int64)
                endpoint_counts = np.asarray(entry.get("pair_endpoint_counts", np.zeros(n_phyla, dtype=np.int64)), dtype=np.int64)
                n_sequences = int(seq_counts.sum())
                n_active_pair_endpoints = int(endpoint_counts.sum())
                n_active_pairs = int(entry.get("n_active_pairs", n_active_pair_endpoints // 2))
                seq_items = [(i, int(seq_counts[i])) for i in range(n_phyla) if int(seq_counts[i]) > 0]
                endpoint_items = [(i, int(endpoint_counts[i])) for i in range(n_phyla) if int(endpoint_counts[i]) > 0]
                seq_items.sort(key=lambda x: (-x[1], safe_labels[x[0]]))
                endpoint_items.sort(key=lambda x: (-x[1], safe_labels[x[0]]))

                f.write(f"\nStep: {step_name}\n")
                f.write(f"  n_sequences={n_sequences}, n_active_pairs={n_active_pairs}\n")
                f.write("  Sequence counts by phylum:\n")
                if not seq_items:
                    f.write("    (none)\n")
                else:
                    for idx, count in seq_items[:max_lines]:
                        seq_prop = (count / float(n_sequences)) if n_sequences > 0 else 0.0
                        f.write(f"    {safe_labels[idx]}: {count} ({seq_prop:.2%})\n")
                    if len(seq_items) > max_lines:
                        f.write(f"    ... {len(seq_items) - max_lines} more phyla\n")

                f.write("  Active pair endpoints by phylum:\n")
                if not endpoint_items:
                    f.write("    (none)\n")
                else:
                    for idx, count in endpoint_items[:max_lines]:
                        endpoint_prop = (count / float(n_active_pair_endpoints)) if n_active_pair_endpoints > 0 else 0.0
                        f.write(f"    {safe_labels[idx]}: {count} ({endpoint_prop:.2%})\n")
                    if len(endpoint_items) > max_lines:
                        f.write(f"    ... {len(endpoint_items) - max_lines} more phyla\n")

            f.write("\nFinal mined pairs by phylum combination:\n")
            if not final_pair_combo_counts:
                f.write("  (none)\n")
            else:
                pair_items = sorted(
                    final_pair_combo_counts.items(),
                    key=lambda kv: (-int(kv[1]), safe_labels[kv[0][0]], safe_labels[kv[0][1]])
                )
                for (a_idx, b_idx), count in pair_items[:max_lines]:
                    prop = (int(count) / float(total_final_pairs)) if total_final_pairs > 0 else 0.0
                    f.write(f"  {safe_labels[a_idx]} <-> {safe_labels[b_idx]}: {int(count)} ({prop:.2%})\n")
                if len(pair_items) > max_lines:
                    f.write(f"  ... {len(pair_items) - max_lines} more phylum combinations\n")

            f.write("\nFinal mined triplet A-N phylum combinations:\n")
            if not final_triplet_an_combo_counts:
                f.write("  (none)\n")
            else:
                triplet_items = sorted(
                    final_triplet_an_combo_counts.items(),
                    key=lambda kv: (-int(kv[1]), safe_labels[kv[0][0]], safe_labels[kv[0][1]])
                )
                for (a_idx, n_idx), count in triplet_items[:max_lines]:
                    prop = (int(count) / float(total_final_triplets)) if total_final_triplets > 0 else 0.0
                    f.write(f"  {safe_labels[a_idx]} -> {safe_labels[n_idx]}: {int(count)} ({prop:.2%})\n")
                if len(triplet_items) > max_lines:
                    f.write(f"  ... {len(triplet_items) - max_lines} more phylum combinations\n")

    except Exception as e:
        print(f"Warning: Failed to write mining phylum log: {e}")


def write_mining_log(batch_num, logs_dir, pairwise_ranks, pre_tsn_ignored=None, taxon_bias_enabled=False,
                     taxon_subsampling_counts_pre=None, taxon_subsampling_counts_post=None):
    """
    Write mining-level statistics to mining.log.
    
    Log files:
        mining.log: Text dump of mining-level metrics with batch headers
    """
    if logs_dir is None or batch_num is None or pairwise_ranks is None:
        return
    log_path = os.path.join(logs_dir, "mining.log")
    try:
        n_seqs = pairwise_ranks.shape[0]
        n_pairs_total = pairwise_ranks.size
        taxon_subsampled_pairs = 0
        taxon_subsampled_prop = 0.0
        per_rank_drop_lines = None
        if taxon_bias_enabled:
            if pre_tsn_ignored is None:
                pre_tsn_ignored = 0
            post_tsn_ignored = np.count_nonzero(pairwise_ranks == -2)
            taxon_subsampled_pairs = max(0, int(post_tsn_ignored - pre_tsn_ignored))
            if n_pairs_total:
                taxon_subsampled_prop = taxon_subsampled_pairs / float(n_pairs_total)
            if taxon_subsampling_counts_pre is not None and taxon_subsampling_counts_post is not None:
                pre_counts = np.asarray(taxon_subsampling_counts_pre, dtype=np.int64)
                post_counts = np.asarray(taxon_subsampling_counts_post, dtype=np.int64)
                if pre_counts.shape[0] >= 10 and post_counts.shape[0] >= 10:
                    rank_names = ["domain", "phylum", "class", "order", "family", "genus",
                                  "species", "sequence", "subseq"]
                    pre_rank_counts = pre_counts[1:10]
                    post_rank_counts = post_counts[1:10]
                    dropped_rank_counts = np.clip(pre_rank_counts - post_rank_counts, 0, None)
                    per_rank_drop_lines = []
                    for rank_name, dropped_count, pre_count in zip(rank_names, dropped_rank_counts, pre_rank_counts):
                        pre_count = int(pre_count)
                        dropped_count = int(dropped_count)
                        drop_prop = (dropped_count / float(pre_count)) if pre_count > 0 else 0.0
                        per_rank_drop_lines.append(
                            f"    {rank_name}: dropped {dropped_count}/{pre_count} pairs ({drop_prop:.2%})"
                        )
        with open(log_path, "a") as f:
            f.write(f"\n---\nBatch {batch_num}\n")
            f.write(f"\nMining Logging -------------------------------------\n")
            if taxon_bias_enabled:
                f.write(f"Taxon-size subsampling: dropped {taxon_subsampled_pairs}/{n_pairs_total} pairs ({taxon_subsampled_prop:.2%}) [n_seqs={n_seqs}]\n")
                if per_rank_drop_lines:
                    f.write("  Per-rank drops:\n")
                    for line in per_rank_drop_lines:
                        f.write(f"{line}\n")
            else:
                f.write(f"Taxon-size subsampling: disabled (0.00% dropped) [n_seqs={n_seqs}]\n")
    except Exception as e:
        print(f"Warning: Failed to write mining log: {e}")

# Bucket Threshold Logging --------------------------

_BUCKET_THRESHOLD_CACHE = {}

def _read_last_bucket_thresholds(filepath):
    """
    Reads the last batch's thresholds from the CSV file.
    Returns a dict: {bucket_index: threshold_value}
    """
    if not os.path.exists(filepath):
        return {}
    
    last_thresholds = {}
    
    try:
        # Read a chunk from the end of the file
        with open(filepath, 'rb') as f:
            try:
                f.seek(0, os.SEEK_END)
                filesize = f.tell()
                # Read last 20KB (should cover a batch easily)
                seek_dist = min(filesize, 1024 * 20)
                f.seek(-seek_dist, os.SEEK_END)
                lines = f.readlines()
            except OSError:
                # Fallback for very small files or seek errors
                return {}
            
        if not lines:
            return {}
            
        # Parse lines from the end to find the last batch number
        # Format: batch_num,rank_name,bucket_index,threshold_value,...
        
        # Find the last non-empty line
        last_line = None
        for line in reversed(lines):
            decoded = line.decode('utf-8', errors='ignore').strip()
            if decoded:
                last_line = decoded
                break
        
        if not last_line:
            return {}
            
        parts = last_line.split(',')
        if len(parts) < 4:
            return {}
            
        try:
            target_batch = int(parts[0])
        except ValueError:
            return {}
            
        # Collect all rows for target_batch
        for line in reversed(lines):
            decoded = line.decode('utf-8', errors='ignore').strip()
            if not decoded:
                continue
            parts = decoded.split(',')
            try:
                batch = int(parts[0])
                if batch != target_batch:
                    # Since we read backwards, if we see a different batch, it must be the previous one
                    # assuming the file is ordered.
                    if batch < target_batch:
                        break
                    continue 
                
                bucket_idx = int(parts[2])
                val_str = parts[3]
                threshold = float(val_str) if val_str and val_str != 'None' else None
                last_thresholds[bucket_idx] = threshold
            except (ValueError, IndexError):
                continue
                
        return last_thresholds
    except Exception as e:
        print(f"Error reading last bucket thresholds from {filepath}: {e}")
        return {}


def log_bucket_thresholds(batch_num, logs_dir, mining_type, per_rank_thresholds, per_rank_stats, rank_map, bucket_configs):
    """
    Log bucket thresholds for pair or triplet mining.
    
    Args:
        batch_num: int
        logs_dir: str
        mining_type: str ("pair" or "triplet")
        per_rank_thresholds: dict {rank_id: [thresholds]}
        per_rank_stats: dict containing 'pool_counts'
        rank_map: dict {rank_id: rank_name}
        bucket_configs: list of (gap, proportion) tuples
    """
    if logs_dir is None:
        return

    subdir = os.path.join(logs_dir, f"{mining_type}_bucket_thresholds")
    os.makedirs(subdir, exist_ok=True)
    
    global _BUCKET_THRESHOLD_CACHE
    
    for rank_id, thresholds in per_rank_thresholds.items():
        rank_name = rank_map.get(rank_id, str(rank_id))
        filename = f"{rank_id}_{rank_name}_bucket_thresholds.csv"
        filepath = os.path.join(subdir, filename)
        
        # Determine pool size
        # Pair mining uses ranks -1..7 (indices 0..8)
        # Triplet mining uses ranks 0..5 (indices 0..5)
        # per_rank_stats['pool_counts'] is indexed by 0..N
        if mining_type == 'pair':
            # Map internal rank (-1..7) to index (0..8)
            pool_idx = rank_id + 1
        else:
            pool_idx = rank_id
            
        # Get available count (total candidates BEFORE subsampling to representative set size)
        # This shows how many pairs/triplets we *could have* sampled from, not the capped pool size.
        available_size = 0
        if per_rank_stats:
            # Use available_counts which captures the total available before representative set subsampling
            if 'available_counts' in per_rank_stats and per_rank_stats['available_counts'] is not None:
                if 0 <= pool_idx < len(per_rank_stats['available_counts']):
                    available_size = int(per_rank_stats['available_counts'][pool_idx])
        
        # Determine target representative set size from config
        if mining_type == 'pair':
            target_sizes = getattr(gb, "PAIR_MINING_REPRESENTATIVE_SET_SIZES", None)
        else:
            target_sizes = getattr(gb, "TRIPLET_MINING_REPRESENTATIVE_SET_SIZES", None)
        
        target_rep_size = None
        if target_sizes is not None and 0 <= pool_idx < len(target_sizes):
            target_rep_size = int(target_sizes[pool_idx])

        # Prepare cache key
        cache_key = filepath
        last_thresholds = _BUCKET_THRESHOLD_CACHE.get(cache_key)
        
        if last_thresholds is None:
            last_thresholds = _read_last_bucket_thresholds(filepath)
            _BUCKET_THRESHOLD_CACHE[cache_key] = last_thresholds
            
        new_cache_entry = {}
        
        # Check if file exists to write header
        file_exists = os.path.exists(filepath)
        
        try:
            with open(filepath, 'a') as f:
                if not file_exists:
                    f.write("batch_num,rank_name,bucket_index,threshold_value,relative_change_from_prev,target_representative_set_size,available_representative_set_size\n")
                
                current_thresholds = thresholds
                
                # Iterate through bucket configs
                for b_idx, (gap, _prop) in enumerate(bucket_configs):
                    # Determine threshold value (upper bound of the bucket)
                    # If we have N gaps (excluding None), we have N-1 thresholds.
                    # Bucket i < N-1 uses thresholds[i].
                    # Bucket N-1 (last gap bucket) and 'Any' bucket have None as upper threshold.
                    threshold_val = None
                    if current_thresholds and b_idx < len(current_thresholds):
                        threshold_val = current_thresholds[b_idx]
                    
                    # Compute relative change
                    rel_change = None
                    if threshold_val is not None:
                        prev_val = last_thresholds.get(b_idx)
                        if prev_val is not None:
                            if abs(prev_val) > 1e-9:
                                rel_change = (threshold_val - prev_val) / abs(prev_val)
                            elif abs(threshold_val) < 1e-9:
                                rel_change = 0.0
                    
                    # Update cache
                    new_cache_entry[b_idx] = threshold_val
                    
                    # Write row
                    row_str = f"{batch_num},{rank_name},{b_idx},{threshold_val if threshold_val is not None else 'None'},"
                    row_str += f"{f'{rel_change:.6f}' if rel_change is not None else 'None'},"
                    if target_rep_size is None:
                        row_str += "None,"
                    else:
                        row_str += f"{target_rep_size},"
                    row_str += f"{available_size}\n"
                    f.write(row_str)
        except Exception as e:
            print(f"Error logging bucket thresholds to {filepath}: {e}")
            continue
        
        # Update global cache
        _BUCKET_THRESHOLD_CACHE[cache_key] = new_cache_entry
