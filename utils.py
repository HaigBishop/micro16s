"""Just some boring utility functions."""

# Imports
import os
import re
import socket
import subprocess
from datetime import datetime

import torch
import globals_config as gb

def synchronize_if_cuda(obj=None):
    """
    Synchronize the CUDA stream (if any) tied to obj so timing measurements include GPU work.
    """
    if not torch.cuda.is_available():
        return

    device = None
    if obj is None:
        device = None
    elif isinstance(obj, torch.Tensor):
        device = obj.device
    elif isinstance(obj, torch.device):
        device = obj
    elif isinstance(obj, str):
        device = torch.device(obj)
    else:
        return

    if device is not None and device.type != 'cuda':
        return

    torch.cuda.synchronize(device if device is not None and device.index is not None else None)


def get_vram_usage_str(device):
    """
    Return a string like " - VRAM: 21.5/24.0GB" for CUDA devices, or "" otherwise.
    Uses memory_reserved() which shows total memory held by PyTorch's allocator,
    not just memory actively allocated to tensors. This gives a more accurate
    picture of actual VRAM usage as seen by nvidia-smi.
    Handles errors gracefully so training is never interrupted.
    """
    if device.type != 'cuda':
        return ""
    try:
        # Synchronize to ensure we get up-to-date memory stats
        torch.cuda.synchronize(device)
        
        # Use memory_reserved() instead of memory_allocated() for more accurate VRAM usage
        # memory_reserved() shows total memory held by PyTorch's caching allocator
        # memory_allocated() only shows memory actively used by tensors (can be much lower)
        vram_reserved = torch.cuda.memory_reserved(device) / (1024**3)  # Convert to GB
        vram_total = torch.cuda.get_device_properties(device).total_memory / (1024**3)  # Convert to GB
        
        return f" - VRAM: {vram_reserved:.1f}/{vram_total:.1f}GB"
    except Exception as e:
        # If synchronization or memory query fails, try without sync as fallback
        try:
            vram_reserved = torch.cuda.memory_reserved(device) / (1024**3)
            vram_total = torch.cuda.get_device_properties(device).total_memory / (1024**3)
            return f" - VRAM: {vram_reserved:.1f}/{vram_total:.1f}GB"
        except Exception:
            return ""


def parent_dir(path):
    """Get the parent directory of a given path."""
    return os.path.dirname(path.rstrip("/").rstrip("\\"))


def get_next_model_name(models_dir):
    """
    Generate the next model name (e.g., m16s_001, m16s_002, ...) by scanning for existing model folders.
    Args:
        models_dir (str): Path to the models directory.
    Returns:
        str: The next model name.
    """
    prefix = "m16s_"
    pattern = re.compile(r"^" + re.escape(prefix) + r"(\d{3})$")
    try:
        entries = os.listdir(models_dir)
    except FileNotFoundError:
        return f"{prefix}001"
    max_idx = 0
    for entry in entries:
        m = pattern.match(entry)
        if m:
            idx = int(m.group(1))
            if idx > max_idx:
                max_idx = idx
    next_idx = max_idx + 1
    return f"{prefix}{next_idx:03d}"


def print_micro16s(m16s_font='slant'):
    if m16s_font == 'slant':
        art = r'''
    __  ___ __ _____ ____  ____  ___ ____ ____
   /  |/  // // ____/ __ \/ __ \/  / ___// __/
  / /|_/ // // /   / /_/ / / / // / __ \ \_ \
 / /  / // // /___/ _  _/ /_/ // / /_/ /__/ /
/_/  /_//_/ \____/_/ |_|\____//_/\____/____/
        '''
    elif m16s_font == 'blocky':
        art = r'''
██     ██ ████  ██████  ████████   ███████     ██    ███████   ██████
███   ███  ██  ██    ██ ██     ██ ██     ██  ████   ██     ██ ██    ██
████ ████  ██  ██       ██     ██ ██     ██    ██   ██        ██
██ ███ ██  ██  ██       ████████  ██     ██    ██   ████████   ██████
██     ██  ██  ██       ██   ██   ██     ██    ██   ██     ██       ██
██     ██  ██  ██    ██ ██    ██  ██     ██    ██   ██     ██ ██    ██
██     ██ ████  ██████  ██     ██  ███████   ██████  ███████   ██████
        '''
    elif m16s_font == 'bloody':
        art = r'''
 ███▄ ▄███▓ ██▓ ▄████▄   ██▀███   ▒█████    ██████
▓██▒▀█▀ ██▒▓██▒▒██▀ ▀█  ▓██ ▒ ██▒▒██▒  ██▒▒██    ▒
▓██    ▓██░▒██▒▒▓█    ▄ ▓██ ░▄█ ▒▒██░  ██▒░ ▓██▄
▒██    ▒██ ░██░▒▓▓▄ ▄██▒▒██▀▀█▄  ▒██   ██░  ▒   ██▒
▒██▒   ░██▒░██░▒ ▓███▀ ░░██▓ ▒██▒░ ████▓▒░▒██████▒▒
░ ▒░   ░  ░░▓  ░ ░▒ ▒  ░░ ▒▓ ░▒▓░░ ▒░▒░▒░ ▒ ▒▓▒ ▒ ░
░  ░      ░ ▒ ░  ░  ▒     ░▒ ░ ▒░  ░ ▒ ▒░ ░ ░▒  ░ ░
░      ░    ▒ ░░          ░░   ░ ░ ░ ░ ▒  ░  ░  ░
       ░    ░  ░ ░         ░         ░ ░        ░
               ░
        '''
    elif m16s_font == 'ansi_shadow':
        art = r'''
███╗   ███╗██╗ ██████╗██████╗  ██████╗  ██╗ ██████╗ ███████╗
████╗ ████║██║██╔════╝██╔══██╗██╔═══██╗███║██╔════╝ ██╔════╝
██╔████╔██║██║██║     ██████╔╝██║   ██║╚██║███████╗ ███████╗
██║╚██╔╝██║██║██║     ██╔══██╗██║   ██║ ██║██╔═══██╗╚════██║
██║ ╚═╝ ██║██║╚██████╗██║  ██║╚██████╔╝ ██║╚██████╔╝███████║
╚═╝     ╚═╝╚═╝ ╚═════╝╚═╝  ╚═╝ ╚═════╝  ╚═╝ ╚═════╝ ╚══════╝
        '''
    else:
        art = 'Micro16S'

    print(art)


# Training summaries ------------------

def _format_duration(seconds):
    total_seconds = max(float(seconds), 0.0)
    minutes, secs = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)
    parts = []
    if int(days):
        parts.append(f"{int(days)}d")
    if int(hours):
        parts.append(f"{int(hours)}h")
    if int(minutes):
        parts.append(f"{int(minutes)}m")
    parts.append(f"{secs:.1f}s")
    return " ".join(parts)


def _device_description(device):
    if getattr(device, "type", "cpu") == "cuda":
        device_index = device.index if device.index is not None else torch.cuda.current_device()
        name = torch.cuda.get_device_name(device_index)
        major, minor = torch.cuda.get_device_capability(device_index)
        return f"cuda:{device_index} - {name} (capability {major}.{minor})"
    return "cpu"


def _git_commit_info(repo_path=None):
    """
    Return the latest git commit hash and message for the repo rooted at repo_path.
    Falls back to a descriptive error if git is unavailable or the command fails.
    """
    repo_path = repo_path or os.getcwd()
    try:
        result = subprocess.run(
            ["git", "-C", repo_path, "log", "-1", "--pretty=format:%H%x01%s"],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return {"error": "git executable not found"}
    except Exception as exc:  # pragma: no cover - defensive fallback
        return {"error": f"git invocation failed ({exc})"}

    if result.returncode != 0:
        err = result.stderr.strip() or "git log returned an error"
        return {"error": err}

    output = result.stdout.strip()
    if not output:
        return {"error": "git log returned no data"}

    parts = output.split("\x01", 1)
    commit_hash = parts[0].strip()
    message = parts[1].strip() if len(parts) > 1 else ""
    if not commit_hash:
        return {"error": "git log produced an empty hash"}
    return {"hash": commit_hash, "message": message}


def _has_rows(df):
    return df is not None and hasattr(df, "empty") and not df.empty


def _loss_snapshot(loss_df):
    if not _has_rows(loss_df):
        return None
    last_row = loss_df.iloc[-1]
    return {
        "batch": last_row.get("Batch"),
        "total": last_row.get("Total_Loss"),
        "triplet": last_row.get("Triplet_Loss"),
        "pair": last_row.get("Pair_Loss"),
    }


def _qt_summary(df, label):
    if not _has_rows(df):
        return None
    last_row = df.iloc[-1]
    last_batch = last_row.get("Batch", "n/a")
    return f"{label}: {len(df)} run(s), last batch {last_batch}"


def _fmt_float(value, precision=6):
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.{precision}f}"
    except Exception:
        return str(value)


def write_about_training(output_path, model_name, model_dir, training_run,
                         training_regimine=None, description=None, start_time=None,
                         end_time=None, batches_completed=None, device=None,
                         logs_dir=None, qt_results_dir=None, qt_plots_dir=None,
                         latest_model_path=None, loss_results_df=None,
                         clustering_scores_df=None, classification_scores_df=None,
                         macro_classification_scores_df=None, ssc_scores_df=None,
                         is_placeholder_about_file=False):
    """
    Write a summary text file that captures metadata about a completed training run.
    When is_placeholder_about_file=True, only metadata that is known before training
    begins is written so we have breadcrumbs if training exits early.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    start_dt = datetime.fromtimestamp(start_time) if start_time is not None else None
    finish_dt = datetime.fromtimestamp(end_time) if end_time is not None else None
    created_dt = finish_dt or start_dt or datetime.now()
    elapsed = None
    if start_time is not None and end_time is not None:
        elapsed = max(end_time - start_time, 0.0)
    avg_batch_time = elapsed / batches_completed if elapsed is not None and batches_completed else None

    host = socket.gethostname()
    device_desc = _device_description(device) if device is not None else "n/a"
    torch_version = torch.__version__
    repo_root = os.path.dirname(os.path.abspath(__file__))
    git_info = _git_commit_info(repo_root) or {}

    training_cfg_path = os.path.join(model_dir, 'cfg', 'training_config.txt') if model_dir else "n/a"
    run_desc = training_run or "n/a"
    regimine_desc = training_regimine or "n/a"
    with open(output_path, "w") as f:
        f.write("----------------------------------------\n")
        f.write(f"Micro16S Training Summary for {model_name}\n")
        f.write("----------------------------------------\n\n")
        if is_placeholder_about_file:
            f.write("Note: This about_training.txt file was written just before training started, and has not been overwritten at the end of the training run (as it should be when training completes). This suggests that training has not/did not complete.\n\n")
        f.write(f"Created: {created_dt.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Host: {host}\n")
        f.write(f"Device: {device_desc}\n")
        f.write(f"PyTorch: {torch_version}\n")
        if git_info.get("hash"):
            f.write(f"Git commit: {git_info['hash']}\n")
            git_message = git_info.get("message", "")
            if git_message:
                f.write(f"Git message: {git_message}\n")
        elif git_info.get("error"):
            f.write(f"Git commit: unavailable ({git_info['error']})\n")
        f.write(f"Training run: {run_desc} ({regimine_desc})\n")
        f.write(f"Description: {description or '-'}\n")
        f.write(f"Dataset split: {getattr(gb, 'DATASET_SPLIT_DIR', 'n/a')}\n")
        f.write(f"Model directory: {model_dir or 'n/a'}\n")
        f.write(f"Latest checkpoint: {latest_model_path or 'n/a'}\n")
        f.write(f"Logs directory: {logs_dir or 'n/a'}\n")
        f.write(f"Quick test results: {qt_results_dir or 'n/a'}\n")
        f.write(f"Quick test plots: {qt_plots_dir or 'n/a'}\n")
        f.write(f"Training config: {training_cfg_path}\n")

        if not is_placeholder_about_file:
            planned_batches = getattr(gb, "NUM_BATCHES", batches_completed if batches_completed is not None else "n/a")
            approx_pairs = None
            approx_triplets = None
            if getattr(gb, "N_PAIRS_PER_BATCH", None) is not None and batches_completed is not None:
                approx_pairs = int(batches_completed * gb.N_PAIRS_PER_BATCH)
            if getattr(gb, "N_TRIPLETS_PER_BATCH", None) is not None and batches_completed is not None:
                approx_triplets = int(batches_completed * gb.N_TRIPLETS_PER_BATCH)
            scheduler_desc = "None"
            if getattr(gb, "LR_SCHEDULER_TYPE", None):
                scheduler_desc = gb.LR_SCHEDULER_TYPE
                if getattr(gb, "LR_SCHEDULER_KWARGS", None):
                    scheduler_desc += f" (kwargs={gb.LR_SCHEDULER_KWARGS})"

            f.write("\nTiming:\n")
            if start_dt is not None:
                f.write(f"  - Training started: {start_dt.strftime('%Y-%m-%d %H:%M:%S')}\n")
            if finish_dt is not None:
                f.write(f"  - Training finished: {finish_dt.strftime('%Y-%m-%d %H:%M:%S')}\n")
            if elapsed is not None:
                f.write(f"  - Total duration: {_format_duration(elapsed)} ({elapsed:.2f} seconds)\n")
            if avg_batch_time is not None:
                f.write(f"  - Average batch time: {avg_batch_time:.2f} seconds\n")
            f.write(f"  - Planned batches: {planned_batches}\n")
            if batches_completed is not None:
                f.write(f"  - Completed batches: {batches_completed}\n")
            f.write(f"  - LR scheduler: {scheduler_desc}\n")

            f.write("\nThroughput Estimates:\n")
            if approx_pairs is not None:
                f.write(f"  - Approx. pairs seen: {approx_pairs:,}\n")
            if approx_triplets is not None:
                f.write(f"  - Approx. triplets seen: {approx_triplets:,}\n")
            f.write(f"  - Triplet loss weight: {getattr(gb, 'TRIPLET_LOSS_WEIGHT', 'n/a')}\n")
            f.write(f"  - Pair loss weight: {getattr(gb, 'PAIR_LOSS_WEIGHT', 'n/a')}\n")

            f.write("\nLoss Snapshot:\n")
            loss_info = _loss_snapshot(loss_results_df)
            if loss_info:
                batch_val = loss_info['batch']
                loss_entry_count = len(loss_results_df) if loss_results_df is not None else 0
                f.write(f"  - Last recorded batch: {batch_val}\n")
                f.write(f"  - Total loss: {_fmt_float(loss_info['total'])}\n")
                f.write(f"  - Triplet loss: {_fmt_float(loss_info['triplet'])}\n")
                f.write(f"  - Pair loss: {_fmt_float(loss_info['pair'])}\n")
                f.write(f"  - Loss entries recorded: {loss_entry_count}\n")
            else:
                f.write("  - No loss entries were recorded.\n")

            f.write("\nQuick Test Summary:\n")
            qt_enabled = bool(getattr(gb, "DO_QT", False))
            if qt_enabled:
                qt_before_first = bool(getattr(gb, "QT_BEFORE_FIRST_BATCH", False))
                qt_every_n = getattr(gb, "QT_EVERY_N_BATCHES", "n/a")
                f.write(f"  - Enabled every {qt_every_n} batch(es); before first batch: {qt_before_first}\n")
                qt_lines = [
                    _qt_summary(clustering_scores_df, "Clustering scores"),
                    _qt_summary(classification_scores_df, "Classification scores"),
                    _qt_summary(macro_classification_scores_df, "Macro-classification scores"),
                    _qt_summary(ssc_scores_df, "Subsequence congruency scores"),
                ]
                qt_lines = [line for line in qt_lines if line is not None]
                if qt_lines:
                    for line in qt_lines:
                        f.write(f"  - {line}\n")
                else:
                    f.write("  - Quick test scheduling was enabled but no runs were saved.\n")
            else:
                f.write("  - Quick tests disabled for this run.\n")
            f.write(f"  - Results stored at: {qt_results_dir or 'n/a'}\n")
            f.write(f"  - Plots stored at: {qt_plots_dir or 'n/a'}\n")
