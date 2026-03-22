"""
Generates sequence length histogram plots for FASTA files and calculates length statistics.

Reads FASTA files matching the pattern {TARGET_DIR}/*_seqs.fasta.
For each input file, it calculates the lengths of all sequences and
generates a histogram plot.

It also generates a summary statistics file 'length_stats.txt' containing
distribution metrics for passed and failed sequences separately.

The plots are saved as PNG files in the {TARGET_DIR}/seq_len_plots/
directory, named like *_seqs_lengths.png. The output directory
is created if it does not exist.

Configuration is done via constants at the top of the script.
"""


# --- Imports ---
import os
import glob
import numpy as np
from Bio import SeqIO
import matplotlib.pyplot as plt



# --- Constants ---
TARGET_DIR = "/mnt/secondary/micro16s_dbs/16s_databases/m16s_db_004/seqs/"
SHOW_PLOTS = False # Display plots interactively?
# Histogram bin widths for sequence lengths
BIN_SIZE_FULL_SEQ = 20       # For files with "FULL" in filename
BIN_SIZE_OTHER_SEQS = 2      # For files without "FULL" in filename



# --- Script ---
def calculate_stats(lengths):
    """Calculates statistics for a list of sequence lengths."""
    if not lengths:
        return [0] * 11 # Return zeros if no sequences
    
    # Convert to numpy array for efficient percentile calculation
    lengths_arr = np.array(lengths)
    
    stats = [
        np.min(lengths_arr),
        np.percentile(lengths_arr, 1),
        np.percentile(lengths_arr, 5),
        np.percentile(lengths_arr, 10),
        np.percentile(lengths_arr, 25),
        np.median(lengths_arr), # 50p
        np.percentile(lengths_arr, 75),
        np.percentile(lengths_arr, 90),
        np.percentile(lengths_arr, 95),
        np.percentile(lengths_arr, 99),
        np.max(lengths_arr)
    ]
    return stats

def plot_sequence_lengths(fasta_file, output_dir, bin_size, show_plot):
    """Reads a FASTA file, calculates sequence lengths, plots a histogram, and returns lengths."""
    lengths = []
    try:
        for record in SeqIO.parse(fasta_file, "fasta"):
            lengths.append(len(record.seq))
    except FileNotFoundError:
        print(f"Error: Input file not found: {fasta_file}")
        return None
    except Exception as e:
        print(f"Error reading {fasta_file}: {e}")
        return None

    if not lengths:
        print(f"Warning: No sequences found in {fasta_file}")
        return []

    num_sequences = len(lengths)
    base_filename = os.path.basename(fasta_file)
    output_filename = os.path.splitext(base_filename)[0] + "_lengths.png"
    output_path = os.path.join(output_dir, output_filename)

    # Determine bins
    min_len = min(lengths)
    max_len = max(lengths)
    bins = range(min_len, max_len + bin_size, bin_size)

    # Create plot
    plt.figure(figsize=(10, 6))
    plt.hist(lengths, bins=bins, edgecolor='black')
    plt.title(f'Sequence Length Distribution ({num_sequences} seqs)\n{base_filename}')
    plt.xlabel('Sequence Length (bp)')
    plt.ylabel('Number of Sequences')
    plt.grid(axis='y', alpha=0.75)

    # Save plot
    try:
        plt.savefig(output_path)
        print(f"Saved plot: {output_path}")
    except Exception as e:
        print(f"Error saving plot {output_path}: {e}")


    # Show plot if requested
    if show_plot:
        plt.show()

    plt.close() # Close the figure to free memory
    
    return lengths



if __name__ == "__main__":
    output_dir = os.path.join(TARGET_DIR, "seq_len_plots")
    input_pattern = os.path.join(TARGET_DIR, "*_seqs.fasta")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Find and process FASTA files
    fasta_files = glob.glob(input_pattern)

    if not fasta_files:
        print(f"No FASTA files found matching pattern: {input_pattern}")
    else:
        print(f"Found {len(fasta_files)} FASTA files to process.")
        
        # Dictionaries to store stats
        passed_stats = []
        failed_stats = []

        for fast_file in sorted(fasta_files):
            bin_size = BIN_SIZE_FULL_SEQ if "FULL" in fast_file else BIN_SIZE_OTHER_SEQS
            print(f"Processing: {fast_file}")
            lengths = plot_sequence_lengths(fast_file, output_dir, bin_size, SHOW_PLOTS)
            
            if lengths is not None:
                # Determine region and status
                filename = os.path.basename(fast_file)
                name_no_ext = filename.replace("_seqs.fasta", "")
                
                if "failed" in name_no_ext:
                    # Handle case where failed might be prefix or somewhere else if needed, 
                    # but assuming suffix based on typical patterns. 
                    if name_no_ext.endswith("_failed"):
                         region = name_no_ext[:-7]
                    else:
                         region = name_no_ext.replace("failed", "").strip("_")
                    
                    stats = calculate_stats(lengths)
                    failed_stats.append((region, stats))
                else:
                    region = name_no_ext
                    stats = calculate_stats(lengths)
                    passed_stats.append((region, stats))

        print("Finished processing all files.")
        
        # Write stats to file
        stats_file_path = os.path.join(TARGET_DIR, "length_stats.txt")
        print(f"Writing stats to {stats_file_path}...")
        
        with open(stats_file_path, "w") as f:
            # Header for Passed
            f.write("Passed Sequence Length Stats:\n")
            f.write("region,min,1p,5p,10p,25p,50p,75p,90p,95p,99p,max\n")
            for region, s in passed_stats:
                # Format stats to 1 decimal place if float, or int if int
                stats_str = ",".join([f"{x:.1f}" for x in s])
                f.write(f"{region},{stats_str}\n")
            
            f.write("\n\n")
            
            # Header for Failed
            f.write("Failed Sequence Length Stats:\n")
            f.write("region,min,1p,5p,10p,25p,50p,75p,90p,95p,99p,max\n")
            for region, s in failed_stats:
                stats_str = ",".join([f"{x:.1f}" for x in s])
                f.write(f"{region},{stats_str}\n")
                
        print("Done.")
