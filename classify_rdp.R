# classify_rdp.R
#
# Classify sequences using DADA2 taxonomy assignment
#
# Args:
#   query_file: path to query sequences (test)
#   ref_fasta: path to reference sequences (train)
#   output_file: path to output CSV
#   batch_size: batch size for assignTaxonomy
#
# This is essentially a wrapper around the DADA2 assignTaxonomy function to be used from Python.
# It takes a test and a train set in the form of 2 FASTA files and outputs a CSV file with the taxonomy assignments.
# The CSV file has the following columns:
#   - ASV_ID: the ASV identifier from the FASTA header (e.g., "42_V4-004")
#   - Seq_Index: the sequence index (e.g., 42)
#   - Region_Train: always "FULL" for RDP output
#   - Region_Test: the region name (e.g., "V4-004")
#   - Domain: the domain of the sequence
#   - Phylum: the phylum of the sequence
#   - Class: the class of the sequence
#   - Order: the order of the sequence
#   - Family: the family of the sequence
#   - Genus: the genus of the sequence
#   - Species: the species of the sequence
#   - Domain_Boot: the bootstrap confidence score for Domain classification [0-100]
#   - Phylum_Boot: the bootstrap confidence score for Phylum classification [0-100]
#   - Class_Boot: the bootstrap confidence score for Class classification [0-100]
#   - Order_Boot: the bootstrap confidence score for Order classification [0-100]
#   - Family_Boot: the bootstrap confidence score for Family classification [0-100]
#   - Genus_Boot: the bootstrap confidence score for Genus classification [0-100]
#   - Species_Boot: the bootstrap confidence score for Species classification [0-100]
# 
# Output CSV Example:
# "ASV_ID","Seq_Index","Region_Train","Region_Test","Domain","Phylum","Class","Order","Family","Genus","Species","Domain_Boot","Phylum_boot","Class_boot","Order_boot","Family_boot","Genus_boot","Species_boot"
# "3_V3-V5-001",3,"FULL","V3-V5-001","d__Bacteria","p__Pseudomonadota","c__Gammaproteobacteria","o__Enterobacterales_A","f__Enterobacteriaceae_A","g__Wigglesworthia","s__Wigglesworthia glossinidia_A",100,100,100,100,100,100,100
# "3_V4-001",3,"FULL","V4-001","d__Bacteria","p__Pseudomonadota","c__Gammaproteobacteria","o__Enterobacterales_A","f__Enterobacteriaceae_A","g__Wigglesworthia","s__Wigglesworthia glossinidia_A",82,82,82,82,82,82,82
# "3_V4-002",3,"V4-002","d__Bacteria","p__Pseudomonadota","c__Gammaproteobacteria","o__Enterobacterales_A","f__Enterobacteriaceae_A","g__Wigglesworthia","s__Wigglesworthia glossinidia_A",100,100,100,100,100,100,100
# "3_V4-003",3,"V4-003","d__Bacteria","p__Pseudomonadota","c__Gammaproteobacteria","o__Enterobacterales_A","f__Enterobacteriaceae_A","g__Wigglesworthia","s__Wigglesworthia glossinidia_A",100,100,100,100,100,100,100
# ...


# Load DADA2 and Biostrings quietly
suppressPackageStartupMessages(library(dada2))
suppressPackageStartupMessages(library(Biostrings))

# Grab arguments from Python
args <- commandArgs(trailingOnly = TRUE)
query_file <- args[1] # First arg: query sequences (test)
ref_fasta <- args[2] # Second arg: reference sequences (train)
output_file <- args[3] # Third arg: output CSV path
batch_size <- as.integer(args[4]) # Fourth arg: batch size


# =============================================================================
# Helper Functions
# =============================================================================

parse_fasta_headers <- function(fasta_path) {
    # Parse FASTA headers to extract ASV_ID, Seq_Index, and Region.
    # Header format: >{index_region} ...  (e.g., ">{42_V4-004} d__Bacteria;...")
    #
    # Returns a data frame with columns: ASV_ID, Seq_Index, Region
    
    lines <- readLines(fasta_path)
    headers <- lines[grepl("^>", lines)]
    
    # Extract ASV_ID (the part in braces NOT including the braces)
    asv_ids <- sub("^>\\{([^}]+)\\}.*", "\\1", headers)
    
    # Extract the content inside braces (without braces)
    inner <- sub("^\\{(.*)\\}$", "\\1", asv_ids)
    
    # Split into Seq_Index and Region (format: "index_region")
    # Use sub to split on first underscore only (region can contain underscores)
    seq_indices <- as.integer(sub("_.*", "", inner))
    regions <- sub("^[0-9]+_", "", inner)
    
    return(data.frame(
        ASV_ID = asv_ids,
        Seq_Index = seq_indices,
        Region = regions,
        stringsAsFactors = FALSE
    ))
}


# =============================================================================
# Main Execution
# =============================================================================

# Run the taxonomy assignment
tryCatch({
    # Parse metadata from query FASTA headers
    metadata <- parse_fasta_headers(query_file)
        
    # Read the query sequences as DNAStringSet
    dna <- readDNAStringSet(query_file)
    seqs <- as.character(dna)

    # Make sure the names are just the first token in the FASTA header
    seq_names <- sub(" .*", "", names(dna))
    names(seqs) <- seq_names

    # Process in batches
    n_seqs <- length(seqs)
    
    # Initialize lists to store results
    all_taxa <- list()
    all_boot <- list()
    
    # Loop through batches
    # If batch_size is invalid or NA, process all at once (fallback)
    if (is.na(batch_size) || batch_size <= 0) {
        batch_size <- n_seqs
    }

    for (i in seq(1, n_seqs, by = batch_size)) {
        end_idx <- min(i + batch_size - 1, n_seqs)
        batch_seqs <- seqs[i:end_idx]
        
        message(sprintf("Processing sequences %d to %d of %d...", i, end_idx, n_seqs))
        
        # Assign taxonomy using the naive Bayesian classifier
        res <- assignTaxonomy(seqs = batch_seqs, refFasta = ref_fasta, multithread = TRUE, outputBootstraps = TRUE, minBoot = 0)
        
        all_taxa[[length(all_taxa) + 1]] <- res$tax
        all_boot[[length(all_boot) + 1]] <- res$boot
    }
    
    # Combine results
    taxa <- do.call(rbind, all_taxa)
    boot <- do.call(rbind, all_boot)
    
    # Convert to data frame
    df <- as.data.frame(taxa)
    df_boot <- as.data.frame(boot)
    colnames(df_boot) <- paste0(colnames(df_boot), "_Boot")
    df <- cbind(df, df_boot)
    
    # Rename Kingdom to Domain
    if("Kingdom" %in% colnames(df)) {
        colnames(df)[colnames(df) == "Kingdom"] <- "Domain"
    }
    if("Kingdom_Boot" %in% colnames(df)) {
        colnames(df)[colnames(df) == "Kingdom_Boot"] <- "Domain_Boot"
    }

    # Clean Domain column: remove prefix before d__
    if("Domain" %in% colnames(df)) {
        df$Domain <- sub("^.*(d__.*)$", "\\1", df$Domain)
    }

    # Clean Species column: remove bracketed metadata
    if("Species" %in% colnames(df)) {
        df$Species <- sub("\\s*\\[.*$", "", df$Species)
    }
    
    # Rename Region column to Region_Test
    colnames(metadata)[colnames(metadata) == "Region"] <- "Region_Test"
    
    # Add Region_Train column (always "FULL" for RDP)
    metadata$Region_Train <- "FULL"
    
    # Reorder metadata columns
    metadata <- metadata[, c("ASV_ID", "Seq_Index", "Region_Train", "Region_Test")]

    # Combine metadata columns with taxonomy
    df <- cbind(metadata, df)
    
    # Write the results to a CSV file without row names and without quotes
    write.csv(df, output_file, row.names = FALSE, quote = FALSE)
# Handle errors
}, error = function(e) {
    # Write error to stderr
    write(conditionMessage(e), stderr())
    # Exit with error status
    q(status = 1)
})
