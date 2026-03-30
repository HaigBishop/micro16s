
# Micro16S
Micro16S is a small neural network trained to embed 16S rRNA genes into dense vectors which hold phylogenetic/taxonomic identity.

**Preprint:** https://www.biorxiv.org/content/10.64898/2026.03.21.713432v1

### Background
All bacteria (and archaea) have their own variant of the 16S rRNA gene. For years microbiologists have used this gene as a way to identify bacteria. Closely related species will have more similar sequences. These DNA sequences hold useful information regarding a bacterium's phylogenetic identity, but in a form (DNA sequence) which is difficult to give to machine learning models. 

By representing these sequences as small information-dense vectors, they are more consumable to machine learning algorithms. Furthermore, in practice only have small segments of the 16S rRNA gene are sequenced, so when different datasets sequence different regions they are much less compatible. But by projecting any segment into the same embedding space, different data will be represented consistently.


### Training Micro16S
The Micro16S neural network takes as input a segment of DNA and outputs a vector called an embedding. The space in which Micro16S projects 16S genes is supposed to closely resemble phylogenetic identity. Since taxonomy is related to this, we might expect that bacteria from the same genus will cluster together, bacteria from the same family will cluster together, and so on. 


### Training data
To train Micro16S we need full length 16S rRNA sequences and some kind of phylogenetic information source. We use the Genome Taxonomy Database (GTDB), which gives full length 16S sequences and taxonomic labels. GTDB's taxonomy is derived from genomic-level information, which we utilise.

When training Micro16S we need a signal of how closely the bacteria are phylogeneticly related to each other. We use two methods here: 
1) **Pair loss** utilises phylogenetic distances derived from the GTDB tree. Given any pair of bacteria, we can calculate their distance. The model is trained to embed 16S genes to hold these relative distances in the embedding space. 
2) **Triplet loss** utilises the taxonomic labels of all 16S sequences to encourage hierachical clustering of the embeddings. Triplets are selected in which two sequences are the same label (e.g. Bacillus), and the other one is a different label (e.g. Virgibacillus). In training, the model is encouraged to position the two Bacillus sequences closer together relative to the Virgibacillus seqeunce.


### The Micro16S models
There were two Micro16S models developed for the micro16s preprint: m16s_001 and m16s_002. m16s_001 was trained using a train-test-excluded taxa split, while m16s_002 only had a train set, utilising all available sequences.


## Full Usage
These are the steps for training of a new Micro16S model.

1. Clone Repo
    ```bash
    git clone https://github.com/HaigBishop/micro16s.git
    ```
2. Install Dependencies
    ```bash
    conda create -n m16s_env python=3.12
    conda activate m16s_env
    
    # Install pytorch (https://pytorch.org/get-started/locally/)
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
    # Install other dependencies
    conda install -c conda-forge biopython hmmer matplotlib seaborn numpy pandas tqdm scikit-learn scipy numba joblib pyyaml networkx psutil
    pip install xgboost umap-learn imbalanced-learn composition-stats sklearn-compat
    ```
3. Get Processed GTDB Trees (redvals)
  - The `micro16s` repository relies on the `redvals` Python package (only on GitHub)
  - Simply cloning the repository will supply the newest GTDB phylogenetic trees with precomputed RED values and many useful methods
    ```bash
    git clone https://github.com/HaigBishop/redvals.git
    ```
4. Filter the GTDB 16S Database
  - Run `filter_database.py` to take a GTDB 16S rRNA database (e.g. `ssu_all_r226.fna`) resulting in a filtered down version (e.g. `ssu_all_r226_filtered.fna`).
  - The latest version of the GTDB 16S gene database can be downloaded from https://gtdb.ecogenomic.org/downloads under `release226/226.0/genomic_files_all`
 - Filtering of the 16S sequences is based on their: 
    1) Presence in trees (from `redvals`)
    2) Taxonomic resolution
    3) Duplicate sequences
    4) Sequence lengths
    5) Maximum taxon sizes (taxa are downsampled)
    6) Minimum taxon sizes
    7) Maximum genes per genome (configurable cap; default keeps the first sequence seen per genome, or disable with `MAX_GENES_PER_GENOME=None`)
  - The generated `*_filtered_about.txt` report now lists filtering stats, length stats, and final counts for the combined dataset as well as for Bacteria and Archaea separately.
5. Extract Variable 16S Regions
  - You can use the `extract16s` tool to get desired regions from full length 16S genes.
    ```bash
    git clone https://github.com/HaigBishop/extract16s.git
    ```
  - The `extract16s` tool requires a `.truncspec` file that specifies truncation for the desired regions (e.g. V3-V4 and V4). You can create a `.truncspec` file using the `asvs2truncspec` tool in the `extract16s` repository. See the `docs/` folder and the `README.md` in the `extract16s` repository for more information.
  - This example truncates at V4 with 30bp of padding. 
    ```bash
    bash /home/haig/Repos/micro16s/extract16s/Scripts/extract16s.sh \
        /mnt/secondary/micro16s_dbs/16s_databases/m16s_db_001/filtered/ssu_all_r226_filtered.fna \
        /home/haig/Repos/micro16s/extract16s/InputData/bac_16s.hmm \
        /home/haig/Repos/micro16s/extract16s/InputData/arc_16s.hmm \
        /home/haig/Repos/micro16s/extract16s/InputData/trunc_micro16s.truncspec \
        --inter_dir /mnt/secondary/micro16s_dbs/16s_databases/m16s_db_001/extract16s_intermediates/ \
        --out_dir /mnt/secondary/micro16s_dbs/16s_databases/m16s_db_001/seqs/ --add_indices \
        --trunc_padding 30 --no_filter_ambiguous_for_full
    ```
  - This results in files: `FULL_seqs.fasta`, `V3_seqs.fasta`, `V3-V4_seqs.fasta`
6. Plot 16S Gene Sequence Lengths (Optional)
  - Now is a good time to use `plot_seq_lengths.py` to investigate the lengths of the extracted sequences.
7. Encode 16S Gene Sequences
  - Run `encode_seqs.py` to take the 16S sequences and create `.npy` files which contain the encoded sequences. It can encode them as 3-bit and/or K-mers.
  - Inputs:
    - `/mnt/secondary/micro16s_dbs/16s_databases/m16s_db_001/seqs/FULL_seqs.fasta`
    - `/mnt/secondary/micro16s_dbs/16s_databases/m16s_db_001/seqs/V3_seqs.fasta`
    - `/mnt/secondary/micro16s_dbs/16s_databases/m16s_db_001/seqs/V3-V4_seqs.fasta`
  - Outputs:
    - `/mnt/secondary/micro16s_dbs/16s_databases/m16s_db_001/seqs/encoded/3bit_seq_reps.npy`
    - `/mnt/secondary/micro16s_dbs/16s_databases/m16s_db_001/seqs/encoded/about_3bit_encodings.txt`
    - `/mnt/secondary/micro16s_dbs/16s_databases/m16s_db_001/seqs/encoded/{K}-mer_seq_reps.npy`
    - `/mnt/secondary/micro16s_dbs/16s_databases/m16s_db_001/seqs/encoded/about_{K}-mer_encodings.txt`
8. Construct a Micro16S Dataset
  - Run `construct_dataset.py` to split the 16S sequences into a test set, a train set and an "excluded taxa" set. For each set, we get the indices for all sequences assigned to that set as files like `testing_indices.txt`. We also get "taxonomic objects" for each set which are Python pickle files (mostly dictionaries) that hold the information regarding taxonomies of the sequences in that set. These .pkl files are in the `tax_objs/` subdirectory. We also get "label arrays" for each set which are NumPy arrays (.npy files) containing precomputed label information like pairwise taxonomic ranks. These .npy files are in the `labels/` subdirectory.
9. Train a Micro16S Model
  - To train a model, use `train.py` which relies on:
    - `micro16s_dataset_loader.py` - Loads and validates dataset files into shared global variables
    - `globals_config.py` - Defines shared global variables used across the codebase
    - `triplet_pair_mining.py` - Handles mining of sequence triplets/pairs for training
    - `quick_test.py` - Evaluates model performance through clustering and classification
    - `generate_seq_variants.py` - Creates sequence variants through mutations and truncations
    - `model.py` - Core model architectures and loss functions


## Contact

**Haig Bishop**:   haigvbishop@gmail.com
