import pandas as pd
import os
import gzip
import logging
import concurrent.futures
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from tqdm import tqdm
import numpy as np
import time
import re
import subprocess
import sys

# Initialize logging for console
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Create a 'logs' directory if it doesn't exist
os.makedirs('logs/', exist_ok=True)
# -----------------------------
# Configuration and Setup
# -----------------------------

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# Read the Excel file
metadata_file = '241111_lam_tfx_conditions.xlsx'
df = pd.read_excel(metadata_file)

# Check if required columns are present
required_columns = ['filename', 'condition', 'rep', 'payload_att_site']
if not all(col in df.columns for col in required_columns):
    raise ValueError(f"Excel file must contain the following columns: {required_columns}")

# Define attB and attP sequences
attB_sequence_full = (
    "cgggccggcttgtcgacgacggcggtctccgtcgtcaggatcatccgggctaccggtcgccaccatgccc"
    "gccatgaagatcgagtgccgcatcaccggcaccctgaacggcgtggagttcgagctggtgggcggcggag"
    "agggcaccccc"
)


attP_sequence_full = (
    "gtggtttgtctggtcaaccaccgcggtctcagtggtgtacggtacaaacccagctaccggtcgccaccat"
    "gcccgccatgaagatcgagtgccgcatcaccggcaccctgaacggcgtggagttcgagctggtgggcggc"
    "ggagagggcacccccgagcagggccg"
)

attB_sequence_middle = 'asefhuioerfasdhfiuyasdgf'  # Middle sequence for attB
attB_sequence_left = 'asefhuioerfiaurfhierufh'
attB_sequence_right = attP_sequence_full[3:25] + attB_sequence_full[24:48]

attP_sequence_middle = 'asefhuioerfakiufhifur' # Middle sequence for attP
attP_sequence_left = 'asefhuioerfuerhfiquafh'
attP_sequence_right = attB_sequence_full[3:24] + attP_sequence_full[25:51]

raw_data_dir = 'raw_data/'

# **Modification Start:** Handle only R1 files and ignore R2 files
# Assume that R1 files have '_R1_' in their filenames

def get_sample_prefix(filename):
    """
    Extracts the sample prefix from the filename by removing the '_R1_001.fastq.gz' suffix.

    Parameters:
    - filename (str): The FASTQ filename.

    Returns:
    - str: The sample prefix.
    """
    return re.sub(r'_R1_\d+\.fastq\.gz$', '', filename)

# Create a list of R1 file paths
r1_file_paths = []
for fname in df['filename'].unique():
    if re.search(r'_R1_\d+\.fastq\.gz$', fname):
        r1_file_paths.append(os.path.join(raw_data_dir, fname))

# Log a warning if no R1 files are found
if not r1_file_paths:
    logging.warning("No R1 FASTQ.gz files found in the metadata.")

# -----------------------------
# Precompute All Sequence Orientations
# -----------------------------

def generate_orientations(seq):
    """
    Generate all orientations (forward, reverse, complement, reverse complement) of a given sequence.

    Parameters:
    - seq (str): The input sequence.

    Returns:
    - dict: A dictionary mapping orientation names to sequences.
    """
    seq_obj = Seq(seq)
    return {
        'forward': str(seq_obj),
        'reverse': str(seq_obj.reverse_complement()[::-1]),
        'complement': str(seq_obj.complement()),
        'reverse_complement': str(seq_obj.reverse_complement())
    }

# Generate orientations for all target sequences
attB_middle_orientations = generate_orientations(attB_sequence_middle)
attP_middle_orientations = generate_orientations(attP_sequence_middle)

attB_left_orientations = generate_orientations(attB_sequence_left)
attB_right_orientations = generate_orientations(attB_sequence_right)

attP_left_orientations = generate_orientations(attP_sequence_left)
attP_right_orientations = generate_orientations(attP_sequence_right)

# **Modification Start:** Define adapter_bridge and its reverse complement
adapter_bridge = 'GACTATAGGGCACGCGTGGACAGAT'
adapter_bridge_rev_comp = str(Seq(adapter_bridge).reverse_complement())

# Convert target sequences to uppercase NumPy arrays for efficient comparison
def seq_to_np(seq):
    return np.array(list(seq.upper()), dtype='<U1')

attB_middle_np = {ori: seq_to_np(seq) for ori, seq in attB_middle_orientations.items()}
attP_middle_np = {ori: seq_to_np(seq) for ori, seq in attP_middle_orientations.items()}

attB_left_np = {ori: seq_to_np(seq) for ori, seq in attB_left_orientations.items()}
attB_right_np = {ori: seq_to_np(seq) for ori, seq in attB_right_orientations.items()}

attP_left_np = {ori: seq_to_np(seq) for ori, seq in attP_left_orientations.items()}
attP_right_np = {ori: seq_to_np(seq) for ori, seq in attP_right_orientations.items()}

# Threshold for weighted mismatches (adjust based on your data quality and requirements)
weighted_mismatch_threshold = 20  # Example threshold

# -----------------------------
# Helper Functions
# -----------------------------

def check_bowtie2_installed():
    """
    Check if Bowtie2 is installed. If not, attempt to install it.
    """
    try:
        subprocess.run(['bowtie2', '--version'], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        logging.info("Bowtie2 is already installed.")
    except (subprocess.CalledProcessError, FileNotFoundError):
        logging.info("Bowtie2 is not installed. Attempting to install Bowtie2...")
        install_bowtie2()

def install_bowtie2():
    """
    Install Bowtie2 using conda or prompt the user to install it manually.
    """
    try:
        # Attempt to install Bowtie2 via conda
        subprocess.run(['conda', 'install', '-y', 'bowtie2'], check=True)
        logging.info("Bowtie2 installed successfully via conda.")
    except (subprocess.CalledProcessError, FileNotFoundError):
        logging.error("Failed to install Bowtie2 via conda. Please install Bowtie2 manually.")
        sys.exit(1)

def download_reference_genome(ref_genome_path='GRCh38.primary_assembly.genome.fa'):
    """
    Download the human reference genome (GRCh38) from Ensembl if not already present.

    Parameters:
    - ref_genome_path (str): Path to save the reference genome.

    Returns:
    - str: Path to the downloaded reference genome.
    """
    if os.path.exists(ref_genome_path):
        logging.info(f"Reference genome already exists at {ref_genome_path}.")
        return ref_genome_path

    logging.info("Downloading the human reference genome (GRCh38)...")
    # URL for GRCh38 from Ensembl
    url = "ftp://ftp.ensembl.org/pub/release-106/fasta/homo_sapiens/dna/Homo_sapiens.GRCh38.dna.primary_assembly.fa.gz"
    compressed_path = ref_genome_path + ".gz"

    try:
        subprocess.run(['wget', '-O', compressed_path, url], check=True)
        logging.info("Download completed. Unzipping the reference genome...")
        subprocess.run(['gunzip', compressed_path], check=True)
        logging.info(f"Reference genome saved to {ref_genome_path}.")
        return ref_genome_path
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to download the reference genome: {e}")
        sys.exit(1)

def build_bowtie2_index(ref_genome_path, index_prefix='GRCh38_index'):
    """
    Build Bowtie2 index for the reference genome.

    Parameters:
    - ref_genome_path (str): Path to the reference genome FASTA file.
    - index_prefix (str): Prefix for the Bowtie2 index files.

    Returns:
    - str: Prefix path of the Bowtie2 index.
    """
    index_files = [f"{index_prefix}.{ext}" for ext in ['1.bt2', '2.bt2', '3.bt2', '4.bt2', 'rev.1.bt2', 'rev.2.bt2']]
    if all(os.path.exists(f) for f in index_files):
        logging.info(f"Bowtie2 index already exists with prefix '{index_prefix}'.")
        return index_prefix

    logging.info(f"Building Bowtie2 index for {ref_genome_path}...")
    try:
        subprocess.run(['bowtie2-build', ref_genome_path, index_prefix], check=True)
        logging.info(f"Bowtie2 index built with prefix '{index_prefix}'.")
        return index_prefix
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to build Bowtie2 index: {e}")
        sys.exit(1)

def calculate_weighted_mismatch(seq_str, qual_scores, target_seq_np_dict, threshold):
    """
    Check if any orientation of the target sequence matches within the mismatch threshold.

    Parameters:
    - seq_str (str): The sequence string.
    - qual_scores (list of int): Phred quality scores.
    - target_seq_np_dict (dict): Dictionary mapping orientation to target sequence NumPy arrays.
    - threshold (int): Weighted mismatch threshold.

    Returns:
    - list of dict: Each dict contains 'orientation', 'start', 'end'.
    """
    matches = []
    seq_len = len(seq_str)
    qual_array = np.array(qual_scores, dtype=np.int32)
    seq_array = np.array(list(seq_str.upper()), dtype='<U1')

    for orientation, target_seq_np in target_seq_np_dict.items():
        target_len = len(target_seq_np)
        if seq_len < target_len:
            continue

        # Generate all possible windows using sliding_window_view (requires NumPy >= 1.20)
        try:
            from numpy.lib.stride_tricks import sliding_window_view
            windows = sliding_window_view(seq_array, window_shape=target_len)
            qual_windows = sliding_window_view(qual_array, window_shape=target_len)
        except AttributeError:
            # For older NumPy versions, use a list comprehension (less efficient)
            windows = np.array([seq_array[i:i+target_len] for i in range(seq_len - target_len + 1)])
            qual_windows = np.array([qual_array[i:i+target_len] for i in range(seq_len - target_len + 1)])

        # Compare windows to target sequence
        mismatches = (windows != target_seq_np)  # Boolean array: True where mismatches occur

        # Calculate weighted mismatches: sum of Phred scores where mismatches occur
        mismatch_int = mismatches.astype(int)
        weighted_mismatch = np.sum(qual_windows * mismatch_int, axis=1)  # Sum along each window

        # Find windows where weighted mismatch <= threshold
        matching_indices = np.where(weighted_mismatch <= threshold)[0]

        for idx in matching_indices:
            match_info = {
                'orientation': orientation,
                'start': idx,
                'end': idx + len(target_seq_np)
            }
            matches.append(match_info)

    return matches

def find_matching_positions(seq_str, qual_scores, target_seq_np_dict, threshold):
    """
    Find all matching positions of target sequences in all orientations within the mismatch threshold.

    Parameters:
    - seq_str (str): The sequence string.
    - qual_scores (list of int): Phred quality scores.
    - target_seq_np_dict (dict): Dictionary mapping orientation to target sequence NumPy arrays.
    - threshold (int): Weighted mismatch threshold.

    Returns:
    - list of dict: Each dict contains 'orientation', 'start', 'end'.
    """
    return calculate_weighted_mismatch(seq_str, qual_scores, target_seq_np_dict, threshold)

def chunked_iterable_single(read_sequences, read_qualities, chunk_size):
    """
    Yield successive chunks of single-ended sequences and qualities along with their starting index.

    Parameters:
    - read_sequences (list of str): List of read sequence strings.
    - read_qualities (list of list of int): List of read quality scores.
    - chunk_size (int): The number of reads per chunk.

    Yields:
    - tuple: (start_idx, list of sequences, list of qualities)
    """
    for i in range(0, len(read_sequences), chunk_size):
        yield (
            i + 1,  # Assuming 1-based indexing
            read_sequences[i:i+chunk_size],
            read_qualities[i:i+chunk_size]
        )

def calculate_a_s(a, orientation):
    """
    Calculate a*s based on att site type and sequence orientation.

    Parameters:
    - a (int): Att site type (+1 for left, -1 for right).
    - orientation (str): Orientation of the sequence ('forward', 'reverse', etc.).

    Returns:
    - int: Product of a and s.
    """
    if orientation in ['forward', 'complement']:
        s = 1
    elif orientation in ['reverse', 'reverse_complement']:
        s = -1
    else:
        s = 1  # Default to +1 if unknown orientation
    return a * s

# **Modification Start:** Update the processing function to handle adapter_bridge trimming
def process_single_reads_chunks(read_start_idx, read_sequences, read_qualities, 
                                attB_middle_np, attP_middle_np, threshold, chunk_index, 
                                payload_att_site, 
                                attB_left_np, attB_right_np, attP_left_np, attP_right_np,
                                fasta_filename):
    """
    Process a chunk of single-ended FASTQ records, filter reads based on middle sequences,
    then filter based on left/right sequences according to payload_att_site with orientation tracking
    and extract additional 24 bp sequences for mapping.

    Additionally trims reads containing the adapter_bridge from the adapter bridge onwards.

    Parameters:
    - read_start_idx (int): Starting index of the reads in the original FASTQ file.
    - read_sequences (list of str): List of read sequence strings.
    - read_qualities (list of list of int): List of read quality scores.
    - attB_middle_np (dict): Middle target sequences for attB in all orientations.
    - attP_middle_np (dict): Middle target sequences for attP in all orientations.
    - threshold (int): Weighted mismatch threshold.
    - chunk_index (int): Index of the current chunk.
    - payload_att_site (str): 'attB' or 'attP' indicating which left/right to search for.
    - attB_left_np (dict): Left target sequences for attB in all orientations.
    - attB_right_np (dict): Right target sequences for attB in all orientations.
    - attP_left_np (dict): Left target sequences for attP in all orientations.
    - attP_right_np (dict): Right target sequences for attP in all orientations.
    - fasta_filename (str): Name of the FASTA file without extensions.

    Returns:
    - tuple: (filtered_out_middle_count, kept_reads_half_site_count, filtered_reads, extracted_sequences, trimmed_adapter_bridge_count, trimmed_adapter_bridge_bases)
    """
    filtered_reads = []
    filtered_out_middle = 0
    kept_reads_half_site = 0
    extracted_sequences = []
    
    # Initialize counters for adapter bridge trimming
    trimmed_adapter_bridge_count = 0
    trimmed_adapter_bridge_bases = 0
    
    # Determine which target sequences to use
    if payload_att_site == 'attB':
        target_middle_np_dict = attB_middle_np
        target_left_np_dict = attB_left_np
        target_right_np_dict = attB_right_np
    elif payload_att_site == 'attP':
        target_middle_np_dict = attP_middle_np
        target_left_np_dict = attP_left_np
        target_right_np_dict = attP_right_np
    else:
        logging.error(f"Unknown payload_att_site '{payload_att_site}'. Skipping this chunk.")
        return 0, 0, [], [], 0, 0
    
    for i, (seq, qual) in enumerate(zip(read_sequences, read_qualities)):
        current_row = read_start_idx + i  # Original row number
        
        # **Modification:** Trim sequences containing the adapter_bridge
        pos_forward = seq.find(adapter_bridge)
        pos_reverse = seq.find(adapter_bridge_rev_comp)
        
        if pos_forward != -1 and (pos_reverse == -1 or pos_forward < pos_reverse):
            # Adapter bridge found in forward orientation
            trimmed_seq = seq[:pos_forward]
            trimmed_qual = qual[:pos_forward]
            bases_trimmed = len(seq) - pos_forward
            trimmed_adapter_bridge_count += 1
            trimmed_adapter_bridge_bases += bases_trimmed
        elif pos_reverse != -1:
            # Adapter bridge found in reverse complement orientation
            trimmed_seq = seq[:pos_reverse]
            trimmed_qual = qual[:pos_reverse]
            bases_trimmed = len(seq) - pos_reverse
            trimmed_adapter_bridge_count += 1
            trimmed_adapter_bridge_bases += bases_trimmed
        else:
            # No adapter bridge found; keep the entire read
            trimmed_seq = seq
            trimmed_qual = qual
        
        # Step 1: Filter out middle sequences
        middle_matches = find_matching_positions(trimmed_seq, trimmed_qual, target_middle_np_dict, threshold)
        if middle_matches:
            filtered_out_middle += 1
            continue  # Skip this read
        
        # Step 2: Filter based on left/right sequences
        # Initialize flags and positions
        contains_left_or_right = False
        positions = []
        
        # Check for left sequences
        left_matches = find_matching_positions(trimmed_seq, trimmed_qual, target_left_np_dict, threshold)
        for match in left_matches:
            contains_left_or_right = True
            positions.append({
                'type': 'left',
                'orientation': match['orientation'],
                'start': match['start'],
                'end': match['end'],
                'payload_att_site': payload_att_site
            })
            # Determine 'a' and 's'
            a = 1  # 'left' corresponds to a=+1
            orientation = match['orientation']
            a_s = calculate_a_s(a, orientation)
            
            # Determine extraction coordinates based on a*s
            if a_s < 0:
                extract_start = max(match['start'] - 50, 0)
                extract_end = match['start']
            else:
                extract_start = match['end']
                extract_end = min(match['end'] + 50, len(trimmed_seq))
            
            extracted_seq = trimmed_seq[extract_start:extract_end]
            if len(extracted_seq) > 0:
                extracted_sequences.append({
                    'read_id': f"{fasta_filename}_row{current_row}",
                    'sequence': extracted_seq,
                    'orientation': orientation,
                    'original_read': 'R1',
                    'payload_att_site': payload_att_site
                })
        
        # Check for right sequences
        right_matches = find_matching_positions(trimmed_seq, trimmed_qual, target_right_np_dict, threshold)
        for match in right_matches:
            contains_left_or_right = True
            positions.append({
                'type': 'right',
                'orientation': match['orientation'],
                'start': match['start'],
                'end': match['end'],
                'payload_att_site': payload_att_site
            })
            # Determine 'a' and 's'
            a = -1  # 'right' corresponds to a=-1
            orientation = match['orientation']
            a_s = calculate_a_s(a, orientation)
            
            # Determine extraction coordinates based on a*s
            if a_s < 0:
                extract_start = max(match['start'] -50, 0)
                extract_end = match['start']
            else:
                extract_start = match['end']
                extract_end = min(match['end'] + 50, len(trimmed_seq))
            
            extracted_seq = trimmed_seq[extract_start:extract_end]
            if len(extracted_seq) > 0:
                extracted_sequences.append({
                    'read_id': f"{fasta_filename}_row{current_row}",
                    'sequence': extracted_seq,
                    'orientation': orientation,
                    'original_read': 'R1',
                    'payload_att_site': payload_att_site
                })
        
        # Decide whether to keep the read
        if contains_left_or_right:
            kept_reads_half_site += 1
            # Create SeqRecord
            record = SeqRecord(
                Seq(trimmed_seq),
                id=f"{fasta_filename}_row{current_row}",
                description=""
            )
            record.letter_annotations["phred_quality"] = trimmed_qual
            if positions:
                record.annotations['left_right_positions'] = positions  # List of dicts
            filtered_reads.append(record)
        else:
            # Read does not contain half-site; filter out
            pass
    
    return filtered_out_middle, kept_reads_half_site, filtered_reads, extracted_sequences, trimmed_adapter_bridge_count, trimmed_adapter_bridge_bases

def map_sequences(extracted_fasta_path, bowtie2_index_prefix, output_sam_path):
    """
    Map extracted sequences to the human genome using Bowtie2.

    Parameters:
    - extracted_fasta_path (str): Path to the FASTA file with extracted sequences.
    - bowtie2_index_prefix (str): Prefix path of the Bowtie2 index.
    - output_sam_path (str): Path to save the Bowtie2 SAM output.
    """
    if not os.path.exists(extracted_fasta_path):
        logging.info(f"No extracted sequences found at {extracted_fasta_path}. Skipping mapping.")
        return
    
    logging.info(f"Starting mapping of extracted sequences from {extracted_fasta_path} to the human genome...")
    try:
        subprocess.run(
            ['bowtie2', '-x', bowtie2_index_prefix, '-f', extracted_fasta_path, '-S', output_sam_path, '--very-sensitive-local'],
            check=True
        )
        logging.info(f"Mapping completed. SAM file saved to {output_sam_path}.")
    except subprocess.CalledProcessError as e:
        logging.error(f"Bowtie2 mapping failed: {e}")
        sys.exit(1)

def parse_sam_file(sam_path, mapping_output_csv, extracted_fasta_path):
    import pysam

    if not os.path.exists(sam_path):
        logging.error(f"SAM file {sam_path} does not exist.")
        return

    logging.info(f"Parsing SAM file {sam_path}...")

    # Load extracted_fasta to map read_id to sequence
    read_id_to_seq = {}
    try:
        with open(extracted_fasta_path, 'r') as fasta_file:
            for record in SeqIO.parse(fasta_file, 'fasta'):
                read_id_to_seq[record.id] = str(record.seq)
    except OSError as e:
        logging.error(f"Error reading the extracted FASTA file {extracted_fasta_path}: {e}")
        sys.exit(1)

    # Initialize list to hold mapping results
    mapping_results = []

    try:
        samfile = pysam.AlignmentFile(sam_path, "r")
    except ValueError as e:
        logging.error(f"Error opening SAM file {sam_path}: {e}")
        sys.exit(1)

    for read in samfile.fetch(until_eof=True):
        # **Modified Condition:** Include mapping_quality >= 20
        if not read.is_unmapped and read.mapping_quality >= 20:
            read_id = read.query_name
            # Retrieve the cryptic_site sequence
            cryptic_site = read_id_to_seq.get(read_id, "")
            mapping_info = {
                'read_id': read_id,
                'reference_name': read.reference_name,
                'reference_start': read.reference_start,
                'mapping_quality': read.mapping_quality,
                'cigar': read.cigarstring,
                'strand': '-' if read.is_reverse else '+',
                'cryptic_site': cryptic_site
            }
            # Optionally, include more information here if needed
            mapping_results.append(mapping_info)

    samfile.close()

    # Convert to DataFrame
    df_mapping = pd.DataFrame(mapping_results)

    # Save to CSV
    df_mapping.to_csv(mapping_output_csv, index=False)
    logging.info(f"Mapping results saved to {mapping_output_csv}.")

# -----------------------------
# Main Filtering Function
# -----------------------------
# (No changes needed here as per user request)

# -----------------------------
# Mapping Function
# -----------------------------
# Note: The duplicate definition has been removed. Only one definition exists above.

# -----------------------------
# Example Usage
# -----------------------------

if __name__ == "__main__":
    # Ensure Bowtie2 is installed and the reference genome is ready
    check_bowtie2_installed()
    ref_genome_path = download_reference_genome()
    bowtie2_index_prefix = build_bowtie2_index(ref_genome_path)
    
    for r1_path in r1_file_paths:
        # Extract sample prefix from R1 filename
        sample_prefix = get_sample_prefix(os.path.basename(r1_path))
        
        # Retrieve the payload_att_site and condition from the metadata dataframe
        payload_att_site_rows = df[df['filename'] == os.path.basename(r1_path)]
        if not payload_att_site_rows.empty:
            payload_att_site = payload_att_site_rows.iloc[0]['payload_att_site']
            condition = payload_att_site_rows.iloc[0]['condition']  # Optional: retrieve condition
        else:
            logging.warning(f"No payload_att_site found for sample prefix: {sample_prefix}. Skipping this file.")
            continue  # Skip this file if payload_att_site is not found
        
        # **Modification Start:** Skip processing positive controls
        if pd.isna(condition):
            logging.warning(f"Condition is missing for sample prefix: {sample_prefix}. Skipping this file.")
            continue  # Skip if condition is missing
        
        if 'positive' not in str(condition).lower():
            logging.info(f"Skipping negative control sample: {sample_prefix} (Condition: {condition})")
            continue  # Skip processing this file
        
        # Validate payload_att_site
        if payload_att_site not in ['attB', 'attP']:
            logging.warning(f"Invalid payload_att_site '{payload_att_site}' for sample prefix: {sample_prefix}. Skipping this file.")
            continue  # Skip if payload_att_site is neither 'attB' nor 'attP'
        
        # **Per-Sample Logging Setup**
        log_file = f"logs/{sample_prefix}.txt"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger = logging.getLogger()
        logger.addHandler(file_handler)
        
        # Log the start of processing for the sample
        logging.info(f"Processing R1 reads for sample: {sample_prefix}")
        logging.info(f"Starting processing of R1 file: {r1_path}...")
        # Define output prefixes and paths
        total_reads_original = 0  # Initialize to capture total reads
        output_filtered_fastq = f"filtered_{sample_prefix}.fastq"
        chunk_size = 32000
        max_workers = 40
        extracted_fasta_dir = "extracted_sequences/"
        mapping_output_dir = "mapping_results/"
        
        os.makedirs(extracted_fasta_dir, exist_ok=True)
        os.makedirs(mapping_output_dir, exist_ok=True)
        os.makedirs('logs/', exist_ok=True)  # Optional: Separate logs directory if needed
        
        # Call the filtering function with the R1 reads
        logging.info(f"Processing R1 reads for sample: {sample_prefix}")
        r1_fastq_path = r1_path
        logging.info(f"Starting processing of R1 file: {r1_fastq_path}...")
        overall_start_time = time.time()
        
        # -----------------------------
        # Step 1: Preload the Entire FASTQ into Memory
        # -----------------------------
        
        logging.info(f"Loading all reads from {r1_fastq_path} into memory...")
        start_time = time.time()
        read_sequences = []
        read_qualities = []
        try:
            with gzip.open(r1_fastq_path, 'rt') as infile:  # Changed to gzip.open with 'rt' mode
                for record in SeqIO.parse(infile, 'fastq'):
                    read_sequences.append(str(record.seq))
                    read_qualities.append(record.letter_annotations["phred_quality"])
        except OSError as e:
            logging.error(f"Error reading the FASTQ file {r1_fastq_path}: {e}")
            sys.exit(1)
        total_reads_original = len(read_sequences)
        loading_time = time.time() - start_time
        logging.info(f"Total reads loaded: {total_reads_original} in {loading_time:.2f} seconds.")
        
        # -----------------------------
        # Step 2: Split Data into Chunks
        # -----------------------------
        
        logging.info(f"Splitting reads into chunks of {chunk_size}...")
        start_time = time.time()
        chunks = list(chunked_iterable_single(read_sequences, read_qualities, chunk_size))
        total_chunks = len(chunks)
        splitting_time = time.time() - start_time
        logging.info(f"Total chunks created: {total_chunks} in {splitting_time:.2f} seconds.")
        
        # -----------------------------
        # Step 3: Initialize In-Memory Storage for Filtered Reads and Extracted Sequences
        # -----------------------------
        
        os.makedirs(extracted_fasta_dir, exist_ok=True)
        os.makedirs(mapping_output_dir, exist_ok=True)
        in_memory_filtered_reads = []
        all_extracted_sequences = []  # To store sequences for mapping
        
        # Initialize counters
        filtered_out_middle_total = 0
        kept_reads_half_site_total = 0
        trimmed_adapter_bridge_count_total = 0
        trimmed_adapter_bridge_bases_total = 0
        
        # -----------------------------
        # Step 4: Initialize Multi-Processing Pool
        # -----------------------------
        
        try:
            # Initialize the progress bar
            with tqdm(total=total_reads_original, desc=f"Filtering R1 reads {os.path.basename(r1_fastq_path)}") as pbar:
                # Use ProcessPoolExecutor to process chunks in parallel
                with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                    # Prepare the arguments for each chunk
                    futures = []
                    fasta_filename = os.path.basename(r1_fastq_path).replace('.fastq.gz', '').replace('.fastq', '')
                    
                    for chunk_index, (start_idx, chunk_seqs, chunk_quals) in enumerate(chunks):
                        futures.append(
                            executor.submit(
                                process_single_reads_chunks,
                                read_start_idx=start_idx,
                                read_sequences=chunk_seqs,
                                read_qualities=chunk_quals,
                                attB_middle_np=attB_middle_np,
                                attP_middle_np=attP_middle_np,
                                threshold=weighted_mismatch_threshold,
                                chunk_index=chunk_index,
                                payload_att_site=payload_att_site,
                                attB_left_np=attB_left_np,
                                attB_right_np=attB_right_np,
                                attP_left_np=attP_left_np,
                                attP_right_np=attP_right_np,
                                fasta_filename=fasta_filename
                            )
                        )
                    
                    # As each future completes, collect filtered records and update counters
                    for future in concurrent.futures.as_completed(futures):
                        try:
                            (filtered_out_middle, kept_reads_half_site, filtered_records, 
                             extracted_sequences, trimmed_adapter_bridge_count, 
                             trimmed_adapter_bridge_bases) = future.result()
                        except Exception as e:
                            logging.error(f"Error processing a chunk: {e}")
                            continue
                        
                        # Append filtered records to in-memory lists
                        if filtered_records:
                            in_memory_filtered_reads.extend(filtered_records)
                            kept_reads_half_site_total += kept_reads_half_site
                        
                        # Append extracted sequences to the main list
                        if extracted_sequences:
                            all_extracted_sequences.extend(extracted_sequences)
                        
                        # Update counters
                        filtered_out_middle_total += filtered_out_middle
                        trimmed_adapter_bridge_count_total += trimmed_adapter_bridge_count
                        trimmed_adapter_bridge_bases_total += trimmed_adapter_bridge_bases
                        pbar.update(len(filtered_records))  # Assuming len(filtered_records) corresponds to reads kept in this chunk
        finally:
            pass  # No cleanup needed
        
        # -----------------------------
        # Step 5: Write Filtered Reads to Output File
        # -----------------------------
        
        logging.info(f"Writing filtered reads to {output_filtered_fastq}...")
        try:
            with open(output_filtered_fastq, 'wt') as outfile:  # Use 'wt' mode for writing text
                SeqIO.write(in_memory_filtered_reads, outfile, 'fastq')
        except OSError as e:
            logging.error(f"Error writing to the output FASTQ file {output_filtered_fastq}: {e}")
            sys.exit(1)
        logging.info(f"Filtered reads saved to {output_filtered_fastq}")
        
        # -----------------------------
        # Step 6: Write Extracted Sequences to FASTA for Mapping
        # -----------------------------
        
        if all_extracted_sequences:
            extracted_fasta_path = os.path.join(extracted_fasta_dir, f"extracted_{fasta_filename}.fasta")
            logging.info(f"Writing extracted sequences to {extracted_fasta_path} for mapping...")
            try:
                with open(extracted_fasta_path, 'wt') as fasta_out:  # Use 'wt' mode for writing text
                    for seq_info in all_extracted_sequences:
                        # Assign 'read_id' to FASTA header
                        fasta_out.write(f">{seq_info['read_id']}\n")
                        fasta_out.write(f"{seq_info['sequence']}\n")
            except OSError as e:
                logging.error(f"Error writing to the extracted FASTA file {extracted_fasta_path}: {e}")
                sys.exit(1)
            logging.info(f"Extracted sequences saved to {extracted_fasta_path}")
        else:
            logging.info("No additional sequences were extracted for mapping.")
        
        # -----------------------------
        # Step 7: Mapping Extracted Sequences
        # -----------------------------
        
        if all_extracted_sequences:
            # Modify the CSV filename to include the total number of reads
            mapping_csv_filename = f"mapped_{fasta_filename}_reads{total_reads_original}.csv"
            mapping_sam_path = os.path.join(mapping_output_dir, f"mapped_{fasta_filename}.sam")
            mapping_csv_path = os.path.join(mapping_output_dir, mapping_csv_filename)
            
            # Perform mapping using Bowtie2
            map_sequences(extracted_fasta_path, bowtie2_index_prefix, mapping_sam_path)
            
            # Parse SAM file and save results
            parse_sam_file(mapping_sam_path, mapping_csv_path, extracted_fasta_path)
        else:
            logging.info("No sequences to map for this sample.")
        
        # -----------------------------
        # Step 8: Logging the Results
        # -----------------------------
        
        overall_time = time.time() - overall_start_time
        logging.info(f"Finished processing of R1 file: {r1_fastq_path}.")
        logging.info(f"Total reads processed: {total_reads_original}")
        logging.info(f"{filtered_out_middle_total} reads filtered out because they contain full attB/P")
        logging.info(f"{kept_reads_half_site_total} reads kept because they contain half-site")
        logging.info(f"Reads filtered out because they do not contain half-site: {total_reads_original - kept_reads_half_site_total - filtered_out_middle_total}")
        logging.info(f"{trimmed_adapter_bridge_count_total} reads trimmed due to adapter bridge.")
        logging.info(f"Total bases trimmed due to adapter bridge: {trimmed_adapter_bridge_bases_total}")
        if trimmed_adapter_bridge_count_total > 0:
            average_trimmed_bases = trimmed_adapter_bridge_bases_total / trimmed_adapter_bridge_count_total
            logging.info(f"Average bases trimmed per read due to adapter bridge: {average_trimmed_bases:.2f}")
        else:
            logging.info("No reads were trimmed due to adapter bridge.")
        logging.info(f"Filtered reads saved to {output_filtered_fastq}")
        logging.info(f"Total processing time: {overall_time:.2f} seconds.\n")
        logger.removeHandler(file_handler)
        file_handler.close()
        
        # Optionally, log completion
        logging.info(f"Completed processing for sample: {sample_prefix}")


