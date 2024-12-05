import pandas as pd
import glob
import os
import numpy as np
from collections import defaultdict
from intervaltree import Interval, IntervalTree
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D  # Added for custom legend elements

# Set seaborn style for better aesthetics
sns.set(style="whitegrid")

def reverse_complement(seq):
    complement = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A',
                  'a': 't', 'c': 'g', 'g': 'c', 't': 'a',
                  'N': 'N', 'n': 'n'}
    rc_seq = ''.join([complement.get(base, base) for base in reversed(seq)])
    return rc_seq.upper()


# Parameters for minimum total counts per cryptic site per plate
min_total_counts = [5]  # Adjust the values as needed for each plate

# Data mapping sample IDs to conditions
data = [
    ["plate1_1_LAM_AD_A1_S1", "Bxb1_WT_positive_ctrl_at_AAVS1"],
    ["plate1_1_LAM_AD_B1_S13", "Bxb1_WT_positive_ctrl_at_AAVS1"],
    ["plate1_1_LAM_AD_C1_S25", "Bxb1_WT_positive_ctrl_at_AAVS1"],
    ["plate1_2_LAM_AD_A2_S2", "Bxb1_WT"],
    ["plate1_2_LAM_AD_B2_S14", "Bxb1_WT"],
    ["plate1_2_LAM_AD_C2_S26", "Bxb1_WT"],
    ["plate1_3_LAM_AD_A3_S3", "Bxb1_D14K_N226K_H334K"],
    ["plate1_3_LAM_AD_B3_S15", "Bxb1_D14K_N226K_H334K"],
    ["plate1_3_LAM_AD_C3_S27", "Bxb1_D14K_N226K_H334K"],
    ["plate1_4_LAM_AD_A4_S4", "Bxb1_D14K_H334K_A341K"],
    ["plate1_4_LAM_AD_B4_S16", "Bxb1_D14K_H334K_A341K"],
    ["plate1_4_LAM_AD_C4_S28", "Bxb1_D14K_H334K_A341K"],
    ["plate1_5_LAM_AD_A5_S5", "Bxb1_V12K_H334K_A341K"],
    ["plate1_5_LAM_AD_B5_S17", "Bxb1_V12K_H334K_A341K"],
    ["plate1_5_LAM_AD_C5_S29", "Bxb1_V12K_H334K_A341K"],
    ["plate1_6_LAM_AD_A6_S6", "Bxb1_D14K_N226K_H334K_A341K"],
    ["plate1_6_LAM_AD_B6_S18", "Bxb1_D14K_N226K_H334K_A341K"],
    ["plate1_6_LAM_AD_C6_S30", "Bxb1_D14K_N226K_H334K_A341K"],
    ["plate1_7_LAM_AD_A7_S7", "Bxb1_V12K_N226K_H334K"],
    ["plate1_7_LAM_AD_B7_S19", "Bxb1_V12K_N226K_H334K"],
    ["plate1_7_LAM_AD_C7_S31", "Bxb1_V12K_N226K_H334K"],
    ["plate1_8_LAM_AD_A8_S8", "Bxb1_D14K"],
    ["plate1_8_LAM_AD_B8_S20", "Bxb1_D14K"],
    ["plate1_8_LAM_AD_C8_S32", "Bxb1_D14K"],
    ["plate1_9_LAM_AD_A9_S9", "Bxb1_A341K"],
    ["plate1_9_LAM_AD_B9_S21", "Bxb1_A341K"],
    ["plate1_9_LAM_AD_C9_S33", "Bxb1_A341K"],
    ["plate1_10_LAM_AD_A10_S10", "negative_ctrl"],
    ["plate1_10_LAM_AD_B10_S22", "negative_ctrl"],
    ["plate1_10_LAM_AD_C10_S34", "negative_ctrl"],
    ["plate2_1_LAM_AD_E1_S49", "Bxb1_WT_positive_ctrl_at_CFTR"],
    ["plate2_1_LAM_AD_F1_S61", "Bxb1_WT_positive_ctrl_at_CFTR"],
    ["plate2_1_LAM_AD_G1_S73", "Bxb1_WT_positive_ctrl_at_CFTR"],
    ["plate2_2_LAM_AD_E2_S50", "Bxb1_WT"],
    ["plate2_2_LAM_AD_F2_S62", "Bxb1_WT"],
    ["plate2_2_LAM_AD_G2_S74", "Bxb1_WT"],
    ["plate2_3_LAM_AD_E3_S51", "Bxb1_D14K_N226K_H334K"],
    ["plate2_3_LAM_AD_F3_S63", "Bxb1_D14K_N226K_H334K"],
    ["plate2_3_LAM_AD_G3_S75", "Bxb1_D14K_N226K_H334K"],
    ["plate2_4_LAM_AD_E4_S52", "Bxb1_D14K_H334K_A341K"],
    ["plate2_4_LAM_AD_F4_S64", "Bxb1_D14K_H334K_A341K"],
    ["plate2_4_LAM_AD_G4_S76", "Bxb1_D14K_H334K_A341K"],
    ["plate2_5_LAM_AD_E5_S53", "Bxb1_V12K_H334K_A341K"],
    ["plate2_5_LAM_AD_F5_S65", "Bxb1_V12K_H334K_A341K"],
    ["plate2_5_LAM_AD_G5_S77", "Bxb1_V12K_H334K_A341K"],
    ["plate2_6_LAM_AD_E6_S54", "Bxb1_D14K_N226K_H334K_A341K"],
    ["plate2_6_LAM_AD_F6_S66", "Bxb1_D14K_N226K_H334K_A341K"],
    ["plate2_6_LAM_AD_G6_S78", "Bxb1_D14K_N226K_H334K_A341K"],
    ["plate2_7_LAM_AD_E7_S55", "Bxb1_V12K_N226K_H334K"],
    ["plate2_7_LAM_AD_F7_S67", "Bxb1_V12K_N226K_H334K"],
    ["plate2_7_LAM_AD_G7_S79", "Bxb1_V12K_N226K_H334K"],
    ["plate2_8_LAM_AD_E8_S56", "Bxb1_D14K"],
    ["plate2_8_LAM_AD_F8_S68", "Bxb1_D14K"],
    ["plate2_8_LAM_AD_G8_S80", "Bxb1_D14K"],
    ["plate2_9_LAM_AD_E9_S57", "Bxb1_A341K"],
    ["plate2_9_LAM_AD_F9_S69", "Bxb1_A341K"],
    ["plate2_9_LAM_AD_G9_S81", "Bxb1_A341K"],
    ["plate2_10_LAM_AD_E10_S58", "negative_ctrl"],
    ["plate2_10_LAM_AD_F10_S70", "negative_ctrl"],
    ["plate2_10_LAM_AD_G10_S82", "negative_ctrl"],
]

# Create a dictionary mapping sample IDs to conditions
sample_to_condition = {sample_id: condition for sample_id, condition in data}

# Modified: Rename specific conditions to "positive control"
for sample_id, condition in sample_to_condition.items():
    if condition in ["Bxb1_WT_positive_ctrl_at_AAVS1", "Bxb1_WT_positive_ctrl_at_CFTR"]:
        sample_to_condition[sample_id] = "positive control"

# Re-collect all unique conditions after renaming
all_conditions = sorted(set(sample_to_condition.values()))

# Create a consistent mapping from condition to code
condition_to_code = {condition: idx for idx, condition in enumerate(all_conditions)}

# Create mapping from sample IDs to plates
sample_to_plate = {}
for sample_id in sample_to_condition.keys():
    if sample_id.startswith('plate1_'):
        sample_to_plate[sample_id] = 'plate1'
    elif sample_id.startswith('plate2_'):
        sample_to_plate[sample_id] = 'plate2'
    else:
        sample_to_plate[sample_id] = 'unknown'

# Read all CSV files
csv_files = glob.glob('mapping_results/*.csv')

# Organize sample data per plate
plates = ['plate1']
for idx_plate, plate in enumerate(plates):
    print(f"Processing {plate}...")

    # Get the min_total_counts for this plate
    plate_min_total_counts = min_total_counts[idx_plate]

    # Collect all positions per chromosome for this plate
    positions_per_chr = defaultdict(set)  # Use set to avoid duplicates

    # Collect data per sample for this plate
    sample_data = {}

    # Collect sample IDs for this plate
    plate_sample_ids = {sample_id for sample_id, p in sample_to_plate.items() if p == plate}

    # **Modified: Collect cryptic_site information per position**
    cryptic_site_per_position = defaultdict(lambda: defaultdict(set))  # chrom -> pos -> set of cryptic_sites

    # Read CSV files for samples in this plate
    for csv_file in csv_files:
        # Extract sample ID from filename
        basename = os.path.basename(csv_file)
        sample_id_part = basename.split('_reads')[0]
        # Remove 'mapped_' prefix
        sample_id_full = sample_id_part[len('mapped_'):]
        # Remove '_R1_001' and any subsequent parts
        sample_id = sample_id_full.split('_R1_001')[0]
        # Skip if sample is not in this plate
        if sample_id not in plate_sample_ids:
            continue
        # Map to condition
        condition = sample_to_condition.get(sample_id, 'Unknown')

        # Read CSV with appropriate separator
        try:
            df = pd.read_csv(csv_file, sep=',', header=0)
        except Exception as e:
            print(f"Error reading {csv_file}: {e}")
            continue

        # Strip leading/trailing whitespace from column names
        df.columns = df.columns.str.strip()

        # Print columns to verify they are correct
        print(f"Columns in {csv_file}: {df.columns.tolist()}")

        if df.empty:
            print(f"{csv_file} is empty. Skipping.")
            continue  # Skip empty files

        # Enforce data types
        df['reference_name'] = df['reference_name'].astype(str)
        df['reference_start'] = pd.to_numeric(df['reference_start'], errors='coerce')
        df = df.dropna(subset=['reference_start'])
        df['reference_start'] = df['reference_start'].astype(int)

        # Check if required columns are present
        required_columns = {'reference_name', 'reference_start', 'cryptic_site'}
        if not required_columns.issubset(df.columns):
            print(f"Required columns missing in {csv_file}. Columns found: {df.columns.tolist()}")
            continue

        # Collect positions per chromosome
        for idx, row in df.iterrows():
            chrom = row['reference_name']
            pos = row['reference_start']
            cryptic_site = row['cryptic_site']
            positions_per_chr[chrom].add(pos)
            cryptic_site_per_position[chrom][pos].add(cryptic_site)

        # Store the dataframe per sample
        sample_data[sample_id] = df

    # Now, for each chromosome, cluster positions into bins
    print("Chromosomes and number of positions collected:")
    for chrom, positions in positions_per_chr.items():
        print(f"Chromosome {chrom}: {len(positions)} positions")

    bins_per_chr = {}
    cryptic_sites_per_bin = {}  # bin_id -> set of cryptic_sites
    for chrom, positions in positions_per_chr.items():
        sorted_positions = sorted(positions)
        bins = []
        current_bin = []
        for pos in sorted_positions:
            if not current_bin:
                current_bin = [pos]
            elif pos - current_bin[-1] <= 100:
                current_bin.append(pos)
            else:
                bins.append(current_bin)
                current_bin = [pos]
        if current_bin:
            bins.append(current_bin)
        bins_per_chr[chrom] = bins

    # Build interval trees per chromosome
    print("Bins created per chromosome:")
    bin_id_counter = 1
    bin_map = {}  # bin_id -> (chromosome, start, end)
    bins_interval_trees = {}  # chrom -> IntervalTree

    for chrom, bins in bins_per_chr.items():
        itree = IntervalTree()
        for bin_positions in bins:
            start = min(bin_positions) - 100
            end = max(bin_positions) + 100
            bin_id = bin_id_counter
            bin_map[bin_id] = {'chromosome': chrom, 'start': start, 'end': end}
            # Collect cryptic sites for this bin
            bin_cryptic_sites = set()
            for pos in bin_positions:
                bin_cryptic_sites.update(cryptic_site_per_position[chrom][pos])
            cryptic_sites_per_bin[bin_id] = bin_cryptic_sites
            itree[start:end+1] = bin_id
            bin_id_counter += 1
        bins_interval_trees[chrom] = itree
        print(f"Chromosome {chrom}: {len(itree)} bins")

    # Now, for each sample, count the number of reads mapping to each bin
    counts_per_sample = defaultdict(lambda: defaultdict(int))  # sample_id -> bin_id -> count

    for sample_id, df in sample_data.items():
        condition = sample_to_condition.get(sample_id, 'Unknown')
        for idx, row in df.iterrows():
            chrom = str(row['reference_name'])
            pos = int(row['reference_start'])
            itree = bins_interval_trees.get(chrom)
            if not itree:
                print(f"No interval tree found for chromosome {chrom} in sample {sample_id}")
                continue
            overlapping_bins = itree[pos]
            if overlapping_bins:
                for interval in overlapping_bins:
                    bin_id = interval.data
                    counts_per_sample[sample_id][bin_id] += 1
            else:
                print(f"No overlapping bins for position {pos} on chromosome {chrom} in sample {sample_id}")

    # Proceed to aggregate counts per condition
    # Aggregate counts per bin per condition
    counts_per_condition = defaultdict(lambda: defaultdict(int))  # condition -> bin_id -> count

    for sample_id, bin_counts in counts_per_sample.items():
        condition = sample_to_condition.get(sample_id, 'Unknown')
        for bin_id, count in bin_counts.items():
            counts_per_condition[condition][bin_id] += count

    # Now, create a DataFrame with counts per condition per bin
    data_rows = []
    conditions = counts_per_condition.keys()
    bins = set()
    for condition, bin_counts in counts_per_condition.items():
        for bin_id in bin_counts.keys():
            bins.add(bin_id)

    for bin_id in bins:
        if bin_id not in bin_map:
            continue
        row = {'bin_id': bin_id}
        chrom = bin_map[bin_id]['chromosome']
        start = bin_map[bin_id]['start']
        end = bin_map[bin_id]['end']
        # Add leading zeros to chromosome numbers for proper sorting
        if chrom.isdigit():
            chrom_padded = chrom.zfill(2)
        else:
            chrom_padded = chrom
        # Create a label using chromosome and position
        bin_label = f"{chrom_padded}:{start}"
        row['bin_label'] = bin_label
        # **Added: Include cryptic site(s)**
        cryptic_sites_in_bin = cryptic_sites_per_bin.get(bin_id, set())
        row['cryptic_site'] = ';'.join(sorted(cryptic_sites_in_bin))
        for condition in conditions:
            count = counts_per_condition.get(condition, {}).get(bin_id, 0)
            row[condition] = count
        data_rows.append(row)

    df_plot = pd.DataFrame(data_rows)
    if df_plot.empty:
        print(f"No data to plot for {plate}.")
        continue
    df_plot.to_csv(f'counts_{plate}.csv', index=False)

    # Added: Save counts without applying min_total_counts
    df_plot_full = df_plot.copy()
    df_plot_full['cryptic_site'] = df_plot_full['cryptic_site'].apply(reverse_complement)
    df_plot_full.to_csv(f'counts_full_{plate}.csv', index=False)

    # Calculate total counts per bin across all conditions
    df_plot['Total_Counts'] = df_plot[list(conditions)].sum(axis=1)

    # Filter out bins where total counts < plate_min_total_counts
    df_plot = df_plot[df_plot['Total_Counts'] >= plate_min_total_counts]

    if df_plot.empty:
        print(f"No data to plot after applying min_total_counts filter for {plate}.")
        continue

    # Melt the DataFrame for processing
    df_melt = df_plot.melt(
        id_vars=['bin_id', 'bin_label', 'cryptic_site'],
        value_vars=list(conditions),
        var_name='Condition',
        value_name='Counts'
    )

    # Filter out condition-bin combinations where Counts < 1
    df_melt = df_melt[df_melt['Counts'] >= 1]

    # Remove bins that have no remaining conditions after filtering
    bins_with_data = df_melt['bin_id'].unique()
    df_melt = df_melt[df_melt['bin_id'].isin(bins_with_data)]

    if df_melt.empty:
        print(f"No data to plot after filtering for {plate}.")
        continue

    # Create a pivot table for the heatmap using bin_label
    heatmap_data = df_melt.pivot_table(
        index='Condition',
        columns='bin_label',
        values='Counts',
        aggfunc='sum',
        fill_value=0
    )

    # Sort conditions and bins for better visualization
    heatmap_data = heatmap_data.sort_index()

    # Modified: Sort columns based on chromosome and start position
    def extract_chrom_start(bin_label):
        chrom_padded, start = bin_label.split(':')
        chrom = chrom_padded.lstrip('0')  # Remove leading zeros
        start = int(start)
        # Map chromosome names to numbers for sorting
        chrom_to_num = {'X': 23, 'Y': 24, 'MT': 25, 'M': 25}
        if chrom.isdigit():
            chrom_num = int(chrom)
        else:
            chrom_num = chrom_to_num.get(chrom.upper(), 26)  # Use 26 for unknown chromosomes
        return (chrom_num, start)

    sorted_columns = sorted(heatmap_data.columns, key=extract_chrom_start)
    heatmap_data = heatmap_data.reindex(columns=sorted_columns)

    # Normalize counts per condition to percentages
    heatmap_data_percent = heatmap_data.div(heatmap_data.sum(axis=1), axis=0) * 100

    # Round percentages to integers for display
    heatmap_data_percent_rounded = heatmap_data_percent.round(0).astype(int)

    # Plot the heatmap with percentages and annotations
    plt.figure(figsize=(20, 10))
    sns.heatmap(
        heatmap_data_percent_rounded,
        cmap='viridis',
        cbar_kws={'label': 'Percentage (%)'},
        linewidths=0.5,
        linecolor='gray',
        annot=True,
        fmt='d',
        annot_kws={"size": 8}
    )

    plt.title(f'Cryptic Site Percentage Distribution Heatmap for {plate}', fontsize=16)
    plt.xlabel('Cryptic Sites (Chromosome:Position)', fontsize=14)
    plt.ylabel('Conditions', fontsize=14)

    plt.tight_layout()
    plt.savefig(f'counts_heatmap_percentage_{plate}.png', dpi=300)
    plt.close()

    # Generate the scatter plot using raw counts
    # Assign x-axis positions based on the sorted order of bins
    df_melt_scatter = df_melt.copy()

    # Extract 'chromosome' and 'start' from 'bin_label' and sort accordingly
    df_melt_scatter[['chromosome_padded', 'start']] = df_melt_scatter['bin_label'].str.split(':', expand=True)
    df_melt_scatter['start'] = df_melt_scatter['start'].astype(int)
    # Remove leading zeros for sorting
    df_melt_scatter['chromosome'] = df_melt_scatter['chromosome_padded'].str.lstrip('0')
    chrom_to_num = {'X': 23, 'Y': 24, 'MT': 25, 'M': 25}
    df_melt_scatter['chromosome_num'] = df_melt_scatter['chromosome'].apply(
        lambda x: int(x) if x.isdigit() else chrom_to_num.get(x.upper(), 26)
    )
    df_melt_scatter.sort_values(by=['chromosome_num', 'start'], inplace=True)

    # Assign x-coordinates based on the bin index after sorting
    unique_bins = df_melt_scatter['bin_label'].unique()
    bin_to_xcoord = {bin_label: idx for idx, bin_label in enumerate(unique_bins)}
    df_melt_scatter['x_coord'] = df_melt_scatter['bin_label'].map(bin_to_xcoord)

    # Map conditions to consistent codes using the mapping created earlier
    df_melt_scatter['Condition_Code'] = df_melt_scatter['Condition'].map(condition_to_code)

    # Calculate figure width based on the number of unique bins
    fig_width = max(14, len(unique_bins) * 0.3)  # Adjust the multiplier as needed

    # Now, create the scatter plot
    plt.figure(figsize=(fig_width, 8))

    scatter = plt.scatter(
        df_melt_scatter['x_coord'],
        df_melt_scatter['Counts'],
        c=df_melt_scatter['Condition_Code'],  # Use consistent condition codes
        cmap='tab20',
        s=50,  # Increase marker size for better visibility
        alpha=0.7,
        edgecolors='w',  # White edge around markers
        linewidths=0.5
    )

    plt.title(f'Cryptic Site Counts by Position for {plate}', fontsize=16)
    plt.xlabel('Chromosome and Position', fontsize=14)
    plt.ylabel('Counts', fontsize=14)

    # Set x-ticks with all labels
    xticks_positions = [bin_to_xcoord[bin_label] for bin_label in unique_bins]
    xticks_labels = unique_bins
    plt.xticks(xticks_positions, xticks_labels, rotation=90, fontsize=8)
    plt.yticks(fontsize=12)

    # Add gridlines for better readability
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

    # Create custom legend handles with consistent colors
    color_map = plt.get_cmap('tab20')

    legend_elements = [
        Line2D(
            [0], [0],
            marker='o',
            color='w',
            label=condition,
            markerfacecolor=color_map(condition_to_code[condition] % len(color_map.colors)),
            markersize=8,
            markeredgecolor='k',
            linewidth=0
        )
        for condition in all_conditions
    ]

    plt.legend(
        handles=legend_elements,
        title="Condition",
        bbox_to_anchor=(1.05, 1),
        loc='upper left',
        fontsize=10,
        title_fontsize=12,
        frameon=True,
        shadow=True,
        borderpad=1
    )

    plt.tight_layout()
    plt.savefig(f'counts_scatter_{plate}.pdf', format="pdf", dpi=300)
    plt.close()

    # Create DataFrame of counts per sample per bin
    df_counts_sample = []

    for sample_id, bin_counts in counts_per_sample.items():
        condition = sample_to_condition.get(sample_id, 'Unknown')
        for bin_id, count in bin_counts.items():
            df_counts_sample.append({'sample_id': sample_id, 'Condition': condition, 'bin_id': bin_id, 'Counts': count})

    df_counts_sample = pd.DataFrame(df_counts_sample)

    # Filter for counts >= 1
    df_counts_sample_filtered = df_counts_sample[df_counts_sample['Counts'] >= 1]

    # For each condition and bin_id, count number of samples (replicates) with counts >= 1
    replicates_per_site_condition = df_counts_sample_filtered.groupby(['Condition', 'bin_id']).agg({'sample_id': 'nunique'}).reset_index()
    replicates_per_site_condition.rename(columns={'sample_id': 'Replicate_Count'}, inplace=True)

    # Filter for bin_ids where Replicate_Count >= 2
    sites_with_enough_replicates = replicates_per_site_condition[replicates_per_site_condition['Replicate_Count'] >= 2]

    # For each condition, count the number of unique bin_ids
    sites_per_condition = sites_with_enough_replicates.groupby('Condition')['bin_id'].nunique().reset_index()
    sites_per_condition.columns = ['Condition', 'Cryptic_Site_Count']

    # Plot the bar chart with consistent colors
    plt.figure(figsize=(12, 8))
    sns.barplot(
        data=sites_per_condition,
        x='Condition',
        y='Cryptic_Site_Count',
        palette=[color_map(condition_to_code[cond] % len(color_map.colors)) for cond in sites_per_condition['Condition']]
    )
    plt.xticks(rotation=90)
    plt.title(f'Number of Cryptic Sites per Condition (Counts ≥ 1 in at least 2 replicates) for {plate}', fontsize=16)
    plt.xlabel('Condition', fontsize=14)
    plt.ylabel('Number of Cryptic Sites', fontsize=14)

    plt.tight_layout()
    plt.savefig(f'cryptic_sites_per_condition_counts_ge1_reps_ge2_{plate}.png', format="pdf", dpi=300)
    plt.close()

