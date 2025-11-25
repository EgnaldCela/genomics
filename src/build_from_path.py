"""
A file to handle creating torch dictionaries of the genome
"""
import os
import torch
import glob
from collections import defaultdict

def parse_filename(filepath):
    """
    Input: transfer/HG03521_hap2_chr21_model1.bed
    Output: (individual, hap, chromosome)
    """
    name = os.path.basename(filepath)
    parts = name.split("_")
    
    if len(parts) < 3:
        # Handle cases where the filename might not be as expected
        raise ValueError(f"Filename {name} does not have expected format.")
        
    individual = parts[0]   # HG03521
    hap = parts[1]        # hap1 or hap2
    chrom = parts[2]      # chr21
    return individual, hap, chrom

def get_distance_tensor_from_file(bed_filepath):
    """ 
    Input: Full path to a .bed file, e.g., /media/.../HG03521_hap2_chr21_model1.bed
    Output: The array of the distances as a pytorch tensor
    """
    
    distances = []
    try:
        with open(bed_filepath, 'r') as f:
            for line in f:
                if line.strip():
                    parts = line.split("\t")
                    if len(parts) < 4:
                        continue # Skip lines that don't have at least 4 columns
                        
                    val = parts[3]  # 4th column
                    if "DNA" in val: # skip the row named "DNA"
                        continue
                    
                    val = val[1:]  # remove 'D'
                    try:
                        distances.append(int(val))
                    except ValueError:
                        # Skip if conversion to int fails
                        print(f"Warning: Skipping non-integer value '{parts[3]}' in {bed_filepath}")
                        continue

    except FileNotFoundError:
        print(f"Error: File not found {bed_filepath}")
        return None
    except Exception as e:
        print(f"Error processing file {bed_filepath}: {e}")
        return None

    # Convert to torch tensor (int32: enough for values like 1632, faster than int64)
    if distances:
        return torch.tensor(distances, dtype=torch.int32)
    else:
        return None


if __name__ == "__main__":
    
    # 1. Define directories
    INPUT_DIR = "/media/pinas/lfranco/genomics/versioni-t2t/latest"
    OUTPUT_DIR = "/media/pinas/egnald/genomics/data/T2T"

    # 2. Ensure the output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Reading from: {INPUT_DIR}")

    # This will hold all our data, e.g.:
    # { 'HG03521': { (key1): tensor1, (key2): tensor2 },
    #   'HG01975': { (key3): tensor3 } }
    all_genome_dicts = defaultdict(dict)

    # 3. Find all files to process
    search_pattern = os.path.join(INPUT_DIR, "*_model1.bed")
    all_bed_files = glob.glob(search_pattern)

    if not all_bed_files:
        print(f"Warning: No '*_model1.bed' files found in {INPUT_DIR}. Exiting.")
        exit()
        
    print(f"Found {len(all_bed_files)} files to process...")

    # 4. Process all files and build dictionaries in memory
    for f_path in all_bed_files:
        try:
            # Get the key, e.g., ('HG03521', 'hap2', 'chr21')
            dict_key = parse_filename(f_path)
            individual = dict_key[0] # The individual's name
            
            # Get the tensor data
            distance_tensor = get_distance_tensor_from_file(f_path)
            
            if distance_tensor is not None:
                # Add this tensor to the correct individual's dictionary
                all_genome_dicts[individual][dict_key] = distance_tensor
            else:
                print(f"Warning: No data loaded from {os.path.basename(f_path)}")
                
        except Exception as e:
            print(f"ERROR: Could not process {os.path.basename(f_path)}: {e}")

    # 5. Save each completed dictionary to a file
    print(f"\nSaving {len(all_genome_dicts)} individual dictionaries to {OUTPUT_DIR}...")
    for individual, genome_dict in all_genome_dicts.items():
        if genome_dict: # Only save if we actually processed data for it
            output_path = os.path.join(OUTPUT_DIR, f"{individual}.pt")
            torch.save(genome_dict, output_path)
            print(f"  Saved {output_path}")
        else:
            print(f"  Skipping {individual} (no data)")
            
    print("\nProcessing complete.")