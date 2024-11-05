import pandas as pd
import pickle
import gzip
import re
import argparse
from tqdm import tqdm

def extract_database_source(identifier: str) -> str:
    """Extracts the database source from the identifier."""
    return identifier.split('_')[0]  # Extracts part before '_'

def extract_reduced_formula(cif_content: str) -> str:
    """Extracts the reduced formula from CIF content."""
    match = re.search(r"_chemical_formula_structural\s+(\S+)", cif_content)
    if match:
        return match.group(1)
    return None

def process_cifs(input_file: str, output_file: str, max_entries: int = None) -> None:
    """Processes CIF data to a DataFrame and saves it as a pickle file."""
    print(f"Processing CIFs from {input_file} to DataFrame and saving to {output_file}")
    print('Loading data...')
    # Load the original CIF data
    with gzip.open(input_file, 'rb') as f:
        data = pickle.load(f)
    
    # Prepare list for dataframe rows
    rows = []
    
    # Determine the range based on max_entries
    if max_entries == 'max':
        data_range = data  # Process all data
    else:
        data_range = data[:int(max_entries)] if max_entries is not None else data
    
    # Iterate over each entry and process
    for entry in tqdm(data_range, desc="Processing CIFs"):
        identifier, cif_content = entry
        database_source = extract_database_source(identifier)
        reduced_formula = extract_reduced_formula(cif_content)
        
        # Append to rows as tuple
        rows.append((database_source, reduced_formula, cif_content))
    
    # Create DataFrame
    df = pd.DataFrame(rows, columns=["Database", "Reduced Formula", "CIF"])

    # Save as pickle
    df.to_pickle(output_file)
    print(f"Data saved to {output_file}")

if __name__ == "__main__":
    # Parse command-line arguments for scalability on HPC
    parser = argparse.ArgumentParser(description="Process CIF database and save as DataFrame.")
    parser.add_argument('--input', type=str, required=True, help="Path to input CIF pickle file")
    parser.add_argument('--output', type=str, required=True, help="Path to save the output DataFrame pickle")
    parser.add_argument('--max_entries', type=str, default=None, help="Limit processing to first N entries or 'max' for all")
    
    args = parser.parse_args()
    
    # Run the processing function
    process_cifs(args.input, args.output, args.max_entries)
