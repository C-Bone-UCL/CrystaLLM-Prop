import argparse
import os
import time
import pickle
import pandas as pd
from pymatgen.ext.matproj import MPRester
from requests.exceptions import HTTPError
from tqdm import tqdm

def extract_material_id(cif_str):
    """
    Extracts the material ID from the CIF string.
    Assumes the material ID is in the format 'mp-XXXXXXX'.
    """
    try:
        # The material ID is typically the second word in the first line
        first_line = cif_str.split('\n')[0]
        parts = first_line.split()
        for part in parts:
            if part.startswith('mp-'):
                return part
        return None
    except Exception:
        return None

def load_database(input_file):
    """
    Loads the pickle database from the specified input file.
    """
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file '{input_file}' does not exist.")
    with open(input_file, 'rb') as f:
        df = pickle.load(f)
    return df

def save_database(df, output_file):
    """
    Saves the DataFrame to a pickle file at the specified output path.
    """
    # Ensure the output directory exists
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(output_file, 'wb') as f:
        pickle.dump(df, f)

def add_bandgap_column(df, api_key, size):
    """
    Adds a 'Bandgap (eV)' column to the DataFrame by querying the Materials Project.
    """
    mpr = MPRester(api_key)
    bandgaps = []

    # Iterate over the DataFrame with a progress bar
    for idx, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing materials"):
        cif_str = row['CIF']
        material_id = extract_material_id(cif_str)
        
        if material_id is None:
            bandgaps.append("N/A")
            continue
        
        try:
            # Query the band_gap property
            bandgap_info = mpr.query(criteria={"material_id": material_id}, properties=["band_gap"])
            if bandgap_info and "band_gap" in bandgap_info[0]:
                bandgap = bandgap_info[0]["band_gap"]
                bandgap = round(bandgap, 3) if isinstance(bandgap, (float, int)) else "N/A"
            else:
                bandgap = "N/A"
        except HTTPError as http_err:
            print(f"HTTP error occurred for {material_id}: {http_err}")
            bandgap = "N/A"
        except Exception as e:
            print(f"Error retrieving data for material {material_id}: {e}")
            bandgap = "N/A"
        
        bandgaps.append(bandgap)
        time.sleep(0.05)  # To avoid hitting the API rate limit

    df['Bandgap (eV)'] = bandgaps
    return df

def main():
    parser = argparse.ArgumentParser(description="Filter MP entries and add bandgap information.")
    parser.add_argument("--api_key", type=str, required=True, 
                        help="API key for accessing the Materials Project database.")
    parser.add_argument("--size", type=str, default="100", 
                        help="Specify the dataset size or 'max' for all data. Default is 100.")
    parser.add_argument("--input_file", type=str, default="/home/uccacbo/CrystaLLM/compressed_datasets_models/CIFs_db_full.pkl",
                        help="Path to the input pickle database.")
    parser.add_argument("--output_file", type=str, default="/home/uccacbo/CrystaLLM/compressed_datasets_models/CIFs_db_MP.pkl",
                        help="Path to save the filtered and updated pickle database.")
    args = parser.parse_args()

    # Load the full database
    print("Loading the full database...")
    df_full = load_database(args.input_file)

    # Filter for 'MP' entries
    print("Filtering for 'MP' database entries...")
    df_mp = df_full[df_full['Database'] == 'MP'].copy()

    # Determine dataset size
    if args.size.lower() == "max":
        num_samples = df_mp.shape[0]
    else:
        try:
            num_samples = int(args.size)
            if num_samples < 1:
                raise ValueError
            num_samples = min(num_samples, df_mp.shape[0])
        except ValueError:
            raise ValueError("Invalid value for size. Please specify a positive integer or 'max'.")

    # Limit the DataFrame to the desired size
    print(f"Selecting {num_samples} entries...")
    df_mp = df_mp.iloc[:num_samples].reset_index(drop=True)

    # Add the 'Bandgap (eV)' column
    print("Adding 'Bandgap (eV)' column by querying Materials Project...")
    df_mp = add_bandgap_column(df_mp, args.api_key, num_samples)

    # Save the new database
    print(f"Saving the updated database to '{args.output_file}'...")
    save_database(df_mp, args.output_file)

    print("Process completed successfully.")

if __name__ == "__main__":
    main()
