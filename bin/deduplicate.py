import argparse
from tqdm import tqdm
import gzip
import os

from crystallm import (
    extract_formula_nonreduced,
    extract_space_group_symbol,
    extract_volume,
    extract_formula_units,
)

import pandas as pd

try:
    import cPickle as pickle
except ImportError:
    import pickle

import warnings
warnings.filterwarnings("ignore")


def deduplicate_binary(cifs):
    """
    Deduplicate CIF entries from a list of (id, cif) tuples.
    Keeps the entry with the smallest volume per (formula, space_group).
    """
    print("Deduplicating binary CIF data...")
    lowest_vpfu = {}

    for id, cif in tqdm(cifs, desc="Processing CIFs"):
        formula = extract_formula_nonreduced(cif)
        space_group = extract_space_group_symbol(cif)
        formula_units = extract_formula_units(cif)
        if formula_units == 0:
            formula_units = 1
        vpfu = extract_volume(cif) / formula_units

        key = (formula, space_group)
        if key not in lowest_vpfu:
            lowest_vpfu[key] = (id, cif, vpfu)
        else:
            existing_vpfu = lowest_vpfu[key][2]
            if vpfu < existing_vpfu:
                lowest_vpfu[key] = (id, cif, vpfu)

    selected_entries = [(id, cif) for id, cif, _ in lowest_vpfu.values()]
    print(f"Number of entries after deduplication: {len(selected_entries):,}")
    return selected_entries


def deduplicate_table(df):
    """
    Deduplicate CIF entries from a dataframe.
    Keeps the entry with the smallest volume per (formula, space_group).
    """
    print("Deduplicating table CIF data...")
    lowest_vpfu = {}

    for idx, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing DataFrame Rows"):
        cif = row['CIF']
        formula = extract_formula_nonreduced(cif)
        space_group = extract_space_group_symbol(cif)
        formula_units = extract_formula_units(cif)
        if formula_units == 0:
            formula_units = 1
        vpfu = extract_volume(cif) / formula_units

        # Deduplication key based on (formula, space_group)
        key = (formula, space_group)
        if key not in lowest_vpfu:
            lowest_vpfu[key] = (idx, vpfu)
        else:
            existing_vpfu = lowest_vpfu[key][1]
            if vpfu < existing_vpfu:
                lowest_vpfu[key] = (idx, vpfu)

    # Extract the indices to keep
    selected_indices = [idx for idx, _ in lowest_vpfu.values()]
    deduplicated_df = df.loc[selected_indices].reset_index(drop=True)
    print(f"Number of entries after deduplication: {deduplicated_df.shape[0]:,}")
    return deduplicated_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deduplicate CIF files or DataFrame tables.")
    parser.add_argument("name", type=str,
                        help="Path to the input file. For type='binary', it should be a gzipped pickle file containing a list of (id, cif) tuples. For type='table', it should be a gzipped pickle file containing a pandas DataFrame with columns ['Database', 'Reduced Formula', 'CIF', 'Bandgap (eV)'].")
    parser.add_argument("--out", "-o", type=str,
                        required=True,
                        help="Path to the output file. For type='binary', the file will be a gzipped pickle of deduplicated (id, cif) tuples. For type='table', the file will be a gzipped pickle of the deduplicated DataFrame.")
    parser.add_argument("--type", type=str, default="binary", choices=["binary", "table"],
                        help="Type of the input file. 'binary' for gzipped pickle files containing list of (id, cif) tuples, 'table' for gzipped pickle files containing DataFrame. Default is 'binary'.")
    args = parser.parse_args()

    input_fname = args.name
    out_fname = args.out
    data_type = args.type

    if data_type == "binary":
        print(f"Loading binary data from {input_fname}...")
        if not os.path.exists(input_fname):
            raise FileNotFoundError(f"Input file {input_fname} does not exist.")

        with gzip.open(input_fname, "rb") as f:
            try:
                cifs = pickle.load(f)
            except Exception as e:
                raise ValueError(f"Failed to load binary data from {input_fname}: {e}")

        if not isinstance(cifs, list) or not all(isinstance(entry, tuple) and len(entry) == 2 for entry in cifs):
            raise ValueError("Binary input file must contain a list of (id, cif) tuples.")

        print(f"Number of CIFs loaded: {len(cifs):,}")

        deduplicated_cifs = deduplicate_binary(cifs)

        print(f"Saving deduplicated binary data to {out_fname}...")
        with gzip.open(out_fname, "wb") as f:
            pickle.dump(deduplicated_cifs, f, protocol=pickle.HIGHEST_PROTOCOL)

    elif data_type == "table":
        print(f"Loading table data from {input_fname}...")
        if not os.path.exists(input_fname):
            raise FileNotFoundError(f"Input file {input_fname} does not exist.")

        # Assuming the input is a gzipped pickle file containing a DataFrame
        with gzip.open(input_fname, "rb") as f:
            try:
                df = pickle.load(f)
            except Exception as e:
                raise ValueError(f"Failed to load table data from {input_fname}: {e}")

        if not isinstance(df, pd.DataFrame):
            raise ValueError("Table input file must contain a pandas DataFrame.")

        required_columns = ['Database', 'Reduced Formula', 'CIF', 'Bandgap (eV)']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Input table must contain the column '{col}'.")
            
        # remove any rows with non-numerical bandgap values in 'Bandgap (eV)' column
        print(f"Number of rows in the table: {df.shape[0]:,}")
        print("Removing rows with non-numerical bandgap values...")
        df = df[pd.to_numeric(df['Bandgap (eV)'], errors='coerce').notnull()]

        print(f"Number of rows after removing entries with no Bandgap: {df.shape[0]:,}")

        deduplicated_df = deduplicate_table(df)

        print(f"Saving deduplicated table data to {out_fname}...")
        # Save as gzipped pickle file
        with gzip.open(out_fname, "wb") as f:
            pickle.dump(deduplicated_df, f, protocol=pickle.HIGHEST_PROTOCOL)

    else:
        raise ValueError(f"Unsupported type '{data_type}'. Choose either 'binary' or 'table'.")

    print("Deduplication completed successfully.")
