import os
import tarfile
import pickle
import numpy as np
import argparse
from tqdm import tqdm

from crystallm import (
    extract_data_formula,
    extract_space_group_symbol,
)

def get_underrepresented_set(underrepresented_fname):
    with open(underrepresented_fname, "rb") as f:
        comps = pickle.load(f)
    underrepresented_set = set()
    for comp, sg in comps:
        underrepresented_set.add(f"{comp}_{sg}")
    return underrepresented_set

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Identify start token indices.")
    parser.add_argument("--dataset_fname", type=str, required=True,
                        help="Path to the tokenized dataset file (.tar.gz).")
    parser.add_argument("--out_fname", type=str, required=True,
                        help="Path to the file that will contain the serialized Python list of start indices. "
                             "Recommended extension is `.pkl`.")
    parser.add_argument("--underrepresented_fname", type=str, default=None,
                        help="Optional: Path to the file containing underrepresented sample information. "
                             "The file should be .pkl file with a serialized Python list of "
                             "(cell composition, space group) pairs that are under-represented.")
    parser.add_argument("--underrepresented_out_fname", type=str,
                        help="Optional: Path to the file that will contain the under-represented start indices as a "
                             "serialized Python list. Recommended extension is `.pkl`.")
    args = parser.parse_args()

    dataset_fname = args.dataset_fname
    out_fname = args.out_fname
    underrepresented_fname = args.underrepresented_fname
    underrepresented_out_fname = args.underrepresented_out_fname

    base_path = os.path.splitext(os.path.basename(dataset_fname))[0]
    base_path = os.path.splitext(base_path)[0]

    with tarfile.open(dataset_fname, "r:gz") as file:
        # Load meta.pkl and binary files
        try:
            file_content_byte = file.extractfile(f"{base_path}/meta.pkl").read()
            meta = pickle.loads(file_content_byte)
        except KeyError:
            raise FileNotFoundError("The 'meta.pkl' file was not found in the dataset archive.")

        try:
            train_ids = np.frombuffer(file.extractfile(f"{base_path}/train.bin").read(), dtype=np.uint16)
            val_ids = np.frombuffer(file.extractfile(f"{base_path}/val.bin").read(), dtype=np.uint16)
        except KeyError:
            raise FileNotFoundError("The binary files 'train.bin' or 'val.bin' were not found in the dataset archive.")

    underrepresented_set = get_underrepresented_set(underrepresented_fname) if underrepresented_fname else None

    def process_ids(ids, underrepresented_set, all_cif_start_indices, underrepresented_start_indices):
        curr_cif_tokens = []
        for i, id in tqdm(enumerate(ids), total=len(ids), desc="identifying starts..."):
            token = meta["itos"][id]
            if token == "data_":
                all_cif_start_indices.append(i)
            curr_cif_tokens.append(token)

            if len(curr_cif_tokens) > 1 and curr_cif_tokens[-2:] == ['\n', '\n']:
                # Reconstruct CIF to check for underrepresented samples
                cif = ''.join(curr_cif_tokens)
                try:
                    data_formula = extract_data_formula(cif)
                    space_group_symbol = extract_space_group_symbol(cif)
                except Exception as e:
                    print(f"Error processing CIF data: {e}")
                    curr_cif_tokens = []
                    continue
                
                if underrepresented_set and f"{data_formula}_{space_group_symbol}" in underrepresented_set:
                    underrepresented_start_indices.append(all_cif_start_indices[-1])
                
                curr_cif_tokens = []

    # Process training IDs
    all_cif_start_indices, underrepresented_start_indices = [], []
    process_ids(train_ids, underrepresented_set, all_cif_start_indices, underrepresented_start_indices)
    
    with open(out_fname, "wb") as f:
        pickle.dump(all_cif_start_indices, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    if underrepresented_fname and underrepresented_out_fname:
        with open(underrepresented_out_fname, "wb") as f:
            pickle.dump(underrepresented_start_indices, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Process validation IDs
    all_cif_start_indices_val, underrepresented_start_indices_val = [], []
    process_ids(val_ids, underrepresented_set, all_cif_start_indices_val, underrepresented_start_indices_val)

    with open(out_fname.replace(".pkl", "_val.pkl"), "wb") as f:
        pickle.dump(all_cif_start_indices_val, f, protocol=pickle.HIGHEST_PROTOCOL)

    if underrepresented_fname and underrepresented_out_fname:
        with open(underrepresented_out_fname.replace(".pkl", "_val.pkl"), "wb") as f:
            pickle.dump(underrepresented_start_indices_val, f, protocol=pickle.HIGHEST_PROTOCOL)
