import argparse
import gzip
from tqdm import tqdm
import multiprocessing as mp
from queue import Empty
import os

from crystallm import (
    semisymmetrize_cif,
    replace_data_formula_with_nonreduced_formula,
    add_atomic_props_block,
    round_numbers,
    extract_formula_units,
)

try:
    import cPickle as pickle
except ImportError:
    import pickle

import warnings
warnings.filterwarnings("ignore")


def progress_listener(queue, n):
    pbar = tqdm(total=n)
    tot = 0
    while True:
        message = queue.get()
        if message is None:
            break
        tot += message
        pbar.update(message)
        if tot >= n:
            break
    pbar.close()


def augment_cif(progress_queue, task_queue, result_queue, oxi, decimal_places):
    augmented_cifs = []

    while not task_queue.empty():
        try:
            id, cif_str = task_queue.get_nowait()
        except Empty:
            break

        try:
            formula_units = extract_formula_units(cif_str)
            # exclude CIFs with formula units (Z) = 0, which are erroneous
            if formula_units == 0:
                raise Exception()

            cif_str = replace_data_formula_with_nonreduced_formula(cif_str)
            cif_str = semisymmetrize_cif(cif_str)
            cif_str = add_atomic_props_block(cif_str, oxi)
            cif_str = round_numbers(cif_str, decimal_places=decimal_places)
            augmented_cifs.append((id, cif_str))
        except Exception:
            pass

        progress_queue.put(1)

    result_queue.put(augmented_cifs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-process CIF files.")
    parser.add_argument("name", type=str,
                        help="Path to the input file. If type is 'list', it should contain the gzipped contents of a pickled Python list of tuples, of (id, cif) pairs. If type is 'table', it should be a gzipped pickled dataframe.")
    parser.add_argument("--out", "-o", action="store",
                        required=True,
                        help="Path to the file where the pre-processed CIFs will be stored. If type is 'list', it will contain the gzipped contents of a pickled Python list of tuples. If type is 'table', it will contain the gzipped contents of the updated dataframe. It is recommended that the filename end in `.pkl.gz`.")
    parser.add_argument("--oxi", action="store_true",
                        help="Include this flag if the CIFs to be processed contain oxidation state information.")
    parser.add_argument("--decimal-places", type=int, default=4,
                        help="The number of decimal places to round the floating point numbers to in the CIF. Default is 4.")
    parser.add_argument("--workers", type=int, default=4,
                        help="The number of workers to use for processing. Default is 4.")
    parser.add_argument("--type", type=str, choices=['list', 'table'], default='list',
                        help="Type of the input file. 'list' for a list of (id, cif) tuples, 'table' for a dataframe with CIFs.")

    args = parser.parse_args()

    input_fname = args.name
    out_fname = args.out
    oxi = args.oxi
    decimal_places = args.decimal_places
    workers = args.workers
    input_type = args.type

    print(f"Loading data from {input_fname} as type '{input_type}'...")
    if input_type == 'table':
        with gzip.open(input_fname, "rb") as f:
            dataframe = pickle.load(f)
        # Ensure the dataframe has the required columns
        required_columns = {'CIF'}
        if not required_columns.issubset(dataframe.columns):
            raise ValueError(f"The input dataframe must contain the columns: {required_columns}")
            # make all the floats in the 'Bandgap (eV)' column to have 3 decimal places

        # round the 'Bandgap (eV)' column to 3 decimal places
        print(f"Found {len(dataframe)} rows in the dataframe")
        print('Removing all rows with Bandgap (eV) = 0...')
        dataframe = dataframe[dataframe['Bandgap (eV)'] != 0]
        print(f"There are now {len(dataframe)} rows in the dataframe")
        print("Rounding the 'Bandgap (eV)' column to 3 decimal places...")
        dataframe['Bandgap (eV)'] = dataframe['Bandgap (eV)'].apply(lambda x: round(x, 3))
        dataframe.reset_index(drop=True, inplace=True)

        with gzip.open(out_fname, "wb") as f:
            pickle.dump(dataframe, f, protocol=pickle.HIGHEST_PROTOCOL)
        # Extract (id, cif_str) pairs. Assuming the dataframe index serves as the id.
        cifs = [(idx, row['CIF']) for idx, row in dataframe.iterrows()]
        total_cifs = len(cifs)
    else:
        with gzip.open(input_fname, "rb") as f:
            cifs = pickle.load(f)
        total_cifs = len(cifs)

    manager = mp.Manager()
    progress_queue = manager.Queue()
    task_queue = manager.Queue()
    result_queue = manager.Queue()

    for item in cifs:
        task_queue.put(item)

    watcher = mp.Process(target=progress_listener, args=(progress_queue, total_cifs,))

    processes = [mp.Process(target=augment_cif, args=(progress_queue, task_queue, result_queue, oxi, decimal_places))
                 for _ in range(workers)]
    processes.append(watcher)

    for process in processes:
        process.start()

    for process in processes:
        process.join()

    modified_cifs = []

    while not result_queue.empty():
        modified_cifs.extend(result_queue.get())

    print(f"Number of CIFs after preprocessing: {len(modified_cifs)}")

    if input_type == 'table':
        # Create a mapping from id to modified_cif_str
        modified_cif_dict = {id: cif_str for id, cif_str in modified_cifs}
        # Update the dataframe's 'CIF' column with modified CIFs
        for id, cif_str in modified_cifs:
            dataframe.at[id, 'CIF'] = cif_str
        print(f"Saving updated dataframe to {out_fname}...")
        with gzip.open(out_fname, "wb") as f:
            pickle.dump(dataframe, f, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        print(f"Saving preprocessed CIFs to {out_fname}...")
        with gzip.open(out_fname, "wb") as f:
            pickle.dump(modified_cifs, f, protocol=pickle.HIGHEST_PROTOCOL)

    print("Preprocessing completed successfully.")
