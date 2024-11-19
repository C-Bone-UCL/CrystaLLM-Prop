import os
import numpy as np
import gzip
import argparse
import multiprocessing as mp
from tqdm import tqdm
try:
    import cPickle as pickle
except ImportError:
    import pickle

from crystallm import CIFTokenizer, CIFTokenizer_extd
import pandas as pd


def progress_listener(queue, n):
    pbar = tqdm(total=n, desc="Tokenizing...")
    while True:
        message = queue.get()
        if message == "kill":
            break
        pbar.update(message)


def tokenize(chunk_of_cifs, queue=None):
    tokenizer = CIFTokenizer()
    print("Amount of tokens in tokenizer: ", len(tokenizer._tokens_with_unk))
    print(tokenizer._tokens_with_unk)
    tokenized = []
    for cif in tqdm(chunk_of_cifs, disable=queue is not None, desc="Tokenizing..."):
        if queue:
            queue.put(1)
        tokenized_cif = tokenizer.tokenize_cif(cif)
        tokenized.append(tokenized_cif)
    return tokenized


def preprocess(cifs_raw):
    cifs = []
    for cif in tqdm(cifs_raw, desc="Preprocessing CIFs..."):
        # Split the CIF content into lines
        lines = cif.split('\n')
        cif_lines = []
        for line in lines:
            line = line.strip()
            # Keep the bandgap line, and only exclude the pymatgen and comment lines
            # Ensure that lines starting with "_Bandgap_eV:" are not removed
            if len(line) > 0 and not line.startswith("#") and "pymatgen" not in line:
                cif_lines.append(line)
        cif_lines.append("\n")
        cifs.append("\n".join(cif_lines))
    return cifs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tokenize CIF files in a dataframe.")
    parser.add_argument("--input_file", type=str, required=True,
                        help="Path to the input .pkl.gz file containing the dataframe.")
    parser.add_argument("--output_file", type=str, required=True,
                        help="Path to the output .pkl.gz file to save the dataframe with tokenized CIFs.")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of workers to use for processing.")
    args = parser.parse_args()

    input_file = args.input_file
    output_file = args.output_file
    workers = args.workers

    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file '{input_file}' does not exist.")

    print(f"Loading dataframe from {input_file}...")
    with gzip.open(input_file, "rb") as f:
        dataframe = pickle.load(f)

    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("The input file does not contain a pandas DataFrame.")

    # Ensure that the 'CIF' column exists
    if 'CIF' not in dataframe.columns:
        raise KeyError("The dataframe does not contain a 'CIF' column.")

    # Extract CIFs from the dataframe
    cifs_raw = dataframe['CIF'].tolist()

    # Preprocess CIFs
    cifs = preprocess(cifs_raw)

    # Tokenize CIFs using multiprocessing
    print("Tokenizing CIFs...")
    chunks = np.array_split(cifs, workers)
    manager = mp.Manager()
    queue = manager.Queue()
    pool = mp.Pool(workers + 1)  # Add an extra worker for the listener
    watcher = pool.apply_async(progress_listener, (queue, len(cifs),))

    jobs = []
    for i in range(workers):
        chunk = chunks[i]
        job = pool.apply_async(tokenize, (chunk, queue))
        jobs.append(job)

    tokenized_cifs = []
    for job in jobs:
        tokenized_cifs.extend(job.get())

    queue.put("kill")
    pool.close()
    pool.join()

    lens = [len(t) for t in tokenized_cifs]
    unk_counts = [t.count("<unk>") for t in tokenized_cifs]
    print(f"train min tokenized length: {np.min(lens):,}")
    print(f"train max tokenized length: {np.max(lens):,}")
    print(f"train mean tokenized length: {np.mean(lens):.2f} +/- {np.std(lens):.2f}")
    print(f"train total unk counts: {np.sum(unk_counts)}")

    print("Encoding...")
    tokenizer = CIFTokenizer()
    tokenized_cifs = [tokenizer.encode(t) for t in tokenized_cifs]

    # Add the tokenized CIFs to the dataframe
    dataframe['CIFs_tokenized'] = tokenized_cifs

    # Save the updated dataframe to the output file
    print(f"Saving tokenized dataframe to {output_file}...")
    with gzip.open(output_file, "wb") as f:
        pickle.dump(dataframe, f)

    print("Tokenization complete.")
