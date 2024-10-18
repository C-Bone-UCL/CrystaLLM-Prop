import os
import io
import tarfile
import argparse
import re
from pymatgen.io.cif import CifWriter, Structure
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")


def process_cif_files(input_dir, output_tar_gz):
    with tarfile.open(output_tar_gz, "w:gz") as tar:
        for root, _, files in os.walk(input_dir):
            for file in tqdm(files, desc="preparing CIF files..."):
                if file.endswith(".cif"):
                    file_path = os.path.join(root, file)
                    
                    # Read the original CIF file
                    with open(file_path, "r") as f:
                        original_cif_content = f.read()

                    # Extract the bandgap information from the original CIF content
                    match = re.search(r'_Bandgap_eV: (\d+\.?\d*)', original_cif_content)
                    bandgap = None
                    if match:
                        bandgap = float(match.group(1))

                    # Use pymatgen to generate the new CIF content
                    struct = Structure.from_file(file_path)
                    cif_content = CifWriter(struct=struct, symprec=0.1).__str__()

                    # Re-append the bandgap information to the generated CIF content
                    if bandgap is not None:
                        cif_content += f"_Bandgap_eV: {bandgap:.2f}\n\n"

                    # Create the tar entry for the CIF file
                    cif_file = tarfile.TarInfo(name=file)
                    cif_bytes = cif_content.encode("utf-8")
                    cif_file.size = len(cif_bytes)
                    tar.addfile(cif_file, io.BytesIO(cif_bytes))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare custom CIF files and save to a tar.gz file.")
    parser.add_argument("input_dir", help="Path to the directory containing CIF files.")
    parser.add_argument("output_tar_gz", help="Path to the output tar.gz file")
    args = parser.parse_args()

    process_cif_files(args.input_dir, args.output_tar_gz)

    print(f"prepared CIF files have been saved to {args.output_tar_gz}")
