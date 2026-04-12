from Bio import Entrez
import time

# Use your actual email so NCBI doesn't block your IP
Entrez.email = "zamiljahid2002@gmail.com"


def download_natural_set(accession_list, output_file):
    print(f"Connecting to NCBI to download {len(accession_list)} natural sequences...")
    try:
        # Fetching in one batch
        handle = Entrez.efetch(db="nucleotide", id=accession_list, rettype="fasta", retmode="text")
        data = handle.read()

        with open(output_file, "w") as f:
            f.write(data)

        print(f"--- SUCCESS ---")
        print(f"Created '{output_file}'")
        print(f"Total sequences added: {data.count('>')}")

    except Exception as e:
        print(f"An error occurred: {e}")


# --- EXPANDED NATURAL DATASET ---
# Including ASFV, SARS, Poxviruses, and Phages to provide a 'Natural' baseline
natural_ids = [
    # ASFV Strains (To fix your specific false positive)
    "NC_001659.2", "NC_044487.1", "AM712239.1",
    # Respiratory & Common Viruses
    "NC_007374.1", "NC_012532.1", "NC_002549.1", "NC_001405.1", "NC_045512.2",
    # Large DNA Viruses (Often confused with vectors)
    "NC_000852.5", "NC_006273.2", "NC_003310.1", "NC_001359.1",
    # Bacteriophages (Natural 'vectors')
    "NC_001416.1", "NC_001422.1", "NC_001604.1", "NC_001330.1",
    # Diverse Viral Families
    "NC_003977.2", "NC_001477.1", "NC_001489.1", "NC_001563.2",
    "NC_001436.1", "NC_001716.2", "NC_001722.1", "NC_001802.1",
    "NC_001806.2", "NC_001927.1", "NC_002645.1", "NC_003551.1",
    "NC_004102.1", "NC_004148.2", "NC_004355.1", "NC_005084.2"
]

if __name__ == "__main__":
    download_natural_set(natural_ids, "natural_viruses.fasta")