from Bio import Entrez
from Bio import SeqIO
import time

# NCBI requires an email address.
Entrez.email = "zamiljahidofficial@gmail.com"

# Updated, highly reliable queries
queries = {
    # 1. Natural: RefSeq wild-type viruses, explicitly excluding synthetic ones
    "Natural": '("Viruses"[Organism]) AND (srcdb_refseq[PROP]) NOT "synthetic"[All Fields]',

    # 2. Vector: Standard lab cloning and expression vectors (broadened for reliable fetching)
    "Vector": '"cloning vector"[Organism]',

    # 3. Ghost: Synthetic constructs containing viral elements
    "Ghost": '("synthetic construct"[Organism]) AND "virus"[All Fields]'
}


def fetch_sequences(class_name, query, count=120):
    filename = f"independent_test_set_{class_name}.txt"
    print(f"--- Fetching {class_name} sequences ---")

    try:
        # Search the database
        search_handle = Entrez.esearch(db="nucleotide", term=query, retmax=count)
        search_results = Entrez.read(search_handle)
        search_handle.close()

        id_list = search_results["IdList"]

        # SAFETY CHECK: If no IDs are found, log a warning and exit the function gracefully
        if not id_list:
            print(f"[-] Warning: No sequences found matching the query for {class_name}. Skipping download.\n")
            return

        print(f"[+] Found {len(id_list)} IDs for {class_name}. Downloading data...")

        # Fetch the actual FASTA sequences
        fetch_handle = Entrez.efetch(db="nucleotide", id=id_list, rettype="fasta", retmode="text")
        sequences = fetch_handle.read()
        fetch_handle.close()

        # Save to file
        with open(filename, "w") as file:
            file.write(sequences)

        print(f"[+] Successfully saved {class_name} sequences to {filename}\n")

    except Exception as e:
        print(f"[-] Error fetching {class_name}: {e}\n")


if __name__ == "__main__":
    # Fetch 120 of each
    for cls, qry in queries.items():
        fetch_sequences(cls, qry, count=120)
        # Polite pause to prevent API rate-limiting
        time.sleep(2)

    print("All downloads completed.")