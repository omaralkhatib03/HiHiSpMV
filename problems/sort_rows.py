import argparse
import numpy as np
from scipy.io import mmread, mmwrite
from scipy.sparse import csr_matrix

def sort_mtx_rows(input_file, output_file, method="index"):
    A = mmread(input_file).tocsr()  # Convert to CSR format
    print(f"Loaded matrix: {A.shape}, nnz = {A.nnz}")

    if method == "index":
        sorted_indices = np.arange(A.shape[0])  # No-op, default order
    elif method == "nnz":
        row_nnz = np.diff(A.indptr)
        sorted_indices = np.argsort(row_nnz)[::-1]  # Descending by NNZ
    elif method == "rowsum":
        row_sums = A.sum(axis=1).A1
        sorted_indices = np.argsort(row_sums)[::-1]  # Descending by sum
    else:
        raise ValueError("Invalid sort method. Choose from: index, nnz, rowsum")

    A_sorted = A[sorted_indices]

    mmwrite(output_file, A_sorted)
    print(f"Saved sorted matrix to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sort rows of a .mtx matrix file.")
    parser.add_argument("input", help="Input .mtx file")
    parser.add_argument("output", help="Output .mtx file")
    parser.add_argument("--method", choices=["index", "nnz", "rowsum"], default="index",
                        help="Sorting method: index (default), nnz (non-zero count), rowsum")
    args = parser.parse_args()

    sort_mtx_rows(args.input, args.output, args.method)
