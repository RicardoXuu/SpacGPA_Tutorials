

import numpy as np
import scipy.sparse as sp
import gc

# remove_duplicate_genes
def remove_duplicate_genes(adata, tol=1e-8):
    """
    Remove duplicate genes (linearly dependent columns) from an AnnData object by comparing normalized nonzero patterns.
    
    Parameters:
        adata (AnnData): an AnnData object with expression data in .X and gene names in .var_names.
        tol (float): tolerance for numerical comparison (default 1e-8).

    Returns:
        AnnData: a new AnnData object with duplicate genes removed.
    """
    if adata.X is None or adata.var_names is None:
        raise ValueError("AnnData object must have X and var_names.")
    if not np.issubdtype(adata.X.dtype, np.number):
        raise ValueError("Expression data must be numeric.")
    
    x_matrix = adata.X
    if isinstance(x_matrix, np.matrix):
        x_csc = sp.csc_matrix(np.asarray(x_matrix))
    elif isinstance(x_matrix, np.ndarray):
        x_csc = sp.csc_matrix(x_matrix)
    else :
        x_csc = x_matrix.tocsc()
    
    del x_matrix
    gc.collect()

    var_names = adata.var_names

    seen = {}
    keep_cols = []
    
    for j in range(x_csc.shape[1]):
        col_start = x_csc.indptr[j]
        col_end = x_csc.indptr[j+1]
        row_idx = x_csc.indices[col_start:col_end]
        col_data = x_csc.data[col_start:col_end]
        
        if len(col_data) == 0:
            key = "zero"
        else:
            factor = col_data[0]
            if np.abs(factor) < tol:
                norm_data = col_data
            else:
                norm_data = col_data / factor
            norm_data = np.round(norm_data / tol) * tol
            key = (tuple(row_idx), tuple(norm_data))
        
        if key not in seen:
            seen[key] = j
            keep_cols.append(j)
        else:
            print(f"Remove Gene {var_names[j]} due to duplication with Gene {var_names[seen[key]]}")
    
    adata_new = adata[:, keep_cols].copy()

    return adata_new
