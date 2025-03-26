
import scanpy as sc
import pandas as pd
import numpy as np
import scipy.sparse as sp
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
import itertools
import warnings
from scipy.spatial.distance import pdist
from scipy.stats import skew


# construct_spatial_weights 
def construct_spatial_weights(coords, k_neighbors=6):
    """
    Construct a spatial weights matrix using kNN and 1/d as weights.
    The resulting W is NOT row-normalized.
    Diagonal entries are set to 0.
    
    Parameters:
        coords (np.array): Spatial coordinates of cells, shape (N, d).
        k_neighbors (int): Number of nearest neighbors.
        
    Returns:
        W (scipy.sparse.csr_matrix): Spatial weights matrix of shape (N, N).
    """
    N = coords.shape[0]
    nbrs = NearestNeighbors(n_neighbors=k_neighbors, metric='euclidean').fit(coords)
    distances, indices_knn = nbrs.kneighbors(coords)
    
    # calculate weights
    with np.errstate(divide='ignore', invalid='ignore'):
        weights = 1 / distances
    weights[distances == 0] = 0
    
    # construct sparse matrix
    row_idx = np.repeat(np.arange(N), k_neighbors)
    col_idx = indices_knn.flatten()
    data_w = weights.flatten()
    W = sp.coo_matrix((data_w, (row_idx, col_idx)), shape=(N, N)).tocsr()
    # set diagonal to zero
    W.setdiag(0)
    return W

# compute_moran
def compute_moran(x, W):
    """
    Compute global Moran's I for vector x using the classical formula:
    
         I = (N / S0) * (sum_{i,j} w_{ij}(x_i - mean(x)) (x_j - mean(x)) / sum_i (x_i - mean(x))^2)
    
    Parameters:
        x (np.array): 1D expression vector for a gene, shape (N,).
        W (scipy.sparse.csr_matrix): Spatial weights matrix, shape (N, N), with zero diagonal.
    
    Returns:
        float: Moran's I value, or np.nan if variance is zero.
    """
    N = x.shape[0]
    x_bar = np.mean(x)
    z = x - x_bar
    denominator = np.sum(z ** 2)
    if denominator == 0:
        return np.nan
    S0 = W.sum()
    numerator = z.T.dot(W.dot(z))
    return (N / S0) * (numerator / denominator)


# calculate_module_expression 
def calculate_module_expression(adata, 
                                ggm_obj, 
                                ggm_key = 'ggm',
                                top_genes=30, 
                                weighted=True,
                                calculate_moran=False, 
                                embedding_key='spatial',
                                k_neighbors=6,
                                add_go_anno=3):
    """
    Calculate and store module expression in adata based on input modules.
    
    Parameters:
        adata: AnnData object containing gene expression data.
        ggm_obj: GGM object containing module information or a DataFrame with columns 'module_id', 'gene', 'degree', and 'rank'.
        ggm_key: str, key for storing all information about the GGM object in adata.
                 Default 'ggm'. If not 'ggm', all information will be prefixed with ggm_key.
        top_genes: int, Number of top genes to use for module expression calculation.
        weighted: bool, Whether to calculate weighted average expression based on gene degree.
        calculate_moran: bool, if True, compute global Moran's I for each gene in module_info.
        embedding_key: str, key in adata.obsm for spatial coordinates.
        k_neighbors: int, number of nearest neighbors for constructing spatial weights.
        add_go_anno: int, default as 3. If the value is greater than 0 (and between 0 and 10), 
                     for each module, extract the top GO terms from ggm.go_enrichment and integrate them into module_df.
    """
    # get module information
    if isinstance(ggm_obj, pd.DataFrame):
        module_df = ggm_obj.copy()
    else:
        if ggm_obj.modules is None:
            raise ValueError("No modules found in the GGM object. Please run `find_modules` first.")
        module_df = ggm_obj.modules.copy()
    
    if calculate_moran and embedding_key not in adata.obsm:
        raise ValueError(f"{embedding_key} coordinates not found in adata.obsm.")
     
    print(f"\nCalculating module expression using top {top_genes} genes...")
    
    if 'ggm_keys' not in adata.uns:
        adata.uns['ggm_keys'] = {}
    if ggm_key in adata.uns['ggm_keys']:
        print(f"\n{ggm_key} already exists in adata.uns['ggm_keys']. Overwriting all information about {ggm_key}...")
    
    # 1. Filter out genes with rank larger than top_genes
    module_df = module_df[module_df['rank'] <= top_genes].copy()

    # 2. Calculate the weights of each gene in the input module
    if weighted:
        print("\nCalculating gene weights based on degree...")
        module_df['weight']  = module_df.groupby('module_id')['degree'].transform(lambda x: x / x.sum())
    else:
        print("\nUsing unweighted gene expression...")
        module_df['weight'] = module_df.groupby('module_id')['degree'].transform(lambda x: 1 / x.size) 
    
    # 3. Filter the input modules to keep only genes that exist in adata
    genes_in_adata = adata.var_names
    module_df = module_df[module_df['gene'].isin(genes_in_adata)]
    module_df.index = range(module_df.shape[0])
    
    # 4. Set the keys for storing module information and expression
    if ggm_key == 'ggm':
        mod_info_key = 'module_info'
        mod_stats_key = 'module_stats'
        expr_key = 'module_expression'
        expr_scaled_key = 'module_expression_scaled'
        col_prefix = ''
    else:
        mod_info_key = f"{ggm_key}_module_info"
        mod_stats_key = f"{ggm_key}_module_stats"
        expr_key = f"{ggm_key}_module_expression"
        expr_scaled_key = f"{ggm_key}_module_expression_scaled"
        col_prefix = f"{ggm_key}_"
    
    module_df['module_id'] = col_prefix + module_df['module_id'].astype(str)
    
    # (optional) Add GO annotation to module information
    if add_go_anno:
        try:
            add_go_anno = int(add_go_anno)
        except Exception as e:
            raise ValueError("Parameter add_go_anno must be an integer between 0 and 10.") from e
        if add_go_anno < 0 or add_go_anno > 10:
            raise ValueError("Parameter add_go_anno must be between 0 and 10.")
        for i in range(1, add_go_anno+1):
            module_df[f"top_{i}_go_term"] = None
        
        # get GO enrichment information
        go_enrichment_df = None
        if isinstance(ggm_obj, pd.DataFrame):
            print("Warning: ggm_obj is a DataFrame; no GO enrichment information available.")
        else:
            if hasattr(ggm_obj, 'go_enrichment'):
                go_enrichment_df = ggm_obj.go_enrichment
            else:
                print("Warning: ggm_obj does not have go_enrichment attribute; skipping GO annotation integration.")
        
        if go_enrichment_df is not None:
            # add prefix to module_id column
            if col_prefix and 'module_id' in go_enrichment_df.columns:
                go_enrichment_df['module_id'] = col_prefix + go_enrichment_df['module_id'].astype(str)
            
            # find top GO terms for each module
            unique_modules = go_enrichment_df['module_id'].unique()
            for mod in unique_modules:
                module_gene_set = set(module_df.loc[module_df['module_id'] == mod, 'gene'])
                mod_go_df = go_enrichment_df[go_enrichment_df['module_id'] == mod].sort_values(by='go_rank')
                for i in range(1, add_go_anno+1):
                    if i-1 < len(mod_go_df):
                        row = mod_go_df.iloc[i-1]
                        genes_str = row['genes_with_go_in_module']
                        if isinstance(genes_str, str):
                            go_genes = set(genes_str.split('/'))
                        else:
                            go_genes = set()
                        if len(go_genes & module_gene_set) > 0:
                            go_term_val = row['go_term']
                        else:
                            go_term_val = None
                        module_df.loc[module_df['module_id'] == mod, f"top_{i}_go_term"] = go_term_val
                    else:
                        module_df.loc[module_df['module_id'] == mod, f"top_{i}_go_term"] = None

    # 5. Make a mapping from gene to index in adata and from module ID to index in the transformation matrix
    gene_to_index = {gene: i for i, gene in enumerate(adata.var_names)}
    module_ids = module_df['module_id'].unique()
    module_to_index = {module: i for i, module in enumerate(module_ids)}
    
    # Remove existing module-related columns in adata.obs
    for col in list(adata.obs.columns):
        if col.startswith(f'{col_prefix}M') and (col.endswith('_exp') or col.endswith('_anno') or col.endswith('_anno_smooth')):
            adata.obs.drop(columns=col, inplace=True)

    # 6. Construct a transformation matrix
    n_genes = len(adata.var_names)
    n_modules = len(module_ids)
    transformation_matrix = sp.lil_matrix((n_genes, n_modules), dtype=np.float32)
    
    for _, row in module_df.iterrows():
        gene_idx = gene_to_index[row['gene']]
        module_idx = module_to_index[row['module_id']]
        transformation_matrix[gene_idx, module_idx] = row['weight']
    
    transformation_matrix = transformation_matrix.tocsr()
    
    # 7. Multiply adata by the transformation matrix to obtain the weighted-average-expression matrix
    weighted_expression = adata.X.dot(transformation_matrix)
    if sp.issparse(weighted_expression):
        weighted_expression = weighted_expression.toarray()
    
    # 8. Calculate the Moran's I for each gene in module_info (如需计算)
    if calculate_moran:
        coords = adata.obsm[embedding_key]
        W = construct_spatial_weights(coords, k_neighbors=k_neighbors)
        print("Calculating Moran's I for modules in ggm...")
        module_moran = {}
        for mod in module_ids:
            mod_expr = weighted_expression[:, module_to_index[mod]]
            I_mod = compute_moran(mod_expr, W)
            module_moran[mod] = I_mod
        print("Calculating Moran's I for genes in ggm...")
        moran_values = []
        for gene in module_df['gene']:
            i = gene_to_index[gene]
            if sp.issparse(adata.X):
                x_gene = adata.X[:, i].toarray().flatten()
            else:
                x_gene = adata.X[:, i]
            I_gene = compute_moran(x_gene, W)
            moran_values.append(I_gene)
        module_df['module_moran_I'] = module_df['module_id'].map(module_moran)
        module_df['gene_moran_I'] = moran_values        

    # 9. Store module information in adata.uns
    print(f"Storing module information in adata.uns['{mod_info_key}']...")
    adata.uns[mod_info_key] = module_df.copy()
   
    # 10. Store the weighted-average-expression in both obsm and obs
    scaler = StandardScaler()
    adata.obsm[expr_key] = weighted_expression
    adata.obsm[expr_scaled_key] = scaler.fit_transform(weighted_expression)
    
    weighted_expression_df = pd.DataFrame(
        weighted_expression,
        index=adata.obs_names,
        columns=[f'{mod}_exp' for mod in module_ids]
    )
    obs_df = pd.concat([adata.obs, weighted_expression_df], axis=1)
    obs_df = obs_df.loc[:, ~obs_df.columns.duplicated(keep='last')]
    adata.obs = obs_df.copy()
    
    adata.uns['ggm_keys'][ggm_key] = {
        'module_info': mod_info_key,
        'module_stats': mod_stats_key,
        'module_expression': expr_key,
        'module_expression_scaled': expr_scaled_key,
        'module_obs_prefix': col_prefix
    }
    print(f"\nTotal {n_modules} modules' average expression calculated and stored in adata.obs and adata.obsm")


# calculate_gmm_annotations
def calculate_gmm_annotations(adata, 
                              ggm_key='ggm',
                              modules_used=None,
                              modules_excluded=None,
                              embedding_key='spatial',
                              k_neighbors=6,
                              max_iter=200,
                              prob_threshold=0.99,
                              min_samples=10,
                              n_components=3,
                              enable_fallback=True,
                              random_state=42):
    """
    Gaussian Mixture Model annotation with additional module-level statistics.
    
    Statistics added to mod_stats_key include:
        - module_id: Module ID.
        - status: 'success' or 'failed'.
        - anno_one: Number of cells annotated as 1.
        - anno_zero: Number of cells expressed module but not annotated as 1.
        - skew: Skewness of the non-zero expression distribution for the module.
        - top1pct_ratio: Ratio of the average expression among the top 1% high-expressing cells
                        to the overall cells mean.
        - module_moran_I: Global Moran's I computed on the module expression (all cells).
        - positive_moran_I: Moran's I computed on module expression for cells annotated as 1.
        - negative_moran_I: Moran's I computed on module expression for cells annotated as 0.
        - positive_mean_distance: Average pairwise spatial distance among cells annotated as 1.
        - n_components: Number of components in the GMM.
        - final_components: Number of components after fallback.
        - threshold: Threshold for calling a cell positive.
        - components: List of dictionaries with keys 'component', 'mean', 'var', 'weight'.
        - main_component: Index of the main component.
        - error_info: Error message if status is 'failed'.
        - top_go_terms: (新增) 如果检测到GO注释信息，则拼接每个模块在 adata.uns['module_info'] 中所有top_***_go_term 列的内容，以“ || ”分隔.
    
    Parameters:
      adata: AnnData object.
      ggm_key: Key for the GGM object in adata.uns['ggm_keys'].
      modules_used: List of module IDs to process; if None, use all modules in adata.uns[mod_info_key].
      modules_excluded: List of module IDs to exclude.
      embedding_key: Key in adata.obsm containing spatial coordinates.
      k_neighbors: Number of nearest neighbors for spatial weight matrix.
      max_iter: Maximum iterations for GMM.
      prob_threshold: Probability threshold for calling a cell positive.
      min_samples: Minimum number of nonzero samples required.
      n_components: Number of GMM components.
      enable_fallback: Whether to fallback to a 2-component model on failure.
      random_state: Random seed.
    
    Returns:
      Updates adata.obs with annotation columns (categorical, with suffix '_anno'),
      and stores module-level statistics in adata.uns[mod_stats_key].
    """

    # Retrieve keys from adata.uns['ggm_keys']
    ggm_keys = adata.uns.get('ggm_keys', {})
    if ggm_key not in ggm_keys:
        raise ValueError(f"{ggm_key} not found in adata.uns['ggm_keys']")
    mod_info_key = ggm_keys[ggm_key].get('module_info')
    mod_stats_key = ggm_keys[ggm_key].get('module_stats')
    expr_key = ggm_keys[ggm_key].get('module_expression')
    if expr_key not in adata.obsm:
        raise ValueError(f"{expr_key} not found in adata.obsm")
    if mod_info_key not in adata.uns:
        raise ValueError(f"{mod_info_key} not found in adata.uns")
    
    # Extract module expression matrix and module information
    module_expr_matrix = pd.DataFrame(adata.obsm[expr_key], index=adata.obs.index)
    unique_mods = adata.uns[mod_info_key]['module_id'].unique()
    if len(unique_mods) != module_expr_matrix.shape[1]:
        raise ValueError(f"module_info and module_expression dimensions for the ggm '{ggm_key}' do not match")
    else:
        module_expr_matrix.columns = unique_mods
    
    # Determine modules to use
    if modules_used is None and modules_excluded is None and mod_stats_key in adata.uns:
       adata.uns.pop(mod_stats_key, None)

    if modules_used is None:
        modules_used = list(unique_mods)
    if modules_excluded is not None:
        modules_used = [mid for mid in modules_used if mid not in modules_excluded]
    valid_modules = [mid for mid in modules_used if mid in module_expr_matrix.columns]
    if not valid_modules:
        raise ValueError(f"Ensure that the input module IDs exist in adata.uns['{mod_info_key}']")
    
    # Remove existing annotation columns
    for col in list(adata.obs.columns):
        if col.endswith('_anno') and any(col.startswith(mid) for mid in valid_modules):
            adata.obs.drop(columns=col, inplace=True)
    
    # Initialize annotation matrix (0/1) for modules
    anno_cols = valid_modules
    annotations = pd.DataFrame(np.zeros((adata.obs.shape[0], len(anno_cols)), dtype=int),
                               index=adata.obs.index, columns=anno_cols)
    stats_records = []
    
    # Pre-calculate expression ranking for tie-breaks
    expr_score = {}
    for mid in valid_modules:
        module_col = f"{mid}_exp"
        if module_col not in adata.obs.columns:
            raise KeyError(f"'{module_col}' not found in adata.obs.")
        rank_vals = adata.obs[module_col].rank(method='dense', ascending=False).astype(int)
        expr_score[mid] = rank_vals.values
    
    # Construct spatial weights matrix W based on embedding_key
    if embedding_key not in adata.obsm:
        raise ValueError(f"{embedding_key} not found in adata.obsm")
    coords = adata.obsm[embedding_key]
    nbrs = NearestNeighbors(n_neighbors=k_neighbors, metric="euclidean").fit(coords)
    distances, knn_indices = nbrs.kneighbors(coords)
    with np.errstate(divide='ignore'):
        weights = 1 / distances
    weights[distances == 0] = 0
    row_idx = np.repeat(np.arange(coords.shape[0]), k_neighbors)
    col_idx = knn_indices.flatten()
    data_w = weights.flatten()
    W = sp.coo_matrix((data_w, (row_idx, col_idx)), shape=(coords.shape[0], coords.shape[0])).tocsr()
    W.setdiag(0)
    
    # Build mapping from gene to index using adata.var_names
    gene_to_index = {gene: i for i, gene in enumerate(adata.var_names)}
    
    # Process each module for GMM annotation and extra statistics
    for module_id in valid_modules:
        stats = {
            'module_id': module_id,
            'status': 'success',
            'n_components': n_components,
            'final_components': n_components,
            'threshold': np.nan,
            'anno_one': 0,
            'anno_zero': 0,
            'components': [],
            'error_info': 'None',
            'module_moran_I': np.nan,
            'positive_mean_distance': np.nan,
            'positive_moran_I': np.nan,
            'negative_moran_I': np.nan,
            'skew': np.nan,
            'top1pct_ratio': np.nan,
            'effect_size': np.nan
        }
        try:
            expr_values = module_expr_matrix[module_id].values
            non_zero_mask = expr_values != 0
            non_zero_expr = expr_values[non_zero_mask]
            if len(non_zero_expr) == 0:
                raise ValueError("all_zero_expression")
            if len(non_zero_expr) < min_samples:
                raise ValueError(f"insufficient_samples ({len(non_zero_expr)} < {min_samples})")
            if np.var(non_zero_expr) < 1e-6:
                raise ValueError("zero_variance")
            
            # Fit GMM on non-zero expression
            gmm = GaussianMixture(n_components=n_components, random_state=random_state, max_iter=max_iter)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                gmm.fit(non_zero_expr.reshape(-1, 1))
            means = gmm.means_.flatten()
            main_component = int(np.argmax(means))
            main_mean = means[main_component]
            probs = gmm.predict_proba(non_zero_expr.reshape(-1, 1))[:, main_component]
            anno_non_zero = (probs >= prob_threshold).astype(int)
            if np.sum(anno_non_zero) == 0:
                raise ValueError("no_positive_cells")
            positive_expr = non_zero_expr[anno_non_zero == 1]
            threshold = float(np.min(positive_expr))
            if n_components >= 3 and threshold > main_mean:
                raise ValueError(f"threshold {threshold:.2f} > μ ({main_mean:.2f})")
            stats.update({
                'threshold': threshold,
                'anno_one': int(np.sum(anno_non_zero)),
                'anno_zero': int(len(anno_non_zero) - np.sum(anno_non_zero)),
                'components': [
                    {'component': i,
                     'mean': float(gmm.means_[i][0]),
                     'var': float(gmm.covariances_[i][0][0]),
                     'weight': float(gmm.weights_[i])
                    } for i in range(n_components)
                ],
                'main_component': main_component
            })
            
            # Build full annotation vector for module: default 0, update non-zero positions
            module_annotation = np.zeros_like(expr_values, dtype=int)
            module_annotation[non_zero_mask] = anno_non_zero
            annotations[module_id] = module_annotation
            
            # (A) Module-level Moran's I: computed on the entire module expression vector
            mod_I = compute_moran(expr_values, W)
            stats['module_moran_I'] = mod_I
            
            # (B) Positive annotation: Construct masked expression vector (0 where 0, expr where 1)
            pos_expr_masked = np.where(module_annotation == 1, expr_values, 0)
            stats['positive_moran_I'] = compute_moran(pos_expr_masked, W)
            # Calculate mean distance among positive cells
            full_indices = np.where(non_zero_mask)[0]
            pos_idx = full_indices[anno_non_zero == 1]
            if len(pos_idx) > 1:
                pos_coords = coords[pos_idx, :]
                stats['positive_mean_distance'] = float(np.mean(pdist(pos_coords)))
            else:
                stats['positive_mean_distance'] = np.nan
            
            # (C) Negative annotation: Construct masked expression vector (0 where 0, expr where 0)
            neg_expr_masked = np.where(module_annotation == 0, expr_values, 0)
            stats['negative_moran_I'] = compute_moran(neg_expr_masked, W)
            
            # (D) Skewness for non-zero expression
            stats['skew'] = float(skew(non_zero_expr))
            
            # (E) Top 1% ratio: mean of top 1% high-expressing cells / overall mean
            if len(expr_values) > 0:
                top_n = max(1, int(len(expr_values) * 0.01))
                sorted_expr = np.sort(expr_values)
                top1_mean = np.mean(sorted_expr[-top_n:])
                overall_mean = np.mean(expr_values)
                stats['top1pct_ratio'] = top1_mean / overall_mean if overall_mean != 0 else np.nan
            else:
                stats['top1pct_ratio'] = np.nan
            # (F) Effect size: mean of positive cells - mean of negative cells
            if len(positive_expr) > 0:
                std_all = np.std(non_zero_expr)
                neg_expr = non_zero_expr[anno_non_zero == 0]
                stats['effect_size'] = float(np.mean(positive_expr) - np.mean(neg_expr)) / std_all if std_all != 0 else np.nan
            else:
                stats['effect_size'] = np.nan
            
        except Exception as e:
            stats.update({
                'status': 'failed',
                'error_info': str(e),
                'components': [],
                'threshold': np.nan,
                'anno_one': 0,
                'anno_zero': expr_values.size if 'expr_values' in locals() else 0,
                'module_moran_I': np.nan,
                'positive_mean_distance': np.nan,
                'positive_moran_I': np.nan,
                'negative_moran_I': np.nan,
                'skew': np.nan,
                'top1pct_ratio': np.nan,
                'effect_size': np.nan
            })
            if enable_fallback and n_components > 2:
                try:
                    gmm = GaussianMixture(n_components=2, random_state=random_state, max_iter=max_iter)
                    gmm.fit(non_zero_expr.reshape(-1, 1))
                    means = gmm.means_.flatten()
                    main_component = int(np.argmax(means))
                    probs = gmm.predict_proba(non_zero_expr.reshape(-1, 1))[:, main_component]
                    anno_non_zero = (probs >= ((1 - (1 - prob_threshold) * 1e-2))).astype(int)
                    if np.sum(anno_non_zero) > 0:
                        positive_expr = non_zero_expr[anno_non_zero == 1]
                        threshold = float(np.min(positive_expr))
                        stats.update({
                            'status': 'success',
                            'final_components': 2,
                            'threshold': threshold,
                            'anno_one': int(np.sum(anno_non_zero)),
                            'anno_zero': int(len(anno_non_zero) - np.sum(anno_non_zero)),
                            'components': [
                                {'component': 0,
                                 'mean': float(gmm.means_[0][0]),
                                 'var': float(gmm.covariances_[0][0][0]),
                                 'weight': float(gmm.weights_[0])
                                },
                                {'component': 1,
                                 'mean': float(gmm.means_[1][0]),
                                 'var': float(gmm.covariances_[1][0][0]),
                                 'weight': float(gmm.weights_[1])
                                }
                            ],
                            'main_component': main_component
                        })
                        # Update annotation after fallback
                        fallback_annotation = np.zeros_like(expr_values, dtype=int)
                        fallback_annotation[non_zero_mask] = anno_non_zero
                        annotations.loc[non_zero_mask, module_id] = anno_non_zero
                        # Module-level Moran's I
                        fallback_mod_I = compute_moran(expr_values, W)
                        stats['module_moran_I'] = fallback_mod_I
                        # Positive group (using masked expression)
                        pos_expr_masked = np.where(fallback_annotation == 1, expr_values, 0)
                        stats['positive_moran_I'] = compute_moran(pos_expr_masked, W)
                        full_indices = np.where(non_zero_mask)[0]
                        pos_idx = full_indices[anno_non_zero == 1]
                        if len(pos_idx) > 1:
                            pos_coords = coords[pos_idx, :]
                            stats['positive_mean_distance'] = float(np.mean(pdist(pos_coords)))
                        else:
                            stats['positive_mean_distance'] = np.nan
                        # Negative group
                        neg_expr_masked = np.where(fallback_annotation == 0, expr_values, 0)
                        stats['negative_moran_I'] = compute_moran(neg_expr_masked, W)
                        # Skewness and top1pct_ratio
                        stats['skew'] = float(skew(non_zero_expr))
                        if len(non_zero_expr) > 0:
                            top_n = max(1, int(len(non_zero_expr) * 0.01))
                            sorted_expr = np.sort(non_zero_expr)
                            top1_mean = np.mean(sorted_expr[-top_n:])
                            overall_mean = np.mean(non_zero_expr)
                            stats['top1pct_ratio'] = top1_mean / overall_mean if overall_mean != 0 else np.nan
                        else:
                            stats['top1pct_ratio'] = np.nan
                        # Effect size
                        if len(positive_expr) > 0:
                            std_all = np.std(non_zero_expr)
                            neg_expr = non_zero_expr[anno_non_zero == 0]
                            stats['effect_size'] = float(np.mean(positive_expr) - np.mean(neg_expr)) / std_all if std_all != 0 else np.nan
                        else:
                            stats['effect_size'] = np.nan
                except Exception as fallback_e:
                    stats['error_info'] += f"; Fallback failed: {str(fallback_e)}"
        finally:
            if stats['status'] == 'success':
                print(f"{module_id} processed successfully, annotated cells: {stats['anno_one']}")
            else:
                print(f"{module_id} processed, failed: {stats['error_info']}")
            stats['components'] = str(stats['components'])
            stats_records.append(stats)
    
    # Transform annotations to categorical and store in adata.obs
    annotations.columns = [f"{col}_anno" for col in annotations.columns]
    for col in annotations.columns:
        orig_name = col.replace("_anno", "")
        annotations[col] = np.where(annotations[col] == 1, orig_name, None)
        annotations[col] = pd.Categorical(annotations[col])
    adata.obs = pd.concat([adata.obs, annotations], axis=1)
    
    # Add GO annotations to module_stats_key 
    stats_records_df = pd.DataFrame(stats_records)
    module_info_df = adata.uns[mod_info_key]
    go_cols = [col for col in module_info_df.columns if col.startswith("top_") and col.endswith("_go_term")]
    if len(go_cols) > 0:
        def concat_go_terms(mod_id):
            rows = module_info_df[module_info_df['module_id'] == mod_id]
            terms = []
            for col in go_cols:
                vals = rows[col].dropna().unique().tolist()
                if vals:
                    terms.extend(vals)
            if terms:
                return " || ".join(sorted(set(terms)))
            else:
                return None
        stats_records_df["top_go_terms"] = stats_records_df["module_id"].apply(concat_go_terms)
        # Set the order of columns in stats_records_df
        new_order = [
            'module_id', 'status', 'anno_one', 'anno_zero', 'top_go_terms', 'skew', 'top1pct_ratio', 
            'module_moran_I', 'positive_moran_I', 'negative_moran_I', 'positive_mean_distance','effect_size',
            'n_components', 'final_components','threshold', 'components', 'main_component', 'error_info']
        stats_records_df = stats_records_df[new_order]
    else:
        new_order = [
            'module_id', 'status', 'anno_one', 'anno_zero', 'skew', 'top1pct_ratio',
            'module_moran_I', 'positive_moran_I', 'negative_moran_I', 'positive_mean_distance','effect_size',
            'n_components', 'final_components','threshold', 'components', 'main_component', 'error_info']
        stats_records_df = stats_records_df[new_order]
    
    if mod_stats_key in adata.uns:
        existing_stats = adata.uns[mod_stats_key]
        for mid in stats_records_df['module_id'].unique():
            new_row = stats_records_df.loc[stats_records_df['module_id'] == mid].iloc[0]
            mask = existing_stats['module_id'] == mid
            if mask.any():
                num_rows = mask.sum()
                new_update_df = pd.DataFrame([new_row] * num_rows, index=existing_stats.loc[mask].index)
                existing_stats.loc[mask] = new_update_df
            else:
                existing_stats = pd.concat([existing_stats, pd.DataFrame([new_row])], ignore_index=True)
        existing_stats.dropna(how='all', inplace=True)
        adata.uns[mod_stats_key] = existing_stats
    else:
        adata.uns[mod_stats_key] = stats_records_df
    
    return

# smooth_annotations
def smooth_annotations(adata,
                       ggm_key='ggm', 
                       modules_used=None,
                       modules_excluded=None, 
                       embedding_key='spatial', k_neighbors=24, min_annotated_neighbors=1):
    """
    Smooth spatial annotations by processing each module's annotation.
    
    Parameters:
      adata (anndata.AnnData): AnnData object containing spatial transcriptomics data.
      ggm_key (str): Key for the GGM object in adata.uns['ggm_keys'].
      modules_used (list): List of modules to smooth; if None, all modules are used.(default None)
      modules_excluded (list): List of modules to exclude from smoothing (default None).
      embedding_key (str): Key in adata.obsm for spatial coordinates (default 'spatial').
      k_neighbors (int): Number of KNN neighbors (default 24); may need adjustment based on technology and cell density.
      min_annotated_neighbors (int): Minimum number of neighbors with annotation 1 required to retain the annotation (default 1).
    """
    if ggm_key not in adata.uns['ggm_keys']:
        raise ValueError(f"{ggm_key} not found in adata.uns['ggm_keys']")
    
    module_stats_key = adata.uns['ggm_keys'][ggm_key]['module_stats']

    # Check input: ensure the embedding key exists in adata.obsm
    if embedding_key not in adata.obsm:
        raise ValueError(f"{embedding_key} not found in adata.obsm. Please ensure the coordinate exists.")
    
    # If modules_used is not provided, get all modules from adata.uns['module_stats']
    if modules_used is None:
        modules_used = adata.uns[module_stats_key]['module_id'].unique()
    # Exclude modules if the modules_excluded list is provided
    if modules_excluded is not None:
        modules_used = [mid for mid in modules_used if mid not in modules_excluded]
                        
    # Remove existing smoothed annotation columns if they exist
    existing_columns = [f"{mid}_anno_smooth" for mid in modules_used if f"{mid}_anno_smooth" in adata.obs]
    if existing_columns:
        print(f"Removing existing smooth annotation columns: {existing_columns}")
        adata.obs.drop(columns=existing_columns, inplace=True)

    # Extract spatial coordinates and the annotation columns to be smoothed
    embedding_coords = adata.obsm[embedding_key]
    module_annotations = adata.obs.loc[:, [f"{mid}_anno" for mid in modules_used]]
    # Reset the module id anno to 0/1 anno
    for col in module_annotations.columns:
        orig_name = col.replace("_anno", "")
        module_annotations[col] = (module_annotations[col] == orig_name).astype(int)

        n_cells, _ = module_annotations.shape

    # Compute KNN neighbors based on the embedding coordinates
    print(f"\nCalculating {k_neighbors} nearest neighbors for each cell based on {embedding_key} embedding...\n")
    k = k_neighbors + 1  # include self
    nbrs = NearestNeighbors(n_neighbors=k, metric='euclidean').fit(embedding_coords)
    _, indices = nbrs.kneighbors(embedding_coords)  # indices: KNN indices for each cell

    # Initialize the smoothed annotation matrix
    smooth_annotations = np.zeros_like(module_annotations, dtype=int)

    # Smooth each module's annotation
    for mid in module_annotations.columns:
        module_values = module_annotations[mid].values  # current module's annotation values
        smooth_values = np.zeros(n_cells, dtype=int)  # initialize smoothed values

        for i in range(n_cells):
            if module_values[i] == 1:  # only process cells with annotation 1
                neighbor_values = module_values[indices[i, 1:]]  # KNN neighbor annotations (excluding self)
                if np.sum(neighbor_values) >= min_annotated_neighbors:  # if at least min_annotated_neighbors neighbors are 1
                    smooth_values[i] = 1

        # Store the smoothed values in the matrix
        smooth_annotations[:, module_annotations.columns.get_loc(mid)] = smooth_values
        print(f"{mid.replace('_anno', '')} processed. removed cells: {np.sum(module_values) - np.sum(smooth_values)}, remain cells: {np.sum(smooth_values)}")

    # Save the smoothed annotations to adata.obs
    smooth_annotations = pd.DataFrame(smooth_annotations,
                                      index=adata.obs_names,
                                      columns=[f"{mid}_smooth" for mid in module_annotations.columns])
    # Reset the 0/1 anno to module id or None
    for col in smooth_annotations.columns:
        orig_name = col.replace("_anno_smooth", "")
        smooth_annotations[col] = np.where(smooth_annotations[col] == 1, orig_name, None)
        smooth_annotations[col] = pd.Categorical(smooth_annotations[col])
    adata.obs = pd.concat([adata.obs, smooth_annotations], axis=1)

    print("\nAnnotation smoothing completed. Results stored in adata.obs.\n")


# integrate_annotations
def integrate_annotations(adata,
                          ggm_key='ggm',
                          cross_ggm = False,
                          modules_used=None,
                          modules_excluded=None,
                          modules_preferred=None,
                          result_anno='annotation',
                          use_smooth=True,
                          embedding_key='spatial',
                          k_neighbors=24,
                          neighbor_similarity_ratio=0.90
                          ):            
    """
    Integrate cell annotations from multiple modules using the following logic:
      1) Optionally use smoothed annotations (controlled by use_smooth);
      2) Automatically compute k_neighbors nearest neighbors;
      3) For cells annotated by multiple modules:
            3.1 If There are modules_preferred and the cell has these modules in its potential annotation,
                the cell will be given priority to be annotated with these modules. 
            3.2 If the fraction of a cell's neighbors annotated with a particular module exceeds 
                the neighbor_similarity_ratio threshold, the cell is assigned that module.
            3.3 Otherwise, the final annotation is determined based on expression scores, where 
                higher expression scores confer higher priority.
    
    Parameters:
      adata (anndata.AnnData): AnnData object containing module annotations.
      ggm_key (str): Key for the GGM object in adata.uns['ggm_keys'].
      cross_ggm (bool): Whether to integrate annotations from multiple GGMs (default False).
                        When True, modules_used must be provided manually.
      modules_used (list): List of modules to integrate; if None, all modules of the GGM mentioned in ggm_key will be used.
      modules_excluded (list): List of modules to exclude from integration (default None).
      modules_preferred (list): List of preferred modules; if a cell has these modules in its potential annotation, 
                                the cell will be given priority to be annotated with these modules.
      result_anno (str): Column name for the integrated annotation (default 'annotation').
      use_smooth (bool): Whether to use smoothed annotations (default True).
      embedding_key (str): Key in adata.obsm for KNN coordinates (default 'spatial').
      k_neighbors (int): Number of KNN neighbors (default 24); may need adjustment based on technology and cell density.
      neighbor_similarity_ratio (float): A threshold representing the proportion of a cell's annotation that must match 
                                         its neighboring cells' annotations. 
                                         If the matching ratio exceeds this threshold, the cell will be directly annotated 
                                         with its neighbor's annotation. (Valid range: [0, 1]. default 0.90.)
    """
    if ggm_key not in adata.uns['ggm_keys']:
        raise ValueError(f"{ggm_key} not found in adata.uns['ggm_keys']")
    mod_stats_key = adata.uns['ggm_keys'][ggm_key]['module_stats']

    if cross_ggm and len(adata.uns['ggm_keys']) > 2: 
        if modules_used is None:
            raise ValueError("When cross_ggm is True, modules_used must be provided manually.")

    # Check input: ensure embedding_key exists in adata.obsm.
    if neighbor_similarity_ratio < 0 or neighbor_similarity_ratio > 1:
        raise ValueError("neighbor_similarity_ratio must be within [0, 1].")
    if neighbor_similarity_ratio == 0:
        print("When set to 0, the neighbor_similarity_ratio will not be used.")

    if  embedding_key not in adata.obsm:
        raise ValueError(f"{embedding_key} not found in adata.obsm. Please ensure the coordinate exists.")
    
    # Check if the integrated annotation column already exists; if so, remove it.
    if adata.obs.get(result_anno) is not None:
        print(f"NOTE: The '{result_anno}' already exists in adata.obs, which will be overwritten.")
        adata.obs.drop(columns=result_anno, inplace=True)
    
    # Check and extract annotation columns.
    # If modules_used is not provided, use all modules in adata.uns[mod_stats_key].
    if modules_used is None:
        modules_used = adata.uns[mod_stats_key]['module_id'].unique()
    # Exclude modules if the modules_excluded list is provided.
    if modules_excluded is not None:
        modules_used = [mid for mid in modules_used if mid not in modules_excluded]

    if use_smooth:
        # Identify modules missing smoothed annotation columns.
        missing_smooth = [mid for mid in modules_used if f"{mid}_anno_smooth" not in adata.obs]
        if missing_smooth:
            print(f"\nThese modules do not have 'Smooth anno': {missing_smooth}. Using 'anno' instead.")
        # Use smoothed columns if available; otherwise, use the original '_anno' columns.
        existing_columns = []
        for mid in modules_used:
            if f"{mid}_anno_smooth" in adata.obs:
                existing_columns.append(f"{mid}_anno_smooth")
            elif f"{mid}_anno" in adata.obs:
                existing_columns.append(f"{mid}_anno")
    else:
        existing_columns = [f"{mid}_anno" for mid in modules_used if f"{mid}_anno" in adata.obs]
    
    if len(existing_columns) != len(modules_used):
        raise ValueError("The annotation columns for the specified modules do not fully match those in adata.obs. Please check your input.")
    
    
    # 1) Compute nearest neighbors.
    print(f"\nCalculating {k_neighbors} nearest neighbors for each cell based on {embedding_key} embedding...\n")
    embedding_coords = adata.obsm[embedding_key]
    k = k_neighbors + 1  # include self
    nbrs = NearestNeighbors(n_neighbors=k, metric="euclidean").fit(embedding_coords)
    _, indices = nbrs.kneighbors(embedding_coords)

    n_obs = adata.n_obs
    combined_annotation = [None] * n_obs  # Store final integrated annotation for each cell.

    # 2) Pre-calculate expression scores based on ranked expression.
    expr_score = {}
    anno_dict = {}
    for mid in modules_used:
        module_col = f"{mid}_exp"
        if module_col not in adata.obs.columns:
            raise KeyError(f"'{module_col}' not found in adata.obs.")        
        rank_vals = adata.obs[module_col].rank(method='dense', ascending=False).astype(int)
        expr_score[mid] = rank_vals.values
    
    # 3) Build annotation mask.
    for mid in modules_used:
        if f"{mid}_anno" in existing_columns:
            anno_col = f"{mid}_anno"
        else:
            anno_col = f"{mid}_anno_smooth"    
        anno_dict[mid] = (adata.obs[anno_col] == mid).astype(int).values
    
    unclear_cells = 0
    unclear_cells_neighbor = 0
    unclear_cells_rank = 0
    # 3) Integrate annotations for each cell.
    for i in range(n_obs):
        # Find which modules annotate the cell.
        annotated_modules = [p for p in modules_used if anno_dict[p][i]]
        if len(annotated_modules) > 1:
            unclear_cells += 1

        # If modules_preferred is provided, take the intersection (prioritize these modules).
        if modules_preferred is not None:
            intersection = [p for p in annotated_modules if p in modules_preferred]
            if intersection:
                annotated_modules = intersection
        
        if len(annotated_modules) == 0:
            combined_annotation[i] = None
            continue
        if len(annotated_modules) == 1:
            combined_annotation[i] = annotated_modules[0]
            continue
        
        # Multiple annotations => Step 1: neighbor voting.
        neighbor_idx = indices[i, 1:]  # exclude self
        n_nb = len(neighbor_idx)

        # Count annotations among neighbors for each module.
        neighbor_counts = {}
        for mid in annotated_modules:
            neighbor_anno = anno_dict[mid][neighbor_idx]
            neighbor_counts[mid] = np.sum(neighbor_anno)

        # Check if any module meets the neighbor majority fraction.
        annotated_modules_re = []
        for mid in annotated_modules:
            frac = neighbor_counts[mid] / n_nb
            if frac >= neighbor_similarity_ratio:
                annotated_modules_re.append(mid)
        
        if len(annotated_modules_re) == 1:
            combined_annotation[i] = annotated_modules_re[0]
            unclear_cells_neighbor += 1
            continue
        elif len(annotated_modules_re) > 1:
            annotated_modules = annotated_modules_re    

        # Step 2: If neighbor voting is indecisive, use expression score.
        best_mid = None
        best_score = n_obs + 1  # initialize with a large value 
        for mid in annotated_modules:
            sc = expr_score[mid][i]
            if sc < best_score:
                best_score = sc
                best_mid = mid
            
        combined_annotation[i] = best_mid
        unclear_cells_rank += 1
    
    # Print statistics about cells with multiple annotations.
    print(f"\n{unclear_cells} of total {n_obs} cells have multiple annotations. Among them, ")
    print(f"    {unclear_cells_neighbor} cells are resolved by neighbors.")
    print(f"    {unclear_cells_rank} cells are resolved by expression score.")

    # Update integrated annotation in adata.obs.
    annotation_df = pd.Series(combined_annotation, index=adata.obs.index, name=result_anno)
    adata.obs[result_anno] = annotation_df
    print(f"\nIntegrated annotation stored in adata.obs['{result_anno}'].\n")


# calculate_module_overlap
def calculate_module_overlap(adata, ggm_key='ggm', cross_ggm=False,
                             modules_used=None, 
                             modules_excluded=None,
                             use_smooth=True):
    """
    Compute the overlap ratios between modules based on annotation columns in adata.obs.
    
    The returned DataFrame has the following seven columns:
      - module_a: ID of module A
      - module_b: ID of module B
      - overlap_ratio_a: Overlap count divided by the cell count of module A
      - overlap_ratio_b: Overlap count divided by the cell count of module B
      - overlap_ratio_union: Overlap count divided by the cell count of the union of module A and module B
      - count_a: Number of cells annotated by module A
      - count_b: Number of cells annotated by module B
    
    Pairs with zero overlap are omitted. The final DataFrame is sorted by module_a (ascending)
    and then by overlap_ratio_a (descending).

    Parameters:
      adata (AnnData): AnnData object. The annotation columns should exist in adata.obs with names 
                       "{module_id}_anno" or "{module_id}_anno_smooth".
      ggm_key (str): Key for the GGM object in adata.uns
      cross_ggm (bool): Whether to calculate overlaps across multiple GGMs (default False).
      modules_used (list): List of module IDs to compute overlaps.
      modules_excluded (list): List of module IDs to exclude from overlap calculation.
      use_smooth (bool): Whether to use smoothed annotation columns (default True).

    Returns:
      pd.DataFrame: DataFrame containing the overlap statistics.
    """
    if ggm_key not in adata.uns['ggm_keys']:
        raise ValueError(f"{ggm_key} not found in adata.uns['ggm_keys']")
    mod_stats_key = adata.uns['ggm_keys'][ggm_key]['module_stats']

    if cross_ggm and len(adata.uns['ggm_keys']) > 2:
        if modules_used is None:
            raise ValueError("When cross_ggm is True, modules_used must be provided manually.")

    # If modules_used is not provided, get modules from uns['module_stats']
    if modules_used is None:
        modules_used = adata.uns[mod_stats_key]['module_id'].unique()
    # Exclude modules if the modules_excluded list is provided    
    if modules_excluded is not None:
        modules_used = [mid for mid in modules_used if mid not in modules_excluded]

    # Determine which annotation column to use for each module.
    if use_smooth:
        missing_smooth = [mid for mid in modules_used if f"{mid}_anno_smooth" not in adata.obs]
        if missing_smooth:
            print(f"\nThese modules do not have 'smoothed anno': {missing_smooth}. Using 'anno' instead.")
        existing_columns = []
        for mid in modules_used:
            if f"{mid}_anno_smooth" in adata.obs:
                existing_columns.append(f"{mid}_anno_smooth")
            elif f"{mid}_anno" in adata.obs:
                existing_columns.append(f"{mid}_anno")
    else:
        existing_columns = [f"{mid}_anno" for mid in modules_used if f"{mid}_anno" in adata.obs]
    
    if len(existing_columns) != len(modules_used):
        raise ValueError("The annotation columns for the specified modules do not fully match those in adata.obs. Please check your input.")
    
    # Build a dictionary mapping module IDs to their annotation (0/1) arrays.
    anno_dict = {}
    for mid in modules_used:
        if f"{mid}_anno" in existing_columns:
            anno_col = f"{mid}_anno"
        else:
            anno_col = f"{mid}_anno_smooth"    
        anno_dict[mid] = (adata.obs[anno_col] == mid).astype(int).values
    
    overlap_records = []
    # Iterate over all pairs of modules.
    for mod_a, mod_b in itertools.combinations(modules_used, 2):
        a = anno_dict[mod_a]
        b = anno_dict[mod_b]
        count_a = np.sum(a)
        count_b = np.sum(b)
        overlap = np.sum((a == 1) & (b == 1))
        if overlap == 0:
            continue
        
        ratio_a = overlap / count_a if count_a > 0 else np.nan
        ratio_b = overlap / count_b if count_b > 0 else np.nan
        union_count = count_a + count_b - overlap
        ratio_union = overlap / union_count if union_count > 0 else np.nan
        
        overlap_records.append({
            "module_a": mod_a,
            "module_b": mod_b,
            "overlap_ratio_a": ratio_a,
            "overlap_ratio_b": ratio_b,
            "overlap_ratio_union": ratio_union,
            "count_a": count_a,
            "count_b": count_b
        })
    
    df = pd.DataFrame(overlap_records)
    df.sort_values(by=["module_a", "overlap_ratio_a"], ascending=[True, False], inplace=True)
    df.index = range(df.shape[0])
    return df




############################################################################################################
# Old functions
# integrate_annotations_old
def integrate_annotations_old(adata, ggm_key='ggm',cross_ggm = False,
                              modules_used=None, modules_excluded=None,
                              result_anno = "annotation", use_smooth=True):
    """
    Integrate module annotations into a single cell annotation.
    
    Parameters:
        adata (anndata.AnnData): AnnData object containing module annotations.
        ggm_key (str): Key for the GGM object in adata.uns['ggm_keys'].
        cross_ggm (bool): Whether to integrate annotations from multiple GGMs (default False).
        modules_used (list): List of module IDs to integrate; if None, all modules are used.
        modules_excluded (list): List of module IDs to exclude from integration (default None).
        result_anno (str): Column name for the integrated annotation (default 'annotation').
        use_smooth (bool): Whether to use smoothed annotations (default True).
    """
    if ggm_key not in adata.uns['ggm_keys']:    
        raise ValueError(f"{ggm_key} not found in adata.uns['ggm_keys']")
    mod_stats_key = adata.uns['ggm_keys'][ggm_key]['module_stats']

    if cross_ggm and len(adata.uns['ggm_keys']) > 2:
        if modules_used is None:
            raise ValueError("When cross_ggm is True, modules_used must be provided manually.")

    # Check if the integrated annotation column already exists
    if adata.obs.get(result_anno) is not None:
        print(f"NOTE: The '{result_anno}' already exists in adata.obs, which will be overwritten.")
        adata.obs.drop(columns=result_anno, inplace=True)

    # Determine and extract annotation columns
    if modules_used is None:
        modules_used = adata.uns[mod_stats_key]['module_id'].unique()
    # Exclude modules if the modules_excluded list is provided
    if modules_excluded is not None:
        modules_used = [mid for mid in modules_used if mid not in modules_excluded]

    if use_smooth:
        # Identify modules missing smoothed annotation columns
        missing_smooth = [mid for mid in modules_used if f"{mid}_anno_smooth" not in adata.obs]
        if missing_smooth:
            print(f"\nThese modules do not have 'Smooth anno': {missing_smooth}. Using 'anno' instead.")
        # For modules with smoothed columns, use them; otherwise, use the original '_anno' columns
        existing_columns = []
        for mid in modules_used:
            if f"{mid}_anno_smooth" in adata.obs:
                existing_columns.append(f"{mid}_anno_smooth")
            elif f"{mid}_anno" in adata.obs:
                existing_columns.append(f"{mid}_anno")
    else:
        existing_columns = [f"{mid}_anno" for mid in modules_used if f"{mid}_anno" in adata.obs]
    
    if len(existing_columns) != len(modules_used):
        raise ValueError("The annotation columns for the specified modules do not fully match those in adata.obs. Please check your input.")
    
    
    # Extract module annotations
    module_annotations = adata.obs.loc[:, [mid for mid in existing_columns]]
    # Reset the module id anno to 0/1 anno
    for col in module_annotations.columns:
        orig_name_1 = col.replace("_anno", "")
        orig_name_2 = col.replace("_anno_smooth", "")
        if orig_name_1 in module_annotations.columns:
            module_annotations[col] = (module_annotations[col] == orig_name_1).astype(int)
        else:
            module_annotations[col] = (module_annotations[col] == orig_name_2).astype(int)
        
    # Compute the number of annotated cells for each module
    module_counts = module_annotations.sum(axis=0)

    # Sort modules by the number of annotated cells in ascending order
    sorted_modules = module_counts.sort_values().index

    # Initialize an empty Series to store the final cell annotations
    cell_annotations = pd.Series(index=module_annotations.index, dtype=object)

    # Annotate cells iteratively
    for module in sorted_modules:
        module_name = module.replace("_anno_smooth", "")
        module_name = module_name.replace("_anno", "")
        # Find cells with annotation value 1 for the current module
        cells_to_annotate = module_annotations.index[module_annotations[module] == 1]

        # Only annotate cells that have not been annotated yet
        cells_to_annotate = cells_to_annotate[cell_annotations[cells_to_annotate].isna()]

        # Set these cells to the current module name
        cell_annotations[cells_to_annotate] = module_name

    # Add the final cell annotations to adata.obs
    adata.obs[result_anno] = cell_annotations

    print(f"Cell annotation completed. Results stored in adata.obs['{result_anno}'].")


# def calculate_gmm_annotations(adata, 
#                               ggm_key='ggm',
#                               modules_used = None,
#                               modules_excluded = None,
#                               max_iter=200,
#                               prob_threshold=0.99,
#                               min_samples=10,
#                               n_components=3,
#                               enable_fallback=True,
#                               random_state=42
#                               ):
#     """
#     Gaussian Mixture Model annotation (with threshold check for 3 components).
    
#     Parameters:
#       adata: AnnData object.
#       ggm_key: Key for the GGM object in adata.uns['ggm_keys'].
#       modules_used: List of module IDs.(default None)
#       modules_excluded: List of module IDs to exclude.(default None)
#       max_iter: Maximum iterations.
#       prob_threshold: Probability threshold for high expression.
#       min_samples: Minimum valid sample count.
#       n_components: Initial number of GMM components.
#       enable_fallback: Whether to enable fallback to fewer components.
#       random_state: Random seed.
      
#     Returns:
#       adata: Updated AnnData object with:
#         - obs: Integrated annotation data (columns prefixed with "Anno_").
#         - uns['module_stats']: Raw statistics records.
#     """
#     # Input validation
#     if ggm_key not in adata.uns['ggm_keys']:
#         raise ValueError(f"{ggm_key} not found in adata.uns['ggm_keys']")

#     mod_info_key = adata.uns['ggm_keys'][ggm_key]['module_info']
#     expr_key = adata.uns['ggm_keys'][ggm_key]['module_expression']
#     mod_stats_key = adata.uns['ggm_keys'][ggm_key]['module_stats']

#     if expr_key not in adata.obsm:
#         raise ValueError(f"{expr_key} not found in adata.obsm")
#     if mod_info_key not in adata.uns:
#         raise ValueError(f"{mod_info_key} not found in adata.uns")
    
#     module_expr_matrix = adata.obsm[expr_key]
#     module_expr_matrix = pd.DataFrame(module_expr_matrix, index=adata.obs.index)
#     if len(adata.uns[mod_info_key]['module_id'].unique()) != module_expr_matrix.shape[1]:
#         raise ValueError(f"module_info and module_expression dimensions for the ggm '{ggm_key}' do not match")
#     else:
#         module_expr_matrix.columns = adata.uns[mod_info_key]['module_id'].unique()

#     # Check module list
#     # If modules_used is not provided, use all modules in adata.uns[mod_info_key]
#     if modules_used is None:
#         modules_used = adata.uns[mod_info_key]['module_id'].unique()
#     # Exclude modules if the modules_excluded list is provided
#     if modules_excluded is not None:
#         modules_used = [mid for mid in modules_used if mid not in modules_excluded]

#     valid_modules = [mid for mid in modules_used if mid in module_expr_matrix.columns]
    
#     if not valid_modules:
#         raise ValueError(f"Ensure that the input module IDs exist in adata.uns['{mod_info_key}']")
    
#     existing_columns = [f"{mid}_anno" for mid in modules_used if f"{mid}_anno" in adata.obs]
#     if existing_columns:
#         print(f"Removing existing annotation columns: {existing_columns}")
#         adata.obs.drop(columns=existing_columns, inplace=True)

#     # Initialize annotation matrix
#     anno_cols = [f"{mid}" for mid in modules_used]
#     annotations = pd.DataFrame(
#         np.zeros((adata.obs.shape[0], len(anno_cols)), dtype=int),
#         index=adata.obs.index,
#         columns=anno_cols
#     )
#     stats_records = []
    
#     # Process each module
#     for module_id in valid_modules:
#         stats = {
#             'module_id': module_id,
#             'status': 'success',
#             'n_components': n_components,
#             'final_components': n_components,
#             'threshold': np.nan,
#             'anno_one': 0,
#             'anno_zero': 0,
#             'components': [],
#             'error_info': 'None'
#         }
        
#         try:
#             expr_values = module_expr_matrix[module_id].values
#             non_zero_mask = expr_values != 0
#             non_zero_expr = expr_values[non_zero_mask]
            
#             # Basic checks
#             if len(non_zero_expr) == 0:
#                 raise ValueError("all_zero_expression")
#             if len(non_zero_expr) < min_samples:
#                 raise ValueError(f"insufficient_samples ({len(non_zero_expr)}<{min_samples})")
#             if np.var(non_zero_expr) < 1e-6:
#                 raise ValueError("zero_variance")

#             # Fit GMM
#             gmm = GaussianMixture(
#                 n_components=n_components,
#                 random_state=random_state,
#                 max_iter=max_iter
#             )
#             with warnings.catch_warnings():
#                 warnings.filterwarnings("ignore")
#                 gmm.fit(non_zero_expr.reshape(-1, 1))

#             # Select high-expression component
#             means = gmm.means_.flatten()
#             main_component = np.argmax(means)
#             main_mean = means[main_component]
            
#             # Compute probabilities and generate annotation
#             probs = gmm.predict_proba(non_zero_expr.reshape(-1, 1))[:, main_component]
#             anno_non_zero = (probs >= prob_threshold).astype(int)
            
#             if np.sum(anno_non_zero) == 0:
#                 raise ValueError("no_positive_cells")
            
#             # Compute expression threshold
#             positive_expr = non_zero_expr[anno_non_zero == 1]
#             threshold = np.min(positive_expr)
            
#             # For 3-component model, check threshold validity
#             if n_components >= 3 and threshold > main_mean:
#                 raise ValueError(f"threshold {threshold:.2f} > μ ({main_mean:.2f})")
            
#             # Store module annotation
#             anno_col = f"{module_id}"
#             annotations.loc[non_zero_mask, anno_col] = anno_non_zero
            
#             # Update statistics
#             stats.update({
#                 'threshold': threshold,
#                 'anno_one': int(anno_non_zero.sum()),
#                 'anno_zero': int(len(anno_non_zero) - anno_non_zero.sum()),
#                 'components': [
#                     {   'component': i,
#                         'mean': float(gmm.means_[i][0]),
#                         'var': float(gmm.covariances_[i][0][0]),
#                         'weight': float(gmm.weights_[i])
#                     } 
#                     for i in range(n_components)
#                 ],
#                 'main_component': int(main_component)
#             })

#         except Exception as e:
#             stats.update({
#                 'status': 'failed',
#                 'error_info': str(e),
#                 'components': [],
#                 'threshold': np.nan,
#                 'anno_one': 0,
#                 'anno_zero': expr_values.size
#             })
            
#             # Fallback strategy
#             if enable_fallback and n_components > 2:
#                 try:
#                     # Try 2-component model (without threshold check)
#                     gmm = GaussianMixture(
#                         n_components=2,
#                         random_state=random_state,
#                         max_iter=max_iter
#                     )
#                     gmm.fit(non_zero_expr.reshape(-1, 1))
                    
#                     # Select high-expression component
#                     means = gmm.means_.flatten()
#                     main_component = np.argmax(means)
                    
#                     probs = gmm.predict_proba(non_zero_expr.reshape(-1, 1))[:, main_component]
#                     #anno_non_zero = (probs >= 0.9999).astype(int)
#                     #anno_non_zero = (probs >= prob_threshold).astype(int)
#                     anno_non_zero = (probs >= ((1 - (1 - prob_threshold) * 1e-2))).astype(int)
                    
#                     if anno_non_zero.sum() > 0:
#                         positive_expr = non_zero_expr[anno_non_zero == 1]
#                         threshold = np.min(positive_expr)
                        
#                         stats.update({
#                             'status': 'success',
#                             'final_components': 2,
#                             'threshold': threshold,
#                             'anno_one': int(anno_non_zero.sum()),
#                             'anno_zero': int(len(anno_non_zero) - anno_non_zero.sum()),
#                             'components': [
#                                 {   'component': 0,
#                                     'mean': float(gmm.means_[0][0]),
#                                     'var': float(gmm.covariances_[0][0][0]),
#                                     'weight': float(gmm.weights_[0])
#                                 },
#                                 {   'component': 1,
#                                     'mean': float(gmm.means_[1][0]),
#                                     'var': float(gmm.covariances_[1][0][0]),
#                                     'weight': float(gmm.weights_[1])
#                                 }
#                             ],
#                             'main_component': int(main_component)
#                         })
#                         annotations.loc[non_zero_mask, f"{module_id}"] = anno_non_zero
#                 except Exception as fallback_e:
#                     stats['error_info'] += f"; Fallback failed: {str(fallback_e)}"

#         finally:
#             if stats['status'] == 'success':
#                 print(f"{module_id} processed, {stats['status']}, anno cells : {stats['anno_one']}")
#             else:
#                 print(f"{module_id} processed, {stats['status']}")

#             if stats.get('components'):
#                 stats['components'] = str(stats['components'])
#             else:
#                 stats['components'] = 'None'
#             stats_records.append(stats)

#     # Store annotations in adata.obs        
#     annotations.columns = [f"{col}_anno" for col in annotations.columns]
    
#     # Reset the 0/1 anno to module id or None
#     for col in annotations.columns:
#         orig_name = col.replace("_anno", "")
#         annotations[col] = np.where(annotations[col] == 1, orig_name, None)
#         annotations[col] = pd.Categorical(annotations[col])

#     adata.obs = pd.concat([adata.obs, annotations], axis=1)
    
#     # Store statistics in adata.uns
#     stats_records_df = pd.DataFrame(stats_records)
#     stats_records_df = pd.DataFrame(stats_records)
#     if mod_stats_key in adata.uns:
#         existing_stats = adata.uns[mod_stats_key]
#         # For each module, update existing records with new data
#         for mid in stats_records_df['module_id'].unique():
#             new_row = stats_records_df.loc[stats_records_df['module_id'] == mid].iloc[0]
#             mask = existing_stats['module_id'] == mid
#             if mask.any():
#                 num_rows = mask.sum()
#                 new_update_df = pd.DataFrame([new_row] * num_rows, index=existing_stats.loc[mask].index)
#                 existing_stats.loc[mask] = new_update_df
#             else:
#                 existing_stats = pd.concat([existing_stats, pd.DataFrame([new_row])], ignore_index=True)
#         existing_stats.dropna(how='all', inplace=True)
#         adata.uns[mod_stats_key] = existing_stats
#     else:
#         adata.uns[mod_stats_key] = stats_records_df

