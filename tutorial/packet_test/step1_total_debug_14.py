

# %%
# 改正calculate_gmm_annotations计算截断值太慢的问题
# 使用MOATA_E16.5_E1S1 数据进行分析
import numpy as np
import pandas as pd
import random
import time
import torch
import scanpy as sc
import anndata
import os
import gc
import matplotlib.pyplot as plt
import seaborn as sns

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from PIL import Image

# %% 切换工作目录
os.getcwd()
workdir = '/dta/ypxu/SpacGPA/Dev_Version/SpacGPA_dev_1'
os.chdir(workdir)
os.getcwd()

# %%
import SpacGPA as sg

# %%
# 读取GGM
start_time = time.time()
ggm = sg.load_ggm("data/MOSTA_E16.5_E1S1.ggm.h5")
print(f"Time: {time.time() - start_time:.5f} s")

# %%
# 读取数据
adata = sc.read_h5ad("/dta/ypxu/ST_GGM/VS_Code/ST_GGM_dev_1/data/MOSTA/E16.5_E1S1.MOSTA.h5ad")
adata.var_names_make_unique()
print(adata.X.shape)

# %%
# 计算模块的加权表达值
start_time = time.time()
sg.calculate_module_expression(adata, 
                               ggm, 
                               ggm_key='ggm',
                               top_genes=30,
                               weighted=True,
                               calculate_moran=False,
                               embedding_key='spatial',
                               k_neighbors=6,
                               add_go_anno=5,
                               )
print(f"Time: {time.time() - start_time:.5f} s")
print(adata.uns['module_info'])


# %%
# 计算GMM注释
start_time = time.time()
sg.calculate_gmm_annotations(adata, 
                            ggm_key='ggm',
                            max_iter=200,
                            prob_threshold=0.99,
                            min_samples=10,
                            n_components=3,
                            enable_fallback=True,
                            embedding_key='spatial',
                            k_neighbors=6,
                            random_state=42)
print(f"Time: {time.time() - start_time:.5f} s")
print(adata.uns['module_stats'])




# %%
# 改正calculate_gmm_annotations计算截断值太慢的问题
import numpy as np
import pandas as pd
import warnings
import scipy.sparse as sp
from scipy.spatial.distance import pdist
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
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


def calculate_gmm_annotations(adata, 
                              ggm_key='ggm',
                              modules_used=None,
                              modules_excluded=None,
                              calculate_moran=True,
                              embedding_key='spatial',
                              k_neighbors=6,
                              max_iter=200,
                              prob_threshold=0.99,
                              min_samples=10,
                              n_components=3,
                              enable_fallback=True,
                              random_state=42
                              ):
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
        - module_moran_I: Global Moran's I computed on the module expression (all cells) (if calculate_moran True).
        - positive_moran_I: Moran's I computed on module expression for cells annotated as 1 (if calculate_moran True).
        - negative_moran_I: Moran's I computed on module expression for cells annotated as 0 (if calculate_moran True).
        - positive_mean_distance: Average pairwise spatial distance among cells annotated as 1 (if calculate_moran True).
        - n_components: Number of components in the GMM.
        - final_components: Number of components after fallback.
        - threshold: Threshold for calling a cell positive.
        - components: List of dictionaries with keys 'component', 'mean', 'var', 'weight'.
        - main_component: Index of the main component.
        - error_info: Error message if status is 'failed'.
        - top_go_terms: Top GO terms associated with the module.

    Parameters:
      adata: AnnData object.
      ggm_key: Key for the GGM object in adata.uns['ggm_keys'].
      modules_used: List of module IDs to process; if None, use all modules in adata.uns[mod_info_key].
      modules_excluded: List of module IDs to exclude.
      calculate_moran: If True, compute Moran's I and other spatial statistics.
                       For Big Spatial Transcriptomics(eg. 10X Visium_HD(2um), etc.), Set to False to avoid memory issues.
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
        if col.endswith('_exp_trim') and any(col.startswith(mid) for mid in valid_modules):
            adata.obs.drop(columns=col, inplace=True)
        if col.endswith('_anno_smooth') and any(col.startswith(mid) for mid in valid_modules):
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
    
    # If calculate_moran is True, construct spatial weights matrix W based on embedding_key.
    if calculate_moran:
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
    else:
        W = None
        coords = None
    
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
            
            # Calculate additional statistics
            if calculate_moran:
                mod_I = compute_moran(expr_values, W)
                stats['module_moran_I'] = mod_I
                
                pos_expr_masked = np.where(module_annotation == 1, expr_values, 0)
                stats['positive_moran_I'] = compute_moran(pos_expr_masked, W)
                
                full_indices = np.where(non_zero_mask)[0]
                pos_idx = full_indices[anno_non_zero == 1]
                if len(pos_idx) > 1:
                    pos_coords = coords[pos_idx, :]
                    stats['positive_mean_distance'] = float(np.mean(pdist(pos_coords)))
                else:
                    stats['positive_mean_distance'] = np.nan
                
                neg_expr_masked = np.where(module_annotation == 0, expr_values, 0)
                stats['negative_moran_I'] = compute_moran(neg_expr_masked, W)
            else:
                stats['module_moran_I'] = np.nan
                stats['positive_moran_I'] = np.nan
                stats['negative_moran_I'] = np.nan
                stats['positive_mean_distance'] = np.nan
            
            stats['skew'] = float(skew(non_zero_expr))
            
            if len(expr_values) > 0:
                top_n = max(1, int(len(expr_values) * 0.01))
                sorted_expr = np.sort(expr_values)
                top1_mean = np.mean(sorted_expr[-top_n:])
                overall_mean = np.mean(expr_values)
                stats['top1pct_ratio'] = top1_mean / overall_mean if overall_mean != 0 else np.nan
            else:
                stats['top1pct_ratio'] = np.nan
            
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
                        fallback_annotation = np.zeros_like(expr_values, dtype=int)
                        fallback_annotation[non_zero_mask] = anno_non_zero
                        annotations.loc[non_zero_mask, module_id] = anno_non_zero
                        if calculate_moran:
                            fallback_mod_I = compute_moran(expr_values, W)
                            stats['module_moran_I'] = fallback_mod_I
                            pos_expr_masked = np.where(fallback_annotation == 1, expr_values, 0)
                            stats['positive_moran_I'] = compute_moran(pos_expr_masked, W)
                            full_indices = np.where(non_zero_mask)[0]
                            pos_idx = full_indices[anno_non_zero == 1]
                            if len(pos_idx) > 1:
                                pos_coords = coords[pos_idx, :]
                                stats['positive_mean_distance'] = float(np.mean(pdist(pos_coords)))
                            else:
                                stats['positive_mean_distance'] = np.nan
                            neg_expr_masked = np.where(fallback_annotation == 0, expr_values, 0)
                            stats['negative_moran_I'] = compute_moran(neg_expr_masked, W)
                        else:
                            stats['module_moran_I'] = np.nan
                            stats['positive_moran_I'] = np.nan
                            stats['negative_moran_I'] = np.nan
                            stats['positive_mean_distance'] = np.nan
                        stats['skew'] = float(skew(non_zero_expr))
                        if len(non_zero_expr) > 0:
                            top_n = max(1, int(len(non_zero_expr) * 0.01))
                            sorted_expr = np.sort(non_zero_expr)
                            top1_mean = np.mean(sorted_expr[-top_n:])
                            overall_mean = np.mean(non_zero_expr)
                            stats['top1pct_ratio'] = top1_mean / overall_mean if overall_mean != 0 else np.nan
                        else:
                            stats['top1pct_ratio'] = np.nan
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
    
    # Add trimmed expression columns which are 0 if the annotation is None for this module
    for mod in valid_modules:
        exp_col = f"{mod}_exp"
        anno_col = f"{mod}_anno"
        trim_col = f"{mod}_exp_trim"
        if exp_col in adata.obs.columns and anno_col in adata.obs.columns:
            mask = (adata.obs[anno_col] == mod).astype(int)
            adata.obs[trim_col] = adata.obs[exp_col] * mask

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
                return ""
        stats_records_df["top_go_terms"] = stats_records_df["module_id"].apply(concat_go_terms)
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
    

# %%
# 测试
start_time = time.time()
calculate_gmm_annotations(adata, 
                            ggm_key='ggm',
                            max_iter=200,
                            prob_threshold=0.99,
                            min_samples=10,
                            n_components=3,
                            enable_fallback=True,
                            embedding_key='spatial',
                            k_neighbors=6,
                            random_state=42)
print(f"Time: {time.time() - start_time:.5f} s")
print(adata.uns['module_stats'])
# %%
