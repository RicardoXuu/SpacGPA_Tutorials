
# %%
# 尝试在计算出模块表达水平之后直接对表达水平进行平滑处理
# 使用 Mouse_Small_Intestine_FFPE 16um数据
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
# 读取数据
ggm = sg.load_ggm("data/Mouse_Small_Intestine_FFPE_r3.ggm.h5")
adata = sc.read_h5ad("data/Mouse_Small_Intestine_FFPE_ggm_anno_r3.h5ad")


















# %%
# 参考
from sklearn.neighbors import NearestNeighbors

def smooth_all_module_expressions(
    adata,
    module_key='module_expression',
    coord_key='spatial',
    n_neighbors=9,
    sigma_scale=0.6,
    cutoff_scale=3.0
):
    """
    Applies spatial smoothing to all module expression columns in adata.obsm[module_key].
    Smoothed results are saved both in adata.obs and adata.obsm['module_expression_smooth'].
    
    Parameters:
    - adata: AnnData object
    - module_key: key in adata.obsm for the matrix of module expressions
    - coord_key: key in adata.obsm for spatial coordinates
    - n_neighbors: number of neighbors to fetch before filtering
    - sigma_scale: sigma multiplier for Gaussian kernel (based on avg min distance)
    - cutoff_scale: max distance multiplier to exclude distant neighbors
    """
    coords = adata.obsm[coord_key]
    module_expr = adata.obsm[module_key]
    feature_names = module_expr.columns if isinstance(module_expr, pd.DataFrame) else [f'M{i+1}_exp' for i in range(module_expr.shape[1])]
    module_expr = pd.DataFrame(module_expr, columns=feature_names)

    # Step 1: compute average minimal distance
    nn_dist_calc = NearestNeighbors(n_neighbors=2).fit(coords)
    dists, _ = nn_dist_calc.kneighbors(coords)
    avg_min_dist = dists[:, 1].mean()

    sigma = sigma_scale * avg_min_dist
    max_dist = cutoff_scale * avg_min_dist

    # Step 2: get neighbors
    nn = NearestNeighbors(n_neighbors=n_neighbors).fit(coords)
    distances, indices = nn.kneighbors(coords)

    # Output container
    smoothed_df = pd.DataFrame(index=adata.obs_names)

    for col in feature_names:
        expr = module_expr[col].values
        smoothed = np.zeros_like(expr)

        for i in range(len(expr)):
            dists_i = distances[i]
            idxs_i = indices[i]
            
            # Filter neighbors within cutoff
            valid_mask = dists_i <= max_dist
            dists_valid = dists_i[valid_mask]
            idxs_valid = idxs_i[valid_mask]
            
            if len(idxs_valid) == 0:
                smoothed[i] = expr[i]
                continue

            expr_valid = expr[idxs_valid]

            # Remove one max-expressing neighbor
            max_idx = np.argmax(expr_valid)
            keep_mask = np.ones(len(expr_valid), dtype=bool)
            keep_mask[max_idx] = False

            expr_kept = expr_valid[keep_mask]
            dists_kept = dists_valid[keep_mask]

            if len(expr_kept) == 0:
                smoothed[i] = expr[i]
                continue

            weights = np.exp(- (dists_kept ** 2) / (2 * sigma ** 2))
            weights /= weights.sum()

            smoothed[i] = np.sum(weights * expr_kept)

        # Save to obs and smoothed matrix
        obs_col_name = f"{col}_smooth"
        adata.obs[obs_col_name] = smoothed
        smoothed_df[obs_col_name] = smoothed

    # Store all smoothed modules in obsm
    adata.obsm['module_expression_smooth'] = smoothed_df
    
def transform_exp_values(values, pct=0.2):
    """
    Transform values in obs['exp']:
    - Values = pct*max are set to 1.
    - Values < pct*max are scaled linearly from 0 to 1 based on their position between 0 and pct*max.
    
    Parameters:
    - obs (dict): Dictionary containing an 'exp' key (list or array-like).
    - pct (float): Percentage threshold (default: 0.1 ? 10% of max).
    
    Returns:
    - dict: Modified `obs` with transformed 'exp' values.
    """
    exp_values = values
    max_val = np.max(exp_values)
    threshold = pct * max_val
    
    # Apply transformation
    transformed = np.where(
        exp_values >= threshold,
        1.0,  # Set values = threshold to 1
        exp_values / threshold  # Scale values < threshold linearly from 0 to 1
    )
    
    # Ensure output matches input type (list or array)
    return transformed
