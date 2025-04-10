
# %%
# 开发新的整合函数，使用ggm_mulit_intersection和ggm_mulit_union做整合
# 使用 CytAssist_FreshFrozen_Mouse_Brain_Rep2 数据
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

# %%
# 切换工作目录
os.getcwd()
workdir = '/dta/ypxu/SpacGPA/Dev_Version/SpacGPA_dev_1'
os.chdir(workdir)
os.getcwd()

# %%
#from SpacGPA import *
import SpacGPA as sg

# %%
# 读取 ggm
start_time = time.time()
ggm = sg.load_ggm("data/CytAssist_FreshFrozen_Mouse_Brain_Rep2.ggm.h5")
print(f"Read ggm: {time.time() - start_time:.5f} s")
# 读取联合分析的ggm
ggm_mulit_intersection = sg.load_ggm("data/CytAssist_FreshFrozen_Mouse_Brain_Rep2.ggm_mulit_intersection.h5")
print(f"Read ggm_mulit_intersection: {time.time() - start_time:.5f} s")
ggm_mulit_union = sg.load_ggm("data/CytAssist_FreshFrozen_Mouse_Brain_Rep2.ggm_mulit_union.h5")
print(f"Read ggm_mulit_union: {time.time() - start_time:.5f} s")
print("=====================================")
print(ggm)
print("=====================================")
print(ggm_mulit_intersection)
print("=====================================")
print(ggm_mulit_union)


# %%
# 读取数据
adata = sc.read_visium("/dta/ypxu/ST_GGM/VS_Code/ST_GGM_dev_1/data/visium/CytAssist_FreshFrozen_Mouse_Brain_Rep2",
                       count_file="CytAssist_FreshFrozen_Mouse_Brain_Rep2_filtered_feature_bc_matrix.h5")
adata.var_names_make_unique()
adata.var_names = adata.var['gene_ids']
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
print(adata.X.shape)

# %%
# 添加注释
graph_cluster = pd.read_csv('/dta/ypxu/ST_GGM/Raw_Datasets/visium/CytAssist_FreshFrozen_Mouse_Brain_Rep2/analysis/clustering/gene_expression_graphclust/clusters.csv',
                            header=0, sep=',', index_col=0)
kmeans_10_clusters = pd.read_csv('/dta/ypxu/ST_GGM/Raw_Datasets/visium/CytAssist_FreshFrozen_Mouse_Brain_Rep2/analysis/clustering/gene_expression_kmeans_10_clusters/clusters.csv',
                                header=0, sep=',', index_col=0)

adata.obs['graph_cluster'] = graph_cluster.loc[adata.obs_names, 'Cluster']
adata.obs['graph_cluster'] = adata.obs['graph_cluster'].astype('category')

adata.obs['kmeans_10_clusters'] = kmeans_10_clusters.loc[adata.obs_names, 'Cluster']
adata.obs['kmeans_10_clusters'] = adata.obs['kmeans_10_clusters'].astype('category')


# %%
sg.annotate_with_ggm(adata, ggm_mulit_intersection,
                     ggm_key='ggm')

# %%
#使用leiden聚类和louvain聚类基于模块表达矩阵归一化矩阵进行聚类
start_time = time.time()
sc.pp.neighbors(adata, n_neighbors=18, use_rep='module_expression_scaled',n_pcs=adata.obsm['module_expression_scaled'].shape[1])
sc.tl.leiden(adata, resolution=0.5, key_added='leiden_0.5_ggm')
sc.tl.leiden(adata, resolution=1, key_added='leiden_1_ggm')
sc.tl.louvain(adata, resolution=0.5, key_added='louvan_0.5_ggm')
sc.tl.louvain(adata, resolution=1, key_added='louvan_1_ggm')
print(f"Time: {time.time() - start_time:.5f} s")

# %%
sc.pl.spatial(adata, alpha_img = 0.5, size = 1.6, title= "graph_cluster", frameon = False, color="graph_cluster", show=True)
sc.pl.spatial(adata, alpha_img = 0.5, size = 1.6, title= "kmeans_10_clusters", frameon = False, color="kmeans_10_clusters", show=True)
sc.pl.spatial(adata, alpha_img = 0.5, size = 1.6, title= "leiden_0.5_ggm", frameon = False, color="leiden_0.5_ggm", show=True)
sc.pl.spatial(adata, alpha_img = 0.5, size = 1.6, title= "leiden_1_ggm", frameon = False, color="leiden_1_ggm", show=True)
sc.pl.spatial(adata, alpha_img = 0.5, size = 1.6, title= "louvan_0.5_ggm", frameon = False, color="louvan_0.5_ggm", show=True)
sc.pl.spatial(adata, alpha_img = 0.5, size = 1.6, title= "louvan_1_ggm", frameon = False, color="louvan_1_ggm", show=True)

# %%
# 设计配色
import colorsys
# Set the number of colors required (122)
n_colors = 122

# Fixed saturation and value (brightness) chosen to produce appealing qualitative colors
saturation = 0.65  
value = 0.9        

# Generate the palette: evenly spaced hues over [0, 1)
palette_hex = []
for i in range(n_colors):
    hue = i / n_colors
    r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
    hex_color = '#{:02x}{:02x}{:02x}'.format(int(r * 255), int(g * 255), int(b * 255))
    palette_hex.append(hex_color)

# Build the dictionary mapping modules "M1" to "M122" to their hex colors
color_dict = {f"M{i+1}": palette_hex[i] for i in range(n_colors)}


# %%
sg.integrate_annotations(adata,
                        ggm_key='ggm',
                        result_anno='annotation',
                        use_smooth=True,
                        modules_excluded=['M15','M18'],
                        embedding_key='spatial',
                        k_neighbors=18,
                        neighbor_similarity_ratio=0.9,
                        )
sc.pl.spatial(adata, alpha_img = 0.5, size = 1.6, title= "annotation", frameon = False, color="annotation",  
              palette=color_dict, show=True)


# %%
# 优化integrate_annotations函数，方案1
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

def integrate_annotations_improved(
    adata,
    ggm_key='ggm',
    cross_ggm=False,
    modules_used=None,
    modules_excluded=None,
    modules_preferred=None,
    result_anno='annotation',
    use_smooth=True,
    embedding_key='spatial',
    k_neighbors=24,
    neighbor_threshold=0.8,       # Threshold for neighbor support in voting
    enable_propagation=True,        # Flag to enable label propagation/smoothing
    propagation_iterations=2,       # Number of iterations for label propagation
    propagation_weight=0.5,         # Weight for neighbor influence during propagation (gamma)
    cluster_prior=None              # Optional prior clustering info as a dict {cell_index: cluster_id}
):
    """
    integrate_annotations_improved: An improved method for integrating multi-module annotations.

    This function integrates module annotations in spatial transcriptomics data by combining 
    neighbor support, gene expression ranking, user-defined module preferences, and optional cluster prior.
    It further refines the annotation result via an optional label propagation (smoothing) procedure and 
    a simple module removal step for isolated, low-frequency modules.

    Parameters:
        adata (anndata.AnnData): An AnnData object containing spatial transcriptomics data and preliminary module annotations.
        ggm_key (str): Key in adata.uns['ggm_keys'] that stores module-related information. Default is 'ggm'.
        cross_ggm (bool): Whether to integrate modules from multiple GGMs. If True and multiple keys exist,
                          modules_used must be provided manually. Default is False.
        modules_used (list): List of module IDs to integrate. If None, all modules in adata.uns[mod_stats_key] are used.
        modules_excluded (list): List of module IDs to exclude from integration.
        modules_preferred (list): List of preferred module IDs. If a cell's candidate annotation includes any of
                                  these modules, they are prioritized.
        result_anno (str): Column name for the integrated annotation in adata.obs. Default is 'annotation'.
        use_smooth (bool): Whether to use the smoothed annotation columns (*_anno_smooth) over the raw annotation columns (*_anno). Default is True.
        embedding_key (str): Key in adata.obsm that stores the spatial coordinates used for neighbor computation. Default is 'spatial'.
        k_neighbors (int): Number of nearest neighbors (excluding the cell itself) used in constructing the KNN graph. Default is 24.
        neighbor_threshold (float): Threshold for the fraction of neighbors supporting a module for direct decision (range 0-1). Typical values are between 0.7 and 0.9.
        enable_propagation (bool): Whether to enable the subsequent label propagation/smoothing step. Default is True.
        propagation_iterations (int): Number of label propagation iterations. Default is 2.
        propagation_weight (float): Weight factor (gamma) for combining self-information and neighbor information during propagation.
                                    A value of 0.5 gives equal weight to self and neighbor information. Default is 0.5.
        cluster_prior (dict or None): Optional dictionary providing clustering labels for each cell (e.g., {cell_index: cluster_id}).
                                      If provided, it contributes to the decision by evaluating cluster consistency.
    
    Returns:
        adata (anndata.AnnData): The input AnnData object updated with the integrated annotation in adata.obs[result_anno].
    """
    # Input checks
    if ggm_key not in adata.uns.get('ggm_keys', {}):
        raise ValueError(f"{ggm_key} not found in adata.uns['ggm_keys']")
    mod_stats_key = adata.uns['ggm_keys'][ggm_key]['module_stats']
    if cross_ggm and len(adata.uns['ggm_keys']) > 2:
        if modules_used is None:
            raise ValueError("When cross_ggm is True, modules_used must be provided manually.")
    # Check for embedding coordinates
    if embedding_key not in adata.obsm:
        raise ValueError(f"Embedding coordinates '{embedding_key}' not found in adata.obsm")
    # Remove pre-existing integrated annotation to avoid warnings
    if result_anno in adata.obs:
        adata.obs.drop(columns=[result_anno], inplace=True)
    
    # Determine the list of modules for integration
    if modules_used is None:
        modules_used = list(adata.uns[mod_stats_key]['module_id'].unique())
    if modules_excluded is not None:
        modules_used = [mid for mid in modules_used if mid not in modules_excluded]
    if len(modules_used) == 0:
        raise ValueError("No modules to integrate after applying exclusions.")
    
    # Prepare annotation matrix and expression ranking scores
    # Select the appropriate annotation columns according to use_smooth flag
    existing_columns = []
    for mid in modules_used:
        if use_smooth and f"{mid}_anno_smooth" in adata.obs:
            existing_columns.append(f"{mid}_anno_smooth")
        elif f"{mid}_anno" in adata.obs:
            existing_columns.append(f"{mid}_anno")
    if len(existing_columns) != len(modules_used):
        missing = set(modules_used) - {int(col.split('_')[0]) for col in existing_columns}
        raise ValueError(f"Some modules missing annotation columns: {missing}")
    # Build a dictionary of binary annotation vectors: 1 indicates a positive annotation for the module
    anno_dict = {}
    for mid in modules_used:
        col = f"{mid}_anno_smooth" if use_smooth and f"{mid}_anno_smooth" in adata.obs else f"{mid}_anno"
        anno_dict[mid] = (adata.obs[col].values == mid).astype(int)
    
    # Compute expression ranking scores for each module (lower rank indicates higher expression)
    expr_score = {}
    for mid in modules_used:
        exp_col = f"{mid}_exp"
        if exp_col not in adata.obs.columns:
            raise KeyError(f"Missing expression column '{exp_col}' for module {mid}")
        rank_vals = adata.obs[exp_col].rank(method='dense', ascending=False).astype(int)
        expr_score[mid] = rank_vals.values
    
    # Compute spatial neighbors (KNN graph)
    coords = adata.obsm[embedding_key]
    nbrs_model = NearestNeighbors(n_neighbors=k_neighbors+1, metric='euclidean')
    nbrs_model.fit(coords)
    _, knn_indices = nbrs_model.kneighbors(coords)  # knn_indices[i] contains the cell itself and k neighbors
    n_cells = adata.n_obs
    
    # Initial integration: combine neighbor voting, expression, user preferences, and optional cluster prior
    initial_labels = [None] * n_cells
    preferred_set = set(modules_preferred) if modules_preferred is not None else None
    
    for i in range(n_cells):
        # Get candidate modules that annotate the cell positively
        candidate_modules = [mid for mid in modules_used if anno_dict[mid][i] == 1]
        if len(candidate_modules) == 0:
            initial_labels[i] = None
            continue
        if preferred_set:
            pref_candidates = [mid for mid in candidate_modules if mid in preferred_set]
            if pref_candidates:
                candidate_modules = pref_candidates
        if len(candidate_modules) == 1:
            initial_labels[i] = candidate_modules[0]
            continue
        # For multiple candidates, compute neighbor support
        neighbor_idx = knn_indices[i, 1:]  # exclude self
        neighbor_counts = {mid: 0 for mid in candidate_modules}
        for mid in candidate_modules:
            neighbor_counts[mid] = np.sum(anno_dict[mid][neighbor_idx])
        neighbor_frac = {mid: neighbor_counts[mid] / len(neighbor_idx) for mid in candidate_modules}
        
        # Check if any module exceeds the neighbor threshold
        chosen = None
        for mid in candidate_modules:
            if neighbor_frac[mid] >= neighbor_threshold:
                if chosen is None or neighbor_frac[mid] > neighbor_frac.get(chosen, 0):
                    chosen = mid
        if chosen:
            initial_labels[i] = chosen
            continue
        
        # Otherwise, compute a composite score using neighbor support, expression, and cluster prior (if available)
        best_score = -1
        best_module = None
        for mid in candidate_modules:
            # Normalize expression score to a value between 0 and 1 (higher is better)
            expr_support = (n_cells - expr_score[mid][i] + 1) / n_cells
            neighbor_support = neighbor_frac[mid]
            cluster_support = 0.0
            if cluster_prior is not None:
                cid = cluster_prior.get(i, None)
                if cid is not None:
                    cluster_idx = [idx for idx, cc in cluster_prior.items() if cc == cid]
                    if cluster_idx:
                        cluster_support = np.mean([anno_dict[mid][idx] for idx in cluster_idx])
            score = 0.5 * neighbor_support + 0.4 * expr_support + 0.1 * cluster_support
            if score > best_score:
                best_score = score
                best_module = mid
        initial_labels[i] = best_module
    
    # Label propagation for spatial smoothing (optional)
    if enable_propagation and propagation_iterations > 0:
        # Convert initial_labels to a probability matrix P (shape: n_cells x len(modules_used))
        module_index = {mid: idx for idx, mid in enumerate(modules_used)}
        P = np.zeros((n_cells, len(modules_used)), dtype=float)
        for i, label in enumerate(initial_labels):
            if label is not None:
                idx = module_index[label]
                P[i, idx] = 1.0
        # Propagation iterations
        for t in range(propagation_iterations):
            P_new = np.zeros_like(P)
            for i in range(n_cells):
                neighbor_idx = knn_indices[i, 1:]
                if len(neighbor_idx) == 0:
                    P_new[i] = P[i]
                else:
                    avg_neighbors = np.mean(P[neighbor_idx], axis=0)
                    P_new[i] = (1 - propagation_weight) * P[i] + propagation_weight * avg_neighbors
            # Normalize each cell's probability vector
            for i in range(n_cells):
                s = P_new[i].sum()
                if s > 0:
                    P_new[i] = P_new[i] / s
            P = P_new
        propagated_labels = [None] * n_cells
        for i in range(n_cells):
            if P[i].max() <= 0:
                propagated_labels[i] = None
            else:
                mid_idx = int(np.argmax(P[i]))
                propagated_labels[i] = modules_used[mid_idx]
        final_labels = propagated_labels
    else:
        final_labels = initial_labels
    
    # Module removal and re-optimization (simplified)
    label_to_cells = {}
    for i, label in enumerate(final_labels):
        if label is None:
            continue
        label_to_cells.setdefault(label, []).append(i)
    modules_to_remove = []
    for mid, cells in label_to_cells.items():
        # If there are few cells (<=3) for a module and they are isolated (neighbors lack same module),
        # mark it for removal.
        if len(cells) <= 3:
            isolated = True
            for c in cells:
                neighbor_idx = set(knn_indices[c, 1:])
                if any(((other in neighbor_idx) for other in cells if other != c)):
                    isolated = False
                    break
            if isolated:
                modules_to_remove.append(mid)
    if modules_to_remove:
        for mid in modules_to_remove:
            for idx in label_to_cells.get(mid, []):
                final_labels[idx] = None
        for i, label in enumerate(final_labels):
            if label is None:
                neighbor_idx = knn_indices[i, 1:]
                neighbor_labels = [final_labels[j] for j in neighbor_idx if final_labels[j] is not None]
                if neighbor_labels:
                    most_label = max(set(neighbor_labels), key=neighbor_labels.count)
                    final_labels[i] = most_label
    
    # Write final labels to adata.obs
    sorted_modules = sorted(modules_used, key=lambda x: int(x.lstrip('M')))
    adata.obs[result_anno] = pd.Categorical(final_labels, categories=sorted_modules, ordered=True)



# %%
# 测试
integrate_annotations_improved(
    adata,
    ggm_key='ggm',
    cross_ggm=False,
    modules_used=None,
    modules_preferred=None,
    result_anno='annotation_new_1',
    use_smooth=True,
    embedding_key='spatial',
    k_neighbors=24,
    neighbor_threshold=0.9,       
    enable_propagation=True,      
    propagation_iterations=1,    
    propagation_weight=0.4,      
)

# %%
sc.pl.spatial(adata, alpha_img = 0.5, size = 1.6, title= "annotation_new_1", frameon = False, color="annotation_new_1",  
              palette=color_dict, show=True)


# %%
# 优化integrate_annotations函数，方案2
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import networkx as nx
import igraph as ig
import leidenalg

def integrate_annotations_improved(adata,
                                   ggm_key='ggm',
                                   cross_ggm=False,
                                   modules_used=None,
                                   modules_excluded=None,
                                   modules_preferred=None,
                                   result_anno='annotation_new_2',
                                   use_smooth=True,
                                   embedding_key='spatial',
                                   k_neighbors=24,
                                   neighbor_similarity_ratio=0.90,
                                   alpha=0.5,
                                   sigma=None,
                                   max_iter=10,
                                   energy_tol=1e-3):
    """
    Improved function for integrating cell annotations from multiple gene co-expression modules.
    It uses spatial neighborhood, expression ranking, and local clustering information to iteratively optimize the annotations.
    
    Parameters:
      adata (anndata.AnnData): AnnData object containing spatial transcriptomics data. Must include:
          - adata.obsm[embedding_key]: Spatial coordinates (2D or 3D).
          - adata.obs: Contains initial annotations for each module (columns named 'MODULE_anno' or 'MODULE_anno_smooth').
          - adata.obs: Contains expression score columns for each module (columns named 'MODULE_exp').
      ggm_key (str): Key in adata.uns['ggm_keys'] where GGM related information is stored.
      cross_ggm (bool): Whether to integrate across multiple GGMs. If True, 'modules_used' must be provided.
      modules_used (list): List of modules to integrate. If None, all modules in adata.uns[mod_stats_key]['module_id'] are used.
      modules_excluded (list): List of modules to exclude.
      modules_preferred (list): List of preferred modules. If a cell’s candidate annotations include a preferred module, that module is chosen.
      result_anno (str): Column name for the final integrated annotation in adata.obs (default: 'annotation_new_2').
      use_smooth (bool): Whether to use smoothed annotations (default True).
      embedding_key (str): Key for spatial coordinates in adata.obsm (default: 'spatial').
      k_neighbors (int): Number of nearest neighbors for KNN (default: 24, including the cell itself).
      neighbor_similarity_ratio (float): Consistency threshold for neighbor voting (range [0,1], default 0.90).
      alpha (float): Weight between neighbor vote and expression ranking in the integrated score (range [0,1], default 0.5).
      sigma (float): Bandwidth parameter for spatial weight decay; if None, set to mean distance of neighbors (excluding self), minimum set to 1e-6.
      max_iter (int): Maximum number of iterations for updating (default: 10).
      energy_tol (float): Tolerance for relative change in global energy for convergence (default: 1e-3).
      
    """
    # STEP 1: Basic checks and module extraction
    if ggm_key not in adata.uns['ggm_keys']:
        raise ValueError(f"{ggm_key} not found in adata.uns['ggm_keys']")
    mod_stats_key = adata.uns['ggm_keys'][ggm_key]['module_stats']
    
    if cross_ggm and len(adata.uns['ggm_keys']) > 2:
        if modules_used is None:
            raise ValueError("When cross_ggm is True, 'modules_used' must be provided manually.")
    
    # Check for spatial coordinates
    if embedding_key not in adata.obsm:
        raise ValueError(f"{embedding_key} not found in adata.obsm. Check the spatial coordinate data.")
    embedding_coords = adata.obsm[embedding_key]
    
    # Remove existing result column if it exists
    if adata.obs.get(result_anno) is not None:
        print(f"Note: The column '{result_anno}' in adata.obs will be overwritten.")
        adata.obs.drop(columns=result_anno, inplace=True)
    
    # Determine modules_used and exclude modules if specified
    if modules_used is None:
        modules_used = list(pd.unique(adata.uns[mod_stats_key]['module_id']))
    if modules_excluded is not None:
        modules_used = [m for m in modules_used if m not in modules_excluded]
    
    # STEP 2: Extract annotation columns (prefer smoothed columns)
    anno_columns = {}
    for m in modules_used:
        col_smooth = f"{m}_anno_smooth"
        col_ori = f"{m}_anno"
        if use_smooth and col_smooth in adata.obs.columns:
            anno_columns[m] = col_smooth
        elif col_ori in adata.obs.columns:
            anno_columns[m] = col_ori
        else:
            raise ValueError(f"Annotation column for module {m} not found in adata.obs. Check the data.")
    
    # STEP 3: Precompute expression ranking (lower rank means higher expression)
    expr_score = {}
    for m in modules_used:
        exp_col = f"{m}_exp"
        if exp_col not in adata.obs.columns:
            raise KeyError(f"Expression score column '{exp_col}' not found in adata.obs.")
        expr_score[m] = adata.obs[exp_col].rank(method='dense', ascending=False).astype(int).values
    
    n_obs = adata.n_obs
    
    # STEP 4: Construct initial annotation dictionary (binary matrix)
    anno_dict = {}
    for m in modules_used:
        # Mark as 1 if annotation equals module name, else 0.
        anno_dict[m] = (adata.obs[anno_columns[m]] == m).astype(int).values
    
    # STEP 5: Build KNN graph and compute weight matrix
    k = k_neighbors + 1  # Including the cell itself
    nbrs = NearestNeighbors(n_neighbors=k, metric='euclidean').fit(embedding_coords)
    distances, indices = nbrs.kneighbors(embedding_coords)
    # If sigma is not provided, set it to the mean distance of neighbors (excluding self); ensure sigma is not zero.
    if sigma is None:
        sigma = np.mean(distances[:, 1:])
        if sigma == 0:
            sigma = 1e-6
    # Compute weight matrix W: for cell i, weight w(i,j) = exp(-d(i,j)^2 / sigma^2)
    W = np.exp(- (distances**2) / (sigma**2))
    # Note: W[i,0] is the self-weight.
    
    # STEP 6: Helper function to compute weighted neighbor voting score for a module
    def compute_neighbor_vote(cell_idx, module):
        nb_idx = indices[cell_idx, 1:]  # Exclude self
        nb_weights = W[cell_idx, 1:]
        nb_annos = anno_dict[module][nb_idx]
        sum_weights = np.sum(nb_weights)
        if sum_weights == 0:
            vote = 0.0
        else:
            vote = np.sum(nb_weights * nb_annos) / sum_weights
        return vote
    
    # STEP 7: Initial integration of annotations based on neighbor voting and expression ranking
    final_annotation = [None] * n_obs
    energy_values = np.zeros(n_obs)  # Store integrated energy for each cell
    
    # Function to compute integrated score S(i, m)
    def compute_score(cell_idx, module):
        vote = compute_neighbor_vote(cell_idx, module)
        # Normalize expression rank to [0,1] (lower normalized rank is better)
        r = expr_score[module][cell_idx]
        r_norm = (r - 1) / (n_obs - 1) if n_obs > 1 else 0
        score = alpha * (1 - vote) + (1 - alpha) * r_norm
        return score
    
    # Compute initial annotation for each cell
    for i in range(n_obs):
        candidates = [m for m in modules_used if anno_dict[m][i] == 1]
        if modules_preferred is not None:
            candidates = [m for m in candidates if m in modules_preferred]
        if len(candidates) == 0:
            # Fallback: use integrated score across all modules if no candidates are found
            scores_all = {m: compute_score(i, m) for m in modules_used}
            best_m = min(scores_all, key=scores_all.get)
            final_annotation[i] = best_m
            energy_values[i] = scores_all[best_m]
        elif len(candidates) == 1:
            final_annotation[i] = candidates[0]
            energy_values[i] = compute_score(i, candidates[0])
        else:
            scores = {m: compute_score(i, m) for m in candidates}
            best_m = min(scores, key=scores.get)
            final_annotation[i] = best_m
            energy_values[i] = scores[best_m]
    
    # STEP 8: Build spatial graph (using KNN result) for local clustering and global consistency checks
    G = nx.Graph()
    G.add_nodes_from(range(n_obs))
    for i in range(n_obs):
        for j in indices[i, 1:]:
            G.add_edge(i, j)
    
    # STEP 9: Global energy function (sum of cell energies; replace inf with a large number)
    def global_energy(energy_vec):
        finite_energy = np.where(np.isfinite(energy_vec), energy_vec, 1e6)
        return np.nansum(finite_energy)
    
    E_old = global_energy(energy_values)
    print(f"Initial global energy: {E_old:.6f}")
    
    # STEP 10: Iteratively update annotations using local clustering and remove unstable modules
    for iteration in range(max_iter):
        print(f"\nIteration {iteration+1}:")
        # Update node attributes in the networkx graph
        g_nx = nx.convert_node_labels_to_integers(G)
        for i in range(n_obs):
            g_nx.nodes[i]["label"] = final_annotation[i] if final_annotation[i] is not None else "None"
        # Convert networkx graph to igraph graph
        g = ig.Graph.from_networkx(g_nx)
        # Perform Leiden clustering
        partition = leidenalg.find_partition(g, leidenalg.RBConfigurationVertexPartition, weights=None)
        # Compute purity for each cluster
        cluster_purity = {}
        for cid, cluster in enumerate(partition):
            label_count = {}
            for idx in cluster:
                lab = g.vs[idx]["label"]
                label_count[lab] = label_count.get(lab, 0) + 1
            purity = max(label_count.values()) / len(cluster)
            cluster_purity[cid] = purity
        avg_purity = np.mean(list(cluster_purity.values()))
        print(f"Average cluster purity: {avg_purity:.4f}")
        
        # Determine unstable modules based on cluster purity
        module_count = {m: 0 for m in modules_used}
        module_low_purity = {m: 0 for m in modules_used}
        for cid, cluster in enumerate(partition):
            count = {}
            for idx in cluster:
                m_lab = final_annotation[idx]
                if m_lab is not None:
                    count[m_lab] = count.get(m_lab, 0) + 1
            for m in count:
                module_count[m] += count[m]
            for m in modules_used:
                if m in count:
                    ratio = count[m] / len(cluster)
                    if ratio < 0.7:
                        module_low_purity[m] += 1
        modules_to_exclude = [m for m in modules_used if (module_low_purity[m] / len(partition)) > 0.5]
        if modules_to_exclude:
            print(f"Excluding unstable modules: {modules_to_exclude}")
        else:
            print("No unstable modules; retaining all candidates.")
        
        # Update candidates and recompute scores for each cell
        energy_values_new = np.zeros(n_obs)
        updated_annotation = [None] * n_obs
        for i in range(n_obs):
            candidates = [m for m in modules_used if anno_dict[m][i] == 1]
            candidates = [m for m in candidates if m not in modules_to_exclude]
            if modules_preferred is not None:
                candidates = [m for m in candidates if m in modules_preferred]
            if len(candidates) == 0:
                scores_all = {m: compute_score(i, m) for m in modules_used}
                best_m = min(scores_all, key=scores_all.get)
                updated_annotation[i] = best_m
                energy_values_new[i] = scores_all[best_m]
            elif len(candidates) == 1:
                updated_annotation[i] = candidates[0]
                energy_values_new[i] = compute_score(i, candidates[0])
            else:
                scores = {m: compute_score(i, m) for m in candidates}
                best_m = min(scores, key=scores.get)
                updated_annotation[i] = best_m
                energy_values_new[i] = scores[best_m]
        E_new = global_energy(energy_values_new)
        print(f"Global energy this iteration: {E_new:.6f}")
        
        # Check for convergence using relative energy change
        if abs(E_old - E_new) / max(abs(E_old), abs(E_new)) < energy_tol:
            print("Global energy converged. Stopping iterations.")
            final_annotation = updated_annotation
            energy_values = energy_values_new
            break
        else:
            final_annotation = updated_annotation
            energy_values = energy_values_new
            E_old = E_new

    # STEP 11: Write final integrated annotations back to adata.obs
    adata.obs[result_anno] = pd.Series(final_annotation, index=adata.obs.index)
    print(f"\nFinal integrated annotations stored in adata.obs['{result_anno}'] with final global energy: {E_old:.6f}")


# %%
# 测试
integrate_annotations_improved(adata,
                            ggm_key='ggm',
                            cross_ggm=False,
                            modules_used=None,
                            modules_excluded=['M15','M18'],
                            modules_preferred=['M5','M28','M38','M39'],
                            result_anno='annotation_new_2',
                            use_smooth=True,
                            embedding_key='spatial',
                            k_neighbors=24,
                            neighbor_similarity_ratio=0.90,
                            alpha=0.5,
                            sigma=None,
                            max_iter=100,
                            energy_tol=0.01)

# %%
sc.pl.spatial(adata, alpha_img = 0.5, size = 1.6, title= "annotation_new_2", frameon = False, color="annotation_new_2",  
              palette=color_dict, show=True)


# %%
# 计算各组结果的ARI和NMI
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

# Define the columns of interest
clust_columns = ['graph_cluster', 'kmeans_10_clusters', 'leiden_0.5_ggm', 'leiden_1_ggm', 'louvan_0.5_ggm', 'louvan_1_ggm', 'annotation', 'annotation_new_1', 'annotation_new_2']
# Extract clustering labels from adata.obs
df = adata.obs[clust_columns]
# Initialize empty DataFrames for the ARI and NMI matrices
ari_matrix = pd.DataFrame(np.zeros((len(clust_columns), len(clust_columns))), index=clust_columns, columns=clust_columns)
nmi_matrix = pd.DataFrame(np.zeros((len(clust_columns), len(clust_columns))), index=clust_columns, columns=clust_columns)
# Loop over each pair of clustering columns and compute metrics.
for col1 in clust_columns:
    for col2 in clust_columns:
        # Convert to strings to ensure proper handling (especially if the columns are categorical)
        labels1 = df[col1].astype(str).values
        labels2 = df[col2].astype(str).values
        
        # Calculate ARI and NMI
        ari = adjusted_rand_score(labels1, labels2)
        nmi = normalized_mutual_info_score(labels1, labels2)
        
        ari_matrix.loc[col1, col2] = ari
        nmi_matrix.loc[col1, col2] = nmi
# Display the ARI matrix
print("Adjusted Rand Index (ARI) Matrix:")
print(ari_matrix)
# Display the NMI matrix
print("\nNormalized Mutual Information (NMI) Matrix:")
print(nmi_matrix)

# %%
