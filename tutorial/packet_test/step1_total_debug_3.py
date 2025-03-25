
# %%
# 一些问题修复
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

# # %%
# # 读取数据
# adata = sc.read_visium("/dta/ypxu/ST_GGM/VS_Code/ST_GGM_dev_1/data/visium/CytAssist_FreshFrozen_Mouse_Brain_Rep2",
#                        count_file="CytAssist_FreshFrozen_Mouse_Brain_Rep2_filtered_feature_bc_matrix.h5")
# adata.var_names_make_unique()
# adata.var_names = adata.var['gene_ids']
# sc.pp.normalize_total(adata, target_sum=1e4)
# sc.pp.log1p(adata)
# print(adata.X.shape)

# # %%
# # 读取原始数据集提供的两种注释
# graph_cluster = pd.read_csv('/dta/ypxu/ST_GGM/Raw_Datasets/visium/CytAssist_FreshFrozen_Mouse_Brain_Rep2/analysis/clustering/gene_expression_graphclust/clusters.csv',
#                             header=0, sep=',', index_col=0)
# kmeans_10_clusters = pd.read_csv('/dta/ypxu/ST_GGM/Raw_Datasets/visium/CytAssist_FreshFrozen_Mouse_Brain_Rep2/analysis/clustering/gene_expression_kmeans_10_clusters/clusters.csv',
#                                 header=0, sep=',', index_col=0)

# adata.obs['graph_cluster'] = graph_cluster.loc[adata.obs_names, 'Cluster']
# adata.obs['graph_cluster'] = adata.obs['graph_cluster'].astype('category')

# adata.obs['kmeans_10_clusters'] = kmeans_10_clusters.loc[adata.obs_names, 'Cluster']
# adata.obs['kmeans_10_clusters'] = adata.obs['kmeans_10_clusters'].astype('category')


# # %%
# # 读取 ggm
# start_time = time.time()
# ggm = sg.load_ggm("data/ggm_gpu_32.h5")
# print(f"Read ggm: {time.time() - start_time:.5f} s")
# # 读取联合分析的ggm
# ggm_mulit_intersection = sg.load_ggm("data/ggm_mulit_intersection.h5")
# print(f"Read ggm_mulit_intersection: {time.time() - start_time:.5f} s")
# ggm_mulit_union = sg.load_ggm("data/ggm_mulit_union.h5")
# print(f"Read ggm_mulit_union: {time.time() - start_time:.5f} s")
# print("=====================================")
# print(ggm)
# print("=====================================")
# print(ggm_mulit_intersection)
# print("=====================================")
# print(ggm_mulit_union)

# # %%
# adata

# # %%
# # 计算模块的加权表达值
# start_time = time.time()
# sg.calculate_module_expression(adata, 
#                                ggm_obj=ggm, 
#                                top_genes=30,
#                                weighted=True,
#                                calculate_moran=True,
#                                embedding_key='spatial',
#                                k_neighbors=6)  
# print(f"Time1: {time.time() - start_time:.5f} s")

# sg.calculate_module_expression(adata, 
#                                ggm_obj=ggm_mulit_intersection, 
#                                ggm_key='intersection',
#                                top_genes=30,
#                                weighted=True,
#                                calculate_moran=True,
#                                embedding_key='spatial',
#                                k_neighbors=6)  
# print(f"Time2: {time.time() - start_time:.5f} s")

# sg.calculate_module_expression(adata, 
#                                ggm_obj=ggm_mulit_union, 
#                                ggm_key='union',
#                                top_genes=30,
#                                weighted=True,
#                                calculate_moran=True,
#                                embedding_key='spatial',
#                                k_neighbors=6)  
# print(f"Time3 {time.time() - start_time:.5f} s")


# # %%
# sc.pp.neighbors(adata, n_neighbors=18, use_rep='module_expression_scaled',n_pcs=40)
# sc.tl.leiden(adata, resolution=0.5, key_added='leiden_0.5')
# sc.tl.leiden(adata, resolution=1, key_added='leiden_1')
# sc.tl.louvain(adata, resolution=0.5, key_added='louvan_0.5')
# sc.tl.louvain(adata, resolution=1, key_added='louvan_1')

# # %%
# sc.pp.neighbors(adata, n_neighbors=18, use_rep='intersection_module_expression_scaled',
#                 n_pcs=adata.obsm['intersection_module_expression_scaled'].shape[1])
# sc.tl.leiden(adata, resolution=0.5, key_added='leiden_0.5_intersection')
# sc.tl.leiden(adata, resolution=1, key_added='leiden_1_intersection')
# sc.tl.louvain(adata, resolution=0.5, key_added='louvan_0.5_intersection')
# sc.tl.louvain(adata, resolution=1, key_added='louvan_1_intersection')

# # %%
# sc.pp.neighbors(adata, n_neighbors=18, use_rep='union_module_expression_scaled',
#                 n_pcs=adata.obsm['union_module_expression_scaled'].shape[1])
# sc.tl.leiden(adata, resolution=0.5, key_added='leiden_0.5_union')
# sc.tl.leiden(adata, resolution=1, key_added='leiden_1_union')
# sc.tl.louvain(adata, resolution=0.5, key_added='louvan_0.5_union')
# sc.tl.louvain(adata, resolution=1, key_added='louvan_1_union')


# # %%
# adata.write("data/CytAssist_FreshFrozen_Mouse_Brain_Rep2_ggm_anno_union_intersection.h5ad")
# del adata, ggm, ggm_mulit_intersection, ggm_mulit_union
# gc.collect()

# %%
adata = sc.read("data/CytAssist_FreshFrozen_Mouse_Brain_Rep2_ggm_anno_union_intersection.h5ad")
adata

# # %%
# sc.pl.spatial(adata, color='graph_cluster', size=1.6, alpha_img=0.5, frameon=False, show=True,
#               save="/CytAssist_FreshFrozen_Mouse_Brain_Rep2_graph_cluster.pdf")
# # %%
# sc.pl.spatial(adata, color='kmeans_10_clusters', size=1.6, alpha_img=0.5, frameon=False, show=True,
#               save="/CytAssist_FreshFrozen_Mouse_Brain_Rep2_kmeans_10_clusters.pdf")

# # %%
# sc.pl.spatial(adata, color='leiden_0.5', size=1.6, alpha_img=0.5, frameon=False, show=True,
#               save="/CytAssist_FreshFrozen_Mouse_Brain_Rep2_ggm_leiden_0.5.pdf")
# # %%
# sc.pl.spatial(adata, color='leiden_1', size=1.6, alpha_img=0.5, frameon=False, show=True,
#               save="/CytAssist_FreshFrozen_Mouse_Brain_Rep2_ggm_leiden_1.pdf")
# # %%
# sc.pl.spatial(adata, color='louvan_0.5', size=1.6, alpha_img=0.5, frameon=False, show=True,
#               save="/CytAssist_FreshFrozen_Mouse_Brain_Rep2_ggm_louvan_0.5.pdf")
# # %%
# sc.pl.spatial(adata, color='louvan_1', size=1.6, alpha_img=0.5, frameon=False, show=True,
#               save="/CytAssist_FreshFrozen_Mouse_Brain_Rep2_ggm_louvan_1.pdf")


# # %%
# sc.pl.spatial(adata, color='leiden_0.5_intersection', size=1.6, alpha_img=0.5, frameon=False, show=True,
#               save="/CytAssist_FreshFrozen_Mouse_Brain_Rep2_ggm_intersection_leiden_0.5.pdf")
# # %%
# sc.pl.spatial(adata, color='leiden_1_intersection', size=1.6, alpha_img=0.5, frameon=False, show=True,
#               save="/CytAssist_FreshFrozen_Mouse_Brain_Rep2_ggm_intersection_leiden_1.pdf")
# # %%
# sc.pl.spatial(adata, color='louvan_0.5_intersection', size=1.6, alpha_img=0.5, frameon=False, show=True,
#               save="/CytAssist_FreshFrozen_Mouse_Brain_Rep2_ggm_intersection_louvan_0.5.pdf")
# # %%
# sc.pl.spatial(adata, color='louvan_1_intersection', size=1.6, alpha_img=0.5, frameon=False, show=True,
#               save="/CytAssist_FreshFrozen_Mouse_Brain_Rep2_ggm_intersection_louvan_1.pdf")


# # %%
# sc.pl.spatial(adata, color='leiden_0.5_union', size=1.6, alpha_img=0.5, frameon=False, show=True,
#               save="/CytAssist_FreshFrozen_Mouse_Brain_Rep2_ggm_union_leiden_0.5.pdf")
# # %%
# sc.pl.spatial(adata, color='leiden_1_union', size=1.6, alpha_img=0.5, frameon=False, show=True,
#               save="/CytAssist_FreshFrozen_Mouse_Brain_Rep2_ggm_union_leiden_1.pdf")
# # %%
# sc.pl.spatial(adata, color='louvan_0.5_union', size=1.6, alpha_img=0.5, frameon=False, show=True,
#               save="/CytAssist_FreshFrozen_Mouse_Brain_Rep2_ggm_union_louvan_0.5.pdf")
# # %%
# sc.pl.spatial(adata, color='louvan_1_union', size=1.6, alpha_img=0.5, frameon=False, show=True,
#               save="/CytAssist_FreshFrozen_Mouse_Brain_Rep2_ggm_union_louvan_1.pdf")



# %%
# 计算GMM注释
start_time = time.time()
sg.calculate_gmm_annotations(adata, 
                            ggm_key='ggm',
                            #modules_used=None,
                            #modules_used=['M01', 'M02', 'M03', 'M04', 'M05', 'M06', 'M07', 'M08', 'M09', 'M10'],
                            #modules_used=['M11', 'M12', 'M13', 'M14', 'M15', 'M16', 'M17', 'M18', 'M19', 'M20'],
                            #modules_used={'M01', 'M02', 'M03', 'M04', 'M05', 'M06', 'M07', 'M08', 'M09', 'M10'},
                            #modules_excluded=['M01', 'M02', 'M03', 'M04', 'M05', 'M06', 'M07', 'M08', 'M09', 'M10'],
                            max_iter=200,
                            prob_threshold=0.99,
                            min_samples=10,
                            n_components=3,
                            enable_fallback=True,
                            random_state=42)
print(f"Time: {time.time() - start_time:.5f} s")
print(adata.uns['module_stats'])

# %%
sc.pl.spatial(adata, alpha_img = 0.5, size = 1.6, title= "", frameon = False, color="M01_anno", show=True)

# %%
# 平滑注释
start_time = time.time()
sg.smooth_annotations(adata, 
                        ggm_key='ggm',
                        #modules_used=None,
                        #modules_used=['M01', 'M02', 'M03', 'M04', 'M05', 'M06', 'M07', 'M08', 'M09', 'M10'],
                        #modules_used=['M11', 'M12', 'M13', 'M14', 'M15', 'M16', 'M17', 'M18', 'M19', 'M20'],
                        #modules_used={'M01', 'M02', 'M03', 'M04', 'M05', 'M06', 'M07', 'M08', 'M09', 'M10'},
                        #modules_excluded=['M01', 'M02', 'M03', 'M04', 'M05', 'M06', 'M07', 'M08', 'M09', 'M10'],
                        embedding_key='spatial',
                        k_neighbors=18,
                        min_annotated_neighbors=2
                        )
print(f"Time: {time.time() - start_time:.5f} s")    

# %%
sc.pl.spatial(adata, alpha_img = 0.5, size = 1.6, title= "", frameon = False, color="M01_anno_smooth", show=True)

# %%
for module in adata.uns['module_stats']['module_id'].unique():
    print(module)
    print(adata.uns['module_info'][adata.uns['module_info']['module_id']==module]['module_moran_I'].unique())
    sc.pl.spatial(adata, color=[f"{module}_exp",f"{module}_anno",f"{module}_anno_smooth"], 
                  color_map="Reds", alpha_img = 0.5, size = 1.6, frameon = False, show=True)


# %%
# 合并注释（考虑空间坐标和模块表达值）
start_time = time.time()
sg.integrate_annotations(adata,
                        ggm_key='ggm',
                        #modules_used=None,
                        modules_used=adata.uns['module_info'][adata.uns['module_info']['module_moran_I'] > 0.6]['module_id'].unique(),
                        modules_preferred=adata.uns['module_info'][adata.uns['module_info']['module_moran_I'] > 0.8]['module_id'].unique(),
                        #modules_used = adata.uns['module_info']['module_id'].unique()[0:20], 
                        #modules_used=['M01', 'M02', 'M03', 'M04', 'M05', 'M06', 'M07', 'M08', 'M09', 'M10'],
                        #modules_used=['M11', 'M12', 'M13', 'M14', 'M15', 'M16', 'M17', 'M18', 'M19', 'M20'],
                        #modules_used={'M01', 'M02', 'M03', 'M04', 'M05', 'M06', 'M07', 'M08', 'M09', 'M10'},
                        #modules_excluded=['M01', 'M02', 'M03', 'M04', 'M05', 'M06', 'M07', 'M08', 'M09', 'M10'],
                        result_anno='annotation',
                        embedding_key='spatial',
                        k_neighbors=18,
                        use_smooth=True,
                        neighbor_similarity_ratio=0
                        )
print(f"Time: {time.time() - start_time:.5f} s")


# %%
sc.pl.spatial(adata, alpha_img = 0.5, size = 1.6, title= "", frameon = False, color="annotation", 
              na_color="black", show=True)

# %%
# 合并注释（仅考虑模块注释的细胞数目）
start_time = time.time()
sg.integrate_annotations_old(adata,
                            ggm_key='ggm',
                            #modules_used=None,
                            #modules_used=adata.uns['module_stats']['module_id'].unique()[0:30], 
                            #modules_used=['M01', 'M02', 'M03', 'M04', 'M05', 'M06', 'M07', 'M08', 'M09', 'M10'],
                            #modules_used=['M11', 'M12', 'M13', 'M14', 'M15', 'M16', 'M17', 'M18', 'M19', 'M20'],
                            #modules_excluded=['M01', 'M02', 'M03', 'M04', 'M05', 'M06', 'M07', 'M08', 'M09', 'M10'],
                            result_anno = "annotation_old",
                         use_smooth=True
                         )
print(f"Time: {time.time() - start_time:.5f} s")

# %%
sc.pl.spatial(adata, alpha_img = 0.5, size = 1.6, title= "", frameon = False, color="annotation_old", show=True)


# %%
# 计算模块重叠
start_time = time.time()
overlap_records = sg.calculate_module_overlap(adata, 
                                              #modules_used = adata.uns['module_stats']['module_id'].unique()
                                              modules_excluded=['M01', 'M02', 'M03', 'M04', 'M05', 'M06', 'M07', 'M08', 'M09', 'M10']
                                              )
print(f"Time: {time.time() - start_time:.5f} s")
print(overlap_records[overlap_records['module_a'] == 'M11'])

# %%

# %%
# 问题5，关于高斯混合分布，设计activity模块的排除标准。尽量不使用先验知识，
# 待定

# %%
# 问题6，关于高斯混合分布，阈值和主成分数目的关系优化。
# 待定

# %%
# 问题7，关于高斯混合分布，除了使用高斯混合分布，也考虑表达值的排序。
#       对于一个模块，只有那些表达水平大于模块最大表达水平（或者为了防止一些离散的点，可以考虑前20个或者30个细胞的平均值作为模块最大表达水平）的一定比例的细胞才被认为是注释为该模块的
# 待定

# %%
# 问题10，关于合并注释，尝试引入模块的整体莫兰指数，来评估模块的空间分布。如果一个模块的莫兰指数很高，则优先考虑该模块的细胞的可信度。
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import warnings
from scipy.spatial.distance import pdist
from scipy.stats import skew
from sklearn.neighbors import NearestNeighbors

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
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import warnings
from scipy.spatial.distance import pdist
from scipy.stats import skew
def calculate_gmm_annotations(adata, 
                              ggm_key='ggm',
                              modules_used=None,
                              modules_excluded=None,
                              embedding_key='spatial',
                              k_neighbors=24,
                              max_iter=200,
                              prob_threshold=0.99,
                              min_samples=10,
                              n_components=3,
                              enable_fallback=True,
                              random_state=42):
    """
    Gaussian Mixture Model annotation with additional module-level statistics.
    
    Additional statistics added to mod_stats_key include:
      - module_moran: Global Moran's I computed on the module expression (all cells).
      - positive_mean_distance: Average pairwise spatial distance among cells annotated as 1.
      - positive_moran: Moran's I computed on module expression for cells annotated as 1.
      - negative_moran: Moran's I computed on module expression for cells annotated as 0.
      - skew: Skewness of the non-zero expression distribution for the module.
      - top1pct_ratio: Ratio of the average expression among the top 1% high-expressing nonzero cells
                       to the overall nonzero mean.
    
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
    import numpy as np
    import pandas as pd
    import scipy.sparse as sp
    from sklearn.mixture import GaussianMixture
    from sklearn.neighbors import NearestNeighbors
    from scipy.spatial.distance import pdist
    from scipy.stats import skew
    import warnings

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
    if modules_used is None:
        modules_used = list(unique_mods)
    if modules_excluded is not None:
        modules_used = [mid for mid in modules_used if mid not in modules_excluded]
    valid_modules = [mid for mid in modules_used if mid in module_expr_matrix.columns]
    if not valid_modules:
        raise ValueError(f"Ensure that the input module IDs exist in adata.uns['{mod_info_key}']")
    
    # Remove existing annotation columns
    for col in adata.obs.columns:
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
            'module_moran': np.nan,
            'positive_mean_distance': np.nan,
            'positive_moran': np.nan,
            'negative_moran': np.nan,
            'skew': np.nan,
            'top1pct_ratio': np.nan
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
            stats['module_moran'] = mod_I
            
            # (B) Positive annotation: Construct masked expression vector (0 where 0, expr where 1)
            pos_expr_masked = np.where(module_annotation == 1, expr_values, 0)
            stats['positive_moran'] = compute_moran(pos_expr_masked, W)
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
            stats['negative_moran'] = compute_moran(neg_expr_masked, W)
            
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
            
        except Exception as e:
            stats.update({
                'status': 'failed',
                'error_info': str(e),
                'components': [],
                'threshold': np.nan,
                'anno_one': 0,
                'anno_zero': expr_values.size if 'expr_values' in locals() else 0,
                'module_moran': np.nan,
                'positive_mean_distance': np.nan,
                'positive_moran': np.nan,
                'negative_moran': np.nan,
                'skew': np.nan,
                'top1pct_ratio': np.nan
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
                        # 更新 fallback 后的 annotation
                        fallback_annotation = np.zeros_like(expr_values, dtype=int)
                        fallback_annotation[non_zero_mask] = anno_non_zero
                        annotations.loc[non_zero_mask, module_id] = anno_non_zero
                        # 计算新统计：模块级 Moran's I
                        fallback_mod_I = compute_moran(expr_values, W)
                        stats['module_moran'] = fallback_mod_I
                        # Positive group (用掩蔽方法)
                        pos_expr_masked = np.where(fallback_annotation == 1, expr_values, 0)
                        stats['positive_moran'] = compute_moran(pos_expr_masked, W)
                        full_indices = np.where(non_zero_mask)[0]
                        pos_idx = full_indices[anno_non_zero == 1]
                        if len(pos_idx) > 1:
                            pos_coords = coords[pos_idx, :]
                            stats['positive_mean_distance'] = float(np.mean(pdist(pos_coords)))
                        else:
                            stats['positive_mean_distance'] = np.nan
                        # Negative group
                        neg_expr_masked = np.where(fallback_annotation == 0, expr_values, 0)
                        stats['negative_moran'] = compute_moran(neg_expr_masked, W)
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
                except Exception as fallback_e:
                    stats['error_info'] += f"; Fallback failed: {str(fallback_e)}"
        finally:
            if stats['status'] == 'success':
                print(f"{module_id} processed successfully, annotated cells: {stats['anno_one']}")
            else:
                print(f"{module_id} processed, failed: {stats['error_info']}")
            stats['components'] = str(stats['components'])
            stats_records.append(stats)
    
    # 后处理：转换 annotations 的 0/1 为模块ID或 None，并转换为 categorical 类型
    annotations.columns = [f"{col}_anno" for col in annotations.columns]
    for col in annotations.columns:
        orig_name = col.replace("_anno", "")
        annotations[col] = np.where(annotations[col] == 1, orig_name, None)
        annotations[col] = pd.Categorical(annotations[col])
    adata.obs = pd.concat([adata.obs, annotations], axis=1)
    
    # 存储统计记录到 adata.uns[mod_stats_key]
    stats_records_df = pd.DataFrame(stats_records)
    new_order = [
    'module_id', 'status', 'anno_one', 'anno_zero',
    'module_moran', 'positive_moran', 'negative_moran', 'positive_mean_distance',
    'skew', 'top1pct_ratio', 'n_components', 'final_components',
    'threshold', 'components', 'main_component', 'error_info']
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



# %%
# 测试
start_time = time.time()
calculate_gmm_annotations(adata, 
                            ggm_key='ggm',
                            #modules_used=None,
                            #modules_used=['M01', 'M02', 'M03', 'M04', 'M05', 'M06', 'M07', 'M08', 'M09', 'M10'],
                            #modules_used=['M11', 'M12', 'M13', 'M14', 'M15', 'M16', 'M17', 'M18', 'M19', 'M20'],
                            #modules_used={'M01', 'M02', 'M03', 'M04', 'M05', 'M06', 'M07', 'M08', 'M09', 'M10'},
                            #modules_excluded=['M01', 'M02', 'M03', 'M04', 'M05', 'M06', 'M07', 'M08', 'M09', 'M10'],
                            embedding_key='spatial',
                            k_neighbors=6,
                            max_iter=200,
                            prob_threshold=0.99,
                            min_samples=10,
                            n_components=3,
                            enable_fallback=True,
                            random_state=42)
print(f"Time: {time.time() - start_time:.5f} s")
print(adata.uns['module_stats'])

# %%
adata.uns['union_module_stats'].to_csv("union_module_stats.csv")

# %%
adata.uns['module_info']['module_moran_I'].unique()
















# %%
# 问题11，关于合并注释，尝试结合louvain或者leiden的聚类结果，在每个聚类之内使用模块来精准注释。

# %%
# 问题13，关于合并注释，在adata的uns中添加一个配色方案，为每个模块指定配色，特别是模块过多的时候。
adata.uns['module_stats']

# %%
# 计算GMM注释
start_time = time.time()
sg.calculate_gmm_annotations(adata, 
                             ggm_key='intersection',
                         #modules_used=None,
                         #modules_used=['M01', 'M02', 'M03', 'M04', 'M05', 'M06', 'M07', 'M08', 'M09', 'M10'],
                         #modules_used=['M11', 'M12', 'M13', 'M14', 'M15', 'M16', 'M17', 'M18', 'M19', 'M20'],
                         #modules_used={'M01', 'M02', 'M03', 'M04', 'M05', 'M06', 'M07', 'M08', 'M09', 'M10'},
                         max_iter=200,
                         prob_threshold=0.99,
                         min_samples=10,
                         n_components=3,
                         enable_fallback=True,
                         random_state=42)
print(f"Time: {time.time() - start_time:.5f} s")
print(adata.uns['module_stats'])

# %%
# 平滑注释
start_time = time.time()
sg.smooth_annotations(adata, 
                      ggm_key='intersection',
                    #modules_used=None,
                    #modules_used=['M01', 'M02', 'M03', 'M04', 'M05', 'M06', 'M07', 'M08', 'M09', 'M10'],
                    #modules_used=['M11', 'M12', 'M13', 'M14', 'M15', 'M16', 'M17', 'M18', 'M19', 'M20'],
                    #modules_used={'M01', 'M02', 'M03', 'M04', 'M05', 'M06', 'M07', 'M08', 'M09', 'M10'},
                    embedding_key='spatial',
                    k_neighbors=18,
                    min_annotated_neighbors=2
                    )
print(f"Time: {time.time() - start_time:.5f} s")    

# %%
# 合并注释（考虑空间坐标和模块表达值）
start_time = time.time()
sg.integrate_annotations(adata,
                         ggm_key='ggm',
                  #modules_used=None,
                  #modules_used = adata.uns['module_info']['module_id'].unique(), 
                  #modules_used=['M01', 'M02', 'M03', 'M04', 'M05', 'M06', 'M07', 'M08', 'M09', 'M10'],
                  #modules_used=['M11', 'M12', 'M13', 'M14', 'M15', 'M16', 'M17', 'M18', 'M19', 'M20'],
                  #modules_used={'M01', 'M02', 'M03', 'M04', 'M05', 'M06', 'M07', 'M08', 'M09', 'M10'},
                  result_anno='annotation',
                  embedding_key='spatial',
                  k_neighbors=18,
                  use_smooth=True,
                  neighbor_similarity_ratio=1
                  )
sg.integrate_annotations(adata,
                         ggm_key='union',
                  #modules_used=None,
                  #modules_used = adata.uns['module_info']['module_id'].unique(), 
                  #modules_used=['M01', 'M02', 'M03', 'M04', 'M05', 'M06', 'M07', 'M08', 'M09', 'M10'],
                  #modules_used=['M11', 'M12', 'M13', 'M14', 'M15', 'M16', 'M17', 'M18', 'M19', 'M20'],
                  #modules_used={'M01', 'M02', 'M03', 'M04', 'M05', 'M06', 'M07', 'M08', 'M09', 'M10'},
                  result_anno='union_annotation',
                  embedding_key='spatial',
                  k_neighbors=18,
                  use_smooth=True,
                  neighbor_similarity_ratio=1
                  )
sg.integrate_annotations(adata,
                         ggm_key='intersection',
                  #modules_used=None,
                  #modules_used = adata.uns['module_info']['module_id'].unique(), 
                  #modules_used=['M01', 'M02', 'M03', 'M04', 'M05', 'M06', 'M07', 'M08', 'M09', 'M10'],
                  #modules_used=['M11', 'M12', 'M13', 'M14', 'M15', 'M16', 'M17', 'M18', 'M19', 'M20'],
                  #modules_used={'M01', 'M02', 'M03', 'M04', 'M05', 'M06', 'M07', 'M08', 'M09', 'M10'},
                  result_anno='intersection_annotation',
                  embedding_key='spatial',
                  k_neighbors=18,
                  use_smooth=True,
                  neighbor_similarity_ratio=1
                  )
print(f"Time: {time.time() - start_time:.5f} s")

# %%
sg.integrate_annotations(adata,
                         ggm_key='intersection',
                         cross_ggm=True,
                         modules_used=['M01', 'M02', 'M03', 'M04', 'M05', 'M06', 'M07', 'M08', 'M09', 'M10',
                                      'intersection_M001', 'intersection_M002', 'intersection_M003', 'intersection_M004', 'intersection_M005',
                                      'intersection_M006', 'intersection_M007', 'intersection_M008', 'intersection_M009', 'intersection_M010',
                                      'union_M001', 'union_M002', 'union_M003', 'union_M004', 'union_M005',
                                      'union_M006', 'union_M007', 'union_M008', 'union_M009', 'union_M010'],
                  #modules_used=None,
                  #modules_used = adata.uns['module_info']['module_id'].unique(), 
                  #modules_used=['M01', 'M02', 'M03', 'M04', 'M05', 'M06', 'M07', 'M08', 'M09', 'M10'],
                  #modules_used=['M11', 'M12', 'M13', 'M14', 'M15', 'M16', 'M17', 'M18', 'M19', 'M20'],
                  #modules_used={'M01', 'M02', 'M03', 'M04', 'M05', 'M06', 'M07', 'M08', 'M09', 'M10'},
                  result_anno='merge_annotation',
                  embedding_key='spatial',
                  k_neighbors=18,
                  use_smooth=True,
                  neighbor_similarity_ratio=1
                  )
# %%
# %%
# 合并注释（仅考虑模块注释的细胞数目）
start_time = time.time()
sg.integrate_annotations_old(adata,
                             ggm_key='ggm',
                         #modules_used=None,
                         #modules_used=['M01', 'M02', 'M03', 'M04', 'M05', 'M06', 'M07', 'M08', 'M09', 'M10'],
                         #modules_used=['M11', 'M12', 'M13', 'M14', 'M15', 'M16', 'M17', 'M18', 'M19', 'M20'],
                         #modules_used={'M01', 'M02', 'M03', 'M04', 'M05', 'M06', 'M07', 'M08', 'M09', 'M10'},
                         result_anno = "annotation_old",
                         use_smooth=True
                         )
sg.integrate_annotations_old(adata,
                             ggm_key='intersection',
                         #modules_used=None,
                         #modules_used=['M01', 'M02', 'M03', 'M04', 'M05', 'M06', 'M07', 'M08', 'M09', 'M10'],
                         #modules_used=['M11', 'M12', 'M13', 'M14', 'M15', 'M16', 'M17', 'M18', 'M19', 'M20'],
                         #modules_used={'M01', 'M02', 'M03', 'M04', 'M05', 'M06', 'M07', 'M08', 'M09', 'M10'},
                         result_anno = "intersection_annotation_old",
                         use_smooth=True
                         )
sg.integrate_annotations_old(adata,
                             ggm_key='union',
                         #modules_used=None,
                         #modules_used=['M01', 'M02', 'M03', 'M04', 'M05', 'M06', 'M07', 'M08', 'M09', 'M10'],
                         #modules_used=['M11', 'M12', 'M13', 'M14', 'M15', 'M16', 'M17', 'M18', 'M19', 'M20'],
                         #modules_used={'M01', 'M02', 'M03', 'M04', 'M05', 'M06', 'M07', 'M08', 'M09', 'M10'},
                         result_anno = "union_annotation_old",
                         use_smooth=True
                         )
sg.integrate_annotations_old(adata,
                             ggm_key='ggm',
                             cross_ggm=True,
                             modules_used=['M01', 'M02', 'M03', 'M04', 'M05', 'M06', 'M07', 'M08', 'M09', 'M10',
                                      'intersection_M001', 'intersection_M002', 'intersection_M003', 'intersection_M004', 'intersection_M005',
                                      'intersection_M006', 'intersection_M007', 'intersection_M008', 'intersection_M009', 'intersection_M010',
                                      'union_M001', 'union_M002', 'union_M003', 'union_M004', 'union_M005',
                                      'union_M006', 'union_M007', 'union_M008', 'union_M009', 'union_M010'],
                         #modules_used=None,
                         #modules_used=['M01', 'M02', 'M03', 'M04', 'M05', 'M06', 'M07', 'M08', 'M09', 'M10'],
                         #modules_used=['M11', 'M12', 'M13', 'M14', 'M15', 'M16', 'M17', 'M18', 'M19', 'M20'],
                         #modules_used={'M01', 'M02', 'M03', 'M04', 'M05', 'M06', 'M07', 'M08', 'M09', 'M10'},
                         result_anno = "merge_annotation_old",
                         use_smooth=True
                         )
print(f"Time: {time.time() - start_time:.5f} s")




# %%
for module in adata.uns['intersection_module_stats']['module_id'].unique():
    print(module)
    print(adata.uns['intersection_module_info'][adata.uns['intersection_module_info']['module_id']==module]['module_moran_I'].unique())
    sc.pl.spatial(adata, color=[f"{module}_exp",f"{module}_anno",f"{module}_anno_smooth"],
                    color_map="Reds", alpha_img = 0.5, size = 1.6, frameon = False, show=True)
    

# %%
for module in adata.uns['union_module_stats']['module_id'].unique():
    print(module)
    print(adata.uns['union_module_info'][adata.uns['union_module_info']['module_id']==module]['module_moran_I'].unique())
    sc.pl.spatial(adata, color=[f"{module}_exp",f"{module}_anno",f"{module}_anno_smooth"],
                    color_map="Reds", alpha_img = 0.5, size = 1.6, frameon = False, show=True)




# %%
# 计算模块重叠
start_time = time.time()
overlap_records = sg.calculate_module_overlap(adata, 
                                           modules_used = adata.uns['module_stats']['module_id'].unique())
print(f"Time: {time.time() - start_time:.5f} s")
print(overlap_records[overlap_records['module_a'] == 'M01'])

# %%
# 注释结果可视化
sc.pl.spatial(adata, alpha_img = 0.5, size = 1.6, title= "", frameon = False, color="annotation", show=True)

# %%
sc.pl.spatial(adata, alpha_img = 0.5, size = 1.6, title= "", frameon = False, color="union_annotation", show=True)

# %%
sc.pl.spatial(adata, alpha_img = 0.5, size = 1.6, title= "", frameon = False, color="intersection_annotation", show=True)

# %%
sc.pl.spatial(adata, alpha_img = 0.5, size = 1.6, title= "", frameon = False, color="merge_annotation", show=True)

# %%
sc.pl.spatial(adata, alpha_img = 0.5, size = 1.6, title= "", frameon = False, color="annotation_old", show=True)

# %%
sc.pl.spatial(adata, alpha_img = 0.5, size = 1.6, title= "", frameon = False, color="union_annotation_old", show=True)

# %%
sc.pl.spatial(adata, alpha_img = 0.5, size = 1.6, title= "", frameon = False, color="intersection_annotation_old", show=True)

# %%
sc.pl.spatial(adata, alpha_img = 0.5, size = 1.6, title= "", frameon = False, color="merge_annotation_old", show=True)

# %%
# 保存adata
adata.write("data/visium/CytAssist_FreshFrozen_Mouse_Brain_Rep2_ggm_anno.h5ad")


# %%
# 逐个可视化各个模块的注释结果
anno_modules = adata.uns['module_stats']['module_id'].unique()
# 1. 原始注释绘图
pdf_file = "figures/visium/All_modules_in_CytAssist_FreshFrozen_Mouse_Brain_Rep2_Module_Anno_Raw.pdf"
c = canvas.Canvas(pdf_file, pagesize=letter)
image_files = []
for module in anno_modules:
    adata.obs['plot_anno'] = adata.obs[f"{module}_anno"].apply(lambda x: module if x else np.nan)
    if len(adata.obs['plot_anno'][adata.obs['plot_anno'] == module]) > 1:
        plt.figure()    
        sc.pl.spatial(adata, img_key = "hires", alpha_img = 0.5, size = 1.6, title= f"{module}_anno", frameon = False, color="plot_anno",show=False)
        raw_png_file = f"figures/visium/{module}_in_CytAssist_FreshFrozen_Mouse_Brain_Rep2_Module_Anno_Raw.png"
        plt.savefig(raw_png_file, format="png", dpi=300, bbox_inches="tight")
        plt.close()
        image_files.append(raw_png_file)

for image_file in image_files:
    img = Image.open(image_file)
    c.setPageSize((img.width, img.height))
    c.drawImage(image_file, 0, 0, width=img.width, height=img.height)
    c.showPage()
c.save()    

# 2. 平滑注释绘图
pdf_file = "figures/visium/All_modules_in_CytAssist_FreshFrozen_Mouse_Brain_Rep2_Module_Anno_Smooth.pdf"
c = canvas.Canvas(pdf_file, pagesize=letter)
image_files = []
for module in anno_modules:
    adata.obs['plot_anno'] = adata.obs[f"{module}_anno_smooth"].apply(lambda x: module if x else np.nan)
    if len(adata.obs['plot_anno'][adata.obs['plot_anno'] == module]) > 1:
        plt.figure()
        sc.pl.spatial(adata, img_key = "hires", alpha_img = 0.5, size = 1.6, title= f"{module}_anno_smooth", frameon = False, color="plot_anno",show=False)
        smooth_png_file = f"figures/visium/{module}_in_CytAssist_FreshFrozen_Mouse_Brain_Rep2_Module_Anno_Smooth.png"
        plt.savefig(smooth_png_file, format="png", dpi=300, bbox_inches="tight")
        plt.close()
        image_files.append(smooth_png_file)

for image_file in image_files:
    img = Image.open(image_file)
    c.setPageSize((img.width, img.height))
    c.drawImage(image_file, 0, 0, width=img.width, height=img.height)
    c.showPage()
c.save()    

# 3. 模块加权表达图
pdf_file = "figures/visium/All_modules_in_CytAssist_FreshFrozen_Mouse_Brain_Rep2_Module_Exp.pdf"
c = canvas.Canvas(pdf_file, pagesize=letter)
image_files = []
for module in anno_modules:
    plt.figure()    
    sc.pl.spatial(adata, img_key = "hires", alpha_img = 0.5, size = 1.6, title= f"{module}_exp", frameon = False, color=f"{module}_exp", color_map="Reds", show=False)
    raw_png_file = f"figures/visium/{module}_in_CytAssist_FreshFrozen_Mouse_Brain_Rep2_Module_Exp.png"
    plt.savefig(raw_png_file, format="png", dpi=300, bbox_inches="tight")
    plt.close()
    image_files.append(raw_png_file)

for image_file in image_files:
    img = Image.open(image_file)
    c.setPageSize((img.width, img.height))
    c.drawImage(image_file, 0, 0, width=img.width, height=img.height)
    c.showPage()
c.save()    

# %%






# %%
# 细胞注释相关的问题
# 问题1，关于计算平均表达值。当使用新的ggm结果注释已经存在module expression的adata时，会报错
# 添加ggm_key 参数，为GGM指定一个key，用来区分不同的ggm结果。
# 解决

# %%
# 问题2，关于计算平均表达值。添加可选参数，计算模块内每个基因的莫兰指数以及模块整体的莫兰指数。
# 解决

# %%
# 问题3，关于模块注释的全部函数，添加反选参数，用来反向排除模块。
# 解决

# %%
# 问题4，关于模块注释的全部函数, 细胞按模块的注释结果改为category类型。而不是现在的0，1，int类型。并注意，之后在涉及到使用这些数据的时候还要换回int类型。
# 解决

# %%
# 问题8，关于平滑处理，在使用的时候，无法仅处理部分模块。
# 解决

# %%
# 问题9，关于合并注释，优化keep_modules的参数。
# 解决

# %%
# 问题12，关于合并注释，注释结果中，字符串None改为空值的None。
# 解决

# %%
# 问题14，关于合并注释，neighbor_similarity_ratio参数似乎会导致activity模块的权重过高。考虑将其设置为0或者1来避免考虑neighbor_similarity_ratio
# 解决
