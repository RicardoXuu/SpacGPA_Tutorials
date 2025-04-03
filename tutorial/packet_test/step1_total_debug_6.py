
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

# %%
# 读取 ggm
start_time = time.time()
ggm = sg.load_ggm("data/ggm_gpu_32.h5")
print(f"Read ggm: {time.time() - start_time:.5f} s")
# 读取联合分析的ggm
ggm_mulit_intersection = sg.load_ggm("data/ggm_mulit_intersection.h5")
print(f"Read ggm_mulit_intersection: {time.time() - start_time:.5f} s")
ggm_mulit_union = sg.load_ggm("data/ggm_mulit_union.h5")
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
ggm.find_modules(methods='louvain', resolution=0.5,
                        expansion=2, inflation=2, max_iter=1000, tol=1e-6, pruning_threshold=1e-5,
                        min_module_size=10, topology_filtering=False, 
                        convert_to_symbols=True, species='mouse')
print(ggm.modules_summary)






# %%
# 测试
sg.annotate_with_ggm(adata,
                 ggm_obj=ggm,
                 ggm_key='ggm')
sg.annotate_with_ggm(adata,
                 ggm_obj=ggm_mulit_union,
                 ggm_key='union')
sg.annotate_with_ggm(adata,
                 ggm_obj=ggm_mulit_intersection,
                 ggm_key='intersection')

# %%
# 测试重复
sg.annotate_with_ggm(adata,
                 ggm_obj=ggm,
                 ggm_key='ggm')
sg.annotate_with_ggm(adata,
                 ggm_obj=ggm_mulit_union,
                 ggm_key='union')
sg.annotate_with_ggm(adata,
                 ggm_obj=ggm_mulit_intersection,
                 ggm_key='intersection')


# %%
# 设计快速分析函数
from SpacGPA import calculate_module_expression
from SpacGPA import calculate_gmm_annotations
from SpacGPA import smooth_annotations

def annotate_with_ggm(
    adata,
    ggm_obj,
    ggm_key='ggm',
    top_genes=30,
    weighted=True,
    calculate_gene_moran=False,  
    calculate_module_moran=True, 
    embedding_key='spatial',  
    k_neighbors_for_moran=6,          
    add_go_anno=3,
    
    max_iter=200,
    prob_threshold=0.99,
    min_samples=10,
    n_components=3,
    enable_fallback=True,
    random_state=42,
    
    k_neighbors_for_smooth=24,
    min_annotated_neighbors=1
):
    """
    Execute the Annotate and Smooth pipeline for GGM analysis in one step:
      1. Compute module average expression using provided GGM information (via calculate_module_expression).
      2. Annotate cells based on module expression with a Gaussian Mixture Model (GMM) and calculate additional module-level statistics (via calculate_gmm_annotations).
      3. Perform spatial smoothing on the annotation results (via smooth_annotations).

    Parameters:
      --- For calculate_module_expression ---
      adata: AnnData object containing gene expression data.
      ggm_obj: GGM object or DataFrame containing module information.
      ggm_key: Key for storing GGM information in adata, default 'ggm'.
      top_genes: Number of top genes used for module expression calculation, default 30.
      weighted: Whether to compute weighted average expression based on gene degree, default True.
      calculate_gene_moran: Whether to compute Moran's I for genes during module expression calculation, default False.
      embedding_key: Key in adata.obsm that stores spatial coordinates, default 'spatial'.
      k_neighbors_for_moran: Number of neighbors used for constructing the spatial weight matrix, default 6.
      add_go_anno: Parameter for GO annotation integration; default 3 (extracts top 3 GO terms).

      --- For calculate_gmm_annotations ---
      ggm_key: Key for storing GGM information in adata, default 'ggm'. same as above.
      calculate_module_moran: Whether to compute Moran's I for modules during GMM annotation, default True.
      embedding_key: Key in adata.obsm that stores spatial coordinates, default 'spatial'. same as above.
      k_neighbors_for_moran: Number of neighbors used for constructing the spatial weight matrix, default 6. same as above.
      max_iter: Maximum iterations for the GMM, default 200.
      prob_threshold: Probability threshold for calling a cell positive, default 0.99.
      min_samples: Minimum number of non-zero samples required for GMM analysis, default 10.
      n_components: Number of components in the GMM, default 3.
      enable_fallback: Whether to fallback to a 2-component model if GMM fitting fails, default True.
      random_state: Random seed for reproducibility, default 42.
      (Note: Optional parameters like modules_used and modules_excluded are handled within the function.)

      --- For smooth_annotations ---
      ggm_key: Key for storing GGM information in adata, default 'ggm'. same as above.
      embedding_key: Key in adata.obsm that stores spatial coordinates, default 'spatial'. same as above.
      k_neighbors_for_smooth: Number of KNN neighbors used for smoothing annotations, default 24.
      min_annotated_neighbors: Minimum number of annotated neighbors required to retain a positive annotation, default 1.
      (Note: Optional parameters like modules_used and modules_excluded are handled within the function.)

    Returns:
      The updated AnnData object with module expression, cell annotations, and smoothed results stored in .obs and .obsm.
    """
    # Compute module average expression
    print("============ Calculating module average expression ============")
    calculate_module_expression(
        adata=adata,
        ggm_obj=ggm_obj,
        ggm_key=ggm_key,
        top_genes=top_genes,
        weighted=weighted,
        calculate_moran=calculate_gene_moran,
        embedding_key=embedding_key,
        k_neighbors=k_neighbors_for_moran,
        add_go_anno=add_go_anno
    )
    
    # Annotate cells based on module expression
    print("\n======== Annotating cells based on module expression ========")
    calculate_gmm_annotations(
        adata=adata,
        ggm_key=ggm_key,
        calculate_moran=calculate_module_moran,
        embedding_key=embedding_key,
        k_neighbors=k_neighbors_for_moran,
        max_iter=max_iter,
        prob_threshold=prob_threshold,
        min_samples=min_samples,
        n_components=n_components,
        enable_fallback=enable_fallback,
        random_state=random_state
    )
    
    # Smooth cell annotations spatially
    print("\n=================== Smoothing annotations ===================")
    smooth_annotations(
        adata=adata,
        ggm_key=ggm_key,
        embedding_key=embedding_key,
        k_neighbors=k_neighbors_for_smooth,
        min_annotated_neighbors=min_annotated_neighbors
    )
    print("\n============= Finished annotating and smoothing =============")


# %%
# 测试
annotate_with_ggm(adata,
                 ggm_obj=ggm,
                 ggm_key='ggm')
annotate_with_ggm(adata,
                 ggm_obj=ggm_mulit_union,
                 ggm_key='union')
annotate_with_ggm(adata,
                 ggm_obj=ggm_mulit_intersection,
                 ggm_key='intersection')

# %%
# 测试重复
annotate_with_ggm(adata,
                 ggm_obj=ggm,
                 ggm_key='ggm')
annotate_with_ggm(adata,
                 ggm_obj=ggm_mulit_union,
                 ggm_key='union')
annotate_with_ggm(adata,
                 ggm_obj=ggm_mulit_intersection,
                 ggm_key='intersection')

# %%
adata.obs.head()
# %%
adata.uns['ggm_keys']
# %%
adata.obsm
